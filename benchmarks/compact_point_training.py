#!/usr/bin/env python3
"""Sealed fixed-topology training against compact 2D Gaussian teachers.

This is a closed scientific harness for the protocol in
``20260716_compact_point_training_PREREG.md``.  Importing it is outcome-free: the literal
official fixture is guarded by the once-only attempt marker, and every seed--arm execution
runs in a fresh ``multiprocessing`` spawn child.  Pure validation/statistics helpers are public
so tests and the independent audit can recompute the consequential transformations.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.core.sh import eval_sh_preactivation, rgb_to_sh
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.render.torch_points import TorchPointRasterizer
from rtgs.render.torch_ref import (
    _DILATION,
    _MAX_ALPHA,
    _NEAR,
    KERNEL_SUPPORT_CUTOFF,
    kernel_support_weight,
)

ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = Path("benchmarks/results/20260716_compact_point_training_PREREG.md")
PREREGISTRATION_SHA256 = "865f86d35805c265d27caf4b5f6e02b99e4679f53162dbdd23c17681354065ea"
PREREGISTRATION_REVIEW = Path("benchmarks/results/20260716_compact_point_training_PREREG_REVIEW.md")
IMPLEMENTATION_REVIEW = Path(
    "benchmarks/results/20260716_compact_point_training_IMPLEMENTATION_REVIEW.md"
)
HARNESS = Path("benchmarks/compact_point_training.py")
FOCUSED_TEST = Path("tests/test_compact_point_training.py")

SEAL = ROOT / "benchmarks/results/20260716_compact_point_training_SEAL.json"
ATTEMPT = ROOT / "benchmarks/results/20260716_compact_point_training_ATTEMPT.json"
RAW = ROOT / "benchmarks/results/20260716_compact_point_training_RAW.json"
RESULT = ROOT / "benchmarks/results/20260716_compact_point_training_RESULT.json"
AUDIT = ROOT / "benchmarks/results/20260716_compact_point_training_AUDIT.md"

RUN_DIR = ROOT / "runs/compact_point_training_20260716"
CALIBRATED_PLAN = RUN_DIR / "CALIBRATED_PLAN.json"
CALIBRATED_ATTEMPT = RUN_DIR / "CALIBRATED_ATTEMPT.json"
CALIBRATED_BUNDLE = RUN_DIR / "reconstruction_inputs"
CALIBRATED_ACQUISITION = RUN_DIR / "teacher_acquisition.json"
CALIBRATED_TRAINING_RAW = RUN_DIR / "compact_training_raw.json"
CALIBRATED_INIT = RUN_DIR / "gaussians_init.ply"
CALIBRATED_FINAL = RUN_DIR / "gaussians.ply"
CALIBRATED_HELDOUT = RUN_DIR / "heldout_evaluation.json"
CALIBRATED_VIEWER = RUN_DIR / "viewer_smoke.json"
CALIBRATED_SNAPSHOTS = RUN_DIR / "viewer_snapshots"
CALIBRATED_RESULT = RUN_DIR / "calibrated_result.json"

OFFICIAL_SEEDS = (74101, 74102, 74103)
ARMS = ("pixel_uniform", "pixel_gaussian", "area_uniform", "area_gaussian")
DOMAIN_PAIRS = {
    "pixel": ("pixel_uniform", "pixel_gaussian", "pixel"),
    "area": ("area_uniform", "area_gaussian", "area"),
}
CHECKPOINTS = (0, 30, 60, 120)
ITERATIONS = 120
ATTEMPTS_PER_STEP = 128
ETA = 0.25
OFFICIAL_N_INIT_2D = (8, 8, 8)
OFFICIAL_N_OPT_2D = (4, 5, 6)
OFFICIAL_N_3D = 4
GAUSSIAN_FIELDS = ("means", "quats", "log_scales", "opacity", "sh")
MOTION_FAMILIES = ("means", "quaternions", "log_scales", "opacity_logits", "sh0")

SEAL_ARTIFACT_TYPE = "compact_point_training_implementation_seal_v1"
ATTEMPT_ARTIFACT_TYPE = "compact_point_training_once_only_attempt_v1"
RAW_ARTIFACT_TYPE = "compact_point_training_raw_v1"
RESULT_ARTIFACT_TYPE = "compact_point_training_result_v1"

MAX_VIEWS = 64
MAX_FITTED_PIXELS_PER_VIEW = 50_000_000
MAX_COMPONENTS_PER_VIEW = 2_000_000
MAX_TILE_ENTRIES_PER_VIEW = 16_000_000
MAX_CANDIDATES_PER_TILE = 200_000
MAX_MANIFEST_BYTES = 8_388_608
MAX_TEACHER_ARCHIVES = 64
MAX_COMPRESSED_ARCHIVE_BYTES = 268_435_456
MAX_TOTAL_COMPRESSED_BYTES = 2_147_483_648
MAX_ZIP_MEMBERS = 64
MAX_UNCOMPRESSED_MEMBER_BYTES = 268_435_456
MAX_UNCOMPRESSED_ARCHIVE_BYTES = 1_073_741_824

CALIBRATED_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
CALIBRATED_HELDOUT_VIEW = "C1004"
CALIBRATED_SCENE = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
CALIBRATED_VIEWER_HOST = "127.0.0.1"
CALIBRATED_VIEWER_PORT = 8876
CALIBRATED_LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
REQUIRED_CXXABI_SYMBOL = "__cxa_call_terminate"
REQUIRED_CXXABI_VERSION = "CXXABI_1.3.15"
CALIBRATED_VIEWER_COMMAND = (
    ".venv/bin/rtgs",
    "view",
    "--gaussians",
    "runs/compact_point_training_20260716/gaussians.ply",
    "--initial",
    "runs/compact_point_training_20260716/gaussians_init.ply",
    "--scene",
    "dataset/2025_03_07_stage_with_fabric/frame_00008",
    "--downscale",
    "1",
    "--max-images",
    "8",
    "--rasterizer",
    "gsplat",
    "--device",
    "cuda",
    "--snapshot-dir",
    "runs/compact_point_training_20260716/viewer_snapshots",
    "--host",
    CALIBRATED_VIEWER_HOST,
    "--port",
    str(CALIBRATED_VIEWER_PORT),
    "--no-open",
)

# Incremented only inside the guarded official constructor.  Import and ordinary unit tests must
# leave this at zero; it is intentionally observable without exposing any official outcome.
_OFFICIAL_CONSTRUCTION_COUNT = 0
_OFFICIAL_AUTHORIZATION_SECRET = object()


class ProtocolInvalid(RuntimeError):
    """Fail-closed protocol violation distinct from a neutral scientific result."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _abi_preload_file_binding(requested_path_text: str) -> dict[str, str]:
    """Freeze the exact C++ runtime and versioned ABI contract for a future plan."""
    requested = Path(requested_path_text)
    if not requested.is_absolute() or not requested.is_file():
        raise ProtocolInvalid("calibrated LD_PRELOAD must name an existing absolute file")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ProtocolInvalid("calibrated LD_PRELOAD cannot be resolved") from error
    if not resolved.is_file():
        raise ProtocolInvalid("resolved calibrated LD_PRELOAD is not a file")
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "sha256": sha256_file(resolved),
        "required_symbol": REQUIRED_CXXABI_SYMBOL,
        "required_version": REQUIRED_CXXABI_VERSION,
    }


def _verify_abi_preload_file_binding(binding: Mapping[str, Any]) -> dict[str, str]:
    """Rebind a planned preload path, including symlink target and bytes, fail-closed."""
    expected_keys = {
        "requested_path",
        "resolved_path",
        "sha256",
        "required_symbol",
        "required_version",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_keys:
        raise ProtocolInvalid("calibrated ABI preload binding has the wrong key set")
    if (
        not all(isinstance(binding[name], str) for name in expected_keys)
        or not _is_sha256(binding["sha256"])
        or binding["required_symbol"] != REQUIRED_CXXABI_SYMBOL
        or binding["required_version"] != REQUIRED_CXXABI_VERSION
    ):
        raise ProtocolInvalid("calibrated ABI preload binding has invalid fields")
    requested = Path(binding["requested_path"])
    resolved = Path(binding["resolved_path"])
    if not requested.is_absolute() or not resolved.is_absolute():
        raise ProtocolInvalid("calibrated ABI preload paths must be absolute")
    try:
        current_resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ProtocolInvalid("calibrated ABI preload requested path no longer resolves") from error
    if current_resolved != resolved or not resolved.is_file():
        raise ProtocolInvalid("calibrated ABI preload symlink target changed")
    if sha256_file(resolved) != binding["sha256"]:
        raise ProtocolInvalid("calibrated ABI preload bytes changed")
    return {name: binding[name] for name in expected_keys}


def _default_namespace_abi_resolution(binding: Mapping[str, Any]) -> dict[str, str]:
    """Prove a versioned symbol resolves from the planned library in this process."""
    if sys.platform != "linux":
        raise ProtocolInvalid("versioned default-namespace ABI verification requires Linux")
    expected = _verify_abi_preload_file_binding(binding)

    import ctypes

    class DlInfo(ctypes.Structure):
        _fields_ = (
            ("dli_fname", ctypes.c_char_p),
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        )

    loader = ctypes.CDLL(None)
    try:
        dlvsym = loader.dlvsym
        dladdr = loader.dladdr
    except AttributeError as error:
        raise ProtocolInvalid("dynamic loader does not expose dlvsym/dladdr") from error
    dlvsym.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
    dlvsym.restype = ctypes.c_void_p
    address = dlvsym(
        None,
        expected["required_symbol"].encode("ascii"),
        expected["required_version"].encode("ascii"),
    )
    if not address:
        raise ProtocolInvalid(
            "planned C++ ABI symbol/version is absent from the default linker namespace"
        )
    dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(DlInfo))
    dladdr.restype = ctypes.c_int
    info = DlInfo()
    if dladdr(address, ctypes.byref(info)) == 0 or not info.dli_fname or not info.dli_sname:
        raise ProtocolInvalid("dladdr could not bind the resolved C++ ABI symbol")
    dladdr_path = Path(os.fsdecode(info.dli_fname))
    try:
        dladdr_resolved = dladdr_path.resolve(strict=True)
    except OSError as error:
        raise ProtocolInvalid("dladdr reported an unavailable C++ runtime") from error
    dladdr_symbol = os.fsdecode(info.dli_sname)
    if dladdr_symbol != expected["required_symbol"]:
        raise ProtocolInvalid("dladdr reported a different C++ ABI symbol")

    mapped_line = None
    mapped_path = None
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        try:
            lower_text, upper_text = fields[0].split("-", maxsplit=1)
            lower, upper = int(lower_text, 16), int(upper_text, 16)
        except ValueError:
            continue
        if lower <= int(address) < upper:
            mapped_line = line
            mapped_path = Path(fields[5])
            break
    if mapped_line is None or mapped_path is None:
        raise ProtocolInvalid("resolved C++ ABI symbol has no file mapping in /proc/self/maps")
    try:
        mapped_resolved = mapped_path.resolve(strict=True)
    except OSError as error:
        raise ProtocolInvalid("mapped C++ runtime is unavailable") from error
    expected_resolved = Path(expected["resolved_path"])
    if dladdr_resolved != expected_resolved or mapped_resolved != expected_resolved:
        raise ProtocolInvalid("default-namespace C++ ABI symbol resolved from the wrong library")
    if sha256_file(mapped_resolved) != expected["sha256"]:
        raise ProtocolInvalid("default-namespace C++ ABI library bytes differ from the plan")
    return {
        "lookup": "dlvsym(RTLD_DEFAULT)",
        "required_symbol": expected["required_symbol"],
        "required_version": expected["required_version"],
        "symbol_address": f"0x{int(address):x}",
        "dladdr_symbol": dladdr_symbol,
        "dladdr_library_path": str(dladdr_path),
        "dladdr_resolved_library_path": str(dladdr_resolved),
        "proc_maps_entry": mapped_line,
        "proc_maps_library_path": str(mapped_path),
        "proc_maps_resolved_library_path": str(mapped_resolved),
        "library_sha256": expected["sha256"],
    }


def _verify_default_namespace_abi_resolution(
    resolution: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
) -> None:
    """Validate fresh-worker ABI evidence without resolving the symbol in the parent."""
    expected_binding = _verify_abi_preload_file_binding(binding)
    expected_keys = {
        "lookup",
        "required_symbol",
        "required_version",
        "symbol_address",
        "dladdr_symbol",
        "dladdr_library_path",
        "dladdr_resolved_library_path",
        "proc_maps_entry",
        "proc_maps_library_path",
        "proc_maps_resolved_library_path",
        "library_sha256",
    }
    if not isinstance(resolution, Mapping) or set(resolution) != expected_keys:
        raise ProtocolInvalid("default-namespace ABI evidence has the wrong key set")
    if not all(isinstance(resolution[name], str) for name in expected_keys):
        raise ProtocolInvalid("default-namespace ABI evidence has invalid field types")
    expected_resolved = expected_binding["resolved_path"]
    if (
        resolution["lookup"] != "dlvsym(RTLD_DEFAULT)"
        or resolution["required_symbol"] != expected_binding["required_symbol"]
        or resolution["required_version"] != expected_binding["required_version"]
        or resolution["dladdr_symbol"] != expected_binding["required_symbol"]
        or resolution["library_sha256"] != expected_binding["sha256"]
        or resolution["dladdr_resolved_library_path"] != expected_resolved
        or resolution["proc_maps_resolved_library_path"] != expected_resolved
        or re.fullmatch(r"0x[1-9a-f][0-9a-f]*", resolution["symbol_address"]) is None
    ):
        raise ProtocolInvalid("default-namespace ABI evidence differs from the frozen binding")
    address = int(resolution["symbol_address"], 16)
    map_fields = resolution["proc_maps_entry"].split(maxsplit=5)
    try:
        lower_text, upper_text = map_fields[0].split("-", maxsplit=1)
        lower, upper = int(lower_text, 16), int(upper_text, 16)
    except (IndexError, ValueError) as error:
        raise ProtocolInvalid("default-namespace ABI /proc mapping is malformed") from error
    if (
        len(map_fields) != 6
        or "x" not in map_fields[1]
        or not lower <= address < upper
        or map_fields[5] != resolution["proc_maps_library_path"]
    ):
        raise ProtocolInvalid("default-namespace ABI /proc mapping does not contain the symbol")
    for name in ("dladdr_library_path", "proc_maps_library_path"):
        try:
            actual = Path(resolution[name]).resolve(strict=True)
        except OSError as error:
            raise ProtocolInvalid(
                "reported default-namespace ABI library is unavailable"
            ) from error
        if str(actual) != expected_resolved:
            raise ProtocolInvalid("reported default-namespace ABI library path differs")
    if sha256_file(Path(expected_resolved)) != expected_binding["sha256"]:
        raise ProtocolInvalid("default-namespace ABI evidence points to changed bytes")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def strict_json_load_bound(path: Path, expected_sha256: str) -> Any:
    """Decode exactly the bytes whose digest is bound by a preceding commit."""
    encoded = path.read_bytes()
    actual = sha256_bytes(encoded)
    if actual != expected_sha256:
        raise ProtocolInvalid(
            f"committed JSON digest changed before strict reload: {actual} != {expected_sha256}"
        )

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    return json.loads(encoded.decode("utf-8"), parse_constant=reject_constant)


def assert_finite_tree(value: Any, context: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolInvalid(f"{context} contains a non-finite float")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_tree(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{context}[{index}]")


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    header = canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}).encode(
        "utf-8"
    )
    return sha256_bytes(header + b"\0" + tensor.numpy().tobytes(order="C"))


def gaussians_hashes(value: Gaussians3D) -> dict[str, Any]:
    fields = {name: tensor_hash(getattr(value, name)) for name in GAUSSIAN_FIELDS}
    return {
        "n": value.n,
        "sh_degree": value.sh_degree,
        "fields": fields,
        "aggregate": canonical_hash(fields),
    }


def field_hashes(field: GaussianObservationField) -> dict[str, Any]:
    names = (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "amplitudes",
        "mean_residuals",
        "color_grads",
        "filter_variance",
    )
    tensors = {
        name: None if getattr(field, name) is None else tensor_hash(getattr(field, name))
        for name in names
    }
    metadata = {
        "width": field.width,
        "height": field.height,
        "fit_window": list(field.fit_window),
        "blend_mode": field.blend_mode,
        "epsilon": field.epsilon,
        "sigma_cutoff": field.sigma_cutoff,
        "support_fade_alpha": field.support_fade_alpha,
        "aa_dilation": field.aa_dilation,
        "view_id": field.view_id,
        "n_init": field.n_init,
        "n_opt": field.n,
        "provider": field.provider,
        "producer_version": field.producer_version,
        "producer_source_digest": field.producer_source_digest,
        "fit_config_digest": field.fit_config_digest,
    }
    return {
        "tensors": tensors,
        "metadata": metadata,
        "aggregate": canonical_hash([tensors, metadata]),
    }


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    assert_finite_tree(payload, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    encoded = rendered.encode("utf-8")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} for {path}")

    # Finish every potentially fallible serialization check before creating the path.  Once the
    # file and containing directory have both been fsynced, returning the precomputed digest is
    # the commit boundary.
    json.loads(rendered, parse_constant=reject_constant)
    digest = sha256_bytes(encoded)
    created = False
    try:
        stream = path.open("x", encoding="utf-8")
        created = True
        with stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return digest
    except BaseException:
        # A caught pre-commit failure must not leave a path that looks committed.  Do not remove
        # an already-existing path: open('x') failing is a collision, not ownership of that file.
        if created:
            path.unlink(missing_ok=True)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        raise


def _git_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "revision": revision,
        "tracked_diff_sha256": sha256_bytes(diff.encode("utf-8")),
        "status_sha256": sha256_bytes(status.encode("utf-8")),
        "status_lines": status.splitlines(),
    }


def _verify_git_binding(expected: Mapping[str, Any]) -> None:
    """Bind executable tracked state; untracked executable files are source-manifest bound."""
    current = _git_metadata()
    for name in ("revision", "tracked_diff_sha256"):
        if current.get(name) != expected.get(name):
            raise ProtocolInvalid(f"git {name} differs from the implementation seal")


def _reviewed_paths() -> tuple[Path, ...]:
    """Paths whose aggregate can be cited by the non-circular implementation review."""
    paths = {
        PREREGISTRATION,
        PREREGISTRATION_REVIEW,
        HARNESS,
        FOCUSED_TEST,
        Path("pyproject.toml"),
        *(path.relative_to(ROOT) for path in (ROOT / "src/rtgs").rglob("*.py")),
        *(path.relative_to(ROOT) for path in (ROOT / "tests").rglob("*.py")),
        *(path.relative_to(ROOT) for path in (ROOT / "benchmarks").glob("*.py")),
        *(path.relative_to(ROOT) for path in (ROOT / "scripts").glob("*.py")),
        *(path.relative_to(ROOT) for path in ROOT.glob("*.py")),
        *(path.relative_to(ROOT) for path in ROOT.glob("*.pth")),
    }
    return tuple(sorted(paths, key=str))


def source_hashes(paths: Sequence[Path] | None = None) -> tuple[dict[str, str], str]:
    selected = _reviewed_paths() if paths is None else tuple(paths)
    missing = [str(path) for path in selected if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"sealed source files are missing: {missing}")
    hashes = {str(path): sha256_file(ROOT / path) for path in selected}
    return hashes, canonical_hash(hashes)


def sealed_source_hashes() -> tuple[dict[str, str], str]:
    """Final manifest, including the review that binds the non-circular source aggregate."""
    return source_hashes((*_reviewed_paths(), IMPLEMENTATION_REVIEW))


def verify_source_hashes(expected: Mapping[str, str], aggregate: str) -> None:
    current_paths = {str(path) for path in (*_reviewed_paths(), IMPLEMENTATION_REVIEW)}
    if set(expected) != current_paths:
        raise ProtocolInvalid("reviewed source path set differs from the implementation seal")
    actual, actual_aggregate = source_hashes(tuple(Path(path) for path in expected))
    if dict(expected) != actual or aggregate != actual_aggregate:
        raise ProtocolInvalid("repository sources differ from the implementation seal")


def _environment_metadata() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    venv = (ROOT / ".venv").resolve()
    return {
        "python": sys.version,
        "executable": str(executable),
        "prefix": str(Path(sys.prefix).resolve()),
        "expected_venv": str(venv),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": "cpu",
    }


def _configure_official_runtime() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)


def _assert_official_environment(metadata: Mapping[str, Any]) -> None:
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise ProtocolInvalid("official execution requires the repository .venv")
    expected = {
        "torch_num_threads": 4,
        "torch_num_interop_threads": 1,
        "deterministic_algorithms": True,
        "cuda_visible_devices": "",
        "device": "cpu",
    }
    mismatch = {
        name: {"expected": value, "actual": metadata.get(name)}
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if mismatch:
        raise ProtocolInvalid(f"official CPU environment differs from protocol: {mismatch}")


def _environment_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "python",
            "executable",
            "prefix",
            "torch",
            "numpy",
            "platform",
            "torch_num_threads",
            "torch_num_interop_threads",
            "deterministic_algorithms",
            "cuda_visible_devices",
            "device",
        )
    }


def official_fixture_construction_count() -> int:
    """Return an outcome-free import/lifecycle diagnostic."""
    return _OFFICIAL_CONSTRUCTION_COUNT


def _load_attempt(expected_sha256: str | None = None) -> dict[str, Any]:
    if not ATTEMPT.is_file():
        raise ProtocolInvalid("official fixture requires the atomic attempt marker")
    digest = sha256_file(ATTEMPT)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProtocolInvalid("attempt marker digest differs from worker authorization")
    marker = strict_json_load_bound(ATTEMPT, digest)
    if not isinstance(marker, dict) or marker.get("artifact_type") != ATTEMPT_ARTIFACT_TYPE:
        raise ProtocolInvalid("invalid compact-training attempt marker")
    if marker.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ProtocolInvalid("attempt marker binds the wrong preregistration")
    seal = load_and_verify_seal()
    expected_keys = {
        "artifact_type",
        "timestamp_utc",
        "preregistration_sha256",
        "seal_file_sha256",
        "seal_payload_sha256",
        "source_aggregate",
        "environment_fingerprint",
        "seeds",
        "arms",
        "raw_path",
        "result_path",
        "command",
    }
    if set(marker) != expected_keys:
        raise ProtocolInvalid("attempt marker key set differs from the sealed protocol")
    expected_bindings = {
        "seal_file_sha256": seal["seal_file_sha256"],
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "source_aggregate": seal["source_aggregate"],
        "environment_fingerprint": _environment_fingerprint(seal["environment"]),
        "seeds": list(OFFICIAL_SEEDS),
        "arms": list(ARMS),
        "raw_path": str(RAW.relative_to(ROOT)),
        "result_path": str(RESULT.relative_to(ROOT)),
    }
    mismatches = {
        name: {"expected": value, "actual": marker.get(name)}
        for name, value in expected_bindings.items()
        if marker.get(name) != value
    }
    if mismatches:
        raise ProtocolInvalid(f"attempt marker/seal bindings differ: {mismatches}")
    command = marker.get("command")
    if (
        not isinstance(command, list)
        or not command
        or Path(str(command[0])).resolve() != Path(sys.executable).resolve()
    ):
        raise ProtocolInvalid("attempt marker command does not bind this interpreter")
    environment = _environment_metadata()
    _assert_official_environment(environment)
    if _environment_fingerprint(environment) != marker["environment_fingerprint"]:
        raise ProtocolInvalid("current environment differs from the sealed attempt marker")
    return marker


class _OfficialAuthorization:
    """Opaque marker-bound capability created only after the once-only marker exists."""

    __slots__ = ("marker_sha256", "seal_file_sha256", "_secret")

    def __init__(self, marker_sha256: str, seal_file_sha256: str, secret: object):
        self.marker_sha256 = marker_sha256
        self.seal_file_sha256 = seal_file_sha256
        self._secret = secret


def _authorize_official(expected_marker_sha256: str | None = None) -> _OfficialAuthorization:
    marker_sha256 = sha256_file(ATTEMPT) if ATTEMPT.is_file() else ""
    if expected_marker_sha256 is not None and marker_sha256 != expected_marker_sha256:
        raise ProtocolInvalid("attempt marker digest differs from worker authorization")
    marker = _load_attempt(marker_sha256)
    return _OfficialAuthorization(
        marker_sha256,
        marker["seal_file_sha256"],
        _OFFICIAL_AUTHORIZATION_SECRET,
    )


def _require_official_authorization(authorization: object) -> _OfficialAuthorization:
    if (
        not isinstance(authorization, _OfficialAuthorization)
        or authorization._secret is not _OFFICIAL_AUTHORIZATION_SECRET
    ):
        raise ProtocolInvalid("official construction requires marker-bound authorization")
    if (
        not ATTEMPT.is_file()
        or sha256_file(ATTEMPT) != authorization.marker_sha256
        or not SEAL.is_file()
        or sha256_file(SEAL) != authorization.seal_file_sha256
    ):
        raise ProtocolInvalid("official marker/seal changed after authorization")
    return authorization


def _literal_target(*, authorization: object) -> Gaussians3D:
    _require_official_authorization(authorization)
    means = torch.tensor(
        [
            [-0.31, -0.18, -0.22],
            [0.27, -0.13, 0.17],
            [-0.16, 0.29, 0.31],
            [0.35, 0.24, -0.36],
        ],
        dtype=torch.float32,
    )
    quats = torch.tensor(
        [
            [0.9659, 0.0000, 0.2588, 0.0000],
            [0.9239, 0.2706, 0.0000, 0.2706],
            [0.9511, -0.1816, 0.1816, 0.1667],
            [0.9063, 0.1094, -0.3790, 0.1510],
        ],
        dtype=torch.float32,
    )
    scales = torch.tensor(
        [[0.14, 0.09, 0.20], [0.11, 0.18, 0.08], [0.17, 0.10, 0.13], [0.09, 0.15, 0.19]],
        dtype=torch.float32,
    )
    opacity = torch.tensor([0.66, 0.54, 0.73, 0.47], dtype=torch.float32)
    colors = torch.tensor(
        [[0.78, 0.24, 0.18], [0.16, 0.72, 0.29], [0.20, 0.34, 0.82], [0.76, 0.68, 0.19]],
        dtype=torch.float32,
    )
    return Gaussians3D(
        means=means,
        quats=quats,
        log_scales=scales.log(),
        opacity=opacity,
        sh=rgb_to_sh(colors)[:, None, :],
    )


def _literal_cameras(*, authorization: object) -> list[Camera]:
    _require_official_authorization(authorization)
    positions = (
        torch.tensor([0.15, -0.10, -3.20]),
        torch.tensor([2.55, 0.45, -2.25]),
        torch.tensor([-2.35, 0.85, -2.45]),
    )
    targets = (
        torch.tensor([0.02, 0.03, 0.00]),
        torch.tensor([-0.03, 0.01, 0.04]),
        torch.tensor([0.04, -0.02, -0.03]),
    )
    return [
        Camera.look_at(position, target, fov_x_deg=52.0, width=32, height=32)
        for position, target in zip(positions, targets, strict=True)
    ]


def _project_target(
    target: Gaussians3D,
    camera: Camera,
    *,
    authorization: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project means/covariances with the point-renderer's frozen 0.3px dilation."""
    _require_official_authorization(authorization)
    means_camera = camera.world_to_cam(target.means)
    depths = means_camera[:, 2]
    xy, _ = camera.project(target.means)
    covariance_camera = camera.R @ target.covariance() @ camera.R.T
    jacobian = torch.zeros(target.n, 2, 3, dtype=torch.float32)
    jacobian[:, 0, 0] = camera.fx / depths
    jacobian[:, 0, 2] = -camera.fx * means_camera[:, 0] / depths.square()
    jacobian[:, 1, 1] = camera.fy / depths
    jacobian[:, 1, 2] = -camera.fy * means_camera[:, 1] / depths.square()
    covariance = jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
    covariance = covariance + 0.3 * torch.eye(2, dtype=torch.float32)
    return xy, covariance


def _field_from_projection(
    target: Gaussians3D,
    camera: Camera,
    view_index: int,
    *,
    authorization: object,
) -> GaussianObservationField:
    _require_official_authorization(authorization)
    xy, covariance = _project_target(target, camera, authorization=authorization)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    log_scales = eigenvalues.clamp_min(1e-12).sqrt().log()
    rotations = torch.atan2(eigenvectors[:, 1, 0], eigenvectors[:, 0, 0])
    colors = torch.tensor(
        [[0.78, 0.24, 0.18], [0.16, 0.72, 0.29], [0.20, 0.34, 0.82], [0.76, 0.68, 0.19]],
        dtype=torch.float32,
    )
    split_ids = ((0, 1, 2, 3), (0, 0, 1, 2, 3), (0, 0, 1, 2, 2, 3))[view_index]
    multiplicity = {component: split_ids.count(component) for component in set(split_ids)}
    ids = torch.tensor(split_ids, dtype=torch.long)
    amplitudes = torch.stack(
        [target.opacity[component] / multiplicity[component] for component in split_ids]
    )
    fixture_digest = sha256_bytes(b"rtgs-compact-point-training-synthetic-fixture-v1")
    config_digest = sha256_bytes(
        canonical_json(
            {
                "epsilon": 1e-8,
                "sigma_cutoff": math.sqrt(12.0),
                "support": "hard_rectangular",
                "aa_dilation": 0.0,
                "blend": "normalized",
            }
        ).encode("utf-8")
    )
    return GaussianObservationField(
        width=32,
        height=32,
        means=xy[ids],
        log_scales=log_scales[ids],
        rotations=rotations[ids],
        colors=colors[ids],
        amplitudes=amplitudes,
        blend_mode="normalized",
        epsilon=1e-8,
        sigma_cutoff=math.sqrt(12.0),
        support_fade_alpha=0.0,
        aa_dilation=0.0,
        view_id=f"synthetic-{view_index}",
        fit_window=(0, 0, 32, 32),
        n_init=8,
        provider="synthetic_fixture",
        producer_version="compact-point-training-fixture-v1",
        producer_source_digest=fixture_digest,
        fit_config_digest=config_digest,
    )


def _perturbed_initialization(
    target: Gaussians3D,
    seed: int,
    *,
    authorization: object,
) -> Gaussians3D:
    _require_official_authorization(authorization)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    row_sign = torch.tensor([-1.0, 1.0, -1.0, 1.0])[:, None]
    means = target.means + row_sign * torch.tensor([0.025, -0.018, 0.021])
    means = means + 0.018 * torch.randn(target.means.shape, generator=generator)
    quats = target.quats + 0.055 * torch.randn(target.quats.shape, generator=generator)
    log_scales = target.log_scales + 0.085 * torch.randn(
        target.log_scales.shape, generator=generator
    )
    opacity_logits = torch.logit(target.opacity) + 0.20 * torch.randn(
        target.opacity.shape, generator=generator
    )
    target_colors = torch.tensor(
        [[0.78, 0.24, 0.18], [0.16, 0.72, 0.29], [0.20, 0.34, 0.82], [0.76, 0.68, 0.19]],
        dtype=torch.float32,
    )
    colors = target_colors + 0.065 * torch.randn(target_colors.shape, generator=generator)
    return Gaussians3D(
        means=means,
        quats=quats,
        log_scales=log_scales,
        opacity=opacity_logits.sigmoid(),
        sh=rgb_to_sh(colors)[:, None, :],
    )


def renderer_boundary_distances(
    gaussians: Gaussians3D,
    camera: Camera,
    coordinates: torch.Tensor,
) -> dict[str, float]:
    """Distances from the actual hard surfaces used by the frozen point renderer/trainer."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] == 0:
        raise ValueError("boundary coordinates must have shape (S,2) with S>0")
    coordinates = coordinates.to(gaussians.means)
    means_camera = camera.world_to_cam(gaussians.means)
    depths = means_camera[:, 2]
    xy, _ = camera.project(gaussians.means)
    covariance_camera = (
        camera.R.to(gaussians.means) @ gaussians.covariance() @ camera.R.to(gaussians.means).T
    )
    zs = depths.clamp_min(_NEAR)
    jacobian = torch.zeros(
        gaussians.n,
        2,
        3,
        dtype=gaussians.means.dtype,
        device=gaussians.means.device,
    )
    jacobian[:, 0, 0] = camera.fx / zs
    jacobian[:, 0, 2] = -camera.fx * means_camera[:, 0] / zs.square()
    jacobian[:, 1, 1] = camera.fy / zs
    jacobian[:, 1, 2] = -camera.fy * means_camera[:, 1] / zs.square()
    covariance = jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
    covariance = covariance + _DILATION * torch.eye(
        2,
        dtype=gaussians.means.dtype,
        device=gaussians.means.device,
    )
    eigen_max = (
        0.5 * (covariance[:, 0, 0] + covariance[:, 1, 1])
        + (
            0.25 * (covariance[:, 0, 0] - covariance[:, 1, 1]).square()
            + covariance[:, 0, 1].square()
        ).sqrt()
    )
    radii = 3.0 * eigen_max.clamp_min(1e-8).sqrt()
    visibility_surfaces = torch.stack(
        [
            xy[:, 0] - (0.5 - radii),
            xy[:, 0] - (camera.width - 0.5 + radii),
            xy[:, 1] - (0.5 - radii),
            xy[:, 1] - (camera.height - 0.5 + radii),
        ],
        dim=1,
    )
    visible = (depths > _NEAR) & camera.in_image(xy, margin=radii)
    visible_indices = visible.nonzero(as_tuple=True)[0]
    if visible_indices.numel() == 0:
        raise ProtocolInvalid("boundary audit found no visible initialization Gaussian")
    visible_depths = depths[visible_indices]
    pairwise = (visible_depths[:, None] - visible_depths[None, :]).abs()
    pairwise.fill_diagonal_(math.inf)
    depth_tie = float(pairwise.min()) if visible_indices.numel() > 1 else math.inf
    covariance_visible = covariance[visible_indices]
    determinant = (
        covariance_visible[:, 0, 0] * covariance_visible[:, 1, 1]
        - covariance_visible[:, 0, 1].square()
    ).clamp_min(1e-12)
    i00 = covariance_visible[:, 1, 1] / determinant
    i01 = -covariance_visible[:, 0, 1] / determinant
    i11 = covariance_visible[:, 0, 0] / determinant
    delta = coordinates[:, None, :] - xy[visible_indices][None, :, :]
    q = (
        delta[..., 0].square() * i00[None]
        + 2.0 * delta[..., 0] * delta[..., 1] * i01[None]
        + delta[..., 1].square() * i11[None]
    )
    active = q < KERNEL_SUPPORT_CUTOFF
    if not bool(active.any()):
        raise ProtocolInvalid("boundary audit found no active initialization score coordinate")
    weights = kernel_support_weight(q, "hard")
    trained_opacity = gaussians.opacity.clamp(1e-4, 1.0 - 1e-4)
    raw_alpha = trained_opacity[visible_indices][None] * weights
    view_directions = torch.nn.functional.normalize(
        gaussians.means[visible_indices] - camera.position.to(gaussians.means)[None], dim=-1
    )
    sh_preactivation = eval_sh_preactivation(
        0,
        gaussians.sh[visible_indices],
        view_directions,
    )
    return {
        "camera_depth_tie": depth_tie,
        "near_plane_0p05": float((depths - _NEAR).abs().min()),
        "visibility_envelope": float(visibility_surfaces.abs().min()),
        "student_hard_support_q12_all_score_coordinates": float(
            (q - KERNEL_SUPPORT_CUTOFF).abs().min()
        ),
        "trainer_opacity_logit_clamp_1e_4_1m1e_4": min(
            float((trained_opacity - 1e-4).abs().min()),
            float((trained_opacity - (1.0 - 1e-4)).abs().min()),
        ),
        "renderer_alpha_cap_0p999_active_coordinates": float(
            (raw_alpha[active] - _MAX_ALPHA).abs().min()
        ),
        "sh_hard_floor_zero": float(sh_preactivation.abs().min()),
        "quaternion_norm_zero": float(gaussians.quats.norm(dim=-1).min()),
    }


def _fixture_nonvacuity(
    target: Gaussians3D,
    inputs: ReconstructionInputs,
    initialization: Gaussians3D,
    *,
    authorization: object,
) -> dict[str, Any]:
    _require_official_authorization(authorization)
    if inputs.n_opt_2d != list(OFFICIAL_N_OPT_2D) or inputs.n_init_2d != list(OFFICIAL_N_INIT_2D):
        raise ProtocolInvalid("official 2D budgets differ from the preregistration")
    if target.n != OFFICIAL_N_3D or initialization.n != OFFICIAL_N_3D:
        raise ProtocolInvalid("official 3D topology differs from four")
    records = []
    minimum_distances: dict[str, float] = {}
    for camera, field in zip(inputs.cameras, inputs.observations, strict=True):
        coordinates = torch.cat(
            [
                *list(_coordinate_chunks(field, "pixel", 4096)),
                *list(_coordinate_chunks(field, "area", 4096)),
            ]
        )
        renderer_distances = renderer_boundary_distances(initialization, camera, coordinates)
        for name, value in renderer_distances.items():
            minimum_distances[name] = min(minimum_distances.get(name, math.inf), value)
        xy, covariance = _project_target(target, camera, authorization=authorization)
        records.append(
            {
                "initialization_camera_depths": camera.world_to_cam(initialization.means)[
                    :, 2
                ].tolist(),
                "camera_xy_sha256": tensor_hash(xy),
                "projected_covariance_sha256": tensor_hash(covariance),
                "teacher_sha256": field_hashes(field)["aggregate"],
                "renderer_boundary_distances": renderer_distances,
                "score_coordinate_sha256": tensor_hash(coordinates),
            }
        )
    distances = dict(minimum_distances)
    if min(distances.values()) <= 1e-5:
        raise ProtocolInvalid(f"official fixture is vacuous at a hard boundary: {distances}")
    return {
        "minimum_boundary_distances": distances,
        "views": records,
    }


def official_fixture(
    seed: int,
    *,
    expected_attempt_sha256: str | None = None,
) -> tuple[ReconstructionInputs, Gaussians3D, dict[str, Any]]:
    """Construct the literal fixture only after validating the atomic marker."""
    authorization = _authorize_official(expected_attempt_sha256)
    return _construct_official_fixture(seed, authorization=authorization)


def _construct_official_fixture(
    seed: int,
    *,
    authorization: object,
) -> tuple[ReconstructionInputs, Gaussians3D, dict[str, Any]]:
    """Construct after validating the marker-bound capability."""
    global _OFFICIAL_CONSTRUCTION_COUNT
    _require_official_authorization(authorization)
    if seed not in OFFICIAL_SEEDS:
        raise ValueError("not a frozen official seed")
    _OFFICIAL_CONSTRUCTION_COUNT += 1
    target = _literal_target(authorization=authorization)
    cameras = _literal_cameras(authorization=authorization)
    fields = [
        _field_from_projection(
            target,
            camera,
            index,
            authorization=authorization,
        )
        for index, camera in enumerate(cameras)
    ]
    inputs = ReconstructionInputs(
        observations=fields,
        cameras=cameras,
        view_names=[f"synthetic-{index}" for index in range(3)],
        points=None,
        point_visibility=None,
        bounds_hint=None,
        name="compact-point-training-official-synthetic",
    )
    initialization = _perturbed_initialization(
        target,
        seed,
        authorization=authorization,
    )
    record = {
        "seed": seed,
        "target": gaussians_hashes(target),
        "initialization": gaussians_hashes(initialization),
        "teachers": [field_hashes(field) for field in fields],
        "cameras": [
            {
                "fx": camera.fx,
                "fy": camera.fy,
                "cx": camera.cx,
                "cy": camera.cy,
                "width": camera.width,
                "height": camera.height,
                "R_sha256": tensor_hash(camera.R),
                "t_sha256": tensor_hash(camera.t),
            }
            for camera in cameras
        ],
        "nonvacuity": _fixture_nonvacuity(
            target,
            inputs,
            initialization,
            authorization=authorization,
        ),
    }
    record["aggregate_sha256"] = canonical_hash(record)
    return inputs, initialization, record


def _coordinate_chunks(
    field: GaussianObservationField,
    risk: str,
    chunk_size: int,
) -> Iterable[torch.Tensor]:
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    offsets = (
        ((0.5, 0.5),)
        if risk == "pixel"
        else (
            (0.25, 0.25),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.75, 0.75),
        )
    )
    buffer: list[tuple[float, float]] = []
    for pixel_y in range(fit_y, fit_y + fit_height):
        for pixel_x in range(fit_x, fit_x + fit_width):
            for offset_x, offset_y in offsets:
                buffer.append((pixel_x + offset_x, pixel_y + offset_y))
                if len(buffer) == chunk_size:
                    yield torch.tensor(buffer, dtype=field.dtype)
                    buffer.clear()
    if buffer:
        yield torch.tensor(buffer, dtype=field.dtype)


def evaluate_risks(
    inputs: ReconstructionInputs,
    gaussians: Gaussians3D,
    *,
    point_rasterizer: TorchPointRasterizer | None = None,
    query_backends: Sequence[GaussianObservationIndex] | None = None,
    evaluation_chunk: int = 256,
    query_component_chunk: int = 64,
) -> dict[str, Any]:
    """Stream exact equal-view pixel risk and frozen four-offset area quadrature."""
    if evaluation_chunk <= 0 or query_component_chunk <= 0:
        raise ValueError("evaluation/query chunks must be positive")
    renderer = point_rasterizer or TorchPointRasterizer(point_chunk=32, gaussian_chunk=64)
    backends = (
        list(query_backends)
        if query_backends is not None
        else [GaussianObservationIndex(field, tile_size=8) for field in inputs.observations]
    )
    if len(backends) != inputs.n_views:
        raise ValueError("evaluation needs one query backend per view")
    teacher_before = [field_hashes(field) for field in inputs.observations]
    result: dict[str, Any] = {"pixel": [], "area": []}
    started = time.perf_counter()
    with torch.no_grad():
        for risk in ("pixel", "area"):
            for view, (field, camera, backend) in enumerate(
                zip(inputs.observations, inputs.cameras, backends, strict=True)
            ):
                sse = 0.0
                point_count = 0
                teacher_below = teacher_above = prediction_below = prediction_above = 0
                view_started = time.perf_counter()
                for xy in _coordinate_chunks(field, risk, evaluation_chunk):
                    teacher = backend.query(xy, component_chunk=query_component_chunk).color
                    prediction = renderer.render_points(
                        gaussians,
                        camera,
                        xy.to(gaussians.means),
                        background=torch.zeros(3, dtype=gaussians.means.dtype),
                        sh_degree=0,
                    ).color
                    difference = prediction.to(torch.float64) - teacher.to(torch.float64)
                    sse += float(difference.square().sum(dtype=torch.float64))
                    point_count += int(xy.shape[0])
                    teacher_below += int((teacher < 0).sum())
                    teacher_above += int((teacher > 1).sum())
                    prediction_below += int((prediction < 0).sum())
                    prediction_above += int((prediction > 1).sum())
                expected = field.fit_window[2] * field.fit_window[3] * (1 if risk == "pixel" else 4)
                if point_count != expected:
                    raise ProtocolInvalid(f"{risk} evaluation point count differs from protocol")
                result[risk].append(
                    {
                        "view": view,
                        "rgb_squared_error_sum": sse,
                        "point_count": point_count,
                        "risk": sse / (3.0 * point_count),
                        "teacher_below_zero_fraction": teacher_below / (3.0 * point_count),
                        "teacher_above_one_fraction": teacher_above / (3.0 * point_count),
                        "prediction_below_zero_fraction": prediction_below / (3.0 * point_count),
                        "prediction_above_one_fraction": prediction_above / (3.0 * point_count),
                        "elapsed_seconds": time.perf_counter() - view_started,
                    }
                )
    teacher_after = [field_hashes(field) for field in inputs.observations]
    if teacher_before != teacher_after:
        raise ProtocolInvalid("evaluation mutated a frozen teacher")
    means = {
        risk: sum(item["risk"] for item in result[risk]) / inputs.n_views
        for risk in ("pixel", "area")
    }
    payload = {
        "equal_view_risk": means,
        "per_view": result,
        "teacher_hashes_before": teacher_before,
        "teacher_hashes_after": teacher_after,
        "evaluation_chunk": evaluation_chunk,
        "query_component_chunk": query_component_chunk,
        "elapsed_seconds": time.perf_counter() - started,
    }
    assert_finite_tree(payload, "evaluation")
    return payload


def normalized_log_auc(checkpoints: Sequence[Mapping[str, Any]], risk: str) -> float:
    steps = [int(checkpoint["step"]) for checkpoint in checkpoints]
    if steps != list(CHECKPOINTS):
        raise ProtocolInvalid("checkpoint schedule differs from (0,30,60,120)")
    initial = float(checkpoints[0]["risks"][risk])
    if initial < 0 or not math.isfinite(initial):
        raise ProtocolInvalid("initial risk is invalid")
    values = [
        math.log((float(checkpoint["risks"][risk]) + 1e-12) / (initial + 1e-12))
        for checkpoint in checkpoints
    ]
    area = 0.0
    for left, right, value_left, value_right in zip(
        CHECKPOINTS[:-1], CHECKPOINTS[1:], values[:-1], values[1:], strict=True
    ):
        area += ((right - left) / ITERATIONS) * (value_left + value_right) / 2.0
    return area


def classify_domain(
    uniform_records: Sequence[Mapping[str, Any]],
    mixture_records: Sequence[Mapping[str, Any]],
    *,
    risk: str,
) -> dict[str, Any]:
    if len(uniform_records) != len(OFFICIAL_SEEDS) or len(mixture_records) != len(OFFICIAL_SEEDS):
        raise ProtocolInvalid("domain comparison requires all three official seeds")
    uniform = {int(item["seed"]): item for item in uniform_records}
    mixture = {int(item["seed"]): item for item in mixture_records}
    if set(uniform) != set(OFFICIAL_SEEDS) or set(mixture) != set(OFFICIAL_SEEDS):
        raise ProtocolInvalid("domain comparison seed set differs from preregistration")
    per_seed = []
    for seed in OFFICIAL_SEEDS:
        u = uniform[seed]
        m = mixture[seed]
        u0 = float(u["checkpoints"][0]["risks"][risk])
        u_final = float(u["checkpoints"][-1]["risks"][risk])
        m_final = float(m["checkpoints"][-1]["risks"][risk])
        auc_u = normalized_log_auc(u["checkpoints"], risk)
        auc_m = normalized_log_auc(m["checkpoints"], risk)
        per_seed.append(
            {
                "seed": seed,
                "uniform_auc": auc_u,
                "mixture_auc": auc_m,
                "delta_auc": auc_m - auc_u,
                "q_init": (u_final + 1e-12) / (u0 + 1e-12),
                "q_final": (m_final + 1e-12) / (u_final + 1e-12),
                "direction": "mixture_better"
                if auc_m < auc_u
                else ("uniform_better" if auc_m > auc_u else "tie"),
            }
        )
    geometric = lambda name: math.exp(  # noqa: E731 - keeps formula visibly local
        sum(math.log(float(item[name])) for item in per_seed) / len(per_seed)
    )
    g_init = geometric("q_init")
    g_final = geometric("q_final")
    g_auc = math.exp(sum(float(item["delta_auc"]) for item in per_seed) / len(per_seed))
    if g_init > 0.90 or sum(float(item["q_init"]) < 1.0 for item in per_seed) < 2:
        label = "INCONCLUSIVE_TRAINER"
    elif (
        g_auc <= 0.95
        and g_final <= 1.05
        and sum(float(item["delta_auc"]) < 0.0 for item in per_seed) >= 2
        and max(float(item["q_final"]) for item in per_seed) <= 1.20
    ):
        label = "MATERIAL_SAMPLING_WIN"
    elif (
        g_auc <= 1.05
        and g_final <= 1.05
        and max(float(item["q_final"]) for item in per_seed) <= 1.20
    ):
        label = "NONINFERIOR"
    else:
        label = "NEUTRAL_OR_NEGATIVE"
    return {
        "risk": risk,
        "G_init": g_init,
        "G_final": g_final,
        "G_AUC": g_auc,
        "label": label,
        "per_seed": per_seed,
    }


def _strict_record_map(raw: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != len(OFFICIAL_SEEDS) * len(ARMS):
        raise ProtocolInvalid("RAW does not contain exactly twelve seed--arm records")
    mapping: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolInvalid("RAW record is not an object")
        key = (int(record.get("seed", -1)), str(record.get("arm")))
        if key in mapping or key[0] not in OFFICIAL_SEEDS or key[1] not in ARMS:
            raise ProtocolInvalid(f"RAW has an unexpected/duplicate record {key}")
        mapping[key] = record
    expected = {(seed, arm) for seed in OFFICIAL_SEEDS for arm in ARMS}
    if set(mapping) != expected:
        raise ProtocolInvalid("RAW seed--arm Cartesian product is incomplete")
    return mapping


def validate_raw(raw: Mapping[str, Any], *, marker_sha256: str, seal: Mapping[str, Any]) -> None:
    required = {
        "artifact_type",
        "status",
        "timestamp_utc",
        "preregistration_sha256",
        "seal_file_sha256",
        "seal_payload_sha256",
        "attempt_sha256",
        "source_aggregate",
        "environment_fingerprint",
        "records",
    }
    if set(raw) != required:
        raise ProtocolInvalid(f"RAW key set differs: {sorted(set(raw) ^ required)}")
    if raw["artifact_type"] != RAW_ARTIFACT_TYPE or raw["status"] != "PASS":
        raise ProtocolInvalid("RAW is not a protocol-valid PASS artifact")
    bindings = {
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_file_sha256": seal["seal_file_sha256"],
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "attempt_sha256": marker_sha256,
        "source_aggregate": seal["source_aggregate"],
    }
    for name, expected in bindings.items():
        if raw.get(name) != expected:
            raise ProtocolInvalid(f"RAW {name} binding differs")
    assert_finite_tree(raw, "RAW")
    mapping = _strict_record_map(raw)
    for seed in OFFICIAL_SEEDS:
        initialization_hashes = {str(mapping[seed, arm]["initialization_hash"]) for arm in ARMS}
        teacher_hashes = {
            canonical_hash(mapping[seed, arm]["teacher_hashes_before"]) for arm in ARMS
        }
        view_schedules = {canonical_hash(mapping[seed, arm]["view_schedule"]) for arm in ARMS}
        if len(initialization_hashes) != 1 or len(teacher_hashes) != 1 or len(view_schedules) != 1:
            raise ProtocolInvalid(f"seed {seed} arms do not share initialization/teachers/views")
        for arm in ARMS:
            record = mapping[seed, arm]
            if record.get("status") != "PASS" or int(record.get("optimizer_steps", -1)) != 120:
                raise ProtocolInvalid(f"seed {seed}/{arm} did not complete 120 updates")
            if int(record.get("n_initial", -1)) != 4 or int(record.get("n_final", -1)) != 4:
                raise ProtocolInvalid(f"seed {seed}/{arm} changed fixed topology")
            if record.get("teacher_hashes_before") != record.get("teacher_hashes_after"):
                raise ProtocolInvalid(f"seed {seed}/{arm} mutated a teacher")
            if [int(item["step"]) for item in record.get("checkpoints", [])] != list(CHECKPOINTS):
                raise ProtocolInvalid(f"seed {seed}/{arm} checkpoint schedule differs")
            if int(record.get("attempt_count", -1)) != ITERATIONS * ATTEMPTS_PER_STEP:
                raise ProtocolInvalid(f"seed {seed}/{arm} did not use fixed attempts")
            maximum_importance = float(record.get("maximum_importance", math.inf))
            if maximum_importance > 4.00001:
                raise ProtocolInvalid(f"seed {seed}/{arm} exceeded bounded importance")
            if record.get("rgb_access_count") != 0:
                raise ProtocolInvalid(f"seed {seed}/{arm} accessed RGB/source images")
            motions = record.get("parameter_motion", {})
            if set(motions) != set(MOTION_FAMILIES) or any(
                not math.isfinite(float(motions[name])) or float(motions[name]) <= 1e-8
                for name in MOTION_FAMILIES
            ):
                raise ProtocolInvalid(f"seed {seed}/{arm} omitted parameter-family motion")
            clocks = record.get("optimizer_group_clocks")
            if (
                not isinstance(clocks, Mapping)
                or set(clocks)
                != {
                    "means",
                    "quats",
                    "scales",
                    "opacities",
                    "sh0",
                    "shN",
                }
                or any(int(value) != ITERATIONS for value in clocks.values())
            ):
                raise ProtocolInvalid(f"seed {seed}/{arm} Adam clocks are not aligned")
            schedule = record.get("view_schedule")
            if not isinstance(schedule, list) or len(schedule) != ITERATIONS:
                raise ProtocolInvalid(f"seed {seed}/{arm} view schedule length differs")
            if record.get("view_schedule_sha256") != canonical_hash(schedule):
                raise ProtocolInvalid(f"seed {seed}/{arm} view schedule hash differs")
            expected_risk = "discrete_pixels" if arm.startswith("pixel") else "continuous_area"
            if record.get("risk_measure") != expected_risk:
                raise ProtocolInvalid(f"seed {seed}/{arm} risk measure differs")
            invariants = record.get("proposal_invariants")
            required_invariants = {
                "active_implies_inside_fit_window",
                "invalid_zero_joint_density",
                "invalid_zero_target_density",
                "invalid_zero_proposal_density",
                "invalid_zero_importance",
                "no_null_resampling",
            }
            if not isinstance(invariants, Mapping) or not required_invariants.issubset(invariants):
                raise ProtocolInvalid(f"seed {seed}/{arm} omitted proposal invariants")
            if not all(invariants[name] is True for name in required_invariants):
                raise ProtocolInvalid(f"seed {seed}/{arm} proposal invariant failed")
            branches = record.get("proposal_branch_counts")
            if not isinstance(branches, Mapping):
                raise ProtocolInvalid(f"seed {seed}/{arm} omitted proposal branch counts")
            if arm.endswith("uniform"):
                if (
                    float(record["active_fraction"]) != 1.0
                    or float(record["maximum_importance"]) != 1.0
                    or int(branches.get("uniform", -1)) != ITERATIONS * ATTEMPTS_PER_STEP
                ):
                    raise ProtocolInvalid(f"seed {seed}/{arm} uniform sampling is not exact")
            elif any(
                int(branches.get(name, 0)) <= 0
                for name in ("uniform", "gaussian", "gaussian_accepted", "gaussian_rejected")
            ):
                raise ProtocolInvalid(f"seed {seed}/{arm} mixture branches are incomplete")
        initial_risks = {
            canonical_hash(mapping[seed, arm]["checkpoints"][0]["risks"]) for arm in ARMS
        }
        if len(initial_risks) != 1:
            raise ProtocolInvalid(f"seed {seed} initial exact risks differ across arms")


def result_from_raw(
    raw: Mapping[str, Any],
    *,
    raw_sha256: str,
    marker_sha256: str,
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every scientific statistic exclusively from strict-reloaded RAW."""
    validate_raw(raw, marker_sha256=marker_sha256, seal=seal)
    mapping = _strict_record_map(raw)
    domains = {}
    for name, (uniform_arm, mixture_arm, risk) in DOMAIN_PAIRS.items():
        domains[name] = classify_domain(
            [mapping[seed, uniform_arm] for seed in OFFICIAL_SEEDS],
            [mapping[seed, mixture_arm] for seed in OFFICIAL_SEEDS],
            risk=risk,
        )
    global_decision = (
        "SAMPLING_WIN"
        if all(item["label"] == "MATERIAL_SAMPLING_WIN" for item in domains.values())
        else "NO_GLOBAL_SAMPLING_WIN"
    )
    summaries = []
    for seed in OFFICIAL_SEEDS:
        for arm in ARMS:
            record = mapping[seed, arm]
            summaries.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "active_fraction": record["active_fraction"],
                    "importance_ess_per_attempt": record["importance_ess_per_attempt"],
                    "wall_seconds": record["wall_seconds"],
                    "peak_rss_kib": record["peak_rss_kib"],
                    "parameter_motion": record["parameter_motion"],
                    "common_pixel_checkpoints": [
                        {"step": item["step"], "risk": item["risks"]["pixel"]}
                        for item in record["checkpoints"]
                    ],
                }
            )
    result = {
        "artifact_type": RESULT_ARTIFACT_TYPE,
        "status": "PASS",
        "decision": global_decision,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim_boundary": (
            "CPU synthetic fixed-topology point-training mechanism only; continuous and discrete "
            "pairs retain distinct risks; no quality, density, speed, memory, CUDA, or default "
            "claim"
        ),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_file_sha256": seal["seal_file_sha256"],
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "source_aggregate": seal["source_aggregate"],
        "attempt_sha256": marker_sha256,
        "raw_sha256": raw_sha256,
        "domains": domains,
        "records": summaries,
        "thresholds": {
            "material_auc": 0.95,
            "noninferior_auc": 1.05,
            "final_geometric": 1.05,
            "maximum_seed_final": 1.20,
            "trainer_reduction": 0.90,
        },
    }
    assert_finite_tree(result, "RESULT")
    return result


def _implementation_review_text(path: Path = ROOT / IMPLEMENTATION_REVIEW) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            "outcome-blind implementation review is missing; sealing is forbidden"
        )
    return path.read_text(encoding="utf-8")


def _require_pass_review(
    text: str,
    *,
    name: str,
    required_hashes: Sequence[str],
) -> None:
    if re.search(r"verdict\s*:\s*`?PASS`?", text, flags=re.IGNORECASE) is None:
        raise ProtocolInvalid(f"{name} review does not contain Verdict: PASS")
    missing = [digest for digest in required_hashes if digest not in text]
    if missing:
        raise ProtocolInvalid(f"{name} review is missing frozen hash bindings: {missing}")
    unresolved = re.search(r"unresolved findings\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if unresolved is None or unresolved.group(1).strip().lower().strip("`*_. ") != "none":
        raise ProtocolInvalid(f"{name} review has unresolved findings")


def _run_preseal_verification(
    expected_hashes: Mapping[str, str],
    expected_aggregate: str,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    commands = (
        [sys.executable, "-m", "pytest", "-q", str(FOCUSED_TEST)],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "pytest", "-q"],
    )
    records = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        record = {
            "command": list(command),
            "returncode": completed.returncode,
            "output_sha256": sha256_bytes(output.encode("utf-8")),
        }
        records.append(record)
        if completed.returncode != 0:
            raise ProtocolInvalid(f"preseal verification failed: {command}\n{output}")
        verify_source_hashes(expected_hashes, expected_aggregate)
    return {"passed": True, "commands": records}


def _official_namespace_collisions(*, include_seal: bool) -> list[Path]:
    results = ROOT / "benchmarks/results"
    patterns = (
        "20260716_compact_point_training_SEAL*",
        "20260716_compact_point_training_ATTEMPT*",
        "20260716_compact_point_training_RAW*",
        "20260716_compact_point_training_RESULT*",
        "20260716_compact_point_training_AUDIT*",
    )
    collisions: set[Path] = set()
    for pattern in patterns:
        collisions.update(
            path for path in results.glob(pattern) if os.path.lexists(os.fspath(path))
        )
    if not include_seal:
        collisions.discard(SEAL)
    return sorted(collisions)


def create_seal() -> dict[str, Any]:
    if sha256_file(ROOT / PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ProtocolInvalid("preregistration differs from frozen SHA-256")
    preregistration_review = (ROOT / PREREGISTRATION_REVIEW).read_text(encoding="utf-8")
    _require_pass_review(
        preregistration_review,
        name="preregistration",
        required_hashes=(PREREGISTRATION_SHA256,),
    )
    collisions = _official_namespace_collisions(include_seal=True)
    if collisions:
        raise FileExistsError(f"refusing to seal over experiment artifacts: {collisions}")
    reviewed_hashes, reviewed_aggregate = source_hashes()
    implementation_review = _implementation_review_text()
    _require_pass_review(
        implementation_review,
        name="implementation",
        required_hashes=(PREREGISTRATION_SHA256, reviewed_aggregate),
    )
    hashes, aggregate = sealed_source_hashes()
    verification = _run_preseal_verification(hashes, aggregate)
    _configure_official_runtime()
    environment = _environment_metadata()
    _assert_official_environment(environment)
    payload: dict[str, Any] = {
        "artifact_type": SEAL_ARTIFACT_TYPE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
        },
        "preregistration_review": {
            "path": str(PREREGISTRATION_REVIEW),
            "sha256": sha256_file(ROOT / PREREGISTRATION_REVIEW),
        },
        "implementation_review": {
            "path": str(IMPLEMENTATION_REVIEW),
            "sha256": sha256_file(ROOT / IMPLEMENTATION_REVIEW),
        },
        "source_hashes": hashes,
        "source_aggregate": aggregate,
        "reviewed_source_hashes": reviewed_hashes,
        "reviewed_source_aggregate": reviewed_aggregate,
        "verification": verification,
        "git": _git_metadata(),
        "environment": environment,
        "official_protocol": {
            "seeds": list(OFFICIAL_SEEDS),
            "arms": list(ARMS),
            "iterations": ITERATIONS,
            "attempts_per_step": ATTEMPTS_PER_STEP,
            "checkpoints": list(CHECKPOINTS),
            "eta": ETA,
        },
        "command": [sys.executable, *sys.argv],
    }
    payload["seal_payload_sha256"] = canonical_hash(payload)
    return payload


def load_and_verify_seal() -> dict[str, Any]:
    seal = strict_json_load(SEAL)
    if not isinstance(seal, dict) or seal.get("artifact_type") != SEAL_ARTIFACT_TYPE:
        raise ProtocolInvalid("invalid compact-training implementation seal")
    expected = seal.get("seal_payload_sha256")
    body = dict(seal)
    body.pop("seal_payload_sha256", None)
    if expected != canonical_hash(body):
        raise ProtocolInvalid("seal self-digest mismatch")
    if seal.get("preregistration", {}).get("sha256") != PREREGISTRATION_SHA256:
        raise ProtocolInvalid("seal binds the wrong preregistration")
    if seal.get("verification", {}).get("passed") is not True:
        raise ProtocolInvalid("seal does not bind passing verification")
    verify_source_hashes(seal["source_hashes"], seal["source_aggregate"])
    _verify_git_binding(seal["git"])
    if sha256_file(ROOT / IMPLEMENTATION_REVIEW) != seal["implementation_review"]["sha256"]:
        raise ProtocolInvalid("implementation review differs from the seal")
    seal["seal_file_sha256"] = sha256_file(SEAL)
    return seal


def _claim_attempt(
    seal: Mapping[str, Any], environment: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    collisions = _official_namespace_collisions(include_seal=False)
    if collisions:
        raise FileExistsError(f"once-only namespace already consumed: {collisions}")
    if _environment_fingerprint(environment) != _environment_fingerprint(seal["environment"]):
        raise ProtocolInvalid("run environment differs from sealed environment")
    payload = {
        "artifact_type": ATTEMPT_ARTIFACT_TYPE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_file_sha256": seal["seal_file_sha256"],
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "source_aggregate": seal["source_aggregate"],
        "environment_fingerprint": _environment_fingerprint(environment),
        "seeds": list(OFFICIAL_SEEDS),
        "arms": list(ARMS),
        "raw_path": str(RAW.relative_to(ROOT)),
        "result_path": str(RESULT.relative_to(ROOT)),
        "command": [sys.executable, *sys.argv],
    }
    digest = _exclusive_json(ATTEMPT, payload)
    return payload, digest


def _official_config(arm: str, seed: int, *, authorization: object) -> Any:
    from rtgs.optim.compact_trainer import CompactTrainConfig

    _require_official_authorization(authorization)
    if arm not in ARMS or seed not in OFFICIAL_SEEDS:
        raise ValueError("official config requires a frozen seed and arm")
    return CompactTrainConfig(
        iterations=120,
        attempts_per_step=128,
        proposal_mode=arm,
        uniform_fraction=1.0 if arm.endswith("uniform") else 0.25,
        seed=seed,
        extent=1.0,
        device="cpu",
        lr_means=1.6e-4,
        lr_quats=1e-3,
        lr_scales=5e-3,
        lr_opacity=5e-2,
        lr_sh=2.5e-3,
        lr_sh_rest=2.5e-3 / 20.0,
        point_chunk=32,
        gaussian_chunk=64,
        outer_microbatch=32,
        query_component_chunk=64,
        teacher_tile_size=8,
        evaluation_chunk=256,
        checkpoints=(0, 30, 60, 120),
        sh_degree=0,
        sh_color_activation="hard",
        sh_smu1_mu=2.0 / 255.0,
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
        max_views=MAX_VIEWS,
        max_fitted_pixels_per_view=MAX_FITTED_PIXELS_PER_VIEW,
        max_components_per_view=MAX_COMPONENTS_PER_VIEW,
        max_index_entries_per_view=MAX_TILE_ENTRIES_PER_VIEW,
        max_candidates_per_tile=MAX_CANDIDATES_PER_TILE,
        max_manifest_bytes=MAX_MANIFEST_BYTES,
        max_teacher_archives=MAX_TEACHER_ARCHIVES,
        max_teacher_archive_bytes=MAX_COMPRESSED_ARCHIVE_BYTES,
        max_total_teacher_archive_bytes=MAX_TOTAL_COMPRESSED_BYTES,
        max_zip_members=MAX_ZIP_MEMBERS,
        max_member_uncompressed_bytes=MAX_UNCOMPRESSED_MEMBER_BYTES,
        max_archive_uncompressed_bytes=MAX_UNCOMPRESSED_ARCHIVE_BYTES,
    )


def _checkpoint_records(history: Mapping[str, Any]) -> list[dict[str, Any]]:
    checkpoints = history.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ProtocolInvalid("trainer history is missing checkpoint evaluations")
    records = []
    for checkpoint in checkpoints:
        evaluation = checkpoint["evaluation"]
        risks = {
            "pixel": float(evaluation["J_pixel"]),
            "area": float(evaluation["J_area"]),
        }
        records.append(
            {
                "step": int(checkpoint["step"]),
                "risks": risks,
                "evaluation": evaluation,
            }
        )
    if [record["step"] for record in records] != list(CHECKPOINTS):
        raise ProtocolInvalid("trainer returned the wrong checkpoint schedule")
    return records


def _motion_values(history: Mapping[str, Any]) -> dict[str, float]:
    source = history.get("parameter_motion")
    if not isinstance(source, Mapping) or not set(MOTION_FAMILIES).issubset(source):
        raise ProtocolInvalid("trainer omitted an effective parameter-family motion")
    if "shN" in source and float(source["shN"]["max_abs"]) != 0.0:
        raise ProtocolInvalid("degree-zero official run changed the empty shN family")
    values = {name: float(source[name]["max_abs"]) for name in MOTION_FAMILIES}
    if any(not math.isfinite(value) or value <= 1e-8 for value in values.values()):
        raise ProtocolInvalid(
            f"official training has vacuous/non-finite parameter motion: {values}"
        )
    return values


def _optimizer_clock(history: Mapping[str, Any]) -> dict[str, int]:
    clocks = history.get("optimizer_steps")
    expected = {"means", "quats", "scales", "opacities", "sh0", "shN"}
    if not isinstance(clocks, Mapping) or set(clocks) != expected:
        raise ProtocolInvalid("trainer optimizer clocks differ from six frozen groups")
    converted = {str(name): int(value) for name, value in clocks.items()}
    if any(value != ITERATIONS for value in converted.values()):
        raise ProtocolInvalid(f"trainer Adam clocks are not aligned: {converted}")
    return converted


def _validate_training_steps(history: Mapping[str, Any], arm: str) -> dict[str, Any]:
    steps = history.get("steps")
    if not isinstance(steps, list) or len(steps) != ITERATIONS:
        raise ProtocolInvalid("trainer did not record exactly 120 steps")
    total_attempts = total_active = total_ess = 0.0
    maximum_importance = 0.0
    for expected_step, item in enumerate(steps, start=1):
        if int(item["step"]) != expected_step or int(item["attempts"]) != ATTEMPTS_PER_STEP:
            raise ProtocolInvalid("trainer step/attempt accounting differs")
        attempts = int(item["attempts"])
        active = int(item["active_count"])
        null = int(item["null_count"])
        invalid = int(item["invalid_count"])
        if active + null != attempts or invalid > null:
            raise ProtocolInvalid("active/null/invalid accounting is inconsistent")
        if (
            int(item["teacher_query_attempts"]) != attempts
            or int(item["student_query_attempts"]) != attempts
        ):
            raise ProtocolInvalid("teacher/student did not receive every attempted coordinate")
        if int(item["teacher_query_calls"]) != int(item["student_query_calls"]):
            raise ProtocolInvalid("teacher/student outer query calls are not matched")
        maximum_importance = max(maximum_importance, float(item["importance_max"]))
        total_attempts += attempts
        total_active += active
        total_ess += float(item["importance_ess"])
        if int(item["cardinality"]) != OFFICIAL_N_3D:
            raise ProtocolInvalid("fixed topology changed inside a step")
    if maximum_importance > 4.00001:
        raise ProtocolInvalid("importance exceeded 1/eta")
    branch_counts = history.get("proposal_branch_counts")
    proposal_invariants = history.get("proposal_invariants")
    if not isinstance(branch_counts, Mapping) or not isinstance(proposal_invariants, Mapping):
        raise ProtocolInvalid("trainer omitted proposal branch/invariant diagnostics")
    required_invariants = {
        "active_implies_inside_fit_window",
        "invalid_zero_joint_density",
        "invalid_zero_target_density",
        "invalid_zero_proposal_density",
        "invalid_zero_importance",
        "no_null_resampling",
    }
    if not required_invariants.issubset(proposal_invariants) or not all(
        proposal_invariants[name] is True for name in required_invariants
    ):
        raise ProtocolInvalid(f"proposal invariants failed: {proposal_invariants}")
    if arm.endswith("uniform"):
        if total_active != total_attempts or maximum_importance != 1.0:
            raise ProtocolInvalid("uniform arm is not exact active unit-importance sampling")
        if int(branch_counts.get("uniform", -1)) != int(total_attempts):
            raise ProtocolInvalid("uniform arm branch count differs from attempts")
    else:
        required = ("uniform", "gaussian", "gaussian_accepted", "gaussian_rejected")
        if any(int(branch_counts.get(name, 0)) <= 0 for name in required):
            raise ProtocolInvalid(
                f"mixture proposal did not exercise every branch: {branch_counts}"
            )
    return {
        "attempt_count": int(total_attempts),
        "active_fraction": total_active / total_attempts,
        "importance_ess_per_attempt": total_ess / total_attempts,
        "maximum_importance": maximum_importance,
        "proposal_branch_counts": {str(key): int(value) for key, value in branch_counts.items()},
        "proposal_invariants": dict(proposal_invariants),
    }


def _execute_seed_arm(seed: int, arm: str, marker_sha256: str) -> dict[str, Any]:
    from rtgs.optim.compact_trainer import CompactTrainer

    started = time.perf_counter()
    authorization = _authorize_official(marker_sha256)
    inputs, initialization, fixture = _construct_official_fixture(
        seed,
        authorization=authorization,
    )
    before = [field_hashes(field) for field in inputs.observations]
    init_hashes = gaussians_hashes(initialization)
    backends = [GaussianObservationIndex(field, tile_size=8) for field in inputs.observations]
    renderer = TorchPointRasterizer(
        point_chunk=32,
        gaussian_chunk=64,
        sh_color_activation="hard",
        sh_smu1_mu=2.0 / 255.0,
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
    )
    config = _official_config(arm, seed, authorization=authorization)
    trainer = CompactTrainer(config, point_rasterizer=renderer)
    final, history = trainer.train(
        inputs,
        initialization.detach(),
        query_backends=backends,
        bundle_path=None,
    )
    after = [field_hashes(field) for field in inputs.observations]
    if before != after:
        raise ProtocolInvalid("official training mutated a teacher")
    if final.n != OFFICIAL_N_3D or final.sh_degree != 0:
        raise ProtocolInvalid("official final topology/SH degree differs")
    clocks = _optimizer_clock(history)
    steps = _validate_training_steps(history, arm)
    checkpoints = _checkpoint_records(history)
    schedule = [int(value) for value in history["view_schedule"]]
    if len(schedule) != ITERATIONS or history.get("view_schedule_sha256") != canonical_hash(
        schedule
    ):
        raise ProtocolInvalid("view schedule or its digest differs")
    # Re-evaluate the final checkpoint through the harness-local streaming reducer.
    independent = evaluate_risks(
        inputs,
        final,
        point_rasterizer=renderer,
        query_backends=backends,
        evaluation_chunk=256,
        query_component_chunk=64,
    )
    for risk in ("pixel", "area"):
        if not math.isclose(
            checkpoints[-1]["risks"][risk],
            independent["equal_view_risk"][risk],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ProtocolInvalid(f"trainer and harness final {risk} evaluation differ")
    record = {
        "status": "PASS",
        "seed": seed,
        "arm": arm,
        "risk_measure": history["risk_measure"],
        "configuration": asdict(config),
        "fixture": fixture,
        "initialization_hash": init_hashes["aggregate"],
        "final_hash": gaussians_hashes(final)["aggregate"],
        "n_initial": initialization.n,
        "n_final": final.n,
        "teacher_hashes_before": before,
        "teacher_hashes_after": after,
        "view_schedule": schedule,
        "view_schedule_sha256": history["view_schedule_sha256"],
        "checkpoints": checkpoints,
        "optimizer_group_clocks": clocks,
        "optimizer_steps": ITERATIONS,
        "parameter_motion": _motion_values(history),
        **steps,
        "rgb_access_count": 0,
        "independent_final_evaluation": independent,
        "trainer_history_sha256": canonical_hash(history),
        "trainer_history": history,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    assert_finite_tree(record, f"seed {seed}/{arm}")
    return record


def _worker_entry(connection: Any, seed: int, arm: str, marker_sha256: str) -> None:
    try:
        _configure_official_runtime()
        environment = _environment_metadata()
        _assert_official_environment(environment)
        seal = load_and_verify_seal()
        marker = _load_attempt(marker_sha256)
        if marker.get("seal_file_sha256") != seal["seal_file_sha256"]:
            raise ProtocolInvalid("worker marker/seal binding differs")
        if _environment_fingerprint(environment) != marker["environment_fingerprint"]:
            raise ProtocolInvalid("worker environment differs from attempt marker")
        record = _execute_seed_arm(seed, arm, marker_sha256)
        verify_source_hashes(seal["source_hashes"], seal["source_aggregate"])
        if sha256_file(ATTEMPT) != marker_sha256:
            raise ProtocolInvalid("attempt marker changed during worker execution")
        connection.send({"ok": True, "record": record})
    except BaseException as error:  # worker must return finite failure evidence to the parent
        connection.send(
            {
                "ok": False,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "traceback": traceback.format_exc(),
                "seed": seed,
                "arm": arm,
            }
        )
    finally:
        connection.close()


def _spawn_seed_arm(
    seed: int, arm: str, marker_sha256: str, timeout: float = 900.0
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(send, seed, arm, marker_sha256),
        name=f"compact-{seed}-{arm}",
    )
    process.start()
    send.close()
    deadline = time.monotonic() + timeout
    payload = None
    while time.monotonic() < deadline:
        if receive.poll(1.0):
            payload = receive.recv()
            break
        if not process.is_alive():
            break
    if payload is None and process.is_alive():
        process.terminate()
        process.join(timeout=10.0)
        raise TimeoutError(f"fresh worker timed out for seed {seed}/{arm}")
    process.join(timeout=30.0)
    receive.close()
    if process.exitcode != 0:
        raise ProtocolInvalid(f"fresh worker exit code {process.exitcode} for seed {seed}/{arm}")
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise ProtocolInvalid(f"fresh worker failed for seed {seed}/{arm}: {payload}")
    return dict(payload["record"])


def _failure_payload(
    *,
    artifact_type: str,
    error: BaseException,
    seal: Mapping[str, Any],
    marker_sha256: str,
    records: Sequence[Mapping[str, Any]],
    failure_phase: str,
    raw_sha256: str | None = None,
) -> dict[str, Any]:
    if failure_phase not in {"pre_raw_commit", "post_raw_commit"}:
        raise ProtocolInvalid(f"unknown RAW failure phase {failure_phase!r}")
    payload = {
        "artifact_type": artifact_type,
        "status": "FAIL",
        "failure_phase": failure_phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "traceback": traceback.format_exc(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_file_sha256": seal["seal_file_sha256"],
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "source_aggregate": seal["source_aggregate"],
        "attempt_sha256": marker_sha256,
        "raw_sha256": raw_sha256,
        "completed_records": list(records),
    }
    if failure_phase == "pre_raw_commit":
        payload["decision"] = "MECHANISM_FAIL"
    return payload


def _write_failure_artifacts(
    *,
    error: BaseException,
    seal: Mapping[str, Any],
    marker_sha256: str,
    records: Sequence[Mapping[str, Any]],
    raw_commit_sha256: str | None,
) -> tuple[str, str]:
    """Commit the frozen pre- or post-PASS-RAW failure transition."""
    failure_phase = "post_raw_commit" if raw_commit_sha256 is not None else "pre_raw_commit"
    if failure_phase == "pre_raw_commit":
        raw_commit_sha256 = _exclusive_json(
            RAW,
            _failure_payload(
                artifact_type=RAW_ARTIFACT_TYPE,
                error=error,
                seal=seal,
                marker_sha256=marker_sha256,
                records=records,
                failure_phase=failure_phase,
            ),
        )
    if raw_commit_sha256 is None:  # pragma: no cover - narrowed by the branch above
        raise ProtocolInvalid("failure transition lacks a committed RAW digest")
    _exclusive_json(
        RESULT,
        _failure_payload(
            artifact_type=RESULT_ARTIFACT_TYPE,
            error=error,
            seal=seal,
            marker_sha256=marker_sha256,
            records=records,
            failure_phase=failure_phase,
            raw_sha256=raw_commit_sha256,
        ),
    )
    return failure_phase, raw_commit_sha256


def run_once() -> int:
    _configure_official_runtime()
    environment = _environment_metadata()
    _assert_official_environment(environment)
    seal = load_and_verify_seal()
    _, marker_sha256 = _claim_attempt(seal, environment)
    records: list[dict[str, Any]] = []
    raw_commit_sha256: str | None = None
    try:
        for seed in OFFICIAL_SEEDS:
            for arm in ARMS:
                records.append(_spawn_seed_arm(seed, arm, marker_sha256))
        verify_source_hashes(seal["source_hashes"], seal["source_aggregate"])
        if sha256_file(ATTEMPT) != marker_sha256:
            raise ProtocolInvalid("attempt marker changed before RAW serialization")
        raw_payload = {
            "artifact_type": RAW_ARTIFACT_TYPE,
            "status": "PASS",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "seal_file_sha256": seal["seal_file_sha256"],
            "seal_payload_sha256": seal["seal_payload_sha256"],
            "attempt_sha256": marker_sha256,
            "source_aggregate": seal["source_aggregate"],
            "environment_fingerprint": _environment_fingerprint(environment),
            "records": records,
        }
        raw_sha256 = _exclusive_json(RAW, raw_payload)
        raw_commit_sha256 = raw_sha256
        strict_raw = strict_json_load_bound(RAW, raw_sha256)
        result_payload = result_from_raw(
            strict_raw,
            raw_sha256=raw_sha256,
            marker_sha256=marker_sha256,
            seal=seal,
        )
        result_sha256 = _exclusive_json(RESULT, result_payload)
    except BaseException as error:
        _write_failure_artifacts(
            error=error,
            seal=seal,
            marker_sha256=marker_sha256,
            records=records,
            raw_commit_sha256=raw_commit_sha256,
        )
        raise
    # A successfully returned RESULT write is terminal.  Keep display-only work outside the
    # protocol handler so an I/O failure cannot contradict or replace committed PASS evidence.
    print(
        f"saved {RESULT.relative_to(ROOT)} (sha256={result_sha256}, "
        f"decision={result_payload['decision']})",
        flush=True,
    )
    return 0


def validate_audit_authorization(
    text: str,
    *,
    bindings: Mapping[str, str],
) -> str:
    """Require a scientist verdict, literal authorization, and every current binding."""
    match = re.search(
        r"(?:verdict\s*:|#{1,6}\s*verdict\s*\n)\s*[*_`]*(PASS|QUALIFIED)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ProtocolInvalid("scientist audit verdict does not authorize calibrated integration")
    if "CALIBRATED_INTEGRATION_AUTHORIZED: YES" not in text:
        raise ProtocolInvalid("scientist audit lacks literal calibrated authorization")
    missing = [name for name, digest in bindings.items() if digest not in text]
    if missing:
        raise ProtocolInvalid(f"scientist audit is missing current bindings: {missing}")
    return match.group(1).upper()


def _audit_authorization(seal: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    if strict_json_load(RESULT).get("status") != "PASS":
        raise ProtocolInvalid("calibrated integration requires a protocol-valid official PASS")
    bindings = {
        "preregistration": PREREGISTRATION_SHA256,
        "seal": seal["seal_file_sha256"],
        "marker": sha256_file(ATTEMPT),
        "raw": sha256_file(RAW),
        "result": sha256_file(RESULT),
        "source": seal["source_aggregate"],
    }
    verdict = validate_audit_authorization(
        AUDIT.read_text(encoding="utf-8"),
        bindings=bindings,
    )
    return verdict, bindings


def _capture_image_paths() -> dict[str, Path]:
    rgb = CALIBRATED_SCENE / "rgb"
    if not rgb.is_dir():
        raise FileNotFoundError(f"missing calibrated RGB directory: {rgb}")
    result: dict[str, Path] = {}
    for path in rgb.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = re.fullmatch(r"[Cc](\d+)|rgb_(\d+)", path.stem, flags=re.IGNORECASE)
        if match is None:
            continue
        digits = match.group(1) or match.group(2)
        camera_id = f"C{int(digits):04d}"
        if camera_id in result:
            raise ProtocolInvalid(f"multiple calibrated RGB candidates for {camera_id}")
        result[camera_id] = path
    required = (*CALIBRATED_VIEWS, CALIBRATED_HELDOUT_VIEW)
    missing = [name for name in required if name not in result]
    if missing:
        raise FileNotFoundError(f"calibrated capture is missing selected views: {missing}")
    return {name: result[name] for name in required}


def _viewer_rgb_directory_binding() -> dict[str, Any]:
    """Bind the full directory ordering that the unchanged viewer CLI will sample."""
    rgb = CALIBRATED_SCENE / "rgb"
    mask_directory = CALIBRATED_SCENE / "mask"
    calibration_ids = set(_calibration_records())
    calibration_candidates = [
        directory / "calibration_dome.json"
        for directory in (CALIBRATED_SCENE, *CALIBRATED_SCENE.parents[:3])
    ]
    selected_calibration = next(
        (candidate for candidate in calibration_candidates if candidate.is_file()),
        None,
    )
    if selected_calibration is None or selected_calibration.resolve() != CALIBRATION.resolve():
        raise ProtocolInvalid("viewer calibration discovery does not resolve the frozen file")
    entries = []
    accepted: list[tuple[str, Path]] = []
    for path in sorted(rgb.iterdir(), key=lambda candidate: candidate.name):
        kind = (
            "symlink"
            if path.is_symlink()
            else ("file" if path.is_file() else ("directory" if path.is_dir() else "other"))
        )
        entries.append({"name": path.name, "kind": kind})
        match = re.fullmatch(r"[Cc](\d+)|rgb_(\d+)", path.stem, flags=re.IGNORECASE)
        if (
            kind == "file"
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and not path.stem.lower().startswith("mask_")
            and match is not None
        ):
            digits = match.group(1) or match.group(2)
            camera_id = f"C{int(digits):04d}"
            if camera_id in calibration_ids:
                accepted.append((camera_id, path))
    accepted.sort(key=lambda item: item[0])
    if not accepted:
        raise ProtocolInvalid("viewer RGB loader would find no calibrated images")
    maximum = 8
    if len(accepted) <= maximum:
        selected = accepted
    else:
        indices = torch.linspace(0, len(accepted) - 1, maximum).round().long().unique().tolist()
        selected = [accepted[index] for index in indices]
    mask_entries = (
        [
            {
                "name": path.name,
                "kind": (
                    "symlink"
                    if path.is_symlink()
                    else ("file" if path.is_file() else ("directory" if path.is_dir() else "other"))
                ),
            }
            for path in sorted(mask_directory.iterdir(), key=lambda candidate: candidate.name)
        ]
        if mask_directory.is_dir()
        else None
    )

    def mask_binding(camera_id: str) -> dict[str, str] | None:
        candidates = (
            mask_directory / f"mask_{camera_id}.png",
            mask_directory / f"mask_{camera_id}.jpg",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            return None
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}

    return {
        "directory": rgb.relative_to(ROOT).as_posix(),
        "calibration_search": [
            {
                "path": candidate.relative_to(ROOT).as_posix(),
                "kind": (
                    "symlink"
                    if candidate.is_symlink()
                    else (
                        "file"
                        if candidate.is_file()
                        else ("directory" if candidate.is_dir() else "absent")
                    )
                ),
                "sha256": sha256_file(candidate) if candidate.is_file() else None,
            }
            for candidate in calibration_candidates
        ],
        "entries": entries,
        "mask_directory": mask_directory.relative_to(ROOT).as_posix(),
        "mask_entries": mask_entries,
        "accepted_camera_order": [camera_id for camera_id, _ in accepted],
        "selected": [
            {
                "camera_id": camera_id,
                "view_name": path.stem,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "expected_native_dimensions": [5328, 4608],
                "mask": mask_binding(camera_id),
            }
            for camera_id, path in selected
        ],
    }


def _heldout_mask_path() -> Path:
    """Freeze the dataset loader's PNG-first precedence to one exact held-out mask."""
    path = CALIBRATED_SCENE / "mask" / f"mask_{CALIBRATED_HELDOUT_VIEW}.png"
    if not path.is_file():
        raise FileNotFoundError(f"frozen PNG held-out mask is unavailable: {path}")
    return path


def _external_structsplat_binding() -> dict[str, Any]:
    """Read-only source/version binding; this harness never edits the external repository."""
    try:
        import importlib.metadata

        import structsplat
    except ImportError as error:
        raise RuntimeError("the preregistered StructSplat provider is unavailable") from error
    source_root = Path(structsplat.__file__).resolve().parent
    repository = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    repository_root = (
        Path(repository.stdout.strip()).resolve() if repository.returncode == 0 else source_root
    )
    provider_suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    repository_suffixes = {*provider_suffixes, ".json", ".toml", ".yaml", ".yml", ".cfg"}
    build_names = {"CMakeLists.txt", "Makefile", "setup.py", "setup.cfg"}
    provider_digest = hashlib.sha256()
    provider_files = []
    for path in sorted(
        candidate
        for candidate in source_root.rglob("*")
        if candidate.is_file() and candidate.suffix in provider_suffixes
    ):
        relative = path.relative_to(source_root).as_posix()
        content = path.read_bytes()
        provider_digest.update(relative.encode("utf-8"))
        provider_digest.update(b"\0")
        provider_digest.update(content)
        provider_digest.update(b"\0")
        provider_files.append({"path": relative, "sha256": sha256_bytes(content)})

    relevant_paths = {
        repository_root / "pyproject.toml",
        *(
            candidate
            for candidate in source_root.rglob("*")
            if candidate.is_file()
            and (candidate.suffix in repository_suffixes or candidate.name in build_names)
        ),
    }
    repository_digest = hashlib.sha256()
    repository_files = []
    for path in sorted(path for path in relevant_paths if path.is_file()):
        relative = path.relative_to(repository_root).as_posix()
        content = path.read_bytes()
        repository_digest.update(relative.encode("utf-8"))
        repository_digest.update(b"\0")
        repository_digest.update(content)
        repository_digest.update(b"\0")
        repository_files.append({"path": relative, "sha256": sha256_bytes(content)})
    try:
        version = importlib.metadata.version("structsplat")
    except importlib.metadata.PackageNotFoundError:
        version = None
    git = None
    if repository.returncode == 0:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                "pyproject.toml",
                "src/structsplat",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "diff",
                "--binary",
                "HEAD",
                "--",
                "pyproject.toml",
                "src/structsplat",
            ],
            check=True,
            capture_output=True,
        ).stdout
        git = {
            "revision": revision,
            "status_sha256": sha256_bytes(status.encode("utf-8")),
            "status_lines": status.splitlines(),
            "tracked_diff_sha256": sha256_bytes(diff),
        }
    extension_candidates = {
        *(path for path in source_root.rglob("*.so") if path.is_file()),
        *(path for path in source_root.rglob("*.pyd") if path.is_file()),
    }
    torch_extensions = Path.home() / ".cache/torch_extensions"
    if torch_extensions.is_dir():
        extension_candidates.update(
            path
            for path in torch_extensions.glob("*/structsplat_render_ext/*")
            if path.is_file() and path.suffix in {".so", ".pyd"}
        )
    extension_binaries = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in sorted(extension_candidates)
    ]
    return {
        "provider": "structsplat",
        "version": version,
        "module_root": str(source_root),
        "repository_root": str(repository_root),
        "provider_source_digest": provider_digest.hexdigest(),
        "provider_source_file_count": len(provider_files),
        "provider_source_manifest_sha256": canonical_hash(provider_files),
        "repository_binding_digest": repository_digest.hexdigest(),
        "repository_source_file_count": len(repository_files),
        "repository_source_manifest_sha256": canonical_hash(repository_files),
        "git": git,
        "extension_binaries": extension_binaries,
    }


def _gsplat_binding() -> dict[str, Any]:
    """Bind the installed gsplat package rather than any stale editable alias."""
    try:
        import importlib.metadata

        import gsplat
    except ImportError as error:
        raise RuntimeError("the calibrated viewer requires gsplat") from error
    source_root = Path(gsplat.__file__).resolve().parent
    suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".so", ".pyd"}
    files = []
    aggregate = hashlib.sha256()
    for path in sorted(candidate for candidate in source_root.rglob("*") if candidate.is_file()):
        if path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(source_root).as_posix()
        content = path.read_bytes()
        digest = sha256_bytes(content)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(content)
        aggregate.update(b"\0")
        files.append({"path": relative, "sha256": digest})
    return {
        "version": importlib.metadata.version("gsplat"),
        "module_root": str(source_root),
        "module_file": str(Path(gsplat.__file__).resolve()),
        "source_digest": aggregate.hexdigest(),
        "source_file_count": len(files),
        "source_manifest_sha256": canonical_hash(files),
    }


def _calibrated_configuration() -> dict[str, Any]:
    abi_preload = _abi_preload_file_binding(CALIBRATED_LD_PRELOAD)
    return {
        "scene": str(CALIBRATED_SCENE.relative_to(ROOT)),
        "calibration": str(CALIBRATION.relative_to(ROOT)),
        "selected_order": [*CALIBRATED_VIEWS, CALIBRATED_HELDOUT_VIEW],
        "training_views": list(CALIBRATED_VIEWS),
        "heldout_view": CALIBRATED_HELDOUT_VIEW,
        "test_every": 8,
        "resolution": [5328, 4608],
        "downscale": 1,
        "undistort": True,
        "load_masks_for_acquisition": False,
        "fit": {
            "n_gaussians": 640,
            "max_gaussians": 640,
            "iterations": 100,
            "backend": "structsplat",
            "adaptive_density": False,
            "growth_waves": 1,
            "relocate_fraction": 0.0,
            "structsplat_renderer": "cuda_tiled",
            "lr": 1e-2,
            "grad_init_mix": 0.7,
            "row_chunk": 64,
            "log_every": 20,
            "convergence_patience": 0,
            "convergence_tol": 0.05,
            "convergence_check_every": 25,
            "appearance_parameterization": "weight_color_9p",
            "freeze_geometry": False,
            "view_seeds": list(range(7)),
        },
        "compact_carve": {
            "n_init_3d": 835,
            "candidate_multiplier": 4,
            "samples_per_ray": 48,
            "query_batch_size": 4096,
            "query_component_chunk": 256,
            "max_query_pairs": 1_048_576,
            "tile_size": 16,
            "max_index_entries_per_view": MAX_TILE_ENTRIES_PER_VIEW,
            "max_candidates_per_tile": MAX_CANDIDATES_PER_TILE,
            "seed": 75200,
            "bounds_scale": 0.5,
            "near": 0.05,
            "min_views": 2,
            "hull_fraction": 0.85,
            "coverage_scale": 1.0,
            "coverage_threshold": 0.40,
            "color_std_sigma": 0.20,
            "min_score": 0.05,
            "peak_radius_steps": 3.0,
            "init_opacity": 0.1,
            "sh_degree": 0,
            "max_anchor_rounds": 8,
        },
        "training": {
            "proposal_mode": "pixel_gaussian",
            "seed": 75201,
            "iterations": 40,
            "attempts_per_step": 128,
            "uniform_fraction": 0.25,
            "checkpoints": [0, 10, 20, 40],
            "extent": "initial_mean_cloud",
            "teacher_tile_size": 16,
            "outer_microbatch": 32,
            "point_chunk": 32,
            "gaussian_chunk": 64,
            "query_component_chunk": 64,
        },
        "sample_evaluation": {
            "seed": 75202,
            "points_per_training_view": 4096,
            "with_replacement": True,
        },
        "heldout_evaluation": {
            "seed": 75203,
            "points": 4096,
            "with_replacement": True,
            "metrics": ["all_rgb_mse_psnr", "foreground_rgb_mse_psnr"],
        },
        "viewer_command": list(CALIBRATED_VIEWER_COMMAND),
        "ld_preload": abi_preload["requested_path"],
        "ld_preload_binding": abi_preload,
    }


def _calibrated_outputs() -> dict[str, str]:
    paths = {
        "bundle": CALIBRATED_BUNDLE,
        "teacher_acquisition": CALIBRATED_ACQUISITION,
        "compact_training_raw": CALIBRATED_TRAINING_RAW,
        "gaussians_init": CALIBRATED_INIT,
        "gaussians_final": CALIBRATED_FINAL,
        "heldout_evaluation": CALIBRATED_HELDOUT,
        "viewer_smoke": CALIBRATED_VIEWER,
        "viewer_snapshots": CALIBRATED_SNAPSHOTS,
        "calibrated_result": CALIBRATED_RESULT,
    }
    return {name: str(path.relative_to(ROOT)) for name, path in paths.items()}


def _calibrated_collisions() -> list[Path]:
    if not os.path.lexists(os.fspath(RUN_DIR)):
        return []
    if not RUN_DIR.is_dir():
        return [RUN_DIR]
    named = {
        CALIBRATED_PLAN,
        CALIBRATED_ATTEMPT,
        CALIBRATED_BUNDLE,
        CALIBRATED_ACQUISITION,
        CALIBRATED_TRAINING_RAW,
        CALIBRATED_INIT,
        CALIBRATED_FINAL,
        CALIBRATED_HELDOUT,
        CALIBRATED_VIEWER,
        CALIBRATED_SNAPSHOTS,
        CALIBRATED_RESULT,
    }
    return sorted(path for path in named if os.path.lexists(os.fspath(path)))


def _calibrated_plan(
    *,
    seal: Mapping[str, Any],
    audit_verdict: str,
    audit_bindings: Mapping[str, str],
) -> dict[str, Any]:
    images = _capture_image_paths()
    heldout_mask = _heldout_mask_path()
    external = _external_structsplat_binding()
    gsplat = _gsplat_binding()
    viewer_rgb = _viewer_rgb_directory_binding()
    if [record["camera_id"] for record in viewer_rgb["selected"]] != [
        *CALIBRATED_VIEWS,
        CALIBRATED_HELDOUT_VIEW,
    ]:
        raise ProtocolInvalid("unchanged viewer command selects a different frozen camera order")
    records = _calibration_records()
    for name in (*CALIBRATED_VIEWS, CALIBRATED_HELDOUT_VIEW):
        resolution = records[name]["intrinsics"]["resolution"]
        if [int(value) for value in resolution] != [5328, 4608]:
            raise ProtocolInvalid(
                f"camera {name} calibration resolution differs from full resolution"
            )
    return {
        "artifact_type": "compact_point_training_calibrated_plan_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_bearing": False,
        "audit_verdict": audit_verdict,
        "audit_sha256": sha256_file(AUDIT),
        "audit_bindings": dict(audit_bindings),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_file_sha256": seal["seal_file_sha256"],
        "source_aggregate": seal["source_aggregate"],
        "external_structsplat": external,
        "gsplat": gsplat,
        "input_hashes": {
            "calibration": sha256_file(CALIBRATION),
            "images": {
                name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
                for name, path in images.items()
            },
            "heldout_mask": {
                "path": heldout_mask.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(heldout_mask),
            },
            "viewer_rgb_directory": viewer_rgb,
        },
        "configuration": _calibrated_configuration(),
        "outputs": _calibrated_outputs(),
    }


def _bounded_calibrated_failure(
    error: BaseException,
    *,
    plan_sha256: str,
    attempt_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact_type": "compact_point_training_calibrated_result_v1",
        "status": "FAIL",
        "decision_bearing": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "traceback": traceback.format_exc(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "plan_sha256": plan_sha256,
        "attempt_sha256": attempt_sha256,
        "no_fallback_used": True,
        "forbidden_legacy_inputs_used": False,
    }


def _camera_record(camera: Camera) -> dict[str, Any]:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "R": camera.R.reshape(-1).tolist(),
        "t": camera.t.tolist(),
    }


def _camera_from_record(record: Mapping[str, Any]) -> Camera:
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


def _calibration_records() -> dict[str, Mapping[str, Any]]:
    calibration = strict_json_load(CALIBRATION)
    records: dict[str, Mapping[str, Any]] = {}
    for record in calibration["cameras"]:
        camera_id = str(record["camera_id"]).upper()
        if camera_id in records:
            raise ProtocolInvalid(f"duplicate calibration camera id: {camera_id}")
        records[camera_id] = record
    return records


def _camera_from_calibration(camera_id: str) -> tuple[Camera, list[float]]:
    record = _calibration_records()[camera_id]
    intrinsics = record["intrinsics"]
    width, height = (int(value) for value in intrinsics["resolution"])
    matrix = intrinsics["camera_matrix"]
    view = torch.tensor(record["extrinsics"]["view_matrix"], dtype=torch.float32).reshape(4, 4)
    camera = Camera(
        fx=float(matrix[0]),
        fy=float(matrix[4]),
        cx=float(matrix[2]) + 0.5,
        cy=float(matrix[5]) + 0.5,
        width=width,
        height=height,
        R=view[:3, :3],
        t=view[:3, 3],
    )
    return camera, [float(value) for value in intrinsics.get("distortion_coefficients", [])]


def _calibrated_bundle_limits() -> Any:
    from rtgs.data.reconstruction_inputs import BundleLoadLimits

    return BundleLoadLimits(
        max_manifest_bytes=MAX_MANIFEST_BYTES,
        max_teacher_archives=MAX_TEACHER_ARCHIVES,
        max_archive_compressed_bytes=MAX_COMPRESSED_ARCHIVE_BYTES,
        max_total_compressed_bytes=MAX_TOTAL_COMPRESSED_BYTES,
        max_zip_members=MAX_ZIP_MEMBERS,
        max_member_uncompressed_bytes=MAX_UNCOMPRESSED_MEMBER_BYTES,
        max_archive_uncompressed_bytes=MAX_UNCOMPRESSED_ARCHIVE_BYTES,
    )


def _load_calibrated_plan_and_seal(plan_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate exact plan bytes and all sealed repository sources for each phase."""
    plan = strict_json_load_bound(CALIBRATED_PLAN, plan_sha256)
    seal = load_and_verify_seal()
    expected = {
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_file_sha256": seal["seal_file_sha256"],
        "source_aggregate": seal["source_aggregate"],
    }
    if any(plan.get(name) != value for name, value in expected.items()):
        raise ProtocolInvalid("calibrated plan binds different preregistration/seal/source state")
    return plan, seal


def _worker_failure(path: Path, error: BaseException, **context: Any) -> None:
    payload = {
        "status": "FAIL",
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "traceback": traceback.format_exc(),
        **context,
    }
    if not path.exists():
        _exclusive_json(path, payload)


def _acquisition_worker(
    plan_sha256: str,
    camera_id: str,
    seed: int,
    teacher_path_text: str,
    result_path_text: str,
) -> None:
    teacher_path = Path(teacher_path_text)
    result_path = Path(result_path_text)
    try:
        plan, _ = _load_calibrated_plan_and_seal(plan_sha256)
        external = _external_structsplat_binding()
        if external != plan["external_structsplat"]:
            raise ProtocolInvalid("external StructSplat source/version drifted before acquisition")
        image_path = _capture_image_paths()[camera_id]
        image_binding = plan["input_hashes"]["images"][camera_id]
        if (
            image_path.relative_to(ROOT).as_posix() != image_binding["path"]
            or sha256_file(image_path) != image_binding["sha256"]
        ):
            raise ProtocolInvalid(f"input image changed before acquisition: {camera_id}")
        if sha256_file(CALIBRATION) != plan["input_hashes"]["calibration"]:
            raise ProtocolInvalid("calibration changed before acquisition")
        if not torch.cuda.is_available():
            raise RuntimeError("full-resolution StructSplat acquisition requires CUDA")

        from PIL import Image as PILImage

        from rtgs.data.calibrated import _resize_image, _undistort
        from rtgs.image2gs.fit import FitConfig, fit_image

        camera, distortion = _camera_from_calibration(camera_id)
        with PILImage.open(image_path) as source:
            native_size = tuple(int(value) for value in source.size)
        if native_size != (5328, 4608):
            raise ProtocolInvalid(
                f"source {camera_id} native dimensions differ from (5328,4608): {native_size}"
            )
        image = _resize_image(image_path, camera.width, camera.height)
        image = _undistort(
            image,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            distortion,
        ).to("cuda")
        observations: list[GaussianObservationField] = []
        config = FitConfig(
            n_gaussians=640,
            max_gaussians=640,
            iterations=100,
            backend="structsplat",
            adaptive_density=False,
            growth_waves=1,
            relocate_fraction=0.0,
            structsplat_renderer="cuda_tiled",
            lr=1e-2,
            grad_init_mix=0.7,
            row_chunk=64,
            log_every=20,
            convergence_patience=0,
            convergence_tol=0.05,
            convergence_check_every=25,
            appearance_parameterization="weight_color_9p",
            freeze_geometry=False,
        )
        _, history = fit_image(
            image,
            config,
            seed=seed,
            mask=None,
            observation_callback=observations.append,
            observation_view_id=camera_id,
        )
        if len(observations) != 1:
            raise ProtocolInvalid("StructSplat did not export exactly one lossless teacher")
        teacher = observations[0].to("cpu")
        if (
            teacher.provider != "structsplat"
            or teacher.producer_source_digest != external["provider_source_digest"]
            or teacher.fit_window != (0, 0, 5328, 4608)
            or teacher.n_init != 640
            or teacher.n != 640
        ):
            raise ProtocolInvalid("acquired teacher semantics/provenance differ from frozen plan")
        external_after = _external_structsplat_binding()
        if external_after != plan["external_structsplat"]:
            raise ProtocolInvalid("StructSplat source/binary binding changed during acquisition")
        import structsplat.cuda_render as cuda_render

        extension = getattr(cuda_render, "_EXT", None)
        extension_path = None if extension is None else Path(extension.__file__).resolve()
        extension_record = (
            None
            if extension_path is None
            else {"path": str(extension_path), "sha256": sha256_file(extension_path)}
        )
        if (
            extension_record is not None
            and extension_record not in external_after["extension_binaries"]
        ):
            raise ProtocolInvalid("loaded StructSplat CUDA extension is outside the frozen binding")
        _load_calibrated_plan_and_seal(plan_sha256)
        teacher.save_npz(teacher_path)
        payload = {
            "status": "PASS",
            "camera_id": camera_id,
            "seed": seed,
            "camera": _camera_record(camera),
            "teacher_sha256": sha256_file(teacher_path),
            "teacher_semantic_hashes": field_hashes(teacher),
            "native_source_dimensions": list(native_size),
            "history": history,
            "external_structsplat": external_after,
            "loaded_cuda_extension": extension_record,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        _exclusive_json(result_path, payload)
    except BaseException as error:
        _worker_failure(result_path, error, camera_id=camera_id, seed=seed)
        raise


def _spawn_file_worker(
    target: Any,
    args: tuple[Any, ...],
    result_path: Path,
    *,
    name: str,
    timeout: float,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=target, args=args, name=name)
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30.0)
        raise TimeoutError(f"calibrated worker timed out: {name}")
    if not result_path.is_file():
        raise ProtocolInvalid(f"calibrated worker produced no result: {name}")
    result_sha256 = sha256_file(result_path)
    payload = strict_json_load_bound(result_path, result_sha256)
    if process.exitcode != 0 or payload.get("status") != "PASS":
        raise ProtocolInvalid(f"calibrated worker failed: {name}: {payload}")
    return payload


def _acquire_calibrated_teachers(
    plan: Mapping[str, Any],
    plan_sha256: str,
    scratch: Path,
) -> tuple[ReconstructionInputs, list[dict[str, Any]]]:
    fields = []
    cameras = []
    records = []
    for seed, camera_id in enumerate(CALIBRATED_VIEWS):
        teacher_path = scratch / f"{seed:04d}.teacher.npz"
        result_path = scratch / f"{seed:04d}.acquisition.json"
        record = _spawn_file_worker(
            _acquisition_worker,
            (plan_sha256, camera_id, seed, str(teacher_path), str(result_path)),
            result_path,
            name=f"compact-acquire-{camera_id}",
            timeout=7200.0,
        )
        if record["external_structsplat"] != plan["external_structsplat"]:
            raise ProtocolInvalid("StructSplat source changed across acquisition workers")
        if sha256_file(teacher_path) != record["teacher_sha256"]:
            raise ProtocolInvalid("acquisition teacher archive changed after worker exit")
        field = GaussianObservationField.load_npz(teacher_path, verify=True)
        if field_hashes(field) != record["teacher_semantic_hashes"]:
            raise ProtocolInvalid("acquisition teacher semantics changed after reload")
        fields.append(field)
        cameras.append(_camera_from_record(record["camera"]))
        records.append(record)
    if _external_structsplat_binding() != plan["external_structsplat"]:
        raise ProtocolInvalid("StructSplat source/binary binding changed across acquisition views")
    extension_records = [record["loaded_cuda_extension"] for record in records]
    if (
        any(record is None for record in extension_records)
        or len({canonical_hash(record) for record in extension_records}) != 1
    ):
        raise ProtocolInvalid("acquisition workers loaded different StructSplat CUDA binaries")
    inputs = ReconstructionInputs(
        observations=fields,
        cameras=cameras,
        view_names=list(CALIBRATED_VIEWS),
        points=None,
        point_visibility=None,
        bounds_hint=None,
        name="compact_point_training_20260716",
    )
    inputs.save(CALIBRATED_BUNDLE)
    reloaded = ReconstructionInputs.load(
        CALIBRATED_BUNDLE,
        device="cpu",
        strict=True,
        limits=_calibrated_bundle_limits(),
    )
    if [field_hashes(field) for field in reloaded.observations] != [
        field_hashes(field) for field in inputs.observations
    ]:
        raise ProtocolInvalid("strict calibrated bundle reload changed teacher semantics")
    return reloaded, records


def _deny_rgb_access() -> tuple[dict[str, int], Any]:
    """Patch every prohibited Stage-2/3 RGB surface in the fresh training child."""
    counters = {
        "dataset_open": 0,
        "dataset_os_open": 0,
        "pil_open": 0,
        "calibrated_loader": 0,
        "scene_data": 0,
        "negative_control_dataset_denials": 0,
        "negative_control_os_open_denials": 0,
        "negative_control_pil_denials": 0,
        "negative_control_calibrated_loader_denials": 0,
        "negative_control_scene_data_denials": 0,
    }
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    dataset_root = (ROOT / "dataset").resolve()

    def is_dataset_path(file: Any) -> bool:
        if not isinstance(file, (str, os.PathLike)):
            return False
        candidate = Path(file).expanduser()
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        return resolved == dataset_root or dataset_root in resolved.parents

    def make_guard(original: Any) -> Any:
        def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if is_dataset_path(file):
                counters["dataset_open"] += 1
                raise ProtocolInvalid("dataset/source-image open denied after Stage-1 boundary")
            return original(file, *args, **kwargs)

        return guarded_open

    def guarded_os_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if is_dataset_path(file):
            counters["dataset_os_open"] += 1
            raise ProtocolInvalid("dataset os.open denied after Stage-1 boundary")
        return original_os_open(file, *args, **kwargs)

    builtins.open = make_guard(original_builtin_open)
    io.open = make_guard(original_io_open)
    os.open = guarded_os_open
    from PIL import Image as PILImage

    import rtgs.data as data_package
    import rtgs.data.calibrated as calibrated_module
    import rtgs.data.scene as scene_module

    def deny_pil(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        counters["pil_open"] += 1
        raise ProtocolInvalid("PIL decode denied after Stage-1 boundary")

    def deny_calibrated(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        counters["calibrated_loader"] += 1
        raise ProtocolInvalid("calibrated RGB loader denied after Stage-1 boundary")

    class DeniedSceneData:
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs
            counters["scene_data"] += 1
            raise ProtocolInvalid("SceneData denied after Stage-1 boundary")

    PILImage.open = deny_pil
    calibrated_module.load_calibrated_scene = deny_calibrated
    calibrated_module._resize_image = deny_calibrated
    calibrated_module.SceneData = DeniedSceneData
    scene_module.SceneData = DeniedSceneData
    data_package.load_calibrated_scene = deny_calibrated
    data_package.SceneData = DeniedSceneData

    def expect_denial(callback: Any, counter: str, label: str) -> None:
        before = counters[counter]
        try:
            callback()
        except ProtocolInvalid:
            if counters[counter] != before + 1:
                raise ProtocolInvalid(f"{label} denial did not increment its counter") from None
        else:
            raise ProtocolInvalid(f"{label} negative control bypassed RGB denial")

    expect_denial(CALIBRATION.read_bytes, "dataset_open", "Path.read_bytes")
    expect_denial(
        lambda: os.open(CALIBRATION, os.O_RDONLY),
        "dataset_os_open",
        "os.open",
    )
    expect_denial(lambda: PILImage.open(CALIBRATION), "pil_open", "PIL.Image.open")
    expect_denial(
        lambda: calibrated_module.load_calibrated_scene(CALIBRATED_SCENE),
        "calibrated_loader",
        "calibrated-module loader",
    )
    expect_denial(
        lambda: data_package.load_calibrated_scene(CALIBRATED_SCENE),
        "calibrated_loader",
        "cached data-package loader",
    )
    expect_denial(
        lambda: calibrated_module._resize_image(CALIBRATION, 1, 1),
        "calibrated_loader",
        "calibrated image loader",
    )
    expect_denial(lambda: scene_module.SceneData(), "scene_data", "scene-module SceneData")
    expect_denial(
        lambda: calibrated_module.SceneData(),
        "scene_data",
        "calibrated-module SceneData",
    )
    expect_denial(lambda: data_package.SceneData(), "scene_data", "cached SceneData")
    counters["negative_control_dataset_denials"] = counters["dataset_open"]
    counters["negative_control_os_open_denials"] = counters["dataset_os_open"]
    counters["negative_control_pil_denials"] = counters["pil_open"]
    counters["negative_control_calibrated_loader_denials"] = counters["calibrated_loader"]
    counters["negative_control_scene_data_denials"] = counters["scene_data"]
    for name in ("dataset_open", "dataset_os_open", "pil_open", "calibrated_loader", "scene_data"):
        counters[name] = 0
    return counters, (original_builtin_open, original_io_open, original_os_open)


def _validate_rgb_denial_counters(counters: Mapping[str, int]) -> None:
    expected_negative = {
        "negative_control_dataset_denials": 1,
        "negative_control_os_open_denials": 1,
        "negative_control_pil_denials": 1,
        "negative_control_calibrated_loader_denials": 3,
        "negative_control_scene_data_denials": 3,
    }
    operational = ("dataset_open", "dataset_os_open", "pil_open", "calibrated_loader", "scene_data")
    if any(counters.get(name) != value for name, value in expected_negative.items()) or any(
        counters.get(name) != 0 for name in operational
    ):
        raise ProtocolInvalid(f"RGB/source denial hooks were triggered or unproven: {counters}")


def validate_calibrated_checkpoint_evidence(
    history: Mapping[str, Any],
    callback_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate measured no-exhaustive-evaluation evidence for the calibrated exception."""
    expected_steps = [0, 10, 20, 40]
    if [int(item["step"]) for item in callback_events] != expected_steps:
        raise ProtocolInvalid("calibrated checkpoint callbacks differ from frozen schedule")
    if not all(item.get("detached") is True for item in callback_events):
        raise ProtocolInvalid("calibrated checkpoint callback received attached tensors")
    checkpoints = history.get("checkpoints")
    if (
        not isinstance(checkpoints, list)
        or [int(item["step"]) for item in checkpoints] != expected_steps
    ):
        raise ProtocolInvalid("calibrated checkpoint history differs from frozen schedule")
    if any(item.get("evaluation") is not None for item in checkpoints):
        raise ProtocolInvalid("calibrated trainer invoked exhaustive checkpoint evaluation")
    history_hashes = [item["snapshot_sha256"] for item in checkpoints]
    callback_hashes = [item["snapshot_sha256"] for item in callback_events]
    if history_hashes != callback_hashes:
        raise ProtocolInvalid("checkpoint history/callback snapshot hashes differ")
    evidence = {
        "enabled": history["checkpoint_risk_evaluation_enabled"],
        "evaluator_call_count": history["checkpoint_risk_evaluation_call_count"],
        "snapshot_count": history["checkpoint_snapshot_count"],
        "callback_call_count": history["checkpoint_callback_call_count"],
    }
    expected = {
        "enabled": False,
        "evaluator_call_count": 0,
        "snapshot_count": 4,
        "callback_call_count": 4,
    }
    if evidence != expected:
        raise ProtocolInvalid(f"calibrated checkpoint evidence differs: {evidence}")
    return evidence


def _validate_calibrated_training_bundle(
    inputs: ReconstructionInputs,
    acquisition: Mapping[str, Any],
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    """Rebind the strict Stage-2 input to the committed acquisition evidence."""
    if acquisition.get("status") != "PASS" or acquisition.get("plan_sha256") != plan_sha256:
        raise ProtocolInvalid("training acquisition does not bind the active calibrated plan")
    manifest_path = CALIBRATED_BUNDLE / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if acquisition.get("bundle_manifest_sha256") != manifest_sha256:
        raise ProtocolInvalid("training bundle manifest differs from teacher acquisition")
    if inputs.points is not None:
        raise ProtocolInvalid("calibrated compact training bundle unexpectedly contains points")
    if inputs.point_visibility is not None:
        raise ProtocolInvalid(
            "calibrated compact training bundle unexpectedly contains point visibility"
        )
    if inputs.bounds_hint is not None:
        raise ProtocolInvalid("calibrated compact training bundle unexpectedly contains bounds")
    if inputs.name != "compact_point_training_20260716":
        raise ProtocolInvalid("calibrated training bundle name differs from acquisition")
    if inputs.view_names != list(CALIBRATED_VIEWS):
        raise ProtocolInvalid("calibrated training view order differs from acquisition")
    views = acquisition.get("views")
    if not isinstance(views, list) or len(views) != len(CALIBRATED_VIEWS):
        raise ProtocolInvalid("teacher acquisition view records are incomplete")
    teacher_hashes = [field_hashes(field) for field in inputs.observations]
    if acquisition.get("bundle_teacher_hashes") != teacher_hashes:
        raise ProtocolInvalid("loaded teacher semantics differ from acquisition bundle binding")
    for expected_name, field, camera, record in zip(
        CALIBRATED_VIEWS,
        inputs.observations,
        inputs.cameras,
        views,
        strict=True,
    ):
        if (
            record.get("camera_id") != expected_name
            or field.view_id != expected_name
            or record.get("teacher_semantic_hashes") != field_hashes(field)
            or record.get("camera") != _camera_record(camera)
        ):
            raise ProtocolInvalid(
                f"loaded teacher/camera semantics differ for acquired view {expected_name}"
            )
    manifest = strict_json_load(manifest_path)
    if (
        manifest.get("name") != inputs.name
        or manifest.get("geometry") is not None
        or [record.get("view_id") for record in manifest.get("views", [])] != list(CALIBRATED_VIEWS)
        or [record.get("camera") for record in manifest.get("views", [])]
        != [_camera_record(camera) for camera in inputs.cameras]
    ):
        raise ProtocolInvalid("strict bundle manifest semantics differ from acquisition")
    return {
        "manifest_sha256": manifest_sha256,
        "manifest_semantic_digest": manifest["semantic_digest"],
        "teacher_semantic_aggregate": canonical_hash(teacher_hashes),
        "camera_semantic_aggregate": canonical_hash(
            [_camera_record(camera) for camera in inputs.cameras]
        ),
        "geometry_absent": True,
    }


def _initialize_compact_carve_after_preflight(
    inputs: ReconstructionInputs,
    train_config: Any,
    carve_config: Any,
    *,
    bundle_path: Path | None,
) -> tuple[Any, dict[str, Any], list[GaussianObservationIndex]]:
    """Enforce all frozen observation/archive budgets before Carve allocates an index."""
    from rtgs.lift.compact_carve import CompactCarveInitializer
    from rtgs.optim.compact_trainer import preflight_observations

    preflight = preflight_observations(inputs, train_config, bundle_path=bundle_path)
    if (
        carve_config.tile_size != train_config.teacher_tile_size
        or carve_config.max_index_entries_per_view != train_config.max_index_entries_per_view
        or carve_config.max_candidates_per_tile != train_config.max_candidates_per_tile
    ):
        raise ProtocolInvalid("Carve/trainer teacher-index caps or tile size differ")
    backends = [
        GaussianObservationIndex(
            field,
            tile_size=carve_config.tile_size,
            max_entries=carve_config.max_index_entries_per_view,
            max_candidates=carve_config.max_candidates_per_tile,
        )
        for field in inputs.observations
    ]
    result = CompactCarveInitializer(carve_config).initialize(inputs, backends=backends)
    return result, preflight, backends


def _calibrated_training_worker(
    plan_sha256: str,
    acquisition_sha256: str,
    result_path_text: str,
) -> None:
    result_path = Path(result_path_text)
    try:
        plan, seal = _load_calibrated_plan_and_seal(plan_sha256)
        counters, _ = _deny_rgb_access()
        inputs = ReconstructionInputs.load(
            CALIBRATED_BUNDLE,
            device="cpu",
            strict=True,
            limits=_calibrated_bundle_limits(),
        )
        acquisition = strict_json_load_bound(CALIBRATED_ACQUISITION, acquisition_sha256)
        bundle_binding = _validate_calibrated_training_bundle(
            inputs,
            acquisition,
            plan_sha256=plan_sha256,
        )
        from rtgs.lift.compact_carve import CompactCarveConfig
        from rtgs.optim.compact_trainer import (
            CompactTrainConfig,
            CompactTrainer,
            _gaussians_sha256,
        )

        carve_config = CompactCarveConfig(**plan["configuration"]["compact_carve"])
        callback_events = []

        def checkpoint_callback(snapshot: Gaussians3D, step: int) -> None:
            callback_events.append(
                {
                    "step": int(step),
                    "snapshot_sha256": _gaussians_sha256(snapshot),
                    "harness_snapshot_sha256": gaussians_hashes(snapshot)["aggregate"],
                    "detached": all(
                        not getattr(snapshot, field).requires_grad for field in GAUSSIAN_FIELDS
                    ),
                }
            )

        train_config = CompactTrainConfig(
            iterations=40,
            attempts_per_step=128,
            proposal_mode="pixel_gaussian",
            uniform_fraction=0.25,
            seed=75201,
            extent=None,
            device="cpu",
            lr_means=1.6e-4,
            lr_quats=1e-3,
            lr_scales=5e-3,
            lr_opacity=5e-2,
            lr_sh=2.5e-3,
            lr_sh_rest=2.5e-3 / 20.0,
            point_chunk=32,
            gaussian_chunk=64,
            outer_microbatch=32,
            query_component_chunk=64,
            teacher_tile_size=16,
            evaluation_chunk=256,
            checkpoints=(0, 10, 20, 40),
            evaluate_checkpoint_risks=False,
            sh_degree=0,
            sh_color_activation="hard",
            sh_smu1_mu=2.0 / 255.0,
            kernel_support_mode="hard",
            visibility_margin_sigma=3.0,
            max_views=MAX_VIEWS,
            max_fitted_pixels_per_view=MAX_FITTED_PIXELS_PER_VIEW,
            max_components_per_view=MAX_COMPONENTS_PER_VIEW,
            max_index_entries_per_view=MAX_TILE_ENTRIES_PER_VIEW,
            max_candidates_per_tile=MAX_CANDIDATES_PER_TILE,
            max_manifest_bytes=MAX_MANIFEST_BYTES,
            max_teacher_archives=MAX_TEACHER_ARCHIVES,
            max_teacher_archive_bytes=MAX_COMPRESSED_ARCHIVE_BYTES,
            max_total_teacher_archive_bytes=MAX_TOTAL_COMPRESSED_BYTES,
            max_zip_members=MAX_ZIP_MEMBERS,
            max_member_uncompressed_bytes=MAX_UNCOMPRESSED_MEMBER_BYTES,
            max_archive_uncompressed_bytes=MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        )
        initialization_result, preflight, query_backends = (
            _initialize_compact_carve_after_preflight(
                inputs,
                train_config,
                carve_config,
                bundle_path=CALIBRATED_BUNDLE,
            )
        )
        initialization = initialization_result.gaussians.detach()
        if initialization.n != 835 or initialization.sh_degree != 0:
            raise ProtocolInvalid("compact Carve did not produce the frozen 835-row initialization")
        initialization.save_ply(CALIBRATED_INIT)
        trainer = CompactTrainer(train_config)
        final, history = trainer.train(
            inputs,
            initialization,
            query_backends=query_backends,
            bundle_path=CALIBRATED_BUNDLE,
            checkpoint_callback=checkpoint_callback,
        )
        if final.n != 835 or final.sh_degree != 0:
            raise ProtocolInvalid("calibrated compact training changed frozen topology")
        final.save_ply(CALIBRATED_FINAL)
        checkpoint_evidence = validate_calibrated_checkpoint_evidence(history, callback_events)
        _validate_rgb_denial_counters(counters)
        _load_calibrated_plan_and_seal(plan_sha256)
        payload = {
            "status": "PASS",
            "source_aggregate": seal["source_aggregate"],
            "teacher_acquisition_sha256": acquisition_sha256,
            "bundle_manifest_sha256": sha256_file(CALIBRATED_BUNDLE / "manifest.json"),
            "bundle_binding": bundle_binding,
            "archive_stats": asdict(inputs.archive_stats)
            if inputs.archive_stats is not None
            else None,
            "compact_carve_config": asdict(carve_config),
            "preflight_before_compact_carve": preflight,
            "compact_carve_diagnostics": initialization_result.diagnostics,
            "train_config": asdict(train_config),
            "history": history,
            "history_sha256": canonical_hash(history),
            "checkpoint_callbacks": callback_events,
            "checkpoint_evidence": checkpoint_evidence,
            "denied_access_counts": counters,
            "gaussians_init_sha256": sha256_file(CALIBRATED_INIT),
            "gaussians_final_sha256": sha256_file(CALIBRATED_FINAL),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        _exclusive_json(result_path, payload)
    except BaseException as error:
        _worker_failure(result_path, error)
        raise


def _uniform_teacher_sample(
    field: GaussianObservationField,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    x = torch.randint(fit_width, (count,), generator=generator) + fit_x
    y = torch.randint(fit_height, (count,), generator=generator) + fit_y
    return torch.stack([x, y], dim=-1).to(field.dtype) + 0.5


def _sampled_teacher_evaluation_worker(
    plan_sha256: str,
    acquisition_sha256: str,
    initial_sha256: str,
    final_sha256: str,
    result_path_text: str,
) -> None:
    result_path = Path(result_path_text)
    try:
        _load_calibrated_plan_and_seal(plan_sha256)
        if (
            sha256_file(CALIBRATED_INIT) != initial_sha256
            or sha256_file(CALIBRATED_FINAL) != final_sha256
        ):
            raise ProtocolInvalid("frozen PLY changed before bounded teacher evaluation")
        counters, _ = _deny_rgb_access()
        inputs = ReconstructionInputs.load(
            CALIBRATED_BUNDLE,
            device="cpu",
            strict=True,
            limits=_calibrated_bundle_limits(),
        )
        acquisition = strict_json_load_bound(CALIBRATED_ACQUISITION, acquisition_sha256)
        bundle_binding = _validate_calibrated_training_bundle(
            inputs,
            acquisition,
            plan_sha256=plan_sha256,
        )
        models = {
            "initial": Gaussians3D.load_ply(CALIBRATED_INIT),
            "final": Gaussians3D.load_ply(CALIBRATED_FINAL),
        }
        renderer = TorchPointRasterizer(point_chunk=32, gaussian_chunk=64)
        backends = [
            GaussianObservationIndex(
                field,
                tile_size=16,
                max_entries=MAX_TILE_ENTRIES_PER_VIEW,
                max_candidates=MAX_CANDIDATES_PER_TILE,
            )
            for field in inputs.observations
        ]
        generator = torch.Generator(device="cpu").manual_seed(75202)
        records = []
        for view, (field, camera, backend) in enumerate(
            zip(inputs.observations, inputs.cameras, backends, strict=True)
        ):
            xy = _uniform_teacher_sample(field, 4096, generator)
            teacher = backend.query(xy, component_chunk=64).color
            model_records = {}
            for name, model in models.items():
                prediction = renderer.render_points(
                    model,
                    camera,
                    xy,
                    background=torch.zeros(3),
                    sh_degree=0,
                ).color
                difference = prediction.double() - teacher.double()
                sse = float(difference.square().sum(dtype=torch.float64))
                mse = sse / (4096 * 3)
                model_records[name] = {
                    "rgb_squared_error_sum": sse,
                    "scalar_count": 4096 * 3,
                    "mse": mse,
                    "psnr": -10.0 * math.log10(max(mse, 1e-30)),
                }
            records.append(
                {
                    "view": view,
                    "view_name": inputs.view_names[view],
                    "xy_sha256": tensor_hash(xy),
                    "teacher_sha256": tensor_hash(teacher),
                    "models": model_records,
                }
            )
        _validate_rgb_denial_counters(counters)
        means = {
            name: sum(item["models"][name]["mse"] for item in records) / len(records)
            for name in models
        }
        if (
            sha256_file(CALIBRATED_INIT) != initial_sha256
            or sha256_file(CALIBRATED_FINAL) != final_sha256
        ):
            raise ProtocolInvalid("frozen PLY changed during bounded teacher evaluation")
        _load_calibrated_plan_and_seal(plan_sha256)
        _exclusive_json(
            result_path,
            {
                "status": "PASS",
                "seed": 75202,
                "samples_per_view": 4096,
                "with_replacement": True,
                "gaussians_init_sha256": initial_sha256,
                "gaussians_final_sha256": final_sha256,
                "teacher_acquisition_sha256": acquisition_sha256,
                "bundle_binding": bundle_binding,
                "records": records,
                "equal_view_mse": means,
                "denied_access_counts": counters,
            },
        )
    except BaseException as error:
        _worker_failure(result_path, error)
        raise


def _heldout_rgb_evaluation_worker(
    plan_sha256: str,
    initial_sha256: str,
    final_sha256: str,
    result_path_text: str,
) -> None:
    result_path = Path(result_path_text)
    try:
        plan, _ = _load_calibrated_plan_and_seal(plan_sha256)
        if (
            sha256_file(CALIBRATED_INIT) != initial_sha256
            or sha256_file(CALIBRATED_FINAL) != final_sha256
        ):
            raise ProtocolInvalid("frozen PLY changed before held-out decode")
        if sha256_file(CALIBRATION) != plan["input_hashes"]["calibration"]:
            raise ProtocolInvalid("calibration changed before held-out decode")
        image_path = _capture_image_paths()[CALIBRATED_HELDOUT_VIEW]
        image_binding = plan["input_hashes"]["images"][CALIBRATED_HELDOUT_VIEW]
        if (
            image_path.relative_to(ROOT).as_posix() != image_binding["path"]
            or sha256_file(image_path) != image_binding["sha256"]
        ):
            raise ProtocolInvalid("held-out RGB changed before decode")
        mask_binding = plan["input_hashes"]["heldout_mask"]
        mask_path = _heldout_mask_path()
        if (
            mask_path.relative_to(ROOT).as_posix() != mask_binding["path"]
            or sha256_file(mask_path) != mask_binding["sha256"]
        ):
            raise ProtocolInvalid("held-out mask changed before decode")
        from PIL import Image as PILImage

        from rtgs.data.calibrated import _resize_image, _undistort

        camera, distortion = _camera_from_calibration(CALIBRATED_HELDOUT_VIEW)
        with PILImage.open(image_path) as source:
            image_native_size = tuple(int(value) for value in source.size)
        with PILImage.open(mask_path) as source:
            mask_native_size = tuple(int(value) for value in source.size)
        if image_native_size != (5328, 4608) or mask_native_size != (5328, 4608):
            raise ProtocolInvalid(
                "held-out RGB/mask native dimensions differ from (5328,4608): "
                f"rgb={image_native_size}, mask={mask_native_size}"
            )
        image = _undistort(
            _resize_image(image_path, camera.width, camera.height),
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            distortion,
        )
        mask = _undistort(
            _resize_image(mask_path, camera.width, camera.height, mask=True),
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            distortion,
            mask=True,
        )
        generator = torch.Generator(device="cpu").manual_seed(75203)
        x = torch.randint(camera.width, (4096,), generator=generator)
        y = torch.randint(camera.height, (4096,), generator=generator)
        xy = torch.stack([x, y], dim=-1).float() + 0.5
        reference = image[y, x]
        foreground = mask[y, x] > 0.5
        if not bool(foreground.any()):
            raise ProtocolInvalid("held-out uniform sample contains no foreground")
        renderer = TorchPointRasterizer(point_chunk=32, gaussian_chunk=64)
        records = {}
        for name, path in (("initial", CALIBRATED_INIT), ("final", CALIBRATED_FINAL)):
            model = Gaussians3D.load_ply(path)
            prediction = renderer.render_points(
                model,
                camera,
                xy,
                background=torch.zeros(3),
                sh_degree=0,
            ).color
            error = prediction.double() - reference.double()
            all_mse = float(error.square().mean(dtype=torch.float64))
            foreground_mse = float(error[foreground].square().mean(dtype=torch.float64))
            records[name] = {
                "all_mse": all_mse,
                "all_psnr": -10.0 * math.log10(max(all_mse, 1e-30)),
                "foreground_mse": foreground_mse,
                "foreground_psnr": -10.0 * math.log10(max(foreground_mse, 1e-30)),
                "prediction_below_zero_fraction": float((prediction < 0).float().mean()),
                "prediction_above_one_fraction": float((prediction > 1).float().mean()),
            }
        _load_calibrated_plan_and_seal(plan_sha256)
        if (
            sha256_file(CALIBRATED_INIT) != initial_sha256
            or sha256_file(CALIBRATED_FINAL) != final_sha256
        ):
            raise ProtocolInvalid("frozen PLY changed during held-out evaluation")
        _exclusive_json(
            result_path,
            {
                "status": "PASS",
                "seed": 75203,
                "view_name": CALIBRATED_HELDOUT_VIEW,
                "samples": 4096,
                "foreground_samples": int(foreground.sum()),
                "with_replacement": True,
                "native_rgb_dimensions": list(image_native_size),
                "native_mask_dimensions": list(mask_native_size),
                "heldout_mask_binding": mask_binding,
                "xy_sha256": tensor_hash(xy),
                "reference_sha256": tensor_hash(reference),
                "mask_sample_sha256": tensor_hash(foreground),
                "gaussians_init_sha256": initial_sha256,
                "gaussians_final_sha256": final_sha256,
                "models": records,
            },
        )
    except BaseException as error:
        _worker_failure(result_path, error)
        raise


def _linux_process_listener_binding(pid: int, host: str, port: int) -> dict[str, Any]:
    """Bind a Linux TCP listener to an FD owned by one exact process."""
    if sys.platform != "linux":
        raise ProtocolInvalid("viewer listener ownership verification requires Linux /proc")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(host, str)
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 0 < port < 65536
    ):
        raise ProtocolInvalid("viewer listener ownership received an invalid pid/host/port")

    import ipaddress
    import socket

    try:
        target_address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ProtocolInvalid("viewer listener ownership received an invalid host") from error

    process_inodes = set()
    descriptor_directory = Path(f"/proc/{pid}/fd")
    try:
        descriptors = list(descriptor_directory.iterdir())
    except (FileNotFoundError, PermissionError) as error:
        raise ProtocolInvalid("cannot inspect launched viewer file descriptors") from error
    for descriptor in descriptors:
        try:
            target = os.readlink(descriptor)
        except (FileNotFoundError, PermissionError):
            continue
        match = re.fullmatch(r"socket:\[(\d+)\]", target)
        if match is not None:
            process_inodes.add(match.group(1))

    def decode_address(table: str, encoded: str) -> str:
        try:
            packed = bytes.fromhex(encoded)
            if table == "tcp":
                return socket.inet_ntop(socket.AF_INET, packed[::-1])
            reordered = b"".join(
                packed[offset : offset + 4][::-1] for offset in range(0, len(packed), 4)
            )
            return socket.inet_ntop(socket.AF_INET6, reordered)
        except (OSError, ValueError) as error:
            raise ProtocolInvalid("cannot decode Linux TCP listener address") from error

    listeners = []
    for table in ("tcp", "tcp6"):
        table_path = Path(f"/proc/{pid}/net/{table}")
        try:
            lines = table_path.read_text(encoding="ascii").splitlines()[1:]
        except (FileNotFoundError, PermissionError) as error:
            raise ProtocolInvalid("cannot inspect launched viewer TCP namespace") from error
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                encoded_address, port_text = fields[1].split(":", maxsplit=1)
                local_port = int(port_text, 16)
            except ValueError as error:
                raise ProtocolInvalid("malformed Linux TCP listener record") from error
            if local_port != port:
                continue
            address = ipaddress.ip_address(decode_address(table, encoded_address))
            accepts_target = address == target_address or (
                address.is_unspecified and address.version == target_address.version
            )
            if (
                isinstance(address, ipaddress.IPv6Address)
                and address.ipv4_mapped is not None
                and address.ipv4_mapped == target_address
            ):
                accepts_target = True
            if accepts_target:
                listeners.append(
                    {
                        "table": table,
                        "local_address": str(address),
                        "local_port": local_port,
                        "inode": fields[9],
                    }
                )
    listeners.sort(key=lambda record: (record["table"], record["local_address"], record["inode"]))
    owned = [record for record in listeners if record["inode"] in process_inodes]
    return {
        "method": "linux_proc_fd_tcp_v1",
        "pid": pid,
        "host": host,
        "port": port,
        "process_socket_inodes": sorted(process_inodes, key=int),
        "target_listener_records": listeners,
        "owned_target_listener_records": owned,
        "owned": bool(owned),
    }


def viewer_http_response_is_valid(
    *,
    process_alive: bool,
    listener_owned: bool,
    status: int,
    body: bytes,
) -> bool:
    """Require branded HTML from a port owned by the launched viewer process."""
    lowered = body.lower()
    return (
        process_alive
        and listener_owned
        and status == 200
        and (b"viser" in lowered or b"realtime-gs" in lowered)
    )


def _loaded_gsplat_extension_binding() -> dict[str, str]:
    import gsplat.cuda._backend as backend

    extension = getattr(backend, "_C", None)
    path_text = None if extension is None else getattr(extension, "__file__", None)
    if path_text is None:
        raise ProtocolInvalid("gsplat render did not expose a loaded CUDA extension")
    path = Path(path_text).resolve()
    if not path.is_file() or path.suffix not in {".so", ".pyd"}:
        raise ProtocolInvalid(f"loaded gsplat extension is not a binary module: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _verify_binary_file_binding(binding: Mapping[str, Any], *, context: str) -> None:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ProtocolInvalid(f"{context} binary binding has the wrong key set")
    if not isinstance(binding["path"], str) or not _is_sha256(binding["sha256"]):
        raise ProtocolInvalid(f"{context} binary binding has invalid field types")
    path = Path(binding["path"]).resolve()
    if str(path) != binding["path"] or not path.is_file() or path.suffix not in {".so", ".pyd"}:
        raise ProtocolInvalid(f"{context} binary binding is not a resolved extension module")
    if sha256_file(path) != binding["sha256"]:
        raise ProtocolInvalid(f"{context} binary changed after it was loaded")


def _save_exact_snapshot(path: Path, color: torch.Tensor) -> dict[str, Any]:
    from PIL import Image as PILImage

    if tuple(color.shape) != (4608, 5328, 3) or not bool(torch.isfinite(color).all()):
        raise ProtocolInvalid(f"exact viewer snapshot has invalid shape/values: {color.shape}")
    array = np.ascontiguousarray(
        (color.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    )
    with path.open("xb") as stream:
        PILImage.fromarray(array).save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if path.stat().st_size <= 0:
        raise ProtocolInvalid("exact viewer snapshot PNG is empty")
    with PILImage.open(path) as decoded:
        if decoded.size != (5328, 4608):
            raise ProtocolInvalid("saved exact viewer snapshot has wrong dimensions")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dimensions": [5328, 4608],
        "color_tensor_sha256": tensor_hash(color),
    }


def _render_programmatic_viewer_snapshots(
    *,
    plan: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    initial_sha256: str,
    final_sha256: str,
) -> dict[str, Any]:
    from rtgs.viewer import render_exact_snapshot

    if _gsplat_binding() != plan["gsplat"]:
        raise ProtocolInvalid("installed gsplat package changed before exact viewer render")
    camera_record = next(
        (
            record["camera"]
            for record in acquisition["views"]
            if record.get("camera_id") == CALIBRATED_VIEWS[0]
        ),
        None,
    )
    if camera_record is None:
        raise ProtocolInvalid("acquisition lacks the frozen exact-snapshot camera")
    camera = _camera_from_record(camera_record)
    if (camera.width, camera.height) != (5328, 4608):
        raise ProtocolInvalid("exact-snapshot camera is not native full resolution")
    records = {}
    for name, path, expected_sha256 in (
        ("initial", CALIBRATED_INIT, initial_sha256),
        ("final", CALIBRATED_FINAL, final_sha256),
    ):
        if sha256_file(path) != expected_sha256:
            raise ProtocolInvalid(f"{name} PLY changed before exact gsplat render")
        model = Gaussians3D.load_ply(path)
        snapshot = render_exact_snapshot(
            model,
            camera,
            device="cuda",
            rasterizer="gsplat",
            packed=False,
            antialiased=False,
        )
        if snapshot.backend != "rtgs.render.gsplat_backend.GsplatRasterizer":
            raise ProtocolInvalid(f"exact snapshot resolved the wrong backend: {snapshot.backend}")
        if not snapshot.device.startswith("cuda"):
            raise ProtocolInvalid(f"exact snapshot did not execute on CUDA: {snapshot.device}")
        png = CALIBRATED_SNAPSHOTS / f"{name}_{CALIBRATED_VIEWS[0]}_gsplat.png"
        records[name] = {
            **_save_exact_snapshot(png, snapshot.color),
            "backend": snapshot.backend,
            "device": snapshot.device,
            "gaussians_sha256": expected_sha256,
            "n_gaussians": model.n,
            "packed": False,
            "antialiased": False,
        }
        del snapshot, model
        torch.cuda.empty_cache()
    return {
        "camera_id": CALIBRATED_VIEWS[0],
        "camera": camera_record,
        "models": records,
        "gsplat_package": plan["gsplat"],
        "loaded_cuda_extension": _loaded_gsplat_extension_binding(),
    }


def _exact_snapshot_worker(
    plan_sha256: str,
    acquisition_sha256: str,
    initial_sha256: str,
    final_sha256: str,
    result_path_text: str,
) -> None:
    # This is intentionally the first observation made by the spawn target.  ``spawn`` starts a
    # fresh interpreter, so a plan-bound LD_PRELOAD set by the parent is active before torch or
    # gsplat imports and is not merely assigned inside an already-running Miniconda process.
    ld_preload_at_startup = os.environ.get("LD_PRELOAD")
    default_namespace_abi = None
    result_path = Path(result_path_text)
    try:
        plan, _ = _load_calibrated_plan_and_seal(plan_sha256)
        expected_preload = plan["configuration"]["ld_preload"]
        if ld_preload_at_startup != expected_preload:
            raise ProtocolInvalid("exact-snapshot worker did not start with the frozen LD_PRELOAD")
        abi_preload_binding = _verify_abi_preload_file_binding(
            plan["configuration"]["ld_preload_binding"]
        )
        if abi_preload_binding["requested_path"] != expected_preload:
            raise ProtocolInvalid("exact-snapshot worker preload path/binding differ")
        default_namespace_abi = _default_namespace_abi_resolution(abi_preload_binding)
        acquisition = strict_json_load_bound(CALIBRATED_ACQUISITION, acquisition_sha256)
        if acquisition.get("plan_sha256") != plan_sha256:
            raise ProtocolInvalid("exact-snapshot worker loaded the wrong acquisition plan")
        if (
            sha256_file(CALIBRATED_INIT) != initial_sha256
            or sha256_file(CALIBRATED_FINAL) != final_sha256
        ):
            raise ProtocolInvalid("frozen PLY changed before exact-snapshot worker render")
        exact_snapshots = _render_programmatic_viewer_snapshots(
            plan=plan,
            acquisition=acquisition,
            initial_sha256=initial_sha256,
            final_sha256=final_sha256,
        )
        plan_after, _ = _load_calibrated_plan_and_seal(plan_sha256)
        acquisition_after = strict_json_load_bound(
            CALIBRATED_ACQUISITION,
            acquisition_sha256,
        )
        if plan_after != plan or acquisition_after != acquisition:
            raise ProtocolInvalid("exact-snapshot inputs changed during worker render")
        if _default_namespace_abi_resolution(abi_preload_binding) != default_namespace_abi:
            raise ProtocolInvalid("default-namespace ABI resolution changed during rendering")
        if (
            sha256_file(CALIBRATED_INIT) != initial_sha256
            or sha256_file(CALIBRATED_FINAL) != final_sha256
        ):
            raise ProtocolInvalid("frozen PLY changed during exact-snapshot worker render")
        _exclusive_json(
            result_path,
            {
                "artifact_type": "compact_point_training_exact_snapshot_worker_v2",
                "status": "PASS",
                "plan_sha256": plan_sha256,
                "acquisition_sha256": acquisition_sha256,
                "gaussians_init_sha256": initial_sha256,
                "gaussians_final_sha256": final_sha256,
                "ld_preload_at_startup": ld_preload_at_startup,
                "default_namespace_abi": default_namespace_abi,
                "exact_gsplat_snapshots": exact_snapshots,
            },
        )
    except BaseException as error:
        _worker_failure(
            result_path,
            error,
            ld_preload_at_startup=ld_preload_at_startup,
            default_namespace_abi=default_namespace_abi,
        )
        raise


def _validate_exact_snapshot_worker_result(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    plan_sha256: str,
    acquisition_sha256: str,
    initial_sha256: str,
    final_sha256: str,
) -> dict[str, Any]:
    expected_keys = {
        "artifact_type",
        "status",
        "plan_sha256",
        "acquisition_sha256",
        "gaussians_init_sha256",
        "gaussians_final_sha256",
        "ld_preload_at_startup",
        "default_namespace_abi",
        "exact_gsplat_snapshots",
    }
    if set(payload) != expected_keys:
        raise ProtocolInvalid("exact-snapshot worker result has the wrong key set")
    expected_bindings = {
        "artifact_type": "compact_point_training_exact_snapshot_worker_v2",
        "status": "PASS",
        "plan_sha256": plan_sha256,
        "acquisition_sha256": acquisition_sha256,
        "gaussians_init_sha256": initial_sha256,
        "gaussians_final_sha256": final_sha256,
        "ld_preload_at_startup": plan["configuration"]["ld_preload"],
    }
    if any(payload.get(name) != value for name, value in expected_bindings.items()):
        raise ProtocolInvalid("exact-snapshot worker returned the wrong artifact bindings")
    _verify_default_namespace_abi_resolution(
        payload["default_namespace_abi"],
        binding=plan["configuration"]["ld_preload_binding"],
    )
    exact = payload["exact_gsplat_snapshots"]
    if not isinstance(exact, Mapping) or set(exact) != {
        "camera_id",
        "camera",
        "models",
        "gsplat_package",
        "loaded_cuda_extension",
    }:
        raise ProtocolInvalid("exact-snapshot worker returned the wrong snapshot schema")
    camera_record = next(
        (
            record["camera"]
            for record in acquisition["views"]
            if record.get("camera_id") == CALIBRATED_VIEWS[0]
        ),
        None,
    )
    if (
        exact["camera_id"] != CALIBRATED_VIEWS[0]
        or exact["camera"] != camera_record
        or exact["gsplat_package"] != plan["gsplat"]
    ):
        raise ProtocolInvalid("exact-snapshot worker returned the wrong camera/package binding")
    _verify_binary_file_binding(
        exact["loaded_cuda_extension"],
        context="exact-snapshot gsplat CUDA extension",
    )
    models = exact["models"]
    if not isinstance(models, Mapping) or set(models) != {"initial", "final"}:
        raise ProtocolInvalid("exact-snapshot worker did not return exactly two models")
    expected_models = {
        "initial": (CALIBRATED_INIT, initial_sha256),
        "final": (CALIBRATED_FINAL, final_sha256),
    }
    expected_model_keys = {
        "path",
        "sha256",
        "bytes",
        "dimensions",
        "color_tensor_sha256",
        "backend",
        "device",
        "gaussians_sha256",
        "n_gaussians",
        "packed",
        "antialiased",
    }
    for name, (ply_path, ply_sha256) in expected_models.items():
        record = models[name]
        snapshot_path = CALIBRATED_SNAPSHOTS / f"{name}_{CALIBRATED_VIEWS[0]}_gsplat.png"
        if not isinstance(record, Mapping) or set(record) != expected_model_keys:
            raise ProtocolInvalid(f"exact-snapshot {name} record has the wrong schema")
        if (
            record["path"] != snapshot_path.relative_to(ROOT).as_posix()
            or record["gaussians_sha256"] != ply_sha256
            or sha256_file(ply_path) != ply_sha256
            or record["dimensions"] != [5328, 4608]
            or record["backend"] != "rtgs.render.gsplat_backend.GsplatRasterizer"
            or not str(record["device"]).startswith("cuda")
            or record["packed"] is not False
            or record["antialiased"] is not False
            or record["n_gaussians"] != 835
            or isinstance(record["n_gaussians"], bool)
            or not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] <= 0
            or not all(
                _is_sha256(record[field])
                for field in ("sha256", "color_tensor_sha256", "gaussians_sha256")
            )
            or not snapshot_path.is_file()
            or snapshot_path.stat().st_size != record["bytes"]
            or sha256_file(snapshot_path) != record["sha256"]
        ):
            raise ProtocolInvalid(f"exact-snapshot {name} artifact binding differs")
    return dict(exact)


def _spawn_exact_snapshot_worker(
    *,
    plan: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    plan_sha256: str,
    acquisition_sha256: str,
    initial_sha256: str,
    final_sha256: str,
    result_path: Path,
) -> dict[str, Any]:
    abi_preload_binding = _verify_abi_preload_file_binding(
        plan["configuration"]["ld_preload_binding"]
    )
    if (
        abi_preload_binding["requested_path"] != plan["configuration"]["ld_preload"]
        or os.environ.get("LD_PRELOAD") != abi_preload_binding["requested_path"]
    ):
        raise ProtocolInvalid("parent did not set frozen LD_PRELOAD before snapshot spawn")
    payload = _spawn_file_worker(
        _exact_snapshot_worker,
        (
            plan_sha256,
            acquisition_sha256,
            initial_sha256,
            final_sha256,
            str(result_path),
        ),
        result_path,
        name="compact-exact-gsplat-snapshots",
        timeout=1800.0,
    )
    return _validate_exact_snapshot_worker_result(
        payload,
        plan=plan,
        acquisition=acquisition,
        plan_sha256=plan_sha256,
        acquisition_sha256=acquisition_sha256,
        initial_sha256=initial_sha256,
        final_sha256=final_sha256,
    )


def _viewer_smoke(
    initial_sha256: str,
    final_sha256: str,
    *,
    plan_sha256: str,
    acquisition_sha256: str,
) -> dict[str, Any]:
    if (
        sha256_file(CALIBRATED_INIT) != initial_sha256
        or sha256_file(CALIBRATED_FINAL) != final_sha256
    ):
        raise ProtocolInvalid("PLY changed before viewer smoke")
    plan = strict_json_load_bound(CALIBRATED_PLAN, plan_sha256)
    acquisition = strict_json_load_bound(CALIBRATED_ACQUISITION, acquisition_sha256)
    viewer_rgb_binding = _viewer_rgb_directory_binding()
    if viewer_rgb_binding != plan["input_hashes"]["viewer_rgb_directory"]:
        raise ProtocolInvalid("viewer RGB directory/listing changed after calibrated planning")
    from PIL import Image as PILImage

    for record in viewer_rgb_binding["selected"]:
        with PILImage.open(ROOT / record["path"]) as source:
            dimensions = list(source.size)
        if dimensions != record["expected_native_dimensions"]:
            raise ProtocolInvalid(
                f"viewer source {record['view_name']} is not native full resolution"
            )
        if record["mask"] is not None:
            with PILImage.open(ROOT / record["mask"]["path"]) as source:
                mask_dimensions = list(source.size)
            if mask_dimensions != record["expected_native_dimensions"]:
                raise ProtocolInvalid(
                    f"viewer mask {record['mask']['path']} is not native full resolution"
                )
    with tempfile.TemporaryDirectory(prefix="rtgs-exact-gsplat-snapshots-") as temporary:
        exact_snapshots = _spawn_exact_snapshot_worker(
            plan=plan,
            acquisition=acquisition,
            plan_sha256=plan_sha256,
            acquisition_sha256=acquisition_sha256,
            initial_sha256=initial_sha256,
            final_sha256=final_sha256,
            result_path=Path(temporary) / "result.json",
        )
    if _viewer_rgb_directory_binding() != viewer_rgb_binding:
        raise ProtocolInvalid("viewer RGB/mask inputs changed immediately before launch")
    process = subprocess.Popen(
        CALIBRATED_VIEWER_COMMAND,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    connected = False
    http_status = None
    http_body_sha256 = None
    listener_binding = None
    viewer_url = f"http://{CALIBRATED_VIEWER_HOST}:{CALIBRATED_VIEWER_PORT}"
    started = time.monotonic()
    try:
        while time.monotonic() - started < 180.0:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(viewer_url, timeout=1.0) as response:
                    body = response.read(1_048_576)
                    http_status = int(response.status)
                candidate_listener = _linux_process_listener_binding(
                    process.pid,
                    CALIBRATED_VIEWER_HOST,
                    CALIBRATED_VIEWER_PORT,
                )
                if viewer_http_response_is_valid(
                    process_alive=process.poll() is None,
                    listener_owned=candidate_listener["owned"],
                    status=http_status,
                    body=body,
                ):
                    time.sleep(1.0)
                    if process.poll() is not None:
                        continue
                    candidate_listener = _linux_process_listener_binding(
                        process.pid,
                        CALIBRATED_VIEWER_HOST,
                        CALIBRATED_VIEWER_PORT,
                    )
                    if not candidate_listener["owned"]:
                        continue
                    connected = True
                    listener_binding = candidate_listener
                    http_body_sha256 = sha256_bytes(body)
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(1.0)
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=30.0)
    if not connected:
        raise RuntimeError(f"viewer did not accept connections; output:\n{output}")
    if listener_binding is None or not listener_binding["owned"]:
        raise ProtocolInvalid("viewer smoke lacks launched-process listener ownership")
    _load_calibrated_plan_and_seal(plan_sha256)
    if _gsplat_binding() != plan["gsplat"]:
        raise ProtocolInvalid("installed gsplat package changed during viewer execution")
    _verify_binary_file_binding(
        exact_snapshots["loaded_cuda_extension"],
        context="exact-snapshot gsplat CUDA extension",
    )
    if _viewer_rgb_directory_binding() != viewer_rgb_binding:
        raise ProtocolInvalid("viewer RGB/mask inputs changed during viewer execution")
    if (
        sha256_file(CALIBRATED_INIT) != initial_sha256
        or sha256_file(CALIBRATED_FINAL) != final_sha256
    ):
        raise ProtocolInvalid("viewer execution changed a frozen PLY")
    snapshots = sorted(
        {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in CALIBRATED_SNAPSHOTS.rglob("*")
            if path.is_file()
        }.items()
    )
    expected_snapshot_paths = {record["path"] for record in exact_snapshots["models"].values()}
    if not expected_snapshot_paths or not expected_snapshot_paths.issubset(dict(snapshots)):
        raise ProtocolInvalid("viewer smoke lacks required nonempty exact gsplat snapshots")
    return {
        "status": "PASS",
        "command": list(CALIBRATED_VIEWER_COMMAND),
        "url": viewer_url,
        "connected": connected,
        "http_status": http_status,
        "http_body_sha256": http_body_sha256,
        "listener_binding": listener_binding,
        "stdout_sha256": sha256_bytes(output.encode("utf-8")),
        "snapshots": dict(snapshots),
        "exact_gsplat_snapshots": exact_snapshots,
        "viewer_rgb_binding": viewer_rgb_binding,
        "gaussians_init_sha256": initial_sha256,
        "gaussians_final_sha256": final_sha256,
    }


def run_calibrated() -> int:
    """Execute the frozen full-resolution, RGB-boundary-safe calibrated interaction."""
    seal = load_and_verify_seal()
    verdict, bindings = _audit_authorization(seal)
    collisions = _calibrated_collisions()
    if collisions:
        raise FileExistsError(f"calibrated once-only namespace is already consumed: {collisions}")
    plan = _calibrated_plan(seal=seal, audit_verdict=verdict, audit_bindings=bindings)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    plan_sha256 = _exclusive_json(CALIBRATED_PLAN, plan)
    attempt = {
        "artifact_type": "compact_point_training_calibrated_attempt_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "plan_sha256": plan_sha256,
        "audit_sha256": sha256_file(AUDIT),
        "external_structsplat": plan["external_structsplat"],
        "outputs": plan["outputs"],
    }
    attempt_sha256 = _exclusive_json(CALIBRATED_ATTEMPT, attempt)
    try:
        preload_binding = _verify_abi_preload_file_binding(
            plan["configuration"]["ld_preload_binding"]
        )
        if preload_binding["requested_path"] != plan["configuration"]["ld_preload"]:
            raise ProtocolInvalid("calibrated LD_PRELOAD path/binding differ")
        preload = Path(preload_binding["requested_path"])
        if not preload.is_file():
            raise RuntimeError(f"frozen LD_PRELOAD library is unavailable: {preload}")
        if not torch.cuda.is_available():
            raise RuntimeError("full-resolution StructSplat acquisition requires CUDA")
        os.environ["LD_PRELOAD"] = str(preload)
        with tempfile.TemporaryDirectory(prefix="rtgs-compact-calibrated-") as temporary:
            scratch = Path(temporary)
            _load_calibrated_plan_and_seal(plan_sha256)
            inputs, acquisition_records = _acquire_calibrated_teachers(
                plan,
                plan_sha256,
                scratch,
            )
            _load_calibrated_plan_and_seal(plan_sha256)
            acquisition_payload = {
                "artifact_type": "compact_point_training_teacher_acquisition_v1",
                "status": "PASS",
                "plan_sha256": plan_sha256,
                "external_structsplat": plan["external_structsplat"],
                "views": acquisition_records,
                "bundle_manifest_sha256": sha256_file(CALIBRATED_BUNDLE / "manifest.json"),
                "bundle_teacher_hashes": [field_hashes(field) for field in inputs.observations],
                "archive_stats": (
                    asdict(inputs.archive_stats) if inputs.archive_stats is not None else None
                ),
            }
            acquisition_sha256 = _exclusive_json(
                CALIBRATED_ACQUISITION,
                acquisition_payload,
            )

            training_result_path = scratch / "training.json"
            _load_calibrated_plan_and_seal(plan_sha256)
            training = _spawn_file_worker(
                _calibrated_training_worker,
                (plan_sha256, acquisition_sha256, str(training_result_path)),
                training_result_path,
                name="compact-rgb-denied-training",
                timeout=7200.0,
            )
            initial_sha256 = sha256_file(CALIBRATED_INIT)
            final_sha256 = sha256_file(CALIBRATED_FINAL)
            if (
                training["gaussians_init_sha256"] != initial_sha256
                or training["gaussians_final_sha256"] != final_sha256
            ):
                raise ProtocolInvalid("training child PLY bindings differ after freeze")
            if training["bundle_manifest_sha256"] != acquisition_payload["bundle_manifest_sha256"]:
                raise ProtocolInvalid("training child bundle differs from teacher acquisition")
            if training["teacher_acquisition_sha256"] != acquisition_sha256:
                raise ProtocolInvalid("training child binds the wrong teacher acquisition")

            sample_result_path = scratch / "sampled_teacher_evaluation.json"
            _load_calibrated_plan_and_seal(plan_sha256)
            sampled = _spawn_file_worker(
                _sampled_teacher_evaluation_worker,
                (
                    plan_sha256,
                    acquisition_sha256,
                    initial_sha256,
                    final_sha256,
                    str(sample_result_path),
                ),
                sample_result_path,
                name="compact-bounded-teacher-evaluation",
                timeout=1800.0,
            )
            if (
                sampled["gaussians_init_sha256"] != initial_sha256
                or sampled["gaussians_final_sha256"] != final_sha256
                or sampled["teacher_acquisition_sha256"] != acquisition_sha256
                or sampled["bundle_binding"] != training["bundle_binding"]
            ):
                raise ProtocolInvalid("bounded teacher evaluator returned wrong artifact bindings")
            compact_payload = {
                "artifact_type": "compact_point_training_calibrated_raw_v1",
                "status": "PASS",
                "plan_sha256": plan_sha256,
                "attempt_sha256": attempt_sha256,
                "teacher_acquisition_sha256": acquisition_sha256,
                "gaussians_init_sha256": initial_sha256,
                "gaussians_final_sha256": final_sha256,
                "training": training,
                "sampled_teacher_evaluation": sampled,
            }
            _load_calibrated_plan_and_seal(plan_sha256)
            compact_sha256 = _exclusive_json(CALIBRATED_TRAINING_RAW, compact_payload)

            _load_calibrated_plan_and_seal(plan_sha256)
            heldout = _spawn_file_worker(
                _heldout_rgb_evaluation_worker,
                (
                    plan_sha256,
                    initial_sha256,
                    final_sha256,
                    str(CALIBRATED_HELDOUT),
                ),
                CALIBRATED_HELDOUT,
                name="compact-heldout-evaluation",
                timeout=1800.0,
            )
            heldout_sha256 = sha256_file(CALIBRATED_HELDOUT)
            if (
                sha256_file(CALIBRATED_INIT) != initial_sha256
                or sha256_file(CALIBRATED_FINAL) != final_sha256
            ):
                raise ProtocolInvalid("held-out evaluation changed a frozen PLY")
            CALIBRATED_SNAPSHOTS.mkdir(parents=False, exist_ok=False)
            _load_calibrated_plan_and_seal(plan_sha256)
            viewer = _viewer_smoke(
                initial_sha256,
                final_sha256,
                plan_sha256=plan_sha256,
                acquisition_sha256=acquisition_sha256,
            )
            viewer_sha256 = _exclusive_json(CALIBRATED_VIEWER, viewer)
            result_payload = {
                "artifact_type": "compact_point_training_calibrated_result_v1",
                "status": "PASS",
                "decision_bearing": False,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "claim_boundary": (
                    "bounded full-resolution integration diagnostics only; no official decision, "
                    "novel-view quality, RGB-baseline, speed, memory, density, or default claim"
                ),
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "plan_sha256": plan_sha256,
                "attempt_sha256": attempt_sha256,
                "teacher_acquisition_sha256": acquisition_sha256,
                "compact_training_raw_sha256": compact_sha256,
                "heldout_evaluation_sha256": heldout_sha256,
                "viewer_smoke_sha256": viewer_sha256,
                "gaussians_init_sha256": initial_sha256,
                "gaussians_final_sha256": final_sha256,
                "n_init_3d": 835,
                "n_opt_3d": 835,
                "n_init_2d": inputs.n_init_2d,
                "n_opt_2d": inputs.n_opt_2d,
                "sampled_teacher_equal_view_mse": sampled["equal_view_mse"],
                "heldout_models": heldout["models"],
                "viewer_command": list(CALIBRATED_VIEWER_COMMAND),
                "viewer_url": viewer["url"],
                "external_structsplat": plan["external_structsplat"],
                "gsplat": plan["gsplat"],
            }
        # Let temporary-directory cleanup complete before the terminal RESULT commit.
        _load_calibrated_plan_and_seal(plan_sha256)
        digest = _exclusive_json(CALIBRATED_RESULT, result_payload)
    except BaseException as error:
        payload = _bounded_calibrated_failure(
            error,
            plan_sha256=plan_sha256,
            attempt_sha256=attempt_sha256,
        )
        digest = _exclusive_json(CALIBRATED_RESULT, payload)
        print(
            f"saved bounded calibrated FAIL {CALIBRATED_RESULT.relative_to(ROOT)} "
            f"(sha256={digest})",
            flush=True,
        )
        return 2
    # A returned PASS RESULT commit is terminal; display failures cannot route into FAIL writing.
    print(
        f"saved calibrated integration {CALIBRATED_RESULT.relative_to(ROOT)} (sha256={digest})",
        flush=True,
    )
    return 0


def _write_seal() -> int:
    payload = create_seal()
    digest = _exclusive_json(SEAL, payload)
    print(f"saved {SEAL.relative_to(ROOT)} (sha256={digest})", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("seal", "run", "calibrated"))
    args = parser.parse_args(argv)
    if args.operation == "seal":
        return _write_seal()
    if args.operation == "run":
        return run_once()
    return run_calibrated()


if __name__ == "__main__":
    raise SystemExit(main())
