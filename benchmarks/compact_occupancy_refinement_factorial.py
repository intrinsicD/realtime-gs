"""Sealed RGB-free compact occupancy-point refinement factorial, runtime-safe iter3.

This benchmark is deliberately separate from the production CLI.  The public ``seal`` and
``run`` operations implement an append-only, once-only lifecycle.  A private worker operation is
used only by the claimed parent attempt so every seed/arm receives a fresh CUDA process.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import resource
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianPointProposal,
    ObservationSamples,
)
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer
from rtgs.render.torch_points import TorchPointRasterizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
PREREGISTRATION = RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_PREREG.md"
IMPLEMENTATION_REVIEW = (
    RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_IMPLEMENTATION_REVIEW.md"
)
SEAL = RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_SEAL.json"
ATTEMPT = RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_ATTEMPT.json"
RESULT = RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json"
RUN_DIR = ROOT / "runs/compact_occupancy_refinement_factorial_iter3_20260717"

ITER1_PREREGISTRATION = RESULTS / "20260717_compact_occupancy_refinement_factorial_PREREG.md"
ITER1_IMPLEMENTATION_REVIEW = (
    RESULTS / "20260717_compact_occupancy_refinement_factorial_IMPLEMENTATION_REVIEW.md"
)
ITER1_SEAL = RESULTS / "20260717_compact_occupancy_refinement_factorial_SEAL.json"
ITER1_ATTEMPT = RESULTS / "20260717_compact_occupancy_refinement_factorial_ATTEMPT.json"
ITER1_RESULT = RESULTS / "20260717_compact_occupancy_refinement_factorial_RESULT.json"
ITER1_FAILURE_AUDIT = RESULTS / "20260717_compact_occupancy_refinement_factorial_FAILURE_AUDIT.md"
ITER1_EXECUTED_SOURCES = (
    RESULTS / "20260717_compact_occupancy_refinement_factorial_EXECUTED_SOURCES.tar"
)

ITER2_PREREGISTRATION = RESULTS / "20260717_compact_occupancy_refinement_factorial_iter2_PREREG.md"
ITER2_IMPLEMENTATION_REVIEW = (
    RESULTS / "20260717_compact_occupancy_refinement_factorial_iter2_IMPLEMENTATION_REVIEW.md"
)
ITER2_SEAL = RESULTS / "20260717_compact_occupancy_refinement_factorial_iter2_SEAL.json"
ITER2_ATTEMPT = RESULTS / "20260717_compact_occupancy_refinement_factorial_iter2_ATTEMPT.json"
ITER2_RESULT = RESULTS / "20260717_compact_occupancy_refinement_factorial_iter2_RESULT.json"
ITER2_FAILURE_AUDIT = (
    RESULTS / "20260717_compact_occupancy_refinement_factorial_iter2_FAILURE_AUDIT.md"
)
ITER2_EXECUTED_SOURCES = (
    RESULTS / "20260717_compact_occupancy_refinement_factorial_iter2_EXECUTED_SOURCES.tar"
)
ITER2_RUN_DIR = ROOT / "runs/compact_occupancy_refinement_factorial_iter2_20260717"
EXPECTED_ITER2_RUN_AGGREGATE_SHA256 = (
    "52643df0cd254f6fe48701929bcddf3fe2b23e36391e3d54f9870ac2fc6739ee"
)

PRIOR_ARTIFACT_SHA256 = {
    ITER1_PREREGISTRATION: "72553e528cbd12185b3845e63ab5367d4e78af3711acfc850383bebd7519f2bf",
    ITER1_IMPLEMENTATION_REVIEW: (
        "3aef740057e5c16bb822b400aef8acfdbd601d8ecb52843770b8950592e971f3"
    ),
    ITER1_SEAL: "8d3299b1c67f1d7aa125423846d96104556d82864115f0f6489335646f66451c",
    ITER1_ATTEMPT: "11c75fd2257041b344481a052fed96267ea78031d7f29d5b651c71bf7a6fe763",
    ITER1_RESULT: "d8030691fba7ebba3a77783473bffd538a1ca4640930e70d311cd6f6c454f520",
    ITER1_FAILURE_AUDIT: "67bf419e696273a7b47d729b7e0c07f5afb468e297568bfc694e6ddec5c0ccc7",
    ITER1_EXECUTED_SOURCES: ("a4dbc184a4288cb50253b40421d7216f8aae585c0870d87a8e1ff98e893fde49"),
    ITER2_PREREGISTRATION: "da4ef58a620c687e6eccfae959113c7e1bf7f25242f2d2f4a05b885c26047278",
    ITER2_IMPLEMENTATION_REVIEW: (
        "a7d0a9ffc136992a2fcf383d537bf0d6f31fff4be220064df188ba650d2e6c00"
    ),
    ITER2_SEAL: "c3b6c665b1255b1021fc1393dae978e36dc8f8f43ea025fdc9080d5c87cb2c01",
    ITER2_ATTEMPT: "dee6d681acf0170ed249a6c792432d9fca5e72ab072fc3e84e99e10bae4ba2f6",
    ITER2_RESULT: "fdc4b5aa5f1b7cd69e32237cfd6d49ec1c1bc624cc6ed29f0650c0c7fa162a6f",
    ITER2_FAILURE_AUDIT: "747b093f41518513f7f0881482df515b92d5028169010fae0be7b481466e29d3",
    ITER2_EXECUTED_SOURCES: ("2ffdae72f066d4936a640e66283070ee101d0a8c5bd8f59f8dfc3fee34d653ac"),
}

TEACHER_BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
PROXY_BUNDLE = ROOT / "runs/compact_occupancy_scalar_ablation_20260717/proxy_bundles/center"
INIT_PLY = ROOT / "runs/compact_occupancy_scalar_ablation_20260717/stage_b/center/gaussians.ply"
EXPECTED_INIT_PLY_SHA256 = "0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e"
EXPECTED_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")

TRAIN_SEEDS = (76801, 76802, 76803)
EVALUATION_SEEDS = (76901, 76902, 76903)
TEST_TRAIN_SEEDS = (992601, 992602, 992603)
TEST_EVALUATION_SEEDS = (992701, 992702)
CONSUMED_SEEDS = (
    76201,
    76401,
    76402,
    76403,
    76501,
    76502,
    76503,
    76601,
    76602,
    76603,
    76701,
    76702,
    76703,
    991601,
    991602,
    991603,
    991701,
    991702,
)
ARMS: dict[str, tuple[str, str]] = {
    "A": ("iid", "uniform"),
    "B": ("balanced_cycle", "uniform"),
    "C": ("iid", "proposal_attempt"),
    "D": ("balanced_cycle", "proposal_attempt"),
}
CHECKPOINTS = (0, 35, 70, 140)
BANK_ATTEMPTS = 4096
TRAIN_ATTEMPTS = 128
UNIFORM_FRACTION = 0.25
EXPLICIT_EXTENT = 1.5469313859939577
WORKER_TIMEOUT_SECONDS = 180
PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33")
EXPECTED_PRELOAD_SHA256 = "1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11"
TORCH_INSTANTIATOR_SHA256 = "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615"
TORCH_REMOTE_MODULE_SHA256 = "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
TORCH_REMOTE_MODULE_BYTES = 2355
TORCH_TEMP_SENTINEL = "<torch.distributed.nn.jit.instantiator-temp>"
REVIEW_AGGREGATE_PREFIX = "Reviewed source aggregate SHA-256: "
FOCUSED_TEST_ENV = "RTGS_FACTORIAL_FOCUSED_TEST"

PAIRED_SAMPLE_KEYS = (
    "view_index",
    "view_name",
    "sample_seed",
    "xy_sha256",
    "active_sha256",
    "inside_fit_window_sha256",
    "proposal_component_ids_sha256",
    "proposal_density_sha256",
    "joint_density_sha256",
)
TARGET_HASH_KEYS = ("target_density_sha256", "importance_sha256")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr"})

SOURCE_PATHS = (
    Path("benchmarks/compact_occupancy_refinement_factorial.py"),
    Path("tests/test_compact_occupancy_refinement_factorial.py"),
    Path("tests/test_compact_trainer.py"),
    Path("tests/test_observation2d.py"),
    Path("src/rtgs/__init__.py"),
    Path("src/rtgs/core/__init__.py"),
    Path("src/rtgs/core/gaussians2d.py"),
    Path("src/rtgs/optim/compact_trainer.py"),
    Path("src/rtgs/optim/__init__.py"),
    Path("src/rtgs/core/observation2d.py"),
    Path("src/rtgs/core/gaussians3d.py"),
    Path("src/rtgs/core/camera.py"),
    Path("src/rtgs/core/sh.py"),
    Path("src/rtgs/data/__init__.py"),
    Path("src/rtgs/data/reconstruction_inputs.py"),
    Path("src/rtgs/render/__init__.py"),
    Path("src/rtgs/render/base.py"),
    Path("src/rtgs/render/point_base.py"),
    Path("src/rtgs/render/projection.py"),
    Path("src/rtgs/render/torch_ref.py"),
    Path("src/rtgs/render/torch_points.py"),
    ITER1_PREREGISTRATION.relative_to(ROOT),
    ITER1_FAILURE_AUDIT.relative_to(ROOT),
    ITER2_PREREGISTRATION.relative_to(ROOT),
    ITER2_FAILURE_AUDIT.relative_to(ROOT),
    PREREGISTRATION.relative_to(ROOT),
    IMPLEMENTATION_REVIEW.relative_to(ROOT),
)


class ProtocolInvalid(RuntimeError):
    """A frozen protocol invariant failed."""


class BankInvariantError(ProtocolInvalid):
    """A bank invariant failed with a finite machine-readable diagnostic."""

    def __init__(self, message: str, diagnostic: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostic = dict(diagnostic)


class BindingInvariantError(ProtocolInvalid):
    """A sealed source/input/runtime/config binding changed with a structured diff."""

    def __init__(self, message: str, diagnostic: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostic = dict(diagnostic)


class WorkerProcessError(ProtocolInvalid):
    """A bounded worker failed and its child-process evidence must survive in RESULT."""

    def __init__(self, message: str, diagnostic: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostic = dict(diagnostic)


def timestamp_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _key_differences(
    expected: object,
    actual: object,
    *,
    path: str = "$",
    limit: int = 128,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def visit(left: object, right: object, current: str) -> None:
        if len(differences) >= limit:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right), key=str):
                child = f"{current}.{key}"
                if key not in left:
                    differences.append(
                        {"path": child, "expected": "<missing>", "actual": right[key]}
                    )
                elif key not in right:
                    differences.append(
                        {"path": child, "expected": left[key], "actual": "<missing>"}
                    )
                else:
                    visit(left[key], right[key], child)
                if len(differences) >= limit:
                    return
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            for index in range(max(len(left), len(right))):
                child = f"{current}[{index}]"
                if index >= len(left):
                    differences.append(
                        {"path": child, "expected": "<missing>", "actual": right[index]}
                    )
                elif index >= len(right):
                    differences.append(
                        {"path": child, "expected": left[index], "actual": "<missing>"}
                    )
                else:
                    visit(left[index], right[index], child)
                if len(differences) >= limit:
                    return
            return
        if left != right:
            differences.append({"path": current, "expected": left, "actual": right})

    visit(expected, actual, path)
    return differences


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = canonical_bytes(dict(payload))
    _exclusive_bytes(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProtocolInvalid(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ProtocolInvalid(f"{path} is not a JSON object")
    canonical_bytes(value)  # Reject NaN/Infinity recursively.
    return value


def bounded_failure(error: BaseException, *, stage: str, **bindings: Any) -> dict[str, Any]:
    message = str(error).replace("\x00", "")[:2000]
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))[-8000:]
    payload: dict[str, Any] = {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_result_v1",
        "timestamp_utc": timestamp_utc(),
        "status": "FAIL",
        "decision": "UNAVAILABLE",
        "scientific_decision": "UNAVAILABLE",
        "promotion_authorized": False,
        "failure": {
            "stage": stage,
            "type": type(error).__name__[:200],
            "message": message,
            "traceback_tail": trace,
        },
        **bindings,
    }
    if isinstance(error, BankInvariantError):
        payload["failure"]["bank_invariant"] = error.diagnostic
    if isinstance(error, BindingInvariantError):
        payload["failure"]["binding_invariant"] = error.diagnostic
    if isinstance(error, WorkerProcessError):
        payload["failure"]["worker_process"] = error.diagnostic
    canonical_bytes(payload)
    return payload


def write_terminal_result(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        canonical_bytes(dict(payload))
        final = dict(payload)
    except BaseException as error:
        safe_bindings = {
            key: value
            for key in ("preregistration_sha256", "seal_file_sha256", "attempt_sha256")
            if isinstance((value := payload.get(key)), str)
        }
        final = bounded_failure(
            error,
            stage="terminal_result_serialization",
            **safe_bindings,
        )
    return exclusive_json(RESULT, final), final


def _directory_binding(directory: Path) -> dict[str, Any]:
    if not directory.is_dir() or directory.is_symlink():
        raise ProtocolInvalid(f"bound input is not an ordinary directory: {directory}")
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ProtocolInvalid(f"bound input contains symlink: {path}")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not records:
        raise ProtocolInvalid(f"bound input directory is empty: {directory}")
    return {
        "path": directory.relative_to(ROOT).as_posix(),
        "files": records,
        "aggregate_sha256": canonical_hash(records),
    }


def input_bindings() -> dict[str, Any]:
    if sha256_file(INIT_PLY) != EXPECTED_INIT_PLY_SHA256:
        raise ProtocolInvalid("common initialization PLY hash changed")
    prior_attempts = {}
    for path, expected_sha256 in PRIOR_ARTIFACT_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ProtocolInvalid(f"prior-attempt provenance changed: {path}")
        prior_attempts[path.relative_to(ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": expected_sha256,
        }
    iter2_run = _directory_binding(ITER2_RUN_DIR)
    if iter2_run["aggregate_sha256"] != EXPECTED_ITER2_RUN_AGGREGATE_SHA256:
        raise ProtocolInvalid("consumed iter2 run-directory binding changed")
    return {
        "teacher_bundle": _directory_binding(TEACHER_BUNDLE),
        "proxy_bundle": _directory_binding(PROXY_BUNDLE),
        "initial_ply": {
            "path": INIT_PLY.relative_to(ROOT).as_posix(),
            "bytes": INIT_PLY.stat().st_size,
            "sha256": EXPECTED_INIT_PLY_SHA256,
        },
        "prior_attempt_provenance": prior_attempts,
        "iter2_run_directory": iter2_run,
    }


def source_hashes() -> tuple[dict[str, str], str]:
    origin_violations = module_origin_violations()
    if origin_violations:
        raise ProtocolInvalid(f"loaded rtgs module origin mismatch: {origin_violations}")
    unbound = unbound_loaded_local_sources()
    if unbound:
        raise ProtocolInvalid(f"loaded local source closure is not sealed: {unbound}")
    records = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise ProtocolInvalid(f"sealed source is missing: {relative}")
        records[relative.as_posix()] = sha256_file(path)
    return records, canonical_hash(records)


def loaded_local_sources() -> tuple[str, ...]:
    """Return the live repository-local Python import closure relevant to this harness."""
    source_root = (ROOT / "src/rtgs").resolve()
    harness = Path(__file__).resolve()
    paths = set()
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if not value:
            continue
        path = Path(value).resolve()
        if path.suffix in {".pyc", ".pyo"}:
            candidate = path.with_suffix(".py")
            if candidate.is_file():
                path = candidate
        if path in (harness, source_root) or source_root in path.parents:
            paths.add(path.relative_to(ROOT.resolve()).as_posix())
    return tuple(sorted(paths))


def unbound_loaded_local_sources() -> tuple[str, ...]:
    sealed = {path.as_posix() for path in SOURCE_PATHS}
    return tuple(path for path in loaded_local_sources() if path not in sealed)


def expected_rtgs_module_origins() -> dict[str, str]:
    origins = {}
    for relative in SOURCE_PATHS:
        parts = relative.parts
        if len(parts) < 3 or parts[:2] != ("src", "rtgs") or relative.suffix != ".py":
            continue
        module_parts = list(parts[1:])
        if module_parts[-1] == "__init__.py":
            module_parts.pop()
        else:
            module_parts[-1] = Path(module_parts[-1]).stem
        origins[".".join(module_parts)] = str((ROOT / relative).resolve())
    return origins


def loaded_rtgs_module_origins() -> dict[str, str | None]:
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name != "rtgs" and not name.startswith("rtgs."):
            continue
        value = getattr(module, "__file__", None)
        origins[name] = None if value is None else str(Path(value).resolve())
    return dict(sorted(origins.items()))


def module_origin_violations() -> tuple[str, ...]:
    expected = expected_rtgs_module_origins()
    actual = loaded_rtgs_module_origins()
    violations = []
    for name, origin in actual.items():
        if name not in expected:
            violations.append(f"unbound:{name}={origin}")
        elif origin != expected[name]:
            violations.append(f"shadowed:{name}={origin};expected={expected[name]}")
    expected_harness = str(
        (ROOT / "benchmarks/compact_occupancy_refinement_factorial.py").resolve()
    )
    if str(Path(__file__).resolve()) != expected_harness:
        violations.append(
            f"shadowed:harness={Path(__file__).resolve()};expected={expected_harness}"
        )
    return tuple(violations)


def reviewed_source_hashes() -> tuple[dict[str, str], str]:
    """Hash the exact reviewed snapshot without the self-referential review file."""
    review_relative = IMPLEMENTATION_REVIEW.relative_to(ROOT)
    records = {
        relative.as_posix(): sha256_file(ROOT / relative)
        for relative in SOURCE_PATHS
        if relative != review_relative
    }
    return records, canonical_hash(records)


def _torch_runtime_import_path_binding() -> dict[str, Any]:
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    importlib.import_module("torch._dynamo")
    cpu_rng_after = torch.random.get_rng_state()
    cuda_rng_after = tuple(torch.cuda.get_rng_state_all())
    if not torch.equal(cpu_rng_before, cpu_rng_after) or len(cuda_rng_before) != len(
        cuda_rng_after
    ):
        raise ProtocolInvalid("TorchDynamo runtime priming changed RNG state")
    if any(
        not torch.equal(before, after)
        for before, after in zip(cuda_rng_before, cuda_rng_after, strict=True)
    ):
        raise ProtocolInvalid("TorchDynamo runtime priming changed CUDA RNG state")

    instantiator = importlib.import_module("torch.distributed.nn.jit.instantiator")
    generated = importlib.import_module("_remote_module_non_scriptable")
    instantiator_path = Path(instantiator.__file__).resolve(strict=True)
    torch_root = Path(torch.__file__).resolve(strict=True).parent
    if torch_root not in instantiator_path.parents:
        raise ProtocolInvalid("PyTorch instantiator origin is outside the bound package")
    if sha256_file(instantiator_path) != TORCH_INSTANTIATOR_SHA256:
        raise ProtocolInvalid("PyTorch instantiator source binding changed")

    path_text = instantiator.INSTANTIATED_TEMPLATE_DIR_PATH
    if path_text != instantiator._TEMP_DIR.name or not isinstance(path_text, str):
        raise ProtocolInvalid("PyTorch instantiator temporary path identity changed")
    path = Path(path_text)
    if not path.is_absolute() or path.is_symlink():
        raise ProtocolInvalid("PyTorch instantiator path is not an absolute ordinary directory")
    path_stat = path.lstat()
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or stat.S_IMODE(path_stat.st_mode) != 0o700
        or path_stat.st_uid != os.getuid()
    ):
        raise ProtocolInvalid("PyTorch instantiator directory ownership/mode changed")
    resolved = path.resolve(strict=True)
    if resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents:
        raise ProtocolInvalid("PyTorch instantiator path entered the repository")
    if sys.path.count(path_text) != 1 or not sys.path or sys.path[-1] != path_text:
        raise ProtocolInvalid("PyTorch instantiator path is not one unique final sys.path entry")

    source = resolved / "_remote_module_non_scriptable.py"
    cache = resolved / "__pycache__"
    if set(item.name for item in resolved.iterdir()) != {source.name, cache.name}:
        raise ProtocolInvalid("PyTorch instantiator directory member allowlist changed")
    source_stat = source.lstat()
    if (
        source.is_symlink()
        or not stat.S_ISREG(source_stat.st_mode)
        or source_stat.st_size != TORCH_REMOTE_MODULE_BYTES
        or sha256_file(source) != TORCH_REMOTE_MODULE_SHA256
    ):
        raise ProtocolInvalid("PyTorch generated remote-module source changed")
    cache_stat = cache.lstat()
    cache_members = tuple(cache.iterdir()) if cache.is_dir() and not cache.is_symlink() else ()
    if (
        not stat.S_ISDIR(cache_stat.st_mode)
        or len(cache_members) != 1
        or cache_members[0].is_symlink()
        or not cache_members[0].is_file()
        or not cache_members[0].name.startswith("_remote_module_non_scriptable.")
        or cache_members[0].suffix != ".pyc"
    ):
        raise ProtocolInvalid("PyTorch generated remote-module cache changed")
    generated_path = Path(generated.__file__).resolve(strict=True)
    if generated_path != source:
        raise ProtocolInvalid("loaded PyTorch generated module origin changed")

    normalized_sys_path = list(sys.path)
    normalized_sys_path[-1] = TORCH_TEMP_SENTINEL
    return {
        "normalized_sys_path": normalized_sys_path,
        "sentinel": TORCH_TEMP_SENTINEL,
        "instantiator": {
            "module": "torch.distributed.nn.jit.instantiator",
            "path": str(instantiator_path),
            "sha256": TORCH_INSTANTIATOR_SHA256,
        },
        "generated_source": {
            "name": source.name,
            "bytes": TORCH_REMOTE_MODULE_BYTES,
            "sha256": TORCH_REMOTE_MODULE_SHA256,
        },
        "directory_policy": {
            "unique_final_sys_path_entry": True,
            "absolute_non_symlink": True,
            "mode": "0700",
            "owned_by_current_uid": True,
            "outside_repository": True,
            "python_source_allowlist": [source.name],
        },
        "rng_state_unchanged": True,
    }


def runtime_binding() -> dict[str, Any]:
    if not PRELOAD.is_file() or sha256_file(PRELOAD) != EXPECTED_PRELOAD_SHA256:
        raise ProtocolInvalid("system libstdc++ binding changed")
    effective_preload = os.environ.get("LD_PRELOAD")
    if effective_preload != str(PRELOAD):
        raise ProtocolInvalid("effective LD_PRELOAD differs from the frozen runtime")
    if not torch.cuda.is_available():
        raise ProtocolInvalid("official factorial requires CUDA")
    import_path = _torch_runtime_import_path_binding()
    origin_violations = module_origin_violations()
    if origin_violations:
        raise ProtocolInvalid(f"loaded rtgs module origin mismatch: {origin_violations}")
    unbound = unbound_loaded_local_sources()
    if unbound:
        raise ProtocolInvalid(f"loaded local source closure is not sealed: {unbound}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 3050" or capability != (8, 6):
        raise ProtocolInvalid("official CUDA device binding changed")
    try:
        gsplat_version = importlib.metadata.version("gsplat")
    except importlib.metadata.PackageNotFoundError:
        gsplat_version = None
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,pci.bus_id,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if driver.returncode != 0 or len(driver.stdout.strip().splitlines()) != 1:
        raise ProtocolInvalid("cannot bind the official NVIDIA driver/device identity")
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "gsplat": gsplat_version,
        "cuda_device": name,
        "cuda_capability": list(capability),
        "cuda_total_memory": int(properties.total_memory),
        "cuda_multiprocessor_count": int(properties.multi_processor_count),
        "cuda_uuid": str(properties.uuid),
        "cuda_pci_bus_id": int(properties.pci_bus_id),
        "nvidia_smi_driver_device": driver.stdout.strip(),
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "module_origins": expected_rtgs_module_origins(),
        "sys_path": import_path["normalized_sys_path"],
        "torch_generated_import_path": {
            key: value for key, value in import_path.items() if key != "normalized_sys_path"
        },
        "pythonpath": os.environ.get("PYTHONPATH"),
        "preload": str(PRELOAD),
        "preload_sha256": EXPECTED_PRELOAD_SHA256,
        "effective_ld_preload": effective_preload,
    }


def _frozen_config(arm: str, seed: int) -> CompactTrainConfig:
    if arm not in ARMS:
        raise ValueError("unknown frozen arm")
    schedule, target = ARMS[arm]
    return CompactTrainConfig(
        iterations=140,
        attempts_per_step=TRAIN_ATTEMPTS,
        proposal_mode="area_gaussian",
        schedule_mode=schedule,
        target_mode=target,
        uniform_fraction=UNIFORM_FRACTION,
        seed=seed,
        extent=EXPLICIT_EXTENT,
        device="cuda:0",
        lr_means=1.6e-4,
        lr_quats=1e-3,
        lr_scales=5e-3,
        lr_opacity=5e-2,
        lr_sh=2.5e-3,
        lr_sh_rest=1.25e-4,
        point_chunk=256,
        gaussian_chunk=256,
        outer_microbatch=128,
        query_component_chunk=640,
        teacher_tile_size=16,
        checkpoints=CHECKPOINTS,
        evaluate_checkpoint_risks=False,
        sh_degree=0,
        sh_color_activation="hard",
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
    )


def frozen_config(arm: str, seed: int) -> CompactTrainConfig:
    if seed not in TRAIN_SEEDS:
        raise ValueError("unknown official iter3 training seed")
    reject_official_seed_in_focused_test(seed)
    return _frozen_config(arm, seed)


def frozen_test_config(arm: str, seed: int) -> CompactTrainConfig:
    if seed not in TEST_TRAIN_SEEDS:
        raise ValueError("unknown test-only training seed")
    return _frozen_config(arm, seed)


def common_train_config_record(config: CompactTrainConfig) -> dict[str, Any]:
    common = asdict(config)
    for arm_specific in ("schedule_mode", "target_mode", "seed"):
        common.pop(arm_specific)
    return json.loads(canonical_bytes(common))


def optimizer_record(common: Mapping[str, Any]) -> dict[str, Any]:
    mean_lr_gamma = 0.01 ** (1.0 / common["iterations"])
    return {
        "groups": ["means", "quats", "scales", "opacities", "sh0", "shN"],
        "algorithm": "Adam",
        "betas": [0.9, 0.999],
        "eps": 1e-15,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "fused": False,
        "means_lr_final_factor": 0.01,
        "means_lr_gamma_per_update": mean_lr_gamma,
        "means_initial_lr_after_extent": common["lr_means"] * common["extent"],
    }


def config_record() -> dict[str, Any]:
    common = common_train_config_record(frozen_config("A", TRAIN_SEEDS[0]))
    return {
        "arms": {
            arm: {"schedule_mode": schedule, "target_mode": target}
            for arm, (schedule, target) in ARMS.items()
        },
        "train_seeds": list(TRAIN_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "rng_domain": "official_iter3",
        "focused_test_seed_sets": {
            "training": list(TEST_TRAIN_SEEDS),
            "evaluation": list(TEST_EVALUATION_SEEDS),
        },
        "seed_sets_disjoint": not bool(
            set(TRAIN_SEEDS + EVALUATION_SEEDS)
            & set(TEST_TRAIN_SEEDS + TEST_EVALUATION_SEEDS + CONSUMED_SEEDS)
        )
        and not bool(set(TEST_TRAIN_SEEDS + TEST_EVALUATION_SEEDS) & set(CONSUMED_SEEDS)),
        "consumed_seed_count": len(CONSUMED_SEEDS),
        "common_train_config": common,
        "bank_attempts": BANK_ATTEMPTS,
        "dtype": "torch.float32",
        "optimizer": optimizer_record(common),
        "continuous_uniform_endpoint_policy": "clamp_to_dtype_predecessor_of_upper",
        "point_render_background_rgb": [0.0, 0.0, 0.0],
        "evaluation_reduction_dtype": "torch.float64",
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
    }


def _expected_binding_state(sealed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_hashes": sealed["source_hashes"],
        "source_aggregate_sha256": sealed["source_aggregate_sha256"],
        "input_sections_sha256": {
            key: canonical_hash(value) for key, value in sorted(sealed["inputs"].items())
        },
        "runtime": sealed["runtime"],
        "config": sealed["config"],
    }


def _current_binding_state() -> dict[str, Any]:
    source_records, source_aggregate = source_hashes()
    inputs = input_bindings()
    return {
        "source_hashes": source_records,
        "source_aggregate_sha256": source_aggregate,
        "input_sections_sha256": {
            key: canonical_hash(value) for key, value in sorted(inputs.items())
        },
        "runtime": runtime_binding(),
        "config": config_record(),
    }


def _binding_receipt(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "semantic_sha256": canonical_hash(state),
        "source_aggregate_sha256": state["source_aggregate_sha256"],
        "input_sections_sha256": state["input_sections_sha256"],
        "runtime_sha256": canonical_hash(state["runtime"]),
        "config_sha256": canonical_hash(state["config"]),
    }


def evaluation_bank_seed(seed: int, view_id: str, kind: str) -> int:
    if seed < 0 or not view_id or kind not in {"uniform", "proposal"}:
        raise ValueError("invalid evaluation-bank seed inputs")
    reject_official_seed_in_focused_test(seed)
    payload = f"rtgs.compact-occupancy-factorial.eval.v1\0{seed}\0{view_id}\0{kind}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def reject_official_seed_in_focused_test(*seeds: int) -> None:
    if os.environ.get(FOCUSED_TEST_ENV) != "1":
        return
    official = set(TRAIN_SEEDS + EVALUATION_SEEDS)
    overlap = sorted(official.intersection(seeds))
    if overlap:
        raise ProtocolInvalid(f"focused test attempted official iter3 seed(s): {overlap}")


def log_auc(risks: Sequence[float]) -> float:
    if len(risks) != len(CHECKPOINTS):
        raise ValueError("risk curve must have exactly four frozen checkpoints")
    x = (0.0, 0.25, 0.5, 1.0)
    values = [math.log(max(float(value), 1e-12)) for value in risks]
    if any(not math.isfinite(value) or value < 0 for value in risks):
        raise ValueError("risks must be finite and non-negative")
    return sum(
        (x[index + 1] - x[index]) * (values[index + 1] + values[index]) * 0.5 for index in range(3)
    )


def geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def compute_decision(
    records: Mapping[int, Mapping[str, Mapping[str, Any]]],
    active_fractions: Sequence[float],
    *,
    seeds: Sequence[int] = TRAIN_SEEDS,
) -> dict[str, Any]:
    final_q_ratios = []
    auc_q_ratios = []
    final_u_ratios = []
    wins = 0
    if len(seeds) != 3:
        raise ValueError("decision requires exactly three paired seeds")
    reject_official_seed_in_focused_test(*seeds)
    for seed in seeds:
        by_arm = records[seed]
        b = by_arm["B"]["checkpoint_metrics"]
        d = by_arm["D"]["checkpoint_metrics"]
        b_q = [float(b[str(step)]["J_Q"]) for step in CHECKPOINTS]
        d_q = [float(d[str(step)]["J_Q"]) for step in CHECKPOINTS]
        b_final_q = b_q[-1]
        d_final_q = d_q[-1]
        b_final_u = float(b[str(CHECKPOINTS[-1])]["J_U"])
        d_final_u = float(d[str(CHECKPOINTS[-1])]["J_U"])
        final_q_ratios.append(d_final_q / b_final_q)
        auc_q_ratios.append(math.exp(log_auc(d_q) - log_auc(b_q)))
        final_u_ratios.append(d_final_u / b_final_u)
        wins += int(d_final_q < b_final_q)
    active_min = min(active_fractions)
    active_ratio = max(active_fractions) / active_min if active_min > 0.0 else 1e300
    gates = {
        "final_J_Q_geometric_ratio_le_0_95": geometric_mean(final_q_ratios) <= 0.95,
        "auc_J_Q_geometric_ratio_le_0_97": geometric_mean(auc_q_ratios) <= 0.97,
        "final_J_Q_wins_ge_2": wins >= 2,
        "final_J_U_geometric_ratio_le_1_05": geometric_mean(final_u_ratios) <= 1.05,
        "final_J_U_each_ratio_le_1_10": max(final_u_ratios) <= 1.10,
        "active_fraction_each_ge_0_95": active_min >= 0.95,
        "active_fraction_ratio_le_1_03": active_ratio <= 1.03,
    }
    passed = all(gates.values())
    return {
        "decision": ("AUTHORIZE_DENSITY_FOLLOWUP" if passed else "NO_REFINEMENT_TARGET_PROMOTION"),
        "gates": gates,
        "per_seed_D_over_B_final_J_Q": final_q_ratios,
        "per_seed_D_over_B_auc_J_Q": auc_q_ratios,
        "per_seed_D_over_B_final_J_U": final_u_ratios,
        "geometric_D_over_B_final_J_Q": geometric_mean(final_q_ratios),
        "geometric_D_over_B_auc_J_Q": geometric_mean(auc_q_ratios),
        "geometric_D_over_B_final_J_U": geometric_mean(final_u_ratios),
        "D_final_J_Q_wins": wins,
        "active_fraction_min": active_min,
        "active_fraction_max_over_min": active_ratio,
    }


def _camera_equal(left: Any, right: Any) -> bool:
    scalar = ("fx", "fy", "cx", "cy", "width", "height")
    return all(getattr(left, name) == getattr(right, name) for name in scalar) and bool(
        torch.equal(left.R, right.R) and torch.equal(left.t, right.t)
    )


def validate_proxy_alignment(
    teachers: ReconstructionInputs,
    proxies: ReconstructionInputs,
) -> list[dict[str, Any]]:
    if tuple(teachers.view_names) != EXPECTED_VIEWS or tuple(proxies.view_names) != EXPECTED_VIEWS:
        raise ProtocolInvalid("ordered compact views changed")
    if teachers.n_views != proxies.n_views:
        raise ProtocolInvalid("teacher/proxy view counts differ")
    records = []
    for index, (name, teacher, proxy, camera, proxy_camera) in enumerate(
        zip(
            teachers.view_names,
            teachers.observations,
            proxies.observations,
            teachers.cameras,
            proxies.cameras,
            strict=True,
        )
    ):
        if not _camera_equal(camera, proxy_camera):
            raise ProtocolInvalid(f"proxy camera differs in {name}")
        scalar_semantics = (
            "width",
            "height",
            "fit_window",
            "blend_mode",
            "sigma_cutoff",
            "support_fade_alpha",
            "aa_dilation",
            "epsilon",
            "n_init",
        )
        if any(getattr(teacher, key) != getattr(proxy, key) for key in scalar_semantics):
            raise ProtocolInvalid(f"proxy support/canvas semantics differ in {name}")
        if teacher.dtype != proxy.dtype:
            raise ProtocolInvalid(f"proxy dtype differs in {name}")
        if teacher.n != proxy.n or teacher.view_id != name or proxy.view_id != name:
            raise ProtocolInvalid(f"proxy cardinality/view id differs in {name}")
        for tensor_name in ("means", "log_scales", "rotations"):
            if not torch.equal(getattr(teacher, tensor_name), getattr(proxy, tensor_name)):
                raise ProtocolInvalid(f"proxy {tensor_name} is positionally misaligned in {name}")
        if (teacher.filter_variance is None) != (proxy.filter_variance is None):
            raise ProtocolInvalid(f"proxy filter presence differs in {name}")
        if teacher.filter_variance is not None and not torch.equal(
            teacher.filter_variance, proxy.filter_variance
        ):
            raise ProtocolInvalid(f"proxy filter variance differs in {name}")
        scalar = proxy.amplitudes
        if not bool(torch.isfinite(scalar).all()) or bool(((scalar < 0) | (scalar > 1)).any()):
            raise ProtocolInvalid(f"proxy occupancy scalar is outside [0,1] in {name}")
        records.append(
            {
                "view_index": index,
                "view_name": name,
                "m_opt_2d": teacher.n,
                "m_init_2d": teacher.n_init,
                "dtype": str(teacher.dtype),
                "blend_mode": teacher.blend_mode,
                "teacher_amplitudes_sha256": tensor_hash(teacher.amplitudes),
                "proxy_amplitudes_sha256": tensor_hash(proxy.amplitudes),
            }
        )
    return records


def build_product_fields(
    teachers: ReconstructionInputs,
    proxies: ReconstructionInputs,
) -> tuple[list[GaussianObservationField], list[dict[str, Any]]]:
    alignment = validate_proxy_alignment(teachers, proxies)
    fields = []
    for name, teacher, proxy, record in zip(
        teachers.view_names,
        teachers.observations,
        proxies.observations,
        alignment,
        strict=True,
    ):
        amplitudes = teacher.amplitudes * proxy.amplitudes
        field = replace(
            teacher,
            colors=torch.ones_like(teacher.colors),
            color_grads=None,
            amplitudes=amplitudes,
        )
        if not torch.equal(field.amplitudes, teacher.amplitudes * proxy.amplitudes):
            raise ProtocolInvalid(f"product amplitude construction changed in {name}")
        record["product_amplitudes_sha256"] = tensor_hash(field.amplitudes)
        record["product_nonzero"] = int((field.amplitudes > 0).sum())
        fields.append(field)
    return fields, alignment


def _save_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite NPZ: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite NPZ: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _bank_tensors(
    samples: ObservationSamples,
    color: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        "xy": samples.xy,
        "active": samples.active,
        "inside_fit_window": samples.inside_fit_window,
        "proposal_component_ids": samples.proposal_component_ids,
        "proposal_density": samples.proposal_density,
        "joint_density": samples.joint_density,
        "target_density": samples.target_density,
        "importance": samples.importance,
    }
    if color is not None:
        tensors["color"] = color
    return tensors


def _first_bad_row(mask: torch.Tensor) -> int | None:
    if mask.ndim == 0:
        return 0 if bool(mask) else None
    if mask.shape[0] == 0:
        return None
    row_mask = mask.reshape(mask.shape[0], -1).any(dim=1)
    indices = row_mask.nonzero(as_tuple=True)[0]
    return int(indices[0]) if indices.numel() else None


def _validate_bank_invariants(
    *,
    evaluation_seed: int,
    seed_domain: str,
    view_index: int,
    view_name: str,
    kind: str,
    generator_seed: int,
    product: GaussianObservationField,
    samples: ObservationSamples,
    color: torch.Tensor | None,
    require_color: bool,
) -> None:
    tensors = _bank_tensors(samples, color)
    expected_shapes = {
        "xy": (BANK_ATTEMPTS, 2),
        "active": (BANK_ATTEMPTS,),
        "inside_fit_window": (BANK_ATTEMPTS,),
        "proposal_component_ids": (BANK_ATTEMPTS,),
        "proposal_density": (BANK_ATTEMPTS,),
        "joint_density": (BANK_ATTEMPTS,),
        "target_density": (BANK_ATTEMPTS,),
        "importance": (BANK_ATTEMPTS,),
    }
    if require_color:
        expected_shapes["color"] = (BANK_ATTEMPTS, 3)

    shape_mismatches: dict[str, dict[str, list[int] | str]] = {}
    predicate_counts: dict[str, int] = {}
    failing_rows: list[int] = []
    for name, expected in expected_shapes.items():
        value = tensors.get(name)
        if not isinstance(value, torch.Tensor):
            shape_mismatches[name] = {
                "expected": list(expected),
                "actual": "missing_or_not_tensor",
            }
            failing_rows.append(0)
            continue
        if tuple(value.shape) != expected:
            shape_mismatches[name] = {
                "expected": list(expected),
                "actual": list(value.shape),
            }
            if value.ndim:
                if value.shape[0] == expected[0]:
                    failing_rows.append(0)
                else:
                    failing_rows.append(min(int(value.shape[0]), expected[0]))
    predicate_counts["shape_mismatch_count"] = len(shape_mismatches)

    for name in (
        "xy",
        "proposal_density",
        "joint_density",
        "target_density",
        "importance",
        "color",
    ):
        value = tensors.get(name)
        if not isinstance(value, torch.Tensor) or name in shape_mismatches:
            continue
        bad = ~torch.isfinite(value)
        count = int(bad.sum())
        predicate_counts[f"nonfinite_{name}_count"] = count
        first = _first_bad_row(bad)
        if first is not None:
            failing_rows.append(first)

    for name in ("proposal_density", "joint_density", "target_density", "importance"):
        value = tensors.get(name)
        if not isinstance(value, torch.Tensor) or name in shape_mismatches:
            continue
        bad = value < 0
        count = int(bad.sum())
        predicate_counts[f"negative_{name}_count"] = count
        first = _first_bad_row(bad)
        if first is not None:
            failing_rows.append(first)

    if kind == "uniform" and not any(
        name in shape_mismatches
        for name in ("xy", "active", "inside_fit_window", "proposal_component_ids")
    ):
        fit_x, fit_y, fit_width, fit_height = product.fit_window
        outside_half_open = (
            (samples.xy[:, 0] < fit_x)
            | (samples.xy[:, 0] >= fit_x + fit_width)
            | (samples.xy[:, 1] < fit_y)
            | (samples.xy[:, 1] >= fit_y + fit_height)
        )
        uniform_predicates = {
            "uniform_inactive_count": ~samples.active,
            "uniform_outside_flag_count": ~samples.inside_fit_window,
            "uniform_non_direct_count": samples.proposal_component_ids != -1,
            "uniform_coordinate_outside_half_open_count": outside_half_open,
        }
        for name, bad in uniform_predicates.items():
            predicate_counts[name] = int(bad.sum())
            first = _first_bad_row(bad)
            if first is not None:
                failing_rows.append(first)

    failed = bool(shape_mismatches) or any(predicate_counts.values())
    if not failed:
        return

    first_failing_index = min(failing_rows) if failing_rows else None
    first_failing_xy: list[float | None] | None = None
    xy = tensors.get("xy")
    if (
        first_failing_index is not None
        and isinstance(xy, torch.Tensor)
        and xy.ndim == 2
        and xy.shape[1] >= 2
        and first_failing_index < xy.shape[0]
    ):
        first_xy = xy[first_failing_index, :2].detach().cpu()
        first_failing_xy = [
            float(value) if bool(torch.isfinite(value)) else None for value in first_xy
        ]
    raise BankInvariantError(
        "evaluation bank invariant failed",
        {
            "evaluation_seed": evaluation_seed,
            "seed_domain": seed_domain,
            "view_index": view_index,
            "view_name": view_name,
            "kind": kind,
            "generator_seed": generator_seed,
            "first_failing_index": first_failing_index,
            "first_failing_xy": first_failing_xy,
            "fit_window": list(product.fit_window),
            "predicate_counts": predicate_counts,
            "shape_mismatches": shape_mismatches,
            "tensor_sha256": {
                name: tensor_hash(value)
                for name, value in sorted(tensors.items())
                if isinstance(value, torch.Tensor)
            },
        },
    )


def _sample_arrays(samples: ObservationSamples, color: torch.Tensor) -> dict[str, np.ndarray]:
    tensors = {
        name: value
        for name, value in _bank_tensors(samples, color).items()
        if name not in {"target_density", "importance"}
    }
    return {name: value.detach().cpu().contiguous().numpy() for name, value in tensors.items()}


def _generate_bank_archive(
    evaluation_seed: int,
    teachers: ReconstructionInputs,
    product_fields: Sequence[GaussianObservationField],
    path: Path,
    *,
    seed_domain: str,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    view_records = []
    for view_index, (view_name, teacher, product) in enumerate(
        zip(teachers.view_names, teachers.observations, product_fields, strict=True)
    ):
        proposal = GaussianPointProposal(product, product)
        bank_records = {}
        for kind, uniform_fraction in (("uniform", 1.0), ("proposal", UNIFORM_FRACTION)):
            seed = evaluation_bank_seed(evaluation_seed, view_name, kind)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            samples = proposal.sample(
                BANK_ATTEMPTS,
                uniform_fraction=uniform_fraction,
                generator=generator,
            )
            _validate_bank_invariants(
                evaluation_seed=evaluation_seed,
                seed_domain=seed_domain,
                view_index=view_index,
                view_name=view_name,
                kind=kind,
                generator_seed=seed,
                product=product,
                samples=samples,
                color=None,
                require_color=False,
            )
            color = teacher.query(samples.xy, component_chunk=640).color
            _validate_bank_invariants(
                evaluation_seed=evaluation_seed,
                seed_domain=seed_domain,
                view_index=view_index,
                view_name=view_name,
                kind=kind,
                generator_seed=seed,
                product=product,
                samples=samples,
                color=color,
                require_color=True,
            )
            values = _sample_arrays(samples, color)
            prefix = f"v{view_index}_{kind}"
            tensor_records = {}
            for name, value in values.items():
                key = f"{prefix}_{name}"
                arrays[key] = value
                tensor_records[name] = {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": array_hash(value),
                }
            active_count = int(values["active"].sum())
            bank_records[kind] = {
                "generator_seed": seed,
                "attempts": BANK_ATTEMPTS,
                "active_count": active_count,
                "null_count": BANK_ATTEMPTS - active_count,
                "active_fraction": active_count / BANK_ATTEMPTS,
                "tensors": tensor_records,
            }
        view_records.append(
            {
                "view_index": view_index,
                "view_name": view_name,
                "m_opt_2d": teacher.n,
                "banks": bank_records,
            }
        )
    metadata: dict[str, Any] = {
        "schema": "rtgs.compact_occupancy_factorial_banks.v2",
        "evaluation_seed": evaluation_seed,
        "seed_domain": seed_domain,
        "attempts_per_bank": BANK_ATTEMPTS,
        "views": view_records,
    }
    metadata["semantic_sha256"] = canonical_hash(metadata)
    arrays["metadata_utf8"] = np.frombuffer(canonical_bytes(metadata), dtype=np.uint8)
    file_sha = _save_npz_exclusive(path, arrays)
    return {
        "path": display_path(path),
        "sha256": file_sha,
        "bytes": path.stat().st_size,
        "metadata": metadata,
    }


def generate_bank_archive(
    evaluation_seed: int,
    teachers: ReconstructionInputs,
    product_fields: Sequence[GaussianObservationField],
    path: Path,
) -> dict[str, Any]:
    if evaluation_seed not in EVALUATION_SEEDS:
        raise ValueError("unknown official iter3 evaluation seed")
    reject_official_seed_in_focused_test(evaluation_seed)
    return _generate_bank_archive(
        evaluation_seed,
        teachers,
        product_fields,
        path,
        seed_domain="official_iter3",
    )


def generate_test_bank_archive(
    evaluation_seed: int,
    teachers: ReconstructionInputs,
    product_fields: Sequence[GaussianObservationField],
    path: Path,
) -> dict[str, Any]:
    if evaluation_seed not in TEST_EVALUATION_SEEDS:
        raise ValueError("unknown test-only evaluation seed")
    return _generate_bank_archive(
        evaluation_seed,
        teachers,
        product_fields,
        path,
        seed_domain="focused_test",
    )


def load_bank_archive(
    path: Path,
    expected_seed: int,
    *,
    expected_seed_domain: str = "official_iter3",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reject_official_seed_in_focused_test(expected_seed)
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_utf8" not in archive.files:
            raise ProtocolInvalid("bank archive lacks metadata")
        metadata = json.loads(np.asarray(archive["metadata_utf8"], dtype=np.uint8).tobytes())
        if not isinstance(metadata, dict):
            raise ProtocolInvalid("bank metadata is not an object")
        digest_payload = dict(metadata)
        stored_digest = digest_payload.pop("semantic_sha256", None)
        if stored_digest != canonical_hash(digest_payload):
            raise ProtocolInvalid("bank semantic digest mismatch")
        if (
            metadata.get("schema") != "rtgs.compact_occupancy_factorial_banks.v2"
            or metadata.get("evaluation_seed") != expected_seed
            or metadata.get("seed_domain") != expected_seed_domain
            or metadata.get("attempts_per_bank") != BANK_ATTEMPTS
        ):
            raise ProtocolInvalid("bank metadata differs from frozen contract")
        expected_keys = {"metadata_utf8"}
        loaded = []
        for view_record in metadata.get("views", []):
            view_index = int(view_record["view_index"])
            if view_index != len(loaded) or view_record["view_name"] != EXPECTED_VIEWS[view_index]:
                raise ProtocolInvalid("bank view ordering changed")
            banks = {}
            for kind in ("uniform", "proposal"):
                record = view_record["banks"][kind]
                values = {}
                for name, descriptor in record["tensors"].items():
                    key = f"v{view_index}_{kind}_{name}"
                    expected_keys.add(key)
                    value = np.asarray(archive[key]).copy()
                    actual = {
                        "dtype": value.dtype.str,
                        "shape": list(value.shape),
                        "sha256": array_hash(value),
                    }
                    if actual != descriptor:
                        raise ProtocolInvalid(f"bank tensor integrity mismatch: {key}")
                    values[name] = value
                active_count = int(values["active"].sum())
                if active_count != record["active_count"]:
                    raise ProtocolInvalid("bank active count changed")
                banks[kind] = values
            loaded.append(banks)
        if len(loaded) != len(EXPECTED_VIEWS) or set(archive.files) != expected_keys:
            raise ProtocolInvalid("bank archive has unexpected arrays or view count")
    return loaded, metadata


class RGBAccessGuard:
    """Deny image/dataset opens and image/calibrated loader imports in an official process."""

    def __init__(self) -> None:
        self.source_rgb_open_attempts = 0
        self.forbidden_import_attempts = 0
        self.negative_control_denials = 0
        self._probe = False
        self._original_open = builtins.open
        self._original_io_open = io.open
        self._original_os_open = os.open
        self._original_import = builtins.__import__
        self._original_import_module = importlib.import_module
        self.forbidden_modules_at_entry: tuple[str, ...] = ()
        self.forbidden_modules_at_exit: tuple[str, ...] = ()

    @staticmethod
    def _forbidden_module(name: str) -> bool:
        return (
            name == "PIL"
            or name.startswith("PIL.")
            or name == "cv2"
            or name.startswith("cv2.")
            or name == "imageio"
            or name.startswith("imageio.")
            or name in {"rtgs.data.calibrated", "rtgs.data.scene"}
        )

    @classmethod
    def _loaded_forbidden_modules(cls) -> tuple[str, ...]:
        return tuple(sorted(name for name in sys.modules if cls._forbidden_module(name)))

    def _forbidden_path(self, value: object) -> bool:
        if isinstance(value, int):
            return False
        try:
            path = Path(os.fspath(value))
        except TypeError:
            return False
        suffix_forbidden = path.suffix.lower() in IMAGE_SUFFIXES
        try:
            resolved = path.resolve(strict=False)
            dataset_forbidden = resolved == ROOT / "dataset" or ROOT / "dataset" in resolved.parents
        except (OSError, RuntimeError):
            dataset_forbidden = "dataset" in path.parts
        return suffix_forbidden or dataset_forbidden

    def _deny_path(self, value: object) -> None:
        if self._forbidden_path(value):
            if self._probe:
                self.negative_control_denials += 1
            else:
                self.source_rgb_open_attempts += 1
            raise PermissionError("official compact factorial denies source image/dataset access")

    def _open(self, file: object, *args: Any, **kwargs: Any) -> Any:
        self._deny_path(file)
        return self._original_open(file, *args, **kwargs)

    def _io_open(self, file: object, *args: Any, **kwargs: Any) -> Any:
        self._deny_path(file)
        return self._original_io_open(file, *args, **kwargs)

    def _os_open(self, path: object, *args: Any, **kwargs: Any) -> int:
        self._deny_path(path)
        return self._original_os_open(path, *args, **kwargs)

    def _import(
        self,
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if self._forbidden_module(name):
            if self._probe:
                self.negative_control_denials += 1
            else:
                self.forbidden_import_attempts += 1
            raise ImportError("official compact factorial denies image/calibrated loaders")
        return self._original_import(name, globals, locals, fromlist, level)

    def _import_module(self, name: str, package: str | None = None) -> Any:
        if self._forbidden_module(name):
            if self._probe:
                self.negative_control_denials += 1
            else:
                self.forbidden_import_attempts += 1
            raise ImportError("official compact factorial denies image/calibrated loaders")
        return self._original_import_module(name, package)

    def __enter__(self) -> RGBAccessGuard:
        loaded = self._loaded_forbidden_modules()
        self.forbidden_modules_at_entry = loaded
        if loaded:
            raise ProtocolInvalid(
                f"RGB-capable modules were loaded before denial boundary: {loaded}"
            )
        builtins.open = self._open
        io.open = self._io_open
        os.open = self._os_open
        builtins.__import__ = self._import
        importlib.import_module = self._import_module
        self._probe = True
        try:
            with contextlib.suppress(PermissionError):
                builtins.open(ROOT / "dataset/negative_control.png", "rb")
            with contextlib.suppress(ImportError):
                builtins.__import__("PIL.Image")
            with contextlib.suppress(ImportError):
                importlib.import_module("rtgs.data.calibrated")
        finally:
            self._probe = False
        if self.negative_control_denials != 3:
            self.__exit__()
            raise ProtocolInvalid("RGB denial negative controls did not all fire")
        return self

    def __exit__(self, *exc: object) -> None:
        builtins.open = self._original_open
        io.open = self._original_io_open
        os.open = self._original_os_open
        builtins.__import__ = self._original_import
        importlib.import_module = self._original_import_module
        loaded = self._loaded_forbidden_modules()
        self.forbidden_modules_at_exit = loaded
        if loaded:
            raise ProtocolInvalid(f"RGB-capable modules crossed denial boundary: {loaded}")

    def record(self) -> dict[str, Any]:
        return {
            "source_rgb_open_attempts": self.source_rgb_open_attempts,
            "forbidden_import_attempts": self.forbidden_import_attempts,
            "negative_control_denials": self.negative_control_denials,
            "forbidden_modules_at_entry": list(self.forbidden_modules_at_entry),
            "forbidden_modules_at_exit": list(self.forbidden_modules_at_exit),
            "passed": self.source_rgb_open_attempts == 0
            and self.forbidden_import_attempts == 0
            and self.negative_control_denials == 3
            and not self.forbidden_modules_at_entry
            and not self.forbidden_modules_at_exit,
        }


def gaussians_hash(gaussians: Gaussians3D) -> str:
    digest = hashlib.sha256()
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        digest.update(name.encode())
        digest.update(tensor_hash(getattr(gaussians, name)).encode())
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def evaluate_snapshot(
    snapshot: Gaussians3D,
    inputs: ReconstructionInputs,
    banks: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    model = snapshot.to(device)
    cameras = [camera.to(device) for camera in inputs.cameras]
    renderer = TorchPointRasterizer(point_chunk=256, gaussian_chunk=256)
    background = torch.zeros(3, device=device, dtype=model.means.dtype)
    per_view = []
    with torch.no_grad():
        for view_index, (view_name, camera, view_banks) in enumerate(
            zip(inputs.view_names, cameras, banks, strict=True)
        ):
            values = {}
            diagnostics = {}
            for kind, risk_name in (("uniform", "J_U"), ("proposal", "J_Q")):
                bank = view_banks[kind]
                xy = torch.from_numpy(bank["xy"]).to(device)
                target = torch.from_numpy(bank["color"]).to(device)
                active = torch.from_numpy(bank["active"]).to(device=device, dtype=torch.bool)
                prediction = renderer.render_points(
                    model,
                    camera,
                    xy,
                    background=background,
                    sh_degree=0,
                ).color
                if not bool(torch.isfinite(prediction).all()):
                    raise ProtocolInvalid("bank point render is non-finite")
                loss = (prediction.double() - target.double()).square().mean(dim=-1)
                weighted = loss if kind == "uniform" else loss * active.double()
                risk = float(weighted.sum(dtype=torch.float64) / BANK_ATTEMPTS)
                values[risk_name] = risk
                active_count = int(active.sum())
                diagnostics[kind] = {
                    "attempts": BANK_ATTEMPTS,
                    "active_count": active_count,
                    "null_count": BANK_ATTEMPTS - active_count,
                    "importance_ess": float(active_count if kind == "proposal" else BANK_ATTEMPTS),
                    "loss_sum": float(weighted.sum(dtype=torch.float64)),
                }
            per_view.append(
                {
                    "view_index": view_index,
                    "view_name": view_name,
                    **values,
                    "banks": diagnostics,
                }
            )
    return {
        "J_U": sum(record["J_U"] for record in per_view) / len(per_view),
        "J_Q": sum(record["J_Q"] for record in per_view) / len(per_view),
        "worst_view_J_U": max(record["J_U"] for record in per_view),
        "worst_view_J_Q": max(record["J_Q"] for record in per_view),
        "per_view": per_view,
    }


def _review_passed() -> bool:
    if not IMPLEMENTATION_REVIEW.is_file():
        return False
    lines = IMPLEMENTATION_REVIEW.read_text(encoding="utf-8").splitlines()
    _, aggregate = reviewed_source_hashes()
    expected_aggregate = f"{REVIEW_AGGREGATE_PREFIX}{aggregate}"
    return (
        sum(line.strip() == "Verdict: PASS" for line in lines) == 1
        and sum(line.strip() == expected_aggregate for line in lines) == 1
    )


def _namespace_clean_for_seal() -> bool:
    return not any(path.exists() for path in (SEAL, ATTEMPT, RESULT, RUN_DIR))


def _namespace_clean_for_run() -> bool:
    return SEAL.is_file() and not any(path.exists() for path in (ATTEMPT, RESULT, RUN_DIR))


def run_focused_verification() -> dict[str, Any]:
    commands = (
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            "tests/test_compact_occupancy_refinement_factorial.py",
            "tests/test_compact_trainer.py",
            "tests/test_observation2d.py",
        ],
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "ruff",
            "check",
            "benchmarks/compact_occupancy_refinement_factorial.py",
            "tests/test_compact_occupancy_refinement_factorial.py",
            "src/rtgs/core/observation2d.py",
            "tests/test_observation2d.py",
            "src/rtgs/optim/compact_trainer.py",
            "tests/test_compact_trainer.py",
            "src/rtgs/data/__init__.py",
            "src/rtgs/optim/__init__.py",
        ],
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "ruff",
            "format",
            "--check",
            "benchmarks/compact_occupancy_refinement_factorial.py",
            "tests/test_compact_occupancy_refinement_factorial.py",
            "src/rtgs/core/observation2d.py",
            "tests/test_observation2d.py",
            "src/rtgs/optim/compact_trainer.py",
            "tests/test_compact_trainer.py",
            "src/rtgs/data/__init__.py",
            "src/rtgs/optim/__init__.py",
        ],
    )
    records = []
    environment = dict(os.environ)
    environment[FOCUSED_TEST_ENV] = "1"
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        record = {
            "command": command,
            "returncode": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        records.append(record)
        if completed.returncode != 0:
            raise ProtocolInvalid(f"focused verification failed: {' '.join(command)}")
    return {"status": "PASS", "records": records}


def create_seal() -> dict[str, Any]:
    if not _namespace_clean_for_seal():
        raise ProtocolInvalid("official factorial namespace is not clean for sealing")
    if not _review_passed():
        raise ProtocolInvalid("independent implementation review has not passed")
    sources, source_aggregate = source_hashes()
    inputs = input_bindings()
    runtime = runtime_binding()
    config = config_record()
    verification = run_focused_verification()
    if (
        source_hashes() != (sources, source_aggregate)
        or input_bindings() != inputs
        or runtime_binding() != runtime
        or config_record() != config
    ):
        raise ProtocolInvalid("seal binding changed during focused verification")
    if not _review_passed() or not _namespace_clean_for_seal():
        raise ProtocolInvalid("review verdict or official namespace changed during verification")
    payload: dict[str, Any] = {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_seal_v1",
        "timestamp_utc": timestamp_utc(),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "implementation_review_sha256": sha256_file(IMPLEMENTATION_REVIEW),
        "source_hashes": sources,
        "source_aggregate_sha256": source_aggregate,
        "inputs": inputs,
        "runtime": runtime,
        "config": config,
        "verification": verification,
    }
    payload["seal_payload_sha256"] = canonical_hash(payload)
    return payload


def validate_unwritten_seal(payload: Mapping[str, Any]) -> None:
    digest_payload = dict(payload)
    stored_digest = digest_payload.pop("seal_payload_sha256", None)
    sources, aggregate = source_hashes()
    if stored_digest != canonical_hash(digest_payload):
        raise ProtocolInvalid("unwritten seal payload digest mismatch")
    if (
        payload.get("source_hashes") != sources
        or payload.get("source_aggregate_sha256") != aggregate
        or payload.get("inputs") != input_bindings()
        or payload.get("runtime") != runtime_binding()
        or payload.get("config") != config_record()
    ):
        raise ProtocolInvalid("unwritten seal binding changed before publication")
    if not _review_passed() or not _namespace_clean_for_seal():
        raise ProtocolInvalid("seal preconditions changed before publication")


def verify_seal() -> dict[str, Any]:
    sealed = strict_json(SEAL)
    digest_payload = dict(sealed)
    stored_digest = digest_payload.pop("seal_payload_sha256", None)
    if stored_digest != canonical_hash(digest_payload):
        raise ProtocolInvalid("seal payload digest mismatch")
    if sealed.get("artifact_type") != "compact_occupancy_refinement_factorial_iter3_seal_v1":
        raise ProtocolInvalid("wrong factorial seal type")
    if sealed.get("preregistration_sha256") != sha256_file(PREREGISTRATION):
        raise ProtocolInvalid("preregistration changed after sealing")
    if sealed.get("implementation_review_sha256") != sha256_file(IMPLEMENTATION_REVIEW):
        raise ProtocolInvalid("implementation review changed after sealing")
    sources, aggregate = source_hashes()
    if sealed.get("source_hashes") != sources or sealed.get("source_aggregate_sha256") != aggregate:
        raise ProtocolInvalid("sealed factorial source changed")
    if sealed.get("inputs") != input_bindings():
        raise ProtocolInvalid("sealed factorial input changed")
    if sealed.get("runtime") != runtime_binding():
        raise ProtocolInvalid("sealed factorial runtime changed")
    if sealed.get("config") != config_record():
        raise ProtocolInvalid("sealed factorial configuration changed")
    return sealed


def _save_snapshot_artifacts(
    arm_dir: Path,
    snapshots: Mapping[int, Gaussians3D],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for step in CHECKPOINTS:
        snapshot = snapshots[step]
        path = arm_dir / f"gaussians_step_{step:03d}.npz"
        if path.exists():
            raise FileExistsError(path)
        snapshot.save_npz(path)
        loaded = Gaussians3D.load_npz(path)
        if gaussians_hash(loaded) != gaussians_hash(snapshot):
            raise ProtocolInvalid("NPZ snapshot round trip changed semantic tensors")
        records.append(
            {
                "step": step,
                "path": display_path(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "semantic_sha256": gaussians_hash(snapshot),
                "n_gaussians": snapshot.n,
            }
        )
    final = snapshots[CHECKPOINTS[-1]]
    ply = arm_dir / "gaussians_final.ply"
    if ply.exists():
        raise FileExistsError(ply)
    final.save_ply(ply)
    replay = Gaussians3D.load_ply(ply)
    if replay.n != final.n or any(
        not bool(torch.isfinite(getattr(replay, name)).all())
        for name in ("means", "quats", "log_scales", "opacity", "sh")
    ):
        raise ProtocolInvalid("final PLY round trip is invalid")
    roundtrip = {
        name: float((getattr(replay, name) - getattr(final, name)).abs().max())
        for name in ("means", "quats", "log_scales", "opacity", "sh")
    }
    return records, {
        "path": display_path(ply),
        "sha256": sha256_file(ply),
        "bytes": ply.stat().st_size,
        "n_gaussians": final.n,
        "source_semantic_sha256": gaussians_hash(final),
        "roundtrip_max_abs": roundtrip,
    }


def run_worker(
    *,
    arm: str,
    seed: int,
    evaluation_seed: int,
    bank_path: Path,
    bank_sha256: str,
    expected_attempt_sha256: str,
    expected_seal_sha256: str,
) -> dict[str, Any]:
    if sha256_file(ATTEMPT) != expected_attempt_sha256 or sha256_file(SEAL) != expected_seal_sha256:
        raise ProtocolInvalid("worker attempt/seal token mismatch")
    sealed = verify_seal()
    expected_binding = _expected_binding_state(sealed)
    entry_binding = _current_binding_state()
    entry_differences = _key_differences(expected_binding, entry_binding)
    if entry_differences:
        raise BindingInvariantError(
            "worker entry bindings differ from the seal",
            {
                "phase": "entry",
                "expected": _binding_receipt(expected_binding),
                "actual": _binding_receipt(entry_binding),
                "differences": entry_differences,
            },
        )
    entry_binding_receipt = _binding_receipt(entry_binding)
    if strict_json(ATTEMPT).get("seal_file_sha256") != expected_seal_sha256:
        raise ProtocolInvalid("worker attempt is bound to another seal")
    if sha256_file(bank_path) != bank_sha256:
        raise ProtocolInvalid("worker bank archive changed")
    if seed not in TRAIN_SEEDS or EVALUATION_SEEDS[TRAIN_SEEDS.index(seed)] != evaluation_seed:
        raise ProtocolInvalid("worker training/evaluation seed pairing changed")
    if arm not in ARMS:
        raise ProtocolInvalid("worker arm changed")

    arm_dir = RUN_DIR / f"seed_{seed}" / f"arm_{arm}"
    arm_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    guard = RGBAccessGuard()
    with guard:
        inputs = ReconstructionInputs.load(TEACHER_BUNDLE, strict=True)
        proxies = ReconstructionInputs.load(PROXY_BUNDLE, strict=True)
        product_fields, alignment = build_product_fields(inputs, proxies)
        init = Gaussians3D.load_ply(INIT_PLY)
        if init.n != 835:
            raise ProtocolInvalid("N_init_3D changed")
        banks, bank_metadata = load_bank_archive(bank_path, evaluation_seed)
        if [record["m_opt_2d"] for record in bank_metadata["views"]] != inputs.n_opt_2d:
            raise ProtocolInvalid("bank m_opt_i_2d list differs from strict bundle")

        config = frozen_config(arm, seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        snapshots: dict[int, Gaussians3D] = {}

        def capture(snapshot: Gaussians3D, step: int) -> None:
            if step in snapshots or step not in CHECKPOINTS:
                raise ProtocolInvalid("checkpoint callback changed")
            snapshots[step] = snapshot.to("cpu")

        torch.cuda.synchronize(0)
        train_started = time.perf_counter()
        final, history = CompactTrainer(config).train(
            inputs,
            init,
            proposal_fields=product_fields,
            bundle_path=TEACHER_BUNDLE,
            checkpoint_callback=capture,
        )
        torch.cuda.synchronize(0)
        train_wall = time.perf_counter() - train_started
        captured_final_hash = gaussians_hash(snapshots[CHECKPOINTS[-1]])
        if (
            tuple(sorted(snapshots)) != CHECKPOINTS
            or gaussians_hash(final.to("cpu")) != captured_final_hash
        ):
            raise ProtocolInvalid("captured checkpoint set/final snapshot changed")
        if (
            history["checkpoint_risk_evaluation_enabled"] is not False
            or history["checkpoint_risk_evaluation_call_count"] != 0
            or history["checkpoint_callback_call_count"] != len(CHECKPOINTS)
            or history["checkpoint_snapshot_count"] != len(CHECKPOINTS)
        ):
            raise ProtocolInvalid("trainer checkpoint-evaluation route changed")
        history_checkpoints = history["checkpoints"]
        if len(history_checkpoints) != len(CHECKPOINTS):
            raise ProtocolInvalid("trainer checkpoint history count changed")
        for expected_step, history_checkpoint in zip(CHECKPOINTS, history_checkpoints, strict=True):
            if (
                history_checkpoint.get("step") != expected_step
                or history_checkpoint.get("snapshot_sha256")
                != gaussians_hash(snapshots[expected_step])
                or history_checkpoint.get("evaluation", object()) is not None
            ):
                raise ProtocolInvalid("trainer/callback checkpoint binding changed")
        if history["n_init_3d"] != 835 or history["n_opt_3d"] != 835 or final.n != 835:
            raise ProtocolInvalid("fixed topology/cardinality changed")
        unbalanced = any(value != 20 for value in history["view_visit_counts"])
        if unbalanced and ARMS[arm][0] == "balanced_cycle":
            raise ProtocolInvalid("balanced schedule no longer visits each view exactly 20 times")

        evaluation_started = time.perf_counter()
        checkpoint_metrics = {
            str(step): evaluate_snapshot(snapshots[step], inputs, banks) for step in CHECKPOINTS
        }
        torch.cuda.synchronize(0)
        evaluation_wall = time.perf_counter() - evaluation_started
        snapshot_records, final_ply = _save_snapshot_artifacts(arm_dir, snapshots)
        history_path = arm_dir / "history.json"
        history_sha = exclusive_json(history_path, history)

    denial = guard.record()
    if not denial["passed"]:
        raise ProtocolInvalid("worker attempted forbidden RGB/image/calibrated access")
    exit_binding = _current_binding_state()
    exit_differences = _key_differences(expected_binding, exit_binding)
    if exit_differences:
        raise BindingInvariantError(
            "worker bindings changed during execution",
            {
                "phase": "exit",
                "expected": _binding_receipt(expected_binding),
                "entry": entry_binding_receipt,
                "actual": _binding_receipt(exit_binding),
                "differences": exit_differences,
            },
        )
    exit_binding_receipt = _binding_receipt(exit_binding)
    return {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_worker_v1",
        "timestamp_utc": timestamp_utc(),
        "status": "PASS",
        "arm": arm,
        "schedule_mode": ARMS[arm][0],
        "target_mode": ARMS[arm][1],
        "training_seed": seed,
        "evaluation_seed": evaluation_seed,
        "n_init_3d": 835,
        "n_opt_3d": final.n,
        "n_init_2d": inputs.n_init_2d,
        "n_opt_2d": inputs.n_opt_2d,
        "sum_m_opt_2d": sum(inputs.n_opt_2d),
        "alignment": alignment,
        "bank": {
            "path": display_path(bank_path),
            "sha256": bank_sha256,
            "semantic_sha256": bank_metadata["semantic_sha256"],
        },
        "rgb_denial": denial,
        "binding_validation": {
            "entry": entry_binding_receipt,
            "exit": exit_binding_receipt,
            "differences": [],
            "passed": True,
        },
        "history": {
            "path": display_path(history_path),
            "sha256": history_sha,
            "schema": history["schema"],
            "view_schedule_sha256": history["view_schedule_sha256"],
            "teacher_digest_before": history["teacher_digest_before"],
            "teacher_digest_after": history["teacher_digest_after"],
            "proposal_digest_before": history["proposal_digest_before"],
            "proposal_digest_after": history["proposal_digest_after"],
            "preflight": history["preflight"],
            "proposal_preflight": history["proposal_preflight"],
            "proposal_normalizers": history["proposal_normalizers"],
            "parameter_motion": history["parameter_motion"],
            "optimizer_group_motion": history["optimizer_group_motion"],
            "proposal_view_diagnostics": history["proposal_view_diagnostics"],
            "gradient_maxima": {
                key: max(step["gradient_max"][key] for step in history["steps"])
                for key in history["steps"][0]["gradient_max"]
            },
            "index_diagnostics": history["index_diagnostics"],
            "proposal_index_diagnostics": history["proposal_index_diagnostics"],
        },
        "snapshots": snapshot_records,
        "final_ply": final_ply,
        "checkpoint_metrics": checkpoint_metrics,
        "timing": {
            "training_wall_seconds": train_wall,
            "evaluation_wall_seconds": evaluation_wall,
            "worker_wall_seconds": time.perf_counter() - started,
        },
        "resources": {
            "peak_rss_bytes": _peak_rss_bytes(),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        },
    }


def worker_entry(args: argparse.Namespace) -> int:
    output = Path(args.worker_output).resolve()
    try:
        payload = run_worker(
            arm=args.arm,
            seed=args.seed,
            evaluation_seed=args.evaluation_seed,
            bank_path=Path(args.bank).resolve(),
            bank_sha256=args.bank_sha256,
            expected_attempt_sha256=args.attempt_sha256,
            expected_seal_sha256=args.seal_sha256,
        )
        digest = exclusive_json(output, payload)
        print(json.dumps({"status": "PASS", "worker": str(output), "sha256": digest}))
        return 0
    except BaseException as error:
        payload = bounded_failure(
            error,
            stage="worker",
            artifact_type="compact_occupancy_refinement_factorial_iter3_worker_v1",
            arm=args.arm,
            training_seed=args.seed,
            evaluation_seed=args.evaluation_seed,
        )
        try:
            digest = exclusive_json(output, payload)
            print(json.dumps({"status": "FAIL", "worker": str(output), "sha256": digest}))
        except BaseException:
            print(json.dumps({"status": "FAIL", "error": str(error)[:1000]}))
        return 2


def _load_worker_record(path: Path) -> dict[str, Any]:
    record = strict_json(path)
    if (
        record.get("artifact_type") != "compact_occupancy_refinement_factorial_iter3_worker_v1"
        or record.get("status") != "PASS"
    ):
        raise ProtocolInvalid(f"factorial worker failed: {path}")
    binding = record.get("binding_validation")
    if (
        not isinstance(binding, dict)
        or binding.get("passed") is not True
        or binding.get("differences") != []
        or binding.get("entry") != binding.get("exit")
    ):
        entry = binding.get("entry") if isinstance(binding, dict) else None
        exit_receipt = binding.get("exit") if isinstance(binding, dict) else None
        raise BindingInvariantError(
            "worker binding-validation receipt failed",
            {
                "phase": "parent_worker_receipt_load",
                "worker": display_path(path),
                "reported_passed": binding.get("passed") if isinstance(binding, dict) else None,
                "reported_differences": (
                    binding.get("differences") if isinstance(binding, dict) else None
                ),
                "receipt_differences": _key_differences(entry, exit_receipt),
            },
        )
    history_path = ROOT / record["history"]["path"]
    if sha256_file(history_path) != record["history"]["sha256"]:
        raise ProtocolInvalid("worker history hash mismatch")
    final_ply = ROOT / record["final_ply"]["path"]
    if sha256_file(final_ply) != record["final_ply"]["sha256"]:
        raise ProtocolInvalid("worker final PLY hash mismatch")
    return record


def _load_worker_record_with_evidence(
    path: Path,
    *,
    seed: int,
    arm: str,
    evaluation_seed: int,
    subprocess_record: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return _load_worker_record(path)
    except BaseException as error:
        artifact: dict[str, Any] = {"path": display_path(path)}
        if path.is_file():
            artifact.update(
                {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
            try:
                payload = strict_json(path)
                artifact.update(
                    {
                        "status": payload.get("status"),
                        "failure": payload.get("failure"),
                        "binding_validation": payload.get("binding_validation"),
                    }
                )
            except BaseException as parse_error:
                artifact["parse_error"] = {
                    "type": type(parse_error).__name__,
                    "message": str(parse_error)[:2000],
                }
        receipt_failure: dict[str, Any] = {
            "type": type(error).__name__,
            "message": str(error)[:2000],
        }
        if isinstance(error, BindingInvariantError):
            receipt_failure["binding_invariant"] = error.diagnostic
        raise WorkerProcessError(
            f"worker receipt rejected for seed={seed} arm={arm}",
            {
                "seed": seed,
                "arm": arm,
                "evaluation_seed": evaluation_seed,
                "process": dict(subprocess_record),
                "worker_artifact": artifact,
                "receipt_failure": receipt_failure,
            },
        ) from error


def validate_paired_workers(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_history = strict_json(ROOT / left["history"]["path"])
    right_history = strict_json(ROOT / right["history"]["path"])
    if len(left_history["steps"]) != 140 or len(right_history["steps"]) != 140:
        raise ProtocolInvalid("paired histories do not have 140 steps")
    for left_step, right_step in zip(left_history["steps"], right_history["steps"], strict=True):
        for key in PAIRED_SAMPLE_KEYS:
            if left_step[key] != right_step[key]:
                raise ProtocolInvalid(f"paired sampling stream differs at {key}")
    differences = {
        key: sum(
            left_step[key] != right_step[key]
            for left_step, right_step in zip(
                left_history["steps"], right_history["steps"], strict=True
            )
        )
        for key in TARGET_HASH_KEYS
    }
    if any(value != 140 for value in differences.values()):
        raise ProtocolInvalid(
            "paired target mode did not change target/importance hashes at every step"
        )
    return {
        "left_arm": left["arm"],
        "right_arm": right["arm"],
        "steps": 140,
        "sampling_keys_exact": list(PAIRED_SAMPLE_KEYS),
        "target_hash_difference_counts": differences,
    }


def validate_step_zero(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    semantic = {record["snapshots"][0]["semantic_sha256"] for record in records.values()}
    metric_hashes = {
        canonical_hash(record["checkpoint_metrics"][str(CHECKPOINTS[0])])
        for record in records.values()
    }
    if len(semantic) != 1 or len(metric_hashes) != 1:
        raise ProtocolInvalid("step-zero snapshot or frozen-bank risks differ between arms")
    return {
        "semantic_snapshot_sha256": next(iter(semantic)),
        "checkpoint_metrics_sha256": next(iter(metric_hashes)),
        "all_four_arms_exact": True,
    }


def contrast_summary(
    records: Mapping[int, Mapping[str, Mapping[str, Any]]],
    numerator: str,
    denominator: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for risk in ("J_U", "J_Q"):
        final_ratios = []
        auc_ratios = []
        for seed in TRAIN_SEEDS:
            num = records[seed][numerator]["checkpoint_metrics"]
            den = records[seed][denominator]["checkpoint_metrics"]
            num_curve = [float(num[str(step)][risk]) for step in CHECKPOINTS]
            den_curve = [float(den[str(step)][risk]) for step in CHECKPOINTS]
            final_ratios.append(num_curve[-1] / den_curve[-1])
            auc_ratios.append(math.exp(log_auc(num_curve) - log_auc(den_curve)))
        result[risk] = {
            "per_seed_final_ratio": final_ratios,
            "geometric_final_ratio": geometric_mean(final_ratios),
            "per_seed_auc_derived_ratio": auc_ratios,
            "geometric_auc_derived_ratio": geometric_mean(auc_ratios),
        }
    return result


def _worker_command(
    *,
    arm: str,
    seed: int,
    evaluation_seed: int,
    bank_path: Path,
    bank_sha256: str,
    worker_output: Path,
    attempt_sha256: str,
    seal_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--evaluation-seed",
        str(evaluation_seed),
        "--bank",
        str(bank_path),
        "--bank-sha256",
        bank_sha256,
        "--worker-output",
        str(worker_output),
        "--attempt-sha256",
        attempt_sha256,
        "--seal-sha256",
        seal_sha256,
    ]


def execute_attempt(
    *,
    sealed: Mapping[str, Any],
    attempt_sha256: str,
    seal_sha256: str,
) -> dict[str, Any]:
    expected_parent_binding = _expected_binding_state(sealed)
    parent_entry_binding = _current_binding_state()
    parent_entry_differences = _key_differences(expected_parent_binding, parent_entry_binding)
    if parent_entry_differences:
        raise BindingInvariantError(
            "parent entry bindings differ from the seal",
            {
                "phase": "parent_entry",
                "expected": _binding_receipt(expected_parent_binding),
                "actual": _binding_receipt(parent_entry_binding),
                "differences": parent_entry_differences,
            },
        )
    parent_entry_receipt = _binding_receipt(parent_entry_binding)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    banks_dir = RUN_DIR / "banks"
    workers_dir = RUN_DIR / "workers"
    banks_dir.mkdir()
    workers_dir.mkdir()

    parent_guard = RGBAccessGuard()
    with parent_guard:
        teachers = ReconstructionInputs.load(TEACHER_BUNDLE, strict=True)
        proxies = ReconstructionInputs.load(PROXY_BUNDLE, strict=True)
        product_fields, alignment = build_product_fields(teachers, proxies)
        bank_records = []
        for evaluation_seed in EVALUATION_SEEDS:
            bank_records.append(
                generate_bank_archive(
                    evaluation_seed,
                    teachers,
                    product_fields,
                    banks_dir / f"banks_{evaluation_seed}.npz",
                )
            )
    parent_denial = parent_guard.record()
    if not parent_denial["passed"]:
        raise ProtocolInvalid("parent attempted forbidden RGB/image/calibrated access")
    bank_manifest = {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_bank_manifest_v1",
        "timestamp_utc": timestamp_utc(),
        "attempt_sha256": attempt_sha256,
        "n_init_2d": teachers.n_init_2d,
        "n_opt_2d": teachers.n_opt_2d,
        "sum_m_opt_2d": sum(teachers.n_opt_2d),
        "alignment": alignment,
        "banks": bank_records,
        "rgb_denial": parent_denial,
    }
    bank_manifest_path = RUN_DIR / "bank_manifest.json"
    bank_manifest_sha = exclusive_json(bank_manifest_path, bank_manifest)

    records: dict[int, dict[str, dict[str, Any]]] = {}
    subprocess_records = []
    for seed, evaluation_seed, bank_record in zip(
        TRAIN_SEEDS, EVALUATION_SEEDS, bank_records, strict=True
    ):
        records[seed] = {}
        bank_path = ROOT / bank_record["path"]
        for arm in ARMS:
            worker_output = workers_dir / f"seed_{seed}_arm_{arm}.json"
            command = _worker_command(
                arm=arm,
                seed=seed,
                evaluation_seed=evaluation_seed,
                bank_path=bank_path,
                bank_sha256=bank_record["sha256"],
                worker_output=worker_output,
                attempt_sha256=attempt_sha256,
                seal_sha256=seal_sha256,
            )
            environment = dict(os.environ)
            environment["LD_PRELOAD"] = str(PRELOAD)
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=WORKER_TIMEOUT_SECONDS,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise ProtocolInvalid(f"worker timed out for seed={seed} arm={arm}") from error
            subprocess_record = {
                "seed": seed,
                "arm": arm,
                "command": command,
                "returncode": completed.returncode,
                "elapsed_seconds": time.perf_counter() - started,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
            subprocess_records.append(subprocess_record)
            if completed.returncode != 0 or not worker_output.is_file():
                worker_artifact = None
                if worker_output.is_file():
                    worker_payload = strict_json(worker_output)
                    worker_artifact = {
                        "path": display_path(worker_output),
                        "sha256": sha256_file(worker_output),
                        "bytes": worker_output.stat().st_size,
                        "status": worker_payload.get("status"),
                        "failure": worker_payload.get("failure"),
                    }
                raise WorkerProcessError(
                    f"worker failed for seed={seed} arm={arm}",
                    {
                        "seed": seed,
                        "arm": arm,
                        "evaluation_seed": evaluation_seed,
                        "process": subprocess_record,
                        "worker_artifact": worker_artifact,
                    },
                )
            records[seed][arm] = _load_worker_record_with_evidence(
                worker_output,
                seed=seed,
                arm=arm,
                evaluation_seed=evaluation_seed,
                subprocess_record=subprocess_record,
            )

    paired = []
    step_zero = {}
    for seed in TRAIN_SEEDS:
        paired.append(validate_paired_workers(records[seed]["A"], records[seed]["C"]))
        paired.append(validate_paired_workers(records[seed]["B"], records[seed]["D"]))
        step_zero[str(seed)] = validate_step_zero(records[seed])

    active_fractions = [
        float(view["banks"]["proposal"]["active_fraction"])
        for bank_record in bank_records
        for view in bank_record["metadata"]["views"]
    ]
    if len(active_fractions) != len(EVALUATION_SEEDS) * len(EXPECTED_VIEWS):
        raise ProtocolInvalid("unique evaluation-bank guard population changed")
    decision = compute_decision(records, active_fractions)
    secondary = {
        "C_over_A_target_under_iid": contrast_summary(records, "C", "A"),
        "B_over_A_schedule_under_uniform": contrast_summary(records, "B", "A"),
        "D_over_C_schedule_under_proposal": contrast_summary(records, "D", "C"),
    }
    interaction = {}
    for risk in ("J_U", "J_Q"):
        values = []
        for seed in TRAIN_SEEDS:
            final = {
                arm: float(records[seed][arm]["checkpoint_metrics"]["140"][risk]) for arm in ARMS
            }
            values.append((final["D"] / final["C"]) / (final["B"] / final["A"]))
        interaction[risk] = {
            "per_seed_ratio_of_ratios": values,
            "geometric_ratio_of_ratios": geometric_mean(values),
        }
    secondary["factorial_interaction"] = interaction

    if sha256_file(SEAL) != seal_sha256 or sha256_file(ATTEMPT) != attempt_sha256:
        raise BindingInvariantError(
            "parent seal/attempt token changed during official execution",
            {
                "phase": "parent_exit",
                "seal_sha256": sha256_file(SEAL),
                "expected_seal_sha256": seal_sha256,
                "attempt_sha256": sha256_file(ATTEMPT),
                "expected_attempt_sha256": attempt_sha256,
            },
        )
    parent_exit_binding = _current_binding_state()
    parent_exit_differences = _key_differences(expected_parent_binding, parent_exit_binding)
    if parent_exit_differences:
        raise BindingInvariantError(
            "parent bindings changed during official execution",
            {
                "phase": "parent_exit",
                "expected": _binding_receipt(expected_parent_binding),
                "entry": parent_entry_receipt,
                "actual": _binding_receipt(parent_exit_binding),
                "differences": parent_exit_differences,
            },
        )
    parent_exit_receipt = _binding_receipt(parent_exit_binding)
    worker_rgb = [records[seed][arm]["rgb_denial"] for seed in TRAIN_SEEDS for arm in ARMS]
    if not all(record["passed"] for record in worker_rgb):
        raise ProtocolInvalid("a worker RGB-denial record failed")
    serializable_records = {
        str(seed): {arm: records[seed][arm] for arm in ARMS} for seed in TRAIN_SEEDS
    }
    return {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_result_v1",
        "timestamp_utc": timestamp_utc(),
        "status": "PASS",
        "decision": decision["decision"],
        "scientific_decision": decision["decision"],
        "promotion_authorized": decision["decision"] == "AUTHORIZE_DENSITY_FOLLOWUP",
        "claim_scope": (
            "single-scene fixed-topology compact-teacher schedule/target factorial; "
            "authorizes at most a later variable-N_opt density experiment"
        ),
        "preregistration_sha256": sealed["preregistration_sha256"],
        "implementation_review_sha256": sealed["implementation_review_sha256"],
        "seal_file_sha256": seal_sha256,
        "attempt_sha256": attempt_sha256,
        "config": sealed["config"],
        "inputs": sealed["inputs"],
        "runtime": sealed["runtime"],
        "bank_manifest": {
            "path": display_path(bank_manifest_path),
            "sha256": bank_manifest_sha,
        },
        "rgb_denial": {
            "parent": parent_denial,
            "workers": worker_rgb,
            "source_rgb_open_attempts": parent_denial["source_rgb_open_attempts"]
            + sum(record["source_rgb_open_attempts"] for record in worker_rgb),
        },
        "binding_validation": {
            "parent_entry": parent_entry_receipt,
            "parent_exit": parent_exit_receipt,
            "differences": [],
            "passed": True,
        },
        "paired_sampling_validation": paired,
        "step_zero_validation": step_zero,
        "decision_metrics": decision,
        "secondary_diagnostics": secondary,
        "workers": serializable_records,
        "worker_processes": subprocess_records,
        "complexity_accounting": {
            "m_init_i_2d": teachers.n_init_2d,
            "m_opt_i_2d": teachers.n_opt_2d,
            "sum_m_opt_i_2d": sum(teachers.n_opt_2d),
            "N_init_3d": 835,
            "N_opt_3d_by_worker": [
                records[seed][arm]["n_opt_3d"] for seed in TRAIN_SEEDS for arm in ARMS
            ],
            "bank_attempts_per_view_measure": BANK_ATTEMPTS,
            "training_attempts_per_step": TRAIN_ATTEMPTS,
            "teacher_tile_overlap_preflight": records[TRAIN_SEEDS[0]]["A"]["history"]["preflight"][
                "views"
            ],
            "proposal_tile_overlap_preflight": records[TRAIN_SEEDS[0]]["A"]["history"][
                "proposal_preflight"
            ]["views"],
            "proposal_normalizers": records[TRAIN_SEEDS[0]]["A"]["history"]["proposal_normalizers"],
            "scaling_claim_authorized": False,
        },
        "visualization": {
            "status": "DEFERRED_POST_RESULT",
            "required_seed": TRAIN_SEEDS[0],
            "required_arms": list(ARMS),
            "required_backend": "gsplat",
            "required_native_resolution": [5328, 4608],
        },
    }


def seal_entry() -> int:
    payload = create_seal()
    validate_unwritten_seal(payload)
    digest = exclusive_json(SEAL, payload)
    print(json.dumps({"status": "PASS", "seal": str(SEAL), "sha256": digest}, indent=2))
    return 0


def run_entry() -> int:
    if not _namespace_clean_for_run():
        raise ProtocolInvalid("official factorial namespace is not clean for one-shot run")
    sealed = verify_seal()
    seal_sha = sha256_file(SEAL)
    attempt_payload = {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_attempt_v1",
        "timestamp_utc": timestamp_utc(),
        "seal_file_sha256": seal_sha,
        "preregistration_sha256": sealed["preregistration_sha256"],
        "config_sha256": canonical_hash(sealed["config"]),
        "once_only": True,
    }
    if not _namespace_clean_for_run():
        raise ProtocolInvalid("official factorial namespace changed before attempt publication")
    attempt_sha = exclusive_json(ATTEMPT, attempt_payload)
    try:
        payload = execute_attempt(
            sealed=sealed,
            attempt_sha256=attempt_sha,
            seal_sha256=seal_sha,
        )
        digest, written = write_terminal_result(payload)
        status = written.get("status")
        print(json.dumps({"status": status, "result": str(RESULT), "sha256": digest}, indent=2))
        return 0 if status == "PASS" else 2
    except BaseException as error:
        payload = bounded_failure(
            error,
            stage="official_attempt",
            preregistration_sha256=sealed["preregistration_sha256"],
            seal_file_sha256=seal_sha,
            attempt_sha256=attempt_sha,
        )
        digest, written = write_terminal_result(payload)
        print(
            json.dumps(
                {"status": written.get("status"), "result": str(RESULT), "sha256": digest},
                indent=2,
            )
        )
        return 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("seal", "run", "worker"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--bank")
    parser.add_argument("--bank-sha256")
    parser.add_argument("--worker-output")
    parser.add_argument("--attempt-sha256")
    parser.add_argument("--seal-sha256")
    args = parser.parse_args(argv)
    if args.operation == "worker":
        required = (
            "arm",
            "seed",
            "evaluation_seed",
            "bank",
            "bank_sha256",
            "worker_output",
            "attempt_sha256",
            "seal_sha256",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"worker is missing required arguments: {', '.join(missing)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operation == "seal":
        return seal_entry()
    if args.operation == "run":
        return run_entry()
    return worker_entry(args)


if __name__ == "__main__":
    raise SystemExit(main())
