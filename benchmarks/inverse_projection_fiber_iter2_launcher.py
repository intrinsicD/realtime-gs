#!/usr/bin/env python3
"""Hash-pinned, parent-owned controller for the one-shot Iteration 2 execution.

An outer bootstrap must hold the workspace, verify these reviewed bytes, inject
``_RTGS_BOOTSTRAP``, and only then compile/execute them.  The scientific worker never owns
the public result or lifecycle transitions.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import re
import signal
import stat
import struct
import subprocess
import sys
import time
import types
import uuid
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

NAMESPACE = "rtgs.inverse-projection-fiber.iter2.v1"
PREREG_SHA256 = "95adcf0f9d03761ca57bb36444a051f5c581e21e06eb399a355437bda9f6d28e"
ITER1_RESULT_SHA256 = "2601a45d19d1d8a636d3c0db5ef8b14adf5f4137baaf718c86e1f80a84cecf9e"
LAUNCHER_RELATIVE = "benchmarks/inverse_projection_fiber_iter2_launcher.py"
SEALED_HELPER_RELATIVE = "benchmarks/inverse_projection_fiber_sealed_archive.py"
BASE_TRANSACTION_RELATIVE = "benchmarks/inverse_projection_fiber_transaction.py"
ITER2_TRANSACTION_RELATIVE = "benchmarks/inverse_projection_fiber_iter2_transaction.py"
WORKER_RELATIVE = "benchmarks/inverse_projection_fiber_iter2.py"
PREREG_RELATIVE = "benchmarks/results/20260717_inverse_projection_fiber_iter2_PREREG.md"
PREREG_REVIEW_RELATIVE = (
    "benchmarks/results/20260717_inverse_projection_fiber_iter2_PREREG_REVIEW.md"
)
IMPLEMENTATION_REVIEW_RELATIVE = (
    "benchmarks/results/20260717_inverse_projection_fiber_iter2_IMPLEMENTATION_REVIEW.json"
)
ITER1_RESULT_RELATIVE = "benchmarks/results/20260717_inverse_projection_fiber_iter1e_RESULT.json"
RESULTS_RELATIVE = "benchmarks/results"
RUNS_RELATIVE = "runs"
RESULT_NAME = "20260717_inverse_projection_fiber_iter2_RESULT.json"
ARTIFACTS_NAME = "inverse_projection_fiber_iter2_official_20260717"
EXECUTED_SOURCES_NAME = "EXECUTED_SOURCES.zip"
ATTEMPT_NAME = "ATTEMPT.json"
WORKER_HANDOFF_NAME = "WORKER_HANDOFF.json"
TERMINAL_NAME = "TERMINAL.json"
LIFECYCLE_NAME = "LIFECYCLE.json"
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_REVIEW_BYTES = 8 * 1024 * 1024
MAX_INPUT_EVIDENCE_BYTES = 512 * 1024 * 1024
MAX_FRESHNESS_FILES = 100_000
MAX_FRESHNESS_DEPTH = 32
WORKER_QUIESCENCE_SECONDS = 5.0
SCIENTIFIC_STATUSES = frozenset({"PASS", "FAIL", "INCONCLUSIVE", "INVALID"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
_DTYPE_RE = re.compile(r"([<>=|])([?bBiufc])([0-9]+)\Z")
_BOOTSTRAP_KEYS = {
    "workspace_fd",
    "workspace_path_hint",
    "launcher_sha256",
    "implementation_review_sha256",
}


class BootstrapError(RuntimeError):
    """The required outer bootstrap context is absent or malformed."""


class ValidationError(RuntimeError):
    """A held input, worker handoff, or prepared result failed closed validation."""


@dataclass(frozen=True, slots=True)
class BootstrapContext:
    workspace_fd: int
    workspace_path_hint: str
    launcher_sha256: str
    implementation_review_sha256: str


@dataclass(frozen=True, slots=True)
class StableCapture:
    relative_path: str
    path_hint: str
    payload: bytes
    sha256: str
    size: int
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "path": self.path_hint,
            "bytes": self.size,
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True, slots=True)
class ReviewedExecution:
    helper: types.ModuleType
    archive: Any
    implementation_review: dict[str, Any]
    implementation_review_capture: StableCapture
    preregistration_capture: StableCapture
    preregistration_review_capture: StableCapture
    prior_result_capture: StableCapture
    source_bytes: dict[str, bytes]
    source_captures: dict[str, StableCapture]
    launcher_capture: StableCapture


@dataclass(frozen=True, slots=True)
class HeldDirectories:
    workspace_fd: int
    results_fd: int
    runs_fd: int
    workspace_path: Path
    results_path: Path
    runs_path: Path
    artifacts_fd: int | None = None
    artifacts_path: Path | None = None

    def close(self) -> None:
        if self.artifacts_fd is not None:
            os.close(self.artifacts_fd)
        os.close(self.runs_fd)
        os.close(self.results_fd)


@dataclass(frozen=True, slots=True)
class WorkerLaunch:
    transaction_id: str
    workspace_fd: int
    artifacts_fd: int
    output_parent_fd: int
    archive_fd: int
    archive_sha256: str
    workspace_path: Path
    attempt_public: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    returncode: int
    stdout: bytes
    stderr: bytes
    process_group_quiescent: bool
    stragglers_killed: bool

    def descriptor(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout_bytes": len(self.stdout),
            "stdout_sha256": _sha256(self.stdout),
            "stderr_bytes": len(self.stderr),
            "stderr_sha256": _sha256(self.stderr),
            "process_group_quiescent": self.process_group_quiescent,
            "stragglers_killed": self.stragglers_killed,
        }


@dataclass(slots=True)
class RootProgress:
    index: int
    bundle: tuple[int, int, int]
    target_name: str
    public: dict[str, Any] | None = None
    status: str | None = None


@dataclass(slots=True)
class TransactionProgress:
    transaction_id: str
    reservation_public: dict[str, Any] | None
    attempt_public: dict[str, Any] | None
    roots: list[RootProgress]
    worker: WorkerOutcome | None = None
    terminal_public: dict[str, Any] | None = None
    lifecycle_published: bool = False


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _nonce(transaction_id: str, label: str) -> str:
    if type(transaction_id) is not str or _NONCE_RE.fullmatch(transaction_id) is None:
        raise ValueError("transaction_id must be 128 lowercase hexadecimal bits")
    if type(label) is not str or not label or "\x00" in label:
        raise ValueError("nonce label must be a non-empty string")
    return hashlib.sha256(f"{transaction_id}:{label}".encode()).hexdigest()[:32]


def _require_bootstrap(namespace: Mapping[str, Any]) -> BootstrapContext:
    raw = namespace.get("_RTGS_BOOTSTRAP")
    if type(raw) is not dict or set(raw) != _BOOTSTRAP_KEYS:
        raise BootstrapError("launcher requires an exact injected _RTGS_BOOTSTRAP")
    workspace_fd = raw["workspace_fd"]
    hint = raw["workspace_path_hint"]
    if type(workspace_fd) is not int or workspace_fd < 0:
        raise BootstrapError("workspace_fd must be a non-negative exact integer")
    try:
        metadata = os.fstat(workspace_fd)
    except OSError as error:
        raise BootstrapError("workspace_fd is not open") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise BootstrapError("workspace_fd does not identify a directory")
    if type(hint) is not str or not hint or "\x00" in hint or not Path(hint).is_absolute():
        raise BootstrapError("workspace_path_hint must be one absolute path hint")
    for key in ("launcher_sha256", "implementation_review_sha256"):
        if not _is_sha256(raw[key]):
            raise BootstrapError(f"{key} must be a canonical SHA-256")
    return BootstrapContext(
        workspace_fd, hint, raw["launcher_sha256"], raw["implementation_review_sha256"]
    )


BOOTSTRAP = _require_bootstrap(globals())


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    if (
        type(relative_path) is not str
        or not relative_path
        or "\x00" in relative_path
        or "\\" in relative_path
        or relative_path.startswith("/")
    ):
        raise ValueError("relative path must be one canonical POSIX path")
    parts = tuple(relative_path.split("/"))
    if any(part in {"", ".", ".."} for part in parts) or "/".join(parts) != relative_path:
        raise ValueError("relative path contains an unsafe component")
    return parts


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_directory_at(root_fd: int, parts: tuple[str, ...]) -> int:
    descriptor = os.dup(root_fd)
    os.set_inheritable(descriptor, False)
    try:
        for part in parts:
            replacement = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = replacement
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise NotADirectoryError("/".join(parts))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_relative(root_fd: int, relative_path: str) -> int:
    return _open_directory_at(root_fd, _relative_parts(relative_path))


def _capture_from_directory(
    directory_fd: int, name: str, *, relative_path: str, path_hint: str, max_bytes: int
) -> StableCapture:
    if len(_relative_parts(name)) != 1:
        raise ValueError("capture name must be one component")
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValidationError(f"not a regular file: {relative_path}")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise ValidationError(f"capture exceeds size bound: {relative_path}")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
            if not block:
                raise ValidationError(f"short read: {relative_path}")
            chunks.append(block)
            offset += len(block)
        if os.pread(descriptor, 1, before.st_size):
            raise ValidationError(f"capture grew: {relative_path}")
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise ValidationError(f"capture metadata changed: {relative_path}")
        if (named.st_dev, named.st_ino, named.st_mode) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
        ):
            raise ValidationError(f"capture identity changed: {relative_path}")
        return StableCapture(
            relative_path,
            path_hint,
            payload,
            _sha256(payload),
            len(payload),
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    finally:
        os.close(descriptor)


def _capture_relative(
    workspace_fd: int,
    relative_path: str,
    *,
    workspace_path_hint: str,
    max_bytes: int,
    expected_sha256: str | None = None,
) -> StableCapture:
    parts = _relative_parts(relative_path)
    parent_fd = _open_directory_at(workspace_fd, parts[:-1])
    try:
        capture = _capture_from_directory(
            parent_fd,
            parts[-1],
            relative_path=relative_path,
            path_hint=str(Path(workspace_path_hint) / relative_path),
            max_bytes=max_bytes,
        )
    finally:
        os.close(parent_fd)
    if expected_sha256 is not None and capture.sha256 != expected_sha256:
        raise ValidationError(f"reviewed hash mismatch: {relative_path}")
    return capture


def _strict_json_object(capture: StableCapture, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(capture.payload)
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{label} is not finite JSON") from error
    if type(value) is not dict:
        raise ValidationError(f"{label} must contain one JSON object")
    return value


def _load_reviewed_helper(capture: StableCapture) -> types.ModuleType:
    name = f"_rtgs_reviewed_sealed_archive_{capture.sha256[:16]}"
    if name in sys.modules:
        raise ValidationError("reviewed helper alias was already loaded")
    module = types.ModuleType(name)
    module.__file__ = f"<reviewed:{capture.relative_path}:{capture.sha256}>"
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(capture.payload, module.__file__, "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    required = {
        "create_sealed_source_archive",
        "verify_sealed_source_archive",
        "require_project_modules_unloaded",
        "verify_loaded_project_modules",
    }
    if any(not callable(getattr(module, item, None)) for item in required):
        sys.modules.pop(name, None)
        raise ValidationError("reviewed helper is missing its required API")
    return module


def _review_source_hashes(review: Mapping[str, Any]) -> dict[str, str]:
    raw = review.get("source_hashes")
    if type(raw) is not dict or not raw:
        raise ValidationError("implementation review has no exact source_hashes map")
    result: dict[str, str] = {}
    for path, digest in raw.items():
        _relative_parts(path)
        if not path.endswith(".py") or not _is_sha256(digest):
            raise ValidationError(f"invalid reviewed source: {path!r}")
        result[path] = digest
    required = {
        LAUNCHER_RELATIVE,
        SEALED_HELPER_RELATIVE,
        BASE_TRANSACTION_RELATIVE,
        ITER2_TRANSACTION_RELATIVE,
        WORKER_RELATIVE,
    }
    if required - set(result):
        raise ValidationError(
            f"implementation review omits controller closure: {sorted(required - set(result))}"
        )
    return result


def _capture_reviewed_execution(bootstrap: BootstrapContext = BOOTSTRAP) -> ReviewedExecution:
    def capture(path: str, bound: int, expected: str | None = None) -> StableCapture:
        return _capture_relative(
            bootstrap.workspace_fd,
            path,
            workspace_path_hint=bootstrap.workspace_path_hint,
            max_bytes=bound,
            expected_sha256=expected,
        )

    launcher = capture(LAUNCHER_RELATIVE, MAX_SOURCE_BYTES, bootstrap.launcher_sha256)
    preregistration = capture(PREREG_RELATIVE, MAX_REVIEW_BYTES, PREREG_SHA256)
    preregistration_review = capture(PREREG_REVIEW_RELATIVE, MAX_REVIEW_BYTES)
    preregistration_review_text = preregistration_review.payload.decode("utf-8", "strict")
    if (
        "Verdict: **PASS**" not in preregistration_review_text
        or PREREG_SHA256 not in preregistration_review_text
    ):
        raise ValidationError("preregistration review is not a matching PASS review")
    prior = capture(ITER1_RESULT_RELATIVE, MAX_SOURCE_BYTES, ITER1_RESULT_SHA256)
    review_capture = capture(
        IMPLEMENTATION_REVIEW_RELATIVE, MAX_REVIEW_BYTES, bootstrap.implementation_review_sha256
    )
    review = _strict_json_object(review_capture, label="implementation review")
    if review.get("verdict") != "PASS":
        raise ValidationError("implementation review verdict is not PASS")
    if review.get("preregistration_sha256") != PREREG_SHA256:
        raise ValidationError("review is not bound to the preregistration")
    if review.get("preregistration_review_sha256") != preregistration_review.sha256:
        raise ValidationError("review is not bound to the preregistration review")
    if review.get("iter1_result_sha256") != ITER1_RESULT_SHA256:
        raise ValidationError("review is not bound to the prior result")
    source_hashes = _review_source_hashes(review)
    if source_hashes[LAUNCHER_RELATIVE] != bootstrap.launcher_sha256:
        raise ValidationError("outer launcher pin differs from review")
    source_captures: dict[str, StableCapture] = {}
    source_bytes: dict[str, bytes] = {}
    for path, expected in sorted(source_hashes.items()):
        item = launcher if path == LAUNCHER_RELATIVE else capture(path, MAX_SOURCE_BYTES, expected)
        if item.sha256 != expected:
            raise ValidationError(f"reviewed source differs: {path}")
        source_captures[path] = item
        source_bytes[path] = item.payload
    helper = _load_reviewed_helper(source_captures[SEALED_HELPER_RELATIVE])
    archive = helper.create_sealed_source_archive(
        source_bytes, source_hashes, name="rtgs-ipf-iter2-reviewed"
    )
    helper.verify_sealed_source_archive(archive)
    return ReviewedExecution(
        helper,
        archive,
        review,
        review_capture,
        preregistration,
        preregistration_review,
        prior,
        source_bytes,
        source_captures,
        launcher,
    )


def _open_control_directories(bootstrap: BootstrapContext = BOOTSTRAP) -> HeldDirectories:
    results_fd = _open_directory_relative(bootstrap.workspace_fd, RESULTS_RELATIVE)
    try:
        runs_fd = _open_directory_relative(bootstrap.workspace_fd, RUNS_RELATIVE)
    except BaseException:
        os.close(results_fd)
        raise
    workspace = Path(bootstrap.workspace_path_hint)
    return HeldDirectories(
        bootstrap.workspace_fd,
        results_fd,
        runs_fd,
        workspace,
        workspace / RESULTS_RELATIVE,
        workspace / RUNS_RELATIVE,
    )


def _entry_absent(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    return False


def _scan_run_names(
    directory_fd: int,
    tokens: tuple[str, ...],
    *,
    prefix: str = "",
    depth: int = 0,
    budget: list[int],
) -> list[str]:
    if depth > MAX_FRESHNESS_DEPTH:
        raise ValidationError("freshness tree depth exceeds bound")
    matches: list[str] = []
    for name in sorted(os.listdir(directory_fd)):
        budget[0] -= 1
        if budget[0] < 0:
            raise ValidationError("freshness entry count exceeds bound")
        relative = f"{prefix}/{name}" if prefix else name
        if any(token in relative for token in tokens):
            matches.append(relative)
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(name, _directory_flags(), dir_fd=directory_fd)
            try:
                matches.extend(
                    _scan_run_names(child, tokens, prefix=relative, depth=depth + 1, budget=budget)
                )
            finally:
                os.close(child)
    return matches


def _freshness(directories: HeldDirectories, roots: tuple[int, ...]) -> dict[str, Any]:
    if (
        type(roots) is not tuple
        or not roots
        or any(type(root) is not int for root in roots)
        or len(set(roots)) != len(roots)
    ):
        raise ValueError("roots must be one unique exact-integer tuple")
    tokens = tuple(str(root) for root in roots)
    run_matches = _scan_run_names(directories.runs_fd, tokens, budget=[MAX_FRESHNESS_FILES])
    allowed = {
        Path(PREREG_RELATIVE).name,
        Path(PREREG_REVIEW_RELATIVE).name,
        Path(IMPLEMENTATION_REVIEW_RELATIVE).name,
        "20260717_inverse_projection_fiber_iter2_PREREG_REVIEW_INITIAL_FAIL.md",
    }
    result_matches: list[str] = []
    for name in sorted(os.listdir(directories.results_fd)):
        if name in allowed:
            continue
        metadata = os.stat(name, dir_fd=directories.results_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_REVIEW_BYTES:
            continue
        item = _capture_from_directory(
            directories.results_fd,
            name,
            relative_path=f"{RESULTS_RELATIVE}/{name}",
            path_hint=str(directories.results_path / name),
            max_bytes=MAX_REVIEW_BYTES,
        )
        if NAMESPACE.encode() in item.payload or any(
            token.encode() in item.payload for token in tokens
        ):
            result_matches.append(name)
    checks = {
        "official_result_absent": _entry_absent(directories.results_fd, RESULT_NAME),
        "official_artifacts_absent": _entry_absent(directories.runs_fd, ARTIFACTS_NAME),
        "no_official_root_named_run_paths": not run_matches,
        "no_unexpected_prior_result_mentions": not result_matches,
    }
    return {
        "checks": checks,
        "run_path_matches": run_matches,
        "unexpected_result_mentions": result_matches,
        "pass": all(checks.values()),
    }


def _import_parent_transactions(reviewed: ReviewedExecution) -> tuple[Any, Any, dict[str, Any]]:
    reviewed.helper.verify_sealed_source_archive(reviewed.archive)
    reviewed.helper.require_project_modules_unloaded()
    archive = reviewed.archive.proc_path
    sys.path.insert(0, archive)
    sys.path.insert(0, f"{archive}/src")
    base = importlib.import_module("benchmarks.inverse_projection_fiber_transaction")
    i2tx = importlib.import_module("benchmarks.inverse_projection_fiber_iter2_transaction")
    origins = reviewed.helper.verify_loaded_project_modules(reviewed.archive)
    return base, i2tx, asdict(origins)


def _write_preflight_file(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-length preflight write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _preflight_rename_exchange(directory_fd: int, base: Any, *, label: str) -> None:
    exchange = getattr(base, "_rename_exchange", None)
    if not callable(exchange):
        raise RuntimeError("reviewed transaction engine has no rename-exchange primitive")
    if not label or not label.isascii() or not label.isalnum():
        raise ValueError("preflight label must be non-empty ASCII alphanumeric text")
    probe_name = f".rtgs-iter2-preflight-{label}-{uuid.uuid4().hex}"
    os.mkdir(probe_name, 0o700, dir_fd=directory_fd)
    os.fsync(directory_fd)
    try:
        probe_fd = os.open(probe_name, _directory_flags(), dir_fd=directory_fd)
    except BaseException:
        os.rmdir(probe_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        raise
    probe_metadata = os.fstat(probe_fd)
    try:
        os.fchmod(probe_fd, 0o700)
        left_payload = f"{label}:left:{probe_name}".encode()
        right_payload = f"{label}:right:{probe_name}".encode()
        _write_preflight_file(probe_fd, "left", left_payload)
        _write_preflight_file(probe_fd, "right", right_payload)
        os.fsync(probe_fd)
        left_before = _capture_from_directory(
            probe_fd,
            "left",
            relative_path=f"{probe_name}/left",
            path_hint=f"<preflight:{label}:left>",
            max_bytes=256,
        )
        right_before = _capture_from_directory(
            probe_fd,
            "right",
            relative_path=f"{probe_name}/right",
            path_hint=f"<preflight:{label}:right>",
            max_bytes=256,
        )
        exchange(probe_fd, "left", "right")
        os.fsync(probe_fd)
        left_after = _capture_from_directory(
            probe_fd,
            "left",
            relative_path=f"{probe_name}/left",
            path_hint=f"<preflight:{label}:left>",
            max_bytes=256,
        )
        right_after = _capture_from_directory(
            probe_fd,
            "right",
            relative_path=f"{probe_name}/right",
            path_hint=f"<preflight:{label}:right>",
            max_bytes=256,
        )
        if (
            left_after.payload != right_payload
            or right_after.payload != left_payload
            or (left_after.device, left_after.inode) != (right_before.device, right_before.inode)
            or (right_after.device, right_after.inode) != (left_before.device, left_before.inode)
        ):
            raise RuntimeError(f"RENAME_EXCHANGE probe did not swap exact entries on {label}")
    finally:
        try:
            for name in ("left", "right"):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(name, dir_fd=probe_fd)
            os.fsync(probe_fd)
            named = os.stat(probe_name, dir_fd=directory_fd, follow_symlinks=False)
            if (named.st_dev, named.st_ino, named.st_mode) != (
                probe_metadata.st_dev,
                probe_metadata.st_ino,
                probe_metadata.st_mode,
            ):
                raise RuntimeError(f"preflight probe directory identity changed on {label}")
        finally:
            os.close(probe_fd)
        os.rmdir(probe_name, dir_fd=directory_fd)
        os.fsync(directory_fd)


def _preflight_pdeathsig() -> None:
    source = (
        "import ctypes,signal,sys\n"
        "libc=ctypes.CDLL(None,use_errno=True)\n"
        "prctl=getattr(libc,'prctl',None)\n"
        "sys.exit(0 if prctl is not None and prctl(1,signal.SIGKILL,0,0,0)==0 else 1)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", source],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=10.0,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("PR_SET_PDEATHSIG preflight child timed out") from error
    if completed.returncode != 0:
        raise RuntimeError(
            "PR_SET_PDEATHSIG preflight child failed: "
            f"returncode={completed.returncode}, stderr_sha256={_sha256(completed.stderr)}"
        )


def _preflight(reviewed: ReviewedExecution, directories: HeldDirectories, base: Any) -> None:
    if not sys.platform.startswith("linux") or not os.path.isdir("/proc/self/fd"):
        raise RuntimeError("official execution requires Linux /proc descriptor paths")
    reviewed.helper.verify_sealed_source_archive(reviewed.archive)
    base.require_rename_exchange()
    _preflight_rename_exchange(directories.results_fd, base, label="results")
    _preflight_rename_exchange(directories.runs_fd, base, label="runs")
    _preflight_pdeathsig()
    for descriptor in (directories.workspace_fd, directories.results_fd, directories.runs_fd):
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RuntimeError("directory authority changed")
        os.fsync(descriptor)


def _create_artifacts_directory(directories: HeldDirectories) -> HeldDirectories:
    os.mkdir(ARTIFACTS_NAME, 0o700, dir_fd=directories.runs_fd)
    os.fsync(directories.runs_fd)
    descriptor = os.open(ARTIFACTS_NAME, _directory_flags(), dir_fd=directories.runs_fd)
    os.fchmod(descriptor, 0o700)
    os.fsync(descriptor)
    return replace(
        directories, artifacts_fd=descriptor, artifacts_path=directories.runs_path / ARTIFACTS_NAME
    )


def _write_blob_exclusive(
    directory_fd: int, directory_path: Path, name: str, payload: bytes
) -> dict[str, Any]:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("zero-length write")
            view = view[count:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (named.st_dev, named.st_ino, named.st_mode, named.st_size) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
    ):
        raise ValidationError("exclusive blob identity changed")
    os.fsync(directory_fd)
    return {
        "path": str(directory_path / name),
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
    }


def _archive_descriptor(reviewed: ReviewedExecution) -> dict[str, Any]:
    observed = reviewed.helper.verify_sealed_source_archive(reviewed.archive)
    return {
        "name": reviewed.archive.name,
        "bytes": observed.size,
        "sha256": observed.sha256,
        "device": observed.device,
        "inode": observed.inode,
        "mode": observed.mode,
        "link_count": observed.link_count,
        "seals": observed.seals,
        "members": [
            {
                "path": member.path,
                "module_name": member.module_name,
                "bytes": member.size,
                "sha256": member.sha256,
            }
            for member in reviewed.archive.image.members
        ],
    }


def _persist_executed_sources(
    reviewed: ReviewedExecution, directories: HeldDirectories
) -> dict[str, Any]:
    if directories.artifacts_fd is None or directories.artifacts_path is None:
        raise RuntimeError("artifact directory authority is absent")
    reviewed.helper.verify_sealed_source_archive(reviewed.archive)
    descriptor = _write_blob_exclusive(
        directories.artifacts_fd,
        directories.artifacts_path,
        EXECUTED_SOURCES_NAME,
        reviewed.archive.image.payload,
    )
    if descriptor["sha256"] != reviewed.archive.image.sha256:
        raise ValidationError("persisted sources differ from sealed image")
    descriptor.update(
        format="deterministic_stored_zip", members=len(reviewed.archive.image.members)
    )
    return descriptor


def _root_bundles(roots: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    if (
        type(roots) is not tuple
        or not roots
        or len(roots) % 3
        or any(type(root) is not int for root in roots)
        or len(set(roots)) != len(roots)
    ):
        raise ValueError("roots must be unique exact integers grouped in triples")
    return tuple(tuple(roots[index : index + 3]) for index in range(0, len(roots), 3))


def _publication_public(publication: Mapping[str, Any]) -> dict[str, Any]:
    public = publication.get("public")
    if type(public) is not dict:
        raise ValidationError("publication omitted public descriptor")
    return public


def _make_root_receipt(
    domain: Any, root: RootProgress, *, transaction_id: str, status: str, phase: str
) -> dict[str, Any]:
    return domain.make_receipt(
        "root_state",
        {
            "transaction_id": transaction_id,
            "root_index": root.index,
            "root_bundle": list(root.bundle),
        },
        root_consumption_status=status,
        roots=root.bundle,
        official_phase=phase,
    )


def _publish_root_state(
    i2tx: Any,
    domain: Any,
    directories: HeldDirectories,
    root: RootProgress,
    *,
    transaction_id: str,
    status: str,
    phase: str,
) -> None:
    if directories.artifacts_fd is None or directories.artifacts_path is None:
        raise RuntimeError("artifact directory authority is absent")
    receipt = _make_root_receipt(
        domain, root, transaction_id=transaction_id, status=status, phase=phase
    )
    arguments = (directories.artifacts_path, root.target_name, "root_state", receipt)
    if root.public is None:
        publication = i2tx.publish_receipt(
            *arguments,
            nonce=_nonce(transaction_id, f"root-{root.index}-{phase.lower()}"),
            directory_fd=directories.artifacts_fd,
        )
    else:
        publication = i2tx.exchange_receipt(
            *arguments,
            expected_public=root.public,
            nonce=_nonce(transaction_id, f"root-{root.index}-{phase.lower()}"),
            directory_fd=directories.artifacts_fd,
        )
    root.public = _publication_public(publication)
    root.status = status


def _worker_environment(launch: WorkerLaunch) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "RTGS_ITER2_WORKSPACE_ROOT": str(launch.workspace_path),
            "RTGS_ITER2_WORKSPACE_FD": str(launch.workspace_fd),
            "RTGS_ITER2_ARTIFACTS_FD": str(launch.artifacts_fd),
            "RTGS_ITER2_OUTPUT_PARENT_FD": str(launch.output_parent_fd),
            "RTGS_ITER2_ARCHIVE_FD": str(launch.archive_fd),
            "RTGS_ITER2_ARCHIVE_SHA256": launch.archive_sha256,
            "RTGS_ITER2_OFFICIAL_ATTEMPT_PUBLIC_JSON": _canonical_json(launch.attempt_public),
            "RTGS_ITER2_EXPECTED_PARENT_PID": str(os.getpid()),
        }
    )
    return environment


def _worker_bootstrap_source() -> str:
    names = (
        "RTGS_ITER2_WORKSPACE_FD",
        "RTGS_ITER2_ARTIFACTS_FD",
        "RTGS_ITER2_OUTPUT_PARENT_FD",
        "RTGS_ITER2_ARCHIVE_FD",
    )
    return (
        "import ctypes,os,runpy,signal,sys\n"
        "parent=int(os.environ.pop('RTGS_ITER2_EXPECTED_PARENT_PID'))\n"
        "libc=ctypes.CDLL(None,use_errno=True)\n"
        "prctl=getattr(libc,'prctl',None)\n"
        "if prctl is None or prctl(1,signal.SIGKILL,0,0,0)!=0:\n"
        " raise RuntimeError('PDEATHSIG failed')\n"
        "if os.getppid()!=parent: os.kill(os.getpid(),signal.SIGKILL)\n"
        f"names={names!r}\n"
        "for name in names: os.set_inheritable(int(os.environ[name]),False)\n"
        "fd=int(os.environ['RTGS_ITER2_ARCHIVE_FD']); archive=f'/proc/self/fd/{fd}'\n"
        "sys.path.insert(0,archive); sys.path.insert(0,archive+'/src')\n"
        "runpy.run_module('benchmarks.inverse_projection_fiber_iter2',run_name='__main__')\n"
    )


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _quiesce_process_group(process_group: int) -> tuple[bool, bool]:
    if not _process_group_exists(process_group):
        return True, False
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return True, False
    deadline = time.monotonic() + WORKER_QUIESCENCE_SECONDS
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group):
            return True, True
        time.sleep(0.01)
    return False, True


def _run_worker(launch: WorkerLaunch) -> WorkerOutcome:
    pass_fds = (
        launch.workspace_fd,
        launch.artifacts_fd,
        launch.output_parent_fd,
        launch.archive_fd,
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", _worker_bootstrap_source()],
        cwd=f"/proc/self/fd/{launch.workspace_fd}",
        env=_worker_environment(launch),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=pass_fds,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    quiescent, killed = _quiesce_process_group(process.pid)
    return WorkerOutcome(process.returncode, stdout, stderr, quiescent, killed)


def _dtype_itemsize(dtype: str) -> int:
    match = _DTYPE_RE.fullmatch(dtype) if type(dtype) is str else None
    if match is None:
        raise ValidationError(f"unsupported or unsafe NPY dtype: {dtype!r}")
    itemsize = int(match.group(3))
    allowed = {
        "?": {1},
        "b": {1},
        "B": {1},
        "i": {1, 2, 4, 8},
        "u": {1, 2, 4, 8},
        "f": {2, 4, 8, 16},
        "c": {8, 16, 32},
    }
    if itemsize not in allowed[match.group(2)]:
        raise ValidationError(f"unsupported dtype itemsize: {dtype!r}")
    return itemsize


def _semantic_dtype_name(dtype: str) -> str:
    """Match NumPy's ``str(array.dtype)`` used by the worker semantic hash."""
    match = _DTYPE_RE.fullmatch(dtype) if type(dtype) is str else None
    if match is None:
        raise ValidationError(f"unsupported or unsafe NPY dtype: {dtype!r}")
    byte_order, kind = match.group(1), match.group(2)
    itemsize = _dtype_itemsize(dtype)
    native = (
        byte_order in {"=", "|"}
        or itemsize == 1
        or (byte_order == "<" and sys.byteorder == "little")
        or (byte_order == ">" and sys.byteorder == "big")
    )
    if not native:
        return dtype
    if kind in {"?", "b"}:
        return "bool"
    if kind == "B":
        return "uint8"
    prefixes = {"i": "int", "u": "uint", "f": "float", "c": "complex"}
    return f"{prefixes[kind]}{itemsize * 8}"


def _parse_npy_member(payload: bytes, *, member_name: str) -> tuple[dict[str, Any], bytes]:
    if not payload.startswith(b"\x93NUMPY") or len(payload) < 10:
        raise ValidationError(f"not an NPY member: {member_name}")
    version = (payload[6], payload[7])
    if version == (1, 0):
        header_size, start, encoding = struct.unpack("<H", payload[8:10])[0], 10, "latin1"
    elif version in {(2, 0), (3, 0)}:
        if len(payload) < 12:
            raise ValidationError(f"truncated NPY header: {member_name}")
        header_size, start = struct.unpack("<I", payload[8:12])[0], 12
        encoding = "utf-8" if version[0] == 3 else "latin1"
    else:
        raise ValidationError(f"unsupported NPY version: {member_name}")
    if header_size <= 0 or header_size > 100_000 or start + header_size > len(payload):
        raise ValidationError(f"invalid NPY header bound: {member_name}")
    try:
        header = ast.literal_eval(payload[start : start + header_size].decode(encoding).strip())
    except Exception as error:
        raise ValidationError(f"invalid NPY header: {member_name}") from error
    if type(header) is not dict or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValidationError(f"noncanonical NPY header: {member_name}")
    dtype, shape = header["descr"], header["shape"]
    if header["fortran_order"] is not False:
        raise ValidationError(f"Fortran-order member: {member_name}")
    if type(shape) is not tuple or any(type(size) is not int or size < 0 for size in shape):
        raise ValidationError(f"invalid NPY shape: {member_name}")
    data = payload[start + header_size :]
    if len(data) != math.prod(shape) * _dtype_itemsize(dtype):
        raise ValidationError(f"NPY byte count differs: {member_name}")
    return {"dtype": dtype, "shape": list(shape)}, data


def _inspect_npz(payload: bytes) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        archive_context = zipfile.ZipFile(io.BytesIO(payload), mode="r")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValidationError("evidence is not a valid NPZ") from error
    members: dict[str, dict[str, Any]] = {}
    semantic: dict[str, tuple[str, list[int], bytes]] = {}
    with archive_context as archive:
        infos = archive.infolist()
        if not infos or len(infos) > 64 or len({item.filename for item in infos}) != len(infos):
            raise ValidationError("NPZ member count or uniqueness differs")
        total_uncompressed = 0
        for info in infos:
            if (
                info.is_dir()
                or info.flag_bits & 1
                or info.compress_type != zipfile.ZIP_STORED
                or "/" in info.filename
                or "\\" in info.filename
                or not info.filename.endswith(".npy")
            ):
                raise ValidationError(f"unsafe NPZ member: {info.filename!r}")
            if info.file_size < 0 or info.file_size > MAX_INPUT_EVIDENCE_BYTES:
                raise ValidationError(f"NPZ member exceeds uncompressed bound: {info.filename!r}")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_INPUT_EVIDENCE_BYTES:
                raise ValidationError("NPZ total uncompressed bytes exceed bound")
            name = info.filename[:-4]
            if not name or "\x00" in name:
                raise ValidationError("invalid NPZ array name")
            spec, data = _parse_npy_member(archive.read(info), member_name=info.filename)
            members[name] = spec
            semantic[name] = (_semantic_dtype_name(spec["dtype"]), spec["shape"], data)
        if archive.testzip() is not None:
            raise ValidationError("NPZ CRC failed")
    digest = hashlib.sha256()
    for name in sorted(semantic):
        dtype, shape, data = semantic[name]
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(dtype.encode())
        digest.update(b"\0")
        digest.update(_canonical_json(shape).encode())
        digest.update(b"\0")
        digest.update(data)
    return dict(sorted(members.items())), digest.hexdigest()


def _validated_scientific_status(
    handoff: Mapping[str, Any], prepared_receipt: Mapping[str, Any]
) -> str:
    scientific_status = handoff.get("scientific_status")
    scientific_gates = prepared_receipt.get("scientific_gates")
    if (
        type(scientific_status) is not str
        or scientific_status not in SCIENTIFIC_STATUSES
        or prepared_receipt.get("status") != scientific_status
        or type(scientific_gates) is not dict
        or scientific_gates.get("overall_status") != scientific_status
    ):
        raise ValidationError("handoff scientific status differs from prepared result")
    return scientific_status


def _validate_input_evidence(
    directories: HeldDirectories, *, receipt: Mapping[str, Any], root: RootProgress
) -> dict[str, Any]:
    if directories.artifacts_fd is None or directories.artifacts_path is None:
        raise RuntimeError("artifact directory authority is absent")
    relative = f"scene_{root.bundle[0]}/input_evidence.npz"
    if receipt.get("input_evidence_relative_path") != relative:
        raise ValidationError("evidence relative path differs")
    descriptor = receipt.get("input_evidence")
    required = {"path", "bytes", "sha256", "device", "inode", "semantic_sha256", "members"}
    if type(descriptor) is not dict or set(descriptor) != required:
        raise ValidationError("evidence descriptor fields differ")
    if not _is_sha256(descriptor["sha256"]) or not _is_sha256(descriptor["semantic_sha256"]):
        raise ValidationError("evidence hashes are malformed")
    if (
        type(descriptor["bytes"]) is not int
        or not 0 <= descriptor["bytes"] <= MAX_INPUT_EVIDENCE_BYTES
        or type(descriptor["device"]) is not int
        or type(descriptor["inode"]) is not int
    ):
        raise ValidationError("evidence identity fields are malformed")
    raw_path = descriptor["path"]
    if (
        type(raw_path) is not str
        or not raw_path.startswith("/proc/self/fd/")
        or not raw_path.endswith("/input_evidence.npz")
    ):
        raise ValidationError("evidence path is not a worker-held descriptor hint")
    root_fd = os.open(
        f"scene_{root.bundle[0]}", _directory_flags(), dir_fd=directories.artifacts_fd
    )
    try:
        capture = _capture_from_directory(
            root_fd,
            "input_evidence.npz",
            relative_path=relative,
            path_hint=str(directories.artifacts_path / relative),
            max_bytes=MAX_INPUT_EVIDENCE_BYTES,
        )
    finally:
        os.close(root_fd)
    for key, observed in (
        ("bytes", capture.size),
        ("sha256", capture.sha256),
        ("device", capture.device),
        ("inode", capture.inode),
    ):
        if descriptor[key] != observed:
            raise ValidationError(f"evidence {key} differs")
    members, semantic_sha256 = _inspect_npz(capture.payload)
    if descriptor["members"] != members or descriptor["semantic_sha256"] != semantic_sha256:
        raise ValidationError("evidence semantic structure differs")
    return {"capture": capture.descriptor(), "semantic_sha256": semantic_sha256, "members": members}


def _capture_worker_handoff(
    i2tx: Any,
    directories: HeldDirectories,
    *,
    transaction_id: str,
    roots: tuple[int, ...],
    root_progress: list[RootProgress],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if directories.artifacts_fd is None or directories.artifacts_path is None:
        raise RuntimeError("artifact directory authority is absent")
    handoff, handoff_public = i2tx.capture_receipt(
        directories.artifacts_path,
        WORKER_HANDOFF_NAME,
        "worker_handoff",
        directory_fd=directories.artifacts_fd,
    )
    if (
        handoff.get("transaction_id") != transaction_id
        or handoff.get("root_consumption_status") != i2tx.PARTIALLY_CONSUMED
        or handoff.get("roots") != list(roots)
        or handoff.get("official_phase") != "WORKER_HANDOFF"
    ):
        raise ValidationError("worker handoff differs from launcher intent")
    publics = handoff.get("input_receipts")
    names = [f"INPUT_ROOT_{root.index}.json" for root in root_progress]
    if (
        type(publics) is not list
        or len(publics) != len(root_progress)
        or handoff.get("input_receipt_names") != names
    ):
        raise ValidationError("input receipt ordering differs")
    validated: list[dict[str, Any]] = []
    for root, name, expected_public in zip(root_progress, names, publics, strict=True):
        if type(expected_public) is not dict:
            raise ValidationError("malformed input public descriptor")
        receipt, public = i2tx.capture_receipt(
            directories.artifacts_path,
            name,
            "input",
            expected_public=expected_public,
            directory_fd=directories.artifacts_fd,
        )
        input_payload = receipt.get("input")
        if (
            receipt.get("transaction_id") != transaction_id
            or receipt.get("root_index") != root.index
            or receipt.get("root_bundle") != list(root.bundle)
            or receipt.get("root_consumption_status") != i2tx.PARTIALLY_CONSUMED
            or receipt.get("roots") != list(root.bundle)
            or receipt.get("official_phase") != "INPUT_MATERIALIZED"
            or type(input_payload) is not dict
            or input_payload.get("scene_root") != root.bundle[0]
            or input_payload.get("depth_root") != root.bundle[1]
            or input_payload.get("observation_order_root") != root.bundle[2]
        ):
            raise ValidationError(f"input receipt differs for root {root.index}")
        evidence = _validate_input_evidence(directories, receipt=receipt, root=root)
        validated.append({"target_name": name, "public": public, "evidence": evidence})
    prepared_descriptor = handoff.get("prepared_result")
    if type(prepared_descriptor) is not dict:
        raise ValidationError("handoff omitted prepared result")
    prepared_receipt, _prepared, prepared_public = i2tx.capture_prepared(
        directories.results_path,
        prepared_descriptor,
        expected_target_name=RESULT_NAME,
        expected_receipt_kind="result",
        expected_transaction_id=transaction_id,
        expected_root_consumption_status=i2tx.CONSUMED,
        expected_roots=roots,
        expected_official_phase="FINAL",
        expected_commit_state=None,
        directory_fd=directories.results_fd,
    )
    _validated_scientific_status(handoff, prepared_receipt)
    return (
        handoff,
        handoff_public,
        validated,
        {"descriptor": prepared_descriptor, "receipt": prepared_receipt, "public": prepared_public},
    )


def _publish_terminal(
    i2tx: Any,
    domain: Any,
    directories: HeldDirectories,
    progress: TransactionProgress,
    *,
    roots: tuple[int, ...],
    transaction_status: str,
    scientific_status: str,
    status: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if directories.artifacts_fd is None or directories.artifacts_path is None:
        raise RuntimeError("artifact directory authority is absent")
    receipt = domain.make_receipt(
        "terminal",
        {
            "transaction_id": progress.transaction_id,
            "transaction_status": transaction_status,
            "scientific_status": scientific_status,
            **dict(payload),
        },
        root_consumption_status=status,
        roots=roots,
        official_phase="TERMINAL",
    )
    publication = i2tx.publish_receipt(
        directories.artifacts_path,
        TERMINAL_NAME,
        "terminal",
        receipt,
        nonce=_nonce(progress.transaction_id, "terminal"),
        directory_fd=directories.artifacts_fd,
    )
    progress.terminal_public = _publication_public(publication)
    return progress.terminal_public


def _publish_lifecycle(
    i2tx: Any,
    domain: Any,
    directories: HeldDirectories,
    progress: TransactionProgress,
    *,
    roots: tuple[int, ...],
    commit_state: str,
    status: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if directories.artifacts_fd is None or directories.artifacts_path is None:
        raise RuntimeError("artifact directory authority is absent")
    receipt = domain.make_receipt(
        "lifecycle",
        {
            "transaction_id": progress.transaction_id,
            "terminal": progress.terminal_public,
            **dict(payload),
        },
        root_consumption_status=status,
        roots=roots,
        official_phase="LIFECYCLE",
        commit_state=commit_state,
    )
    publication = i2tx.publish_receipt(
        directories.artifacts_path,
        LIFECYCLE_NAME,
        "lifecycle",
        receipt,
        nonce=_nonce(progress.transaction_id, "lifecycle"),
        directory_fd=directories.artifacts_fd,
    )
    progress.lifecycle_published = True
    return _publication_public(publication)


def _record_abort(
    i2tx: Any,
    domain: Any,
    directories: HeldDirectories,
    progress: TransactionProgress,
    *,
    roots: tuple[int, ...],
    error: BaseException,
) -> None:
    abort_errors: list[str] = []
    for root in progress.roots:
        if root.status == i2tx.CONSUMED:
            continue
        try:
            _publish_root_state(
                i2tx,
                domain,
                directories,
                root,
                transaction_id=progress.transaction_id,
                status=i2tx.ABORTED_UNKNOWN,
                phase="ABORTED_UNKNOWN",
            )
        except BaseException as abort_error:
            abort_errors.append(f"root_{root.index}:{type(abort_error).__name__}:{abort_error}")
    failure = {
        "error_class": type(error).__name__,
        "error_message": str(error),
        "abort_errors": abort_errors,
        "worker": None if progress.worker is None else progress.worker.descriptor(),
        "root_states": [
            {
                "root_index": root.index,
                "root_bundle": list(root.bundle),
                "status": root.status,
                "public": root.public,
            }
            for root in progress.roots
        ],
        "attempt": progress.attempt_public,
        "reservation": progress.reservation_public,
    }
    try:
        _publish_terminal(
            i2tx,
            domain,
            directories,
            progress,
            roots=roots,
            transaction_status="ABORTED",
            scientific_status="UNKNOWN",
            status=i2tx.ABORTED_UNKNOWN,
            payload=failure,
        )
    except BaseException as terminal_error:
        abort_errors.append(f"terminal:{type(terminal_error).__name__}:{terminal_error}")
    with contextlib.suppress(BaseException):
        _publish_lifecycle(
            i2tx,
            domain,
            directories,
            progress,
            roots=roots,
            commit_state="ABORTED",
            status=i2tx.ABORTED_UNKNOWN,
            payload={"failure": failure, "terminal_publication_errors": list(abort_errors)},
        )


def _execute_transaction(
    *,
    reviewed: ReviewedExecution,
    directories: HeldDirectories,
    executed_sources: dict[str, Any],
    freshness: dict[str, Any],
    base_transaction: Any,
    i2tx: Any,
    module_origins: dict[str, Any],
    roots: tuple[int, ...],
    worker_runner: Callable[[WorkerLaunch], WorkerOutcome] = _run_worker,
    transaction_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> tuple[int, WorkerOutcome | None]:
    del base_transaction
    if directories.artifacts_fd is None or directories.artifacts_path is None:
        raise RuntimeError("artifact directory authority is absent")
    if not freshness.get("pass"):
        raise ValidationError("freshness must pass before reservation")
    bundles = _root_bundles(roots)
    transaction_id = transaction_id_factory()
    if type(transaction_id) is not str or _NONCE_RE.fullmatch(transaction_id) is None:
        raise ValueError("malformed transaction ID")
    domain = i2tx.official_domain()
    progress = TransactionProgress(
        transaction_id,
        None,
        None,
        [
            RootProgress(index, bundle, f"ROOT_STATE_{index}.json")
            for index, bundle in enumerate(bundles)
        ],
    )
    reserved = False
    try:
        reservation = domain.make_receipt(
            "result_reservation",
            {
                "transaction_id": transaction_id,
                "planned_attempt": str(directories.artifacts_path / ATTEMPT_NAME),
                "planned_artifacts": str(directories.artifacts_path),
            },
            root_consumption_status=i2tx.RESERVED_UNCONSUMED,
            roots=roots,
            official_phase="RESULT_RESERVED",
        )
        publication = i2tx.publish_receipt(
            directories.results_path,
            RESULT_NAME,
            "result_reservation",
            reservation,
            nonce=_nonce(transaction_id, "result-reservation"),
            directory_fd=directories.results_fd,
        )
        progress.reservation_public = _publication_public(publication)
        reserved = True

        member_hashes = {member.path: member.sha256 for member in reviewed.archive.image.members}
        attempt_payload = {
            "transaction_id": transaction_id,
            "workspace_root": str(directories.workspace_path),
            "result_path": str(directories.results_path / RESULT_NAME),
            "artifacts_path": str(directories.artifacts_path),
            "reservation_payload": reservation,
            "reservation_public": progress.reservation_public,
            "preregistration_sha256": reviewed.preregistration_capture.sha256,
            "preregistration_review_sha256": reviewed.preregistration_review_capture.sha256,
            "iter1_result_sha256": reviewed.prior_result_capture.sha256,
            "implementation_review_payload": reviewed.implementation_review,
            "implementation_review_sha256": reviewed.implementation_review_capture.sha256,
            "implementation_review_capture": reviewed.implementation_review_capture.descriptor(),
            "source_hashes": dict(sorted(reviewed.implementation_review["source_hashes"].items())),
            "archive_member_hashes": dict(sorted(member_hashes.items())),
            "source_archive": _archive_descriptor(reviewed),
            "executed_sources": executed_sources,
            "preimport_captures": {
                "launcher": reviewed.launcher_capture.descriptor(),
                "preregistration": reviewed.preregistration_capture.descriptor(),
                "preregistration_review": reviewed.preregistration_review_capture.descriptor(),
                "prior_result": reviewed.prior_result_capture.descriptor(),
                "sources": {
                    path: capture.descriptor()
                    for path, capture in sorted(reviewed.source_captures.items())
                },
            },
            "parent_module_origins": module_origins,
            "freshness": freshness,
        }
        attempt = domain.make_receipt(
            "attempt",
            attempt_payload,
            root_consumption_status=i2tx.RESERVED_UNCONSUMED,
            roots=roots,
            official_phase="ATTEMPT_RESERVED",
        )
        publication = i2tx.publish_receipt(
            directories.artifacts_path,
            ATTEMPT_NAME,
            "attempt",
            attempt,
            nonce=_nonce(transaction_id, "attempt"),
            directory_fd=directories.artifacts_fd,
        )
        progress.attempt_public = _publication_public(publication)

        for root in progress.roots:
            _publish_root_state(
                i2tx,
                domain,
                directories,
                root,
                transaction_id=transaction_id,
                status=i2tx.RESERVED_UNCONSUMED,
                phase="UNSTARTED",
            )
        for root in progress.roots:
            _publish_root_state(
                i2tx,
                domain,
                directories,
                root,
                transaction_id=transaction_id,
                status=i2tx.CONSUMPTION_STARTED,
                phase="STARTED",
            )
        _payload, public = i2tx.capture_receipt(
            directories.results_path,
            RESULT_NAME,
            "result_reservation",
            expected_public=progress.reservation_public,
            directory_fd=directories.results_fd,
        )
        progress.reservation_public = public
        launch = WorkerLaunch(
            transaction_id,
            directories.workspace_fd,
            directories.artifacts_fd,
            directories.results_fd,
            reviewed.archive.fileno(),
            reviewed.archive.image.sha256,
            directories.workspace_path,
            progress.attempt_public,
        )
        progress.worker = worker_runner(launch)
        if (
            progress.worker.returncode != 0
            or not progress.worker.process_group_quiescent
            or progress.worker.stragglers_killed
        ):
            raise RuntimeError(f"worker failed or left descendants: {progress.worker.descriptor()}")

        handoff, handoff_public, validated_inputs, prepared = _capture_worker_handoff(
            i2tx,
            directories,
            transaction_id=transaction_id,
            roots=roots,
            root_progress=progress.roots,
        )
        for root in progress.roots:
            _publish_root_state(
                i2tx,
                domain,
                directories,
                root,
                transaction_id=transaction_id,
                status=i2tx.CONSUMED,
                phase="CONSUMED",
            )
        result_exchange = i2tx.exchange_prepared_receipt(
            directories.results_path,
            prepared["descriptor"],
            expected_public=progress.reservation_public,
            expected_target_name=RESULT_NAME,
            expected_receipt_kind="result",
            expected_transaction_id=transaction_id,
            expected_root_consumption_status=i2tx.CONSUMED,
            expected_roots=roots,
            expected_official_phase="FINAL",
            expected_commit_state=None,
            directory_fd=directories.results_fd,
        )
        result_public = _publication_public(result_exchange)
        terminal_public = _publish_terminal(
            i2tx,
            domain,
            directories,
            progress,
            roots=roots,
            transaction_status="COMMITTED",
            scientific_status=handoff["scientific_status"],
            status=i2tx.CONSUMED,
            payload={
                "worker": progress.worker.descriptor(),
                "worker_handoff": handoff_public,
                "validated_inputs": validated_inputs,
                "prepared_result": prepared["public"],
                "result": result_public,
                "root_states": [root.public for root in progress.roots],
            },
        )
        _publish_lifecycle(
            i2tx,
            domain,
            directories,
            progress,
            roots=roots,
            commit_state="COMMITTED",
            status=i2tx.CONSUMED,
            payload={
                "result": result_public,
                "attempt": progress.attempt_public,
                "worker_handoff": handoff_public,
                "terminal_receipt": terminal_public,
                "root_states": [root.public for root in progress.roots],
            },
        )
        return 0, progress.worker
    except BaseException as error:
        if reserved:
            _record_abort(i2tx, domain, directories, progress, roots=roots, error=error)
            return 1, progress.worker
        raise


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("official Iteration 2 launcher accepts no arguments")
    reviewed = _capture_reviewed_execution()
    directories = _open_control_directories()
    outcome: WorkerOutcome | None = None
    try:
        base, i2tx, module_origins = _import_parent_transactions(reviewed)
        roots = tuple(i2tx.OFFICIAL_ROOTS)
        freshness = _freshness(directories, roots)
        if not freshness["pass"]:
            raise RuntimeError(f"official freshness failed before reservation: {freshness}")
        _preflight(reviewed, directories, base)
        directories = _create_artifacts_directory(directories)
        executed_sources = _persist_executed_sources(reviewed, directories)
        returncode, outcome = _execute_transaction(
            reviewed=reviewed,
            directories=directories,
            executed_sources=executed_sources,
            freshness=freshness,
            base_transaction=base,
            i2tx=i2tx,
            module_origins=module_origins,
            roots=roots,
        )
        return returncode
    finally:
        directories.close()
        reviewed.archive.close()
        if outcome is not None:
            if outcome.stdout:
                sys.stdout.buffer.write(outcome.stdout)
                sys.stdout.buffer.flush()
            if outcome.stderr:
                sys.stderr.buffer.write(outcome.stderr)
                sys.stderr.buffer.flush()


if __name__ == "__main__":
    raise SystemExit(main())
