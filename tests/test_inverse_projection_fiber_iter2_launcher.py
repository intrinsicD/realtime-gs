from __future__ import annotations

import hashlib
import io
import os
import struct
import sys
import types
import uuid
import zipfile
from pathlib import Path
from typing import Any

import pytest

LAUNCHER = Path(__file__).parents[1] / "benchmarks/inverse_projection_fiber_iter2_launcher.py"
SYNTHETIC_ROOTS = (101, 201, 301, 102, 202, 302)


def _load_launcher(workspace: Path) -> tuple[types.ModuleType, int, str]:
    payload = LAUNCHER.read_bytes()
    descriptor = os.open(workspace, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    name = f"_iter2_launcher_test_{uuid.uuid4().hex}"
    module = types.ModuleType(name)
    module.__file__ = str(LAUNCHER)
    module.__dict__["_RTGS_BOOTSTRAP"] = {
        "workspace_fd": descriptor,
        "workspace_path_hint": str(workspace),
        "launcher_sha256": hashlib.sha256(payload).hexdigest(),
        "implementation_review_sha256": "0" * 64,
    }
    sys.modules[name] = module
    try:
        exec(compile(payload, str(LAUNCHER), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        os.close(descriptor)
        raise
    return module, descriptor, name


@pytest.fixture
def launcher(tmp_path: Path):
    module, descriptor, name = _load_launcher(tmp_path)
    try:
        yield module
    finally:
        sys.modules.pop(name, None)
        os.close(descriptor)


def test_launcher_rejects_absent_bootstrap(tmp_path: Path) -> None:
    payload = LAUNCHER.read_bytes()
    name = f"_iter2_launcher_missing_bootstrap_{uuid.uuid4().hex}"
    module = types.ModuleType(name)
    module.__file__ = str(LAUNCHER)
    sys.modules[name] = module
    try:
        with pytest.raises(RuntimeError, match="exact injected"):
            exec(compile(payload, str(LAUNCHER), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)


def test_bootstrap_contract_is_exact_and_fd_backed(launcher: Any, tmp_path: Path) -> None:
    valid = dict(launcher.__dict__["_RTGS_BOOTSTRAP"])
    assert (
        launcher._require_bootstrap({"_RTGS_BOOTSTRAP": valid}).workspace_fd
        == valid["workspace_fd"]
    )
    with pytest.raises(launcher.BootstrapError):
        launcher._require_bootstrap({"_RTGS_BOOTSTRAP": {**valid, "extra": True}})
    with pytest.raises(launcher.BootstrapError):
        launcher._require_bootstrap({"_RTGS_BOOTSTRAP": {**valid, "launcher_sha256": "bad"}})
    regular = tmp_path / "regular"
    regular.write_bytes(b"x")
    descriptor = os.open(regular, os.O_RDONLY)
    try:
        with pytest.raises(launcher.BootstrapError):
            launcher._require_bootstrap({"_RTGS_BOOTSTRAP": {**valid, "workspace_fd": descriptor}})
    finally:
        os.close(descriptor)


def test_same_fd_capture_rejects_symlink_components(launcher: Any, tmp_path: Path) -> None:
    source = tmp_path / "safe" / "source.py"
    source.parent.mkdir()
    source.write_bytes(b"VALUE = 1\n")
    captured = launcher._capture_relative(
        launcher.BOOTSTRAP.workspace_fd,
        "safe/source.py",
        workspace_path_hint=str(tmp_path),
        max_bytes=1024,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    assert captured.payload == b"VALUE = 1\n"
    (tmp_path / "safe" / "final-link.py").symlink_to(source)
    (tmp_path / "parent-link").symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(OSError):
        launcher._capture_relative(
            launcher.BOOTSTRAP.workspace_fd,
            "safe/final-link.py",
            workspace_path_hint=str(tmp_path),
            max_bytes=1024,
        )
    with pytest.raises(OSError):
        launcher._capture_relative(
            launcher.BOOTSTRAP.workspace_fd,
            "parent-link/source.py",
            workspace_path_hint=str(tmp_path),
            max_bytes=1024,
        )


def _npy(dtype: str, shape: tuple[int, ...], data: bytes, *, fortran: bool = False) -> bytes:
    header = repr({"descr": dtype, "fortran_order": fortran, "shape": shape}).encode("latin1")
    padding = (-((10 + len(header) + 1) % 64)) % 64
    header += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data


def _npz(payloads: dict[str, bytes], *, compression: int = zipfile.ZIP_STORED) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=compression) as archive:
        for name, payload in payloads.items():
            archive.writestr(f"{name}.npy", payload)
    return stream.getvalue()


def test_npz_inspection_rederives_member_and_semantic_hash(
    launcher: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = struct.pack("<2d", 1.5, -2.0)
    member = _npy("<f8", (2,), data)
    payload = _npz({"values": member})
    members, observed = launcher._inspect_npz(payload)
    digest = hashlib.sha256()
    digest.update(b"values\0float64\0[2]\0")
    digest.update(data)
    assert members == {"values": {"dtype": "<f8", "shape": [2]}}
    assert observed == digest.hexdigest()
    with pytest.raises(launcher.ValidationError, match="Fortran"):
        launcher._inspect_npz(_npz({"values": _npy("<f8", (2,), data, fortran=True)}))
    with pytest.raises(launcher.ValidationError, match="unsafe NPZ member"):
        launcher._inspect_npz(_npz({"values": member}, compression=zipfile.ZIP_DEFLATED))
    monkeypatch.setattr(launcher, "MAX_INPUT_EVIDENCE_BYTES", len(member) - 1)
    with pytest.raises(launcher.ValidationError, match="member exceeds"):
        launcher._inspect_npz(_npz({"values": member}))
    monkeypatch.setattr(launcher, "MAX_INPUT_EVIDENCE_BYTES", len(member))
    with pytest.raises(launcher.ValidationError, match="total uncompressed"):
        launcher._inspect_npz(_npz({"first": member, "second": member}))


def test_scientific_status_must_match_prepared_result(launcher: Any) -> None:
    handoff = {"scientific_status": "FAIL"}
    prepared = {"status": "FAIL", "scientific_gates": {"overall_status": "FAIL"}}
    assert launcher._validated_scientific_status(handoff, prepared) == "FAIL"

    invalid_pairs = (
        ({"scientific_status": "UNKNOWN"}, prepared),
        ({"scientific_status": True}, prepared),
        (handoff, {"status": "PASS", "scientific_gates": {"overall_status": "FAIL"}}),
        (handoff, {"status": "FAIL", "scientific_gates": {"overall_status": "PASS"}}),
        (handoff, {"status": "FAIL"}),
    )
    for invalid_handoff, invalid_prepared in invalid_pairs:
        with pytest.raises(launcher.ValidationError, match="scientific status"):
            launcher._validated_scientific_status(invalid_handoff, invalid_prepared)


def _held_directories(launcher: Any, workspace: Path) -> Any:
    results = workspace / "results"
    runs = workspace / "runs"
    artifacts = runs / "artifacts"
    results.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    return launcher.HeldDirectories(
        workspace_fd=os.open(workspace, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        results_fd=os.open(results, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        runs_fd=os.open(runs, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        workspace_path=workspace,
        results_path=results,
        runs_path=runs,
        artifacts_fd=os.open(artifacts, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        artifacts_path=artifacts,
    )


def _close_held(directories: Any) -> None:
    workspace_fd = directories.workspace_fd
    directories.close()
    os.close(workspace_fd)


def test_input_evidence_is_reopened_beneath_artifact_fd(launcher: Any, tmp_path: Path) -> None:
    directories = _held_directories(launcher, tmp_path)
    try:
        root = launcher.RootProgress(0, SYNTHETIC_ROOTS[:3], "ROOT_STATE_0.json")
        scene = directories.artifacts_path / f"scene_{root.bundle[0]}"
        scene.mkdir()
        data = struct.pack("<2q", 7, 9)
        payload = _npz({"indices": _npy("<i8", (2,), data)})
        evidence_path = scene / "input_evidence.npz"
        evidence_path.write_bytes(payload)
        metadata = evidence_path.stat()
        _members, semantic = launcher._inspect_npz(payload)
        descriptor = {
            "path": "/proc/self/fd/999/input_evidence.npz",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "semantic_sha256": semantic,
            "members": {"indices": {"dtype": "<i8", "shape": [2]}},
        }
        receipt = {
            "input_evidence_relative_path": f"scene_{root.bundle[0]}/input_evidence.npz",
            "input_evidence": descriptor,
        }
        validation = launcher._validate_input_evidence(directories, receipt=receipt, root=root)
        assert validation["semantic_sha256"] == semantic
        receipt["input_evidence"] = {**descriptor, "semantic_sha256": "f" * 64}
        with pytest.raises(launcher.ValidationError, match="semantic"):
            launcher._validate_input_evidence(directories, receipt=receipt, root=root)
    finally:
        _close_held(directories)


class _FakeDomain:
    def make_receipt(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        root_consumption_status: str,
        roots: tuple[int, ...],
        official_phase: str | None = None,
        commit_state: str | None = None,
    ) -> dict[str, Any]:
        receipt = {
            "kind": kind,
            "root_consumption_status": root_consumption_status,
            "roots": list(roots),
            **payload,
        }
        if official_phase is not None:
            receipt["official_phase"] = official_phase
        if commit_state is not None:
            receipt["commit_state"] = commit_state
        return receipt


class _FakeTransactions:
    RESERVED_UNCONSUMED = "RESERVED_UNCONSUMED"
    CONSUMPTION_STARTED = "STARTED_CONSERVATIVELY_CONSUMED"
    PARTIALLY_CONSUMED = "PARTIALLY_CONSUMED"
    CONSUMED = "CONSUMED"
    ABORTED_UNKNOWN = "ABORTED_UNKNOWN"

    def __init__(self) -> None:
        self.domain = _FakeDomain()
        self.entries: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        self.events: list[tuple[str, str, str | None]] = []

    def official_domain(self) -> _FakeDomain:
        return self.domain

    def publish_receipt(
        self,
        _directory: Path,
        target: str,
        _kind: str,
        receipt: dict[str, Any],
        *,
        nonce: str,
        directory_fd: int,
    ) -> dict[str, Any]:
        del nonce
        key = (directory_fd, target)
        assert key not in self.entries
        public = {"target": target, "version": 1}
        self.entries[key] = (receipt, public)
        self.events.append(("publish", target, receipt.get("official_phase")))
        return {"public": public}

    def exchange_receipt(
        self,
        _directory: Path,
        target: str,
        _kind: str,
        receipt: dict[str, Any],
        *,
        expected_public: dict[str, Any],
        nonce: str,
        directory_fd: int,
    ) -> dict[str, Any]:
        del nonce
        key = (directory_fd, target)
        assert self.entries[key][1] == expected_public
        public = {"target": target, "version": expected_public["version"] + 1}
        self.entries[key] = (receipt, public)
        self.events.append(("exchange", target, receipt.get("official_phase")))
        return {"public": public}

    def capture_receipt(
        self,
        _directory: Path,
        target: str,
        _kind: str,
        *,
        expected_public: dict[str, Any] | None = None,
        directory_fd: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        receipt, public = self.entries[(directory_fd, target)]
        if expected_public is not None:
            assert expected_public == public
        self.events.append(("capture", target, receipt.get("official_phase")))
        return receipt, public

    def exchange_prepared_receipt(
        self,
        _directory: Path,
        descriptor: dict[str, Any],
        *,
        expected_public: dict[str, Any],
        expected_target_name: str,
        expected_receipt_kind: str,
        expected_transaction_id: str,
        expected_root_consumption_status: str,
        expected_roots: tuple[int, ...],
        expected_official_phase: str | None,
        expected_commit_state: str | None,
        directory_fd: int,
    ) -> dict[str, Any]:
        del (
            descriptor,
            expected_receipt_kind,
            expected_transaction_id,
            expected_root_consumption_status,
            expected_roots,
            expected_commit_state,
        )
        key = (directory_fd, expected_target_name)
        assert self.entries[key][1] == expected_public
        public = {"target": expected_target_name, "version": expected_public["version"] + 1}
        self.entries[key] = ({"kind": "result", "official_phase": expected_official_phase}, public)
        self.events.append(("exchange_prepared", expected_target_name, expected_official_phase))
        return {"public": public}


def _capture(launcher: Any, label: str) -> Any:
    payload = label.encode()
    return launcher.StableCapture(
        label,
        f"/synthetic/{label}",
        payload,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        1,
        1,
        0o100600,
        1,
        1,
    )


def _reviewed(launcher: Any, tmp_path: Path) -> tuple[Any, int]:
    archive_path = tmp_path / "archive"
    archive_path.write_bytes(b"zip")
    archive_fd = os.open(archive_path, os.O_RDONLY)
    member = types.SimpleNamespace(
        path="benchmarks/fake.py", module_name="benchmarks.fake", size=3, sha256="b" * 64
    )
    image = types.SimpleNamespace(payload=b"zip", sha256="a" * 64, members=(member,))
    archive = types.SimpleNamespace(name="synthetic", image=image, fileno=lambda: archive_fd)
    verification = types.SimpleNamespace(
        size=3, sha256="a" * 64, device=1, inode=2, mode=0o100400, link_count=0, seals=15
    )
    helper = types.SimpleNamespace(verify_sealed_source_archive=lambda _archive: verification)
    review_capture = _capture(launcher, "review")
    reviewed = launcher.ReviewedExecution(
        helper=helper,
        archive=archive,
        implementation_review={"source_hashes": {"benchmarks/fake.py": "b" * 64}},
        implementation_review_capture=review_capture,
        preregistration_capture=_capture(launcher, "prereg"),
        preregistration_review_capture=_capture(launcher, "prereg-review"),
        prior_result_capture=_capture(launcher, "prior"),
        source_bytes={},
        source_captures={},
        launcher_capture=_capture(launcher, "launcher"),
    )
    return reviewed, archive_fd


def _install_fake_handoff(
    monkeypatch: pytest.MonkeyPatch, launcher: Any, transactions: _FakeTransactions
) -> None:
    def capture_handoff(*_args: Any, **_kwargs: Any):
        transactions.events.append(("validate", "WORKER_HANDOFF.json", "WORKER_HANDOFF"))
        return (
            {"scientific_status": "PASS"},
            {"target": "WORKER_HANDOFF.json"},
            [],
            {"descriptor": {"prepared": True}, "public": {"recovery": True}},
        )

    monkeypatch.setattr(launcher, "_capture_worker_handoff", capture_handoff)


def test_transaction_controller_commits_only_after_validated_handoff(
    launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directories = _held_directories(launcher, tmp_path)
    reviewed, archive_fd = _reviewed(launcher, tmp_path)
    transactions = _FakeTransactions()
    _install_fake_handoff(monkeypatch, launcher, transactions)
    calls = 0

    def worker(_launch: Any) -> Any:
        nonlocal calls
        calls += 1
        transactions.events.append(("worker", "subprocess", None))
        return launcher.WorkerOutcome(0, b"", b"", True, False)

    try:
        returncode, _outcome = launcher._execute_transaction(
            reviewed=reviewed,
            directories=directories,
            executed_sources={"sha256": "a" * 64},
            freshness={"pass": True},
            base_transaction=object(),
            i2tx=transactions,
            module_origins={},
            roots=SYNTHETIC_ROOTS,
            worker_runner=worker,
            transaction_id_factory=lambda: "1" * 32,
        )
        assert returncode == 0
        assert calls == 1
        phases = [event[2] for event in transactions.events]
        assert phases[-1] == "LIFECYCLE"
        assert phases.index("RESULT_RESERVED") < phases.index("ATTEMPT_RESERVED")
        assert phases.index("STARTED") < phases.index("WORKER_HANDOFF")
        assert phases.index("WORKER_HANDOFF") < phases.index("CONSUMED")
        assert phases.index("CONSUMED") < phases.index("FINAL")
        assert phases.index("FINAL") < phases.index("TERMINAL") < phases.index("LIFECYCLE")
    finally:
        os.close(archive_fd)
        _close_held(directories)


def test_worker_failure_aborts_without_result_exchange_or_retry(
    launcher: Any, tmp_path: Path
) -> None:
    directories = _held_directories(launcher, tmp_path)
    reviewed, archive_fd = _reviewed(launcher, tmp_path)
    transactions = _FakeTransactions()
    calls = 0

    def worker(_launch: Any) -> Any:
        nonlocal calls
        calls += 1
        transactions.events.append(("worker", "subprocess", None))
        return launcher.WorkerOutcome(7, b"", b"failed", True, False)

    try:
        returncode, _outcome = launcher._execute_transaction(
            reviewed=reviewed,
            directories=directories,
            executed_sources={"sha256": "a" * 64},
            freshness={"pass": True},
            base_transaction=object(),
            i2tx=transactions,
            module_origins={},
            roots=SYNTHETIC_ROOTS,
            worker_runner=worker,
            transaction_id_factory=lambda: "2" * 32,
        )
        assert returncode == 1
        assert calls == 1
        assert not any(event[0] == "exchange_prepared" for event in transactions.events)
        assert transactions.events[-1][2] == "LIFECYCLE"
        root_receipts = [
            receipt
            for (fd, name), (receipt, _public) in transactions.entries.items()
            if fd == directories.artifacts_fd and name.startswith("ROOT_STATE_")
        ]
        assert root_receipts
        assert {receipt["root_consumption_status"] for receipt in root_receipts} == {
            transactions.ABORTED_UNKNOWN
        }
        result_receipt = transactions.entries[(directories.results_fd, launcher.RESULT_NAME)][0]
        assert result_receipt["official_phase"] == "RESULT_RESERVED"
    finally:
        os.close(archive_fd)
        _close_held(directories)


def test_worker_bootstrap_restores_cloexec_and_sets_parent_death_signal(launcher: Any) -> None:
    source = launcher._worker_bootstrap_source()
    assert "prctl(1,signal.SIGKILL" in source
    assert "os.getppid()!=parent" in source
    assert "os.set_inheritable" in source
    assert "runpy.run_module('benchmarks.inverse_projection_fiber_iter2'" in source
    assert "run_path" not in source


def test_preflight_actually_probes_both_filesystems_and_pdeathsig(
    launcher: Any, tmp_path: Path
) -> None:
    directories = _held_directories(launcher, tmp_path)

    class FakeBase:
        calls: list[str] = []

        @classmethod
        def _rename_exchange(cls, directory_fd: int, first: str, second: str) -> None:
            temporary = ".swap"
            os.rename(first, temporary, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.rename(second, first, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.rename(temporary, second, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            cls.calls.append(Path(os.readlink(f"/proc/self/fd/{directory_fd}")).parent.name)

    try:
        launcher._preflight_rename_exchange(directories.results_fd, FakeBase, label="results")
        launcher._preflight_rename_exchange(directories.runs_fd, FakeBase, label="runs")
        launcher._preflight_pdeathsig()
        assert FakeBase.calls == ["results", "runs"]
        assert not any(
            name.startswith(".rtgs-iter2-preflight-") for name in os.listdir(directories.results_fd)
        )
        assert not any(
            name.startswith(".rtgs-iter2-preflight-") for name in os.listdir(directories.runs_fd)
        )
    finally:
        _close_held(directories)
