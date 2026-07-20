"""CPU-only tests for deterministic sealed inverse-projection source archives."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib
import importlib.abc
import io
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType

import pytest
from benchmarks import inverse_projection_fiber_sealed_archive as sealed


def _sources() -> dict[str, bytes]:
    return {
        "src/rtgs/__init__.py": b'VALUE = "package"\n',
        "src/rtgs/sealed_probe.py": b'VALUE = "worker"\n',
        "benchmarks/sealed_task.py": b'VALUE = "benchmark"\n',
    }


def _reviewed(sources: dict[str, bytes]) -> dict[str, str]:
    return {path: hashlib.sha256(payload).hexdigest() for path, payload in sources.items()}


def _create(*, name: str = "sealed-archive-test") -> sealed.SealedSourceArchive:
    sources = _sources()
    return sealed.create_sealed_source_archive(sources, _reviewed(sources), name=name)


def test_deterministic_zip_uses_exact_allowlist_and_real_import_roots() -> None:
    sources = _sources()
    reverse = dict(reversed(tuple(sources.items())))
    first = sealed.build_deterministic_source_zip(sources, _reviewed(sources))
    second = sealed.build_deterministic_source_zip(reverse, _reviewed(reverse))

    assert first == second
    expected_paths = (
        "benchmarks/__init__.py",
        "benchmarks/sealed_task.py",
        "src/rtgs/__init__.py",
        "src/rtgs/sealed_probe.py",
    )
    assert tuple(member.path for member in first.members) == expected_paths
    assert [member.module_name for member in first.members] == [
        "benchmarks",
        "benchmarks.sealed_task",
        "rtgs",
        "rtgs.sealed_probe",
    ]
    assert first.allowlist_sha256 == tuple(sorted(_reviewed(sources).items()))
    with zipfile.ZipFile(io.BytesIO(first.payload), mode="r") as archive:
        assert archive.namelist() == list(expected_paths)
        assert archive.read("benchmarks/__init__.py") == sealed.GENERATED_BENCHMARKS_INIT_BYTES
        for path, payload in sources.items():
            assert archive.read(path) == payload
        for info in archive.infolist():
            assert info.date_time == sealed.FIXED_ZIP_TIMESTAMP
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.create_system == 3
            assert info.create_version == 20
            assert info.extract_version == 20
            assert info.flag_bits == 0
            assert info.external_attr >> 16 == 0o100444
            assert info.extra == b""
            assert info.comment == b""
    sealed.verify_archive_image(first)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute.py",
        "../escape.py",
        "benchmarks/../escape.py",
        "benchmarks/./module.py",
        "benchmarks//module.py",
        "benchmarks\\module.py",
        "C:module.py",
        "benchmarks/",
        "benchmarks/module.pyc",
        "benchmarks/module.txt",
        "benchmarks/__pycache__/module.py",
        "benchmarks/bad-name.py",
        "__init__.py",
        "benchmarks/\x00module.py",
        "b\N{LATIN SMALL LETTER A WITH DIAERESIS}nchmarks/module.py",
        "sealed_probe/module.py",
        "src/other/module.py",
    ],
)
def test_archive_path_validation_rejects_unsafe_or_unapproved_names(path: str) -> None:
    with pytest.raises(sealed.UnsafeArchivePathError):
        sealed.validate_archive_member_path(path)


def test_allowlist_rejects_generated_member_nonbytes_and_missing_package() -> None:
    generated = {sealed.GENERATED_BENCHMARKS_INIT: b"caller controlled\n"}
    with pytest.raises(ValueError, match="generated"):
        sealed.build_deterministic_source_zip(generated, _reviewed(generated))

    with pytest.raises(TypeError, match="exact bytes"):
        sealed.build_deterministic_source_zip(
            {"benchmarks/probe.py": bytearray(b"not exact bytes")},  # type: ignore[dict-item]
            {"benchmarks/probe.py": hashlib.sha256(b"not exact bytes").hexdigest()},
        )

    missing_package = {"src/rtgs/module.py": b"VALUE = 1\n"}
    with pytest.raises(sealed.SealedArchiveError, match="missing package initializer"):
        sealed.build_deterministic_source_zip(missing_package, _reviewed(missing_package))


@pytest.mark.parametrize("case", ["missing", "extra", "changed"])
def test_source_bytes_must_match_exact_reviewed_path_hash_allowlist(case: str) -> None:
    sources = _sources()
    reviewed = _reviewed(sources)
    if case == "missing":
        sources.pop("benchmarks/sealed_task.py")
    elif case == "extra":
        sources["benchmarks/extra.py"] = b"EXTRA = True\n"
    else:
        sources["benchmarks/sealed_task.py"] += b"# changed\n"
    with pytest.raises(sealed.SealedArchiveError, match="allowlist|reviewed SHA-256"):
        sealed.build_deterministic_source_zip(sources, reviewed)


def test_source_allowlist_rejects_module_identity_collisions() -> None:
    sources = {
        "benchmarks/collision.py": b"VALUE = 1\n",
        "benchmarks/collision/__init__.py": b"VALUE = 2\n",
    }
    with pytest.raises(sealed.SealedArchiveError, match="collide as module"):
        sealed.build_deterministic_source_zip(sources, _reviewed(sources))


def test_archive_image_rejects_trailing_polyglot_bytes() -> None:
    sources = _sources()
    image = sealed.build_deterministic_source_zip(sources, _reviewed(sources))
    payload = image.payload + b"unreviewed trailing bytes"
    altered = replace(image, payload=payload, sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(sealed.SealedArchiveError, match="trailing"):
        sealed.verify_archive_image(altered)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux memfd contract")
def test_ctypes_fallback_creates_fully_sealed_same_descriptor_archive(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    original = sealed._memfd_create_ctypes

    def observed_fallback(name: str, flags: int) -> int:
        calls.append((name, flags))
        return original(name, flags)

    monkeypatch.delattr(sealed.os, "memfd_create", raising=False)
    monkeypatch.setattr(sealed, "_memfd_create_ctypes", observed_fallback)
    archive = _create()
    descriptor = archive.fileno()
    try:
        assert calls == [("sealed-archive-test", sealed.MFD_CLOEXEC | sealed.MFD_ALLOW_SEALING)]
        verification = sealed.verify_sealed_source_archive(archive)
        assert verification == archive.verification
        assert verification.seals == sealed.REQUIRED_SEALS
        assert verification.close_on_exec
        assert verification.link_count == 0
        assert verification.proc_path == f"/proc/self/fd/{descriptor}"
        assert verification.link_target == "/memfd:sealed-archive-test (deleted)"

        for operation in (
            lambda: os.pwrite(descriptor, b"x", 0),
            lambda: os.ftruncate(descriptor, verification.size - 1),
            lambda: os.ftruncate(descriptor, verification.size + 1),
            lambda: fcntl.fcntl(descriptor, sealed.F_ADD_SEALS, sealed.F_SEAL_WRITE),
        ):
            with pytest.raises(OSError) as caught:
                operation()
            assert caught.value.errno in {errno.EPERM, errno.EBUSY}
    finally:
        archive.close()
    with pytest.raises(OSError) as caught:
        os.fstat(descriptor)
    assert caught.value.errno == errno.EBADF
    with pytest.raises(ValueError, match="closed"):
        archive.fileno()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux memfd contract")
def test_seal_verification_rejects_unsealed_mismatched_or_reused_descriptor() -> None:
    sources = _sources()
    image = sealed.build_deterministic_source_zip(sources, _reviewed(sources))
    descriptor = sealed._memfd_create("unsealed-source-test")
    try:
        os.write(descriptor, image.payload)
        with pytest.raises(sealed.SealedArchiveError, match="seal mask"):
            sealed.verify_sealed_memfd(
                descriptor,
                image,
                expected_name="unsealed-source-test",
            )
    finally:
        os.close(descriptor)

    with _create() as archive:
        wrong = replace(archive.image, sha256="0" * 64)
        with pytest.raises(sealed.SealedArchiveError, match="SHA-256"):
            sealed.verify_sealed_memfd(
                archive.fileno(),
                wrong,
                expected_name=archive.name,
            )

    first = _create(name="fd-reuse-test")
    first_descriptor = first.fileno()
    os.close(first_descriptor)
    replacement = _create(name="fd-reuse-test")
    if replacement.fileno() != first_descriptor:
        os.dup2(replacement.fileno(), first_descriptor)
    try:
        with pytest.raises(sealed.SealedArchiveError, match="identity changed"):
            sealed.verify_sealed_source_archive(first)
    finally:
        first._closed = True
        if replacement.fileno() != first_descriptor:
            os.close(first_descriptor)
        replacement.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux memfd contract")
def test_create_failure_closes_partially_populated_memfd(monkeypatch) -> None:
    descriptors: list[int] = []
    original_create = sealed._memfd_create

    def observed_create(name: str) -> int:
        descriptor = original_create(name)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(sealed, "_memfd_create", observed_create)
    monkeypatch.setattr(sealed.os, "write", lambda _fd, _payload: 0)
    sources = _sources()
    with pytest.raises(OSError, match="zero-length write"):
        sealed.create_sealed_source_archive(sources, _reviewed(sources))
    assert len(descriptors) == 1
    with pytest.raises(OSError) as caught:
        os.fstat(descriptors[0])
    assert caught.value.errno == errno.EBADF


def test_preimport_guard_scans_the_complete_project_module_table() -> None:
    sealed.require_project_modules_unloaded({"json": ModuleType("json")})
    contaminated = {
        "json": ModuleType("json"),
        "rtgs.workspace_copy": ModuleType("rtgs.workspace_copy"),
    }
    with pytest.raises(sealed.SealedArchiveError, match="loaded before sealed import"):
        sealed.require_project_modules_unloaded(contaminated)


class _ForgedLoader(importlib.abc.Loader):
    def get_data(self, _path: str) -> bytes:
        return _sources()["src/rtgs/sealed_probe.py"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux memfd contract")
def test_module_origin_verification_rejects_workspace_and_forged_loader() -> None:
    with _create() as archive:
        outside = ModuleType("rtgs.sealed_probe")
        outside.__file__ = "/workspace/rtgs/sealed_probe.py"
        outside.__spec__ = importlib.util.spec_from_loader(
            outside.__name__,
            loader=None,
            origin=outside.__file__,
        )
        with pytest.raises(sealed.SealedArchiveError, match="origin"):
            sealed.verify_module_origins(archive, {outside.__name__: outside})

        forged = ModuleType("rtgs.sealed_probe")
        forged.__file__ = f"{archive.proc_path}/src/rtgs/sealed_probe.py"
        forged.__spec__ = importlib.util.spec_from_loader(
            forged.__name__,
            loader=_ForgedLoader(),
            origin=forged.__file__,
        )
        with pytest.raises(sealed.SealedArchiveError, match="not zipimporter"):
            sealed.verify_module_origins(archive, {forged.__name__: forged})


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux memfd contract")
def test_isolated_child_adopts_fd_and_imports_only_from_proc_archive(tmp_path: Path) -> None:
    helper_path = Path(sealed.__file__)
    sources = {
        **_sources(),
        "benchmarks/inverse_projection_fiber_sealed_archive.py": helper_path.read_bytes(),
    }
    archive = sealed.create_sealed_source_archive(
        sources,
        _reviewed(sources),
        name="isolated-child-sources",
    )
    descriptor = archive.fileno()
    capsule = {
        "name": archive.name,
        "size": archive.verification.size,
        "sha256": archive.image.sha256,
        "members": [asdict(member) for member in archive.image.members],
        "allowlist_sha256": archive.image.allowlist_sha256,
        "verification": asdict(archive.verification),
    }
    child = r"""import fcntl, importlib, json, os, runpy, sys
fd = int(sys.argv[1])
capsule = json.loads(sys.argv[2])
initial_cloexec = bool(fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC)
root = f"/proc/self/fd/{fd}"
sys.path[:0] = [root, root + "/src"]
from benchmarks import inverse_projection_fiber_sealed_archive as helper
remaining = capsule["size"]
offset = 0
blocks = []
while remaining:
    block = os.pread(fd, remaining, offset)
    if not block:
        raise RuntimeError("short inherited archive read")
    blocks.append(block)
    remaining -= len(block)
    offset += len(block)
image = helper.SourceArchiveImage(
    payload=b"".join(blocks),
    sha256=capsule["sha256"],
    members=tuple(helper.ArchiveMember(**item) for item in capsule["members"]),
    allowlist_sha256=tuple(tuple(item) for item in capsule["allowlist_sha256"]),
)
expected = helper.MemfdVerification(**capsule["verification"])
archive = helper.adopt_inherited_sealed_source_archive(
    fd,
    name=capsule["name"],
    image=image,
    expected_verification=expected,
)
run = runpy.run_module("benchmarks.sealed_task", run_name="__sealed_entry__")
probe = importlib.import_module("rtgs.sealed_probe")
task = importlib.import_module("benchmarks.sealed_task")
verification = helper.verify_loaded_project_modules(archive)
assert run["__spec__"].origin == root + "/benchmarks/sealed_task.py"
assert probe.__spec__.origin == root + "/src/rtgs/sealed_probe.py"
assert task.__spec__.origin == root + "/benchmarks/sealed_task.py"
print(json.dumps({
    "initial_cloexec": initial_cloexec,
    "restored_cloexec": archive.verification.close_on_exec,
    "modules": [item.module_name for item in verification.modules],
}, sort_keys=True))
archive.close()
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", child, str(descriptor), json.dumps(capsule)],
            cwd=tmp_path,
            pass_fds=(descriptor,),
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        archive.close()
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["initial_cloexec"] is False
    assert result["restored_cloexec"] is True
    assert result["modules"] == [
        "benchmarks",
        "benchmarks.inverse_projection_fiber_sealed_archive",
        "benchmarks.sealed_task",
        "rtgs",
        "rtgs.sealed_probe",
    ]
