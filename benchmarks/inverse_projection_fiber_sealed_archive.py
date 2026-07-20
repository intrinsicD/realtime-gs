#!/usr/bin/env python3
"""Deterministic, sealed source archives for inverse-projection-fiber workers.

This module is deliberately independent of the Iteration 2 launcher and scientific worker.  Its
input is an exact mapping of approved repository-relative Python paths to already-captured bytes;
it never reads source files from the workspace.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import io
import os
import re
import stat
import sys
import zipfile
import zipimport
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import ModuleType

# Linux UAPI values from linux/memfd.h and asm-generic/fcntl.h.  Some supported Python builds do
# not expose os.memfd_create or the fcntl constants, so the reviewed numeric ABI is intentional.
MFD_CLOEXEC = 0x0001
MFD_ALLOW_SEALING = 0x0002
F_ADD_SEALS = 1033
F_GET_SEALS = 1034
F_SEAL_SEAL = 0x0001
F_SEAL_SHRINK = 0x0002
F_SEAL_GROW = 0x0004
F_SEAL_WRITE = 0x0008
REQUIRED_SEALS = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE

GENERATED_BENCHMARKS_INIT = "benchmarks/__init__.py"
GENERATED_BENCHMARKS_INIT_BYTES = (
    b'"""Generated package marker for a sealed benchmark source archive."""\n'
)
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MAX_MEMBER_COUNT = 4096
MAX_MEMBER_SIZE = 64 * 1024 * 1024
MAX_TOTAL_SOURCE_SIZE = 512 * 1024 * 1024
MAX_ARCHIVE_PATH_BYTES = 1024

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MEMFD_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SealedArchiveError(RuntimeError):
    """The archive, memfd, seals, or loaded module failed closed verification."""


class UnsafeArchivePathError(ValueError):
    """An archive member is not one canonical, import-safe POSIX Python path."""


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """Immutable identity of one exact source member."""

    path: str
    module_name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceArchiveImage:
    """Complete deterministic ZIP bytes and their exact member manifest."""

    payload: bytes
    sha256: str
    members: tuple[ArchiveMember, ...]
    allowlist_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class MemfdVerification:
    """Same-descriptor content, identity, flag, and seal verification."""

    proc_path: str
    link_target: str
    sha256: str
    size: int
    device: int
    inode: int
    mode: int
    link_count: int
    seals: int
    close_on_exec: bool


@dataclass(frozen=True, slots=True)
class ModuleOrigin:
    """Verified relationship between one loaded module and one archive member."""

    module_name: str
    member_path: str
    origin: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ModuleOriginVerification:
    """Origin and loader-byte evidence for an exact requested module set."""

    archive_sha256: str
    proc_path: str
    modules: tuple[ModuleOrigin, ...]


@dataclass(slots=True)
class SealedSourceArchive:
    """Owned sealed memfd.  Close it explicitly or use it as a context manager."""

    descriptor: int
    name: str
    image: SourceArchiveImage
    verification: MemfdVerification
    _closed: bool = False

    def fileno(self) -> int:
        if self._closed:
            raise ValueError("sealed source archive is closed")
        return self.descriptor

    @property
    def proc_path(self) -> str:
        self.fileno()
        return self.verification.proc_path

    def close(self) -> None:
        if not self._closed:
            os.close(self.descriptor)
            self._closed = True

    def __enter__(self) -> SealedSourceArchive:
        self.fileno()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_archive_member_path(raw_path: str) -> str:
    """Return one canonical, approved repository-relative member path or reject it."""

    if type(raw_path) is not str or not raw_path:
        raise UnsafeArchivePathError("archive member path must be a non-empty string")
    if not raw_path.isascii():
        raise UnsafeArchivePathError("archive member paths must be ASCII")
    if "\x00" in raw_path or "\\" in raw_path:
        raise UnsafeArchivePathError("archive member paths cannot contain NUL or backslash")
    if len(raw_path.encode("ascii")) > MAX_ARCHIVE_PATH_BYTES:
        raise UnsafeArchivePathError("archive member path exceeds the configured bound")
    if raw_path.startswith("/") or raw_path.endswith("/") or "//" in raw_path:
        raise UnsafeArchivePathError("archive member path must be one relative file path")

    parts = raw_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise UnsafeArchivePathError("archive member path cannot contain empty, dot, or dot-dot")
    if "__pycache__" in parts:
        raise UnsafeArchivePathError("__pycache__ members are prohibited")
    if PurePosixPath(raw_path).as_posix() != raw_path:
        raise UnsafeArchivePathError("archive member path is not canonical POSIX form")
    if not raw_path.endswith(".py"):
        raise UnsafeArchivePathError("sealed source archives accept only .py members")

    stem = parts[-1][:-3]
    if not stem or not _IDENTIFIER.fullmatch(stem):
        raise UnsafeArchivePathError("archive member filename is not a Python identifier")
    if any(not _IDENTIFIER.fullmatch(part) for part in parts[:-1]):
        raise UnsafeArchivePathError("archive member directory is not a Python identifier")
    if parts == ["__init__.py"]:
        raise UnsafeArchivePathError("a root-level __init__.py has no importable package identity")
    if not (raw_path.startswith("benchmarks/") or raw_path.startswith("src/rtgs/")):
        raise UnsafeArchivePathError(
            "archive members must be beneath benchmarks/ or the src/rtgs import root"
        )
    return raw_path


def _module_name(path: str) -> str:
    parts = path[:-3].split("/")
    if parts[0] == "src":
        parts.pop(0)
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        raise UnsafeArchivePathError(f"archive member has no module identity: {path!r}")
    return ".".join(parts)


def _validate_package_closure(paths: set[str]) -> None:
    for path in sorted(paths):
        parts = path.split("/")
        first_package_depth = 2 if parts[0] == "src" else 1
        for depth in range(first_package_depth, len(parts)):
            package_init = "/".join((*parts[:depth], "__init__.py"))
            if package_init not in paths:
                raise SealedArchiveError(
                    f"archive member {path!r} is missing package initializer {package_init!r}"
                )


def _validate_allowlist(
    source_bytes: Mapping[str, bytes],
    expected_sha256: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(expected_sha256, Mapping) or not expected_sha256:
        raise ValueError("expected_sha256 must be a non-empty reviewed mapping")
    if set(source_bytes) != set(expected_sha256):
        missing = sorted(set(expected_sha256) - set(source_bytes))
        extra = sorted(set(source_bytes) - set(expected_sha256))
        raise SealedArchiveError(
            "source bytes differ from the exact reviewed allowlist: "
            f"missing={missing}, extra={extra}"
        )
    observed: list[tuple[str, str]] = []
    for raw_path in sorted(expected_sha256):
        path = validate_archive_member_path(raw_path)
        expected = expected_sha256[raw_path]
        if type(expected) is not str or not _SHA256.fullmatch(expected):
            raise ValueError(f"reviewed SHA-256 is not canonical for {path!r}")
        payload = source_bytes[raw_path]
        if type(payload) is not bytes:
            raise TypeError(f"source member {path!r} must be exact bytes")
        if _sha256(payload) != expected:
            raise SealedArchiveError(f"source member {path!r} differs from its reviewed SHA-256")
        observed.append((path, expected))
    return tuple(observed)


def _archive_members(
    source_bytes: Mapping[str, bytes],
    expected_sha256: Mapping[str, str],
) -> tuple[dict[str, bytes], tuple[ArchiveMember, ...], tuple[tuple[str, str], ...]]:
    if not isinstance(source_bytes, Mapping) or not source_bytes:
        raise ValueError("source_bytes must be a non-empty mapping")
    if len(source_bytes) + 1 > MAX_MEMBER_COUNT:
        raise ValueError("source allowlist exceeds the member-count bound")
    allowlist = _validate_allowlist(source_bytes, expected_sha256)

    captured: dict[str, bytes] = {}
    total_size = 0
    for raw_path, payload in source_bytes.items():
        path = validate_archive_member_path(raw_path)
        if path == GENERATED_BENCHMARKS_INIT:
            raise ValueError(
                f"{GENERATED_BENCHMARKS_INIT!r} is generated and cannot be caller supplied"
            )
        if len(payload) > MAX_MEMBER_SIZE:
            raise ValueError(f"source member {path!r} exceeds the per-member size bound")
        if path in captured:
            raise ValueError(f"duplicate source member {path!r}")
        captured[path] = payload
        total_size += len(payload)

    captured[GENERATED_BENCHMARKS_INIT] = GENERATED_BENCHMARKS_INIT_BYTES
    total_size += len(GENERATED_BENCHMARKS_INIT_BYTES)
    if total_size > MAX_TOTAL_SOURCE_SIZE:
        raise ValueError("source allowlist exceeds the total byte bound")

    _validate_package_closure(set(captured))
    modules: dict[str, str] = {}
    manifest: list[ArchiveMember] = []
    for path in sorted(captured):
        module_name = _module_name(path)
        previous = modules.setdefault(module_name, path)
        if previous != path:
            raise SealedArchiveError(
                f"members {previous!r} and {path!r} collide as module {module_name!r}"
            )
        payload = captured[path]
        manifest.append(
            ArchiveMember(
                path=path,
                module_name=module_name,
                size=len(payload),
                sha256=_sha256(payload),
            )
        )
    return captured, tuple(manifest), allowlist


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, FIXED_ZIP_TIMESTAMP)
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o444) << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _observed_zip_members(payload: bytes) -> tuple[ArchiveMember, ...]:
    end_record = payload.rfind(b"PK\x05\x06")
    if end_record < 0 or end_record + 22 > len(payload):
        raise SealedArchiveError("ZIP has no canonical end-of-central-directory record")
    comment_size = int.from_bytes(payload[end_record + 20 : end_record + 22], "little")
    if end_record + 22 + comment_size != len(payload):
        raise SealedArchiveError("ZIP has trailing or noncanonical end-record bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise SealedArchiveError("ZIP contains duplicate member names")
            if names != sorted(names):
                raise SealedArchiveError("ZIP member order is not deterministic")
            if archive.comment:
                raise SealedArchiveError("ZIP archive comment must be empty")

            members: list[ArchiveMember] = []
            for info in infos:
                path = validate_archive_member_path(info.filename)
                if info.is_dir():
                    raise SealedArchiveError(f"ZIP contains directory entry {path!r}")
                if info.date_time != FIXED_ZIP_TIMESTAMP:
                    raise SealedArchiveError(f"ZIP timestamp differs for {path!r}")
                if info.compress_type != zipfile.ZIP_STORED:
                    raise SealedArchiveError(f"ZIP member {path!r} is compressed")
                if info.create_system != 3:
                    raise SealedArchiveError(f"ZIP creator system differs for {path!r}")
                if info.create_version != 20 or info.extract_version != 20:
                    raise SealedArchiveError(f"ZIP version metadata differs for {path!r}")
                if info.flag_bits != 0:
                    raise SealedArchiveError(f"ZIP flag bits differ for {path!r}")
                if info.external_attr >> 16 != stat.S_IFREG | 0o444:
                    raise SealedArchiveError(f"ZIP mode differs for {path!r}")
                if info.extra or info.comment:
                    raise SealedArchiveError(f"ZIP metadata is non-empty for {path!r}")
                source = archive.read(info)
                if len(source) != info.file_size:
                    raise SealedArchiveError(f"ZIP member size differs for {path!r}")
                members.append(
                    ArchiveMember(
                        path=path,
                        module_name=_module_name(path),
                        size=len(source),
                        sha256=_sha256(source),
                    )
                )
            if archive.testzip() is not None:
                raise SealedArchiveError("ZIP CRC verification failed")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SealedArchiveError(f"invalid deterministic source ZIP: {error}") from error
    return tuple(members)


def build_deterministic_source_zip(
    source_bytes: Mapping[str, bytes],
    expected_sha256: Mapping[str, str],
) -> SourceArchiveImage:
    """Build and verify a byte-deterministic stored ZIP from the exact supplied bytes."""

    captured, expected_members, allowlist = _archive_members(source_bytes, expected_sha256)
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        for path in sorted(captured):
            archive.writestr(_zip_info(path), captured[path])
    payload = stream.getvalue()
    observed_members = _observed_zip_members(payload)
    if observed_members != expected_members:
        raise SealedArchiveError("deterministic ZIP differs from the exact source manifest")
    return SourceArchiveImage(
        payload=payload,
        sha256=_sha256(payload),
        members=expected_members,
        allowlist_sha256=allowlist,
    )


def verify_archive_image(image: SourceArchiveImage) -> None:
    """Fail closed unless an image still matches its bytes and exact ZIP manifest."""

    if not isinstance(image, SourceArchiveImage) or type(image.payload) is not bytes:
        raise TypeError("image must be a SourceArchiveImage containing exact bytes")
    if _sha256(image.payload) != image.sha256:
        raise SealedArchiveError("archive image SHA-256 mismatch")
    if _observed_zip_members(image.payload) != image.members:
        raise SealedArchiveError("archive image manifest mismatch")
    observed_allowlist = tuple(
        (member.path, member.sha256)
        for member in image.members
        if member.path != GENERATED_BENCHMARKS_INIT
    )
    if image.allowlist_sha256 != observed_allowlist:
        raise SealedArchiveError("archive image reviewed allowlist mismatch")
    generated = next(
        (member for member in image.members if member.path == GENERATED_BENCHMARKS_INIT),
        None,
    )
    if generated is None or generated.sha256 != _sha256(GENERATED_BENCHMARKS_INIT_BYTES):
        raise SealedArchiveError("archive image generated package marker mismatch")


def _validate_memfd_name(name: str) -> str:
    if (
        type(name) is not str
        or not name
        or not name.isascii()
        or not _MEMFD_NAME.fullmatch(name)
        or len(name.encode("ascii")) > 200
    ):
        raise ValueError("memfd name must be 1-200 safe ASCII characters")
    return name


def _memfd_create_ctypes(name: str, flags: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "memfd_create", None)
    if function is None:
        raise SealedArchiveError("libc does not expose memfd_create")
    function.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    descriptor = function(name.encode("ascii"), flags)
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), name)
    return descriptor


def _memfd_create(name: str) -> int:
    if not sys.platform.startswith("linux"):
        raise SealedArchiveError("sealed source archives require Linux memfd support")
    if not os.path.isdir("/proc/self/fd"):
        raise SealedArchiveError("/proc/self/fd is required for sealed source imports")
    flags = MFD_CLOEXEC | MFD_ALLOW_SEALING
    native = getattr(os, "memfd_create", None)
    descriptor = native(name, flags) if callable(native) else _memfd_create_ctypes(name, flags)
    if type(descriptor) is not int or descriptor < 0:
        raise SealedArchiveError("memfd_create returned an invalid descriptor")
    return descriptor


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("zero-length write while populating sealed source memfd")
        view = view[written:]


def verify_sealed_memfd(
    descriptor: int,
    image: SourceArchiveImage,
    *,
    expected_name: str,
    require_close_on_exec: bool = True,
) -> MemfdVerification:
    """Verify complete seals, same-FD bytes, and the /proc descriptor identity."""

    if type(descriptor) is not int or descriptor < 0:
        raise ValueError("descriptor must be a non-negative integer")
    expected_name = _validate_memfd_name(expected_name)
    verify_archive_image(image)

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SealedArchiveError("sealed source descriptor is not a regular file")
    if before.st_nlink != 0:
        raise SealedArchiveError("sealed source descriptor unexpectedly has a filesystem link")
    if before.st_size != len(image.payload):
        raise SealedArchiveError("sealed source descriptor size differs from archive image")
    seals = fcntl.fcntl(descriptor, F_GET_SEALS)
    if seals != REQUIRED_SEALS:
        raise SealedArchiveError(
            f"sealed source descriptor has seal mask {seals:#x}, expected {REQUIRED_SEALS:#x}"
        )
    descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
    close_on_exec = bool(descriptor_flags & fcntl.FD_CLOEXEC)
    if require_close_on_exec and not close_on_exec:
        raise SealedArchiveError("sealed source descriptor is missing FD_CLOEXEC")

    blocks: list[bytes] = []
    offset = 0
    remaining = before.st_size
    while remaining:
        block = os.pread(descriptor, min(1 << 20, remaining), offset)
        if not block:
            raise SealedArchiveError("short read from sealed source descriptor")
        blocks.append(block)
        offset += len(block)
        remaining -= len(block)
    if os.pread(descriptor, 1, before.st_size):
        raise SealedArchiveError("sealed source descriptor grew during capture")
    payload = b"".join(blocks)

    after = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise SealedArchiveError("sealed source descriptor metadata changed during capture")
    if payload != image.payload or _sha256(payload) != image.sha256:
        raise SealedArchiveError("sealed source descriptor bytes differ from archive image")
    if fcntl.fcntl(descriptor, F_GET_SEALS) != seals:
        raise SealedArchiveError("sealed source descriptor seal mask changed during capture")

    proc_path = f"/proc/self/fd/{descriptor}"
    proc_metadata = os.stat(proc_path)
    if (
        proc_metadata.st_dev,
        proc_metadata.st_ino,
        proc_metadata.st_mode,
        proc_metadata.st_size,
    ) != (after.st_dev, after.st_ino, after.st_mode, after.st_size):
        raise SealedArchiveError("/proc descriptor path does not identify the captured memfd")
    link_target = os.readlink(proc_path)
    expected_link_target = f"/memfd:{expected_name} (deleted)"
    if link_target != expected_link_target:
        raise SealedArchiveError(
            f"/proc descriptor target differs: expected {expected_link_target!r}, "
            f"observed {link_target!r}"
        )

    return MemfdVerification(
        proc_path=proc_path,
        link_target=link_target,
        sha256=image.sha256,
        size=len(payload),
        device=after.st_dev,
        inode=after.st_ino,
        mode=after.st_mode,
        link_count=after.st_nlink,
        seals=seals,
        close_on_exec=close_on_exec,
    )


def create_sealed_source_archive(
    source_bytes: Mapping[str, bytes],
    expected_sha256: Mapping[str, str],
    *,
    name: str = "rtgs-ipf-iter2-sources",
) -> SealedSourceArchive:
    """Build, populate, fully seal, and verify one owned source memfd."""

    name = _validate_memfd_name(name)
    image = build_deterministic_source_zip(source_bytes, expected_sha256)
    descriptor = _memfd_create(name)
    try:
        _write_all(descriptor, image.payload)
        os.fsync(descriptor)
        fcntl.fcntl(descriptor, F_ADD_SEALS, REQUIRED_SEALS)
        verification = verify_sealed_memfd(descriptor, image, expected_name=name)
    except BaseException:
        os.close(descriptor)
        raise
    return SealedSourceArchive(
        descriptor=descriptor,
        name=name,
        image=image,
        verification=verification,
    )


def _same_archive_identity(first: MemfdVerification, second: MemfdVerification) -> bool:
    fields = (
        "proc_path",
        "link_target",
        "sha256",
        "size",
        "device",
        "inode",
        "mode",
        "link_count",
        "seals",
    )
    return all(getattr(first, field) == getattr(second, field) for field in fields)


def verify_sealed_source_archive(archive: SealedSourceArchive) -> MemfdVerification:
    """Reverify an owned descriptor without accepting FD-number reuse as the same archive."""

    if not isinstance(archive, SealedSourceArchive):
        raise TypeError("archive must be a SealedSourceArchive")
    current = verify_sealed_memfd(
        archive.fileno(),
        archive.image,
        expected_name=archive.name,
    )
    if current != archive.verification:
        raise SealedArchiveError("sealed source descriptor identity changed after creation")
    return current


def adopt_inherited_sealed_source_archive(
    descriptor: int,
    *,
    name: str,
    image: SourceArchiveImage,
    expected_verification: MemfdVerification,
) -> SealedSourceArchive:
    """Restore CLOEXEC and adopt one exact memfd inherited through ``pass_fds``."""

    name = _validate_memfd_name(name)
    try:
        inherited = verify_sealed_memfd(
            descriptor,
            image,
            expected_name=name,
            require_close_on_exec=False,
        )
        if not _same_archive_identity(inherited, expected_verification):
            raise SealedArchiveError("inherited sealed source descriptor identity mismatch")
        flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        fcntl.fcntl(descriptor, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
        restored = verify_sealed_memfd(
            descriptor,
            image,
            expected_name=name,
        )
        if restored != expected_verification:
            raise SealedArchiveError("inherited descriptor differs after restoring FD_CLOEXEC")
    except BaseException:
        os.close(descriptor)
        raise
    return SealedSourceArchive(
        descriptor=descriptor,
        name=name,
        image=image,
        verification=restored,
    )


def _expected_loader_prefix(member_path: str) -> str:
    parts = member_path.split("/")
    if parts[-1] == "__init__.py":
        parts = parts[:-2]
    else:
        parts = parts[:-1]
    return "" if not parts else "/".join(parts) + "/"


def verify_module_origins(
    archive: SealedSourceArchive,
    modules: Mapping[str, ModuleType],
    *,
    aliases: Mapping[str, str] | None = None,
) -> ModuleOriginVerification:
    """Verify exact zipimport origins and loader bytes beneath this archive's /proc path."""

    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("modules must be a non-empty mapping")
    aliases = {} if aliases is None else dict(aliases)
    if any(name not in modules for name in aliases):
        raise ValueError("module aliases must name entries in modules")
    verification = verify_sealed_source_archive(archive)
    expected_by_module: dict[str, ArchiveMember] = {}
    for member in archive.image.members:
        if member.module_name in expected_by_module:
            raise SealedArchiveError(
                f"archive has duplicate module identity {member.module_name!r}"
            )
        expected_by_module[member.module_name] = member

    observed: list[ModuleOrigin] = []
    for raw_name in sorted(modules):
        if type(raw_name) is not str or not raw_name:
            raise SealedArchiveError("module names must be non-empty strings")
        module = modules[raw_name]
        if not isinstance(module, ModuleType) or module.__name__ != raw_name:
            raise SealedArchiveError(f"module mapping entry {raw_name!r} has the wrong identity")
        alias_path = aliases.get(raw_name)
        if alias_path is None:
            member = expected_by_module.get(raw_name)
        else:
            alias_path = validate_archive_member_path(alias_path)
            member = next(
                (value for value in archive.image.members if value.path == alias_path),
                None,
            )
        if member is None:
            raise SealedArchiveError(f"module {raw_name!r} is absent from the source archive")

        spec = module.__spec__
        origin = None if spec is None else spec.origin
        expected_origin = f"{verification.proc_path}/{member.path}"
        if origin != expected_origin or getattr(module, "__file__", None) != expected_origin:
            raise SealedArchiveError(
                f"module {raw_name!r} origin is not the sealed descriptor member"
            )
        if spec is None or spec.loader is None:
            raise SealedArchiveError(f"module {raw_name!r} has no loader")
        if not isinstance(spec.loader, zipimport.zipimporter):
            raise SealedArchiveError(f"module {raw_name!r} loader is not zipimporter")
        if spec.loader.archive != verification.proc_path:
            raise SealedArchiveError(f"module {raw_name!r} loader archive differs from the memfd")
        if spec.loader.prefix != _expected_loader_prefix(member.path):
            raise SealedArchiveError(f"module {raw_name!r} loader prefix differs from its member")
        get_data = getattr(spec.loader, "get_data", None)
        if not callable(get_data):
            raise SealedArchiveError(f"module {raw_name!r} loader cannot recapture source bytes")
        loaded_bytes = get_data(origin)
        if type(loaded_bytes) is not bytes:
            raise SealedArchiveError(f"module {raw_name!r} loader returned non-byte source")
        if len(loaded_bytes) != member.size or _sha256(loaded_bytes) != member.sha256:
            raise SealedArchiveError(f"module {raw_name!r} loader bytes differ from the allowlist")

        is_package = member.path.endswith("/__init__.py")
        if (spec.submodule_search_locations is not None) != is_package:
            raise SealedArchiveError(
                f"module {raw_name!r} package identity differs from its member"
            )
        observed.append(
            ModuleOrigin(
                module_name=raw_name,
                member_path=member.path,
                origin=origin,
                sha256=member.sha256,
                size=member.size,
            )
        )
    return ModuleOriginVerification(
        archive_sha256=archive.image.sha256,
        proc_path=verification.proc_path,
        modules=tuple(observed),
    )


def require_project_modules_unloaded(
    module_table: Mapping[str, ModuleType] | None = None,
) -> None:
    """Fail before archive import if any rtgs/benchmarks module is already resident."""

    table = sys.modules if module_table is None else module_table
    loaded = sorted(
        name
        for name, module in table.items()
        if isinstance(module, ModuleType)
        and (
            name == "rtgs"
            or name.startswith("rtgs.")
            or name == "benchmarks"
            or name.startswith("benchmarks.")
        )
    )
    if loaded:
        raise SealedArchiveError(f"project modules were loaded before sealed import: {loaded}")


def verify_loaded_project_modules(
    archive: SealedSourceArchive,
    *,
    module_table: Mapping[str, ModuleType] | None = None,
    main_member: str | None = None,
) -> ModuleOriginVerification:
    """Scan every loaded rtgs/benchmarks module, plus an optional worker ``__main__`` alias."""

    table = sys.modules if module_table is None else module_table
    selected: dict[str, ModuleType] = {}
    for name, module in table.items():
        if not isinstance(module, ModuleType):
            continue
        if (
            name == "rtgs"
            or name.startswith("rtgs.")
            or name == "benchmarks"
            or name.startswith("benchmarks.")
        ):
            selected[name] = module
    aliases: dict[str, str] = {}
    if main_member is not None:
        main = table.get("__main__")
        if not isinstance(main, ModuleType):
            raise SealedArchiveError("worker __main__ module is unavailable")
        selected["__main__"] = main
        aliases["__main__"] = main_member
    if not selected:
        raise SealedArchiveError("no loaded project modules were available to verify")
    return verify_module_origins(archive, selected, aliases=aliases)
