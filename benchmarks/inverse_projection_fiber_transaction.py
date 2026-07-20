#!/usr/bin/env python3
"""Durable JSON transactions for inverse-projection-fiber experiments.

The module intentionally implements a narrow Linux, single-writer protocol.  Every payload is
validated by one immutable :class:`ReceiptDomain`; prepared entries are recovery-qualified from
birth and are never removed; public creation uses an exclusive hard link; and an owned update
uses ``renameat2(RENAME_EXCHANGE)`` with an ownership-checked accept-or-rollback decision.

The optional event hook is for deterministic, root-free fault injection.  A production caller
must not mutate transaction names through the hook.  Fault tests inject at most one mutation at
one seam and are quiescent thereafter, matching the frozen Iteration 1e threat boundary.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RENAME_EXCHANGE = 2
DEFAULT_CAPTURE_MAX_BYTES = 512 * 1024 * 1024
DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
LAST_OBSERVED_YES_AFTER_EXCHANGE = "YES_AFTER_EXCHANGE"
LAST_OBSERVED_NO_AFTER_ROLLBACK = "NO_AFTER_ROLLBACK"
LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION = "UNKNOWN_AFTER_DISRUPTION"
LAST_OBSERVED_YES_AFTER_LINK = "YES_AFTER_LINK"
LAST_OBSERVED_NO_BEFORE_MUTATION = "NO_BEFORE_MUTATION"

EventHook = Callable[[str, Mapping[str, object]], None]


class ReceiptDomainError(ValueError):
    """A receipt does not belong to its immutable producer domain."""


class OwnershipCaptureError(RuntimeError):
    """A path could not be captured as one stable, regular-file identity."""


@dataclass(frozen=True, slots=True)
class ReceiptDomain:
    """Immutable metadata authority for every receipt produced by one execution domain."""

    protocol_label: str
    label: str
    namespace: str
    schema_family: str
    permitted_root_consumption_statuses: tuple[str, ...]
    permitted_roots: tuple[int, ...] = ()
    forbidden_roots: tuple[int, ...] = ()
    official_phases_permitted: bool = False
    commit_states_permitted: bool = False
    forbidden_literals: tuple[str, ...] = ()
    protocol_generation: str = "iter1e"

    def __post_init__(self) -> None:
        for field_name in ("protocol_label", "label", "namespace", "schema_family"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ReceiptDomainError(f"{field_name} must be a non-empty canonical string")
        if self.protocol_generation not in {"iter1e", "iter2"}:
            raise ReceiptDomainError("protocol_generation must be exactly 'iter1e' or 'iter2'")
        if self.label not in {"official", "development"}:
            raise ReceiptDomainError("label must be exactly 'official' or 'development'")
        if not self.permitted_root_consumption_statuses:
            raise ReceiptDomainError("at least one root-consumption status is required")
        if len(set(self.permitted_root_consumption_statuses)) != len(
            self.permitted_root_consumption_statuses
        ):
            raise ReceiptDomainError("root-consumption statuses must be unique")
        if len(set(self.permitted_roots)) != len(self.permitted_roots):
            raise ReceiptDomainError("permitted roots must be unique")
        if len(set(self.forbidden_roots)) != len(self.forbidden_roots):
            raise ReceiptDomainError("forbidden roots must be unique")
        if set(self.permitted_roots) & set(self.forbidden_roots):
            raise ReceiptDomainError("permitted and forbidden roots must be disjoint")
        if any(type(root) is not int for root in (*self.permitted_roots, *self.forbidden_roots)):
            raise ReceiptDomainError("root literals must be integers")
        if self.label == "development":
            if self.permitted_roots:
                raise ReceiptDomainError("development domains cannot permit roots")
            if self.permitted_root_consumption_statuses != (DEVELOPMENT_ONLY,):
                raise ReceiptDomainError("development domains must permit only DEVELOPMENT_ONLY")
            if self.official_phases_permitted or self.commit_states_permitted:
                raise ReceiptDomainError(
                    "development domains cannot permit official phases or commit states"
                )
            if f"development_{self.protocol_generation}" not in self.schema_family:
                raise ReceiptDomainError(
                    "development schema family must contain its generation-qualified "
                    "development token"
                )
            if "development" not in self.namespace:
                raise ReceiptDomainError("development namespace must contain 'development'")
            if self.protocol_generation not in self.namespace:
                raise ReceiptDomainError("development namespace must contain protocol_generation")
            if not self.forbidden_roots:
                raise ReceiptDomainError("development domains must prohibit official roots")
        elif (
            self.protocol_generation not in self.schema_family
            or "development" in self.schema_family
        ):
            raise ReceiptDomainError(
                "official schema family must contain protocol_generation and exclude 'development'"
            )
        elif "development" in self.namespace:
            raise ReceiptDomainError("official namespace cannot contain 'development'")
        elif self.protocol_generation not in self.namespace:
            raise ReceiptDomainError("official namespace must contain protocol_generation")
        elif not self.permitted_roots:
            raise ReceiptDomainError("official domains must enumerate permitted roots")

    def schema(self, kind: str) -> str:
        """Derive a schema from this domain rather than accepting one from a producer."""
        _validate_kind(kind)
        return f"{self.schema_family}_{kind}_v1"

    def make_receipt(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        root_consumption_status: str,
        roots: tuple[int, ...] = (),
        official_phase: str | None = None,
        commit_state: str | None = None,
    ) -> dict[str, Any]:
        """Return a domain-decorated receipt and validate it before any disk operation."""
        reserved = {
            "schema",
            "protocol_label",
            "receipt_domain",
            "namespace",
            "root_consumption_status",
            "roots",
            "official_phase",
            "commit_state",
        }
        overlap = reserved.intersection(payload)
        if overlap:
            raise ReceiptDomainError(
                f"payload cannot independently supply domain fields: {sorted(overlap)}"
            )
        receipt: dict[str, Any] = {
            "schema": self.schema(kind),
            "protocol_label": self.protocol_label,
            "receipt_domain": self.label,
            "namespace": self.namespace,
            "root_consumption_status": root_consumption_status,
            "roots": list(roots),
            **dict(payload),
        }
        if official_phase is not None:
            receipt["official_phase"] = official_phase
        if commit_state is not None:
            receipt["commit_state"] = commit_state
        self.validate_receipt(kind, receipt)
        return receipt

    def validate_receipt(self, kind: str, value: Mapping[str, Any]) -> None:
        """Reject metadata, roots, phases, or literals outside this domain."""
        if not isinstance(value, Mapping):
            raise ReceiptDomainError("receipt must be a mapping")
        expected = {
            "schema": self.schema(kind),
            "protocol_label": self.protocol_label,
            "receipt_domain": self.label,
            "namespace": self.namespace,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise ReceiptDomainError(
                    f"{key} differs from receipt domain: "
                    f"expected {expected_value!r}, observed {value.get(key)!r}"
                )
        status_value = value.get("root_consumption_status")
        if status_value not in self.permitted_root_consumption_statuses:
            raise ReceiptDomainError(f"root_consumption_status {status_value!r} is not permitted")
        roots_value = value.get("roots")
        if not isinstance(roots_value, list) or any(type(root) is not int for root in roots_value):
            raise ReceiptDomainError("roots must be a JSON integer list")
        if len(set(roots_value)) != len(roots_value):
            raise ReceiptDomainError("receipt roots must be unique")
        if any(root not in self.permitted_roots for root in roots_value):
            raise ReceiptDomainError("receipt contains a root outside the domain")
        if not self.official_phases_permitted and "official_phase" in value:
            raise ReceiptDomainError("official phases are prohibited in this domain")
        if not self.commit_states_permitted and "commit_state" in value:
            raise ReceiptDomainError("commit states are prohibited in this domain")
        forbidden_values: tuple[object, ...] = (
            *self.forbidden_roots,
            *self.forbidden_literals,
        )
        for item in _walk_json(value):
            if item in forbidden_values:
                raise ReceiptDomainError(f"receipt contains forbidden literal {item!r}")
            if isinstance(item, str):
                for root in self.forbidden_roots:
                    if str(root) in item:
                        raise ReceiptDomainError(
                            f"receipt string contains forbidden root literal {root}"
                        )
                for literal in self.forbidden_literals:
                    if literal and literal in item:
                        raise ReceiptDomainError(
                            f"receipt string contains forbidden literal {literal!r}"
                        )


@dataclass(frozen=True, slots=True)
class Ownership:
    """Stable regular-file identity and content observed from one descriptor."""

    sha256: str
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    def same_identity_and_hash(self, other: Ownership) -> bool:
        return (
            self.sha256 == other.sha256
            and self.device == other.device
            and self.inode == other.inode
        )


@dataclass(frozen=True, slots=True)
class EntrySnapshot:
    """One last-observed fact about a directory entry; never a future-state promise."""

    name: str
    state: str
    ownership: Ownership | None = None
    device: int | None = None
    inode: int | None = None
    mode: int | None = None
    error_class: str | None = None
    error_message: str | None = None

    @property
    def is_regular(self) -> bool:
        return self.state == "REGULAR" and self.ownership is not None


@dataclass(frozen=True, slots=True)
class PreparedJSON:
    """A complete, durable recovery-qualified JSON entry retained by the transaction."""

    domain: ReceiptDomain
    receipt_kind: str
    target_name: str
    recovery_name: str
    ownership: Ownership
    payload_sha256: str
    payload_size: int


@dataclass(frozen=True, slots=True)
class MutationReport:
    """Owned mutation evidence, including last-observed public/recovery states."""

    operation: str
    target_name: str
    recovery_name: str
    last_observed: str
    accepted: bool
    expected_ownership: Ownership | None
    prepared_ownership: Ownership
    public_entry: EntrySnapshot | None
    recovery_entry: EntrySnapshot | None
    displaced_entry: EntrySnapshot | None
    events: tuple[str, ...]
    errors: tuple[str, ...]
    recovery_uncertainty: bool
    publication_error: bool


class OwnedMutationError(RuntimeError):
    """A failed mutation carrying enough evidence for conservative recovery reporting."""

    def __init__(self, message: str, prepared: PreparedJSON, report: MutationReport):
        super().__init__(message)
        self.prepared = prepared
        self.report = report


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical, finite JSON bytes including the durable trailing newline."""
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
    )


def open_directory(path: str | Path) -> int:
    """Open and hold a real directory for all subsequent single-component operations."""
    flags = (
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(Path(path), flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise NotADirectoryError(path)
    return descriptor


def require_rename_exchange() -> None:
    """Fail closed unless Linux libc exposes ``renameat2``."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("renameat2(RENAME_EXCHANGE) requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if getattr(libc, "renameat2", None) is None:
        raise RuntimeError("libc does not expose renameat2")


def fsync_directory(
    dir_fd: int,
    *,
    event_hook: EventHook | None = None,
    reason: str = "unspecified",
    _events: list[str] | None = None,
) -> None:
    """Fsync a held directory descriptor while exposing deterministic event seams."""
    _emit(
        _events,
        event_hook,
        "before_directory_fsync",
        dir_fd=dir_fd,
        reason=reason,
    )
    os.fsync(dir_fd)
    _emit(
        _events,
        event_hook,
        "after_directory_fsync",
        dir_fd=dir_fd,
        reason=reason,
    )


def _capture_entry_bytes(
    dir_fd: int,
    name: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    max_bytes: int = DEFAULT_CAPTURE_MAX_BYTES,
    event_hook: EventHook | None = None,
    _events: list[str] | None = None,
) -> tuple[bytes, EntrySnapshot]:
    """Capture identity and bytes through one ``O_NOFOLLOW`` descriptor.

    The expected hash is deliberately checked last, after both descriptor metadata checks and
    the path-to-descriptor identity check.
    """
    _validate_name(name)
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive exact integer")
    if expected_size is not None and (
        type(expected_size) is not int or expected_size < 0 or expected_size > max_bytes
    ):
        raise ValueError("expected_size must be a non-negative exact integer within max_bytes")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise RuntimeError("O_NOFOLLOW is required")
    # O_NONBLOCK prevents a concurrently substituted FIFO/device from hanging before fstat can
    # reject it.  It has no effect on regular-file reads.
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | nofollow
    _emit(_events, event_hook, "before_capture_open", dir_fd=dir_fd, name=name)
    descriptor = os.open(name, flags, dir_fd=dir_fd)
    try:
        _emit(_events, event_hook, "after_capture_open", dir_fd=dir_fd, name=name)
        first = os.fstat(descriptor)
        _emit(_events, event_hook, "after_capture_first_fstat", dir_fd=dir_fd, name=name)
        if not stat.S_ISREG(first.st_mode):
            raise OwnershipCaptureError(f"{name!r} is not a regular file")
        if first.st_size < 0 or first.st_size > max_bytes:
            raise OwnershipCaptureError(
                f"{name!r} size {first.st_size} exceeds capture bound {max_bytes}"
            )
        if expected_size is not None and first.st_size != expected_size:
            raise OwnershipCaptureError(
                f"{name!r} size mismatch: expected {expected_size}, observed {first.st_size}"
            )
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        offset = 0
        while offset < first.st_size:
            block = os.pread(descriptor, min(1 << 20, first.st_size - offset), offset)
            if not block:
                raise OwnershipCaptureError(f"{name!r} ended before its captured size")
            blocks.append(block)
            digest.update(block)
            offset += len(block)
        if os.pread(descriptor, 1, first.st_size):
            raise OwnershipCaptureError(f"{name!r} grew beyond its captured size")
        payload = b"".join(blocks)
        observed_sha256 = digest.hexdigest()
        _emit(_events, event_hook, "after_capture_pread", dir_fd=dir_fd, name=name)
        second = os.fstat(descriptor)
        _emit(_events, event_hook, "after_capture_second_fstat", dir_fd=dir_fd, name=name)
        if _stable_metadata(first) != _stable_metadata(second):
            raise OwnershipCaptureError(f"{name!r} metadata changed during same-FD capture")
        _emit(_events, event_hook, "before_capture_path_stat", dir_fd=dir_fd, name=name)
        path_metadata = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if not stat.S_ISREG(path_metadata.st_mode):
            raise OwnershipCaptureError(f"{name!r} path no longer names a regular file")
        if (path_metadata.st_dev, path_metadata.st_ino) != (second.st_dev, second.st_ino):
            raise OwnershipCaptureError(f"{name!r} path identity changed during capture")
        ownership = Ownership(
            sha256=observed_sha256,
            device=second.st_dev,
            inode=second.st_ino,
            size=second.st_size,
            mode=second.st_mode,
            mtime_ns=second.st_mtime_ns,
            ctime_ns=second.st_ctime_ns,
        )
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise OwnershipCaptureError(
                f"{name!r} hash mismatch: expected {expected_sha256}, observed {observed_sha256}"
            )
        _emit(_events, event_hook, "capture_complete", dir_fd=dir_fd, name=name)
        return (
            payload,
            EntrySnapshot(
                name=name,
                state="REGULAR",
                ownership=ownership,
                device=second.st_dev,
                inode=second.st_ino,
                mode=second.st_mode,
            ),
        )
    finally:
        os.close(descriptor)


def capture_entry(
    dir_fd: int,
    name: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    max_bytes: int = DEFAULT_CAPTURE_MAX_BYTES,
    event_hook: EventHook | None = None,
    _events: list[str] | None = None,
) -> EntrySnapshot:
    """Capture stable ownership while discarding the already-validated bytes."""
    _payload, snapshot = _capture_entry_bytes(
        dir_fd,
        name,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        max_bytes=max_bytes,
        event_hook=event_hook,
        _events=_events,
    )
    return snapshot


def capture_entry_bytes(
    dir_fd: int,
    name: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    max_bytes: int = DEFAULT_CAPTURE_MAX_BYTES,
    event_hook: EventHook | None = None,
) -> tuple[bytes, EntrySnapshot]:
    """Capture stable regular-file ownership and the exact bytes from the same descriptor."""
    return _capture_entry_bytes(
        dir_fd,
        name,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        max_bytes=max_bytes,
        event_hook=event_hook,
    )


def observe_entry(dir_fd: int, name: str) -> EntrySnapshot:
    """Best-effort, no-follow observation for mutation reports."""
    _validate_name(name)
    try:
        metadata = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return EntrySnapshot(name=name, state="ABSENT")
    except OSError as error:
        return EntrySnapshot(
            name=name,
            state="ERROR",
            error_class=type(error).__name__,
            error_message=str(error),
        )
    identity = {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
    }
    if stat.S_ISLNK(metadata.st_mode):
        return EntrySnapshot(name=name, state="SYMLINK", **identity)
    if stat.S_ISDIR(metadata.st_mode):
        return EntrySnapshot(name=name, state="DIRECTORY", **identity)
    if not stat.S_ISREG(metadata.st_mode):
        return EntrySnapshot(name=name, state="NON_REGULAR", **identity)
    try:
        return capture_entry(dir_fd, name)
    except PermissionError as error:
        return EntrySnapshot(
            name=name,
            state="UNREADABLE",
            **identity,
            error_class=type(error).__name__,
            error_message=str(error),
        )
    except OwnershipCaptureError as error:
        return EntrySnapshot(
            name=name,
            state="UNSTABLE",
            **identity,
            error_class=type(error).__name__,
            error_message=str(error),
        )
    except OSError as error:
        return EntrySnapshot(
            name=name,
            state="ERROR",
            **identity,
            error_class=type(error).__name__,
            error_message=str(error),
        )


def prepare_json(
    dir_fd: int,
    target_name: str,
    domain: ReceiptDomain,
    receipt_kind: str,
    value: Mapping[str, Any],
    *,
    nonce: str | None = None,
    event_hook: EventHook | None = None,
) -> PreparedJSON:
    """Create and durably retain one complete recovery-qualified prepared JSON entry."""
    _validate_name(target_name)
    _validate_kind(receipt_kind)
    domain.validate_receipt(receipt_kind, value)
    payload = canonical_json_bytes(value)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    nonce_value = uuid.uuid4().hex if nonce is None else nonce
    if (
        not isinstance(nonce_value, str)
        or len(nonce_value) != 32
        or any(character not in "0123456789abcdef" for character in nonce_value)
    ):
        raise ValueError("nonce must be exactly 128 lowercase hexadecimal bits")
    recovery_name = f".{target_name}.recovery.{nonce_value}.prepared"
    _validate_name(recovery_name)
    events: list[str] = []
    _emit(
        events,
        event_hook,
        "before_prepared_create",
        dir_fd=dir_fd,
        target_name=target_name,
        recovery_name=recovery_name,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(recovery_name, flags, 0o600, dir_fd=dir_fd)
    try:
        os.fchmod(descriptor, 0o600)
        _emit(
            events,
            event_hook,
            "after_prepared_create",
            dir_fd=dir_fd,
            target_name=target_name,
            recovery_name=recovery_name,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "zero-length write while preparing JSON")
            view = view[written:]
        _emit(
            events,
            event_hook,
            "before_prepared_file_fsync",
            dir_fd=dir_fd,
            recovery_name=recovery_name,
        )
        os.fsync(descriptor)
        _emit(
            events,
            event_hook,
            "after_prepared_file_fsync",
            dir_fd=dir_fd,
            recovery_name=recovery_name,
        )
    finally:
        os.close(descriptor)
    fsync_directory(
        dir_fd,
        event_hook=event_hook,
        reason="prepared_json_created",
        _events=events,
    )
    capture_entry(
        dir_fd,
        recovery_name,
        expected_sha256=payload_sha256,
        expected_size=len(payload),
        event_hook=event_hook,
    )
    snapshot = capture_entry(
        dir_fd,
        recovery_name,
        expected_sha256=payload_sha256,
        expected_size=len(payload),
    )
    assert snapshot.ownership is not None
    if not _snapshot_secure(snapshot, snapshot.ownership):
        raise OwnershipCaptureError("prepared recovery is not a private 0600 regular file")
    return PreparedJSON(
        domain=domain,
        receipt_kind=receipt_kind,
        target_name=target_name,
        recovery_name=recovery_name,
        ownership=snapshot.ownership,
        payload_sha256=payload_sha256,
        payload_size=len(payload),
    )


def publish_exclusive(
    dir_fd: int,
    prepared: PreparedJSON,
    *,
    event_hook: EventHook | None = None,
) -> MutationReport:
    """Publish an absent target by exclusive hard link, preserving all colliders."""
    events: list[str] = []
    errors: list[str] = []
    linked = False
    try:
        _emit(
            events,
            event_hook,
            "before_prepared_verification",
            dir_fd=dir_fd,
            name=prepared.recovery_name,
        )
        recovery_before = capture_entry(
            dir_fd,
            prepared.recovery_name,
            expected_sha256=prepared.payload_sha256,
            expected_size=prepared.payload_size,
            event_hook=event_hook,
            _events=events,
        )
        if not _snapshot_exact(recovery_before, prepared.ownership):
            raise OwnershipCaptureError("recovery entry is not the prepared inode before link")
        _emit(
            events,
            event_hook,
            "before_exclusive_link",
            dir_fd=dir_fd,
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
        )
        recovery_ready = capture_entry(
            dir_fd,
            prepared.recovery_name,
            expected_sha256=prepared.payload_sha256,
            expected_size=prepared.payload_size,
        )
        if not _snapshot_exact(recovery_ready, prepared.ownership):
            raise OwnershipCaptureError("recovery entry changed at the exclusive-link seam")
        os.link(
            prepared.recovery_name,
            prepared.target_name,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
            follow_symlinks=False,
        )
        linked = True
        _emit(
            events,
            event_hook,
            "after_exclusive_link",
            dir_fd=dir_fd,
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
        )
        fsync_directory(
            dir_fd,
            event_hook=event_hook,
            reason="exclusive_link",
            _events=events,
        )
        public = capture_entry(dir_fd, prepared.target_name, event_hook=event_hook, _events=events)
        recovery = capture_entry(
            dir_fd, prepared.recovery_name, event_hook=event_hook, _events=events
        )
        # Hooked captures are diagnostic seams.  Acceptance is based only on a final hook-free
        # pair observed after every injected seam, preventing capture_complete stale acceptance.
        public = capture_entry(dir_fd, prepared.target_name)
        recovery = capture_entry(dir_fd, prepared.recovery_name)
        if not _snapshot_secure(public, prepared.ownership) or not _snapshot_secure(
            recovery, prepared.ownership
        ):
            raise OwnershipCaptureError(
                "exclusive link did not leave public and recovery names on prepared inode"
            )
        return MutationReport(
            operation="exclusive_link",
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
            last_observed=LAST_OBSERVED_YES_AFTER_LINK,
            accepted=True,
            expected_ownership=None,
            prepared_ownership=prepared.ownership,
            public_entry=public,
            recovery_entry=recovery,
            displaced_entry=None,
            events=tuple(events),
            errors=(),
            recovery_uncertainty=False,
            publication_error=False,
        )
    except Exception as error:
        errors.append(_error_text(error))
        public = observe_entry(dir_fd, prepared.target_name)
        recovery = observe_entry(dir_fd, prepared.recovery_name)
        report = MutationReport(
            operation="exclusive_link",
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
            last_observed=(
                LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION
                if linked
                else LAST_OBSERVED_NO_BEFORE_MUTATION
            ),
            accepted=False,
            expected_ownership=None,
            prepared_ownership=prepared.ownership,
            public_entry=public,
            recovery_entry=recovery,
            displaced_entry=None,
            events=tuple(events),
            errors=tuple(errors),
            recovery_uncertainty=linked,
            publication_error=True,
        )
        raise OwnedMutationError(
            "exclusive prepared publication failed", prepared, report
        ) from error


def exchange_owned(
    dir_fd: int,
    prepared: PreparedJSON,
    expected: Ownership,
    *,
    event_hook: EventHook | None = None,
) -> MutationReport:
    """Exchange an owned target and accept or roll back from retained evidence."""
    require_rename_exchange()
    events: list[str] = []
    errors: list[str] = []
    exchanged = False
    displaced: EntrySnapshot | None = None
    public: EntrySnapshot | None = None
    recovery: EntrySnapshot | None = None
    try:
        _emit(
            events,
            event_hook,
            "before_prepared_verification",
            dir_fd=dir_fd,
            name=prepared.recovery_name,
        )
        recovery_before = capture_entry(
            dir_fd,
            prepared.recovery_name,
            expected_sha256=prepared.payload_sha256,
            expected_size=prepared.payload_size,
            event_hook=event_hook,
            _events=events,
        )
        if not _snapshot_exact(recovery_before, prepared.ownership):
            raise OwnershipCaptureError("recovery entry is not the prepared inode before exchange")
        _emit(
            events,
            event_hook,
            "before_exchange",
            dir_fd=dir_fd,
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
        )
        recovery_ready = capture_entry(
            dir_fd,
            prepared.recovery_name,
            expected_sha256=prepared.payload_sha256,
            expected_size=prepared.payload_size,
        )
        if not _snapshot_exact(recovery_ready, prepared.ownership):
            raise OwnershipCaptureError("recovery entry changed at the exchange seam")
        _rename_exchange(dir_fd, prepared.recovery_name, prepared.target_name)
        exchanged = True
        _emit(
            events,
            event_hook,
            "after_exchange",
            dir_fd=dir_fd,
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
        )
        fsync_directory(
            dir_fd,
            event_hook=event_hook,
            reason="owned_exchange",
            _events=events,
        )
        _emit(
            events,
            event_hook,
            "before_public_verification",
            dir_fd=dir_fd,
            name=prepared.target_name,
        )
        public = capture_entry(dir_fd, prepared.target_name, event_hook=event_hook, _events=events)
        _emit(
            events,
            event_hook,
            "before_displaced_verification",
            dir_fd=dir_fd,
            name=prepared.recovery_name,
        )
        displaced = observe_entry(dir_fd, prepared.recovery_name)
        recovery = displaced
        if displaced.state in {"ABSENT", "ERROR", "UNSTABLE"}:
            raise OwnershipCaptureError(
                "displaced entry could not be observed completely after exchange"
            )
        # The final public capture deliberately has no event hook.  Under the frozen one-injected-
        # mutation model this closes the seam after the earlier public observation and prevents a
        # stale snapshot from authorizing acceptance.
        public = capture_entry(dir_fd, prepared.target_name)
        if not _snapshot_secure(public, prepared.ownership):
            raise OwnershipCaptureError("public entry is not the prepared inode after exchange")
        if _snapshot_secure(displaced, expected):
            return MutationReport(
                operation="owned_exchange",
                target_name=prepared.target_name,
                recovery_name=prepared.recovery_name,
                last_observed=LAST_OBSERVED_YES_AFTER_EXCHANGE,
                accepted=True,
                expected_ownership=expected,
                prepared_ownership=prepared.ownership,
                public_entry=public,
                recovery_entry=recovery,
                displaced_entry=displaced,
                events=tuple(events),
                errors=(),
                recovery_uncertainty=False,
                publication_error=False,
            )
    except Exception as error:
        errors.append(_error_text(error))
        public = public if public is not None else observe_entry(dir_fd, prepared.target_name)
        recovery = (
            recovery if recovery is not None else observe_entry(dir_fd, prepared.recovery_name)
        )
        report = MutationReport(
            operation="owned_exchange",
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
            last_observed=(
                LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION
                if exchanged
                else LAST_OBSERVED_NO_BEFORE_MUTATION
            ),
            accepted=False,
            expected_ownership=expected,
            prepared_ownership=prepared.ownership,
            public_entry=public,
            recovery_entry=recovery,
            displaced_entry=displaced,
            events=tuple(events),
            errors=tuple(errors),
            recovery_uncertainty=exchanged,
            publication_error=True,
        )
        raise OwnedMutationError("owned exchange was disrupted", prepared, report) from error

    # A complete post-exchange observation found an unowned displacement.  This is the only
    # branch authorized to mutate the contested names again.
    assert displaced is not None
    try:
        _emit(
            events,
            event_hook,
            "before_rollback_exchange",
            dir_fd=dir_fd,
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
        )
        _rename_exchange(dir_fd, prepared.recovery_name, prepared.target_name)
        _emit(
            events,
            event_hook,
            "after_rollback_exchange",
            dir_fd=dir_fd,
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
        )
        fsync_directory(
            dir_fd,
            event_hook=event_hook,
            reason="owned_exchange_rollback",
            _events=events,
        )
        _emit(
            events,
            event_hook,
            "before_rollback_public_verification",
            dir_fd=dir_fd,
            name=prepared.target_name,
        )
        public = observe_entry(dir_fd, prepared.target_name)
        _emit(
            events,
            event_hook,
            "before_rollback_recovery_verification",
            dir_fd=dir_fd,
            name=prepared.recovery_name,
        )
        recovery = capture_entry(
            dir_fd,
            prepared.recovery_name,
        )
        # The explicit recovery-verification seam can mutate either name.  Re-observe public
        # after that seam before accepting the rollback.
        public = observe_entry(dir_fd, prepared.target_name)
        if not _snapshot_same(public, displaced) or not _snapshot_matches(
            recovery, prepared.ownership
        ):
            raise OwnershipCaptureError("rollback verification did not recover observed entries")
        report = MutationReport(
            operation="owned_exchange",
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
            last_observed=LAST_OBSERVED_NO_AFTER_ROLLBACK,
            accepted=False,
            expected_ownership=expected,
            prepared_ownership=prepared.ownership,
            public_entry=public,
            recovery_entry=recovery,
            displaced_entry=displaced,
            events=tuple(events),
            errors=(
                "displaced entry did not match expected ownership; rollback retained both entries",
            ),
            recovery_uncertainty=False,
            publication_error=True,
        )
        raise OwnedMutationError("owned target changed before exchange", prepared, report)
    except OwnedMutationError:
        raise
    except Exception as error:
        errors.append(_error_text(error))
        public = observe_entry(dir_fd, prepared.target_name)
        recovery = observe_entry(dir_fd, prepared.recovery_name)
        report = MutationReport(
            operation="owned_exchange",
            target_name=prepared.target_name,
            recovery_name=prepared.recovery_name,
            last_observed=LAST_OBSERVED_UNKNOWN_AFTER_DISRUPTION,
            accepted=False,
            expected_ownership=expected,
            prepared_ownership=prepared.ownership,
            public_entry=public,
            recovery_entry=recovery,
            displaced_entry=displaced,
            events=tuple(events),
            errors=tuple(errors),
            recovery_uncertainty=True,
            publication_error=True,
        )
        raise OwnedMutationError("owned rollback was disrupted", prepared, report) from error


def _rename_exchange(dir_fd: int, first_name: str, second_name: str) -> None:
    _validate_name(first_name)
    _validate_name(second_name)
    require_rename_exchange()
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        dir_fd,
        os.fsencode(first_name),
        dir_fd,
        os.fsencode(second_name),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), first_name, second_name)


def _snapshot_matches(snapshot: EntrySnapshot, ownership: Ownership) -> bool:
    return snapshot.ownership is not None and snapshot.ownership.same_identity_and_hash(ownership)


def _snapshot_exact(snapshot: EntrySnapshot, ownership: Ownership) -> bool:
    return snapshot.ownership is not None and snapshot.ownership == ownership


def _snapshot_secure(snapshot: EntrySnapshot, ownership: Ownership) -> bool:
    return bool(
        snapshot.ownership is not None
        and snapshot.ownership.same_identity_and_hash(ownership)
        and snapshot.ownership.size == ownership.size
        and stat.S_ISREG(snapshot.ownership.mode)
        and stat.S_IMODE(snapshot.ownership.mode) == 0o600
    )


def _snapshot_same(first: EntrySnapshot, second: EntrySnapshot) -> bool:
    if first.ownership is not None and second.ownership is not None:
        return first.ownership.same_identity_and_hash(second.ownership)
    return (
        first.state == second.state
        and first.device is not None
        and first.inode is not None
        and first.mode is not None
        and (first.device, first.inode, first.mode) == (second.device, second.inode, second.mode)
    )


def _stable_metadata(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _walk_json(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _walk_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_json(item)
    else:
        yield value


def _validate_kind(kind: str) -> None:
    if (
        not isinstance(kind, str)
        or not kind
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in kind)
    ):
        raise ReceiptDomainError(
            "receipt kind must contain only lowercase ASCII letters, digits, and underscores"
        )


def _validate_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError("transaction names must be single non-special path components")
    if len(os.fsencode(name)) > 255:
        raise ValueError("transaction name exceeds the portable directory-entry limit")


def _emit(
    events: list[str] | None,
    hook: EventHook | None,
    event: str,
    **context: object,
) -> None:
    if events is not None:
        events.append(event)
    if hook is not None:
        hook(event, context)


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"
