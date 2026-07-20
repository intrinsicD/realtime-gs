"""Iteration 2 receipt domain and narrow wrappers over the fault-tested JSON engine."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks import inverse_projection_fiber_transaction as tx

NAMESPACE = "rtgs.inverse-projection-fiber.iter2.v1"
PROTOCOL_LABEL = "inverse-projection-fiber-iter2"
SCHEMA_FAMILY = "inverse_projection_fiber_iter2"
SCENE_ROOTS = (27_688_011, 27_688_012, 27_688_013)
DEPTH_ROOTS = (27_688_111, 27_688_112, 27_688_113)
ORDER_ROOTS = (27_688_211, 27_688_212, 27_688_213)
OFFICIAL_ROOTS = tuple(
    value for bundle in zip(SCENE_ROOTS, DEPTH_ROOTS, ORDER_ROOTS, strict=True) for value in bundle
)
RESERVED_UNCONSUMED = "RESERVED_UNCONSUMED"
CONSUMPTION_STARTED = "STARTED_CONSERVATIVELY_CONSUMED"
PARTIALLY_CONSUMED = "PARTIALLY_CONSUMED"
CONSUMED = "CONSUMED"
ABORTED_UNKNOWN = "ABORTED_UNKNOWN"
MAX_RECEIPT_BYTES = 64 * 1024 * 1024


def official_domain() -> tx.ReceiptDomain:
    """Return the immutable official Iteration 2 receipt domain."""
    return tx.ReceiptDomain(
        protocol_label=PROTOCOL_LABEL,
        label="official",
        namespace=NAMESPACE,
        schema_family=SCHEMA_FAMILY,
        permitted_root_consumption_statuses=(
            RESERVED_UNCONSUMED,
            CONSUMPTION_STARTED,
            PARTIALLY_CONSUMED,
            CONSUMED,
            ABORTED_UNKNOWN,
        ),
        permitted_roots=OFFICIAL_ROOTS,
        official_phases_permitted=True,
        commit_states_permitted=True,
        protocol_generation="iter2",
    )


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _require_receipt_size(receipt: dict[str, Any]) -> None:
    if len(tx.canonical_json_bytes(receipt)) > MAX_RECEIPT_BYTES:
        raise ValueError("canonical receipt exceeds the receipt-size bound")


def ownership_dict(value: tx.Ownership) -> dict[str, Any]:
    return _jsonable(asdict(value))


def ownership_from_dict(value: dict[str, Any]) -> tx.Ownership:
    required = {"sha256", "device", "inode", "size", "mode", "mtime_ns", "ctime_ns"}
    if type(value) is not dict:
        raise ValueError("ownership descriptor must be an exact dictionary")
    if set(value) != required:
        raise ValueError(f"ownership descriptor keys differ: {sorted(map(repr, value))}")
    sha256 = value["sha256"]
    if (
        type(sha256) is not str
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("ownership sha256 must be 64 lowercase hexadecimal characters")
    integer_fields = ("device", "inode", "size", "mode", "mtime_ns", "ctime_ns")
    if any(type(value[field]) is not int for field in integer_fields):
        raise ValueError("ownership integer fields must be exact JSON integers")
    if any(value[field] < 0 for field in integer_fields):
        raise ValueError("ownership integer fields must be non-negative")
    ownership = tx.Ownership(
        sha256=sha256,
        device=value["device"],
        inode=value["inode"],
        size=value["size"],
        mode=value["mode"],
        mtime_ns=value["mtime_ns"],
        ctime_ns=value["ctime_ns"],
    )
    if not stat.S_ISREG(ownership.mode):
        raise ValueError("ownership mode must describe a regular file")
    if stat.S_IMODE(ownership.mode) != 0o600:
        raise ValueError("receipt ownership mode must be exactly private 0600")
    return ownership


def _directory_descriptor(directory_fd: int) -> dict[str, int]:
    metadata = os.fstat(directory_fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(directory_fd)
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }


def _directory_from_dict(value: dict[str, Any]) -> dict[str, int]:
    required = {"device", "inode"}
    if type(value) is not dict:
        raise ValueError("directory descriptor must be an exact dictionary")
    if set(value) != required:
        raise ValueError(f"directory descriptor keys differ: {sorted(map(repr, value))}")
    if any(type(value[field]) is not int or value[field] < 0 for field in required):
        raise ValueError("directory descriptor fields must be non-negative exact integers")
    return {field: value[field] for field in ("device", "inode")}


def snapshot_descriptor(
    value: tx.EntrySnapshot,
    *,
    path: Path,
    directory: dict[str, int],
) -> dict[str, Any]:
    if not value.is_regular or value.ownership is None:
        raise tx.OwnershipCaptureError(f"entry is not a captured regular file: {value}")
    return {
        "path_hint": str(path),
        "name": value.name,
        "directory": _directory_from_dict(directory),
        "ownership": ownership_dict(value.ownership),
    }


def prepared_descriptor(
    value: tx.PreparedJSON,
    *,
    directory: dict[str, int],
) -> dict[str, Any]:
    return {
        "receipt_kind": value.receipt_kind,
        "target_name": value.target_name,
        "recovery_name": value.recovery_name,
        "ownership": ownership_dict(value.ownership),
        "payload_sha256": value.payload_sha256,
        "payload_size": value.payload_size,
        "directory": _directory_from_dict(directory),
    }


def _component_name(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\x00" in value
        or len(os.fsencode(value)) > 255
    ):
        raise ValueError(f"{label} must be one non-empty directory-entry name")
    return value


def prepared_from_descriptor(value: dict[str, Any]) -> tx.PreparedJSON:
    """Strictly reconstruct a prepared entry without trusting JSON coercions."""
    required = {
        "receipt_kind",
        "target_name",
        "recovery_name",
        "ownership",
        "payload_sha256",
        "payload_size",
        "directory",
    }
    if type(value) is not dict:
        raise ValueError("prepared descriptor must be an exact dictionary")
    if set(value) != required:
        raise ValueError(f"prepared descriptor keys differ: {sorted(map(repr, value))}")
    receipt_kind = _component_name(value["receipt_kind"], label="receipt_kind")
    target_name = _component_name(value["target_name"], label="target_name")
    recovery_name = _component_name(value["recovery_name"], label="recovery_name")
    ownership = ownership_from_dict(value["ownership"])
    _directory_from_dict(value["directory"])
    payload_sha256 = value["payload_sha256"]
    payload_size = value["payload_size"]
    if (
        type(payload_sha256) is not str
        or len(payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise ValueError("payload_sha256 must be 64 lowercase hexadecimal characters")
    if type(payload_size) is not int or payload_size < 0:
        raise ValueError("payload_size must be a non-negative exact JSON integer")
    if payload_size > MAX_RECEIPT_BYTES:
        raise ValueError("prepared payload exceeds the receipt-size bound")
    if payload_sha256 != ownership.sha256 or payload_size != ownership.size:
        raise ValueError("prepared payload metadata differs from captured ownership")
    prefix = f".{target_name}.recovery."
    suffix = ".prepared"
    if not recovery_name.startswith(prefix) or not recovery_name.endswith(suffix):
        raise ValueError("recovery_name does not belong to target_name")
    nonce = recovery_name[len(prefix) : -len(suffix)]
    if len(nonce) != 32 or any(character not in "0123456789abcdef" for character in nonce):
        raise ValueError("recovery_name nonce is not 128 lowercase hexadecimal bits")
    domain = official_domain()
    domain.schema(receipt_kind)
    return tx.PreparedJSON(
        domain=domain,
        receipt_kind=receipt_kind,
        target_name=target_name,
        recovery_name=recovery_name,
        ownership=ownership,
        payload_sha256=payload_sha256,
        payload_size=payload_size,
    )


def _directory_handle(directory: Path, directory_fd: int | None) -> tuple[int, dict[str, int]]:
    """Return an owned descriptor, duplicating a caller-held identity when supplied."""
    if directory_fd is None:
        descriptor = tx.open_directory(directory)
        return descriptor, _directory_descriptor(descriptor)
    if type(directory_fd) is not int or directory_fd < 0:
        raise ValueError("directory_fd must be a non-negative exact integer")
    descriptor = os.dup(directory_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise NotADirectoryError(directory)
    return descriptor, _directory_descriptor(descriptor)


def mutation_descriptor(value: tx.MutationReport) -> dict[str, Any]:
    return _jsonable(asdict(value))


def descriptor_matches_snapshot(
    descriptor: dict[str, Any],
    snapshot: tx.EntrySnapshot,
    *,
    expected_path: Path,
    directory: dict[str, int],
) -> bool:
    if not snapshot.is_regular or snapshot.ownership is None:
        return False
    if type(descriptor) is not dict or set(descriptor) != {
        "path_hint",
        "name",
        "directory",
        "ownership",
    }:
        return False
    if (
        descriptor.get("path_hint") != str(expected_path)
        or descriptor.get("name") != expected_path.name
    ):
        return False
    try:
        expected = ownership_from_dict(descriptor["ownership"])
        expected_directory = _directory_from_dict(descriptor["directory"])
    except Exception:
        return False
    return snapshot.ownership == expected and expected_directory == directory


def publish_receipt(
    directory: Path,
    target_name: str,
    receipt_kind: str,
    receipt: dict[str, Any],
    *,
    nonce: str,
    directory_fd: int | None = None,
    event_hook: tx.EventHook | None = None,
) -> dict[str, Any]:
    """Prepare, exclusively publish, and stably recapture an absent receipt path."""
    domain = official_domain()
    domain.validate_receipt(receipt_kind, receipt)
    _require_receipt_size(receipt)
    owned_directory_fd, directory_identity = _directory_handle(directory, directory_fd)
    try:
        prepared = tx.prepare_json(
            owned_directory_fd,
            target_name,
            domain,
            receipt_kind,
            receipt,
            nonce=nonce,
            event_hook=event_hook,
        )
        mutation = tx.publish_exclusive(owned_directory_fd, prepared, event_hook=event_hook)
        payload, public = tx.capture_entry_bytes(
            owned_directory_fd,
            target_name,
            expected_sha256=prepared.payload_sha256,
            expected_size=prepared.payload_size,
            max_bytes=MAX_RECEIPT_BYTES,
        )
    finally:
        os.close(owned_directory_fd)
    if payload != tx.canonical_json_bytes(receipt):
        raise tx.OwnershipCaptureError("published receipt bytes are not canonical expected bytes")
    return {
        "prepared": prepared_descriptor(prepared, directory=directory_identity),
        "mutation": mutation_descriptor(mutation),
        "public": snapshot_descriptor(
            public,
            path=directory / target_name,
            directory=directory_identity,
        ),
    }


def exchange_receipt(
    directory: Path,
    target_name: str,
    receipt_kind: str,
    receipt: dict[str, Any],
    *,
    expected_public: dict[str, Any],
    nonce: str,
    directory_fd: int | None = None,
    event_hook: tx.EventHook | None = None,
) -> dict[str, Any]:
    """Prepare and exception-safely exchange one currently owned receipt."""
    domain = official_domain()
    domain.validate_receipt(receipt_kind, receipt)
    _require_receipt_size(receipt)
    owned_directory_fd, directory_identity = _directory_handle(directory, directory_fd)
    try:
        before_payload, before = tx.capture_entry_bytes(
            owned_directory_fd,
            target_name,
            max_bytes=MAX_RECEIPT_BYTES,
        )
        del before_payload
        if not descriptor_matches_snapshot(
            expected_public,
            before,
            expected_path=directory / target_name,
            directory=directory_identity,
        ):
            raise tx.OwnershipCaptureError("public receipt ownership changed before exchange")
        assert before.ownership is not None
        prepared = tx.prepare_json(
            owned_directory_fd,
            target_name,
            domain,
            receipt_kind,
            receipt,
            nonce=nonce,
            event_hook=event_hook,
        )
        mutation = tx.exchange_owned(
            owned_directory_fd,
            prepared,
            before.ownership,
            event_hook=event_hook,
        )
        payload, public = tx.capture_entry_bytes(
            owned_directory_fd,
            target_name,
            expected_sha256=prepared.payload_sha256,
            expected_size=prepared.payload_size,
            max_bytes=MAX_RECEIPT_BYTES,
        )
    finally:
        os.close(owned_directory_fd)
    if payload != tx.canonical_json_bytes(receipt):
        raise tx.OwnershipCaptureError("exchanged receipt bytes are not canonical expected bytes")
    return {
        "prepared": prepared_descriptor(prepared, directory=directory_identity),
        "mutation": mutation_descriptor(mutation),
        "public": snapshot_descriptor(
            public,
            path=directory / target_name,
            directory=directory_identity,
        ),
    }


def capture_receipt(
    directory: Path,
    target_name: str,
    receipt_kind: str,
    *,
    expected_public: dict[str, Any] | None = None,
    directory_fd: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Same-FD capture, canonical parse, domain validation, and optional ownership check."""
    owned_directory_fd, directory_identity = _directory_handle(directory, directory_fd)
    try:
        payload, snapshot = tx.capture_entry_bytes(
            owned_directory_fd,
            target_name,
            max_bytes=MAX_RECEIPT_BYTES,
        )
    finally:
        os.close(owned_directory_fd)
    if expected_public is not None and not descriptor_matches_snapshot(
        expected_public,
        snapshot,
        expected_path=directory / target_name,
        directory=directory_identity,
    ):
        raise tx.OwnershipCaptureError("captured receipt differs from expected public ownership")
    receipt = json.loads(payload)
    if payload != tx.canonical_json_bytes(receipt):
        raise ValueError("receipt is not canonical finite JSON")
    official_domain().validate_receipt(receipt_kind, receipt)
    return receipt, snapshot_descriptor(
        snapshot,
        path=directory / target_name,
        directory=directory_identity,
    )


def prepare_receipt(
    directory: Path,
    target_name: str,
    receipt_kind: str,
    receipt: dict[str, Any],
    *,
    nonce: str,
    directory_fd: int | None = None,
    event_hook: tx.EventHook | None = None,
) -> dict[str, Any]:
    """Durably prepare a recovery entry without mutating its public target."""
    domain = official_domain()
    domain.validate_receipt(receipt_kind, receipt)
    _require_receipt_size(receipt)
    owned_directory_fd, directory_identity = _directory_handle(directory, directory_fd)
    try:
        prepared = tx.prepare_json(
            owned_directory_fd,
            target_name,
            domain,
            receipt_kind,
            receipt,
            nonce=nonce,
            event_hook=event_hook,
        )
    finally:
        os.close(owned_directory_fd)
    return prepared_descriptor(prepared, directory=directory_identity)


def capture_prepared(
    directory: Path,
    descriptor: dict[str, Any],
    *,
    expected_target_name: str,
    expected_receipt_kind: str,
    expected_transaction_id: str,
    expected_root_consumption_status: str,
    expected_roots: tuple[int, ...],
    expected_official_phase: str | None,
    expected_commit_state: str | None,
    directory_fd: int | None = None,
) -> tuple[dict[str, Any], tx.PreparedJSON, dict[str, Any]]:
    """Reopen and validate a worker-prepared recovery entry through one held directory."""
    prepared = prepared_from_descriptor(descriptor)
    if prepared.target_name != _component_name(
        expected_target_name,
        label="expected_target_name",
    ):
        raise tx.OwnershipCaptureError("prepared target differs from launcher intent")
    if prepared.receipt_kind != _component_name(
        expected_receipt_kind,
        label="expected_receipt_kind",
    ):
        raise tx.OwnershipCaptureError("prepared receipt kind differs from launcher intent")
    if (
        type(expected_transaction_id) is not str
        or len(expected_transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in expected_transaction_id)
    ):
        raise ValueError("expected_transaction_id must be 128 lowercase hexadecimal bits")
    if (
        expected_root_consumption_status
        not in official_domain().permitted_root_consumption_statuses
    ):
        raise ValueError("expected_root_consumption_status is outside the official domain")
    if (
        type(expected_roots) is not tuple
        or any(type(root) is not int for root in expected_roots)
        or len(set(expected_roots)) != len(expected_roots)
    ):
        raise ValueError("expected_roots must be a unique exact-integer tuple")
    owned_directory_fd, directory_identity = _directory_handle(directory, directory_fd)
    if _directory_from_dict(descriptor["directory"]) != directory_identity:
        os.close(owned_directory_fd)
        raise tx.OwnershipCaptureError("prepared recovery directory differs from handoff")
    try:
        payload, snapshot = tx.capture_entry_bytes(
            owned_directory_fd,
            prepared.recovery_name,
            expected_sha256=prepared.payload_sha256,
            expected_size=prepared.payload_size,
            max_bytes=MAX_RECEIPT_BYTES,
        )
    finally:
        os.close(owned_directory_fd)
    if not snapshot.is_regular or snapshot.ownership is None:
        raise tx.OwnershipCaptureError("prepared recovery entry is not a regular file")
    if snapshot.ownership != prepared.ownership:
        raise tx.OwnershipCaptureError("prepared recovery ownership differs from handoff")
    if len(payload) != prepared.payload_size:
        raise tx.OwnershipCaptureError("prepared recovery size differs from handoff")
    receipt = json.loads(payload)
    if payload != tx.canonical_json_bytes(receipt):
        raise ValueError("prepared receipt is not canonical finite JSON")
    prepared.domain.validate_receipt(prepared.receipt_kind, receipt)
    if receipt.get("transaction_id") != expected_transaction_id:
        raise tx.OwnershipCaptureError("prepared transaction differs from launcher intent")
    if receipt.get("root_consumption_status") != expected_root_consumption_status:
        raise tx.OwnershipCaptureError("prepared root status differs from launcher intent")
    if receipt.get("roots") != list(expected_roots):
        raise tx.OwnershipCaptureError("prepared roots differ from launcher intent")
    for key, expected in (
        ("official_phase", expected_official_phase),
        ("commit_state", expected_commit_state),
    ):
        if expected is None:
            if key in receipt:
                raise tx.OwnershipCaptureError(
                    f"prepared {key} presence differs from launcher intent"
                )
        elif type(expected) is not str or not expected:
            raise ValueError(f"expected_{key} must be None or a non-empty string")
        elif receipt.get(key) != expected:
            raise tx.OwnershipCaptureError(f"prepared {key} differs from launcher intent")
    return (
        receipt,
        prepared,
        snapshot_descriptor(
            snapshot,
            path=directory / prepared.recovery_name,
            directory=directory_identity,
        ),
    )


def exchange_prepared_receipt(
    directory: Path,
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
    directory_fd: int | None = None,
    event_hook: tx.EventHook | None = None,
) -> dict[str, Any]:
    """Parent-owned exchange of an intent-bound, worker-prepared receipt."""
    owned_directory_fd, directory_identity = _directory_handle(directory, directory_fd)
    try:
        receipt, prepared, _recovery = capture_prepared(
            directory,
            descriptor,
            expected_target_name=expected_target_name,
            expected_receipt_kind=expected_receipt_kind,
            expected_transaction_id=expected_transaction_id,
            expected_root_consumption_status=expected_root_consumption_status,
            expected_roots=expected_roots,
            expected_official_phase=expected_official_phase,
            expected_commit_state=expected_commit_state,
            directory_fd=owned_directory_fd,
        )
        _before_payload, before = tx.capture_entry_bytes(
            owned_directory_fd,
            prepared.target_name,
            max_bytes=MAX_RECEIPT_BYTES,
        )
        if not descriptor_matches_snapshot(
            expected_public,
            before,
            expected_path=directory / prepared.target_name,
            directory=directory_identity,
        ):
            raise tx.OwnershipCaptureError("public reservation changed before prepared exchange")
        assert before.ownership is not None
        mutation = tx.exchange_owned(
            owned_directory_fd,
            prepared,
            before.ownership,
            event_hook=event_hook,
        )
        payload, public = tx.capture_entry_bytes(
            owned_directory_fd,
            prepared.target_name,
            expected_sha256=prepared.payload_sha256,
            expected_size=prepared.payload_size,
            max_bytes=MAX_RECEIPT_BYTES,
        )
    finally:
        os.close(owned_directory_fd)
    if payload != tx.canonical_json_bytes(receipt):
        raise tx.OwnershipCaptureError("public result bytes differ from prepared receipt")
    return {
        "prepared": prepared_descriptor(prepared, directory=directory_identity),
        "mutation": mutation_descriptor(mutation),
        "public": snapshot_descriptor(
            public,
            path=directory / prepared.target_name,
            directory=directory_identity,
        ),
    }
