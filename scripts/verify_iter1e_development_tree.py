#!/usr/bin/env python3
"""Read-only verifier for the inverse-projection-fiber iter1e development tree.

The scanner deliberately receives forbidden byte strings from its caller.  It never embeds or
echoes official namespaces, schemas, roots, or lifecycle literals.  Paths are reported as hex
encoded filesystem bytes so a forbidden path component cannot copy itself into the receipt.

Run:
    python scripts/verify_iter1e_development_tree.py \
        --base PATH \
        --forbidden-needle TEXT [--forbidden-needle TEXT ...]
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from typing import Any

SCHEMA = "inverse_projection_fiber_development_iter1e_tree_scan_v1"
NAMESPACE = "rtgs.inverse-projection-fiber.development.iter1e.v1"
PROTOCOL_LABEL = "inverse-projection-fiber-iteration-1e"
RECEIPT_DOMAIN = "development"
ROOT_CONSUMPTION_STATUS = "DEVELOPMENT_ONLY"
READ_SIZE = 1 << 20


def canonical_bytes(value: Any) -> bytes:
    """Encode one receipt deterministically without accepting non-finite numbers."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _error(code: str, relative_path: bytes | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"code": code}
    if relative_path is not None:
        record["path_hex"] = relative_path.hex()
    return record


def _match_record(
    *,
    kind: str,
    relative_path: bytes,
    needle_index: int,
    needle: bytes,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "needle_index": needle_index,
        "needle_sha256": _sha256(needle),
        "path_hex": relative_path.hex(),
    }


def _scan_regular_file(
    path: bytes,
    relative_path: bytes,
    needles: tuple[bytes, ...],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None, [], _error("FILE_OPEN_FAILED", relative_path)

    digest = hashlib.sha256()
    matches: list[dict[str, Any]] = []
    found = [False] * len(needles)
    longest = max((len(needle) for needle in needles), default=1)
    carry = b""
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return None, [], _error("FILE_CHANGED_TO_NONREGULAR", relative_path)
        while True:
            block = os.read(descriptor, READ_SIZE)
            if not block:
                break
            digest.update(block)
            total += len(block)
            window = carry + block
            for index, needle in enumerate(needles):
                if not found[index] and needle in window:
                    found[index] = True
                    matches.append(
                        _match_record(
                            kind="file_bytes",
                            relative_path=relative_path,
                            needle_index=index,
                            needle=needle,
                        )
                    )
            carry = window[-(longest - 1) :] if longest > 1 else b""
        after = os.fstat(descriptor)
    except OSError:
        return None, [], _error("FILE_READ_FAILED", relative_path)
    finally:
        os.close(descriptor)

    stable_fields = (
        before.st_dev == after.st_dev,
        before.st_ino == after.st_ino,
        before.st_mode == after.st_mode,
        before.st_size == after.st_size,
        before.st_mtime_ns == after.st_mtime_ns,
        total == after.st_size,
    )
    if not all(stable_fields):
        return None, [], _error("FILE_CHANGED_DURING_SCAN", relative_path)
    try:
        current = os.lstat(path)
    except OSError:
        return None, [], _error("FILE_PATH_UNAVAILABLE_AFTER_SCAN", relative_path)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != after.st_dev
        or current.st_ino != after.st_ino
    ):
        return None, [], _error("FILE_PATH_CHANGED_DURING_SCAN", relative_path)

    return (
        {
            "path_hex": relative_path.hex(),
            "sha256": digest.hexdigest(),
            "size_bytes": total,
        },
        matches,
        None,
    )


def scan_tree(
    base: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    needles: list[bytes],
) -> dict:
    """Recursively scan *base* without following links or writing any artifacts."""
    normalized_needles = tuple(bytes(needle) for needle in needles)
    if not normalized_needles or any(not needle for needle in normalized_needles):
        raise ValueError("at least one non-empty forbidden needle is required")

    base_bytes = os.path.abspath(os.fsencode(base))
    needle_receipts = [
        {"index": index, "length_bytes": len(needle), "sha256": _sha256(needle)}
        for index, needle in enumerate(normalized_needles)
    ]
    files: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    directory_count = 0
    path_name_count = 0

    try:
        base_stat = os.lstat(base_bytes)
    except OSError:
        errors.append(_error("BASE_UNAVAILABLE"))
        base_stat = None
    if base_stat is not None:
        if stat.S_ISLNK(base_stat.st_mode):
            errors.append(_error("BASE_IS_SYMLINK"))
        elif not stat.S_ISDIR(base_stat.st_mode):
            errors.append(_error("BASE_IS_NOT_DIRECTORY"))

    pending: list[tuple[bytes, bytes]] = []
    if not errors:
        pending.append((base_bytes, b""))

    while pending:
        directory, relative_directory = pending.pop()
        directory_count += 1
        try:
            with os.scandir(directory) as iterator:
                names = sorted(entry.name for entry in iterator)
        except OSError:
            errors.append(_error("DIRECTORY_SCAN_FAILED", relative_directory))
            continue

        child_directories: list[tuple[bytes, bytes]] = []
        for name in names:
            path_name_count += 1
            relative_path = name if not relative_directory else relative_directory + b"/" + name
            child_path = os.path.join(directory, name)
            for index, needle in enumerate(normalized_needles):
                if needle in name:
                    matches.append(
                        _match_record(
                            kind="path_name",
                            relative_path=relative_path,
                            needle_index=index,
                            needle=needle,
                        )
                    )
            try:
                child_stat = os.lstat(child_path)
            except OSError:
                errors.append(_error("PATH_STAT_FAILED", relative_path))
                continue
            mode = child_stat.st_mode
            if stat.S_ISLNK(mode):
                errors.append(_error("SYMLINK_FORBIDDEN", relative_path))
            elif stat.S_ISDIR(mode):
                child_directories.append((child_path, relative_path))
            elif stat.S_ISREG(mode):
                file_record, file_matches, file_error = _scan_regular_file(
                    child_path,
                    relative_path,
                    normalized_needles,
                )
                if file_record is not None:
                    files.append(file_record)
                matches.extend(file_matches)
                if file_error is not None:
                    errors.append(file_error)
            else:
                errors.append(_error("NONREGULAR_FORBIDDEN", relative_path))
        pending.extend(reversed(child_directories))

    files.sort(key=lambda record: record["path_hex"])
    matches.sort(
        key=lambda record: (
            record["path_hex"],
            record["kind"],
            record["needle_index"],
        )
    )
    errors.sort(key=lambda record: (record.get("path_hex", ""), record["code"]))
    scan_complete = not errors
    status = "PASS" if scan_complete and not matches else "FAIL"
    return {
        "namespace": NAMESPACE,
        "protocol_label": PROTOCOL_LABEL,
        "receipt_domain": RECEIPT_DOMAIN,
        "root_consumption_status": ROOT_CONSUMPTION_STATUS,
        "roots": [],
        "schema": SCHEMA,
        "status": status,
        "scan": {
            "base_path_sha256": _sha256(base_bytes),
            "directory_count": directory_count,
            "errors": errors,
            "file_count": len(files),
            "files": files,
            "forbidden_match_count": len(matches),
            "forbidden_matches": matches,
            "forbidden_needles": needle_receipts,
            "path_name_count": path_name_count,
            "regular_files_sha256": _sha256(canonical_bytes(files)),
            "scan_complete": scan_complete,
        },
    }


def _parse_arguments(arguments: list[str]) -> tuple[str, list[bytes]]:
    base: str | None = None
    needles: list[bytes] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument not in {"--base", "--forbidden-needle"} or index + 1 >= len(arguments):
            raise ValueError("invalid arguments")
        value = arguments[index + 1]
        if argument == "--base":
            if base is not None:
                raise ValueError("base supplied more than once")
            base = value
        else:
            needles.append(os.fsencode(value))
        index += 2
    if base is None:
        raise ValueError("base is required")
    if not needles:
        raise ValueError("at least one forbidden needle is required")
    return base, needles


def _argument_failure() -> dict[str, Any]:
    return {
        "namespace": NAMESPACE,
        "protocol_label": PROTOCOL_LABEL,
        "receipt_domain": RECEIPT_DOMAIN,
        "root_consumption_status": ROOT_CONSUMPTION_STATUS,
        "roots": [],
        "schema": SCHEMA,
        "status": "FAIL",
        "scan": {
            "errors": [{"code": "INVALID_ARGUMENTS"}],
            "scan_complete": False,
        },
    }


def main(arguments: list[str] | None = None) -> int:
    """Run the read-only scan and emit exactly one canonical JSON object."""
    try:
        base, needles = _parse_arguments(sys.argv[1:] if arguments is None else arguments)
        receipt = scan_tree(base, needles)
    except (OSError, TypeError, ValueError):
        receipt = _argument_failure()
    sys.stdout.buffer.write(canonical_bytes(receipt) + b"\n")
    sys.stdout.buffer.flush()
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
