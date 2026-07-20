#!/usr/bin/env python3
"""Execute and attest the exact fresh-base Iteration 1e development verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks import inverse_projection_fiber_iter1e as iteration
from benchmarks import inverse_projection_fiber_protocol as protocol
from benchmarks.inverse_projection_fiber_transaction import (
    DEVELOPMENT_ONLY,
    canonical_json_bytes,
    open_directory,
    prepare_json,
    publish_exclusive,
)

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / iteration.SPEC.verification_scanner
FOCUSED_TESTS = tuple(path.as_posix() for path in iteration.SPEC.verification_test_paths)
_PASS_SUMMARY = re.compile(rb"(?:^|\n)([1-9][0-9]*) passed in [^\n]+\n?\Z")
_COLLECTION_SUMMARY = re.compile(rb"(?:^|\n)([1-9][0-9]*) tests collected in [^\n]+\n?\Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _python_executable() -> Path:
    executable = Path(sys.executable).absolute()
    if not executable.is_file():
        raise RuntimeError("current Python executable is not a file")
    return executable


def _scan_canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _forbidden_needles() -> tuple[bytes, ...]:
    values = (
        iteration.OFFICIAL_NAMESPACE,
        iteration.OFFICIAL_SCHEMA_FAMILY,
        *(str(root) for root in iteration.OFFICIAL_ROOTS),
        *iteration.OFFICIAL_ROOT_STATUSES,
    )
    encoded = tuple(value.encode("utf-8") for value in values)
    if len(encoded) != 11 or len(set(encoded)) != len(encoded):
        raise RuntimeError("frozen forbidden-needle manifest is not exactly eleven unique values")
    return encoded


def _forbidden_manifest() -> list[dict[str, Any]]:
    return [
        {"index": index, "length_bytes": len(needle), "sha256": _sha256(needle)}
        for index, needle in enumerate(_forbidden_needles())
    ]


def _pytest_command(base: Path) -> list[str]:
    return [
        str(_python_executable()),
        *protocol._verification_pytest_args(iteration.SPEC, base, collect_only=False),
    ]


def _collection_command(base: Path) -> list[str]:
    return [
        str(_python_executable()),
        *protocol._verification_pytest_args(iteration.SPEC, base, collect_only=True),
    ]


def _scanner_command(base: Path) -> list[str]:
    command = [str(_python_executable()), str(SCANNER), "--base", str(base)]
    for needle in _forbidden_needles():
        command.extend(["--forbidden-needle", needle.decode("utf-8")])
    return command


def _parse_pytest_pass_count(completed: subprocess.CompletedProcess[bytes]) -> int:
    if completed.returncode != 0 or completed.stderr != b"":
        raise RuntimeError("focused pytest did not exit cleanly with empty stderr")
    matched = _PASS_SUMMARY.search(completed.stdout)
    if matched is None:
        raise RuntimeError("focused pytest summary is not an all-passed terminal summary")
    return int(matched.group(1))


def _parse_pytest_collection(
    completed: subprocess.CompletedProcess[bytes],
) -> tuple[str, ...]:
    if completed.returncode != 0 or completed.stderr != b"":
        raise RuntimeError("focused pytest collection did not exit cleanly with empty stderr")
    matched = _COLLECTION_SUMMARY.search(completed.stdout)
    if matched is None:
        raise RuntimeError("focused pytest collection has no exact terminal summary")
    lines = completed.stdout.splitlines()
    if len(lines) < 3 or lines[-2] != b"":
        raise RuntimeError("focused pytest collection output has an unexpected framing")
    try:
        nodeids = tuple(line.decode("utf-8") for line in lines[:-2])
    except UnicodeDecodeError as error:
        raise RuntimeError("focused pytest collection contains a non-UTF-8 node id") from error
    if len(nodeids) != int(matched.group(1)):
        raise RuntimeError("focused pytest collection count disagrees with its node-id manifest")
    protocol._validate_collected_nodeids(iteration.SPEC, list(nodeids))
    return nodeids


def _validate_scan(scan: dict[str, Any], base: Path) -> None:
    iteration.DEVELOPMENT_DOMAIN.validate_receipt("tree_scan", scan)
    expected_base_hash = _sha256(os.fsencode(str(base)))
    scan_body = scan.get("scan")
    if (
        scan.get("schema") != iteration.DEVELOPMENT_DOMAIN.schema("tree_scan")
        or scan.get("namespace") != iteration.DEVELOPMENT_NAMESPACE
        or scan.get("root_consumption_status") != DEVELOPMENT_ONLY
        or scan.get("status") != "PASS"
        or not isinstance(scan_body, dict)
        or scan_body.get("base_path_sha256") != expected_base_hash
        or scan_body.get("scan_complete") is not True
        or scan_body.get("errors") != []
        or scan_body.get("forbidden_match_count") != 0
        or scan_body.get("forbidden_matches") != []
        or scan_body.get("forbidden_needles") != _forbidden_manifest()
    ):
        raise RuntimeError("development-tree scan is not a clean exact-manifest PASS")

    files = scan_body.get("files")
    file_count = scan_body.get("file_count")
    directory_count = scan_body.get("directory_count")
    path_name_count = scan_body.get("path_name_count")
    if (
        not isinstance(files, list)
        or type(file_count) is not int
        or file_count != len(files)
        or type(directory_count) is not int
        or directory_count <= 0
        or type(path_name_count) is not int
        or path_name_count < file_count
    ):
        raise RuntimeError("development-tree scan counts are internally inconsistent")
    path_hex_values: list[str] = []
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path_hex", "sha256", "size_bytes"}:
            raise RuntimeError("development-tree scan contains a malformed file record")
        path_hex = record["path_hex"]
        digest = record["sha256"]
        size = record["size_bytes"]
        if (
            not isinstance(path_hex, str)
            or len(path_hex) % 2 != 0
            or any(character not in "0123456789abcdef" for character in path_hex)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(size) is not int
            or size < 0
        ):
            raise RuntimeError("development-tree scan contains an invalid file record")
        path_hex_values.append(path_hex)
    if path_hex_values != sorted(path_hex_values) or len(set(path_hex_values)) != len(
        path_hex_values
    ):
        raise RuntimeError("development-tree scan paths are not unique and sorted")
    expected_files_hash = _sha256(_scan_canonical_bytes(files))
    if scan_body.get("regular_files_sha256") != expected_files_hash:
        raise RuntimeError("development-tree scan aggregate file hash is inconsistent")


def _decode_scanner_output(
    completed: subprocess.CompletedProcess[bytes],
    base: Path,
) -> dict[str, Any]:
    if completed.returncode != 0 or completed.stderr != b"":
        raise RuntimeError("development-tree scanner did not exit cleanly")
    scan = json.loads(completed.stdout)
    if completed.stdout != canonical_json_bytes(scan):
        raise RuntimeError("scanner output is not exactly one canonical JSON line")
    if not isinstance(scan, dict):
        raise RuntimeError("scanner output is not a JSON object")
    _validate_scan(scan, base)
    return scan


def _build_verification_receipt(
    *,
    base: Path,
    collection_command: list[str],
    collection_completed: subprocess.CompletedProcess[bytes],
    collection_nodeids: tuple[str, ...],
    pytest_command: list[str],
    pytest_completed: subprocess.CompletedProcess[bytes],
    pytest_test_count: int,
    scanner_completed: subprocess.CompletedProcess[bytes],
    scan: dict[str, Any],
    environment: dict[str, str],
    source_manifest: list[dict[str, str]],
) -> dict[str, Any]:
    scan_body = scan["scan"]
    return iteration.DEVELOPMENT_DOMAIN.make_receipt(
        "verification",
        {
            "status": "PASS",
            "verification_base": str(base),
            "official_roots_used": False,
            "development_roots_used": list(iteration.SPEC.development_roots),
            "pytest_status": "PASS",
            "pytest_returncode": pytest_completed.returncode,
            "pytest_test_count": pytest_test_count,
            "collection_status": "PASS",
            "collection_returncode": collection_completed.returncode,
            "collection_test_count": len(collection_nodeids),
            "collection_command": collection_command,
            "collection_nodeids": list(collection_nodeids),
            "collection_nodeids_sha256": _sha256(_scan_canonical_bytes(list(collection_nodeids))),
            "collection_stdout_sha256": _sha256(collection_completed.stdout),
            "collection_stderr_sha256": _sha256(collection_completed.stderr),
            "focused_pytest_command": pytest_command,
            "focused_pytest_cwd": str(ROOT),
            "verification_environment": environment,
            "pytest_python_realpath": str(_python_executable().resolve()),
            "pytest_python_sha256": _sha256_file(_python_executable().resolve()),
            "pytest_stdout_sha256": _sha256(pytest_completed.stdout),
            "pytest_stderr_sha256": _sha256(pytest_completed.stderr),
            "scanner_program": str(SCANNER),
            "scanner_forbidden_needles": _forbidden_manifest(),
            "scanner_stdout_sha256": _sha256(scanner_completed.stdout),
            "scanner_stderr_sha256": _sha256(scanner_completed.stderr),
            "scan_receipt": scan,
            "scan_file_count": scan_body["file_count"],
            "scan_directory_count": scan_body["directory_count"],
            "scan_regular_files_sha256": scan_body["regular_files_sha256"],
            "forbidden_match_count": 0,
            "scan_error_count": 0,
            "verification_source_stable": True,
            "verification_source_manifest": source_manifest,
            "verification_source_count": len(source_manifest),
            "verification_source_manifest_sha256": protocol._source_manifest_sha256(
                source_manifest
            ),
        },
        root_consumption_status=DEVELOPMENT_ONLY,
    )


def main() -> int:
    base = iteration.VERIFICATION_BASE.resolve()
    out = iteration.VERIFICATION_RECEIPT.resolve()
    if os.path.lexists(base):
        raise FileExistsError(f"frozen verification base is not fresh: {base}")
    if os.path.lexists(out):
        raise FileExistsError(f"frozen verification receipt already exists: {out}")
    if not base.parent.is_dir() or base.parent.is_symlink():
        raise RuntimeError("verification-base parent is not a real directory")
    if not out.parent.is_dir() or out.parent.is_symlink():
        raise RuntimeError("verification-receipt parent is not a real directory")

    environment = protocol._verification_environment(base)
    source_manifest_start = protocol._verification_source_manifest(iteration.SPEC)
    collection_command = _collection_command(base)
    collection_completed = subprocess.run(
        collection_command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )
    collection_nodeids = _parse_pytest_collection(collection_completed)
    if os.path.lexists(base):
        raise RuntimeError("collect-only pytest polluted the frozen fresh verification base")
    pytest_command = _pytest_command(base)
    pytest_completed = subprocess.run(
        pytest_command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )
    pytest_test_count = _parse_pytest_pass_count(pytest_completed)
    if not base.is_dir() or base.is_symlink():
        raise RuntimeError("focused pytest did not create the exact real verification base")

    scanner_command = _scanner_command(base)
    scanner_completed = subprocess.run(
        scanner_command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )
    scan = _decode_scanner_output(scanner_completed, base)
    source_manifest_end = protocol._verification_source_manifest(iteration.SPEC)
    if source_manifest_end != source_manifest_start:
        raise RuntimeError("verification source closure changed during collection, tests, or scan")
    if pytest_test_count != len(collection_nodeids):
        raise RuntimeError("passed-test count disagrees with the exact collection manifest")
    receipt = _build_verification_receipt(
        base=base,
        collection_command=collection_command,
        collection_completed=collection_completed,
        collection_nodeids=collection_nodeids,
        pytest_command=pytest_command,
        pytest_completed=pytest_completed,
        pytest_test_count=pytest_test_count,
        scanner_completed=scanner_completed,
        scan=scan,
        environment=environment,
        source_manifest=source_manifest_start,
    )
    directory_fd = open_directory(out.parent)
    try:
        prepared = prepare_json(
            directory_fd,
            out.name,
            iteration.DEVELOPMENT_DOMAIN,
            "verification",
            receipt,
        )
        report = publish_exclusive(directory_fd, prepared)
    finally:
        os.close(directory_fd)
    if not report.accepted or report.recovery_uncertainty or report.publication_error:
        raise RuntimeError("verification receipt publication was not accepted")
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "namespace": receipt["namespace"],
                "schema": receipt["schema"],
                "status": receipt["status"],
                "sha256": prepared.payload_sha256,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
