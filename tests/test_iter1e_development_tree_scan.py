"""Isolated tests for the iter1e read-only development-tree verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks import inverse_projection_fiber_protocol as protocol
from scripts import write_iter1e_verification_receipt as verification_runner

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "verify_iter1e_development_tree.py"
DUMMY_PATH_NEEDLE = "DUMMY_PATH_FORBIDDEN"
DUMMY_PAYLOAD_NEEDLE = "DUMMY_PAYLOAD_FORBIDDEN"


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    """Module-local temp path without pytest's persistent ``*current`` symlink."""

    digest = hashlib.sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:20]
    path = tmp_path_factory.getbasetemp() / f"case_{digest}"
    path.mkdir()
    return path


def _run_scan(base: Path, *needles: str) -> subprocess.CompletedProcess[bytes]:
    command = [sys.executable, str(SCRIPT), "--base", str(base)]
    for needle in needles:
        command.extend(["--forbidden-needle", needle])
    return subprocess.run(command, check=False, capture_output=True)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _tree_snapshot(base: Path) -> list[tuple[str, str]]:
    snapshot = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            kind = "symlink"
        elif path.is_dir():
            kind = "directory"
        elif path.is_file():
            kind = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            kind = "nonregular"
        snapshot.append((path.relative_to(base).as_posix(), kind))
    return snapshot


def test_scan_detects_forbidden_path_payload_symlink_and_nonregular_then_passes_cleanly(
    tmp_path: Path,
) -> None:
    base = tmp_path / "development"
    nested = base / "nested"
    nested.mkdir(parents=True)
    clean_a = base / "a.bin"
    clean_b = nested / "b.bin"
    clean_a.write_bytes(b"alpha")
    clean_b.write_bytes(b"beta")

    bad_name = base / f"{DUMMY_PATH_NEEDLE}.txt"
    bad_payload = nested / "payload.bin"
    bad_name.write_bytes(b"ordinary bytes")
    bad_payload.write_bytes(b"prefix-" + DUMMY_PAYLOAD_NEEDLE.encode() + b"-suffix")
    symlink = base / "forbidden-link"
    symlink.symlink_to(clean_a)
    fifo = base / "forbidden-fifo"
    os.mkfifo(fifo)

    before = _tree_snapshot(base)
    failed = _run_scan(base, DUMMY_PATH_NEEDLE, DUMMY_PAYLOAD_NEEDLE)
    after = _tree_snapshot(base)

    assert failed.returncode == 1
    assert failed.stderr == b""
    assert before == after
    failed_receipt = json.loads(failed.stdout)
    assert failed.stdout == _canonical(failed_receipt) + b"\n"
    assert failed_receipt["status"] == "FAIL"
    assert failed_receipt["root_consumption_status"] == "DEVELOPMENT_ONLY"
    assert failed_receipt["scan"]["scan_complete"] is False
    assert failed_receipt["scan"]["forbidden_match_count"] == 2
    assert {match["kind"] for match in failed_receipt["scan"]["forbidden_matches"]} == {
        "file_bytes",
        "path_name",
    }
    assert {error["code"] for error in failed_receipt["scan"]["errors"]} == {
        "NONREGULAR_FORBIDDEN",
        "SYMLINK_FORBIDDEN",
    }
    assert DUMMY_PATH_NEEDLE.encode() not in failed.stdout
    assert DUMMY_PAYLOAD_NEEDLE.encode() not in failed.stdout

    # Fault fixtures must not contaminate the final development-tree verification.
    bad_name.unlink()
    bad_payload.unlink()
    symlink.unlink()
    fifo.unlink()

    clean_before = _tree_snapshot(base)
    passed = _run_scan(base, DUMMY_PATH_NEEDLE, DUMMY_PAYLOAD_NEEDLE)
    clean_after = _tree_snapshot(base)

    assert passed.returncode == 0
    assert passed.stderr == b""
    assert clean_before == clean_after
    receipt = json.loads(passed.stdout)
    assert passed.stdout == _canonical(receipt) + b"\n"
    assert receipt["status"] == "PASS"
    assert receipt["scan"]["scan_complete"] is True
    assert receipt["scan"]["errors"] == []
    assert receipt["scan"]["forbidden_match_count"] == 0
    assert receipt["scan"]["forbidden_matches"] == []
    assert receipt["scan"]["file_count"] == 2
    expected_files = {
        path.relative_to(base).as_posix().encode().hex(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in (clean_a, clean_b)
    }
    observed_files = {
        record["path_hex"]: {
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for record in receipt["scan"]["files"]
    }
    assert observed_files == expected_files
    assert DUMMY_PATH_NEEDLE.encode() not in passed.stdout
    assert DUMMY_PAYLOAD_NEEDLE.encode() not in passed.stdout


def test_invalid_arguments_are_bounded_canonical_json() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 1
    assert completed.stderr == b""
    receipt = json.loads(completed.stdout)
    assert completed.stdout == _canonical(receipt) + b"\n"
    assert receipt["status"] == "FAIL"
    assert receipt["scan"] == {
        "errors": [{"code": "INVALID_ARGUMENTS"}],
        "scan_complete": False,
    }


def _exact_clean_scan(base: Path) -> dict[str, object]:
    completed = _run_scan(
        base,
        *(needle.decode("utf-8") for needle in verification_runner._forbidden_needles()),
    )
    assert completed.returncode == 0
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    "mutation",
    ["missing_needle", "extra_needle", "reordered_needles", "file_count", "files_hash"],
)
def test_verification_validator_rejects_inexact_or_inconsistent_scan(
    tmp_path: Path,
    mutation: str,
) -> None:
    base = tmp_path / "clean-scan"
    base.mkdir()
    (base / "ordinary.bin").write_bytes(b"ordinary development bytes")
    scan = _exact_clean_scan(base)
    body = scan["scan"]
    if mutation == "missing_needle":
        body["forbidden_needles"] = body["forbidden_needles"][:-1]
    elif mutation == "extra_needle":
        body["forbidden_needles"] = [
            *body["forbidden_needles"],
            {"index": 11, "length_bytes": 3, "sha256": "0" * 64},
        ]
    elif mutation == "reordered_needles":
        body["forbidden_needles"] = list(reversed(body["forbidden_needles"]))
    elif mutation == "file_count":
        body["file_count"] += 1
    else:
        body["regular_files_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="scan|manifest|count|hash"):
        verification_runner._validate_scan(scan, base.resolve())


def test_pytest_attestation_is_derived_only_from_clean_all_passed_output() -> None:
    valid = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout=b".......................................... [100%]\n42 passed in 1.23s\n",
        stderr=b"",
    )
    assert verification_runner._parse_pytest_pass_count(valid) == 42

    invalid = (
        subprocess.CompletedProcess(["pytest"], 1, b"1 failed in 0.1s\n", b""),
        subprocess.CompletedProcess(["pytest"], 0, b"41 passed, 1 skipped in 0.1s\n", b""),
        subprocess.CompletedProcess(["pytest"], 0, b"42 passed in 0.1s\n", b"warning"),
    )
    for completed in invalid:
        with pytest.raises(RuntimeError, match="pytest"):
            verification_runner._parse_pytest_pass_count(completed)


def test_pytest_collection_attests_exact_unique_nodeids_from_all_frozen_files() -> None:
    nodeids = tuple(
        f"{path}::test_attested_{index}"
        for index, path in enumerate(verification_runner.FOCUSED_TESTS)
    )
    completed = subprocess.CompletedProcess(
        args=["pytest", "--collect-only"],
        returncode=0,
        stdout=("\n".join((*nodeids, "", f"{len(nodeids)} tests collected in 0.1s")) + "\n").encode(
            "utf-8"
        ),
        stderr=b"",
    )
    assert verification_runner._parse_pytest_collection(completed) == nodeids

    invalid = copy.deepcopy(completed)
    invalid.stdout = completed.stdout.replace(nodeids[-1].encode(), nodeids[0].encode())
    with pytest.raises(RuntimeError, match="duplicate|omitted"):
        verification_runner._parse_pytest_collection(invalid)


def test_verification_commands_are_frozen_and_base_must_be_fresh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base = tmp_path / "already-present"
    base.mkdir()
    command = verification_runner._pytest_command(base)
    assert command[1:4] == ["-m", "pytest", "-q"]
    assert command[4:8] == ["-c", os.devnull, "--noconftest", f"--rootdir={REPO}"]
    assert command[-4:-1] == list(verification_runner.FOCUSED_TESTS)
    assert command[-1] == f"--basetemp={base}"
    collection_command = verification_runner._collection_command(base)
    assert "--collect-only" in collection_command
    assert not any(argument.startswith("--basetemp=") for argument in collection_command)
    environment = protocol._verification_environment(base.resolve())
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert "PYTEST_ADDOPTS" not in environment
    scanner_command = verification_runner._scanner_command(base)
    assert scanner_command.count("--forbidden-needle") == 11

    monkeypatch.setattr(verification_runner.iteration, "VERIFICATION_BASE", base)
    monkeypatch.setattr(
        verification_runner.iteration,
        "VERIFICATION_RECEIPT",
        tmp_path / "unwritten.json",
    )
    with pytest.raises(FileExistsError, match="not fresh"):
        verification_runner.main()
    assert not (tmp_path / "unwritten.json").exists()


def test_protocol_accepts_exact_runner_receipt_and_rejects_nested_tampering(
    tmp_path: Path,
) -> None:
    base = (tmp_path / "validated-scan").resolve()
    base.mkdir()
    (base / "ordinary.bin").write_bytes(b"validated development bytes")
    scan = _exact_clean_scan(base)
    nodeids = tuple(
        f"{path}::test_receipt_{index}"
        for index, path in enumerate(verification_runner.FOCUSED_TESTS)
    )
    collection_completed = subprocess.CompletedProcess(
        args=verification_runner._collection_command(base),
        returncode=0,
        stdout=("\n".join((*nodeids, "", f"{len(nodeids)} tests collected in 1.0s")) + "\n").encode(
            "utf-8"
        ),
        stderr=b"",
    )
    pytest_completed = subprocess.CompletedProcess(
        args=verification_runner._pytest_command(base),
        returncode=0,
        stdout=f"... [100%]\n{len(nodeids)} passed in 1.0s\n".encode("ascii"),
        stderr=b"",
    )
    scanner_completed = subprocess.CompletedProcess(
        args=verification_runner._scanner_command(base),
        returncode=0,
        stdout=_canonical(scan) + b"\n",
        stderr=b"",
    )
    spec = replace(verification_runner.iteration.SPEC, verification_base=base)
    environment = protocol._verification_environment(base)
    source_manifest = protocol._verification_source_manifest(spec)
    receipt = verification_runner._build_verification_receipt(
        base=base,
        collection_command=verification_runner._collection_command(base),
        collection_completed=collection_completed,
        collection_nodeids=nodeids,
        pytest_command=verification_runner._pytest_command(base),
        pytest_completed=pytest_completed,
        pytest_test_count=len(nodeids),
        scanner_completed=scanner_completed,
        scan=scan,
        environment=environment,
        source_manifest=source_manifest,
    )
    protocol._validate_verification_receipt(spec, receipt)

    tampered = copy.deepcopy(receipt)
    tampered["scan_receipt"]["scan"]["file_count"] += 1
    with pytest.raises(RuntimeError, match="counts|evidence"):
        protocol._validate_verification_receipt(spec, tampered)

    tampered_environment = copy.deepcopy(receipt)
    tampered_environment["verification_environment"]["PYTEST_ADDOPTS"] = "--ignore=tests"
    with pytest.raises(RuntimeError, match="execution contract"):
        protocol._validate_verification_receipt(spec, tampered_environment)

    tampered_source = copy.deepcopy(receipt)
    tampered_source["verification_source_manifest"][0]["file_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="execution contract"):
        protocol._validate_verification_receipt(spec, tampered_source)
