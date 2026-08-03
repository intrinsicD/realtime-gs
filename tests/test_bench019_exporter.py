"""CPU-only contract tests for the StructSplat BENCH-019 exporter."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from rtgs import bench019 as B


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path, value: object) -> dict:
    _write_json(path, value)
    return B.describe_artifact(path)


def _design_digest(protocol: dict) -> str:
    payload = copy.deepcopy(protocol)
    for key in ("state", "design_sha256", "protocol_sha256", "review"):
        payload.pop(key, None)
    return hashlib.sha256(B.canonical_json(payload)).hexdigest()


def _protocol(
    tmp_path: Path, *, frozen: bool = False, formal_valid: bool = False
) -> tuple[Path, dict]:
    bindings = tmp_path / "bindings"
    task = _artifact(bindings / "task.json", {"task": "BENCH-019"})
    dataset = _artifact(bindings / "dataset.json", {"captures": ["capture_a"]})
    environment = _artifact(bindings / "environment.json", {"python": "test"})
    schedule = _artifact(bindings / "schedule.json", {"iterations": 4, "lr": 0.01})
    shared = _artifact(bindings / "shared.json", {"shared": True})
    families = []
    for family_id, additive, psnr in (
        ("additive", True, 32.0),
        ("normalized", False, 30.0),
    ):
        families.append(
            {
                "id": family_id,
                "field_manifest": _artifact(
                    bindings / f"{family_id}_manifest.json", {"family": family_id}
                ),
                "stage1_metrics": _artifact(
                    bindings / f"{family_id}_stage1.json",
                    {"foreground_psnr": psnr, "query_error": 0.25},
                ),
                "semantics": {
                    "provider": "native" if additive else "structsplat",
                    "equation": "additive_sum" if additive else "normalized_weighted_sum",
                    "blend_mode": "additive" if additive else "normalized",
                    "alpha_policy": "packed_alpha",
                    "coordinate_convention": "top-left pixel center is (0.5,0.5)",
                    "semantic_digest": hashlib.sha256(family_id.encode()).hexdigest(),
                },
            }
        )
    protocol = {
        "schema": B.PROTOCOL_SCHEMA,
        "task_id": "BENCH-019",
        "state": "review",
        "driver": "driver-a",
        "claim_scope": "workload_specific",
        "repositories": [],
        "captures": [
            {
                "id": "capture_a",
                "frames": [
                    {
                        "id": "frame_a",
                        "pixels": shared,
                        "masks": shared,
                        "cameras": shared,
                        "split": {"train": ["C0001"], "heldout": ["C0002"]},
                        "families": families,
                    }
                ],
            }
        ],
        "downstream": {
            "task_manifest": task,
            "dataset_manifest": dataset,
            "environment": environment,
            "schedule_config": schedule,
            "command": ["python", "driver.py", "run"],
            "working_directory": str(tmp_path),
            "outcome_root": str(tmp_path / "outcomes"),
            "seeds": [11],
            "initializers": ["fixed"],
            "result_schema": B.ROW_SCHEMA,
        },
        "predictors": [
            {"name": "foreground_psnr", "direction": "higher"},
            {"name": "query_error", "direction": "lower"},
        ],
        "responses": [
            {"name": "heldout_psnr", "direction": "higher", "primary": True},
            {"name": "fit_seconds", "direction": "lower", "primary": False},
        ],
        "analysis": {},
        "aa_replay": {
            "frame_id": "frame_a",
            "family_id": "additive",
            "seed": 11,
            "initializer": "fixed",
            "primary_replicate": "primary",
            "replay_replicate": "aa",
            "metric_abs_tolerance": {"foreground_psnr": 0.0, "heldout_psnr": 0.0},
        },
    }
    if formal_valid:
        clean_status = hashlib.sha256(b"").hexdigest()
        protocol["repositories"] = [
            {
                "name": name,
                "root": str(tmp_path),
                "commit": commit * 40,
                "branch": "test",
                "dirty": False,
                "status_sha256": clean_status,
                "environment": environment,
            }
            for name, commit in (("structsplat", "a"), ("realtime-gs", "b"))
        ]
        protocol["downstream"]["seeds"] = [11, 12, 13]
        protocol["analysis"] = {
            "bootstrap_replicates": 200,
            "bootstrap_seed": 19019,
            "minimum_capture_groups": 1,
            "minimum_frames": 1,
            "minimum_family_count": 2,
            "minimum_spearman": 0.8,
            "minimum_bootstrap_lower": 0.0,
            "minimum_lofo_top1_agreement": 1.0,
            "selection_priority": ["foreground_psnr", "query_error"],
            "missing_policy": "fail_closed",
        }
    protocol["design_sha256"] = _design_digest(protocol)
    if frozen:
        protocol["review"] = {
            "driver": "driver-a",
            "reviewer": "reviewer-b",
            "verdict": "approved",
            "design_sha256": protocol["design_sha256"],
            "artifact": shared,
        }
        protocol["state"] = "frozen"
        payload = copy.deepcopy(protocol)
        payload.pop("protocol_sha256", None)
        protocol["protocol_sha256"] = hashlib.sha256(B.canonical_json(payload)).hexdigest()
    path = _write_json(tmp_path / "protocol.json", protocol)
    return path, protocol


def _family(protocol: dict, family_id: str) -> dict:
    return next(
        family
        for capture in protocol["captures"]
        for frame in capture["frames"]
        for family in frame["families"]
        if family["id"] == family_id
    )


def _source(
    tmp_path: Path,
    protocol: dict,
    *,
    family_id: str = "additive",
    replicate_id: str = "primary",
    status: str = "ok",
) -> tuple[Path, dict]:
    cell = {
        "capture_id": "capture_a",
        "frame_id": "frame_a",
        "family_id": family_id,
        "seed": 11,
        "initializer": "fixed",
        "replicate_id": replicate_id,
    }
    family = _family(protocol, family_id)
    identity = B.protocol_identity(protocol, allow_review=True)
    factor = B.downstream_factor_digest(
        protocol,
        frame_id="frame_a",
        seed=11,
        initializer="fixed",
        allow_review=True,
    )
    if status == "error":
        value = {
            "schema": B.SOURCE_SCHEMA,
            "protocol_digest": identity,
            "status": "error",
            "error": "worker failed before metrics",
            "cell": cell,
            "sources": {},
            "metric_bindings": {"stage1": {}, "downstream": {}},
            "artifacts": {},
        }
        return _write_json(tmp_path / f"{family_id}_{replicate_id}.source.json", value), value

    run_receipt = {
        "bench019": {
            "schema": B.RUN_BINDING_SCHEMA,
            **cell,
            "field_manifest_sha256": family["field_manifest"]["sha256"],
            "field_semantic_digest": family["semantics"]["semantic_digest"],
            "downstream_factor_digest": factor,
        },
        "status": "passed",
    }
    cell_dir = tmp_path / "cell_sources" / f"{family_id}_{replicate_id}"
    downstream_metrics = _artifact(
        cell_dir / "metrics.json",
        {"heldout": {"psnr": 24.5}, "timing": {"fit_seconds": 3.25}},
    )
    run_receipt_artifact = _artifact(cell_dir / "run_receipt.json", run_receipt)
    artifacts = {
        name: _artifact(cell_dir / f"{name}.json", {"artifact": name})
        for name in B.REQUIRED_CELL_ARTIFACTS
    }
    value = {
        "schema": B.SOURCE_SCHEMA,
        "protocol_digest": identity,
        "status": "ok",
        "error": "",
        "cell": cell,
        "sources": {
            "downstream_metrics": downstream_metrics,
            "run_receipt": run_receipt_artifact,
        },
        "metric_bindings": {
            "stage1": {
                "foreground_psnr": {
                    "source": "stage1_metrics",
                    "pointer": "/foreground_psnr",
                },
                "query_error": {"source": "stage1_metrics", "pointer": "/query_error"},
            },
            "downstream": {
                "heldout_psnr": {
                    "source": "downstream_metrics",
                    "pointer": "/heldout/psnr",
                },
                "fit_seconds": {
                    "source": "downstream_metrics",
                    "pointer": "/timing/fit_seconds",
                },
            },
        },
        "artifacts": artifacts,
    }
    return _write_json(tmp_path / f"{family_id}_{replicate_id}.source.json", value), value


def _export(
    tmp_path: Path,
    protocol_path: Path,
    source_path: Path,
    name: str,
) -> tuple[Path, Path, dict]:
    row_path = tmp_path / "exports" / f"{name}.cell.json"
    receipt_path = tmp_path / "exports" / f"{name}.export.json"
    row = B.export_cell(
        protocol_path,
        source_path,
        row_path,
        receipt_path,
        allow_review_protocol=True,
    )
    return row_path, receipt_path, row


@pytest.mark.parametrize("family_id", ["additive", "normalized"])
def test_export_preserves_exact_family_semantics_and_metric_sources(
    tmp_path: Path, family_id: str
) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, _ = _source(tmp_path, protocol, family_id=family_id)
    row_path, receipt_path, row = _export(tmp_path, protocol_path, source_path, family_id)

    family = _family(protocol, family_id)
    assert set(row) == {
        "schema",
        "status",
        "error",
        "capture_id",
        "frame_id",
        "family_id",
        "seed",
        "initializer",
        "replicate_id",
        "field_manifest_sha256",
        "field_semantic_digest",
        "downstream_factor_digest",
        "stage1",
        "downstream",
        "artifacts",
    }
    assert row["field_semantic_digest"] == family["semantics"]["semantic_digest"]
    assert row["stage1"]["foreground_psnr"] == (32.0 if family_id == "additive" else 30.0)
    assert row["downstream"] == {"heldout_psnr": 24.5, "fit_seconds": 3.25}
    assert set(row["artifacts"]) == set(B.REQUIRED_CELL_ARTIFACTS)
    assert B.validate_row(protocol, row, allow_review_protocol=True) == (
        "frame_a",
        family_id,
        11,
        "fixed",
        "primary",
    )
    assert row_path.is_file() and receipt_path.is_file()


def test_factor_is_family_and_replicate_invariant_but_protocol_bound(tmp_path: Path) -> None:
    _, protocol = _protocol(tmp_path)
    additive_source, _ = _source(tmp_path, protocol, family_id="additive")
    normalized_source, _ = _source(tmp_path, protocol, family_id="normalized")
    aa_source, _ = _source(tmp_path, protocol, family_id="additive", replicate_id="aa")

    manifests = [
        B.load_json_object(path) for path in (additive_source, normalized_source, aa_source)
    ]
    run_digests = []
    for manifest in manifests:
        receipt_path = Path(manifest["sources"]["run_receipt"]["path"])
        run_digests.append(B.load_json_object(receipt_path)["bench019"]["downstream_factor_digest"])
    assert len(set(run_digests)) == 1
    expected = B.downstream_factor_digest(
        protocol,
        frame_id="frame_a",
        seed=11,
        initializer="fixed",
        allow_review=True,
    )
    assert run_digests == [expected, expected, expected]


def test_formal_export_requires_frozen_protocol_and_detects_protocol_tamper(tmp_path: Path) -> None:
    review_path, review_protocol = _protocol(tmp_path / "review")
    source_path, _ = _source(tmp_path / "review", review_protocol)
    with pytest.raises(B.ExportError, match="requires a frozen"):
        B.export_cell(
            review_path,
            source_path,
            tmp_path / "formal.cell.json",
            tmp_path / "formal.receipt.json",
        )

    frozen_path, frozen = _protocol(tmp_path / "frozen", frozen=True)
    frozen["downstream"]["command"].append("--tampered")
    _write_json(frozen_path, frozen)
    with pytest.raises(B.ExportError, match="design digest"):
        B.protocol_identity(frozen)

    frozen_path, frozen = _protocol(tmp_path / "self-reviewed", frozen=True)
    frozen["review"]["reviewer"] = frozen["driver"]
    payload = copy.deepcopy(frozen)
    payload.pop("protocol_sha256", None)
    frozen["protocol_sha256"] = hashlib.sha256(B.canonical_json(payload)).hexdigest()
    _write_json(frozen_path, frozen)
    with pytest.raises(B.ExportError, match="distinct approval"):
        B.protocol_identity(frozen)


def test_formal_protocol_requires_complete_upstream_v1_invariants(tmp_path: Path) -> None:
    _, incomplete = _protocol(tmp_path / "incomplete", frozen=True)
    with pytest.raises(B.ExportError, match="bind both StructSplat and realtime-gs"):
        B.protocol_identity(incomplete)

    protocol_path, protocol = _protocol(tmp_path / "complete", frozen=True, formal_valid=True)
    source_path, _ = _source(tmp_path / "complete", protocol)
    row_path = tmp_path / "complete" / "formal.cell.json"
    receipt_path = tmp_path / "complete" / "formal.export.json"
    row = B.export_cell(protocol_path, source_path, row_path, receipt_path)
    assert row["status"] == "ok"
    assert B.protocol_identity(protocol) == protocol["protocol_sha256"]


def test_run_receipt_factor_or_semantic_drift_fails_closed(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, source = _source(tmp_path, protocol)
    receipt_path = Path(source["sources"]["run_receipt"]["path"])
    receipt = B.load_json_object(receipt_path)
    receipt["bench019"]["downstream_factor_digest"] = "f" * 64
    _write_json(receipt_path, receipt)
    source["sources"]["run_receipt"] = B.describe_artifact(receipt_path)
    _write_json(source_path, source)
    with pytest.raises(B.ExportError, match="run_receipt BENCH-019 binding differs"):
        _export(tmp_path, protocol_path, source_path, "drift")


def test_global_downstream_artifact_tamper_fails_closed(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, _ = _source(tmp_path, protocol)
    schedule_path = Path(protocol["downstream"]["schedule_config"]["path"])
    schedule_path.write_text('{"iterations": 999}\n', encoding="utf-8")

    with pytest.raises(B.ExportError, match="schedule_config (SHA-256|byte length)"):
        _export(tmp_path, protocol_path, source_path, "global-drift")


def test_json_pointer_rejects_invalid_escape(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, source = _source(tmp_path, protocol)
    source["metric_bindings"]["downstream"]["heldout_psnr"]["pointer"] = "/heldout/~2psnr"
    _write_json(source_path, source)

    with pytest.raises(B.ExportError, match="invalid RFC 6901 escape"):
        _export(tmp_path, protocol_path, source_path, "invalid-pointer")


def test_stage1_predictors_cannot_be_sourced_from_downstream(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, source = _source(tmp_path, protocol)
    source["metric_bindings"]["stage1"]["foreground_psnr"] = {
        "source": "downstream_metrics",
        "pointer": "/heldout/psnr",
    }
    _write_json(source_path, source)
    with pytest.raises(B.ExportError, match="Stage-1 predictors must come"):
        _export(tmp_path, protocol_path, source_path, "wrong-source")


def test_nonfinite_boolean_or_missing_metric_fails_closed(tmp_path: Path) -> None:
    for case in ("boolean", "missing", "nan"):
        case_root = tmp_path / case
        protocol_path, protocol = _protocol(case_root)
        source_path, source = _source(case_root, protocol)
        metrics_path = Path(source["sources"]["downstream_metrics"]["path"])
        metrics = B.load_json_object(metrics_path)
        if case == "boolean":
            metrics["heldout"]["psnr"] = True
            _write_json(metrics_path, metrics)
        elif case == "missing":
            del metrics["heldout"]["psnr"]
            _write_json(metrics_path, metrics)
        else:
            metrics_path.write_text(
                '{"heldout":{"psnr":NaN},"timing":{"fit_seconds":3.25}}\n',
                encoding="utf-8",
            )
        source["sources"]["downstream_metrics"] = B.describe_artifact(metrics_path)
        _write_json(source_path, source)
        with pytest.raises(B.ExportError):
            _export(case_root, protocol_path, source_path, case)


def test_metric_extraction_preserves_an_exact_integer(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, source = _source(tmp_path, protocol)
    metrics_path = Path(source["sources"]["downstream_metrics"]["path"])
    metrics = B.load_json_object(metrics_path)
    metrics["heldout"]["psnr"] = 24
    _write_json(metrics_path, metrics)
    source["sources"]["downstream_metrics"] = B.describe_artifact(metrics_path)
    _write_json(source_path, source)

    _, _, row = _export(tmp_path, protocol_path, source_path, "integer")
    assert row["downstream"]["heldout_psnr"] == 24
    assert isinstance(row["downstream"]["heldout_psnr"], int)


def test_source_and_cell_artifact_tampering_is_detected(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, source = _source(tmp_path, protocol)
    field_path = Path(source["artifacts"]["field"]["path"])
    field_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(B.ExportError, match="SHA-256|byte length"):
        _export(tmp_path, protocol_path, source_path, "tampered")


def test_error_cell_is_explicit_and_has_no_success_payload(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, _ = _source(tmp_path, protocol, status="error")
    _, _, row = _export(tmp_path, protocol_path, source_path, "error")
    assert row["status"] == "error"
    assert row["stage1"] == row["downstream"] == row["artifacts"] == {}
    B.validate_row(protocol, row, allow_review_protocol=True)

    source_path, source = _source(tmp_path / "bad-shape", protocol, status="error")
    source["sources"] = []
    _write_json(source_path, source)
    with pytest.raises(B.ExportError, match="empty source/artifact objects"):
        _export(tmp_path / "bad-shape", protocol_path, source_path, "bad-shape")


def test_export_refuses_overwrite_and_removes_partial_pair(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, _ = _source(tmp_path, protocol)
    row_path, receipt_path, _ = _export(tmp_path, protocol_path, source_path, "once")
    with pytest.raises(B.ExportError, match="must both be new"):
        B.export_cell(
            protocol_path,
            source_path,
            row_path,
            receipt_path,
            allow_review_protocol=True,
        )


def test_assembly_orders_cells_and_rejects_missing_or_duplicate_rows(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    rows = []
    for family_id, replicate in (
        ("normalized", "primary"),
        ("additive", "aa"),
        ("additive", "primary"),
    ):
        source_path, _ = _source(tmp_path, protocol, family_id=family_id, replicate_id=replicate)
        row_path, receipt_path, _ = _export(
            tmp_path, protocol_path, source_path, f"{family_id}-{replicate}"
        )
        rows.append((row_path, receipt_path))

    ordered = B.assemble_rows(
        protocol_path,
        [row for row, _ in rows],
        tmp_path / "all.jsonl",
        tmp_path / "all.receipt.json",
        export_receipt_paths=[receipt for _, receipt in rows],
        allow_review_protocol=True,
    )
    assert [(row["family_id"], row["replicate_id"]) for row in ordered] == [
        ("additive", "primary"),
        ("normalized", "primary"),
        ("additive", "aa"),
    ]

    with pytest.raises(B.ExportError, match="duplicate"):
        B.assemble_rows(
            protocol_path,
            [rows[0][0], rows[0][0]],
            tmp_path / "duplicate.jsonl",
            tmp_path / "duplicate.receipt.json",
            export_receipt_paths=[rows[0][1], rows[0][1]],
            allow_review_protocol=True,
            allow_incomplete=True,
        )
    with pytest.raises(B.ExportError, match="missing"):
        B.assemble_rows(
            protocol_path,
            [rows[0][0]],
            tmp_path / "missing.jsonl",
            tmp_path / "missing.receipt.json",
            export_receipt_paths=[rows[0][1]],
            allow_review_protocol=True,
        )
    diagnostic = B.assemble_rows(
        protocol_path,
        [rows[0][0]],
        tmp_path / "diagnostic.jsonl",
        tmp_path / "diagnostic.receipt.json",
        export_receipt_paths=[rows[0][1]],
        allow_review_protocol=True,
        allow_incomplete=True,
    )
    assert len(diagnostic) == 1
    receipt = B.load_json_object(tmp_path / "diagnostic.receipt.json")
    assert receipt["diagnostic"] is True and len(receipt["missing_cells"]) == 2


def test_assembly_requires_and_revalidates_export_receipts(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, _ = _source(tmp_path, protocol)
    row_path, receipt_path, _ = _export(tmp_path, protocol_path, source_path, "receipt")

    with pytest.raises(B.ExportError, match="one export receipt per cell"):
        B.assemble_rows(
            protocol_path,
            [row_path],
            tmp_path / "no-receipt.jsonl",
            tmp_path / "no-receipt.assembly.json",
            allow_review_protocol=True,
            allow_incomplete=True,
        )

    receipt = B.load_json_object(receipt_path)
    receipt["row_canonical_sha256"] = "f" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(B.ExportError, match="canonical row digest"):
        B.assemble_rows(
            protocol_path,
            [row_path],
            tmp_path / "tampered-receipt.jsonl",
            tmp_path / "tampered-receipt.assembly.json",
            export_receipt_paths=[receipt_path],
            allow_review_protocol=True,
            allow_incomplete=True,
        )


@pytest.mark.parametrize("mutation", ["delete", "substitute", "extra"])
def test_assembly_reconciles_receipt_source_set(tmp_path: Path, mutation: str) -> None:
    case_root = tmp_path / mutation
    protocol_path, protocol = _protocol(case_root)
    source_path, _ = _source(case_root, protocol)
    row_path, receipt_path, _ = _export(
        case_root, protocol_path, source_path, f"receipt-{mutation}"
    )
    receipt = B.load_json_object(receipt_path)
    if mutation == "delete":
        del receipt["source_artifacts"]["downstream_metrics"]
    elif mutation == "substitute":
        receipt["source_artifacts"]["downstream_metrics"] = receipt["source_artifacts"][
            "run_receipt"
        ]
    else:
        receipt["source_artifacts"]["extra"] = receipt["source_artifacts"]["run_receipt"]
    _write_json(receipt_path, receipt)

    with pytest.raises(B.ExportError, match="source_artifacts differ"):
        B.assemble_rows(
            protocol_path,
            [row_path],
            case_root / "rows.jsonl",
            case_root / "assembly.json",
            export_receipt_paths=[receipt_path],
            allow_review_protocol=True,
            allow_incomplete=True,
        )


def test_assembly_replays_cooperatively_modified_source_manifest(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, _ = _source(tmp_path, protocol)
    row_path, receipt_path, _ = _export(tmp_path, protocol_path, source_path, "source-mutation")
    source = B.load_json_object(source_path)
    del source["sources"]["downstream_metrics"]
    _write_json(source_path, source)
    receipt = B.load_json_object(receipt_path)
    receipt["source_manifest"] = B.describe_artifact(source_path)
    del receipt["source_artifacts"]["downstream_metrics"]
    _write_json(receipt_path, receipt)

    with pytest.raises(B.ExportError, match="unknown source downstream_metrics"):
        B.assemble_rows(
            protocol_path,
            [row_path],
            tmp_path / "rows.jsonl",
            tmp_path / "assembly.json",
            export_receipt_paths=[receipt_path],
            allow_review_protocol=True,
            allow_incomplete=True,
        )


def test_assembly_rejects_cooperatively_added_unused_source(tmp_path: Path) -> None:
    protocol_path, protocol = _protocol(tmp_path)
    source_path, _ = _source(tmp_path, protocol)
    row_path, receipt_path, _ = _export(tmp_path, protocol_path, source_path, "extra-source")
    extra = _artifact(tmp_path / "cell_sources/extra.json", {"unused": True})
    source = B.load_json_object(source_path)
    source["sources"]["extra"] = extra
    _write_json(source_path, source)
    receipt = B.load_json_object(receipt_path)
    receipt["source_manifest"] = B.describe_artifact(source_path)
    receipt["source_artifacts"]["extra"] = extra
    _write_json(receipt_path, receipt)

    with pytest.raises(B.ExportError, match="unreferenced source artifacts"):
        B.assemble_rows(
            protocol_path,
            [row_path],
            tmp_path / "rows.jsonl",
            tmp_path / "assembly.json",
            export_receipt_paths=[receipt_path],
            allow_review_protocol=True,
            allow_incomplete=True,
        )


def test_factor_rejects_undeclared_coordinates(tmp_path: Path) -> None:
    _, protocol = _protocol(tmp_path)
    for kwargs, message in (
        ({"frame_id": "other", "seed": 11, "initializer": "fixed"}, "frame_id"),
        ({"frame_id": "frame_a", "seed": 99, "initializer": "fixed"}, "seed"),
        ({"frame_id": "frame_a", "seed": 11, "initializer": "other"}, "initializer"),
    ):
        with pytest.raises(B.ExportError, match=message):
            B.downstream_factor_record(protocol, allow_review=True, **kwargs)
