"""Structural tests for the task-first experiment contract and shared report."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_script("experiment_contract")
BUNDLE = _load_script("check_results_bundle")
LIVE_TASK = REPO / "experiments" / "tasks" / "20260728_vram_claim_stage_frames00008_00009.json"
LIVE_V2_TASK = (
    REPO
    / "experiments"
    / "tasks"
    / "20260730_additive_analytic_objective_stage_frames00008_00009.json"
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_body(task_id: str, digest: str, reviewer: str, verdict: str) -> str:
    return (
        "# Prospective Protocol Review\n\n"
        f"- Task ID: `{task_id}`\n"
        f"- Protocol SHA-256: `{digest}`\n"
        f"- Reviewer: `{reviewer}`\n"
        f"- Verdict: `{verdict}`\n"
        "- Outcome Access: `none`\n\n"
        "## Scope\n\nFixture scope.\n\n"
        "## Checks\n\nFixture checks.\n\n"
        "## Findings\n\nFixture findings.\n\n"
        "## Protected Actions Not Taken\n\nNo protected run or outcome access.\n"
    )


def test_registered_three_arm_program_is_valid() -> None:
    assert CONTRACT.validate_repository(root=REPO) == []


def test_protocol_digest_ignores_only_review_and_status() -> None:
    task = json.loads(LIVE_TASK.read_text(encoding="utf-8"))
    initial = CONTRACT.protocol_sha256(task)
    task["status"] = "ready"
    task["protocol_review"] = {
        "reviewer": "reviewer",
        "verdict": "approved",
        "protocol_sha256": initial,
        "artifact": f"experiments/reviews/{task['task_id']}_PROTOCOL_REVIEW.md",
    }
    assert CONTRACT.protocol_sha256(task) == initial
    task["seeds"].append(999)
    assert CONTRACT.protocol_sha256(task) != initial


def test_development_source_state_binds_untracked_file_bytes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=RTGS Test",
            "-c",
            "user.email=rtgs-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    untracked = tmp_path / "new_driver.py"
    untracked.write_text("VALUE = 2\n", encoding="utf-8")
    first = hashlib.sha256(CONTRACT._development_source_state(tmp_path)).hexdigest()
    untracked.write_text("VALUE = 3\n", encoding="utf-8")
    second = hashlib.sha256(CONTRACT._development_source_state(tmp_path)).hexdigest()
    assert first != second


def test_frozen_source_binding_is_enforced_against_live_bytes(tmp_path: Path) -> None:
    source = tmp_path / "src/rtgs/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    binding = {
        "patterns": ["src/rtgs/**/*.py"],
        "file_count": 0,
        "aggregate_sha256": "0" * 64,
    }
    binding = CONTRACT.build_source_binding(binding, root=tmp_path)
    task = {"frozen_configuration": {"source_binding": binding}}

    assert CONTRACT.verify_source_binding(task, root=tmp_path) == []
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert any(
        "behavior-bearing source differs" in error
        for error in CONTRACT.verify_source_binding(task, root=tmp_path)
    )


def _strict_cell_bundle_fixture(tmp_path: Path):
    task_id = "fixture_task"
    run = tmp_path / "runs" / task_id
    run.mkdir(parents=True)
    task_lock = run / "task.lock.json"
    _json(task_lock, {"fixture": True})
    lock = {
        "protocol_sha256": "a" * 64,
        "data_seal_sha256": "b" * 64,
    }
    task = {
        "task_id": task_id,
        "datasets": [{"id": "dataset"}],
        "comparators": [{"id": "arm"}],
        "seeds": [7],
        "primary_metrics": [{"id": "quality"}],
        "splits": {"dataset": {"train": ["C1", "C2"], "heldout": ["C3"]}},
        "frozen_configuration": {
            "source_binding": {"aggregate_sha256": "c" * 64},
            "optimizer_views": ["C1"],
            "validation_views": ["C2"],
            "rgb_refinement": {"iterations": 10},
            "warmup": {
                "dataset_id": "dataset",
                "arm_id": "arm",
                "seed": 7,
                "iterations": 2,
            },
            "cell_receipt_policy": {
                "schema": "rtgs.janelle_gaussian2d_image_cell_receipt.v1",
                "bundle_path": "cell_bundle_receipt.json",
                "warmup_cells": 1,
                "measured_cells": 1,
                "validate_before_resume_and_aggregation": True,
                "hash_every_required_artifact": True,
                "strict_semantic_bundle_replay": True,
                "warmup_artifacts": ["field_lift.json", "summary.json"],
                "measured_artifacts": [
                    "field_lift.json",
                    "heldout_metrics.json",
                    "summary.json",
                ],
                "effective_sha256": {"warmup": {"arm": {}}, "measured": {"arm": {}}},
            },
        },
    }
    identities = [
        ("warmup/dataset/arm", "warmup"),
        ("cells/dataset/seed_7/arm", "measured"),
    ]
    entries = []
    for relative, mode in identities:
        cell = run / relative
        cell.mkdir(parents=True)
        effective = {"mode": mode}
        effective_sha256 = CONTRACT._canonical_sha256(effective)
        task["frozen_configuration"]["cell_receipt_policy"]["effective_sha256"][mode]["arm"][
            "7"
        ] = effective_sha256
        input_binding = {
            "camera_records_sha256": "d" * 64,
            "optimizer_validation_image_sha256": "e" * 64,
        }
        summary = {
            "status": "completed",
            "task_id": task_id,
            "dataset_id": "dataset",
            "arm": "arm",
            "seed": 7,
            "warmup": mode == "warmup",
            "effective": effective,
            "input_binding": input_binding,
        }
        if mode == "warmup":
            summary["heldout_outcome_access"] = False
        else:
            summary.update(
                {
                    "heldout_opened_after_endpoint_saved": True,
                    "measurement_endpoint_before_heldout": True,
                    "metrics": {"quality": 1.0},
                }
            )
            summary["input_binding"] = {**input_binding, "heldout_image_sha256": "f" * 64}
            _json(cell / "heldout_metrics.json", {"quality": 1.0})
        _json(cell / "summary.json", summary)
        _json(
            cell / "field_lift.json",
            {
                "manifest_sha256": "1" * 64,
                "loaded_optimizer_compact_sha256": "2" * 64,
                "camera_alignment": {"records_sha256": "d" * 64},
                "optimizer_validation_image_input": {"records_sha256": "e" * 64},
            },
        )
        artifact_names = task["frozen_configuration"]["cell_receipt_policy"][f"{mode}_artifacts"]
        receipt = {
            "schema": "rtgs.janelle_gaussian2d_image_cell_receipt.v1",
            "task_id": task_id,
            "protocol_sha256": lock["protocol_sha256"],
            "task_lock_sha256": _sha256(task_lock),
            "data_seal_sha256": lock["data_seal_sha256"],
            "source_binding_sha256": "c" * 64,
            "dataset_id": "dataset",
            "arm": "arm",
            "seed": 7,
            "mode": mode,
            "iterations": 2 if mode == "warmup" else 10,
            "output_path": cell.relative_to(tmp_path).as_posix(),
            "partition_sha256": CONTRACT._canonical_sha256(
                {
                    "optimizer": ["C1"],
                    "validation": ["C2"],
                    "heldout": ["C3"],
                }
            ),
            "effective_sha256": effective_sha256,
            "input_binding": {
                "manifest_sha256": "1" * 64,
                "compact_optimizer_sha256": "2" * 64,
                "camera_records_sha256": "d" * 64,
                "optimizer_validation_image_sha256": "e" * 64,
            },
            "artifacts": [
                {
                    "path": name,
                    "bytes": (cell / name).stat().st_size,
                    "sha256": _sha256(cell / name),
                }
                for name in artifact_names
            ],
        }
        receipt_path = cell / "cell_receipt.json"
        _json(receipt_path, receipt)
        entries.append(
            {
                "dataset_id": "dataset",
                "arm": "arm",
                "seed": 7,
                "mode": mode,
                "receipt_path": receipt_path.relative_to(run).as_posix(),
                "receipt_bytes": receipt_path.stat().st_size,
                "receipt_sha256": _sha256(receipt_path),
            }
        )
    _json(
        run / "cell_bundle_receipt.json",
        {
            "schema": "rtgs.janelle_gaussian2d_image_cell_bundle.v1",
            "task_id": task_id,
            "protocol_sha256": lock["protocol_sha256"],
            "task_lock_sha256": _sha256(task_lock),
            "data_seal_sha256": lock["data_seal_sha256"],
            "source_binding_sha256": "c" * 64,
            "warmup_cell_count": 1,
            "measured_cell_count": 1,
            "entries": entries,
        },
    )

    return task, run, lock, entries


def test_cell_bundle_validation_is_transitive_to_artifact_bytes(tmp_path: Path) -> None:
    task, run, lock, _entries = _strict_cell_bundle_fixture(tmp_path)

    assert CONTRACT._cell_bundle_errors(run, tmp_path, task, lock) == []
    (run / "cells/dataset/seed_7/arm/summary.json").write_text("tampered\n", encoding="utf-8")
    assert any(
        "cell artifact changed" in error
        for error in CONTRACT._cell_bundle_errors(run, tmp_path, task, lock)
    )


def test_cell_bundle_rejects_omitted_semantics_and_numeric_aliases(tmp_path: Path) -> None:
    task, run, lock, entries = _strict_cell_bundle_fixture(tmp_path)
    receipt_path = run / entries[1]["receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("partition_sha256")
    _json(receipt_path, receipt)
    entries[1]["receipt_bytes"] = receipt_path.stat().st_size
    entries[1]["receipt_sha256"] = _sha256(receipt_path)
    bundle_path = run / "cell_bundle_receipt.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["entries"] = entries
    _json(bundle_path, bundle)
    assert any(
        "wrong keys" in error for error in CONTRACT._cell_bundle_errors(run, tmp_path, task, lock)
    )

    task, run, lock, entries = _strict_cell_bundle_fixture(tmp_path / "alias")
    receipt_path = run / entries[1]["receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["iterations"] = True
    _json(receipt_path, receipt)
    bundle_path = run / "cell_bundle_receipt.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["entries"][1]["receipt_bytes"] = receipt_path.stat().st_size
    bundle["entries"][1]["receipt_sha256"] = _sha256(receipt_path)
    _json(bundle_path, bundle)
    assert any(
        "typed identity/configuration" in error
        for error in CONTRACT._cell_bundle_errors(run, tmp_path / "alias", task, lock)
    )


def test_cell_bundle_rejects_coherent_artifact_inventory_removal(tmp_path: Path) -> None:
    task, run, lock, _entries = _strict_cell_bundle_fixture(tmp_path)
    bundle_path = run / "cell_bundle_receipt.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    receipt_path = run / bundle["entries"][1]["receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"] = [
        item for item in receipt["artifacts"] if item["path"] != "heldout_metrics.json"
    ]
    _json(receipt_path, receipt)
    bundle["entries"][1]["receipt_bytes"] = receipt_path.stat().st_size
    bundle["entries"][1]["receipt_sha256"] = _sha256(receipt_path)
    _json(bundle_path, bundle)
    assert any(
        "artifact inventory differs" in error
        for error in CONTRACT._cell_bundle_errors(run, tmp_path, task, lock)
    )


def test_protocol_review_requires_distinct_identity_and_no_outcome_access(
    tmp_path: Path,
) -> None:
    task = json.loads(LIVE_TASK.read_text(encoding="utf-8"))
    task["owner"] = "Driver"
    task["status"] = "ready"
    digest = CONTRACT.protocol_sha256(task)
    artifact = f"experiments/reviews/{task['task_id']}_PROTOCOL_REVIEW.md"
    task["protocol_review"] = {
        "reviewer": "driver",
        "verdict": "approved",
        "protocol_sha256": digest,
        "artifact": artifact,
    }
    path = tmp_path / artifact
    path.parent.mkdir(parents=True)
    path.write_text(
        _review_body(task["task_id"], digest, "driver", "approved").replace(
            "Outcome Access: `none`", "Outcome Access: `results-read`"
        ),
        encoding="utf-8",
    )
    errors = CONTRACT._validate_protocol_review(task, root=tmp_path)
    assert any("must differ from the task owner" in error for error in errors)
    assert any("Outcome Access must equal 'none'" in error for error in errors)


def test_task_name_is_composed_from_date_task_and_data() -> None:
    task = json.loads(LIVE_TASK.read_text(encoding="utf-8"))
    task["task_id"] = "descriptive_but_unbound"
    errors = CONTRACT.validate_task(task, LIVE_TASK, root=REPO)
    assert any("task_id must equal" in error for error in errors)


def test_new_task_must_explicitly_select_v2() -> None:
    task = json.loads(LIVE_V2_TASK.read_text(encoding="utf-8"))
    task.pop("report_template_version")
    errors = CONTRACT.validate_task(task, LIVE_V2_TASK, root=REPO)
    assert any("must explicitly set report_template_version to 2" in error for error in errors)


def test_compact_arm_rejects_rgb_in_reconstruction_policy() -> None:
    task = json.loads(LIVE_TASK.read_text(encoding="utf-8"))
    task["input_policy"]["reconstruction_allowed"].append("rgb")
    errors = CONTRACT.validate_task(task, LIVE_TASK, root=REPO)
    assert any("calibration and gaussians2d only" in error for error in errors)


def test_direct_compact_arm_rejects_beam_mechanisms() -> None:
    task = json.loads(LIVE_TASK.read_text(encoding="utf-8"))
    task["input_policy"]["reconstruction_forbidden"].remove("rtgs.lift.beam_fusion")
    task["execution_guards"].remove("deny_beam_imports")
    task["stages"].append(
        {
            "id": "beam_fusion",
            "label": "Beam Fusion",
            "purpose": "This mechanism is forbidden in the direct compact arm.",
        }
    )
    errors = CONTRACT.validate_task(task, LIVE_TASK, root=REPO)
    assert any("direct_compact reconstruction_forbidden" in error for error in errors)
    assert any("deny_beam_imports" in error for error in errors)
    assert any("must not contain Beam/carrier" in error for error in errors)


def test_compact_data_seal_contains_no_rgb_or_mask_path() -> None:
    task = json.loads(LIVE_TASK.read_text(encoding="utf-8"))
    seal = json.loads((REPO / task["data_seal"]).read_text(encoding="utf-8"))
    assert seal["input_profile"] == "compact"
    paths = [item["path"] for item in seal["files"]]
    assert paths
    assert not any("/rgb/" in path or "/mask/" in path for path in paths)


def test_compact_data_seal_binds_optional_bundle_production_manifest(
    tmp_path: Path,
) -> None:
    task = json.loads(LIVE_TASK.read_text(encoding="utf-8"))
    task["datasets"] = [copy.deepcopy(task["datasets"][0])]
    task["splits"] = {"frame_00008": task["splits"]["frame_00008"]}
    dataset = task["datasets"][0]
    dataset["frame_path"] = "dataset/frame"
    dataset["calibration"] = "dataset/calibration.json"
    dataset["compact_manifest"] = "dataset/frame/gaussians2d/manifest.json"
    dataset["production_manifest"] = "dataset/frame/gaussians2d/production_manifest.json"
    task["arm"] = "direct_compact"

    (tmp_path / "dataset/frame/gaussians2d").mkdir(parents=True)
    (tmp_path / "dataset/calibration.json").write_text("calibration\n", encoding="utf-8")
    (tmp_path / "dataset/frame/gaussians2d/C0001.rtgsv").write_text("compact\n", encoding="utf-8")
    _json(
        tmp_path / dataset["compact_manifest"],
        {"views": [{"view_id": "C0001", "path": "C0001.rtgsv"}]},
    )
    _json(
        tmp_path / dataset["production_manifest"],
        {"schema": "rtgs.additive_native_bundle_production.v1"},
    )
    task["splits"]["frame_00008"] = {"train": ["C0001"], "heldout": []}

    seal = CONTRACT.build_data_seal(task, root=tmp_path)
    paths = {item["path"] for item in seal["files"]}
    assert dataset["production_manifest"] in paths
    assert seal["datasets"][0]["production_manifest"] == dataset["production_manifest"]


def test_hybrid_rgb_data_seal_binds_compact_rgb_and_masks(tmp_path: Path) -> None:
    task = json.loads(LIVE_TASK.read_text(encoding="utf-8"))
    task["arm"] = "rgb_3dgs"
    task["input_policy"] = {
        "reconstruction_allowed": [
            "calibration",
            "gaussians2d",
            "gaussians3d_initialization",
            "mask",
            "rgb",
        ],
        "reconstruction_forbidden": ["heldout_rgb_for_training"],
        "evaluation_allowed": ["calibration", "gaussians2d", "mask", "rgb"],
    }
    task["datasets"] = [copy.deepcopy(task["datasets"][0])]
    task["splits"] = {"frame_00008": {"train": ["C0001"], "heldout": ["C0002"]}}
    dataset = task["datasets"][0]
    dataset["frame_path"] = "dataset/frame"
    dataset["calibration"] = "dataset/calibration.json"
    dataset["compact_manifest"] = "dataset/frame/gaussians2d/manifest.json"

    compact = tmp_path / "dataset/frame/gaussians2d"
    rgb = tmp_path / "dataset/frame/rgb"
    mask = tmp_path / "dataset/frame/mask"
    compact.mkdir(parents=True)
    rgb.mkdir()
    mask.mkdir()
    (tmp_path / "dataset/calibration.json").write_text("calibration\n", encoding="utf-8")
    views = []
    for view_id in ("C0001", "C0002"):
        (compact / f"{view_id}.rtgsv").write_text("compact\n", encoding="utf-8")
        (rgb / f"{view_id}.jpg").write_text("rgb\n", encoding="utf-8")
        (mask / f"mask_{view_id}.png").write_text("mask\n", encoding="utf-8")
        views.append({"view_id": view_id, "path": f"{view_id}.rtgsv"})
    _json(tmp_path / dataset["compact_manifest"], {"views": views})

    seal = CONTRACT.build_data_seal(task, root=tmp_path)
    paths = {item["path"] for item in seal["files"]}
    assert seal["input_profile"] == "rgb"
    assert seal["datasets"][0]["selected_modalities"] == [
        "calibration",
        "gaussians2d",
        "rgb",
        "mask",
    ]
    assert "dataset/frame/gaussians2d/C0001.rtgsv" in paths
    assert "dataset/frame/rgb/C0001.jpg" in paths
    assert "dataset/frame/mask/mask_C0001.png" in paths


def _report_fixture(tmp_path: Path, *, report_version: int = 2) -> tuple[Path, dict, dict]:
    root = tmp_path
    task = copy.deepcopy(json.loads(LIVE_TASK.read_text(encoding="utf-8")))
    task_id = (
        "20260728_report_contract_fixture"
        if report_version == 2
        else "20260728_vram_claim_stage_frames00008_00009"
    )
    task.update(
        {
            "task_id": task_id,
            "task_slug": "report_contract" if report_version == 2 else "vram_claim",
            "data_slug": "fixture" if report_version == 2 else "stage_frames00008_00009",
            "status": "ready",
            "owner": "test-agent",
            "data_seal": "experiments/data/fixture.json",
            "run_command": ["python", f"scripts/experiments/{task_id}.py"],
            "blockers": [],
        }
    )
    if report_version == 2:
        task["report_template_version"] = 2
    else:
        task.pop("report_template_version", None)
    digest = CONTRACT.protocol_sha256(task)
    review_artifact = f"experiments/reviews/{task['task_id']}_PROTOCOL_REVIEW.md"
    task["protocol_review"] = {
        "reviewer": "test-reviewer",
        "verdict": "approved",
        "protocol_sha256": digest,
        "artifact": review_artifact,
    }
    task_path = root / "experiments" / "tasks" / f"{task_id}.json"
    seal_path = root / "experiments" / "data" / "fixture.json"
    review_path = root / review_artifact
    _json(task_path, task)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
        _review_body(task["task_id"], digest, "test-reviewer", "approved"),
        encoding="utf-8",
    )
    _json(
        seal_path,
        {
            "schema_version": 1,
            "data_slug": "fixture",
            "input_profile": "compact",
            "files": [],
        },
    )
    driver = root / "scripts" / "experiments" / f"{task['task_id']}.py"
    driver.parent.mkdir(parents=True)
    driver.write_text('"""Fixture driver."""\n', encoding="utf-8")

    run = root / "runs" / task["task_id"]
    run.mkdir(parents=True)
    lock = {
        "schema_version": CONTRACT.TASK_LOCK_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "task_path": task_path.relative_to(root).as_posix(),
        "task_sha256": _sha256(task_path),
        "protocol_sha256": digest,
        "protocol_review": task["protocol_review"],
        "protocol_review_artifact_sha256": _sha256(review_path),
        "data_seal_path": seal_path.relative_to(root).as_posix(),
        "data_seal_sha256": _sha256(seal_path),
        "source_commit": "a" * 40,
        "source_dirty": False,
        "source_diff_sha256": "0" * 64,
        "development": False,
        "started_at_utc": "2026-07-28T00:00:00+00:00",
        "command": task["run_command"],
        "report_template_version": report_version,
    }
    _json(run / "task.lock.json", lock)
    for name in ("gaussians_init.ply", "gaussians.ply"):
        (run / name).write_text("ply\nfixture\n", encoding="utf-8")
    _json(run / "gaussians.config.json", {"fit": {"iterations": 2, "learning_rate": 0.01}})
    _json(run / "input_boundary_receipt.json", {"schema_version": 1, "status": "passed"})
    _json(run / "resource_receipt.json", {"schema_version": 1, "wall_seconds": 2.5})
    _json(
        run / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": task["task_id"],
            "status": "completed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": "2026-07-28T00:00:03+00:00",
            "exit_code": 0,
            "failure_phase": None,
            "message": "Fixture completed successfully.",
        },
    )
    _json(
        run / "environment.json",
        {
            "schema_version": 1,
            "python": "3.12.0",
            "platform": "fixture-linux",
            "packages": {"realtime-gs": "fixture"},
            "device": {"type": "cpu", "name": "fixture cpu", "cuda": None},
        },
    )
    history_records = []
    stage_markers = []
    for stage_index, stage in enumerate(task["stages"]):
        start_step = stage_index * 2
        end_step = start_step + 1
        start_seconds = float(start_step)
        end_seconds = float(end_step)
        for metric_index, metric_id in enumerate(("loss_total", "loss_auxiliary")):
            scale = 1.0 + metric_index
            for step, wall_seconds in (
                (start_step, start_seconds),
                (end_step, end_seconds),
            ):
                history_records.append(
                    {
                        "step": step,
                        "wall_seconds": wall_seconds,
                        "stage": stage["id"],
                        "dataset_id": task["datasets"][0]["id"],
                        "arm_id": "fixture_arm",
                        "seed": task["seeds"][0],
                        "split": "train",
                        "metric_id": metric_id,
                        "value": scale / (step + 1.0),
                    }
                )
        for boundary, step, wall_seconds in (
            ("start", start_step, start_seconds),
            ("end", end_step, end_seconds),
        ):
            stage_markers.append(
                {
                    "step": step,
                    "wall_seconds": wall_seconds,
                    "stage": stage["id"],
                    "dataset_id": task["datasets"][0]["id"],
                    "arm_id": "fixture_arm",
                    "seed": task["seeds"][0],
                    "boundary": boundary,
                    "label": stage["label"],
                }
            )
    history = {
        "schema_version": 2,
        "records": history_records,
        "metric_metadata": {
            "loss_total": {
                "label": "Total objective",
                "unit": "loss",
                "group": "Objective",
                "direction": "lower",
            },
            "loss_auxiliary": {
                "label": "Auxiliary objective",
                "unit": "loss",
                "group": "Objective",
                "direction": "lower",
            },
        },
        "stage_markers": stage_markers,
    }
    _json(run / "training_history.json", history)
    evidence = [
        f"benchmarks/results/{task['task_id']}_{suffix}" for suffix in CONTRACT.EVIDENCE_SUFFIXES
    ]
    for path in evidence:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture evidence\n", encoding="utf-8")

    metrics = {
        "schema_version": report_version,
        "report_template_version": report_version,
        "task_id": task["task_id"],
        "summary": "The fixture completed and its canonical report is renderable.",
        "decision": "diagnostic",
        "claim_boundary": task["claim_boundary"],
        "metrics": {},
        "metric_metadata": {},
        "charts": [
            {
                "id": "quality",
                "title": "Quality",
                "unit": "loss",
                "values": [{"label": "fixture", "value": 0.0125}],
            },
            {
                "id": "resources",
                "title": "Resource use",
                "unit": "bytes",
                "values": [{"label": "allocated", "value": 1024}],
            },
            {
                "id": "stage_runtime",
                "title": "Stage runtime",
                "unit": "seconds",
                "values": [{"label": "total", "value": 2.5}],
            },
        ],
        "artifacts": [
            {"label": "Initial", "path": "gaussians_init.ply"},
            {"label": "Final", "path": "gaussians.ply"},
            {"label": "History", "path": "training_history.json"},
            {"label": "Config", "path": "gaussians.config.json"},
            {"label": "Boundary", "path": "input_boundary_receipt.json"},
            {"label": "Resources", "path": "resource_receipt.json"},
        ],
        "evidence": [
            {"label": suffix, "path": path}
            for suffix, path in zip(CONTRACT.EVIDENCE_SUFFIXES, evidence, strict=True)
        ],
        "notes": ["Synthetic structural fixture; not scientific evidence."],
    }
    viewer = [
        ".venv/bin/rtgs",
        "view",
        "--gaussians",
        f"runs/{task['task_id']}/gaussians.ply",
        "--no-open",
    ]
    if report_version == 2:
        metrics["artifacts"].extend(
            [
                {"label": "Run receipt", "path": "run_receipt.json"},
                {"label": "Environment", "path": "environment.json"},
            ]
        )
        metrics["commands"] = {
            "reproduce": task["run_command"],
            "serve_report": [
                ".venv/bin/python",
                "-m",
                "http.server",
                "8765",
                "--directory",
                f"runs/{task['task_id']}",
            ],
            "viewer": viewer,
        }
    else:
        metrics["viewer_command"] = viewer
    if report_version == 2:
        _json(
            run / "viewer_smoke.json",
            {
                "schema_version": 1,
                "status": "passed",
                "viewer_command": viewer,
                "report": {
                    "target": "index.html",
                    "http_status": 200,
                    "local_targets_ok": True,
                },
                "browser": {
                    "name": "fixture-browser",
                    "version": "1.0",
                    "user_agent": "fixture-browser/1.0",
                    "webgl2": True,
                    "renderer": "fixture WebGL renderer",
                },
                "checks": {
                    "viewer_ready": True,
                    "canvas_count": 2,
                    "rendered_content_visible": True,
                    "framebuffer_nonbackground_pixels": 2048,
                    "orbit_camera_changed": True,
                    "client_errors": [],
                    "client_warnings": [],
                },
            },
        )
    else:
        (run / "smoke_receipt.md").write_text(
            "# Smoke receipt\n"
            "- index.html served with HTTP 200\n"
            f"- .venv/bin/rtgs view --gaussians "
            f"runs/{task['task_id']}/gaussians.ply --no-open\n",
            encoding="utf-8",
        )
    values = {
        "peak_cuda_allocated_bytes": 1024,
        "peak_cuda_reserved_bytes": 2048,
        "peak_process_rss_bytes": 4096,
        "wall_seconds": 2.5,
        "heldout_j_pixel": 0.0125,
        "heldout_j_area": 0.013,
        "final_gaussians": 16,
        "compact_input_bytes": 8192,
    }
    task_metrics = {item["id"]: item for item in task["primary_metrics"]}
    metrics["metrics"] = values
    metrics["metric_metadata"] = {
        metric_id: {
            "label": task_metrics[metric_id]["label"],
            "unit": task_metrics[metric_id]["unit"],
            "group": (
                "Quality"
                if metric_id.startswith("heldout") or metric_id.startswith("containment")
                else "Resources"
            ),
            "direction": task_metrics[metric_id]["direction"],
        }
        for metric_id in values
    }
    _json(run / "metrics.json", metrics)
    return run, task, metrics


def test_shared_report_renders_every_required_section(tmp_path: Path) -> None:
    run, task, _metrics = _report_fixture(tmp_path)
    page = CONTRACT.render_run(run, root=tmp_path)
    body = page.read_text(encoding="utf-8")
    assert 'name="rtgs-experiment-report-template" content="2"' in body
    assert task["claim_boundary"] in body
    assert "Input boundary" in body
    assert "Pipeline" in body
    assert "Fitting process" in body
    assert "<svg" in body
    assert "Total objective" in body
    assert "Auxiliary objective" in body
    assert body.count("worker cell wall time from worker start (s)") == 2
    assert body.count('class="stage-boundary"') == len(task["stages"]) * 2 * 2
    assert body.count('data-boundary="start"') == len(task["stages"]) * 2
    assert body.count('data-boundary="end"') == len(task["stages"]) * 2
    for stage in task["stages"]:
        assert body.count(f"<strong>{stage['label']}</strong>: start") == 2
    assert "fitting step" not in body
    assert "Quality" in body
    assert "Resource use" in body
    assert "Stage runtime" in body
    assert "Prospective review" in body
    assert task["protocol_review"]["protocol_sha256"] in body
    assert "gaussians.ply" in body
    assert "fit.iterations" in body
    assert "Start the orbit viewer" in body
    assert (run / "README.md").is_file()
    assert (run / "manifest.json").is_file()
    readme = (run / "README.md").read_text(encoding="utf-8")
    assert "## Effective parameters" in readme
    assert "### Start the orbit viewer" in readme
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert {item["path"] for item in manifest["entries"]} >= {
        "index.html",
        "README.md",
        "metrics.json",
        "training_history.json",
        "viewer_smoke.json",
    }
    assert CONTRACT.validate_run(run, root=tmp_path) == []
    assert BUNDLE.check_bundle(run, previews=False) == []


def test_v2_report_renders_one_canonical_child_page_per_dataset(tmp_path: Path) -> None:
    run, task, metrics = _report_fixture(tmp_path)
    summaries = {}
    for dataset in task["datasets"]:
        summaries[dataset["id"]] = {
            "title": f"{dataset['id']} comparison",
            "summary": "Fixture native-versus-candidate comparison.",
            "metrics": {"quality": 0.25},
            "metric_metadata": {
                "quality": {
                    "label": "Quality",
                    "unit": "MSE",
                    "group": "quality",
                    "direction": "lower",
                }
            },
            "charts": [
                {
                    "id": "quality",
                    "title": "Quality by arm",
                    "unit": "MSE",
                    "values": [
                        {"label": "native", "value": 0.3},
                        {"label": "candidate", "value": 0.2},
                    ],
                }
            ],
            "curves": [
                {
                    "id": "quality",
                    "title": "Quality across seeds",
                    "x_label": "seed",
                    "unit": "MSE",
                    "direction": "lower",
                    "series": [
                        {
                            "label": "candidate",
                            "points": [
                                {"x": 1, "value": 0.3},
                                {"x": 2, "value": 0.2},
                            ],
                        }
                    ],
                }
            ],
            "artifacts": [{"label": "Model", "path": "gaussians.ply"}],
            "commands": {"viewer": metrics["commands"]["viewer"]},
            "notes": ["Fixture child report."],
        }
    metrics["dataset_summaries"] = summaries
    _json(run / "metrics.json", metrics)

    root_page = CONTRACT.render_run(run, root=tmp_path)
    root_body = root_page.read_text(encoding="utf-8")
    assert "Per-dataset reports" in root_body
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {item["path"] for item in manifest["entries"]}
    for dataset in task["datasets"]:
        relative = f"datasets/{dataset['id']}/index.html"
        child = run / relative
        assert child.is_file()
        body = child.read_text(encoding="utf-8")
        assert "All final metrics across measured seeds" in body
        assert "Optimizer and stage curves" in body
        assert "Orbit viewer" in body
        assert relative in root_body
        assert relative in manifest_paths
    assert CONTRACT.validate_run(run, root=tmp_path) == []


def test_v1_report_is_grandfathered(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path, report_version=1)
    page = CONTRACT.render_run(run, root=tmp_path)
    assert 'name="rtgs-experiment-report-template" content="1"' in page.read_text(encoding="utf-8")
    assert not (run / "manifest.json").exists()
    assert CONTRACT.validate_run(run, root=tmp_path) == []


def test_v2_history_rejects_heldout_fitting_records(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    history_path = run / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["records"][0]["split"] = "heldout"
    _json(history_path, history)
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("must not expose heldout/test data" in error for error in errors)


def test_v2_history_requires_every_stage_start_and_end(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    history_path = run / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["stage_markers"].pop()
    _json(history_path, history)
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("ordered start/end boundaries for every frozen stage" in error for error in errors)


def test_v2_history_rejects_records_outside_stage_time_bounds(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    history_path = run / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["records"][1]["wall_seconds"] = 1.5
    _json(history_path, history)
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("follows its stage end boundary" in error for error in errors)


def test_v2_history_binds_boundary_labels_to_frozen_stages(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    history_path = run / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["stage_markers"][0]["label"] = "Unfrozen label"
    _json(history_path, history)
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("stage_markers[0] is invalid" in error for error in errors)


def test_v2_history_reports_a_malformed_boundary_without_crashing(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    history_path = run / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["stage_markers"][0]["boundary"] = []
    _json(history_path, history)
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("stage_markers[0] is invalid" in error for error in errors)


def test_v2_manifest_detects_artifact_tampering(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    CONTRACT.render_run(run, root=tmp_path)
    config_path = run / "gaussians.config.json"
    _json(config_path, {"fit": {"iterations": 99}})
    errors = CONTRACT.validate_run(run, root=tmp_path)
    assert any("manifest SHA-256 mismatch" in error for error in errors)
    assert any(
        "manifest SHA-256 mismatch" in error for error in BUNDLE.check_bundle(run, previews=False)
    )


def test_v2_report_requires_generated_readme(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    CONTRACT.render_run(run, root=tmp_path)
    (run / "README.md").unlink()
    errors = CONTRACT.validate_run(run, root=tmp_path)
    assert any("missing generated README.md" in error for error in errors)


def test_v2_rejects_noncanonical_report_server_command(tmp_path: Path) -> None:
    run, _task, metrics = _report_fixture(tmp_path)
    metrics["commands"]["serve_report"][-1] = "."
    _json(run / "metrics.json", metrics)
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("canonical repository-root HTTP command" in error for error in errors)


def test_v2_bundle_requires_exact_viewer_smoke_command(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    receipt_path = run / "viewer_smoke.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["viewer_command"][0] = "rtgs"
    _json(receipt_path, receipt)
    CONTRACT.render_run(run, root=tmp_path)
    problems = BUNDLE.check_bundle(run, previews=False)
    assert any("viewer_command must exactly match" in problem for problem in problems)


def test_v2_bundle_rejects_http_only_viewer_smoke(tmp_path: Path) -> None:
    run, task, _metrics = _report_fixture(tmp_path)
    (run / "viewer_smoke.json").unlink()
    (run / "smoke_receipt.md").write_text(
        "# Smoke receipt\n"
        "- index.html served with HTTP 200\n"
        f"- .venv/bin/rtgs view --gaussians "
        f"runs/{task['task_id']}/gaussians.ply --no-open\n",
        encoding="utf-8",
    )
    CONTRACT.render_run(run, root=tmp_path)
    problems = BUNDLE.check_bundle(run, previews=False)
    assert any("missing viewer_smoke.json" in problem for problem in problems)


def test_v2_bundle_requires_an_exercised_orbit(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    receipt_path = run / "viewer_smoke.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checks"]["orbit_camera_changed"] = False
    _json(receipt_path, receipt)
    CONTRACT.render_run(run, root=tmp_path)
    problems = BUNDLE.check_bundle(run, previews=False)
    assert any("confirm an orbit changed the camera" in problem for problem in problems)


def test_v2_bundle_rejects_a_blank_viewer_framebuffer(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    receipt_path = run / "viewer_smoke.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checks"]["rendered_content_visible"] = False
    receipt["checks"]["framebuffer_nonbackground_pixels"] = 0
    _json(receipt_path, receipt)
    CONTRACT.render_run(run, root=tmp_path)
    problems = BUNDLE.check_bundle(run, previews=False)
    assert any("confirm visible rendered scene content" in problem for problem in problems)
    assert any(
        "framebuffer_nonbackground_pixels must be at least one" in problem for problem in problems
    )


def test_multi_dataset_bundle_requires_one_exact_browser_smoke_per_child(
    tmp_path: Path,
) -> None:
    dataset_ids = [f"dataset_{index}" for index in range(6)]
    expected = [
        {
            "dataset_id": dataset_id,
            "viewer_command": ["rtgs", "view", "--port", str(8400 + index)],
            "report_target": f"datasets/{dataset_id}/index.html",
        }
        for index, dataset_id in enumerate(dataset_ids)
    ]

    def entry(item):
        return {
            "dataset_id": item["dataset_id"],
            "viewer_command": item["viewer_command"],
            "report": {
                "target": item["report_target"],
                "http_status": 200,
                "local_targets_ok": True,
            },
            "browser": {
                "name": "fixture-browser",
                "version": "1",
                "user_agent": "fixture/1",
                "webgl2": True,
                "renderer": "fixture renderer",
            },
            "checks": {
                "viewer_ready": True,
                "canvas_count": 1,
                "rendered_content_visible": True,
                "framebuffer_nonbackground_pixels": 10,
                "orbit_camera_changed": True,
                "client_errors": [],
                "client_warnings": [],
            },
        }

    _json(
        tmp_path / "viewer_smoke.json",
        {"schema_version": 2, "status": "passed", "entries": [entry(item) for item in expected]},
    )
    assert BUNDLE._check_v2_viewer_smoke(tmp_path, expected) == []

    receipt = json.loads((tmp_path / "viewer_smoke.json").read_text(encoding="utf-8"))
    receipt["entries"].pop()
    _json(tmp_path / "viewer_smoke.json", receipt)
    assert any(
        "entry count differs" in problem
        for problem in BUNDLE._check_v2_viewer_smoke(tmp_path, expected)
    )

    receipt["entries"].append(entry(expected[-1]))
    receipt["entries"][-1]["checks"]["orbit_camera_changed"] = False
    _json(tmp_path / "viewer_smoke.json", receipt)
    assert any(
        "confirm an orbit changed" in problem
        for problem in BUNDLE._check_v2_viewer_smoke(tmp_path, expected)
    )


def test_v2_failure_report_is_renderable_but_not_results_bearing(tmp_path: Path) -> None:
    run, _task, metrics = _report_fixture(tmp_path)
    receipt_path = run / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "status": "failed",
            "exit_code": 17,
            "failure_phase": "fitting",
            "message": "Synthetic optimizer failure.",
        }
    )
    _json(receipt_path, receipt)
    _json(
        run / "training_history.json",
        {"schema_version": 2, "records": [], "metric_metadata": {}, "stage_markers": []},
    )
    metrics["metrics"] = {}
    metrics["metric_metadata"] = {}
    metrics["charts"] = []
    metrics["evidence"] = []
    metrics["commands"]["viewer"] = None
    metrics["artifacts"] = [
        item
        for item in metrics["artifacts"]
        if item["path"] not in {"gaussians_init.ply", "gaussians.ply"}
    ]
    _json(run / "metrics.json", metrics)
    (run / "gaussians_init.ply").unlink()
    (run / "gaussians.ply").unlink()
    page = CONTRACT.render_run(run, root=tmp_path)
    body = page.read_text(encoding="utf-8")
    assert "Run status: failed" in body
    assert "Synthetic optimizer failure" in body
    assert CONTRACT.validate_run(run, root=tmp_path) == []
    problems = BUNDLE.check_bundle(run, previews=False)
    assert any("not a results-bearing bundle" in problem for problem in problems)


def test_report_rejects_a_missing_required_diagram(tmp_path: Path) -> None:
    run, task, metrics = _report_fixture(tmp_path)
    metrics["charts"] = metrics["charts"][:-1]
    _json(run / "metrics.json", metrics)
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("required_charts order" in error for error in errors)


def test_run_lock_rejects_review_artifact_drift(tmp_path: Path) -> None:
    run, task, _metrics = _report_fixture(tmp_path)
    review_path = tmp_path / task["protocol_review"]["artifact"]
    review_path.write_text(
        review_path.read_text(encoding="utf-8") + "\nPost-run edit.\n",
        encoding="utf-8",
    )
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("review artifact changed or disappeared" in error for error in errors)


def test_run_lock_rejects_command_drift(tmp_path: Path) -> None:
    run, _task, _metrics = _report_fixture(tmp_path)
    lock_path = run / "task.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["command"] = ["python", "different_driver.py"]
    _json(lock_path, lock)
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("lock command does not match" in error for error in errors)
