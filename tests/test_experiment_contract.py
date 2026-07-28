"""Structural tests for the task-first experiment contract and shared report."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_three_arm_program_is_valid() -> None:
    assert CONTRACT.validate_repository(root=REPO) == []


def test_task_name_is_composed_from_date_task_and_data() -> None:
    task = json.loads(LIVE_TASK.read_text(encoding="utf-8"))
    task["task_id"] = "descriptive_but_unbound"
    errors = CONTRACT.validate_task(task, LIVE_TASK, root=REPO)
    assert any("task_id must equal" in error for error in errors)


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


def _report_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    root = tmp_path
    task = copy.deepcopy(json.loads(LIVE_TASK.read_text(encoding="utf-8")))
    task.update(
        {
            "task_id": "20260728_report_contract_fixture",
            "task_slug": "report_contract",
            "data_slug": "fixture",
            "status": "ready",
            "owner": "test-agent",
            "data_seal": "experiments/data/fixture.json",
            "run_command": ["python", "scripts/experiments/20260728_report_contract_fixture.py"],
            "blockers": [],
        }
    )
    task_path = root / "experiments" / "tasks" / "20260728_report_contract_fixture.json"
    seal_path = root / "experiments" / "data" / "fixture.json"
    _json(task_path, task)
    _json(
        seal_path,
        {
            "schema_version": 1,
            "data_slug": "fixture",
            "input_profile": "compact",
            "files": [],
        },
    )

    run = root / "runs" / task["task_id"]
    run.mkdir(parents=True)
    lock = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "task_path": task_path.relative_to(root).as_posix(),
        "task_sha256": _sha256(task_path),
        "data_seal_path": seal_path.relative_to(root).as_posix(),
        "data_seal_sha256": _sha256(seal_path),
        "source_commit": "a" * 40,
        "source_dirty": False,
        "source_diff_sha256": "0" * 64,
        "development": False,
        "started_at_utc": "2026-07-28T00:00:00+00:00",
        "command": task["run_command"],
        "report_template_version": 1,
    }
    _json(run / "task.lock.json", lock)
    for name in CONTRACT.REQUIRED_MODEL_ARTIFACTS:
        (run / name).write_text("fixture\n", encoding="utf-8")
    (run / "smoke_receipt.md").write_text(
        "# Smoke receipt\n"
        "- index.html served with HTTP 200\n"
        f"- rtgs view --gaussians runs/{task['task_id']}/gaussians.ply --no-open\n",
        encoding="utf-8",
    )
    evidence = [
        f"benchmarks/results/{task['task_id']}_{suffix}" for suffix in CONTRACT.EVIDENCE_SUFFIXES
    ]
    for path in evidence:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture evidence\n", encoding="utf-8")

    metrics = {
        "schema_version": 1,
        "report_template_version": 1,
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
        "viewer_command": [
            ".venv/bin/rtgs",
            "view",
            "--gaussians",
            f"runs/{task['task_id']}/gaussians.ply",
            "--no-open",
        ],
        "notes": ["Synthetic structural fixture; not scientific evidence."],
    }
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
    assert 'name="rtgs-experiment-report-template" content="1"' in body
    assert task["claim_boundary"] in body
    assert "Input boundary" in body
    assert "Pipeline" in body
    assert "Quality" in body
    assert "Resource use" in body
    assert "Stage runtime" in body
    assert "gaussians.ply" in body
    assert CONTRACT.validate_run(run, root=tmp_path) == []
    assert BUNDLE.check_bundle(run, previews=False) == []


def test_report_rejects_a_missing_required_diagram(tmp_path: Path) -> None:
    run, task, metrics = _report_fixture(tmp_path)
    metrics["charts"] = metrics["charts"][:-1]
    _json(run / "metrics.json", metrics)
    errors = CONTRACT.validate_run(run, root=tmp_path, require_index=False)
    assert any("required_charts order" in error for error in errors)
