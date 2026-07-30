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
    history = {
        "schema_version": 2,
        "records": [
            {
                "step": step,
                "wall_seconds": float(step),
                "stage": task["stages"][0]["id"],
                "dataset_id": task["datasets"][0]["id"],
                "arm_id": "fixture_arm",
                "seed": task["seeds"][0],
                "split": "train",
                "metric_id": "loss_total",
                "value": value,
            }
            for step, value in ((0, 1.0), (1, 0.5), (2, 0.25))
        ],
        "metric_metadata": {
            "loss_total": {
                "label": "Total objective",
                "unit": "loss",
                "group": "Objective",
                "direction": "lower",
            }
        },
        "stage_markers": [
            {"step": 0, "stage": task["stages"][0]["id"], "label": "Fixture stage start"}
        ],
    }
    _json(run / "training_history.json", history)
    (run / "smoke_receipt.md").write_text(
        "# Smoke receipt\n"
        "- index.html served with HTTP 200\n"
        f"- .venv/bin/rtgs view --gaussians runs/{task['task_id']}/gaussians.ply --no-open\n",
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
    assert "Fixture stage start" in body
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
    }
    assert CONTRACT.validate_run(run, root=tmp_path) == []
    assert BUNDLE.check_bundle(run, previews=False) == []


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
    run, task, _metrics = _report_fixture(tmp_path)
    (run / "smoke_receipt.md").write_text(
        "# Smoke receipt\n"
        "- index.html served with HTTP 200\n"
        f"- rtgs view --gaussians runs/{task['task_id']}/gaussians.ply --no-open\n",
        encoding="utf-8",
    )
    CONTRACT.render_run(run, root=tmp_path)
    problems = BUNDLE.check_bundle(run, previews=False)
    assert any("does not contain commands.viewer exactly" in problem for problem in problems)


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
