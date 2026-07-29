"""Contract tests for the preregistered fixed-anchor field-sweep driver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/experiments/20260729_field_sweep_placement_stage_frames00008_00009.py"
TASK = ROOT / "experiments/tasks/20260729_field_sweep_placement_stage_frames00008_00009.json"


def _driver():
    spec = importlib.util.spec_from_file_location("field_sweep_experiment_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summaries(module, *, robust_rgb: float = 0.90) -> tuple[dict, list[dict]]:
    task = {
        "datasets": [{"id": "frame_a"}, {"id": "frame_b"}],
        "seeds": [1, 2, 3],
    }
    rows = []
    for dataset in task["datasets"]:
        for seed in task["seeds"]:
            for arm in module.ARM_ORDER:
                final_rgb = {
                    "bounded_midpoint": 1.0,
                    "all_view_consensus": 0.95,
                    "source_excluded_robust": robust_rgb,
                }[arm]
                rows.append(
                    {
                        "dataset_id": dataset["id"],
                        "seed": seed,
                        "arm": arm,
                        "metrics": {
                            "final_heldout_rgb_mse": final_rgb,
                            "supported_track_fraction": (
                                0.97 if arm == "source_excluded_robust" else 1.0
                            ),
                            "source_projection_max_error": 1e-5,
                        },
                    }
                )
    return task, rows


def test_task_frozen_configuration_matches_driver_constants() -> None:
    module = _driver()
    task = json.loads(TASK.read_text(encoding="utf-8"))
    frozen = task["frozen_configuration"]

    assert frozen["driver"] == DRIVER.relative_to(ROOT).as_posix()
    assert frozen["arms"] == module.ARM_MODES
    assert frozen["compact_loading"] == module.COMPACT_LOADING
    assert frozen["field_lift"] == module.FIELD_CONFIG
    assert frozen["refit"] == module.REFIT_CONFIG
    assert (
        module._protocol_sha256(task)
        == "9ff7057e47e3ea6a2af0532edbecf13036a892d9805e02cfe26aee7f33732844"
    )


def test_run_binding_rejects_data_seal_drift_after_initialization(tmp_path) -> None:
    module = _driver()
    module.ROOT = tmp_path
    task = json.loads(TASK.read_text(encoding="utf-8"))
    task["status"] = "ready"
    review_relative = Path("experiments/reviews") / f"{module.TASK_ID}_PROTOCOL_REVIEW.md"
    review_path = tmp_path / review_relative
    review_path.parent.mkdir(parents=True)
    review_path.write_text("prospective review", encoding="utf-8")
    task["protocol_review"] = {
        "reviewer": "Codex-protocol-reviewer",
        "verdict": "approved",
        "protocol_sha256": module._protocol_sha256(task),
        "artifact": review_relative.as_posix(),
    }
    task_path = tmp_path / module.TASK_RELATIVE
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(task), encoding="utf-8")
    seal_path = tmp_path / task["data_seal"]
    seal_path.parent.mkdir(parents=True)
    seal_path.write_text("sealed-v1", encoding="utf-8")
    run = tmp_path / module.RUN_RELATIVE
    run.mkdir(parents=True)
    lock = {
        "task_id": module.TASK_ID,
        "task_path": module.TASK_RELATIVE.as_posix(),
        "task_sha256": module._sha256_file(task_path),
        "protocol_sha256": module._protocol_sha256(task),
        "protocol_review": task["protocol_review"],
        "protocol_review_artifact_sha256": module._sha256_file(review_path),
        "data_seal_path": task["data_seal"],
        "data_seal_sha256": module._sha256_file(seal_path),
        "command": task["run_command"],
        "development": True,
        "source_dirty": True,
    }
    (run / "task.lock.json").write_text(json.dumps(lock), encoding="utf-8")

    assert module._validate_run_binding(task_path, run)["task_id"] == module.TASK_ID

    seal_path.write_text("sealed-v2", encoding="utf-8")
    with pytest.raises(ValueError, match="data_seal_sha256"):
        module._validate_run_binding(task_path, run)


def test_frozen_producer_decision_applies_ratio_seed_and_guardrail_rules() -> None:
    module = _driver()
    task, rows = _summaries(module)
    decision = module._decision(task, rows)
    assert decision["producer_rule_passed"] is True
    assert decision["robust_to_midpoint_final_rgb_geometric_mean_ratio"] == pytest.approx(0.90)
    assert decision["robust_seed_wins_by_dataset"] == {"frame_a": 3, "frame_b": 3}

    failing_task, failing_rows = _summaries(module, robust_rgb=0.96)
    assert module._decision(failing_task, failing_rows)["producer_rule_passed"] is False


def test_execution_data_seal_rehashes_exact_selected_compact_bytes(tmp_path) -> None:
    module = _driver()
    compact = tmp_path / "data/frame/gaussians2d"
    compact.mkdir(parents=True)
    calibration = tmp_path / "data/calibration.json"
    calibration.write_bytes(b"calibration")
    manifest = compact / "manifest.json"
    manifest.write_bytes(b"manifest")
    view = compact / "C0001.rtgsv"
    view.write_bytes(b"compact-view")

    def record(path: Path) -> dict:
        payload = path.read_bytes()
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    seal_path = tmp_path / "experiments/data/seal.json"
    seal_path.parent.mkdir(parents=True)
    seal = {
        "schema_version": 1,
        "data_slug": "fixture",
        "input_profile": "compact",
        "files": [record(calibration), record(manifest), record(view)],
    }
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    task = {
        "data_slug": "fixture",
        "data_seal": seal_path.relative_to(tmp_path).as_posix(),
        "datasets": [
            {
                "id": "frame",
                "frame_path": "data/frame",
                "calibration": "data/calibration.json",
            }
        ],
    }
    module.ROOT = tmp_path

    receipt = module._verify_data_seal(task, "frame")
    assert receipt["checked_file_count"] == 3
    assert receipt["checked_bytes"] == len(b"calibrationmanifestcompact-view")

    view.write_bytes(b"drift")
    with pytest.raises(ValueError, match="sealed input changed"):
        module._verify_data_seal(task, "frame")


def test_supported_import_route_passes_live_direct_compact_boundary() -> None:
    code = f"""
import importlib.util
from pathlib import Path
path = Path({str(DRIVER)!r})
spec = importlib.util.spec_from_file_location("field_sweep_experiment_driver", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
with module.NoImageGuard() as guard:
    from rtgs.data.compact_views import CompactDataset
    from rtgs.data.field_inputs import SceneFits
    from rtgs.lift.field_lifter import FieldLifter
    assert CompactDataset and SceneFits and FieldLifter
    record = guard.record()
assert record["passed"], record
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
