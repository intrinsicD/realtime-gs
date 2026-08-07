"""Contracts for the reviewed-before-outcome probabilistic-field experiment plan."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rtgs.lift.fiber_correspondence import CorrespondencePlan
from rtgs.lift.field_lifter import FieldLiftConfig

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "experiments/tasks/20260805_probabilistic_field_pipeline_mixed.json"
DRIVER = ROOT / "scripts/experiments/20260805_probabilistic_field_pipeline_mixed.py"
RETRY_TASK = ROOT / "experiments/tasks/20260805_probabilistic_field_pipeline_retry_mixed.json"
RETRY_DRIVER = ROOT / "scripts/experiments/20260805_probabilistic_field_pipeline_retry_mixed.py"
INPUT_RETRY_TASK = (
    ROOT / "experiments/tasks/20260805_probabilistic_field_pipeline_input_retry_mixed.json"
)
INPUT_RETRY_DRIVER = (
    ROOT / "scripts/experiments/20260805_probabilistic_field_pipeline_input_retry_mixed.py"
)
COMPLETION_TASK = (
    ROOT / "experiments/tasks/20260805_probabilistic_field_pipeline_completion_mixed.json"
)
COMPLETION_DRIVER = (
    ROOT / "scripts/experiments/20260805_probabilistic_field_pipeline_completion_mixed.py"
)
SUPPORT_FALLBACK_TASK = (
    ROOT / "experiments/tasks/20260805_probabilistic_field_pipeline_support_fallback_mixed.json"
)
SUPPORT_FALLBACK_DRIVER = (
    ROOT / "scripts/experiments/20260805_probabilistic_field_pipeline_support_fallback_mixed.py"
)
AABB_ELIGIBLE_TASK = (
    ROOT / "experiments/tasks/20260805_probabilistic_field_pipeline_aabb_eligible_mixed.json"
)
AABB_ELIGIBLE_DRIVER = (
    ROOT / "scripts/experiments/20260805_probabilistic_field_pipeline_aabb_eligible_mixed.py"
)
ASSOCIATION_ROLLBACK_TASK = (
    ROOT / "experiments/tasks/20260805_probabilistic_field_pipeline_association_rollback_mixed.json"
)
ASSOCIATION_ROLLBACK_DRIVER = (
    ROOT / "scripts/experiments/20260805_probabilistic_field_pipeline_association_rollback_mixed.py"
)


def _module(driver: Path = DRIVER):
    spec = importlib.util.spec_from_file_location(
        f"probabilistic_field_experiment_{driver.stem}",
        driver,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _immutable_plan_payload(module, task: dict[str, object]) -> dict[str, object]:
    """Replay a historical plan without pretending its bound source is still current."""

    stored = task["frozen_configuration"]["source_binding"]["sha256"]
    namespaces = []
    for function_name in ("plan_payload", "_assert_task_contract", "_base_assert_task_contract"):
        function = getattr(module, function_name, None)
        namespace = getattr(function, "__globals__", None)
        if (
            isinstance(namespace, dict)
            and "_source_tree_sha256" in namespace
            and all(namespace is not item[0] for item in namespaces)
        ):
            namespaces.append((namespace, namespace["_source_tree_sha256"]))
            namespace["_source_tree_sha256"] = lambda: stored
    try:
        return module.plan_payload(task)
    finally:
        for namespace, live_digest in namespaces:
            namespace["_source_tree_sha256"] = live_digest


def test_frozen_protocol_expands_unique_mechanism_isolated_cells() -> None:
    module = _module()
    task = json.loads(TASK.read_text(encoding="utf-8"))
    payload = _immutable_plan_payload(module, task)

    assert payload["outcome_access"] == "guarded_after_review"
    assert payload["result_producer_enabled"] is True
    assert payload["cell_count"] == 549
    assert payload["stage_counts"] == {
        "exact_shape_recovery": 324,
        "recomponentized_association": 60,
        "support_mask_factorial": 81,
        "topology_factorial": 6,
        "schedule_factorial": 6,
        "independent_half_stability": 6,
        "calibrated_compact_operability": 66,
    }
    identifiers = [cell["cell_id"] for cell in payload["cells"]]
    assert len(identifiers) == len(set(identifiers))
    assert task["frozen_configuration"]["calibrated_followup"]["target_component_cap"] == 512
    assert len(task["frozen_configuration"]["source_binding"]["sha256"]) == 64
    assert task["frozen_configuration"]["pipeline"]["association_failure_policy"] == "raise"
    schedule = [cell for cell in payload["cells"] if cell["stage"] == "schedule_factorial"]
    assert schedule
    assert {cell["factors"]["final_cleanup_iterations"] for cell in schedule} == {5}
    assert {
        item["source"]
        for item in payload["cells"]
        if item["stage"] == "calibrated_compact_operability"
    } == {item["id"] for item in task["datasets"]}


def test_infrastructure_retry_preserves_the_reviewable_549_cell_protocol() -> None:
    module = _module(RETRY_DRIVER)
    task = json.loads(RETRY_TASK.read_text(encoding="utf-8"))
    payload = _immutable_plan_payload(module, task)

    assert task["depends_on"] == []
    assert payload["cell_count"] == 549
    assert len(task["frozen_configuration"]["source_binding"]["sha256"]) == 64


def test_infrastructure_retry_guard_accepts_builtin_fromlist_none() -> None:
    code = f"""
import importlib.util
import sys
from pathlib import Path
path = Path({str(RETRY_DRIVER)!r})
spec = importlib.util.spec_from_file_location("retry_guard_subprocess", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
with module.NoImageGuard() as guard:
    builtins_result = __import__("torch", globals(), locals(), None, 0)
    assert builtins_result.__name__ == "torch"
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


def test_input_retry_freezes_loader_and_dilation_contract_per_dataset() -> None:
    module = _module(INPUT_RETRY_DRIVER)
    task = json.loads(INPUT_RETRY_TASK.read_text(encoding="utf-8"))
    payload = _immutable_plan_payload(module, task)
    followup = task["frozen_configuration"]["calibrated_followup"]
    dataset_ids = {item["id"] for item in task["datasets"]}

    assert task["depends_on"] == []
    assert task["status"] == "ready"
    assert payload["cell_count"] == 549
    assert followup["projection_dilation"] == 0.0
    assert set(followup["compact_view_byte_caps"]) == dataset_ids
    assert set(followup["compact_view_byte_caps"].values()) == {168_000, 8_388_608}
    calibrated = [
        item for item in payload["cells"] if item["stage"] == "calibrated_compact_operability"
    ]
    assert len(calibrated) == 66
    assert {item["factors"]["projection_dilation"] for item in calibrated} == {0.0}
    assert {item["factors"]["compact_view_byte_cap"] for item in calibrated} == {168_000, 8_388_608}
    assert (
        module._calibrated_config(
            task, seed=task["seeds"][0], arm="all_candidate_mechanisms"
        ).projection_dilation
        == 0.0
    )
    source = INPUT_RETRY_DRIVER.read_text(encoding="utf-8")
    assert "byte_cap=compact_view_byte_cap" in source
    assert "observation_dilations != [projection_dilation]" in source


def test_completion_retry_preserves_gates_and_freezes_failure_accounting() -> None:
    module = _module(COMPLETION_DRIVER)
    task = json.loads(COMPLETION_TASK.read_text(encoding="utf-8"))
    payload = _immutable_plan_payload(module, task)
    followup = task["frozen_configuration"]["calibrated_followup"]
    calibrated = [
        item for item in payload["cells"] if item["stage"] == "calibrated_compact_operability"
    ]

    assert task["status"] == "ready"
    assert task["protocol_review"] == {
        "reviewer": "Codex-probabilistic-field-protocol-reviewer",
        "verdict": "approved",
        "protocol_sha256": "1b0382109dfaaba43807daf59afac6e646c5cfb6f062e428be50ae3544b257dd",
        "artifact": (
            "experiments/reviews/"
            "20260805_probabilistic_field_pipeline_completion_mixed_PROTOCOL_REVIEW.md"
        ),
    }
    assert task["depends_on"] == []
    assert payload["cell_count"] == 549
    assert followup["cell_failure_policy"] == "continue_structured_hard_invariant_failure"
    assert (
        followup["continued_failure_receipt_contract"] == module.CONTINUED_FAILURE_RECEIPT_CONTRACT
    )
    assert followup["failure_reporting_contract"] == module.FAILURE_REPORTING_CONTRACT
    assert followup["conditional_aggregate_metrics"] == module.CONDITIONAL_AGGREGATE_METRICS
    assert followup["preserve_rejected_models_for_viewer"] is True
    assert {item["factors"]["cell_failure_policy"] for item in calibrated} == {
        "continue_structured_hard_invariant_failure"
    }
    assert {item["factors"]["preserve_rejected_models_for_viewer"] for item in calibrated} == {True}
    assert {
        item["factors"]["continued_failure_receipt_contract_sha256"] for item in calibrated
    } == {module._canonical_sha256(module.CONTINUED_FAILURE_RECEIPT_CONTRACT)}
    assert {item["factors"]["failure_reporting_contract_sha256"] for item in calibrated} == {
        module._canonical_sha256(module.FAILURE_REPORTING_CONTRACT)
    }
    assert {item["factors"]["conditional_aggregate_metrics_sha256"] for item in calibrated} == {
        module._canonical_sha256(module.CONDITIONAL_AGGREGATE_METRICS)
    }
    assert task["frozen_configuration"]["invariant_gates"]["minimum_transport_real_mass"] == 1e-10
    assert task["frozen_configuration"]["pipeline"]["association_failure_policy"] == "raise"
    assert len(task["frozen_configuration"]["source_binding"]["sha256"]) == 64


def test_completion_accepts_only_guarded_structured_field_fit_failure(
    tmp_path: Path,
) -> None:
    module = _module(COMPLETION_DRIVER)
    task = json.loads(COMPLETION_TASK.read_text(encoding="utf-8"))
    dataset = task["datasets"][0]
    dataset_id = dataset["id"]
    seed = task["seeds"][0]
    arm = "all_candidate_mechanisms"
    followup = task["frozen_configuration"]["calibrated_followup"]
    byte_cap, projection_dilation = module._calibrated_input_contract(task, dataset_id)
    cell = tmp_path / "cells" / dataset_id / f"seed_{seed}" / arm
    cell.mkdir(parents=True)
    rejected_artifacts = []
    for name in ("rejected_gaussians_init.ply", "rejected_gaussians.ply"):
        path = cell / name
        path.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
        rejected_artifacts.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": module._sha256_file(path),
            }
        )
    loaded_compact_files = module._expected_compact_file_records(dataset)
    input_bytes = sum(record["bytes"] for record in loaded_compact_files)
    message = "hard invariant violation: transport real mass"
    context = {
        "dataset_id": dataset_id,
        "seed": seed,
        "arm": arm,
        "warmup": False,
        "guard_passed": True,
        "rejected_model_artifacts": rejected_artifacts,
        "rejection_preservation_error": None,
    }
    failure = {
        "schema_version": 1,
        "task_id": module.TASK_ID,
        "status": "failed",
        "phase": "field_fit",
        "failed_at_utc": "2026-08-05T00:00:00+00:00",
        "exception_type": "RuntimeError",
        "message": message,
        "traceback": f"RuntimeError: {message}\n",
        "context": context,
    }
    boundary = {
        "schema_version": 1,
        "task_id": module.TASK_ID,
        "status": "failed",
        "dataset_id": dataset_id,
        "seed": seed,
        "arm": arm,
        "warmup": False,
        "failure_phase": "field_fit",
        "allowed_modalities": ["calibration", "gaussians2d"],
        "compact_alpha_loaded": True,
        "external_mask_access": False,
        "heldout_training_access": False,
        "loaded_compact_files": loaded_compact_files,
        "input_bytes": input_bytes,
        "guard": {
            "passed": True,
            "denied_paths": 0,
            "denied_imports": 0,
            "negative_control_denials": 3,
            "forbidden_modules_loaded": [],
        },
        "component_cap": followup["target_component_cap"],
        "compact_view_byte_cap": byte_cap,
        "observation_aa_dilations": [projection_dilation],
        "projection_dilation": projection_dilation,
        "rejected_model_artifacts": rejected_artifacts,
        "rejection_preservation_error": None,
    }
    resource = {
        "schema_version": 1,
        "task_id": module.TASK_ID,
        "status": "failed",
        "dataset_id": dataset_id,
        "seed": seed,
        "arm": arm,
        "warmup": False,
        "failure_phase": "field_fit",
        "cpu_threads": 1,
        "torch_threads": 1,
        "cpu_model": "fixture-cpu",
        "cuda_used": False,
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "input_bytes": input_bytes,
        "output_bytes": sum(record["bytes"] for record in rejected_artifacts),
        "fit_wall_seconds": 2.0,
        "wall_seconds": 3.0,
        "process_wall_seconds": 4.0,
        "stage_wall_seconds": {"compact_loading": 1.0, "field_fit": 2.0},
        "ru_maxrss_bytes": 4096,
    }
    (cell / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
    (cell / "input_boundary_receipt.json").write_text(json.dumps(boundary), encoding="utf-8")
    (cell / "resource_receipt.json").write_text(json.dumps(resource), encoding="utf-8")

    loaded = module._load_structured_field_fit_failure(
        cell,
        task=task,
        dataset=dataset,
        seed=seed,
        arm=arm,
    )
    assert loaded["failure"]["phase"] == "field_fit"

    boundary["guard"]["passed"] = False
    (cell / "input_boundary_receipt.json").write_text(json.dumps(boundary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not safely continuable"):
        module._load_structured_field_fit_failure(
            cell,
            task=task,
            dataset=dataset,
            seed=seed,
            arm=arm,
        )
    boundary["guard"]["passed"] = True
    failure["exception_type"] = "ValueError"
    (cell / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
    (cell / "input_boundary_receipt.json").write_text(json.dumps(boundary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not safely continuable"):
        module._load_structured_field_fit_failure(
            cell,
            task=task,
            dataset=dataset,
            seed=seed,
            arm=arm,
        )

    failure["exception_type"] = "RuntimeError"
    forged_failure = json.loads(json.dumps(failure))
    forged_failure["context"]["guard_passed"] = False
    forged_boundary = json.loads(json.dumps(boundary))
    forged_boundary.update(
        {
            "dataset_id": "wrong-dataset",
            "seed": -1,
            "arm": "wrong-arm",
            "warmup": True,
            "external_mask_access": True,
            "heldout_training_access": True,
        }
    )
    forged_boundary["guard"]["passed"] = True
    forged_boundary["guard"]["violations"] = ["opened image.png"]
    forged_resource = {
        "task_id": module.TASK_ID,
        "status": "failed",
        "dataset_id": "wrong-dataset",
        "seed": -1,
        "arm": "wrong-arm",
        "warmup": True,
        "failure_phase": "field_fit",
    }
    (cell / "failure.json").write_text(json.dumps(forged_failure), encoding="utf-8")
    (cell / "input_boundary_receipt.json").write_text(json.dumps(forged_boundary), encoding="utf-8")
    (cell / "resource_receipt.json").write_text(json.dumps(forged_resource), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not safely continuable"):
        module._load_structured_field_fit_failure(
            cell,
            task=task,
            dataset=dataset,
            seed=seed,
            arm=arm,
        )


def test_completion_serve_report_command_is_schema_exact() -> None:
    module = _module(COMPLETION_DRIVER)
    contract = _module(ROOT / "scripts/experiment_contract.py")
    task = json.loads(COMPLETION_TASK.read_text(encoding="utf-8"))
    commands = {
        "reproduce": task["run_command"],
        "serve_report": module._serve_report_command(),
        "viewer": [
            ".venv/bin/rtgs",
            "view",
            "--comparison-manifest",
            f"runs/{module.TASK_ID}/datasets/fixture/viewer_comparison.json",
        ],
    }
    assert commands["serve_report"][0] == ".venv/bin/python"
    assert (
        contract._v2_commands_errors(
            commands,
            {"command": task["run_command"], "task_id": module.TASK_ID},
            completed=True,
        )
        == []
    )


def test_completion_dataset_curves_show_failures_without_metric_imputation(
    tmp_path: Path,
) -> None:
    module = _module(COMPLETION_DRIVER)
    task = json.loads(COMPLETION_TASK.read_text(encoding="utf-8"))
    dataset = task["datasets"][0]
    source_run = tmp_path / "run"
    publish_root = tmp_path / "staging"
    publish_root.mkdir()
    successes = []
    failures = []
    for seed in task["seeds"]:
        for arm in module.CALIBRATED_ARMS:
            cell = source_run / "cells" / dataset["id"] / f"seed_{seed}" / arm
            cell.mkdir(parents=True)
            if arm == "all_candidate_mechanisms" and seed == task["seeds"][0]:
                (cell / "failure.json").write_text("{}", encoding="utf-8")
                failures.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "arm": arm,
                        "cell": cell,
                        "failure": {"message": "hard invariant violation: transport real mass"},
                    }
                )
                continue
            successes.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "arm": arm,
                    "cell": cell,
                    "summary": {
                        "metrics": {
                            "heldout_field_rgb_mse": float(seed),
                            "peak_rss_bytes": 10.0,
                            "refit_wall_seconds": 2.0,
                            "embedded_alpha_view_fraction": 0.0,
                        }
                    },
                }
            )

    summary = module._dataset_summary(
        task,
        source_run,
        publish_root,
        dataset,
        successes,
        failures,
        port=8300,
    )
    status_curve = summary["curves"][0]
    candidate = next(
        item for item in status_curve["series"] if item["label"] == "all candidate mechanisms"
    )
    assert [point["value"] for point in candidate["points"]] == [0.0, 1.0, 1.0]
    assert summary["metrics"][
        "all_candidate_mechanisms_calibrated_cell_success_fraction"
    ] == pytest.approx(2 / 3)
    candidate_quality = next(
        item for item in summary["curves"] if item["id"] == "heldout_field_rgb_mse"
    )["series"][1]
    assert [point["x"] for point in candidate_quality["points"]] == task["seeds"][1:]
    contract = _module(ROOT / "scripts/experiment_contract.py")
    dataset_task = {**task, "datasets": [dataset]}
    assert (
        contract._dataset_summary_errors({dataset["id"]: summary}, dataset_task, completed=True)
        == []
    )


def test_completion_all_failure_dataset_is_explicit_and_schema_valid(tmp_path: Path) -> None:
    module = _module(COMPLETION_DRIVER)
    task = json.loads(COMPLETION_TASK.read_text(encoding="utf-8"))
    dataset = task["datasets"][0]
    source_run = tmp_path / "run"
    publish_root = tmp_path / "staging"
    publish_root.mkdir()
    failures = []
    for seed in task["seeds"]:
        for arm in module.CALIBRATED_ARMS:
            cell = source_run / "cells" / dataset["id"] / f"seed_{seed}" / arm
            cell.mkdir(parents=True)
            (cell / "failure.json").write_text("{}", encoding="utf-8")
            failures.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "arm": arm,
                    "cell": cell,
                    "failure": {"message": "hard invariant violation: transport real mass"},
                }
            )

    summary = module._dataset_summary(
        task,
        source_run,
        publish_root,
        dataset,
        successes=[],
        failures=failures,
        port=8300,
    )
    for arm in module.CALIBRATED_ARMS:
        assert summary["metrics"][f"{arm}_calibrated_cell_success_fraction"] == 0.0
        assert summary["metrics"][f"{arm}_successful_cell_count"] == 0.0
        assert summary["metrics"][f"{arm}_attempt_count"] == 3.0
    assert all(chart["values"] for chart in summary["charts"])
    assert all("unavailable" in chart["title"] for chart in summary["charts"])
    assert all(chart["unit"] == "successful cells" for chart in summary["charts"])
    assert {value["value"] for chart in summary["charts"] for value in chart["values"]} == {0.0}
    assert {
        point["value"] for series in summary["curves"][0]["series"] for point in series["points"]
    } == {0.0}

    contract = _module(ROOT / "scripts/experiment_contract.py")
    dataset_task = {**task, "datasets": [dataset]}
    assert (
        contract._dataset_summary_errors({dataset["id"]: summary}, dataset_task, completed=True)
        == []
    )
    history = module._history_bundle(task, [], failures)
    assert contract._history_errors(history, task, completed=True) == []
    assert {record["metric_id"] for record in history["records"]} == {"calibrated_cell_success"}
    assert {record["value"] for record in history["records"]} == {0.0}
    assert module._median_or_none([]) is None
    assert module._conditional_metric_result([]) == {
        "value": None,
        "successful_cell_count": 0,
    }
    assert module._conditional_metric_result([1.0, 3.0]) == {
        "value": 2.0,
        "successful_cell_count": 2,
    }
    primary_ids = {item["id"] for item in task["primary_metrics"]}
    assert primary_ids.isdisjoint({item["id"] for item in module.CONDITIONAL_AGGREGATE_METRICS})


def test_completion_coordinator_continues_one_validated_field_fit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(COMPLETION_DRIVER)
    task = json.loads(COMPLETION_TASK.read_text(encoding="utf-8"))
    run = tmp_path / "run"
    measured_calls = []
    aggregate_calls = []
    validation_calls = []

    def fake_internal_command(
        _task_path: Path,
        _run: Path,
        *,
        synthetic: bool = False,
        synthetic_cell_id: str | None = None,
        dataset_id: str | None = None,
        seed: int | None = None,
        arm: str | None = None,
        warmup: bool = False,
    ) -> list[str]:
        assert synthetic_cell_id is None
        if synthetic:
            return ["synthetic"]
        return [
            "cell",
            str(dataset_id),
            str(seed),
            str(arm),
            "warmup" if warmup else "measured",
        ]

    def fake_run(command: list[str], **_kwargs: object) -> None:
        if command[0] not in {"cell", "synthetic"}:
            return
        if command[0] == "synthetic" or command[-1] == "warmup":
            return
        _kind, dataset_id, seed_text, arm, _mode = command
        seed = int(seed_text)
        measured_calls.append((dataset_id, seed, arm))
        if len(measured_calls) != 2:
            return
        cell = run / module._cell_relative(dataset_id, seed, arm, False)
        cell.mkdir(parents=True)
        raise subprocess.CalledProcessError(1, command)

    def fake_load_failure(
        cell: Path,
        *,
        task: dict[str, object],
        dataset: dict[str, object],
        seed: int,
        arm: str,
    ) -> dict[str, object]:
        validation_calls.append((cell, task["task_id"], dataset["id"], seed, arm))
        return {"failure": {"message": "hard invariant violation: transport real mass"}}

    monkeypatch.setattr(module, "_internal_command", fake_internal_command)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_load_structured_field_fit_failure", fake_load_failure)
    monkeypatch.setattr(
        module,
        "_publish_aggregate",
        lambda received_task, received_run: aggregate_calls.append(
            (received_task["task_id"], received_run)
        ),
    )

    module._orchestrate_body(COMPLETION_TASK, run, task)

    assert len(measured_calls) == 66
    assert len(validation_calls) == 1
    assert aggregate_calls == [(module.TASK_ID, run)]


def test_completion_terminal_count_excludes_worker_temp_directories(tmp_path: Path) -> None:
    module = _module(COMPLETION_DRIVER)
    task = json.loads(COMPLETION_TASK.read_text(encoding="utf-8"))
    run = tmp_path / "run"
    expected_cells = [
        run / module._cell_relative(dataset["id"], seed, arm, False)
        for dataset in task["datasets"]
        for seed in task["seeds"]
        for arm in module.CALIBRATED_ARMS
    ]
    for cell in expected_cells[:65]:
        cell.mkdir(parents=True)
        (cell / "summary.json").write_text("{}", encoding="utf-8")
    final = expected_cells[65]
    temporary = final.with_name(f".{final.name}.worker-123")
    temporary.mkdir(parents=True)
    (temporary / "failure.json").write_text("{}", encoding="utf-8")

    assert module._measured_terminal_count(task, run) == 65
    final.mkdir(parents=True)
    (final / "failure.json").write_text("{}", encoding="utf-8")
    assert module._measured_terminal_count(task, run) == 66
    (final / "summary.json").write_text("{}", encoding="utf-8")
    assert module._measured_terminal_count(task, run) == 65


@pytest.mark.parametrize(
    ("aggregation_started", "expected_phase"),
    [(False, "orchestration"), (True, "aggregation")],
)
def test_completion_root_failure_phase_uses_actual_aggregation_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregation_started: bool,
    expected_phase: str,
) -> None:
    module = _module(COMPLETION_DRIVER)
    task = json.loads(COMPLETION_TASK.read_text(encoding="utf-8"))
    run = tmp_path / expected_phase
    run.mkdir()
    (run / "task.lock.json").write_text(
        json.dumps({"started_at_utc": "2026-08-05T00:00:00+00:00"}),
        encoding="utf-8",
    )

    def fail_body(
        _task_path: Path,
        received_run: Path,
        received_task: dict[str, object],
        *,
        orchestration_state: dict[str, bool],
    ) -> None:
        for dataset in received_task["datasets"]:
            for seed in received_task["seeds"]:
                for arm in module.CALIBRATED_ARMS:
                    cell = received_run / module._cell_relative(dataset["id"], seed, arm, False)
                    cell.mkdir(parents=True)
                    (cell / "failure.json").write_text("{}", encoding="utf-8")
        orchestration_state["aggregation_started"] = aggregation_started
        raise RuntimeError("fixture terminal failure")

    monkeypatch.setattr(module, "_orchestrate_body", fail_body)
    with pytest.raises(RuntimeError, match="fixture terminal failure"):
        module._orchestrate(COMPLETION_TASK, run, task)

    failure = json.loads((run / "failure.json").read_text(encoding="utf-8"))
    receipt = json.loads((run / "run_receipt.json").read_text(encoding="utf-8"))
    assert failure["phase"] == expected_phase
    assert failure["context"]["measured_cell_count"] == 66
    assert receipt["failure_phase"] == expected_phase


def test_support_fallback_retries_the_entire_fit_once_and_discloses_every_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(SUPPORT_FALLBACK_DRIVER)
    task = json.loads(SUPPORT_FALLBACK_TASK.read_text(encoding="utf-8"))

    @dataclass(frozen=True)
    class FakePlacement:
        diagnostics: dict[str, object]

    @dataclass(frozen=True)
    class FakeResult:
        placement: FakePlacement
        diagnostics: dict[str, object]

    @dataclass(frozen=True)
    class FakePipeline:
        reconstruction: FakeResult
        half_reconstructions: tuple[FakeResult, FakeResult] | None
        stability: object

    def fake_result(mask_mode: str) -> FakeResult:
        diagnostics = {"mask_mode": mask_mode}
        return FakeResult(FakePlacement(dict(diagnostics)), dict(diagnostics))

    calls: list[str] = []

    def fake_once(**kwargs: object):
        config = kwargs["config"]
        assert isinstance(config, FieldLiftConfig)
        calls.append(config.mask_mode)
        if len(calls) == 1:
            raise ValueError(module.EMPTY_SUPPORT_EXCEPTION_MESSAGE)
        primary = fake_result(config.mask_mode)
        halves = (fake_result(config.mask_mode), fake_result(config.mask_mode))
        pipeline = FakePipeline(primary, halves, SimpleNamespace())
        return primary, pipeline.stability, pipeline

    monkeypatch.setattr(module, "_run_calibrated_fit_once", fake_once)
    result, _stability, pipeline, effective, record = (
        module._run_calibrated_fit_with_support_fallback(
            task=task,
            fits=SimpleNamespace(),
            config=FieldLiftConfig(mask_mode="hard"),
            seed=task["frozen_configuration"]["calibrated_followup"]["independent_half_seed"],
            warmup=False,
        )
    )

    assert calls == ["hard", "none"]
    assert effective.mask_mode == "none"
    assert record == {
        "requested_mask_mode": "hard",
        "effective_mask_mode": "none",
        "used": True,
        "retry_count": 1,
        "checked_fit_count": 3,
        "fallback_fit_count": 3,
        "rng_reset_seed": task["frozen_configuration"]["calibrated_followup"][
            "independent_half_seed"
        ],
        "trigger_exception_type": "ValueError",
        "trigger_exception_message": module.EMPTY_SUPPORT_EXCEPTION_MESSAGE,
        "interpretation": "unmasked_operability_only_not_mask_mode_evidence",
    }
    assert pipeline is not None and pipeline.half_reconstructions is not None
    checked = (result, *pipeline.half_reconstructions)
    assert all(item.diagnostics["requested_mask_mode"] == "hard" for item in checked)
    assert all(item.diagnostics["effective_mask_mode"] == "none" for item in checked)
    assert all(item.diagnostics["unmasked_support_fallback_used"] is True for item in checked)
    module._enforce_support_fallback_provenance(
        result,
        pipeline,
        record,
        seed=task["frozen_configuration"]["calibrated_followup"]["independent_half_seed"],
    )
    with pytest.raises(RuntimeError, match="provenance values are inconsistent"):
        module._enforce_support_fallback_provenance(
            result,
            pipeline,
            record | {"effective_mask_mode": "hard"},
            seed=task["frozen_configuration"]["calibrated_followup"]["independent_half_seed"],
        )


@pytest.mark.parametrize(
    "error",
    [
        ValueError("different field-fit input error"),
        RuntimeError("support-mask policy rejected every field-placement source"),
    ],
)
def test_support_fallback_is_exact_exception_only(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    module = _module(SUPPORT_FALLBACK_DRIVER)
    task = json.loads(SUPPORT_FALLBACK_TASK.read_text(encoding="utf-8"))
    calls = []

    def fail_once(**kwargs: object):
        calls.append(kwargs["config"])
        raise error

    monkeypatch.setattr(module, "_run_calibrated_fit_once", fail_once)
    with pytest.raises(type(error), match=str(error)):
        module._run_calibrated_fit_with_support_fallback(
            task=task,
            fits=SimpleNamespace(),
            config=FieldLiftConfig(mask_mode="probability"),
            seed=80502,
            warmup=False,
        )
    assert len(calls) == 1


def test_support_fallback_success_provenance_must_match_every_serialized_surface(
    tmp_path: Path,
) -> None:
    module = _module(SUPPORT_FALLBACK_DRIVER)
    task = json.loads(SUPPORT_FALLBACK_TASK.read_text(encoding="utf-8"))
    seed = task["frozen_configuration"]["calibrated_followup"]["independent_half_seed"]
    arm = "native_controls"
    dataset_id = task["datasets"][0]["id"]
    record = {
        "requested_mask_mode": "hard",
        "effective_mask_mode": "none",
        "used": True,
        "retry_count": 1,
        "checked_fit_count": 3,
        "fallback_fit_count": 3,
        "rng_reset_seed": seed,
        "trigger_exception_type": "ValueError",
        "trigger_exception_message": module.EMPTY_SUPPORT_EXCEPTION_MESSAGE,
        "interpretation": "unmasked_operability_only_not_mask_mode_evidence",
    }
    summary = {
        "task_id": module.TASK_ID,
        "dataset_id": dataset_id,
        "seed": seed,
        "arm": arm,
        "warmup": False,
        "support_fallback": record,
        "metrics": {
            "unmasked_support_fallback_used": 1.0,
            "unmasked_support_fallback_retry_count": 1.0,
            "unmasked_support_fallback_fit_count": 3.0,
        },
        "diagnostics": {
            "mask_mode": "none",
            "requested_mask_mode": "hard",
            "effective_mask_mode": "none",
            "unmasked_support_fallback_used": True,
            "unmasked_support_fallback_retry_count": 1,
            "unmasked_support_fallback_trigger": module.EMPTY_SUPPORT_EXCEPTION_MESSAGE,
            "unmasked_support_fallback_interpretation": module.SUPPORT_FALLBACK_INTERPRETATION,
        },
    }
    boundary = {"support_fallback": record}
    resource = {"support_fallback": record}
    config = module._support_fallback_config_payload(
        task,
        dataset_id=dataset_id,
        seed=seed,
        arm=arm,
        warmup=False,
        support_fallback=record,
    )
    (tmp_path / "gaussians.config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "gaussians_init.ply").write_text("fixture", encoding="utf-8")
    (tmp_path / "gaussians.ply").write_text("fixture", encoding="utf-8")

    module._validate_serialized_support_fallback(
        task=task,
        cell=tmp_path,
        summary=summary,
        boundary=boundary,
        resource_record=resource,
        dataset_id=dataset_id,
        seed=seed,
        arm=arm,
    )
    with pytest.raises(RuntimeError, match="provenance is inconsistent"):
        module._validate_serialized_support_fallback(
            task=task,
            cell=tmp_path,
            summary=summary,
            boundary={"support_fallback": record | {"effective_mask_mode": "hard"}},
            resource_record=resource,
            dataset_id=dataset_id,
            seed=seed,
            arm=arm,
        )
    for key, tampered_value in (
        ("unmasked_support_fallback_retry_count", 0),
        ("unmasked_support_fallback_trigger", None),
        ("unmasked_support_fallback_interpretation", None),
    ):
        tampered_summary = json.loads(json.dumps(summary))
        tampered_summary["diagnostics"][key] = tampered_value
        with pytest.raises(RuntimeError, match="provenance is inconsistent"):
            module._validate_serialized_support_fallback(
                task=task,
                cell=tmp_path,
                summary=tampered_summary,
                boundary=boundary,
                resource_record=resource,
                dataset_id=dataset_id,
                seed=seed,
                arm=arm,
            )
    original_config = (tmp_path / "gaussians.config.json").read_text(encoding="utf-8")
    for field, tampered_value, diagnostic_key, metric_key in (
        (
            "retry_count",
            True,
            "unmasked_support_fallback_retry_count",
            "unmasked_support_fallback_retry_count",
        ),
        (
            "retry_count",
            1.0,
            "unmasked_support_fallback_retry_count",
            "unmasked_support_fallback_retry_count",
        ),
        ("checked_fit_count", 3.0, None, None),
        (
            "fallback_fit_count",
            3.0,
            None,
            "unmasked_support_fallback_fit_count",
        ),
        ("rng_reset_seed", float(seed), None, None),
    ):
        tampered_record = json.loads(json.dumps(record))
        tampered_record[field] = tampered_value
        tampered_summary = json.loads(json.dumps(summary))
        tampered_summary["support_fallback"] = tampered_record
        if diagnostic_key is not None:
            tampered_summary["diagnostics"][diagnostic_key] = tampered_value
        if metric_key is not None:
            tampered_summary["metrics"][metric_key] = tampered_value
        tampered_config = json.loads(original_config)
        tampered_config["support_fallback"] = tampered_record
        (tmp_path / "gaussians.config.json").write_text(
            json.dumps(tampered_config), encoding="utf-8"
        )
        try:
            with pytest.raises(RuntimeError):
                module._validate_serialized_support_fallback(
                    task=task,
                    cell=tmp_path,
                    summary=tampered_summary,
                    boundary={"support_fallback": tampered_record},
                    resource_record={"support_fallback": tampered_record},
                    dataset_id=dataset_id,
                    seed=seed,
                    arm=arm,
                )
        finally:
            (tmp_path / "gaussians.config.json").write_text(original_config, encoding="utf-8")
    for metric_key, tampered_value in (
        ("unmasked_support_fallback_used", True),
        ("unmasked_support_fallback_retry_count", 1),
        ("unmasked_support_fallback_fit_count", 3),
    ):
        tampered_summary = json.loads(json.dumps(summary))
        tampered_summary["metrics"][metric_key] = tampered_value
        with pytest.raises(RuntimeError, match="provenance is inconsistent"):
            module._validate_serialized_support_fallback(
                task=task,
                cell=tmp_path,
                summary=tampered_summary,
                boundary=boundary,
                resource_record=resource,
                dataset_id=dataset_id,
                seed=seed,
                arm=arm,
            )
    one_fit_record = module._expected_support_fallback_record(
        task,
        seed=task["seeds"][1],
        arm=arm,
        warmup=False,
        used=True,
    )
    for field, tampered_value in (
        ("retry_count", True),
        ("checked_fit_count", True),
        ("fallback_fit_count", True),
        ("retry_count", 1.0),
        ("checked_fit_count", 1.0),
        ("fallback_fit_count", 1.0),
        ("rng_reset_seed", float(task["seeds"][1])),
    ):
        assert not module._support_fallback_record_is_exact(
            one_fit_record | {field: tampered_value}, one_fit_record
        )
    no_fallback_record = module._expected_support_fallback_record(
        task,
        seed=task["seeds"][1],
        arm=arm,
        warmup=False,
        used=False,
    )
    for field, tampered_value in (
        ("used", 0),
        ("rng_reset_seed", 0),
        ("trigger_exception_type", "ValueError"),
        ("trigger_exception_message", module.EMPTY_SUPPORT_EXCEPTION_MESSAGE),
        ("interpretation", module.SUPPORT_FALLBACK_INTERPRETATION),
    ):
        assert not module._support_fallback_record_is_exact(
            no_fallback_record | {field: tampered_value}, no_fallback_record
        )


def test_support_fallback_hard_gate_failure_preserves_and_validates_every_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(SUPPORT_FALLBACK_DRIVER)
    task = json.loads(SUPPORT_FALLBACK_TASK.read_text(encoding="utf-8"))
    dataset = task["datasets"][0]
    dataset_id = dataset["id"]
    seed = task["seeds"][1]
    arm = "native_controls"
    manifest = json.loads((ROOT / dataset["compact_manifest"]).read_text(encoding="utf-8"))
    _byte_cap, projection_dilation = module._calibrated_input_contract(task, dataset_id)
    views = tuple(
        SimpleNamespace(
            view_id=item["view_id"],
            observation=SimpleNamespace(aa_dilation=projection_dilation),
        )
        for item in manifest["views"]
    )
    compact = SimpleNamespace(views=views, n_views=len(views))
    train, heldout = module._split_indices(compact, task["splits"][dataset_id])
    fits = SimpleNamespace(
        train_view_indices=train,
        heldout_view_indices=heldout,
        n_views=len(views),
    )

    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.data.compact_views import CompactDataset
    from rtgs.data.field_inputs import SceneFits

    model = Gaussians3D.from_means_covs(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.eye(3, dtype=torch.float64).unsqueeze(0) * 0.01,
        torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    record = module._expected_support_fallback_record(
        task,
        seed=seed,
        arm=arm,
        warmup=False,
        used=True,
    )
    diagnostics = {
        "mask_mode": "none",
        "requested_mask_mode": "hard",
        "effective_mask_mode": "none",
        "unmasked_support_fallback_used": True,
        "unmasked_support_fallback_retry_count": 1,
        "unmasked_support_fallback_trigger": module.EMPTY_SUPPORT_EXCEPTION_MESSAGE,
        "unmasked_support_fallback_interpretation": module.SUPPORT_FALLBACK_INTERPRETATION,
    }
    fit_result = SimpleNamespace(
        diagnostics=diagnostics,
        gaussians_init=model,
        gaussians=model,
        optimized_view_indices=train,
        heldout_view_indices=heldout,
        semantic_validation=SimpleNamespace(heldout=SimpleNamespace()),
        placement_semantic_validation=SimpleNamespace(heldout=SimpleNamespace()),
    )
    requested = module._calibrated_config(task, seed=seed, arm=arm)
    effective = replace(requested, mask_mode="none")

    class FixtureGuard:
        def __enter__(self):
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        @staticmethod
        def record() -> dict[str, object]:
            return {
                "passed": True,
                "denied_paths": 0,
                "denied_imports": 0,
                "negative_control_denials": 3,
                "forbidden_modules_loaded": [],
            }

    monkeypatch.setattr(module, "NoImageGuard", FixtureGuard)
    monkeypatch.setattr(CompactDataset, "load", staticmethod(lambda *_args, **_kwargs: compact))
    monkeypatch.setattr(
        SceneFits,
        "from_compact_dataset",
        staticmethod(lambda *_args, **_kwargs: fits),
    )
    monkeypatch.setattr(torch, "set_num_threads", lambda _count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _count: None)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda _enabled: None)
    monkeypatch.setattr(
        module,
        "_run_calibrated_fit_with_support_fallback",
        lambda **_kwargs: (fit_result, None, None, effective, record),
    )

    def reject_invariants(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("hard invariant violation: fixture gate")

    monkeypatch.setattr(module, "_enforce_result_invariants", reject_invariants)
    run = tmp_path / "run"
    with pytest.raises(RuntimeError, match="hard invariant violation: fixture gate"):
        module._calibrated_worker(
            task=task,
            run=run,
            dataset_id=dataset_id,
            seed=seed,
            arm=arm,
            warmup=False,
        )

    cell = run / module._cell_relative(dataset_id, seed, arm, False)
    loaded = module._load_structured_field_fit_failure(
        cell,
        task=task,
        dataset=dataset,
        seed=seed,
        arm=arm,
    )
    assert loaded["support_fallback"] == record
    presentation_source = module._presentation_model_source(
        arm,
        successes=[],
        failures=[loaded],
        preferred_seed=seed,
    )
    assert presentation_source is not None
    assert presentation_source["status"] == "rejected"
    assert presentation_source["support_fallback"] == record
    surfaces = {
        "failure.json": ("context", "support_fallback"),
        "input_boundary_receipt.json": ("support_fallback",),
        "resource_receipt.json": ("support_fallback",),
        "gaussians.config.json": ("support_fallback",),
        "gaussians.config.json#effective": ("field_lift_effective",),
    }
    for surface, keys in surfaces.items():
        filename = surface.split("#", maxsplit=1)[0]
        path = cell / filename
        original = path.read_text(encoding="utf-8")
        payload = json.loads(original)
        if keys == ("context", "support_fallback"):
            payload["context"]["support_fallback"]["effective_mask_mode"] = "hard"
        elif keys == ("support_fallback",):
            payload["support_fallback"]["effective_mask_mode"] = "hard"
        else:
            payload["field_lift_effective"]["mask_mode"] = "hard"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            with pytest.raises(RuntimeError, match="not safely continuable"):
                module._load_structured_field_fit_failure(
                    cell,
                    task=task,
                    dataset=dataset,
                    seed=seed,
                    arm=arm,
                )
        finally:
            path.write_text(original, encoding="utf-8")

    for filename in (
        "failure.json",
        "input_boundary_receipt.json",
        "resource_receipt.json",
        "gaussians.config.json",
    ):
        path = cell / filename
        original = path.read_text(encoding="utf-8")
        payload = json.loads(original)
        if filename == "failure.json":
            payload["context"]["support_fallback"]["retry_count"] = True
        else:
            payload["support_fallback"]["retry_count"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            with pytest.raises(RuntimeError, match="not safely continuable"):
                module._load_structured_field_fit_failure(
                    cell,
                    task=task,
                    dataset=dataset,
                    seed=seed,
                    arm=arm,
                )
        finally:
            path.write_text(original, encoding="utf-8")

    no_fallback = module._expected_support_fallback_record(
        task,
        seed=seed,
        arm=arm,
        warmup=False,
        used=False,
    )
    for filename in ("failure.json", "input_boundary_receipt.json", "resource_receipt.json"):
        path = cell / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        if filename == "failure.json":
            payload["context"]["support_fallback"] = no_fallback
        else:
            payload["support_fallback"] = no_fallback
        path.write_text(json.dumps(payload), encoding="utf-8")
    (cell / "gaussians.config.json").write_text(
        json.dumps(
            module._support_fallback_config_payload(
                task,
                dataset_id=dataset_id,
                seed=seed,
                arm=arm,
                warmup=False,
                support_fallback=no_fallback,
            )
        ),
        encoding="utf-8",
    )
    assert (
        module._load_structured_field_fit_failure(
            cell,
            task=task,
            dataset=dataset,
            seed=seed,
            arm=arm,
        )["support_fallback"]
        == no_fallback
    )


def test_support_fallback_mixed_outcome_keeps_rejection_in_report_and_viewer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(SUPPORT_FALLBACK_DRIVER)
    task = json.loads(SUPPORT_FALLBACK_TASK.read_text(encoding="utf-8"))
    dataset = task["datasets"][0]
    arm = "native_controls"
    rejected_seed = task["seeds"][0]
    accepted_seed = task["seeds"][1]
    masked_rejected_seed = task["seeds"][2]
    accepted_cell = tmp_path / "accepted"
    rejected_cell = tmp_path / "rejected"
    masked_rejected_cell = tmp_path / "masked_rejected"
    accepted_cell.mkdir()
    rejected_cell.mkdir()
    masked_rejected_cell.mkdir()

    from PIL import Image

    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.data.compact_views import CompactDataset

    model = Gaussians3D.from_means_covs(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.eye(3, dtype=torch.float64).unsqueeze(0) * 0.01,
        torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    for directory, names in (
        (accepted_cell, ("gaussians_init.ply", "gaussians.ply")),
        (
            rejected_cell,
            ("rejected_gaussians_init.ply", "rejected_gaussians.ply"),
        ),
        (
            masked_rejected_cell,
            ("rejected_gaussians_init.ply", "rejected_gaussians.ply"),
        ),
    ):
        for name in names:
            model.save_ply(directory / name)

    accepted_record = module._expected_support_fallback_record(
        task,
        seed=accepted_seed,
        arm=arm,
        warmup=False,
        used=False,
    )
    rejected_record = module._expected_support_fallback_record(
        task,
        seed=rejected_seed,
        arm=arm,
        warmup=False,
        used=True,
    )
    masked_rejected_record = module._expected_support_fallback_record(
        task,
        seed=masked_rejected_seed,
        arm=arm,
        warmup=False,
        used=False,
    )
    successes = [
        {
            "arm": arm,
            "seed": accepted_seed,
            "cell": accepted_cell,
            "summary": {
                "support_fallback": accepted_record,
                "metrics": {
                    "heldout_field_rgb_mse": 0.1,
                    "peak_rss_bytes": 4096.0,
                    "refit_wall_seconds": 1.0,
                },
            },
        }
    ]
    failures = [
        {
            "arm": arm,
            "seed": rejected_seed,
            "cell": rejected_cell,
            "failure": {"message": "hard invariant violation: fixture gate"},
            "support_fallback": rejected_record,
        },
        {
            "arm": arm,
            "seed": masked_rejected_seed,
            "cell": masked_rejected_cell,
            "failure": {"message": "hard invariant violation: second fixture gate"},
            "support_fallback": masked_rejected_record,
        },
    ]
    fake_compact = SimpleNamespace(
        views=[SimpleNamespace(camera=SimpleNamespace(position=torch.tensor([0.0, 0.0, 2.0])))]
    )
    monkeypatch.setattr(
        CompactDataset, "load", staticmethod(lambda *_args, **_kwargs: fake_compact)
    )
    monkeypatch.setattr(module, "_scaled_camera", lambda camera: camera)
    monkeypatch.setattr(module, "_orbit_cameras", lambda *_args, **_kwargs: [None])
    monkeypatch.setattr(
        module,
        "_image_from_render",
        lambda *_args, **_kwargs: Image.new("RGB", (8, 8), "black"),
    )

    publish_root = tmp_path / "published"
    receipt = module._save_dataset_presentation(
        task,
        publish_root,
        dataset,
        successes,
        failures,
        preferred_seed=rejected_seed,
    )
    manifest = json.loads(
        (publish_root / "datasets" / dataset["id"] / "viewer_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["methods"]) == 3
    labels = [item["name"] for item in manifest["methods"]]
    assert any(f"seed {accepted_seed}" in label and "accepted" in label for label in labels)
    assert any(
        f"seed {rejected_seed}" in label
        and "presentation-only" in label
        and "effective unmasked fallback" in label
        for label in labels
    )
    assert any(
        f"seed {masked_rejected_seed}" in label
        and "presentation-only" in label
        and "no unmasked fallback" in label
        for label in labels
    )
    assert receipt["representative_models"][arm]["seed"] == accepted_seed
    assert receipt["rejected_models"] == [
        {
            "arm": arm,
            "seed": rejected_seed,
            "status": "rejected",
            "failure": "hard invariant violation: fixture gate",
            "support_fallback": rejected_record,
            "presentation_id": f"{arm}_seed_{rejected_seed}_rejected",
        },
        {
            "arm": arm,
            "seed": masked_rejected_seed,
            "status": "rejected",
            "failure": "hard invariant violation: second fixture gate",
            "support_fallback": masked_rejected_record,
            "presentation_id": f"{arm}_seed_{masked_rejected_seed}_rejected",
        },
    ]
    report = module._dataset_summary(
        task,
        tmp_path,
        publish_root,
        dataset,
        successes,
        failures,
        port=8300,
    )
    assert (
        "1 rejected cells also used that fallback and remain presentation-only" in report["summary"]
    )
    failure_notes = [note for note in report["notes"] if "failed during field_fit" in note]
    fallback_note = next(note for note in failure_notes if f"seed {rejected_seed}" in note)
    masked_note = next(note for note in failure_notes if f"seed {masked_rejected_seed}" in note)
    assert "Preserved rejected model is presentation-only" in fallback_note
    assert "requested=hard, effective=none, used=true" in fallback_note
    assert f"rng_reset_seed={rejected_seed}" in fallback_note
    assert module.SUPPORT_FALLBACK_INTERPRETATION in fallback_note
    assert "requested=hard, effective=hard, used=false" in masked_note
    assert "rng_reset_seed=none, trigger=none, interpretation=none" in masked_note
    assert any(
        f"{arm}_seed_{rejected_seed}_rejected_gaussians.ply" in item["path"]
        for item in report["artifacts"]
    )
    assert any(
        f"{arm}_seed_{masked_rejected_seed}_rejected_gaussians.ply" in item["path"]
        for item in report["artifacts"]
    )


def test_support_fallback_plan_preserves_science_and_discloses_the_retry() -> None:
    module = _module(SUPPORT_FALLBACK_DRIVER)
    completion = _module(COMPLETION_DRIVER)
    task = json.loads(SUPPORT_FALLBACK_TASK.read_text(encoding="utf-8"))
    completion_task = json.loads(COMPLETION_TASK.read_text(encoding="utf-8"))
    payload = _immutable_plan_payload(module, task)
    prior = _immutable_plan_payload(completion, completion_task)

    assert task["depends_on"] == []
    assert payload["cell_count"] == prior["cell_count"] == 549
    assert payload["stage_counts"] == prior["stage_counts"]
    prior_by_id = {item["cell_id"]: item for item in prior["cells"]}
    for cell in payload["cells"]:
        old = prior_by_id[cell["cell_id"]]
        if cell["stage"] != "calibrated_compact_operability":
            assert cell == old
            continue
        added = {
            "empty_support_policy",
            "empty_support_exception_type",
            "empty_support_exception_message",
            "empty_support_max_retries",
            "empty_support_retry_rng_reset",
            "fallback_interpretation",
            "continued_failure_receipt_contract_sha256",
        }
        assert {key: value for key, value in cell["factors"].items() if key not in added} == {
            key: value for key, value in old["factors"].items() if key not in added
        }
        assert cell["factors"]["continued_failure_receipt_contract_sha256"] == (
            module._canonical_sha256(module.CONTINUED_FAILURE_RECEIPT_CONTRACT)
        )
        assert cell["factors"]["empty_support_policy"] == module.EMPTY_SUPPORT_POLICY
        assert cell["factors"]["empty_support_exception_type"] == "ValueError"
        assert (
            cell["factors"]["empty_support_exception_message"]
            == module.EMPTY_SUPPORT_EXCEPTION_MESSAGE
        )
        assert cell["factors"]["empty_support_max_retries"] == 1
        assert cell["factors"]["empty_support_retry_rng_reset"] == ("torch_manual_seed_cell_seed")
    assert any(
        item["id"] == "unmasked_support_fallback_cell_fraction" for item in task["primary_metrics"]
    )


def test_aabb_eligible_successor_preserves_cells_and_binds_only_geometry_domain_repair() -> None:
    module = _module(AABB_ELIGIBLE_DRIVER)
    task = json.loads(AABB_ELIGIBLE_TASK.read_text(encoding="utf-8"))
    payload = _immutable_plan_payload(module, task)

    assert task["status"] == "ready"
    assert task["depends_on"] == []
    assert task["protocol_review"]["verdict"] == "approved"
    assert len(task["protocol_review"]["protocol_sha256"]) == 64
    assert task["blockers"] == []
    assert payload["cell_count"] == 549
    assert module._canonical_sha256(payload["cells"]) == (
        "1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc"
    )
    configuration = task["frozen_configuration"]
    assert len(configuration["source_binding"]["sha256"]) == 64
    assert configuration["base_driver_binding"] == {
        "path": module.BASE_DRIVER_RELATIVE.as_posix(),
        "algorithm": "sha256-bytes-v1",
        "sha256": module.BASE_DRIVER_SHA256,
    }
    assert configuration["fixed_anchor_eligibility"] == module.ANCHOR_ELIGIBILITY_CONTRACT


def test_aabb_eligible_draft_to_ready_transition_preserves_protocol_digest() -> None:
    module = _module(AABB_ELIGIBLE_DRIVER)
    ready_task = json.loads(AABB_ELIGIBLE_TASK.read_text(encoding="utf-8"))
    task = json.loads(json.dumps(ready_task))
    task["status"] = "draft"
    task["protocol_review"] = {
        "reviewer": None,
        "verdict": "pending",
        "protocol_sha256": None,
        "artifact": None,
    }
    reviewed_digest = module._protocol_sha256(task)
    ready = json.loads(json.dumps(task))
    ready["status"] = "ready"
    ready["protocol_review"] = {
        "reviewer": "distinct-reviewer",
        "verdict": "approved",
        "protocol_sha256": reviewed_digest,
        "artifact": "experiments/reviews/review.md",
    }

    assert ready["blockers"] == []
    assert module._protocol_sha256(ready) == reviewed_digest


def test_aabb_eligible_successor_rejects_task_policy_or_base_driver_tampering() -> None:
    module = _module(AABB_ELIGIBLE_DRIVER)
    task = json.loads(AABB_ELIGIBLE_TASK.read_text(encoding="utf-8"))

    changed_policy = json.loads(json.dumps(task))
    changed_policy["frozen_configuration"]["fixed_anchor_eligibility"]["policy"] = (
        "accept_every_ray"
    )
    with pytest.raises(ValueError, match="fixed-anchor eligibility contract"):
        _immutable_plan_payload(module, changed_policy)

    changed_base = json.loads(json.dumps(task))
    changed_base["frozen_configuration"]["base_driver_binding"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="base-driver binding"):
        _immutable_plan_payload(module, changed_base)


def test_aabb_eligibility_diagnostic_validator_is_json_type_strict() -> None:
    module = _module(AABB_ELIGIBLE_DRIVER)
    diagnostics = {
        "anchor_eligibility_policy": "forward_search_aabb_intersection_v1",
        "n_init_3d": 3,
        "anchor_candidate_count": 6,
        "anchor_forward_aabb_eligible_count": 5,
        "anchor_forward_aabb_rejected_count": 1,
        "anchor_forward_aabb_eligible_counts_per_view": [1, 2, 2],
        "anchor_forward_aabb_rejected_counts_per_view": [1, 0, 0],
    }
    result = SimpleNamespace(
        diagnostics={"placement": diagnostics, "target_component_counts_used": [2, 2, 2]},
        optimized_view_indices=(0, 1, 2),
    )
    module._validate_anchor_eligibility_diagnostics(result)

    for key, value in (
        ("anchor_forward_aabb_eligible_count", True),
        ("anchor_forward_aabb_rejected_count", 1.0),
        ("anchor_candidate_count", 5),
    ):
        changed = dict(diagnostics)
        changed[key] = value
        with pytest.raises(RuntimeError):
            module._validate_anchor_eligibility_diagnostics(
                SimpleNamespace(
                    diagnostics={
                        "placement": changed,
                        "target_component_counts_used": [2, 2, 2],
                    },
                    optimized_view_indices=(0, 1, 2),
                )
            )


def test_aabb_eligibility_diagnostic_validator_binds_each_optimized_view_capacity() -> None:
    module = _module(AABB_ELIGIBLE_DRIVER)
    placement = {
        "anchor_eligibility_policy": "forward_search_aabb_intersection_v1",
        "n_init_3d": 3,
        "anchor_candidate_count": 6,
        "anchor_forward_aabb_eligible_count": 5,
        "anchor_forward_aabb_rejected_count": 1,
        "anchor_forward_aabb_eligible_counts_per_view": [1, 2, 2],
        "anchor_forward_aabb_rejected_counts_per_view": [1, 0, 0],
    }

    def result(
        changed_placement: dict[str, object] | None = None,
        component_counts: object = (2, 2, 2),
        optimized_views: object = (0, 1, 2),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            diagnostics={
                "placement": placement if changed_placement is None else changed_placement,
                "target_component_counts_used": list(component_counts),
            },
            optimized_view_indices=optimized_views,
        )

    for budget in (0, -1):
        changed = dict(placement)
        changed["n_init_3d"] = budget
        with pytest.raises(RuntimeError, match="requested budget"):
            module._validate_anchor_eligibility_diagnostics(result(changed))

    redistributed = dict(placement)
    redistributed["anchor_forward_aabb_eligible_counts_per_view"] = [1, 3, 1]
    redistributed["anchor_forward_aabb_rejected_counts_per_view"] = [1, 0, 0]
    with pytest.raises(RuntimeError, match="per-view eligibility exceeds"):
        module._validate_anchor_eligibility_diagnostics(result(redistributed))

    false_total = dict(placement)
    false_total["anchor_candidate_count"] = 5
    false_total["anchor_forward_aabb_eligible_count"] = 4
    false_total["anchor_forward_aabb_eligible_counts_per_view"] = [1, 2, 1]
    false_total["anchor_forward_aabb_rejected_counts_per_view"] = [1, 0, 0]
    false_total["anchor_forward_aabb_rejected_count"] = 1
    with pytest.raises(RuntimeError):
        module._validate_anchor_eligibility_diagnostics(result(false_total))

    for capacities in ((2, 2), (2, 2.0, 2), (2, True, 2)):
        with pytest.raises(RuntimeError, match="candidate-capacity"):
            module._validate_anchor_eligibility_diagnostics(result(component_counts=capacities))

    with pytest.raises(RuntimeError, match="candidate-capacity"):
        module._validate_anchor_eligibility_diagnostics(result(optimized_views=(0, 0, 2)))


def test_association_rollback_successor_preserves_cells_and_binds_failure_completion() -> None:
    module = _module(ASSOCIATION_ROLLBACK_DRIVER)
    prior_module = _module(AABB_ELIGIBLE_DRIVER)
    task = json.loads(ASSOCIATION_ROLLBACK_TASK.read_text(encoding="utf-8"))
    prior_task = json.loads(AABB_ELIGIBLE_TASK.read_text(encoding="utf-8"))
    payload = _immutable_plan_payload(module, task)
    prior = _immutable_plan_payload(prior_module, prior_task)

    assert task["status"] == "ready"
    assert task["depends_on"] == []
    assert task["protocol_review"] == {
        "reviewer": "Codex-probabilistic-field-protocol-reviewer",
        "verdict": "approved",
        "protocol_sha256": "e57d58112fd6f95467e8ddacdb4daad7fc9d83ed48b8b9f336a32b1966a92e87",
        "artifact": (
            "experiments/reviews/"
            "20260805_probabilistic_field_pipeline_association_rollback_mixed_"
            "PROTOCOL_REVIEW.md"
        ),
    }
    assert task["blockers"] == []
    assert payload["cell_count"] == prior["cell_count"] == 549
    assert payload["cells"] == prior["cells"]
    configuration = task["frozen_configuration"]
    assert len(configuration["source_binding"]["sha256"]) == 64
    assert configuration["base_driver_binding"] == {
        "path": module.BASE_DRIVER_RELATIVE.as_posix(),
        "algorithm": "sha256-bytes-v1",
        "sha256": module.BASE_DRIVER_SHA256,
    }
    assert configuration["fixed_anchor_eligibility"] == module.ANCHOR_ELIGIBILITY_CONTRACT
    assert configuration["association_failure_completion"] == module.ASSOCIATION_ROLLBACK_CONTRACT
    assert configuration["pipeline"]["association_failure_policy"] == "rollback"
    candidate = module._calibrated_config(
        task,
        seed=task["seeds"][0],
        arm="all_candidate_mechanisms",
    )
    native = module._calibrated_config(
        task,
        seed=task["seeds"][0],
        arm="native_controls",
    )
    assert candidate.association is not None
    assert candidate.association.failure_policy == "rollback"
    assert native.association is None
    assert (ROOT / "runs" / task["task_id"]).is_dir()


def test_association_rollback_draft_to_ready_transition_preserves_protocol_digest() -> None:
    module = _module(ASSOCIATION_ROLLBACK_DRIVER)
    task = json.loads(ASSOCIATION_ROLLBACK_TASK.read_text(encoding="utf-8"))
    reviewed_digest = module._protocol_sha256(task)
    ready = json.loads(json.dumps(task))
    ready["status"] = "ready"
    ready["protocol_review"] = {
        "reviewer": "distinct-reviewer",
        "verdict": "approved",
        "protocol_sha256": reviewed_digest,
        "artifact": "experiments/reviews/review.md",
    }

    assert ready["blockers"] == []
    assert module._protocol_sha256(ready) == reviewed_digest


def test_association_rollback_successor_rejects_policy_or_contract_tampering() -> None:
    module = _module(ASSOCIATION_ROLLBACK_DRIVER)
    task = json.loads(ASSOCIATION_ROLLBACK_TASK.read_text(encoding="utf-8"))

    changed_policy = json.loads(json.dumps(task))
    changed_policy["frozen_configuration"]["pipeline"]["association_failure_policy"] = "raise"
    with pytest.raises(ValueError, match="transactional rollback"):
        _immutable_plan_payload(module, changed_policy)

    changed_contract = json.loads(json.dumps(task))
    changed_contract["frozen_configuration"]["association_failure_completion"][
        "failed_cell_semantics"
    ] = "treat_as_success"
    with pytest.raises(ValueError, match="association-rollback contract"):
        _immutable_plan_payload(module, changed_contract)


def _association_result_fixture(
    *,
    status: str,
    association: object | None,
    failure: str | None,
) -> SimpleNamespace:
    placement_diagnostics: dict[str, object] = {
        "anchor_eligibility_policy": "forward_search_aabb_intersection_v1",
        "n_init_3d": 3,
        "anchor_candidate_count": 6,
        "anchor_forward_aabb_eligible_count": 5,
        "anchor_forward_aabb_rejected_count": 1,
        "anchor_forward_aabb_eligible_counts_per_view": [1, 2, 2],
        "anchor_forward_aabb_rejected_counts_per_view": [1, 0, 0],
        "association_status": status,
    }
    diagnostics: dict[str, object] = {
        "placement": placement_diagnostics,
        "target_component_counts_used": [2, 2, 2],
        "association_status": status,
    }
    if failure is not None:
        placement_diagnostics["association_failure"] = failure
        diagnostics["association_failure"] = failure
    return SimpleNamespace(
        diagnostics=diagnostics,
        placement=SimpleNamespace(diagnostics=placement_diagnostics),
        optimized_view_indices=(0, 1, 2),
        association=association,
        association_failure=failure,
    )


def test_association_rollback_diagnostics_are_exact_and_type_strict() -> None:
    module = _module(ASSOCIATION_ROLLBACK_DRIVER)
    committed = _association_result_fixture(
        status="committed",
        association=object(),
        failure=None,
    )
    rolled_back = _association_result_fixture(
        status="rolled_back",
        association=None,
        failure="RuntimeError: a supported projection left the valid camera domain during M-step",
    )
    module._validate_association_completion_diagnostics(
        committed,
        association_required=True,
    )
    module._validate_association_completion_diagnostics(
        rolled_back,
        association_required=True,
    )

    malformed = [
        _association_result_fixture(status="committed", association=None, failure=None),
        _association_result_fixture(status="rolled_back", association=None, failure=None),
        _association_result_fixture(
            status="rolled_back",
            association=None,
            failure="KeyError: unsupported type",
        ),
        _association_result_fixture(
            status="rolled_back",
            association=None,
            failure="RuntimeError: ",
        ),
        _association_result_fixture(
            status="rolled_back",
            association=object(),
            failure="ValueError: fixture",
        ),
    ]
    mismatch = _association_result_fixture(
        status="rolled_back",
        association=None,
        failure="ValueError: fixture",
    )
    mismatch.diagnostics["association_failure"] = "ValueError: forged"
    malformed.append(mismatch)
    for result in malformed:
        with pytest.raises(RuntimeError, match="association diagnostics"):
            module._validate_association_completion_diagnostics(
                result,
                association_required=True,
            )


def test_association_rollback_still_reaches_the_existing_missing_transport_hard_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ASSOCIATION_ROLLBACK_DRIVER)
    rolled_back = _association_result_fixture(
        status="rolled_back",
        association=None,
        failure="RuntimeError: a supported projection left the valid camera domain during M-step",
    )

    def hard_gate(_task: object, result: object, *, association_required: bool):
        assert result is rolled_back
        assert association_required is True
        raise RuntimeError("hard invariant violation: transport plan missing, transport real mass")

    monkeypatch.setattr(module, "_base_enforce_result_invariants", hard_gate)
    with pytest.raises(RuntimeError, match="hard invariant violation: transport plan missing"):
        module._enforce_result_invariants({}, rolled_back, association_required=True)


def test_completion_viewer_falls_back_to_labeled_rejected_model(tmp_path: Path) -> None:
    module = _module(COMPLETION_DRIVER)
    cell = tmp_path / "failed"
    cell.mkdir()
    for name in ("rejected_gaussians_init.ply", "rejected_gaussians.ply"):
        (cell / name).write_text("fixture", encoding="utf-8")
    source = module._presentation_model_source(
        "all_candidate_mechanisms",
        successes=[],
        failures=[
            {
                "arm": "all_candidate_mechanisms",
                "seed": 80501,
                "cell": cell,
                "failure": {"message": "hard invariant violation: transport real mass"},
            }
        ],
        preferred_seed=80501,
    )

    assert source is not None
    assert source["status"] == "rejected"
    assert source["failure"] == "hard invariant violation: transport real mass"


def test_hard_invariant_probes_are_exact_and_failures_are_structured() -> None:
    module = _module()
    conservation = module._split_conservation_invariants()
    assert conservation == {
        "split_density_mass_error": 0.0,
        "split_optical_thickness_error": 0.0,
    }
    failure = module._failure_payload(phase="fixture", error=RuntimeError("boom"))
    assert failure["status"] == "failed"
    assert failure["phase"] == "fixture"
    assert failure["exception_type"] == "RuntimeError"
    assert "boom" in failure["traceback"]


def test_heldout_isolation_is_measured_from_realized_indices() -> None:
    module = _module()
    task = json.loads(TASK.read_text(encoding="utf-8"))
    fits = SimpleNamespace(train_view_indices=(0, 1), heldout_view_indices=(2,), n_views=3)
    isolated = SimpleNamespace(optimized_view_indices=(0, 1), heldout_view_indices=(2,))
    leaked = SimpleNamespace(optimized_view_indices=(0, 2), heldout_view_indices=(2,))

    assert module._heldout_fit_access_count(task, isolated, fits) == 0
    with pytest.raises(RuntimeError, match="held-out view entered fitting"):
        module._heldout_fit_access_count(task, leaked, fits)

    first = SimpleNamespace(optimized_view_indices=(0,), heldout_view_indices=(1, 2))
    second = SimpleNamespace(optimized_view_indices=(1,), heldout_view_indices=(0, 2))
    pipeline = SimpleNamespace(
        reconstruction=isolated,
        half_reconstructions=(first, second),
        stability=SimpleNamespace(first_train_views=(0,), second_train_views=(1,)),
    )
    assert module._pipeline_fit_access_metrics(task, pipeline, fits) == {
        "heldout_fit_access_count": 0,
        "heldout_fit_checked_fit_count": 3,
    }
    pipeline.half_reconstructions = (first, leaked)
    with pytest.raises(RuntimeError, match="held-out view entered fitting"):
        module._pipeline_fit_access_metrics(task, pipeline, fits)


def test_finite_penalty_transport_uses_supported_fail_closed_gates() -> None:
    task = json.loads(TASK.read_text(encoding="utf-8"))
    gates = task["frozen_configuration"]["invariant_gates"]

    assert "dustbin_capacity_balance_tolerance" not in gates
    assert "transport_fixed_point_residual_tolerance" in gates
    source = DRIVER.read_text(encoding="utf-8")
    assert "hard association invariant violation in view" in source
    assert "augmented.sum(dim=1).sum() - augmented.sum(dim=0).sum()" not in source


def test_calibrated_candidate_gate_violation_is_measured_and_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    task = json.loads(TASK.read_text(encoding="utf-8"))
    plan = CorrespondencePlan(
        real_mass=torch.tensor([[0.25]], dtype=torch.float64),
        track_dustbin_mass=torch.zeros(1, dtype=torch.float64),
        observation_dustbin_mass=torch.zeros(1, dtype=torch.float64),
        dustbin_dustbin_mass=torch.tensor(0.0, dtype=torch.float64),
        track_capacities=torch.ones(1, dtype=torch.float64),
        observation_capacities=torch.ones(1, dtype=torch.float64),
        method="unbalanced_sinkhorn",
        iterations=1,
        fixed_point_residual=0.0,
        candidate_mask=torch.tensor([[False]]),
    )
    result = SimpleNamespace(association=SimpleNamespace(plans=(plan,)))
    monkeypatch.setattr(
        module,
        "_source_projection_invariants",
        lambda _result: {
            "source_mean_max_error": 0.0,
            "source_covariance_max_error": 0.0,
            "source_covariance_relative_error": 0.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_split_conservation_invariants",
        lambda: {
            "split_density_mass_error": 0.0,
            "split_optical_thickness_error": 0.0,
        },
    )

    with pytest.raises(RuntimeError, match="candidate gate"):
        module._enforce_result_invariants(task, result, association_required=True)

    good_plan = replace(plan, candidate_mask=torch.tensor([[True]]))
    good_result = SimpleNamespace(association=SimpleNamespace(plans=(good_plan,)))
    pipeline = SimpleNamespace(
        reconstruction=good_result,
        half_reconstructions=(good_result, result),
    )
    with pytest.raises(RuntimeError, match="candidate gate"):
        module._enforce_pipeline_result_invariants(
            task,
            pipeline,
            association_required=True,
        )
    pipeline.half_reconstructions = (good_result, good_result)
    aggregate = module._enforce_pipeline_result_invariants(
        task,
        pipeline,
        association_required=True,
    )
    assert aggregate["hard_invariant_checked_fit_count"] == 3
    assert aggregate["candidate_gate_violation_mass_max"] == 0.0


def test_draft_protocol_refuses_outcome_execution_before_run_lock() -> None:
    module = _module(RETRY_DRIVER)
    task = json.loads(RETRY_TASK.read_text(encoding="utf-8"))
    task["status"] = "draft"
    task["protocol_review"] = {
        "reviewer": None,
        "verdict": "pending",
        "protocol_sha256": None,
        "artifact": None,
    }

    with pytest.raises(ValueError, match="task is draft"):
        module._validate_run_binding(RETRY_TASK, ROOT / "runs" / module.TASK_ID, task)
