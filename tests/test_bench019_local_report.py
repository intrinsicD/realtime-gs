"""Synthetic end-to-end publication, strict export, and report-schema tests."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from rtgs import bench019 as B
from rtgs import bench019_local_report as report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return B.describe_artifact(path)


def _bytes(path, value=b"synthetic fixture bytes"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return B.describe_artifact(path)


def _relative(path, root):
    return {**B.describe_artifact(path), "path": path.relative_to(root).as_posix()}


def _contract(filename="experiment_contract.py"):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location("report_contract_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task():
    return {
        "task_id": "20260907_bench019_synthetic_contract",
        "title": "Synthetic publication contract",
        "evidence_phase": "development",
        "input_policy": {
            "reconstruction_allowed": ["synthetic"],
            "reconstruction_forbidden": ["heldout"],
            "evaluation_allowed": ["synthetic"],
        },
        "protocol_review": {
            "artifact": "synthetic_review.md",
            "reviewer": "fixture reviewer",
            "protocol_sha256": "a" * 64,
        },
        "claim_boundary": "Synthetic schema fixture only; no empirical research claim.",
        "run_command": ["python", "synthetic_fixture.py"],
        "datasets": [
            {"id": "frame_00008", "role": "development"},
            {"id": "frame_00009", "role": "replication"},
        ],
        "comparators": [
            {"id": family}
            for family in ("native_additive", "structsplat_normalized", "structsplat_contained")
        ],
        "splits": {
            frame: {
                "train": [f"C{i:04d}" for i in range(8)],
                "heldout": ["C0100", "C0101", "C0102"],
            }
            for frame in ("frame_00008", "frame_00009")
        },
        "seeds": [19001, 19002, 19003],
        "stages": [
            {
                "id": "refine",
                "label": "Fixed-topology RGB refinement",
                "purpose": "Synthetic source-format test",
            }
        ],
        "required_charts": ["quality", "resources", "stage_runtime"],
        "primary_metrics": [
            {"id": name, "label": name, "unit": unit, "direction": direction}
            for name, unit, direction in (
                ("heldout_foreground_psnr", "dB", "higher"),
                ("heldout_alpha_iou", "fraction", "higher"),
                ("heldout_exterior_leakage", "fraction", "lower"),
                ("field_bytes", "bytes", "descriptive"),
                ("wall_seconds", "seconds", "descriptive"),
                ("peak_cuda_allocated_bytes", "bytes", "descriptive"),
                ("final_gaussians", "gaussians", "descriptive"),
            )
        ],
        "decision_policy": {"family_materiality_floor_db": 0.25, "alpha_iou_maximum_loss": 0.02},
        "resource_protocol": {"scope": "Actual recorded scope; shared acquisition once."},
        "resource_monitor": {"warmup_seed": 19999},
        "aa_replay": {
            "frame_id": "frame_00008",
            "seed": 19001,
            "family_id": "native_additive",
            "additional_family_id": "structsplat_contained",
        },
    }


def _fixture(tmp_path):
    task = _task()
    run = tmp_path / "runs" / task["task_id"]
    run.mkdir(parents=True)
    lock = {
        "task_id": task["task_id"],
        "command": task["run_command"],
        "started_at_utc": "2026-09-07T00:00:00+00:00",
        "report_template_version": 2,
        "source_commit": "a" * 40,
        "task_path": "task.json",
        "data_seal_path": "synthetic_seal.json",
    }
    _write(run / "task.lock.json", lock)
    task_artifact = _write(tmp_path / "task.json", task)
    environment = _write(
        run / "environment.json",
        {
            "schema_version": 1,
            "python": "synthetic",
            "platform": "synthetic",
            "packages": {"torch": "test"},
            "device": {"type": "cpu", "name": "test", "cuda": None},
        },
    )
    dataset_artifact = _write(run / "generated_inputs.json", {"synthetic": True})
    schedule = _write(run / "gaussians.config.json", task)
    _write(run / "aa_replay.json", {"status": "passed", "checks": []})
    resource = {
        "nvml_process_peak_bytes": None,
        "nvml_errors": ["synthetic unavailable"],
        "contended": False,
        "samples": [],
        "wall_seconds": 25.0,
    }
    frames = []
    for frame_index, dataset in enumerate(task["datasets"]):
        frame = dataset["id"]
        families = []
        for family_index, item in enumerate(task["comparators"]):
            family = item["id"]
            directory = run / "inputs" / frame / family
            manifest = _write(directory / "manifest.json", {"synthetic": frame + family})
            views = []
            for view_index, name in enumerate(task["splits"][frame]["train"]):
                source_history = _write(
                    directory / f"{name}.history.json",
                    {
                        "view_id": name,
                        "seed": 19000 + frame_index * 100 + view_index,
                        "history": [
                            {"step": 0, "elapsed_seconds": 0.001, "pixel_l2": 1.0},
                            {"step": 1000, "elapsed_seconds": 2.0, "pixel_l2": 0.1},
                        ],
                    },
                )
                views.append({"view_id": name, "history": source_history, "config": {"count": 512}})
            stage1 = _write(
                directory / "stage1_metrics.json",
                {
                    "metrics": {
                        "foreground_psnr": 25.0 + family_index,
                        "boundary_mae": 0.1,
                        "support_iou": 0.8,
                        "field_bytes": 12345,
                        "rows": 4096,
                    }
                },
            )
            _write(
                directory / "production.json",
                {
                    "status": "complete",
                    "frame_id": frame,
                    "family_id": family,
                    "train_views": task["splits"][frame]["train"],
                    "views": views,
                    "wall_seconds": 30.0,
                },
            )
            _write(directory / "resource_receipt.json", resource)
            families.append(
                {
                    "id": family,
                    "field_manifest": manifest,
                    "stage1_metrics": stage1,
                    "semantics": {
                        "provider": "native" if family_index == 0 else "structsplat",
                        "equation": "additive_sum"
                        if family_index == 0
                        else "normalized_weighted_sum",
                        "blend_mode": "additive" if family_index == 0 else "normalized",
                        "alpha_policy": "packed_alpha",
                        "coordinate_convention": "pixel centers",
                        "semantic_digest": hashlib.sha256(family.encode()).hexdigest(),
                    },
                }
            )
            warmup = report._cell_dir(run, frame, family, 19999, "warmup")
            _write(warmup / "receipt.json", {"status": "ok"})
            _write(warmup / "resource_receipt.json", resource)
            for seed in task["seeds"]:
                replicates = ["primary"]
                if frame_index == 0 and family_index in (0, 2) and seed == 19001:
                    replicates.append("aa")
                for replicate in replicates:
                    cell = report._cell_dir(run, frame, family, seed, replicate)
                    initial_ply = cell / "checkpoint_0000.ply"
                    final_ply, final_npz = cell / "gaussians.ply", cell / "gaussians.npz"
                    for path in (initial_ply, final_ply, final_npz):
                        _bytes(path)
                    config = {"seed": seed, "synthetic": True}
                    _write(cell / "config.json", config)
                    _write(cell / "initializer.json", {"supported": 256})
                    native = {
                        "executed_iterations": 1000,
                        "stop_reason": "max_iterations",
                        "loss": [1 / (step + 1) for step in range(1000)],
                        "elapsed": [[step, step / 50] for step in range(50, 1001, 50)],
                        "n_gaussians": [[step, 256] for step in range(50, 1001, 50)],
                    }
                    _write(
                        cell / "history.json",
                        {
                            "heldout_evaluation": False,
                            "native_trainer": native,
                            "sampled_training_view_ids": ["C0000"] * 1000,
                        },
                    )
                    cell_metrics = {
                        "field_bytes": 12345,
                        "wall_seconds": 25.0,
                        "final_gaussians": 256,
                        "peak_cuda_allocated_bytes": 1000000,
                        "lift_seconds": 2.0,
                        "refine_seconds": 21.0,
                        "peak_cuda_reserved_bytes": 2000000,
                    }
                    _write(
                        cell / "receipt.json",
                        {
                            "task_id": task["task_id"],
                            "status": "ok",
                            "frame_id": frame,
                            "family_id": family,
                            "seed": seed,
                            "metrics": cell_metrics,
                            "fields": {"manifest": manifest, "camera_geometry_sha256": "a" * 64},
                            "bounds": {
                                "bounds_source": "camera_axis_fallback",
                                "bounds_extent": 1.0,
                            },
                            "unsupported_midpoint_count": family_index * 20,
                            "artifacts": {
                                "history": _relative(cell / "history.json", cell),
                                "config": _relative(cell / "config.json", cell),
                                "initial": {"ply": _relative(initial_ply, cell)},
                                "final": {
                                    "ply": _relative(final_ply, cell),
                                    "npz": _relative(final_npz, cell),
                                },
                            },
                        },
                    )
                    evaluation = cell / "evaluation"
                    psnr = 25 + family_index * 0.3 + (seed - 19001) * 0.01
                    per_view = []
                    for name in task["splits"][frame]["heldout"]:
                        previews = {}
                        for kind in ("target", "reconstruction", "error", "alpha"):
                            path = evaluation / f"{name}_{kind}.png"
                            _bytes(path)
                            previews[kind] = _relative(path, evaluation)
                        per_view.append(
                            {
                                "view_id": name,
                                "foreground_psnr": psnr,
                                "alpha_iou": 0.8,
                                "exterior_leakage": 0.1,
                                "previews": previews,
                            }
                        )
                    _write(
                        evaluation / "evaluation.json",
                        {
                            "task_id": task["task_id"],
                            "status": "ok",
                            "frame_id": frame,
                            "family_id": family,
                            "seed": seed,
                            "view_ids": task["splits"][frame]["heldout"],
                            "model_sha256": B.sha256_file(final_npz),
                            "per_view": per_view,
                            "metrics": {
                                "heldout_foreground_psnr": psnr,
                                "heldout_alpha_iou": 0.8,
                                "heldout_exterior_leakage": 0.1,
                                "evaluation_seconds": 0.3,
                            },
                        },
                    )
                    _write(cell / "resource_receipt.json", resource)
        frames.append(
            {
                "id": frame,
                "pixels": dataset_artifact,
                "masks": dataset_artifact,
                "cameras": dataset_artifact,
                "split": task["splits"][frame],
                "families": families,
            }
        )
    protocol = {
        "schema": B.PROTOCOL_SCHEMA,
        "task_id": "BENCH-019",
        "state": "frozen",
        "driver": "synthetic_driver",
        "claim_scope": "workload_specific",
        "repositories": [
            {
                "name": name,
                "root": str(tmp_path),
                "commit": character * 40,
                "branch": "synthetic",
                "dirty": False,
                "status_sha256": hashlib.sha256(b"").hexdigest(),
                "environment": environment,
            }
            for name, character in (("structsplat", "a"), ("realtime-gs", "b"))
        ],
        "captures": [{"id": "synthetic_capture", "frames": frames}],
        "downstream": {
            "task_manifest": task_artifact,
            "dataset_manifest": dataset_artifact,
            "environment": environment,
            "schedule_config": schedule,
            "command": task["run_command"],
            "working_directory": str(tmp_path),
            "outcome_root": str(run / "downstream"),
            "seeds": task["seeds"],
            "initializers": ["field_sweep"],
            "result_schema": B.ROW_SCHEMA,
        },
        "predictors": [
            {"name": "foreground_psnr", "direction": "higher"},
            {"name": "boundary_mae", "direction": "lower"},
        ],
        "responses": [
            {"name": "heldout_foreground_psnr", "direction": "higher", "primary": True},
            {"name": "wall_seconds", "direction": "lower", "primary": False},
        ],
        "analysis": {
            "bootstrap_replicates": 200,
            "bootstrap_seed": 192019,
            "minimum_capture_groups": 3,
            "minimum_frames": 2,
            "minimum_family_count": 3,
            "minimum_spearman": 0.8,
            "minimum_bootstrap_lower": 0.0,
            "minimum_lofo_top1_agreement": 0.67,
            "selection_priority": ["foreground_psnr", "boundary_mae"],
            "missing_policy": "fail_closed",
        },
        "aa_replay": {
            "frame_id": "frame_00008",
            "family_id": "native_additive",
            "seed": 19001,
            "initializer": "field_sweep",
            "primary_replicate": "primary",
            "replay_replicate": "aa",
            "metric_abs_tolerance": {"foreground_psnr": 0.0, "heldout_foreground_psnr": 0.05},
        },
    }
    design = copy.deepcopy(protocol)
    design.pop("state")
    protocol["design_sha256"] = hashlib.sha256(B.canonical_json(design)).hexdigest()
    protocol["review"] = {
        "driver": "synthetic_driver",
        "reviewer": "synthetic_reviewer",
        "verdict": "approved",
        "design_sha256": protocol["design_sha256"],
        "artifact": dataset_artifact,
    }
    protocol["protocol_sha256"] = hashlib.sha256(B.canonical_json(protocol)).hexdigest()
    protocol_path = run / "protocol" / "structsplat.frozen.json"
    _write(protocol_path, protocol)
    return task, run, protocol_path


def test_publication_passes_shared_schemas_and_formal_nineteen_cell_export(tmp_path):
    task, run, protocol_path = _fixture(tmp_path)
    result = report.publish_result_sources(task, run, protocol_path)
    checker = _contract()
    metrics, history = (
        report._read(run / "metrics.json"),
        report._read(run / "training_history.json"),
    )
    lock = report._read(run / "task.lock.json")
    assert checker._metric_errors_v2(metrics, task, lock, completed=True) == []
    assert checker._history_errors(history, task, completed=True) == []
    assert checker._run_receipt_errors(report._read(run / "run_receipt.json"), task, lock) == []
    assert result["bench019"]["cell_count"] == 19
    rows = [
        json.loads(line)
        for line in Path(result["bench019"]["rows"]["path"]).read_text().splitlines()
    ]
    assert len([row for row in rows if row["replicate_id"] == "primary"]) == 18
    assert [
        (row["family_id"], row["replicate_id"]) for row in rows if row["replicate_id"] == "aa"
    ] == [("native_additive", "aa")]
    assert result["surrogate_verdict"] == "not_evaluated_insufficient_scope"
    assert result["audit_status"] == "pending_independent_audit"
    assert len(result["cells"]) == 20
    assert not list((tmp_path / "benchmarks" / "results").glob("*AUDIT*"))
    assert len(list((tmp_path / "benchmarks" / "results").glob("*RESULT*"))) == 2
    resource = report._read(run / "resource_receipt.json")
    assert len(resource["stage1_acquisitions"]) == 6
    assert len(resource["primary_cells"]) == 18
    assert not resource["stage1_timing_replication"]
    assert resource["stage1_acquisitions"][0]["resource"]["nvml_process_peak_bytes"] is None
    assert all(
        record["stage"] == "refine" and record["split"] == "train" for record in history["records"]
    )
    assert len(metrics["dataset_summaries"]["frame_00008"]["curves"][1]["series"]) == 24
    with pytest.raises(FileExistsError):
        report.publish_result_sources(task, run, protocol_path)


@pytest.mark.parametrize(
    "failure", ["missing_cell", "wrong_model", "aggregate", "heldout_schedule"]
)
def test_publication_fails_before_aggregate_on_invalid_cell_source(tmp_path, failure):
    task, run, protocol_path = _fixture(tmp_path)
    directory = report._cell_dir(run, "frame_00009", "structsplat_contained", 19003)
    if failure == "missing_cell":
        (directory / "receipt.json").unlink()
    elif failure == "heldout_schedule":
        history_path = directory / "history.json"
        history = report._read(history_path)
        history["sampled_training_view_ids"][0] = "C0100"
        _write(history_path, history)
        receipt = report._read(directory / "receipt.json")
        receipt["artifacts"]["history"] = _relative(history_path, directory)
        _write(directory / "receipt.json", receipt)
    else:
        path = directory / "evaluation" / "evaluation.json"
        evaluation = report._read(path)
        if failure == "wrong_model":
            evaluation["model_sha256"] = "0" * 64
        else:
            evaluation["metrics"]["heldout_foreground_psnr"] += 1.0
        _write(path, evaluation)
    with pytest.raises((ValueError, B.ExportError)):
        report.publish_result_sources(task, run, protocol_path)
    assert not (run / "metrics.json").exists()
    assert not (run / "downstream" / "bench019").exists()


def test_history_uses_actual_times_and_rejects_repeated_acquisition_stages():
    task = _task()
    native = {
        "executed_iterations": 1000,
        "stop_reason": "max_iterations",
        "elapsed": [[50, 0.8], [1000, 17.3]],
        "n_gaussians": [[50, 256], [1000, 256]],
        "loss": [float(index) for index in range(1000)],
    }
    cell = {
        "frame_id": "frame_00008",
        "family_id": "native_additive",
        "seed": 19001,
        "history": {"native_trainer": native},
    }
    history = report.training_history(task, [cell])
    losses = [row for row in history["records"] if row["metric_id"] == "refine_loss"]
    assert [(r["step"], r["wall_seconds"], r["value"]) for r in losses] == [
        (50, 0.8, 49.0),
        (1000, 17.3, 999.0),
    ]
    task["stages"].insert(0, {"id": "stage1", "label": "Acquisition"})
    with pytest.raises(ValueError, match="acquisition histories are separate"):
        report.training_history(task, [cell])


def test_materiality_requires_positive_pairs_and_alpha_guard_in_both_frames():
    task = _task()
    cells = []
    for dataset in task["datasets"]:
        for family in task["comparators"]:
            for seed in task["seeds"]:
                delta = 0 if family["id"] == "native_additive" else 0.3
                cells.append(
                    {
                        "frame_id": dataset["id"],
                        "family_id": family["id"],
                        "seed": seed,
                        "metrics": {
                            "heldout_foreground_psnr": 25 + delta,
                            "heldout_alpha_iou": 0.8,
                        },
                    }
                )
    assert all(row["meets_rule_in_both_frames"] for row in report.paired_comparisons(task, cells))
    cells[-1]["metrics"]["heldout_alpha_iou"] = 0.7
    results = report.paired_comparisons(task, cells)
    assert results[0]["meets_rule_in_both_frames"]
    assert not results[1]["meets_rule_in_both_frames"]


def test_shared_renderer_consumes_synthetic_sources_without_real_audit_artifacts(
    tmp_path, monkeypatch
):
    task, run, protocol_path = _fixture(tmp_path)
    report.publish_result_sources(task, run, protocol_path)
    # These explicit synthetic fixtures exist only below pytest's temporary root.
    # The publisher itself never writes production AUDIT evidence.
    evidence = tmp_path / "benchmarks" / "results"
    _write(evidence / f"{task['task_id']}_AUDIT.json", {"synthetic_fixture": True})
    _bytes(evidence / f"{task['task_id']}_AUDIT.md", b"Synthetic renderer fixture, not an audit.\n")
    _bytes(tmp_path / "synthetic_review.md", b"Synthetic renderer fixture.\n")
    _write(tmp_path / "synthetic_seal.json", {"synthetic_fixture": True})
    checker = _contract()
    lock = report._read(run / "task.lock.json")
    # Isolate existing source-authority validation; retain real v2 schema, artifact,
    # chart/history rendering, page inventory, link and checksum validators.
    monkeypatch.setattr(checker, "_locked_task", lambda *_args, **_kwargs: (task, lock, []))
    index = checker.render_run(run, root=tmp_path)
    assert index.is_file()
    assert checker.validate_run(run, root=tmp_path) == []
    assert (run / "datasets" / "frame_00008" / "index.html").is_file()
    assert (run / "datasets" / "frame_00009" / "index.html").is_file()
    assert (
        "Stage1 per-view acquisition"
        in (run / "datasets" / "frame_00008" / "index.html").read_text()
    )
    bundle_checker = _contract("check_results_bundle.py")
    assert bundle_checker._check_v2_manifest(run) == []
    nested = "inputs/frame_00008/native_additive/manifest.json"
    manifest_path = run / "manifest.json"
    complete_manifest = report._read(manifest_path)
    omitted = copy.deepcopy(complete_manifest)
    omitted["entries"] = [entry for entry in omitted["entries"] if entry["path"] != nested]
    _write(manifest_path, omitted)
    metrics = report._read(run / "metrics.json")
    assert any(nested in error for error in checker._manifest_errors(run, tmp_path, task, metrics))
    assert any(nested in error for error in bundle_checker._check_v2_manifest(run))
    _write(manifest_path, complete_manifest)
    (run / nested).write_bytes((run / nested).read_bytes() + b"tampered")
    assert any(nested in error for error in checker._manifest_errors(run, tmp_path, task, metrics))
    assert any(nested in error for error in bundle_checker._check_v2_manifest(run))


def test_failure_sources_preserve_failed_run_receipt_and_do_not_invent_metrics(tmp_path):
    task, run, _protocol = _fixture(tmp_path)
    frozen_receipt = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "status": "failed",
        "started_at_utc": "2026-09-07T00:00:00+00:00",
        "finished_at_utc": "2026-09-07T00:00:01+00:00",
        "exit_code": 1,
        "failure_phase": "phase1",
        "message": "synthetic failure",
    }
    descriptor = _write(run / "run_receipt.json", frozen_receipt)
    result = report.publish_failure_sources(None, run, RuntimeError("synthetic failure"))
    assert result["task_context_available"] and result["v2_sources_written"]
    assert B.describe_artifact(run / "run_receipt.json") == descriptor
    metrics = report._read(run / "metrics.json")
    assert metrics["metrics"] == {} and metrics["charts"] == [] and metrics["evidence"] == []
    assert report._read(run / "resource_receipt.json")["aggregate_metrics"] is None
    checker = _contract()
    assert (
        checker._metric_errors_v2(
            metrics, task, report._read(run / "task.lock.json"), completed=False
        )
        == []
    )
    assert (
        checker._history_errors(report._read(run / "training_history.json"), task, completed=False)
        == []
    )


def test_failure_without_task_context_retains_only_truthful_minimal_record(tmp_path):
    run = tmp_path / "runs" / "synthetic_failure"
    run.mkdir(parents=True)
    result = report.publish_failure_sources(None, run, ValueError("missing task"))
    assert result["task_id"] == run.name
    assert not result["task_context_available"] and not result["v2_sources_written"]
    assert result["metrics_claim"].startswith("none")
    assert not (run / "metrics.json").exists()


def test_early_failure_environment_is_explicitly_unprobed(tmp_path):
    task = _task()
    run = tmp_path / "runs" / task["task_id"]
    run.mkdir(parents=True)
    report.publish_failure_sources(task, run, RuntimeError("preflight failed"))
    environment = report._read(run / "environment.json")
    assert environment["device"]["type"] == "unprobed"
    assert environment["device"]["cuda"] is None
    assert _contract()._environment_errors(environment) == []
