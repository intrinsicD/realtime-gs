from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import scripts.experiment_contract as contract

ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "scripts/experiments/20260801_paper_three_provider_fullres_stage_frame00008.py"
TASK_PATH = ROOT / "experiments/tasks/20260801_paper_three_provider_fullres_stage_frame00008.json"


def _driver():
    spec = importlib.util.spec_from_file_location("rtgs_three_provider_driver", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task() -> dict:
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


def test_task_provider_factor_preserves_native_compositor_semantics():
    task = _task()
    providers = task["frozen_configuration"]["provider_factor"]

    assert providers["gaussianimage"]["rtgsv_provider"] == "native"
    assert providers["gaussianimage"]["blend_mode"] == "additive"
    assert providers["structsplat_mask_contained"]["rtgsv_provider"] == "structsplat"
    assert providers["structsplat_mask_contained"]["blend_mode"] == "normalized"
    assert providers["structsplat_mask_contained"]["boundary_specialization"] == ("mask_contained")
    assert providers["structsplat_no_boundary"]["blend_mode"] == "normalized"
    assert providers["structsplat_no_boundary"]["boundary_specialization"] == "none"
    assert {dataset["id"]: task["splits"][dataset["id"]] for dataset in task["datasets"]} == {
        dataset["id"]: task["splits"][task["datasets"][0]["id"]] for dataset in task["datasets"]
    }


def test_driver_constants_match_frozen_compact_controls():
    module = _driver()
    task = _task()
    frozen = task["frozen_configuration"]
    compact = frozen["compact_3dgs"]

    assert {key: compact[key] for key in module.TRAIN_CONFIG} == {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in module.TRAIN_CONFIG.items()
    }
    assert frozen["classic_density"] == module.DENSITY_CONFIG
    assert frozen["compact_query_index"] == module.QUERY_INDEX_CONFIG
    assert frozen["point_renderer"] == module.POINT_RENDER_CONFIG
    assert compact["convergence_evaluation"] == module.CONVERGENCE_CONFIG
    assert frozen["resource_measurement"] == module.RESOURCE_CONFIG
    assert frozen["presentation"] == module.PRESENTATION_CONFIG
    initializer = module._initializer_settings(task)
    assert initializer["max_starting_gaussians"] == 3293
    assert task["frozen_configuration"]["initializer"]["common_starting_count"] == 3293


def test_v2_history_projection_has_complete_ordered_stage_boundaries(tmp_path):
    module = _driver()
    task = _task()
    stages = {stage["id"]: 0.25 for stage in task["stages"]}
    cell = {
        "dataset": task["datasets"][0],
        "seed": task["seeds"][0],
        "arm": "bounded_random",
        "receipt": {"stage_wall_seconds": stages},
        "history": {
            "steps": [
                {"step": 1, "elapsed_seconds": 0.01, "total_sampled_loss": 0.5},
                {"step": 10000, "elapsed_seconds": 0.01, "total_sampled_loss": 0.2},
            ]
        },
        "summary": {
            "sampled_heldout_evaluation": {"equal_view_uniform_fit_window_mse": 0.1},
            "sampled_heldout_convergence": [
                {
                    "step": 10000,
                    "evaluation": {"equal_view_uniform_fit_window_mse": 0.1},
                }
            ],
        },
    }

    history = module._history_bundle(task, tmp_path, [cell])

    assert contract._history_errors(history, task, completed=True) == []


def test_canonical_parser_does_not_expose_dense_or_image_backed_commands(monkeypatch):
    module = _driver()
    monkeypatch.setattr(
        "sys.argv",
        [str(DRIVER_PATH), "train-standard"],
    )

    try:
        module._parse_args()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("forbidden train-standard command unexpectedly parsed")


def test_execution_plan_has_one_global_warmup_and_three_measured_repeats_per_cell():
    module = _driver()
    task = _task()

    jobs = module._execution_jobs(task)
    warmups = [job for job in jobs if job[3]]
    measured = [job for job in jobs if not job[3]]
    expected_pairs = {(dataset["id"], arm) for dataset in task["datasets"] for arm in module.ARMS}

    assert warmups == [(task["datasets"][0]["id"], min(task["seeds"]) - 1, module.ARMS[0], True)]
    assert len(measured) == len(expected_pairs) * len(task["seeds"])
    assert {(dataset_id, seed, arm) for dataset_id, seed, arm, _warmup in measured} == {
        (dataset["id"], seed, arm)
        for dataset in task["datasets"]
        for seed in task["seeds"]
        for arm in module.ARMS
    }


def test_convergence_summary_uses_frozen_tail_and_stable_band_rules():
    module = _driver()
    records = [
        {
            "step": step,
            "evaluation": {"equal_view_uniform_fit_window_mse": risk},
        }
        for step, risk in ((0, 1.0), (5000, 0.5), (8000, 0.4), (10000, 0.398))
    ]

    summary = module._convergence_summary(
        records,
        stable_best_multiplier=1.05,
        maximum_absolute_tail_relative_change=0.01,
    )

    assert summary["best_step"] == 10000
    assert summary["iterations_to_stable_best_band"] == 8000
    assert summary["tail_relative_risk_change"] == pytest.approx(-0.005)
    assert summary["converged_by_frozen_rule"] is True


def test_resource_receipt_requires_every_frozen_measurement():
    module = _driver()
    output_files = [{"path": "gaussians.npz", "bytes": 11, "sha256": "0" * 64}]
    receipt = {
        "wall_seconds": 1.0,
        "peak_cuda_allocated_bytes": 2,
        "peak_cuda_reserved_bytes": 3,
        "nvml_process_peak_bytes": 4,
        "background_device_memory_bytes": 5,
        "background_device_memory_peak_bytes": 6,
        "device_total_bytes": 7,
        "driver_version": "driver",
        "torch": "torch",
        "torch_cuda": "cuda",
        "ru_maxrss_bytes": 8,
        "compact_input_bytes": 9,
        "compact_field_bytes": 10,
        "final_model_npz_bytes": 11,
        "final_model_ply_bytes": 12,
        "compact_to_model_compression_ratio": 10 / 11,
        "output_bytes": 11,
        "output_files": output_files,
        "idle_guard": {"passed": True},
        "foreign_compute_processes": [],
    }

    module._validate_resource_record(receipt)
    receipt.pop("nvml_process_peak_bytes")
    with pytest.raises(RuntimeError, match="nvml_process_peak_bytes"):
        module._validate_resource_record(receipt)


def test_completed_and_failed_v2_sources_match_contract_and_comparison_viewer():
    module = _driver()
    task = _task()
    cells = []
    for dataset in task["datasets"]:
        for seed in task["seeds"]:
            for arm in module.ARMS:
                cells.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "arm": arm,
                        "summary": {
                            "sampled_heldout_evaluation": {
                                "equal_view_uniform_fit_window_mse": 0.1
                            },
                            "final_gaussians": 3293,
                            "convergence_summary": {
                                "final_to_best_risk_ratio": 1.0,
                                "converged_by_frozen_rule": True,
                            },
                        },
                        "resource": {
                            "peak_cuda_allocated_bytes": 100,
                            "wall_seconds": 1.0,
                            "compact_to_model_compression_ratio": 2.0,
                        },
                    }
                )

    metrics = module._metrics_bundle(task, ROOT / "unused", cells)
    viewer = metrics["commands"]["viewer"]
    assert "--comparison-manifest" in viewer
    assert any(item.endswith("viewer_comparison.json") for item in viewer)
    assert [chart["id"] for chart in metrics["charts"]] == task["required_charts"]
    assert (
        contract._metric_errors_v2(
            metrics,
            task,
            {"command": task["run_command"], "task_id": task["task_id"]},
            completed=True,
        )
        == []
    )

    failed = module._failed_metrics_bundle(task)
    assert (
        contract._metric_errors_v2(
            failed,
            task,
            {"command": task["run_command"], "task_id": task["task_id"]},
            completed=False,
        )
        == []
    )


def test_protocol_narrows_provider_interpretation_and_freezes_failure_evidence():
    task = _task()

    assert (
        "no directional boundary-leakage or provider-superiority prediction" in task["hypothesis"]
    )
    assert task["frozen_configuration"]["compact_3dgs"]["sampling_comparability"][
        "provider_conditioned"
    ] == ["proposal distributions", "fit windows", "realized 2D coordinates"]
    assert "structured_worker_and_run_failure_publication" in task["execution_guards"]
    assert "complete_root_preview_publication" in task["execution_guards"]
