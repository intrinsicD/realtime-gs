"""Contract tests for the preregistered GPS field-proxy experiment driver."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import scripts.experiment_contract as contract

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (
    ROOT / "scripts/experiments/20260804_gps_field_proxy_depth_stage_frame00008_gaussianimage.py"
)
TASK = ROOT / "experiments/tasks/20260804_gps_field_proxy_depth_stage_frame00008_gaussianimage.json"


def _driver():
    spec = importlib.util.spec_from_file_location("rtgs_gps_field_experiment_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task() -> dict:
    return json.loads(TASK.read_text(encoding="utf-8"))


def _cells(module, task: dict) -> list[dict]:
    stages = [item["id"] for item in task["stages"]]
    cells = []
    for seed in task["seeds"]:
        for arm in module.ARMS:
            start = 0.0
            trace = {}
            for stage in stages:
                duration = 2.5 if stage == "matched_compact_refinement" else 0.25
                trace[stage] = {"start": start, "end": start + duration}
                start += duration
            initial = 0.9 if arm == "gps_field_proxy" else 1.0
            if arm == "gps_shuffled_field":
                initial = 1.0
            final = 0.81 if arm == "gps_field_proxy" else 0.8
            cells.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "directory": Path("unused") / str(seed) / arm,
                    "receipt": {
                        "stage_trace": trace,
                        "initial_evaluation": {"equal_view_j_area": initial},
                        "final_evaluation": {"equal_view_j_area": final},
                        "initializer": {
                            "diagnostics": {
                                "valid_depth_candidate_fraction": 0.75,
                                "selected_left_right_consistency_median_px": 0.2,
                            }
                        },
                        "final_gaussians": 3000,
                    },
                    "resource": {
                        "peak_cuda_allocated_bytes": 100,
                        "peak_cuda_reserved_bytes": 120,
                        "nvml_process_peak_bytes": 150,
                        "peak_process_rss_bytes": 200,
                        "wall_seconds": start,
                        "stage_seconds": {
                            name: values["end"] - values["start"] for name, values in trace.items()
                        },
                    },
                    "history": {
                        "steps": [
                            {
                                "step": step,
                                "elapsed_seconds": 0.01,
                                "total_sampled_loss": 1.0 / step,
                            }
                            for step in range(1, 251)
                        ]
                    },
                }
            )
    return cells


def test_execution_plan_matches_frozen_warmups_and_rotated_measured_orders():
    module = _driver()
    task = _task()
    warmups, measured = module._execution_jobs(task)

    assert warmups == [(module.WARMUP_SEED, arm, True) for arm in module.ARMS]
    assert len(measured) == len(module.ARMS) * len(task["seeds"])
    assert [(seed, arm) for seed, arm, _warmup in measured] == [
        (int(seed), arm)
        for seed, arms in task["frozen_configuration"]["execution_order"]["measured_orders"].items()
        for arm in arms
    ]


def test_driver_materializes_exact_frozen_gps_and_compact_training_configs():
    module = _driver()
    task = _task()
    correct = module._gps_initializer_config(task, shuffled=False)
    shuffled = module._gps_initializer_config(task, shuffled=True)
    training = module._train_config(task, seed=804001, extent=1.25, device="cuda:0")

    assert correct.n_init_3d == 3000
    assert correct.left_view == "C0001"
    assert correct.right_view == correct.proxy_right_view == "C0022"
    assert shuffled.right_view == "C0022"
    assert shuffled.proxy_right_view == "C0005"
    assert correct.proxy.resolution == 1024
    assert training.iterations == 250
    assert training.checkpoints == (0, 50, 100, 250)
    assert training.seed == 804001
    assert training.extent == 1.25
    assert training.device == "cuda:0"


def test_decision_applies_all_three_paired_seed_gates_and_completion_requirement():
    module = _driver()
    task = _task()
    cells = _cells(module, task)
    decision = module._decision(task, cells)

    assert decision["completion_requirement_met"] is True
    assert decision["initial_seed_wins"] == 3
    assert decision["final_seed_wins"] == 3
    assert decision["shuffled_seed_wins"] == 3
    assert decision["geometry_pass_opens_field_native_successor"] is True

    incomplete = [
        cell
        for cell in cells
        if not (cell["seed"] == task["seeds"][0] and cell["arm"] == "beam_fusion")
    ]
    failed = module._decision(task, incomplete)
    assert failed["completion_requirement_met"] is False
    assert failed["geometry_pass_opens_field_native_successor"] is False
    assert failed["verdict"] == "inconclusive_incomplete_cells"


def test_v2_history_and_metrics_projection_satisfy_common_experiment_contract():
    module = _driver()
    task = _task()
    cells = _cells(module, task)
    decision = module._decision(task, cells)

    history = module._history_bundle(task, cells)
    metrics = module._metrics_bundle(task, cells, decision)
    lock = {"command": task["run_command"], "task_id": task["task_id"]}

    assert contract._history_errors(history, task, completed=True) == []
    assert contract._metric_errors_v2(metrics, task, lock, completed=True) == []
    failed = module._failed_metrics_bundle(task, "warmup failed")
    assert contract._metric_errors_v2(failed, task, lock, completed=False) == []


def test_live_guard_denies_all_negative_controls_without_importing_image_stack():
    script = f"""
import importlib.util
import json
from pathlib import Path
path = Path({str(DRIVER)!r})
spec = importlib.util.spec_from_file_location('guard_driver', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
guard = module.NoImageGuard()
with guard:
    import torch
print(json.dumps(guard.record(), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "schema": "rtgs.no_image_guard.v1",
        "passed": True,
        "negative_control_expected": 4,
        "negative_control_denials": 4,
        "unexpected_denied_paths": 0,
        "unexpected_denied_imports": 0,
        "forbidden_modules_loaded": [],
        "source_rgb_or_mask_opened": False,
    }
