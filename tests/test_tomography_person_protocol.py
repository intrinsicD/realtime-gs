"""CPU contracts for the person-reference tomography comparison boundary."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import (
    COMPACT_VIEW_BYTE_CAP,
    save_compact_view,
    write_compact_dataset_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "experiments/tasks/20260906_tomography_source_constraints_haelyn_dome.json"
DRIVER = ROOT / "scripts/experiments/20260906_tomography_source_constraints_haelyn_dome.py"


def _driver():
    spec = importlib.util.spec_from_file_location("_person_protocol_test", DRIVER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_constraint_configs_have_only_the_declared_treatment_difference() -> None:
    driver = _driver()
    task = json.loads(TASK.read_text())
    for dataset in task["frozen_configuration"]["phase_order"]:
        configs = [
            driver.configs(task, arm, task["seeds"][0], dataset_id=dataset)
            for arm in ("hard", "soft", "free")
        ]
        values = [asdict(config[0]) for config in configs]
        assert values[0]["refit"]["source_constraint"] == "hard"
        assert values[1]["refit"]["source_constraint"] == "soft"
        assert values[2]["refit"]["source_anchor_weight"] == 0
        for value in values:
            value["refit"].pop("source_constraint")
            value["refit"].pop("source_anchor_weight")
        assert values[0] == values[1] == values[2]
        assert [asdict(c) for c in configs[0][1:]] == [asdict(c) for c in configs[1][1:]]
        assert [asdict(c) for c in configs[0][1:]] == [asdict(c) for c in configs[2][1:]]
        assert configs[0][0].mask_mode == ("none" if dataset == "haelyn_unmasked" else "hard")


def _fixture(tmp_path: Path) -> dict:
    directory = tmp_path / "compact"
    directory.mkdir()
    paths = []
    for i in range(5):
        name = f"C{i + 1:04d}"
        camera = Camera.look_at(
            torch.tensor([-0.4 + 0.2 * i, 0.0, -3.0]), torch.zeros(3), width=12, height=10
        )
        field = GaussianObservationField(
            width=12,
            height=10,
            means=torch.tensor([[4.0 + 0.1 * i, 4.0], [7.0 + 0.1 * i, 5.5]]),
            log_scales=torch.log(torch.tensor([[0.9, 1.1], [1.2, 0.8]])),
            rotations=torch.tensor([0.15, -0.25]),
            colors=torch.tensor([[0.2, 0.5, 0.8], [0.8, 0.3, 0.15]]),
            amplitudes=torch.tensor([0.7, 0.45]),
            blend_mode="additive",
            view_id=name,
            provider="synthetic_fixture",
        )
        p = directory / f"{name}.rtgsv"
        save_compact_view(
            p,
            field,
            camera,
            calibration_sha256="a" * 64,
            source_rgb_name=f"{name}.jpg",
            source_rgb_sha256="b" * 64,
        )
        paths.append(p)
    write_compact_dataset_manifest(
        directory,
        name="fixture",
        calibration_sha256="a" * 64,
        view_paths=paths,
        bounds_hint=(torch.zeros(3), 2.0),
    )
    task = json.loads(TASK.read_text())
    task["datasets"] = [
        {"id": "haelyn_unmasked", "compact_manifest": str(directory / "manifest.json")}
    ]
    task["splits"] = {
        "haelyn_unmasked": {"train": [f"C{i:04d}" for i in range(1, 5)], "heldout": ["C0005"]}
    }
    frozen = task["frozen_configuration"]
    frozen.pop("source_binding", None)
    frozen["converter"]["view_byte_cap"] = COMPACT_VIEW_BYTE_CAP
    frozen["beam_reference"]["beam"].update(min_views=2, max_components=2, pair_limit=3)
    frozen["beam_reference"]["repair"]["covariance_steps"] = 2
    frozen["validation_views"] = {"haelyn_unmasked": ["C0004"]}
    frozen["field_lift"].update(
        max_tracks=2,
        max_train_views=3,
        depth_samples=2,
        candidate_multiplier=1,
        min_views=1,
        target_component_cap=2,
    )
    frozen["field_refit"].update(iterations=2, appearance_start=0)
    frozen["native_refinement"].update(iterations=2, checkpoints=[0, 2], attempts_per_step=8)
    return task


def test_tiny_worker_preserves_init_and_delays_heldout_until_saved_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    task = _fixture(tmp_path)
    task_path = tmp_path / "fixture.json"
    task_path.write_text(json.dumps(task))
    runner = tmp_path / "worker.py"
    runner.write_text(
        "import importlib.util,json\nfrom pathlib import Path\n"
        f"p=Path({str(DRIVER)!r})\n"
        "spec=importlib.util.spec_from_file_location('fixture_driver',p)\n"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
        f"t=json.loads(Path({str(task_path)!r}).read_text())\n"
        f"m.worker(t,Path({str(tmp_path)!r}),'haelyn_unmasked',__import__('sys').argv[1],90601,False)\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    summaries = []
    for arm in ("hard", "soft", "free"):
        result = subprocess.run(
            [sys.executable, str(runner), arm],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        output = tmp_path / "cells/haelyn_unmasked/seed_90601" / arm
        summary = json.loads((output / "summary.json").read_text())
        assert summary["input_boundary"]["passed"]
        heldout = [
            r for r in summary["input_boundary"]["opened_compact"] if r["view_id"] == "C0005"
        ]
        assert heldout and all(r["phase"] == "reporting_after_saved_endpoint" for r in heldout)
        assert not any(r["split"] == "heldout" for r in summary["records"])
        assert summary["initial_gaussians"] == summary["final_gaussians"]
        assert (output / "gaussians.ply").is_file()
        summaries.append(summary)
    assert len({s["initial_sha256"] for s in summaries}) == 1
    contract = _driver().module(ROOT / "scripts/experiment_contract.py", "_person_history_contract")
    history = {
        "schema_version": 2,
        "records": [r for s in summaries for r in s["records"]],
        "metric_metadata": {
            "native_teacher_mse": {
                "label": "Validation pixel MSE",
                "unit": "MSE",
                "group": "Quality",
                "direction": "lower",
            }
        },
        "stage_markers": [r for s in summaries for r in s["stage_markers"]],
    }
    assert not contract._history_errors(history, task, completed=True)
    for summary in summaries:
        assert (
            summary["selected_input_binding"]["entry"] == summary["selected_input_binding"]["exit"]
        )
    result = subprocess.run(
        [sys.executable, str(runner), "beam_reference"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # The report-only process can evaluate saved endpoints using image references.
    # These generated fixture values are test artifacts, never research evidence.
    from types import SimpleNamespace

    from rtgs.data import calibrated
    from rtgs.data.compact_views import CompactDataset

    task["seeds"] = [90601]
    production = tmp_path / "production.json"
    production.write_text(json.dumps({"downscale": 1}))
    task["datasets"][0].update(
        production_manifest=str(production),
        frame_path=str(tmp_path),
        calibration=str(tmp_path / "calibration.json"),
    )
    held = CompactDataset.load(tmp_path / "compact", view_ids=["C0005"], load_alpha=False)
    scene = SimpleNamespace(
        cameras=[v.camera for v in held.views],
        images=[torch.full((10, 12, 3), 0.25)],
        masks=[torch.ones(10, 12)],
    )
    monkeypatch.setattr(calibrated, "load_calibrated_scene", lambda *a, **kw: scene)
    d = _driver()
    monkeypatch.setattr(d, "verify_external_seal", lambda task: {})
    report = d.module(DRIVER.with_name(DRIVER.stem + "_report.py"), "_person_report_test")
    monkeypatch.setattr(report, "driver", lambda: d)
    report.external_phase(task, tmp_path, "haelyn_unmasked")
    result = json.loads((tmp_path / "phases/haelyn_unmasked/external_evaluation.json").read_text())
    assert len(result["rows"]) == 12
    assert {r["endpoint"] for r in result["rows"]} == {"initial", "proxy", "final"}
    assert len(list(tmp_path.glob("cells/**/preview_*.png"))) == 12


def test_relative_run_aggregation_and_success_receipt_are_complete(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise the original relative-argv failure with small fabricated producer sources."""
    import copy
    import shutil

    from rtgs.core.gaussians3d import Gaussians3D

    d = _driver()
    report = d.module(DRIVER.with_name(DRIVER.stem + "_report.py"), "_relative_report_test")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(
        ROOT / "scripts/experiment_contract.py", tmp_path / "scripts/experiment_contract.py"
    )
    (tmp_path / "benchmarks/results").mkdir(parents=True)
    task = json.loads(TASK.read_text())
    task["seeds"] = [90601]
    task["datasets"] = [copy.deepcopy(task["datasets"][0])]
    task["frozen_configuration"]["phase_order"] = ["haelyn_masked"]
    task["splits"] = {"haelyn_masked": task["splits"]["haelyn_masked"]}
    task["data_seal"] = "seal.json"
    (tmp_path / "seal.json").write_text("{}\n")
    task["datasets"][0]["production_manifest"] = "production.json"
    (tmp_path / "production.json").write_text("{}\n")
    relative = Path("runs") / d.TASK_ID
    run = tmp_path / relative
    run.mkdir(parents=True)
    lock = {
        "task_id": d.TASK_ID,
        "command": task["run_command"],
        "started_at_utc": "2020-01-01T00:00:00+00:00",
    }
    (run / "task.lock.json").write_text(json.dumps(lock))
    for arm in d.ARMS:
        cell = run / "cells/haelyn_masked/seed_90601" / arm
        cell.mkdir(parents=True)
        model = Gaussians3D(
            means=torch.tensor([[0.0, 0.0, 0.0]]),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            log_scales=torch.full((1, 3), -2.0),
            opacity=torch.tensor([0.5]),
            sh=torch.ones(1, 1, 3),
        )
        models = {}
        for name in ("gaussians_init.ply", "gaussians_proxy.ply", "gaussians.ply"):
            model.save_ply(cell / name)
            models[name] = d.sha(cell / name)
        records, markers = [], []
        for i, stage in enumerate(task["stages"]):
            identity = {
                "dataset_id": "haelyn_masked",
                "arm_id": arm,
                "seed": 90601,
                "stage": stage["id"],
            }
            for boundary, offset in (("start", 0), ("end", 1)):
                markers.append(
                    {
                        **identity,
                        "step": i + offset,
                        "wall_seconds": float(i + offset),
                        "boundary": boundary,
                        "label": stage["label"],
                    }
                )
            records.append(
                {
                    **identity,
                    "step": i + 1,
                    "wall_seconds": float(i + 1),
                    "split": "validation",
                    "metric_id": "native_teacher_mse",
                    "value": 0.1,
                }
            )
        summary = {
            "status": "completed",
            "dataset_id": "haelyn_masked",
            "arm_id": arm,
            "seed": 90601,
            "models": models,
            "input_boundary": {"passed": True},
            "selected_input_binding": {},
            "effective": [],
            "records": records,
            "stage_markers": markers,
            "heldout_metrics": {k: {"J_pixel": 0.1} for k in ("initial", "proxy", "final")},
            "worker_endpoint_seconds": 3.0,
            "endpoint_excluding_validation_seconds": 2.0,
            "peak_host_bytes": 1024,
            "source_mean_drift": None if arm == "beam_reference" else 0.0,
        }
        (cell / "summary.json").write_text(json.dumps(summary))
        (cell / "process_receipt.json").write_text(json.dumps({"fixture_only": True}))
    report.aggregate(task, relative)
    payload = json.loads((run / "metrics.json").read_text())
    assert payload["commands"]["serve_report"][-1] == relative.as_posix()
    assert json.loads((run / "run_receipt.json").read_text())["status"] == "completed"
    assert (tmp_path / "benchmarks/results" / f"{d.TASK_ID}_RESULT.json").exists()
    assert len(payload["charts"][-1]["values"]) == 12


def test_coordinator_failure_is_renderable_and_retry_preserves_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    import shutil

    import pytest

    d = _driver()
    report = d.module(DRIVER.with_name(DRIVER.stem + "_report.py"), "_failure_report_test")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(
        ROOT / "scripts/experiment_contract.py", tmp_path / "scripts/experiment_contract.py"
    )
    original_module = d.module
    monkeypatch.setattr(
        d,
        "module",
        lambda path, name: (
            report if name == "_rtgs016_failed_report" else original_module(path, name)
        ),
    )
    task = json.loads(TASK.read_text())
    run = tmp_path / "runs" / d.TASK_ID
    run.mkdir(parents=True)
    lock = {
        "task_id": d.TASK_ID,
        "command": task["run_command"],
        "started_at_utc": "2020-01-01T00:00:00+00:00",
    }
    (run / "task.lock.json").write_text(json.dumps(lock))
    partial = b'{"partial": "must retain these exact bytes"}\n'

    def fail(task, run, progress):
        progress["phase"] = "warmup/haelyn_masked/hard/90601"
        (run / "metrics.json").write_bytes(partial)
        raise RuntimeError("forced fixture failure")

    monkeypatch.setattr(d, "_coordinate", fail)
    with pytest.raises(RuntimeError, match="forced fixture failure"):
        d.coordinator(task, Path("runs") / d.TASK_ID)
    assert (run / "failure_preserved/metrics.json").read_bytes() == partial
    contract = original_module(ROOT / "scripts/experiment_contract.py", "_failure_contract_test")
    completed, errors = contract._v2_source_errors(run, task, lock)
    assert not completed and not errors
    payload = json.loads((run / "metrics.json").read_text())
    assert not contract._metric_errors_v2(payload, task, lock, completed=False)
    assert payload["metrics"] == {} and payload["charts"] == []
    hashes = {str(p): d.sha(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        d.coordinator(task, run)
    assert hashes == {str(p): d.sha(p) for p in run.rglob("*") if p.is_file()}


def test_frozen_driver_rejects_fallback_nonfinite_history_and_cropped_windows() -> None:
    from types import SimpleNamespace

    import pytest

    d = _driver()
    with pytest.raises(RuntimeError, match="fallback"):
        d.check_placement({"placement_fallback_reason": "carve failed"})
    for field in ("objective_history", "elapsed_seconds"):
        values = {"objective_history": [1.0, 0.5], "elapsed_seconds": [0.0, 1.0]}
        values[field] = [float("nan")]
        with pytest.raises(RuntimeError, match="non-finite"):
            d.check_proxy(SimpleNamespace(**values))
    cropped = SimpleNamespace(width=12, height=10, fit_window=(1, 1, 10, 8))
    with pytest.raises(RuntimeError, match="fit window"):
        d.check_windows(SimpleNamespace(views=[SimpleNamespace(observation=cropped)]))


def test_warmup_uses_frozen_counts_and_main_resolves_relative_argv(
    tmp_path: Path, monkeypatch
) -> None:
    d = _driver()
    task = json.loads(TASK.read_text())
    task["frozen_configuration"]["warmup"].update(proxy_iterations=3, native_iterations=4)
    field, native, _, repair = d.configs(
        task, "hard", 90601, dataset_id="haelyn_masked", warmup=True
    )
    assert field.refit.iterations == repair.covariance_steps == 3
    assert native.iterations == 4 and native.checkpoints == (0, 4)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["driver", "--task", str(TASK), "--run", "runs/test"])
    monkeypatch.setattr(d, "validate_binding", lambda task, run: {"fixture": True})
    calls = []
    monkeypatch.setattr(d, "coordinator", lambda task, run: calls.append(run))
    d.main()
    assert calls == [(tmp_path / "runs/test").resolve()]
