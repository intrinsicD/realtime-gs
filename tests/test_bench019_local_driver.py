"""Coordinator fail-closed tests with synthetic bytes; never open local captures."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver():
    path = (
        Path(__file__).parents[1]
        / "scripts/experiments/20260907_bench019_local_stage_frames00008_00009.py"
    )
    spec = importlib.util.spec_from_file_location("bench019_coordinator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_rejects_tampering_and_symlink_parent(tmp_path, driver):
    base = tmp_path / "base"
    base.mkdir()
    path = base / "field"
    path.write_bytes(b"frozen")
    inventory = {"field": driver.sha256(path)}
    driver.verify_file_inventory(base, inventory)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        driver.verify_file_inventory(base, inventory)
    external = tmp_path / "external"
    external.mkdir()
    (external / "field").write_bytes(b"frozen")
    (base / "link").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="nonordinary"):
        driver.verify_file_inventory(base, {"link/field": inventory["field"]})
    with pytest.raises(ValueError, match="unsafe"):
        driver.verify_file_inventory(base, {"../external/field": inventory["field"]})


def test_canonical_root_rejects_alias_even_if_destination_matches(tmp_path, driver):
    task = {"task_id": driver.TASK_ID}
    run = tmp_path / "runs" / driver.TASK_ID
    run.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(run, target_is_directory=True)
    assert driver.canonical_run(task, run, root=tmp_path) == run
    with pytest.raises(ValueError, match="canonical"):
        driver.canonical_run(task, alias, root=tmp_path)


def test_external_source_addition_is_not_an_allowed_metadata_commit(tmp_path, driver):
    source = tmp_path / "src"
    source.mkdir()
    code = source / "existing.py"
    code.write_text("a = 1\n")
    binding = {
        "root": str(tmp_path),
        "patterns": ["src/**/*.py"],
        "files": {"src/existing.py": driver.sha256(code)},
    }
    driver.verify_external_sources(binding)
    (tmp_path / "review.md").write_text("Independent metadata-only review\n")
    driver.verify_external_sources(binding)
    (source / "added.py").write_text("a = 2\n")
    with pytest.raises(ValueError, match="inventory changed"):
        driver.verify_external_sources(binding)


def _task():
    return {
        "seeds": [19001, 19002, 19003],
        "downstream_config": {"n_init_3d": 256},
        "resource_monitor": {"warmup_seed": 19999},
        "aa_replay": {
            "frame_id": "frame_00008",
            "family_id": "native_additive",
            "additional_family_id": "structsplat_contained",
            "seed": 19001,
            "foreground_psnr_absolute_tolerance_db": 0.05,
            "alpha_iou_absolute_tolerance": 0.01,
        },
    }


@pytest.mark.parametrize(
    "mode,frame,family,seed,replicate",
    [
        ("input-worker", "frame_00008", "native_additive", 19001, "primary"),
        ("cell-worker", "frame_00009", "native_additive", 19001, "aa"),
        ("cell-worker", "frame_00008", "structsplat_normalized", 19001, "aa"),
        ("evaluate-worker", "frame_00008", "native_additive", 19999, "warmup"),
        ("cell-worker", "frame_00008", "native_additive", 19999, "primary"),
        ("preview-worker", "frame_00008", "structsplat_contained", 19001, "primary"),
    ],
)
def test_private_worker_cannot_expand_frozen_matrix(driver, mode, frame, family, seed, replicate):
    args = SimpleNamespace(mode=mode, frame=frame, family=family, seed=seed, replicate=replicate)
    with pytest.raises(ValueError):
        driver.validate_worker_selection(_task(), args)


def _aa_fixture(driver, run):
    for family in ("native_additive", "structsplat_contained"):
        for replicate in ("primary", "aa"):
            cell = driver.cell_directory(run, "frame_00008", family, 19001, replicate)
            driver.write_json(cell / "config.json", {"family": family, "frozen": True})
            driver.write_json(
                cell / "receipt.json",
                {"fields": {"digest": "a" * 64}, "metrics": {"final_gaussians": 256}},
            )
            driver.write_json(
                cell / "evaluation/evaluation.json",
                {"metrics": {"heldout_foreground_psnr": 25.0, "heldout_alpha_iou": 0.8}},
            )


def test_aa_accepts_exact_replay_and_records_both_families(tmp_path, driver):
    _aa_fixture(driver, tmp_path)
    driver.aa_check(_task(), tmp_path)
    receipt = driver.read_json(tmp_path / "aa_replay.json")
    assert receipt["status"] == "passed"
    assert {check["family"] for check in receipt["checks"]} == {
        "native_additive",
        "structsplat_contained",
    }


@pytest.mark.parametrize("changed", ["config", "field", "quality", "count", "nonfinite"])
def test_aa_rejects_drift_before_main_matrix(tmp_path, driver, changed):
    _aa_fixture(driver, tmp_path)
    cell = driver.cell_directory(tmp_path, "frame_00008", "native_additive", 19001, "aa")
    if changed == "config":
        driver.write_json(cell / "config.json", {"frozen": False}, exclusive=False)
    elif changed in {"field", "count"}:
        receipt = driver.read_json(cell / "receipt.json")
        if changed == "field":
            receipt["fields"]["digest"] = "b" * 64
        else:
            receipt["metrics"]["final_gaussians"] = 257
        driver.write_json(cell / "receipt.json", receipt, exclusive=False)
    else:
        path = cell / "evaluation/evaluation.json"
        if changed == "nonfinite":
            path.write_text('{"metrics":{"heldout_foreground_psnr":NaN,"heldout_alpha_iou":0.8}}')
        else:
            driver.write_json(
                path,
                {"metrics": {"heldout_foreground_psnr": 25.2, "heldout_alpha_iou": 0.8}},
                exclusive=False,
            )
    with pytest.raises((ValueError, RuntimeError)):
        driver.aa_check(_task(), tmp_path)
    assert not (tmp_path / "aa_replay.json").exists()


@pytest.mark.parametrize(
    "drift", ["added_input", "changed_input", "seeds", "schedule", "unapproved"]
)
def test_phase2_rechecks_generated_inputs_and_both_protocols(tmp_path, monkeypatch, driver, drift):
    from rtgs import bench019

    monkeypatch.setattr(driver, "ROOT", tmp_path)
    run = tmp_path / "runs" / driver.TASK_ID
    task = {
        "datasets": [{"id": "frame_00008"}],
        "comparators": [{"id": "native_additive"}],
        "splits": {"frame_00008": {"train": ["C1"], "heldout": ["C2"]}},
        "seeds": [19001, 19002, 19003],
        "run_command": ["python", "frozen.py"],
    }
    task_path = tmp_path / "experiments/tasks" / f"{driver.TASK_ID}.json"
    driver.write_json(task_path, task)
    family_dir = run / "inputs/frame_00008/native_additive"
    for name in ("manifest.json", "stage1_metrics.json"):
        driver.write_json(family_dir / name, {"frozen": True})
    driver.write_json(run / "generated_inputs.json", {"files": driver.input_inventory(run)})
    for name in ("environment.json", "gaussians.config.json"):
        driver.write_json(run / name, {"frozen": True})
    protocol = {
        "state": "frozen",
        "downstream": {
            "outcome_root": str(run / "downstream"),
            "task_manifest": {"sha256": driver.sha256(task_path)},
            "dataset_manifest": {"sha256": driver.sha256(run / "generated_inputs.json")},
            "environment": {"sha256": driver.sha256(run / "environment.json")},
            "schedule_config": {"sha256": driver.sha256(run / "gaussians.config.json")},
            "seeds": task["seeds"],
            "initializers": ["field_sweep"],
            "command": task["run_command"],
            "working_directory": str(tmp_path),
        },
        "captures": [
            {
                "id": "stage",
                "frames": [
                    {
                        "id": "frame_00008",
                        "split": task["splits"]["frame_00008"],
                        "families": [
                            {
                                "id": "native_additive",
                                "field_manifest": {
                                    "sha256": driver.sha256(family_dir / "manifest.json")
                                },
                                "stage1_metrics": {
                                    "sha256": driver.sha256(family_dir / "stage1_metrics.json")
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }

    def formal_guard(value, **kwargs):
        assert kwargs["allow_review"] is False
        if value["state"] != "frozen":
            raise ValueError("independent approval missing")

    monkeypatch.setattr(bench019, "protocol_identity", formal_guard)
    protocol_path = run / "protocol/structsplat.frozen.json"
    driver.write_json(protocol_path, protocol)
    driver.verify_phase2(task, run)
    if drift == "added_input":
        (family_dir / "extra.bin").write_bytes(b"extra")
    elif drift == "changed_input":
        (family_dir / "manifest.json").write_text("{}")
    elif drift == "seeds":
        protocol["downstream"]["seeds"] = [19004]
    elif drift == "schedule":
        protocol["downstream"]["schedule_config"]["sha256"] = "0" * 64
    else:
        protocol["state"] = "review"
    driver.write_json(protocol_path, protocol, exclusive=False)
    with pytest.raises(ValueError):
        driver.verify_phase2(task, run)


def test_initial_malformed_task_still_writes_failure_receipt(tmp_path, monkeypatch, driver):
    run = tmp_path / "runs" / driver.TASK_ID
    driver.write_json(run / "task.lock.json", {"started_at_utc": "2026-09-07T00:00:00+00:00"})
    task = tmp_path / "broken.json"
    task.write_text("{invalid json")
    args = SimpleNamespace(
        mode="run", task=task, run_dir=run, frame=None, family=None, seed=0, replicate="primary"
    )
    monkeypatch.setattr(driver, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    original = driver.canonical_run
    monkeypatch.setattr(
        driver, "canonical_run", lambda value, path: original(value, path, root=tmp_path)
    )
    assert driver.main() == 1
    receipt = driver.read_json(run / "run_receipt.json")
    assert receipt["status"] == "failed" and receipt["exit_code"] == 1
    assert receipt["started_at_utc"] == "2026-09-07T00:00:00+00:00"
    assert (run / "failures/run-None-None-0-primary.json").is_file()
