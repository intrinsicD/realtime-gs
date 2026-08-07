"""CPU protocol checks for the six-folder image-backed Janelle experiment."""

from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "experiments/tasks/20260806_gaussian2d_image_refinement_janelle_frame00008.json"
DRIVER = ROOT / "scripts/experiments/20260806_gaussian2d_image_refinement_janelle_frame00008.py"
CONTRACT = ROOT / "scripts/experiment_contract.py"


def _module():
    spec = importlib.util.spec_from_file_location("janelle_image_experiment", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _task() -> dict:
    return json.loads(TASK.read_text(encoding="utf-8"))


def _contract_module():
    spec = importlib.util.spec_from_file_location("janelle_experiment_contract", CONTRACT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_owner_selected_six_folder_matrix_is_frozen() -> None:
    module = _module()
    task = _task()

    assert tuple(item["id"] for item in task["datasets"]) == module.DATASET_IDS
    assert module.DATASET_IDS == (
        "gaussians2d",
        "gaussians2d_additive",
        "gaussians2d_gaussianimage_fullres",
        "gaussians2d_native_fullres",
        "gaussians2d_structsplat_mask_contained_fullres",
        "gaussians2d_structsplat_no_boundary_fullres",
    )
    assert tuple(item["id"] for item in task["comparators"]) == module.ARMS
    assert len(task["datasets"]) * len(task["comparators"]) * len(task["seeds"]) == 36
    module._assert_task(task, require_ready=False)


@pytest.mark.parametrize(
    "dataset_id",
    (
        "gaussians2d",
        "gaussians2d_additive",
        "gaussians2d_gaussianimage_fullres",
        "gaussians2d_native_fullres",
        "gaussians2d_structsplat_mask_contained_fullres",
        "gaussians2d_structsplat_no_boundary_fullres",
    ),
)
def test_every_folder_uses_the_same_disjoint_twenty_three_three_partition(
    dataset_id: str,
) -> None:
    module = _module()
    partition = module._partition(_task(), dataset_id)

    assert len(partition["optimizer"]) == 20
    assert partition["validation"] == ("C0009", "C0022", "C0037")
    assert partition["heldout"] == ("C0014", "C0028", "C1001")
    assert not (set(partition["optimizer"]) & set(partition["validation"]))
    assert not (set(partition["optimizer"]) & set(partition["heldout"]))
    assert not (set(partition["validation"]) & set(partition["heldout"]))


def test_masked_and_unmasked_configs_change_only_the_declared_support_policy() -> None:
    module = _module()
    task = _task()
    seed = task["seeds"][0]

    masked_lift = module._field_config(task, "masked_pipeline", seed)
    unmasked_lift = module._field_config(task, "unmasked_pipeline", seed)
    assert masked_lift.mask_mode == "hard"
    assert unmasked_lift.mask_mode == "none"
    assert masked_lift.max_tracks == unmasked_lift.max_tracks == 256
    assert masked_lift.target_component_cap == unmasked_lift.target_component_cap == 2048

    masked_train = module._train_config(task, "masked_pipeline", seed, iterations=20)
    unmasked_train = module._train_config(task, "unmasked_pipeline", seed, iterations=20)
    assert masked_train.use_masks is True
    assert unmasked_train.use_masks is False
    assert masked_train.iterations == unmasked_train.iterations == 20
    assert masked_train.density.absgrad is unmasked_train.density.absgrad is True
    assert masked_train.internal_checkpoint_evaluation is False
    assert masked_train.reset_cuda_peak_stats is False


def test_hybrid_data_seal_contains_all_compact_fields_janelle_rgb_and_masks() -> None:
    task = _task()
    seal = json.loads((ROOT / task["data_seal"]).read_text(encoding="utf-8"))
    paths = {item["path"] for item in seal["files"]}

    assert seal["input_profile"] == "rgb"
    assert len(paths) == 215
    assert sum(path.endswith(".rtgsv") for path in paths) == 156
    assert sum("/rgb/" in path and path.endswith(".jpg") for path in paths) == 26
    assert sum("/mask/" in path and path.endswith(".png") for path in paths) == 26
    for dataset in task["datasets"]:
        assert dataset["compact_manifest"] in paths


def test_full_canvas_psnr_is_measured_against_the_unmasked_janelle_photo() -> None:
    module = _module()
    height = width = 16
    image = torch.ones((height, width, 3), dtype=torch.float32)
    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[4:12, 4:12] = 1
    prediction = image.clone()
    renderer = SimpleNamespace(
        render=lambda *_args, **_kwargs: SimpleNamespace(
            color=prediction,
            alpha=torch.ones((height, width), dtype=torch.float32),
        )
    )
    scene = SimpleNamespace(
        images=[image],
        cameras=[SimpleNamespace(to=lambda _device: object())],
        masks=[mask],
        view_names=["C0014"],
    )
    gaussians = SimpleNamespace(means=torch.empty(0))

    metrics = module._per_view_metrics(scene, gaussians, renderer, [0])["aggregate"]
    assert metrics["psnr_full"] == pytest.approx(120.0)


def test_validation_auc_and_heldout_endpoint_order_are_fixed() -> None:
    module = _module()
    records = [
        {"optimizer_wall_seconds": 0.0, "metrics": {"psnr_fg": 10.0}},
        {"optimizer_wall_seconds": 2.0, "metrics": {"psnr_fg": 14.0}},
        {"optimizer_wall_seconds": 4.0, "metrics": {"psnr_fg": 18.0}},
    ]
    assert module._validation_auc(records) == pytest.approx(14.0)

    source = inspect.getsource(module._run_worker)
    endpoint = source.index('final.save_ply(output / "gaussians.ply")')
    heldout_load = source.index('view_ids=partition["heldout"]')
    assert endpoint < heldout_load
    assert 'if mode != "measured":' in source[:heldout_load]
    assert '"heldout_outcome_access": False' in source[:heldout_load]


def test_old_direct_worker_scratch_surface_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "forbidden-worker-output"
    process = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--worker",
            "--scratch",
            "--output",
            str(output),
            "--iterations",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert process.returncode == 2
    assert "unrecognized arguments: --worker" in process.stderr
    assert not output.exists()


def test_measured_worker_without_authenticated_binding_cannot_open_heldout(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "unauthenticated-measured-cell"

    code = module._run_worker(
        task_path=TASK,
        output=output,
        dataset_id=module.DATASET_IDS[0],
        seed=80601,
        arm="masked_pipeline",
        iterations=1,
        mode="measured",
        official_binding=None,
    )

    assert code == 1
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["phase"] == "input_alignment"
    assert "authenticated worker binding" in failure["message"]
    assert not (output / "heldout_metrics.json").exists()


def test_tampered_worker_ticket_fails_before_task_or_input_access(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    run = tmp_path / module.RUN_RELATIVE
    tickets = run / "worker_tickets"
    tickets.mkdir(parents=True)
    ticket = tickets / "tampered.json"
    module._write_json(
        ticket,
        {
            "body": {"task_id": module.TASK_ID, "mode": "measured"},
            "hmac_sha256": "0" * 64,
        },
    )
    monkeypatch.setenv(module.WORKER_SECRET_ENV, "1" * 64)

    with pytest.raises(RuntimeError, match="authentication failed"):
        module._run_ticket_worker(str(ticket))


def _official_lock_fixture(module, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_verify_source_binding", lambda _task: {})
    task = _task()
    task["status"] = "ready"
    task["protocol_review"] = {
        "reviewer": "independent-reviewer",
        "verdict": "approved",
        "protocol_sha256": "a" * 64,
        "artifact": module.CANONICAL_REVIEW_RELATIVE.as_posix(),
    }
    task_path = tmp_path / module.TASK_RELATIVE
    module._write_json(task_path, task)
    review_path = tmp_path / module.CANONICAL_REVIEW_RELATIVE
    review_path.parent.mkdir(parents=True)
    review_path.write_text("approved prospective review\n", encoding="utf-8")
    seal_path = tmp_path / task["data_seal"]
    module._write_json(seal_path, {"fixture": True})
    run = tmp_path / module.RUN_RELATIVE
    run.mkdir(parents=True)
    lock = {
        "schema_version": 2,
        "task_id": module.TASK_ID,
        "task_path": module.TASK_RELATIVE.as_posix(),
        "task_sha256": module._sha256_file(task_path),
        "protocol_sha256": "a" * 64,
        "protocol_review": task["protocol_review"],
        "protocol_review_artifact_sha256": module._sha256_file(review_path),
        "data_seal_path": task["data_seal"],
        "data_seal_sha256": module._sha256_file(seal_path),
        "source_commit": "b" * 40,
        "source_dirty": True,
        "source_diff_sha256": "c" * 64,
        "development": True,
        "started_at_utc": "2026-08-06T00:00:00+00:00",
        "command": task["run_command"],
        "report_template_version": 2,
    }
    module._write_json(run / "task.lock.json", lock)
    return task, task_path, run, review_path, lock


def test_official_lock_rejects_review_deletion_drift_wrong_path_and_malformed_lock(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    task, task_path, run, review_path, lock = _official_lock_fixture(module, tmp_path, monkeypatch)
    assert module._official_lock(task_path, run, task) == lock

    original = review_path.read_bytes()
    review_path.unlink()
    with pytest.raises(RuntimeError, match="absent or not a regular file"):
        module._official_lock(task_path, run, task)
    review_path.write_bytes(original + b"drift")
    with pytest.raises(RuntimeError, match="protocol review artifact bytes"):
        module._official_lock(task_path, run, task)
    review_path.write_bytes(original)

    wrong_path_task = json.loads(json.dumps(task))
    wrong_path_task["protocol_review"]["artifact"] = "experiments/reviews/wrong.md"
    with pytest.raises(RuntimeError, match="canonical approved protocol review"):
        module._official_lock(task_path, run, wrong_path_task)

    malformed = dict(lock)
    malformed.pop("protocol_review_artifact_sha256")
    module._write_json(run / "task.lock.json", malformed)
    with pytest.raises(RuntimeError, match="wrong keys"):
        module._official_lock(task_path, run, task)


def test_process_owned_cuda_peak_boundary_is_explicit(monkeypatch) -> None:
    module = _module()
    calls = []

    class FakeCuda:
        peak_allocated = 123
        peak_reserved = 456

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            calls.append("current")
            return 0

        @staticmethod
        def empty_cache():
            calls.append("empty")

        @staticmethod
        def reset_peak_memory_stats(device):
            calls.append(("reset_peak", device))

        @staticmethod
        def reset_accumulated_memory_stats(device):
            calls.append(("reset_accumulated", device))

        @staticmethod
        def synchronize(device):
            calls.append(("synchronize", device))

        @classmethod
        def max_memory_allocated(cls, _device):
            return cls.peak_allocated

        @classmethod
        def max_memory_reserved(cls, _device):
            return cls.peak_reserved

    fake_torch = SimpleNamespace(cuda=FakeCuda)
    assert module._start_cuda_measurement(fake_torch) == 0
    assert calls == [
        "current",
        "empty",
        ("reset_peak", 0),
        ("reset_accumulated", 0),
        ("synchronize", 0),
    ]

    monkeypatch.setattr(module.time, "perf_counter", lambda: 20.0)
    endpoint = module._freeze_resource_endpoint(
        fake_torch,
        SimpleNamespace(type="cuda"),
        cell_started=2.0,
        measurement_started=5.0,
    )
    FakeCuda.peak_allocated = 999
    FakeCuda.peak_reserved = 999
    assert endpoint["measurement_endpoint_wall_seconds"] == 18.0
    assert endpoint["measurement_total_wall_seconds"] == 15.0
    assert endpoint["peak_cuda_allocated_bytes"] == 123
    assert endpoint["peak_cuda_reserved_bytes"] == 456


def _make_valid_cell_bundle(module, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    task = _task()
    run = tmp_path / "runs" / module.TASK_ID
    run.mkdir(parents=True)
    lock = {
        "protocol_sha256": "a" * 64,
        "data_seal_sha256": "b" * 64,
    }
    module._write_json(run / "task.lock.json", lock)
    dataset_id = module.DATASET_IDS[0]
    arm = module.ARMS[0]
    seed = task["seeds"][0]
    output = module._cell_dir(run, dataset_id, seed, arm)
    output.mkdir(parents=True)
    effective = {
        "field_lift": module._json_safe(module._field_config(task, arm, seed)),
        "rgb_refinement": module._json_safe(
            module._train_config(
                task,
                arm,
                seed,
                iterations=task["frozen_configuration"]["rgb_refinement"]["iterations"],
            )
        ),
    }
    summary = {
        "status": "completed",
        "task_id": module.TASK_ID,
        "dataset_id": dataset_id,
        "arm": arm,
        "seed": seed,
        "warmup": False,
        "heldout_opened_after_endpoint_saved": True,
        "measurement_endpoint_before_heldout": True,
        "metrics": {item["id"]: 1.0 for item in task["primary_metrics"]},
        "input_binding": {
            "camera_records_sha256": "c" * 64,
            "optimizer_validation_image_sha256": "d" * 64,
            "heldout_image_sha256": "1" * 64,
        },
        "effective": effective,
    }
    field = {
        "manifest_sha256": "e" * 64,
        "loaded_optimizer_compact_sha256": "f" * 64,
    }
    module._write_json(output / "summary.json", summary)
    module._write_json(output / "field_lift.json", field)
    for name in module.MEASURED_CELL_ARTIFACTS:
        path = output / name
        if path.exists():
            continue
        if path.suffix == ".json":
            module._write_json(path, {"fixture": name})
        else:
            path.write_bytes(f"fixture:{name}".encode("ascii"))
    binding = {
        "mode": "measured",
        "iterations": task["frozen_configuration"]["rgb_refinement"]["iterations"],
        "task_lock_sha256": module._sha256_file(run / "task.lock.json"),
        "protocol_sha256": lock["protocol_sha256"],
        "data_seal_sha256": lock["data_seal_sha256"],
    }
    module._write_cell_receipt(
        output=output,
        task=task,
        official_binding=binding,
        summary=summary,
        compact_receipt=field,
        camera_receipt={"records_sha256": "c" * 64},
        image_input_receipt={"records_sha256": "d" * 64},
    )
    return task, run, lock, output, dataset_id, arm, seed


def test_cell_resume_rejects_copied_identity_and_tampered_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    task, run, lock, output, dataset_id, arm, seed = _make_valid_cell_bundle(
        module, tmp_path, monkeypatch
    )
    module._validate_cell_bundle(
        run=run,
        task=task,
        lock=lock,
        dataset_id=dataset_id,
        seed=seed,
        arm=arm,
        mode="measured",
    )

    copied_arm = module.ARMS[1]
    copied = module._cell_dir(run, dataset_id, seed, copied_arm)
    shutil.copytree(output, copied)
    with pytest.raises(RuntimeError, match="receipt (arm|output_path) differs"):
        module._validate_cell_bundle(
            run=run,
            task=task,
            lock=lock,
            dataset_id=dataset_id,
            seed=seed,
            arm=copied_arm,
            mode="measured",
        )

    with (output / "gaussians.ply").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="artifact hash differs"):
        module._validate_cell_bundle(
            run=run,
            task=task,
            lock=lock,
            dataset_id=dataset_id,
            seed=seed,
            arm=arm,
            mode="measured",
        )


def test_viewers_are_frozen_to_open_only_after_measurement_matrix() -> None:
    module = _module()
    task = _task()
    for dataset_id in module.DATASET_IDS:
        command = module._viewer_command(
            task,
            dataset_id,
            port=module._viewer_port(task, dataset_id),
        )
        assert "--open" in command
        assert "--no-open" not in command
    source = inspect.getsource(module._run_parent)
    assert source.index("exit_data_check") < source.index("_aggregate_run")
    assert source.index("_aggregate_run") < source.index("_launch_viewers")


def test_post_matrix_failure_is_schema_recorded_and_same_root_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    task = _task()
    task["status"] = "ready"
    task_path = tmp_path / module.TASK_RELATIVE
    module._write_json(task_path, task)
    run = tmp_path / module.RUN_RELATIVE
    run.mkdir(parents=True)
    lock = {"started_at_utc": "2026-08-06T00:00:00+00:00"}
    module._write_json(run / "task.lock.json", lock)
    warmup = task["frozen_configuration"]["warmup"]
    (run / "warmup" / warmup["dataset_id"] / warmup["arm_id"]).mkdir(parents=True)
    for dataset_id in module.DATASET_IDS:
        for seed in task["seeds"]:
            for arm in module.ARMS:
                module._cell_dir(run, dataset_id, seed, arm).mkdir(parents=True)

    monkeypatch.setattr(module, "_assert_task", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_official_lock", lambda *_args, **_kwargs: lock)
    monkeypatch.setattr(module, "_verify_full_data_seal", lambda _task: {"data_seal_sha256": "x"})
    monkeypatch.setattr(module, "_verify_source_binding", lambda _task: {"source": "x"})
    monkeypatch.setattr(module, "_validate_cell_bundle", lambda **kwargs: dict(kwargs))
    monkeypatch.setattr(module, "_environment", lambda: {})
    calls = {"aggregate": 0}

    def aggregate(_run, _task):
        calls["aggregate"] += 1
        if calls["aggregate"] == 1:
            raise RuntimeError("preview fixture failure")
        return {}, []

    monkeypatch.setattr(module, "_aggregate_run", aggregate)
    monkeypatch.setattr(module, "_launch_viewers", lambda *_args: None)
    monkeypatch.setattr(module, "_write_result_evidence", lambda *_args: None)

    assert module._run_parent(task_path, run) == 1
    failed = json.loads((run / "run_receipt.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "post_matrix_publication"
    failed_metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert failed_metrics["decision"] == "failed"
    assert failed_metrics["charts"] == []
    assert failed_metrics["evidence"] == []

    assert module._run_parent(task_path, run) == 0
    completed = json.loads((run / "run_receipt.json").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert calls["aggregate"] == 2


@pytest.mark.parametrize("failed_filename", ("progress.json", "run_receipt.json"))
def test_terminal_completion_write_failure_fails_closed_and_same_root_resumes(
    tmp_path: Path, monkeypatch, failed_filename: str
) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    task = _task()
    task["status"] = "ready"
    task_path = tmp_path / module.TASK_RELATIVE
    module._write_json(task_path, task)
    run = tmp_path / module.RUN_RELATIVE
    run.mkdir(parents=True)
    lock = {"started_at_utc": "2026-08-06T00:00:00+00:00"}
    module._write_json(run / "task.lock.json", lock)
    warmup = task["frozen_configuration"]["warmup"]
    (run / "warmup" / warmup["dataset_id"] / warmup["arm_id"]).mkdir(parents=True)
    for dataset_id in module.DATASET_IDS:
        for seed in task["seeds"]:
            for arm in module.ARMS:
                module._cell_dir(run, dataset_id, seed, arm).mkdir(parents=True)

    monkeypatch.setattr(module, "_assert_task", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_official_lock", lambda *_args, **_kwargs: lock)
    monkeypatch.setattr(module, "_verify_full_data_seal", lambda _task: {"data_seal_sha256": "x"})
    monkeypatch.setattr(module, "_verify_source_binding", lambda _task: {"source": "x"})
    monkeypatch.setattr(module, "_validate_cell_bundle", lambda **kwargs: dict(kwargs))
    monkeypatch.setattr(module, "_environment", lambda: {})
    monkeypatch.setattr(module, "_aggregate_run", lambda *_args: ({}, []))
    monkeypatch.setattr(module, "_launch_viewers", lambda *_args: None)
    monkeypatch.setattr(module, "_write_result_evidence", lambda *_args: None)

    original_write_json = module._write_json
    injected = {"done": False}

    def fail_one_completed_terminal_write(path, value):
        if (
            not injected["done"]
            and path.name == failed_filename
            and isinstance(value, dict)
            and value.get("status") == "completed"
        ):
            injected["done"] = True
            raise OSError(f"injected completed {failed_filename} write failure")
        original_write_json(path, value)

    monkeypatch.setattr(module, "_write_json", fail_one_completed_terminal_write)

    assert module._run_parent(task_path, run) == 1
    failed_receipt = json.loads((run / "run_receipt.json").read_text(encoding="utf-8"))
    failed_progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    assert failed_receipt["status"] == "failed"
    assert failed_receipt["failure_phase"] == "post_matrix_publication"
    assert failed_progress["status"] == "failed"
    assert "completed" not in {
        failed_receipt["status"],
        failed_progress["status"],
    }

    assert module._run_parent(task_path, run) == 0
    completed_receipt = json.loads((run / "run_receipt.json").read_text(encoding="utf-8"))
    completed_progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    assert completed_receipt["status"] == "completed"
    assert completed_progress["status"] == "completed"


def test_viewer_startup_failure_receipt_is_retry_safe(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    task = _task()
    run = tmp_path / module.RUN_RELATIVE
    for dataset_id in module.DATASET_IDS:
        (run / "datasets" / dataset_id).mkdir(parents=True)

    attempts: dict[int, int] = {}
    alive_pids: set[int] = set()
    ready_ports: set[int] = set()

    class FakeProcess:
        def __init__(self, command):
            self.port = int(command[command.index("--port") + 1])
            attempts[self.port] = attempts.get(self.port, 0) + 1
            self.pid = 10000 + self.port + attempts[self.port]
            self.failed = self.port == 8402 and attempts[self.port] == 1
            if not self.failed:
                alive_pids.add(self.pid)
                ready_ports.add(self.port)

        def poll(self):
            return 1 if self.failed else None

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda command, **_kwargs: FakeProcess(command),
    )
    monkeypatch.setattr(module, "_pid_alive", lambda pid: pid in alive_pids)
    monkeypatch.setattr(module, "_viewer_port_ready", lambda _host, port: port in ready_ports)

    with pytest.raises(RuntimeError, match="process/HTTP startup probe"):
        module._launch_viewers(run, task)
    first = json.loads((run / "viewer_launch_receipt.json").read_text(encoding="utf-8"))
    assert first["status"] == "incomplete"
    assert len(first["records"]) == 6

    module._launch_viewers(run, task)
    second = json.loads((run / "viewer_launch_receipt.json").read_text(encoding="utf-8"))
    assert second["status"] == "completed"
    assert len(second["records"]) == 6
    assert sum(bool(item["reused"]) for item in second["records"]) == 5
    assert attempts[8402] == 2


def test_result_publication_recovers_exact_partial_evidence_and_rejects_conflict(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    task = _task()
    summaries = {
        dataset_id: {
            "metrics": {
                "masked_pipeline_heldout_foreground_psnr": 1.0,
                "unmasked_pipeline_heldout_foreground_psnr": 2.0,
            }
        }
        for dataset_id in module.DATASET_IDS
    }
    result_json = tmp_path / "benchmarks/results" / f"{module.TASK_ID}_RESULT.json"
    result_md = tmp_path / "benchmarks/results" / f"{module.TASK_ID}_RESULT.md"
    module._write_result_evidence(task, summaries, [{}] * 36)
    original_json = result_json.read_bytes()
    result_md.unlink()
    module._write_result_evidence(task, summaries, [{}] * 36)
    assert result_json.read_bytes() == original_json
    assert result_md.is_file()

    result_json.write_text("conflict\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing canonical evidence differs"):
        module._write_result_evidence(task, summaries, [{}] * 36)


def test_dataset_summary_exposes_every_resource_metric_and_native_optimizer_clock(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    task = _task()
    dataset_id = module.DATASET_IDS[0]
    primary_ids = [item["id"] for item in task["primary_metrics"]]
    cells = []
    validation_metrics = {
        "psnr_fg": 10.0,
        "psnr_full": 9.0,
        "ssim_crop": 0.5,
        "alpha_iou": 0.4,
        "alpha_outside": 0.1,
    }
    for seed in task["seeds"]:
        for arm in module.ARMS:
            cell = module._cell_dir(tmp_path, dataset_id, seed, arm)
            module._write_json(
                cell / "field_lift.json",
                {"source_component_counts_all_views": {"C0001": 2}},
            )
            module._write_json(
                cell / "validation_metrics.json",
                {
                    "records": [
                        {
                            "step": 0,
                            "optimizer_wall_seconds": 0.0,
                            "cell_wall_seconds": 2.0,
                            "metrics": validation_metrics,
                        },
                        {
                            "step": 100,
                            "optimizer_wall_seconds": 1.0,
                            "cell_wall_seconds": 4.0,
                            "metrics": validation_metrics,
                        },
                    ]
                },
            )
            cells.append(
                {
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "arm": arm,
                    "metrics": {metric_id: 1.0 for metric_id in primary_ids},
                    "resource": {},
                }
            )

    summary = module._dataset_summary(
        tmp_path,
        task,
        dataset_id,
        cells,
        {arm: [] for arm in module.ARMS},
    )
    curve_ids = {item["id"] for item in summary["curves"]}
    required_resources = {
        "field_lift_wall_seconds",
        "rgb_refinement_wall_seconds",
        "validation_observer_seconds",
        "measurement_endpoint_wall_seconds",
        "total_cell_wall_seconds",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "peak_rss_bytes",
        "final_gaussians",
    }
    assert required_resources <= curve_ids
    native_curves = [
        item for item in summary["curves"] if item["id"].endswith("native_optimizer_time")
    ]
    assert len(native_curves) == 5
    assert all(
        item["x_label"] == "native optimizer time excluding validation observers (s)"
        for item in native_curves
    )


def test_all_six_rendered_children_enumerate_required_metrics_and_both_clocks(
    tmp_path: Path,
) -> None:
    contract = _contract_module()
    task = _task()
    primary = task["primary_metrics"]
    metadata = {
        item["id"]: {
            "label": item["label"],
            "unit": item["unit"],
            "group": "Resources",
            "direction": item["direction"],
        }
        for item in primary
    }
    summaries = {}
    history_records = []
    for dataset in task["datasets"]:
        dataset_id = dataset["id"]
        curves = [
            {
                "id": item["id"],
                "title": item["id"],
                "x_label": "seed",
                "unit": item["unit"],
                "direction": item["direction"],
                "series": [{"label": "masked", "points": [{"x": 80601, "value": 1.0}]}],
            }
            for item in primary
        ]
        curves.append(
            {
                "id": "validation_psnr_fg_native_optimizer_time",
                "title": "native_optimizer_clock_fixture",
                "x_label": "native optimizer time excluding validation observers (s)",
                "unit": "dB",
                "direction": "higher",
                "series": [{"label": "masked", "points": [{"x": 0.0, "value": 1.0}]}],
            }
        )
        summaries[dataset_id] = {
            "title": dataset_id,
            "summary": "fixture",
            "metrics": {item["id"]: 1.0 for item in primary},
            "metric_metadata": metadata,
            "charts": [],
            "curves": curves,
            "artifacts": [],
            "commands": {"viewer": ["rtgs", "view", dataset_id]},
            "notes": ["fixture"],
        }
        history_records.append(
            {
                "step": 0,
                "wall_seconds": 0.0,
                "stage": "input_alignment",
                "dataset_id": dataset_id,
                "arm_id": "masked_pipeline",
                "seed": 80601,
                "split": "diagnostic",
                "metric_id": "clock_fixture",
                "value": 1.0,
            }
        )
    links = contract._render_dataset_pages(
        tmp_path,
        task,
        {"dataset_summaries": summaries},
        {
            "records": history_records,
            "stage_markers": [],
            "metric_metadata": {
                "clock_fixture": {
                    "label": "Cell-wall clock fixture",
                    "unit": "score",
                    "group": "Boundary",
                    "direction": "descriptive",
                }
            },
        },
    )
    assert tuple(links) == tuple(item["id"] for item in task["datasets"])
    for dataset_id, relative in links.items():
        body = (tmp_path / relative).read_text(encoding="utf-8")
        for item in primary:
            assert item["id"] in body
        assert "native optimizer time excluding validation observers (s)" in body
        assert "worker cell wall time from worker start (s)" in body
        assert dataset_id in body
