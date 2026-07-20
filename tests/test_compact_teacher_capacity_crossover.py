from __future__ import annotations

import builtins
import dataclasses
import hashlib
import json
import math
import os
import socket
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from benchmarks import compact_teacher_capacity_crossover as experiment

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.data.reconstruction_inputs import ReconstructionInputs


def _camera(eye: tuple[float, float, float]) -> Camera:
    return Camera.look_at(
        torch.tensor(eye),
        torch.tensor([0.0, 0.0, 0.0]),
        width=experiment.NATIVE_SIZE[0],
        height=experiment.NATIVE_SIZE[1],
        fov_x_deg=60.0,
    )


def _field(
    view_id: str,
    count: int,
    *,
    residuals: bool,
    fit_window: tuple[int, int, int, int] = (0, 0, 64, 64),
) -> GaussianObservationField:
    index = torch.arange(count, dtype=torch.float32)
    local = torch.stack(
        [
            8.125 + torch.remainder(index, 16),
            8.375 + torch.remainder(torch.div(index, 16, rounding_mode="floor"), 16),
        ],
        dim=-1,
    )
    origin = torch.tensor([fit_window[0] + 0.5, fit_window[1] + 0.5])
    stored = local + origin
    correction = torch.full((count, 2), 0.125, dtype=torch.float32) if residuals else None
    return GaussianObservationField(
        width=experiment.NATIVE_SIZE[0],
        height=experiment.NATIVE_SIZE[1],
        means=stored,
        log_scales=torch.zeros(count, 2),
        rotations=torch.zeros(count),
        colors=torch.full((count, 3), 0.5),
        amplitudes=torch.ones(count),
        mean_residuals=correction,
        fit_window=fit_window,
        view_id=view_id,
        n_init=count,
        provider="synthetic_fixture",
        producer_version="capacity-crossover-test",
        producer_source_digest="a" * 64,
        fit_config_digest="b" * 64,
    )


def _paired_inputs() -> tuple[ReconstructionInputs, ReconstructionInputs]:
    eyes = (
        (2.0, 0.0, 0.5),
        (1.2, 1.6, 0.5),
        (-0.4, 2.0, 0.5),
        (-1.8, 0.8, 0.5),
        (-1.8, -0.8, 0.5),
        (-0.4, -2.0, 0.5),
        (1.2, -1.6, 0.5),
    )
    cameras = [_camera(eye) for eye in eyes]
    old = ReconstructionInputs(
        observations=[
            _field(view, experiment.OLD_COMPONENTS_PER_VIEW, residuals=False)
            for view in experiment.TRAIN_VIEWS
        ],
        cameras=cameras,
        view_names=list(experiment.TRAIN_VIEWS),
        name="old-test",
    )
    new = ReconstructionInputs(
        observations=[
            _field(view, experiment.NEW_COMPONENTS_PER_VIEW, residuals=True)
            for view in experiment.TRAIN_VIEWS
        ],
        cameras=cameras,
        view_names=list(experiment.TRAIN_VIEWS),
        bounds_hint=(
            torch.tensor(experiment.BOUNDS_CENTER, dtype=torch.float32),
            experiment.BOUNDS_EXTENT,
        ),
        name="new-test",
    )
    return old, new


def _replace_inputs(
    inputs: ReconstructionInputs,
    **changes,
) -> ReconstructionInputs:
    values = {
        "observations": inputs.observations,
        "cameras": inputs.cameras,
        "view_names": inputs.view_names,
        "points": inputs.points,
        "point_visibility": inputs.point_visibility,
        "bounds_hint": inputs.bounds_hint,
        "name": inputs.name,
        "archive_stats": inputs.archive_stats,
    }
    values.update(changes)
    return ReconstructionInputs(**values)


def _tiny_inputs(*, residuals: bool = True) -> ReconstructionInputs:
    views = ("T0", "T1", "T2")
    camera = _camera((2.0, 0.0, 0.5))
    return ReconstructionInputs(
        observations=[
            _field(
                view,
                2,
                residuals=residuals,
                fit_window=(2_640, 2_280, 48, 48),
            )
            for view in views
        ],
        cameras=[camera, camera, camera],
        view_names=list(views),
        bounds_hint=(torch.zeros(3), 1.0),
        name="tiny-capacity-crossover",
    )


def test_frozen_configs_are_causal_isolation_protocol() -> None:
    lift = experiment.lift_config()
    train = experiment.train_config()
    assert lift.anchor_mode == "component_centers"
    assert lift.samples_per_ray == 48
    assert lift.n_init_3d == 835
    assert lift.seed == 75_200
    assert train.device == "cpu"
    assert train.proposal_mode == "pixel_uniform"
    assert train.schedule_mode == "balanced_cycle"
    assert train.target_mode == "uniform"
    assert train.uniform_fraction == 1.0
    assert train.iterations == 140
    assert train.attempts_per_step == 128
    assert train.checkpoints == (0, 28, 70, 140)
    assert not train.evaluate_checkpoint_risks
    assert (train.point_chunk, train.gaussian_chunk) == (32, 64)
    assert (train.outer_microbatch, train.query_component_chunk) == (32, 64)
    assert train.extent == experiment.BOUNDS_EXTENT
    conditional = experiment.config_record()["protocol"]["conditional_next_arm"]
    assert not conditional["enabled_in_this_plan"]
    assert conditional["n_init_3d"] == 2_610
    assert not experiment.config_record()["protocol"]["count_only_ablation"]
    assert (
        "complete old_640 versus new_2000" in experiment.config_record()["protocol"]["intervention"]
    )
    assert json.loads(experiment.canonical_bytes(experiment.config_record())) == (
        experiment.config_record()
    )


def test_pair_validation_and_exact_bounds_injection() -> None:
    old, new = _paired_inputs()
    record = experiment.validate_paired_inputs(old, new)
    injected = experiment.inject_new_bounds(old, new)
    assert record["heldout_absent"]
    assert record["points_absent"]
    assert not record["index"]["ordering_claimed"]
    assert all(
        value <= experiment.INDEX_ENTRY_CAP
        for value in (
            *record["index"]["old_per_view_entries"],
            *record["index"]["new_per_view_entries"],
        )
    )
    assert injected.bounds_hint is not None and new.bounds_hint is not None
    assert torch.equal(injected.bounds_hint[0], new.bounds_hint[0])
    assert injected.bounds_hint[1] == new.bounds_hint[1]
    assert all(
        left is right for left, right in zip(injected.observations, old.observations, strict=True)
    )
    assert all(left is right for left, right in zip(injected.cameras, old.cameras, strict=True))
    assert injected.points is None and injected.point_visibility is None


@pytest.mark.parametrize("side", ["old", "new"])
def test_pair_rejects_sparse_points(side: str) -> None:
    old, new = _paired_inputs()
    points = torch.zeros(1, 3)
    if side == "old":
        old = _replace_inputs(old, points=points)
    else:
        new = _replace_inputs(new, points=points)
    with pytest.raises(ValueError, match="forbids sparse points"):
        experiment.validate_paired_inputs(old, new)


def test_pair_rejects_camera_and_fit_window_drift() -> None:
    old, new = _paired_inputs()
    changed_camera = _camera((3.0, 0.0, 0.5))
    new_camera = _replace_inputs(
        new,
        cameras=[changed_camera, *new.cameras[1:]],
    )
    with pytest.raises(ValueError, match="camera mismatch"):
        experiment.validate_paired_inputs(old, new_camera)

    fields = list(new.observations)
    fields[0] = _field(
        experiment.TRAIN_VIEWS[0],
        experiment.NEW_COMPONENTS_PER_VIEW,
        residuals=True,
        fit_window=(1, 0, 64, 64),
    )
    new_window = _replace_inputs(new, observations=fields)
    with pytest.raises(ValueError, match="fit-window mismatch"):
        experiment.validate_paired_inputs(old, new_window)

    fields = list(new.observations)
    fields[0] = dataclasses.replace(fields[0], epsilon=1e-7)
    new_semantics = _replace_inputs(new, observations=fields)
    with pytest.raises(ValueError, match="renderer/provider semantics mismatch"):
        experiment.validate_paired_inputs(old, new_semantics)


def test_pair_rejects_heldout_and_residual_contract_drift() -> None:
    old, new = _paired_inputs()
    names = [experiment.HELDOUT_VIEW, *experiment.TRAIN_VIEWS[1:]]
    fields = [
        _field(experiment.HELDOUT_VIEW, experiment.OLD_COMPONENTS_PER_VIEW, residuals=False),
        *old.observations[1:],
    ]
    leaked = _replace_inputs(old, observations=fields, view_names=names)
    with pytest.raises(ValueError, match="frozen seven training views"):
        experiment.validate_paired_inputs(leaked, new)

    fields = list(new.observations)
    fields[0] = _field(
        experiment.TRAIN_VIEWS[0],
        experiment.NEW_COMPONENTS_PER_VIEW,
        residuals=False,
    )
    missing_residual = _replace_inputs(new, observations=fields)
    with pytest.raises(ValueError, match="exact mean residuals"):
        experiment.validate_paired_inputs(old, missing_residual)


def test_candidate_lineage_uses_every_exact_native_mean_once() -> None:
    inputs = _tiny_inputs(residuals=True)
    config = dataclasses.replace(experiment.lift_config(), n_init_3d=2)
    arrays, record = experiment.candidate_lineage(inputs, config)
    expected_xy = torch.cat(
        [field.native_means(dtype=torch.float64) for field in inputs.observations]
    ).numpy()
    assert record["candidate_count"] == 6
    assert record["per_view"] == [2, 2, 2]
    assert record["attempt_count"] == 6
    assert record["identity_checks"]["every_component_exactly_once"]
    assert arrays["source_xy"].dtype == torch.empty((), dtype=torch.float64).numpy().dtype
    assert (arrays["source_xy"] == expected_xy).all()
    assert arrays["source_view_indices"].tolist() == [0, 0, 1, 1, 2, 2]
    assert arrays["source_component_indices"].tolist() == [0, 1, 0, 1, 0, 1]
    audit = record["candidate_complete_best_depth_audit"]
    assert audit["available"].startswith("pending")
    assert not audit["depth_profile_available"]
    assert not audit["scoring_forked_or_duplicated"]


def test_production_candidate_audit_covers_scores_depth_validity_and_selection() -> None:
    inputs = _tiny_inputs(residuals=True)
    config = dataclasses.replace(experiment.lift_config(), n_init_3d=2)
    candidate_arrays, _ = experiment.candidate_lineage(inputs, config)
    audits = []
    depth_batches = []
    result = experiment.CompactCarveInitializer(config).initialize(
        inputs,
        candidate_audit_callback=audits.append,
        ray_depth_audit_callback=depth_batches.append,
    )
    assert len(audits) == 1
    arrays = experiment._validate_candidate_audit(audits[0], candidate_arrays, result)
    assert set(arrays) == {
        "candidate_source_view_indices",
        "candidate_source_component_indices",
        "candidate_source_xy",
        "candidate_best_depths",
        "candidate_depth_sigmas",
        "candidate_best_means",
        "candidate_best_scores",
        "candidate_best_depth_indices",
        "candidate_best_coverages",
        "candidate_best_color_variances",
        "candidate_best_n_seen",
        "candidate_best_n_covered",
        "candidate_second_best_scores",
        "candidate_score_margins",
        "candidate_half_max_widths",
        "candidate_consensus_colors",
        "candidate_valid_mask",
        "candidate_eligible_mask",
        "selected_candidate_indices",
    }
    selected = arrays["selected_candidate_indices"]
    assert arrays["candidate_valid_mask"][selected].all()
    assert arrays["candidate_eligible_mask"][selected].all()
    profile = experiment._validate_ray_depth_audit(
        depth_batches,
        candidate_arrays,
        audits[0],
        config,
    )
    assert profile["scores"].shape == (6, 48)
    assert profile["consensus_colors"].shape == (6, 48, 3)
    rows = np.arange(6)
    winners = arrays["candidate_best_depth_indices"]
    np.testing.assert_array_equal(
        profile["scores"][rows, winners],
        arrays["candidate_best_scores"],
    )
    np.testing.assert_array_equal(
        profile["depths"][rows, winners],
        arrays["candidate_best_depths"],
    )


def test_scoring_backends_are_exact_ordered_indices_without_proxy() -> None:
    inputs = _tiny_inputs()
    config = dataclasses.replace(experiment.lift_config(), n_init_3d=2)
    backends, records = experiment.exact_scoring_backends(inputs, config)
    assert all(isinstance(backend, GaussianObservationIndex) for backend in backends)
    assert all(
        backend.field is field for backend, field in zip(backends, inputs.observations, strict=True)
    )
    assert [record["backend"] for record in records] == [
        "GaussianObservationIndex",
    ] * inputs.n_views
    assert not any("proxy" in record["backend"].lower() for record in records)


def test_scene_guard_blocks_multiple_open_routes_and_allows_other_files(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    raster = scene / "frame.png"
    raster.write_bytes(b"not-an-image")
    allowed = tmp_path / "allowed.txt"
    allowed.write_text("ok", encoding="utf-8")

    with experiment.deny_scene_rasters(scene) as guard:
        with pytest.raises(RuntimeError, match="forbidden before selection receipt"):
            raster.open("rb")
        with pytest.raises(RuntimeError, match="forbidden before selection receipt"):
            builtins.open(raster, "rb")  # noqa: SIM115 - the guard raises before a handle exists
        with pytest.raises(RuntimeError, match="forbidden before selection receipt"):
            os.open(raster, os.O_RDONLY)
        assert allowed.read_text(encoding="utf-8") == "ok"
    assert len(guard.attempts) == 3
    assert guard.negative_control_denials == 4
    assert not guard.record()["passed"]
    assert raster.read_bytes() == b"not-an-image"
    with experiment.deny_scene_rasters(scene) as clean_guard:
        assert allowed.read_text(encoding="utf-8") == "ok"
    assert clean_guard.record()["passed"]


def _fake_history() -> dict:
    steps = []
    for index in range(experiment.train_config().iterations):
        record = {key: f"{key}-{index}" for key in experiment.SAMPLE_HASH_KEYS}
        record["view_index"] = index % len(experiment.TRAIN_VIEWS)
        record["view_name"] = experiment.TRAIN_VIEWS[record["view_index"]]
        record["sample_seed"] = index + 1
        steps.append(record)
    return {"steps": steps}


def test_paired_sampling_requires_view_xy_density_and_importance_identity() -> None:
    old = _fake_history()
    new = _fake_history()
    record = experiment.assert_paired_sampling(old, new)
    assert record["identical"]
    assert record["steps"] == 140
    assert "xy_sha256" in record["keys"]
    assert "proposal_density_sha256" in record["keys"]
    assert "importance_sha256" in record["keys"]

    new["steps"][17]["importance_sha256"] = "tampered"
    with pytest.raises(RuntimeError, match="step 18 key importance_sha256"):
        experiment.assert_paired_sampling(old, new)


def test_tree_binding_rejects_symlinks_and_detects_byte_drift(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payload = bundle / "manifest.json"
    payload.write_text("{}", encoding="utf-8")
    before = experiment.bundle_binding(bundle)
    payload.write_text('{"changed":true}', encoding="utf-8")
    after = experiment.bundle_binding(bundle)
    assert before["aggregate_sha256"] != after["aggregate_sha256"]

    (bundle / "link").symlink_to(payload)
    with pytest.raises(ValueError, match="contains symlink"):
        experiment.bundle_binding(bundle)


def test_source_binding_covers_complete_rtgs_python_tree_and_benchmark_helpers() -> None:
    expected = {
        path.relative_to(experiment.ROOT) for path in (experiment.ROOT / "src/rtgs").rglob("*.py")
    }
    expected |= {
        Path("benchmarks/compact_bounds_crossover.py"),
        Path("benchmarks/compact_teacher_capacity_crossover.py"),
    }
    assert set(experiment.SOURCE_PATHS) == expected
    assert Path("src/rtgs/render/torch_ref.py") in experiment.SOURCE_PATHS
    assert Path("src/rtgs/cli.py") in experiment.SOURCE_PATHS
    assert Path("src/rtgs/data/scene.py") in experiment.SOURCE_PATHS


def test_exclusive_json_publication_is_no_replace_and_cleans_temporaries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    digest = experiment._write_json_exclusive(path, {"value": 1})
    assert digest == hashlib.sha256(experiment.canonical_bytes({"value": 1})).hexdigest()
    assert path.read_bytes() == experiment.canonical_bytes({"value": 1})
    with pytest.raises(FileExistsError):
        experiment._write_json_exclusive(path, {"value": 2})
    assert path.read_bytes() == experiment.canonical_bytes({"value": 1})
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_verified_selection_rehashes_executed_source_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    plan_sha = "a" * 64
    source_payload = b"bound source bytes"
    source_files = {"src/rtgs/example.py": hashlib.sha256(source_payload).hexdigest()}
    source_aggregate = experiment.canonical_hash(source_files)
    source_binding = {
        "files": source_files,
        "aggregate_sha256": source_aggregate,
    }
    monkeypatch.setattr(
        experiment,
        "_verify_frozen_plan",
        lambda _: (
            {
                "source_binding": source_binding,
                "cpu_phase_runtime": experiment._cpu_phase_runtime_record(),
            },
            plan_sha,
        ),
    )
    arms = {}
    for arm in ("old_640", "new_2000"):
        artifacts = {}
        for key, digest_key in (
            ("candidate_audit", "candidate_audit_sha256"),
            ("ray_depth_profiles", "ray_depth_profiles_sha256"),
            ("selected_lineage", "selected_lineage_sha256"),
            ("gaussians_init", "gaussians_init_sha256"),
            ("history", "history_sha256"),
            ("gaussians", "gaussians_sha256"),
        ):
            path = out / "arms" / arm / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{arm}-{key}".encode())
            artifacts[key] = str(path)
            artifacts[digest_key] = experiment.sha256_file(path)
        arms[arm] = {"artifacts": artifacts}
    archive = out / "artifacts" / "EXECUTED_SOURCES.tar"
    archive.parent.mkdir()
    source = tmp_path / "example.py"
    source.write_bytes(source_payload)
    with tarfile.open(archive, "w") as stream:
        stream.add(source, arcname="src/rtgs/example.py")
    verification = experiment._verify_executed_sources_archive(archive, source_binding)
    selection = {
        "artifact_type": f"{experiment.SCHEMA}_selection_receipt_v1",
        "status": "PASS",
        "plan_sha256": plan_sha,
        "cpu_phase_runtime": experiment._cpu_phase_runtime_record(),
        "source_denial_before_receipt": {"passed": True},
        "arms": arms,
        "executed_sources": {
            "path": str(archive),
            "sha256": experiment.sha256_file(archive),
            "source_aggregate_sha256": source_aggregate,
            "verification": verification,
        },
    }
    selection["selection_payload_sha256"] = experiment.canonical_hash(selection)
    experiment._write_json_exclusive(out / "selection_receipt.json", selection)
    experiment._verified_selection(out)
    archive.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="executed-source archive changed"):
        experiment._verified_selection(out)


def test_executed_source_archive_rejects_coherent_wrong_member_bytes(tmp_path: Path) -> None:
    payload = b"expected"
    binding = {
        "files": {"src/rtgs/example.py": hashlib.sha256(payload).hexdigest()},
        "aggregate_sha256": "a" * 64,
    }
    source = tmp_path / "example.py"
    source.write_bytes(b"wrong")
    archive = tmp_path / "sources.tar"
    with tarfile.open(archive, "w") as stream:
        stream.add(source, arcname="src/rtgs/example.py")
    with pytest.raises(ValueError, match="bytes differ"):
        experiment._verify_executed_sources_archive(archive, binding)


def test_viewer_command_compares_old_final_to_new_final_at_native_settings(tmp_path: Path) -> None:
    command = experiment.viewer_command(tmp_path, port=8_899)
    assert command[:3] == [
        "env",
        f"LD_PRELOAD={experiment.ABI_PRELOAD}",
        ".venv/bin/python",
    ]
    assert command[3:6] == [
        "-m",
        "benchmarks.compact_teacher_capacity_crossover",
        "serve-viewer",
    ]
    assert command[command.index("--out") + 1] == str(tmp_path.resolve())
    assert command[command.index("--port") + 1] == "8899"
    binding = experiment._viewer_binding_digest(tmp_path)
    assert len(binding) == 64
    token = "a" * 64
    marked = experiment.viewer_command(tmp_path, port=8_899, handshake_token=token)
    assert marked[marked.index("--handshake-token") + 1] == token
    marker = experiment._viewer_handshake_marker(
        tmp_path,
        token=token,
        pid=123,
        port=8_899,
    )
    assert marker.startswith(experiment.VIEWER_MARKER_PREFIX)
    assert f"binding={binding}" in marker
    assert "pid=123 port=8899" in marker


def test_viewer_handshake_rejects_occupied_port_and_binds_listener_pid() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = int(listener.getsockname()[1])
        with pytest.raises(RuntimeError, match="already occupied"):
            experiment._assert_port_available(port)
        assert experiment._process_owns_listening_port(os.getpid(), port)
        assert not experiment._process_owns_listening_port(os.getpid() + 1_000_000, port)


def test_native_evaluation_requires_exact_versioned_abi(monkeypatch) -> None:
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    with pytest.raises(RuntimeError, match="first and sole"):
        experiment._assert_evaluation_abi()
    monkeypatch.setenv("LD_PRELOAD", f"/tmp/other.so:{experiment.ABI_PRELOAD}")
    with pytest.raises(RuntimeError, match="first and sole"):
        experiment._assert_evaluation_abi()
    monkeypatch.setenv("LD_PRELOAD", f"{experiment.ABI_PRELOAD}:/tmp/other.so")
    with pytest.raises(RuntimeError, match="first and sole"):
        experiment._assert_evaluation_abi()
    monkeypatch.setenv("LD_PRELOAD", str(experiment.ABI_PRELOAD))
    record = experiment._assert_evaluation_abi()
    assert record["required_path"] == str(experiment.ABI_PRELOAD)
    assert record["first_and_sole_entry"]
    assert len(record["required_sha256"]) == 64


def test_cpu_plan_and_selection_require_empty_preload(monkeypatch) -> None:
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    assert experiment._assert_cpu_phase_runtime() == {
        "effective_ld_preload": "",
        "policy": "plan_and_selection_require_no_preloaded_shared_libraries",
    }
    monkeypatch.setenv("LD_PRELOAD", str(experiment.ABI_PRELOAD))
    with pytest.raises(RuntimeError, match="requires an empty LD_PRELOAD"):
        experiment._assert_cpu_phase_runtime()


def test_evaluation_acquisition_rejects_any_input_or_source_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    calibration_path = experiment.display_path(experiment.CALIBRATION)
    files = {calibration_path: "a" * 64}
    for view_id in experiment.EVALUATION_VIEWS:
        files[experiment.display_path(experiment.SCENE / "rgb" / f"{view_id}.jpg")] = "b" * 64
        files[experiment.display_path(experiment.SCENE / "mask" / f"mask_{view_id}.png")] = "c" * 64
    source = {"aggregate_sha256": "c" * 64}
    monkeypatch.setattr(experiment, "_evaluation_input_files", lambda: dict(files))
    monkeypatch.setattr(experiment, "source_binding", lambda: dict(source))
    calibration = {"path": calibration_path, "sha256": "a" * 64}
    views = [
        {
            "view_id": view_id,
            "split": split,
            "calibration": calibration,
            "distortion_coefficients": [],
            "rgb": {
                "path": experiment.display_path(experiment.SCENE / "rgb" / f"{view_id}.jpg"),
                "sha256": "b" * 64,
            },
            "mask": {
                "path": experiment.display_path(experiment.SCENE / "mask" / f"mask_{view_id}.png"),
                "sha256": "c" * 64,
            },
            "camera": {"view_id": view_id},
            "decoded_rgb_tensor_sha256": f"{index:064x}",
            "decoded_mask_tensor_sha256": f"{index + 8:064x}",
        }
        for index, (view_id, split) in enumerate(
            zip(experiment.EVALUATION_VIEWS, experiment.EVALUATION_SPLITS, strict=True)
        )
    ]
    receipt = {
        "artifact_type": f"{experiment.SCHEMA}_evaluation_acquisition_v1",
        "status": "PASS",
        "selection_receipt_file_sha256": "d" * 64,
        "view_order": list(experiment.EVALUATION_VIEWS),
        "splits": list(experiment.EVALUATION_SPLITS),
        "calibration": calibration,
        "input_files": dict(files),
        "views": views,
        "source_aggregate_sha256": source["aggregate_sha256"],
        "runtime_abi": {"effective_ld_preload": str(experiment.ABI_PRELOAD)},
        "decoded_once_per_view_in_this_phase": True,
    }
    receipt["acquisition_payload_sha256"] = experiment.canonical_hash(receipt)
    experiment._write_json_exclusive(out / "evaluation_acquisition.json", receipt)
    verified, _ = experiment._verify_evaluation_acquisition(
        out,
        selection_sha="d" * 64,
        runtime_abi=receipt["runtime_abi"],
    )
    assert verified == receipt

    first_rgb = experiment.display_path(
        experiment.SCENE / "rgb" / f"{experiment.EVALUATION_VIEWS[0]}.jpg"
    )
    files[first_rgb] = "e" * 64
    with pytest.raises(RuntimeError, match="changed after acquisition"):
        experiment._verify_evaluation_acquisition(out, selection_sha="d" * 64)
    files[first_rgb] = "b" * 64
    source["aggregate_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="source changed"):
        experiment._verify_evaluation_acquisition(out, selection_sha="d" * 64)


def _fake_native_evaluation(
    out: Path,
) -> tuple[dict, dict]:
    acquisition_views = []
    records = []
    for view_index, (view_id, split) in enumerate(
        zip(experiment.EVALUATION_VIEWS, experiment.EVALUATION_SPLITS, strict=True)
    ):
        target = {
            "view_id": view_id,
            "split": split,
            "binding": f"target-{view_index}",
        }
        acquisition_views.append(target)
        models = {}
        for model_index, model_name in enumerate(experiment.MODEL_NAMES):
            path = out / "native" / model_name / f"{view_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{view_id}-{model_name}".encode()
            path.write_bytes(payload)
            all_sse = float(view_index + model_index + 1)
            foreground_sse = float(view_index + model_index + 2)
            metrics = {}
            for stratum, sse, count in (
                ("all", all_sse, 6),
                ("foreground", foreground_sse, 3),
            ):
                mse = sse / count
                metrics[stratum] = {
                    "sse": sse,
                    "scalar_count": count,
                    "mse": mse,
                    "psnr_db": -10.0 * math.log10(max(mse, 1e-30)),
                }
            models[model_name] = {
                "n_gaussians": experiment.N_INIT_3D,
                "backend": "rtgs.render.gsplat_backend.GsplatRasterizer",
                "device": "cuda:0",
                "packed": False,
                "antialiased": False,
                "render": {
                    "path": str(path),
                    "sha256": experiment.sha256_file(path),
                    "bytes": len(payload),
                    "dimensions": list(experiment.NATIVE_SIZE),
                    "color_tensor_sha256": f"{view_index + model_index:064x}",
                },
                "metrics": metrics,
            }
        records.append(
            {
                "view_id": view_id,
                "split": split,
                "target": target,
                "models": models,
            }
        )
    contact_path = out / "CONTACT_SHEET.png"
    contact_path.write_bytes(b"contact")
    evaluation = {
        "records": records,
        "contact_sheet": {
            "path": str(contact_path),
            "sha256": experiment.sha256_file(contact_path),
            "rows": ["target", *experiment.MODEL_NAMES],
            "columns": list(experiment.EVALUATION_VIEWS),
        },
    }
    evaluation["pooled"] = experiment._recompute_pooled(records)
    evaluation["deltas"] = experiment._metric_deltas(evaluation)
    return evaluation, {"views": acquisition_views}


def test_native_artifact_verifier_recomputes_metrics_and_rehashes_every_output(
    tmp_path: Path,
) -> None:
    evaluation, acquisition = _fake_native_evaluation(tmp_path)
    experiment._verify_native_artifacts(tmp_path, evaluation, acquisition)
    render = evaluation["records"][3]["models"]["new_final"]["render"]
    Path(render["path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="render artifact changed"):
        experiment._verify_native_artifacts(tmp_path, evaluation, acquisition)


def test_native_artifact_verifier_accepts_canonical_json_key_order(tmp_path: Path) -> None:
    evaluation, acquisition = _fake_native_evaluation(tmp_path)
    round_tripped = json.loads(experiment.canonical_bytes(evaluation))
    assert tuple(round_tripped["records"][0]["models"]) != experiment.MODEL_NAMES
    experiment._verify_native_artifacts(tmp_path, round_tripped, acquisition)


def test_failed_viewer_smoke_is_retry_safe(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "run"
    out.mkdir()
    selection_sha = "a" * 64
    evaluation_sha = "b" * 64
    monkeypatch.setattr(
        experiment,
        "_verified_selection",
        lambda _: ({"arms": {}}, selection_sha),
    )
    monkeypatch.setattr(
        experiment,
        "_verified_evaluation",
        lambda *_: ({}, evaluation_sha),
    )
    monkeypatch.setattr(experiment, "viewer_command", lambda *_args, **_kwargs: ["viewer"])
    failed_smoke = {
        "status": "FAIL",
        "command": ["viewer"],
        "process_handshake": {
            "marker": "failed-marker",
            "marker_observed": False,
            "listener_owner_verified": False,
        },
    }
    monkeypatch.setattr(
        experiment,
        "_smoke_explicit_viewer",
        lambda *_args, **_kwargs: failed_smoke,
    )
    with pytest.raises(RuntimeError, match="viewer smoke did not pass"):
        experiment.run_view_smoke(out, port=8_899)
    assert not (out / "viewer_receipt.json").exists()

    passed_smoke = {
        "status": "PASS",
        "command": ["viewer"],
        "process_handshake": {
            "marker": "passed-marker",
            "marker_observed": True,
            "listener_owner_verified": True,
        },
    }
    monkeypatch.setattr(
        experiment,
        "_smoke_explicit_viewer",
        lambda *_args, **_kwargs: passed_smoke,
    )
    receipt = experiment.run_view_smoke(out, port=8_899)
    assert receipt["status"] == "PASS"
    assert (out / "viewer_receipt.json").is_file()


def test_tiny_select_arm_persists_complete_production_depth_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _tiny_inputs(residuals=True)
    bundle = tmp_path / "bundle"
    source.save(bundle)
    inputs = ReconstructionInputs.load(bundle, device="cpu", strict=True)
    lift = dataclasses.replace(experiment.lift_config(), n_init_3d=2)
    train = dataclasses.replace(
        experiment.train_config(),
        iterations=1,
        attempts_per_step=4,
        point_chunk=2,
        gaussian_chunk=2,
        outer_microbatch=2,
        evaluation_chunk=4,
        checkpoints=(0, 1),
    )
    monkeypatch.setattr(experiment, "N_INIT_3D", 2)
    monkeypatch.setattr(experiment, "OLD_CANDIDATES", 6)
    monkeypatch.setattr(experiment, "lift_config", lambda: lift)
    monkeypatch.setattr(experiment, "train_config", lambda: train)
    out = tmp_path / "out"
    out.mkdir()
    record, history = experiment._select_arm(
        out=out,
        arm="old_640",
        inputs=inputs,
        bundle_path=bundle,
    )
    assert record["n_init_3d"] == record["n_opt_3d"] == 2
    assert history["n_init_3d"] == history["n_opt_3d"] == 2
    profile_path = experiment.bound_path(record["artifacts"]["ray_depth_profiles"])
    with np.load(profile_path, allow_pickle=False) as profile:
        assert profile["scores"].shape == (6, 48)
        assert profile["consensus_colors"].shape == (6, 48, 3)
        assert profile["source_view_indices"].tolist() == [0, 0, 1, 1, 2, 2]
    assert record["candidate_contract"]["complete_ray_depth_profile"]["winner_gathers_exact"]


def test_evaluation_is_impossible_without_committed_selection_receipt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        experiment._verified_selection(tmp_path)


def test_cli_exposes_only_frozen_nonmonolithic_phases() -> None:
    for phase in ("plan", "select", "evaluate", "view-smoke"):
        assert experiment.parse_args([phase]).phase == phase
    serve = experiment.parse_args(["serve-viewer", "--out", "/tmp/run", "--port", "8899"])
    assert serve.phase == "serve-viewer"
    with pytest.raises(SystemExit):
        experiment.parse_args(["run"])
