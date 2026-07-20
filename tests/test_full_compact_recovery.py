"""CPU-only contracts for the full compact reconstruction recovery harness."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rtgs.data.synthetic import make_synthetic_scene


@pytest.fixture(scope="module")
def recovery_harness():
    path = Path(__file__).resolve().parents[1] / "benchmarks/full_compact_reconstruction.py"
    spec = importlib.util.spec_from_file_location("rtgs_full_compact_recovery_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _target_record(*, crop_sha256: str = "a" * 64) -> dict:
    return {
        "global_index": 3,
        "view_id": "C0006",
        "fit_window": [10, 20, 30, 40],
        "renderer": "additive",
        "blend_mode": "additive",
        "components": 5000,
        "has_alpha": True,
        "alpha_applied": True,
        "crop_sha256": crop_sha256,
        "elapsed_seconds": 99.0,
        "unclamped_min": -0.1,
        "unclamped_max": 1.1,
    }


def _write_target_receipt(
    path: Path,
    record: dict,
    *,
    deterministic_origin: bool = False,
) -> None:
    payload = {
        "schema": (
            "rtgs.full_compact_reconstruction.targets.v2"
            if deterministic_origin
            else "rtgs.full_compact_reconstruction.targets.v1"
        ),
        "views": [record],
    }
    if deterministic_origin:
        payload["deterministic_algorithms"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_legacy_target_replay_records_hash_mismatch_without_claiming_equivalence(
    recovery_harness,
    tmp_path,
):
    reference = tmp_path / "compact_targets.json"
    frozen = _target_record(crop_sha256="a" * 64)
    _write_target_receipt(reference, frozen)
    replayed = _target_record(crop_sha256="b" * 64)
    replayed["elapsed_seconds"] = 0.001
    replayed["unclamped_min"] += 5e-7
    independent = copy.deepcopy(replayed)

    receipt = recovery_harness._verify_target_replay(
        reference,
        [replayed],
        [independent],
    )

    assert receipt["view_count"] == 1
    assert receipt["all_metadata_match"] is True
    assert receipt["frozen_raw_crop_hash_match_count"] == 0
    assert receipt["all_frozen_raw_crop_hashes_match"] is False
    assert receipt["original_tensor_equivalence_verified"] is False
    assert receipt["deterministic_recovery_replay_verified"] is True
    assert receipt["max_frozen_extrema_delta"] == pytest.approx(5e-7)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("global_index", 4),
        ("view_id", "C0007"),
        ("fit_window", [11, 20, 30, 40]),
        ("renderer", "normalized"),
        ("blend_mode", "normalized"),
        ("components", 4999),
        ("has_alpha", False),
        ("alpha_applied", False),
    ],
)
def test_target_replay_rejects_each_metadata_change(
    recovery_harness,
    tmp_path,
    key,
    value,
):
    reference = tmp_path / "compact_targets.json"
    frozen = _target_record()
    _write_target_receipt(reference, frozen)
    replayed = copy.deepcopy(frozen)
    replayed[key] = value

    with pytest.raises(RuntimeError, match="metadata differs from the frozen fit"):
        recovery_harness._verify_target_replay(
            reference,
            [replayed],
            [copy.deepcopy(replayed)],
        )


def test_target_replay_rejects_independent_deterministic_hash_change(
    recovery_harness,
    tmp_path,
):
    reference = tmp_path / "compact_targets.json"
    frozen = _target_record()
    _write_target_receipt(reference, frozen)
    replayed = _target_record(crop_sha256="b" * 64)
    independent = copy.deepcopy(replayed)
    independent["crop_sha256"] = "c" * 64

    with pytest.raises(RuntimeError, match="differs bitwise"):
        recovery_harness._verify_target_replay(reference, [replayed], [independent])


def test_target_replay_rejects_frozen_extrema_outside_tolerance(
    recovery_harness,
    tmp_path,
):
    reference = tmp_path / "compact_targets.json"
    frozen = _target_record()
    _write_target_receipt(reference, frozen)
    replayed = copy.deepcopy(frozen)
    replayed["unclamped_max"] += 2e-6

    with pytest.raises(RuntimeError, match="unclamped_max differs"):
        recovery_harness._verify_target_replay(
            reference,
            [replayed],
            [copy.deepcopy(replayed)],
        )


def test_deterministic_origin_requires_exact_frozen_crop_hash(
    recovery_harness,
    tmp_path,
):
    reference = tmp_path / "compact_targets.json"
    frozen = _target_record()
    _write_target_receipt(reference, frozen, deterministic_origin=True)

    receipt = recovery_harness._verify_target_replay(
        reference,
        [copy.deepcopy(frozen)],
        [copy.deepcopy(frozen)],
    )
    assert receipt["original_tensor_equivalence_verified"] is True

    replayed = _target_record(crop_sha256="b" * 64)
    with pytest.raises(RuntimeError, match="deterministic-origin"):
        recovery_harness._verify_target_replay(
            reference,
            [replayed],
            [copy.deepcopy(replayed)],
        )


def test_deterministic_target_receipt_requires_marker(recovery_harness, tmp_path):
    reference = tmp_path / "compact_targets.json"
    record = _target_record()
    reference.write_text(
        json.dumps(
            {
                "schema": "rtgs.full_compact_reconstruction.targets.v2",
                "views": [record],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lacks its determinism marker"):
        recovery_harness._verify_target_replay(
            reference,
            [copy.deepcopy(record)],
            [copy.deepcopy(record)],
        )


def _provenance_record() -> dict:
    camera = {
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        "t": [0.0, 0.0, 0.0],
        "fx": 10.0,
        "fy": 10.0,
        "cx": 5.0,
        "cy": 5.0,
        "width": 10,
        "height": 10,
    }
    views = []
    for index, view_id in enumerate(("C0001", "C0002")):
        views.append(
            {
                "index": index,
                "view_id": view_id,
                "bundle": f"/data/{view_id}.rtgsv",
                "bundle_sha256": str(index + 1) * 64,
                "bundle_bytes": 100 + index,
                "n_components": 5,
                "fit_window": [0, 0, 10, 10],
                "has_alpha": True,
                "source": {
                    "rgb": {"name": f"{view_id}.jpg", "sha256": "a" * 64},
                    "mask": {"name": f"mask_{view_id}.png", "sha256": "b" * 64},
                },
                "camera": copy.deepcopy(camera),
            }
        )
    return {
        "schema": "rtgs.full_compact_reconstruction.provenance.v1",
        "written_before_source_rgb_access": True,
        "created_utc": "ignored",
        "manifest": "/data/manifest.json",
        "manifest_sha256": "c" * 64,
        "calibration_sha256": "d" * 64,
        "bounds_hint": [[0.0, 0.0, 0.0], 1.0],
        "fit_indices": [0, 1],
        "expected_component_center_candidates": 10,
        "views": views,
        "environment": {"torch": "test", "cuda_available": False},
    }


def test_provenance_gate_is_exact_for_manifest_bundle_camera_source_and_order(
    recovery_harness,
):
    frozen = _provenance_record()
    current = copy.deepcopy(frozen)
    current["created_utc"] = "different-and-ignored"
    receipt = recovery_harness._verify_provenance_identity(frozen, current)
    assert receipt["all_fields_match"] is True

    mutations = []
    changed = copy.deepcopy(current)
    changed["manifest_sha256"] = "e" * 64
    mutations.append(changed)
    changed = copy.deepcopy(current)
    changed["views"][0]["bundle_sha256"] = "f" * 64
    mutations.append(changed)
    changed = copy.deepcopy(current)
    changed["views"][0]["camera"]["fx"] = 11.0
    mutations.append(changed)
    changed = copy.deepcopy(current)
    changed["views"][0]["source"]["rgb"]["sha256"] = "0" * 64
    mutations.append(changed)
    changed = copy.deepcopy(current)
    changed["views"].reverse()
    mutations.append(changed)

    for changed in mutations:
        with pytest.raises(RuntimeError, match="current compact provenance differs"):
            recovery_harness._verify_provenance_identity(frozen, changed)


def test_independent_target_replay_checks_exact_alpha_without_second_scene(
    recovery_harness,
    monkeypatch,
):
    image = torch.tensor([[[0.25, 0.5, 0.75]]], dtype=torch.float32)
    alpha = torch.ones((1, 1), dtype=torch.bool)
    crop_sha256 = recovery_harness._tensor_sha256(image)
    receipt = _target_record(crop_sha256=crop_sha256)
    receipt["view_id"] = "C0006"
    receipt["fit_window"] = [0, 0, 1, 1]
    receipt.pop("global_index")
    view = SimpleNamespace(view_id="C0006")
    dataset = SimpleNamespace(views=[view])
    scene = SimpleNamespace(images=[image], masks=[alpha])
    config = {
        "fit_indices": [0],
        "validation_indices": [],
        "device": "cpu",
        "structsplat_renderer": "reference",
        "structsplat_chunk": 1,
    }
    replayed = [{"global_index": 0, **receipt}]

    def render_exact(*args, **kwargs):
        return image.clone(), alpha.clone(), copy.deepcopy(receipt)

    monkeypatch.setattr(recovery_harness, "_render_compact_crop", render_exact)
    records, alpha_receipt = recovery_harness._independently_replay_training_targets(
        dataset,
        config,
        scene,
        replayed,
    )
    assert records[0]["crop_sha256"] == crop_sha256
    assert alpha_receipt["all_alpha_masks_match"] is True

    def render_bad_alpha(*args, **kwargs):
        return image.clone(), torch.zeros_like(alpha), copy.deepcopy(receipt)

    monkeypatch.setattr(recovery_harness, "_render_compact_crop", render_bad_alpha)
    with pytest.raises(RuntimeError, match="alpha differs"):
        recovery_harness._independently_replay_training_targets(
            dataset,
            config,
            scene,
            replayed,
        )


def test_deterministic_algorithm_context_restores_prior_setting(recovery_harness):
    prior_enabled = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with recovery_harness._deterministic_algorithms():
        assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.are_deterministic_algorithms_enabled() is prior_enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled() is prior_warn_only


def test_resume_checkpoint_requires_containment_and_exact_name(recovery_harness, tmp_path):
    out = tmp_path / "run"
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "gaussians_step_004000.ply"
    checkpoint.write_bytes(b"ply")

    resolved, step = recovery_harness._resume_checkpoint(out, checkpoint)

    assert resolved == checkpoint.resolve()
    assert step == 4000
    bad_name = checkpoints / "step_004000.ply"
    bad_name.write_bytes(b"ply")
    with pytest.raises(RuntimeError, match="gaussians_step"):
        recovery_harness._resume_checkpoint(out, bad_name)
    outside = tmp_path / "gaussians_step_004000.ply"
    outside.write_bytes(b"ply")
    with pytest.raises(RuntimeError, match="inside"):
        recovery_harness._resume_checkpoint(out, outside)


def test_recovery_receipts_are_numbered_without_overwrite(recovery_harness, tmp_path):
    first = recovery_harness._next_recovery_receipt(tmp_path)
    assert first.name == "recovery_attempt_001.json"
    recovery_harness._write_json_new(first, {"attempt": 1})

    second = recovery_harness._next_recovery_receipt(tmp_path)

    assert second.name == "recovery_attempt_002.json"
    assert json.loads(first.read_text(encoding="utf-8")) == {"attempt": 1}


def _write_polish_parent(recovery_harness, parent: Path, dataset: Path) -> dict[str, str]:
    parent.mkdir()
    manifest = dataset / "manifest.json"
    dataset.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    names = list(recovery_harness.EXPECTED_VIEW_NAMES)
    config = {
        "schema": "rtgs.full_compact_reconstruction.config.v1",
        "dataset": str(dataset),
        "fit_mode": "all",
        "fit_indices": list(range(len(names))),
        "validation_indices": list(recovery_harness.V_INDICES),
        "view_names": names,
        "iterations": 30_000,
        "eval_every": 1000,
        "device": "cpu",
        "densify": True,
        "density_strategy": "classic",
        "max_gaussians": 100,
        "seed": 0,
        "smoke": False,
        "structsplat_renderer": "reference",
        "structsplat_chunk": 1,
        "validation_interpretation": "V is fitted-view validation, not held-out evidence.",
    }
    trainer = recovery_harness.TrainConfig(
        iterations=30_000,
        device="cpu",
        densify=True,
        density_strategy="classic",
        density=recovery_harness.DensityConfig(max_gaussians=100),
        eval_every=1000,
        target_sh_degree=3,
        sh_degree_interval=1000,
        seed=0,
    )
    recovery_trainer = recovery_harness.dataclasses.replace(
        trainer,
        iterations=26_000,
        iteration_offset=4_000,
        schedule_iterations=30_000,
    )
    provenance = {
        "schema": "rtgs.full_compact_reconstruction.provenance.v1",
        "written_before_source_rgb_access": True,
        "manifest": str(manifest),
        "manifest_sha256": recovery_harness._sha256_file(manifest),
        "calibration_sha256": "d" * 64,
        "bounds_hint": [[0.0, 0.0, 0.0], 1.0],
        "fit_indices": list(range(len(names))),
        "expected_component_center_candidates": 130_000,
        "views": [{"view_id": name} for name in names],
        "environment": {"test": True},
    }
    targets = {
        "schema": recovery_harness.LEGACY_NONDETERMINISTIC_TARGET_SCHEMA,
        "views": [{"global_index": index, "view_id": name} for index, name in enumerate(names)],
    }
    initial_payloads = {
        "config.json": config,
        "provenance.json": provenance,
        "training_config.json": recovery_harness.dataclasses.asdict(trainer),
        "compact_targets.json": targets,
        "compact_metrics.json": {},
    }
    for name, payload in initial_payloads.items():
        (parent / name).write_text(json.dumps(payload), encoding="utf-8")
    (parent / "gaussians_final.ply").write_bytes(b"test-parent-ply")

    recovery_dir = parent / "recovery"
    recovery_dir.mkdir()
    recovery_path = recovery_dir / "recovery_attempt_001.json"
    frozen_names = (
        "config.json",
        "provenance.json",
        "training_config.json",
        "compact_targets.json",
    )
    deterministic_identity = "e" * 64
    alpha_identity = "f" * 64
    recovery = {
        "schema": "rtgs.full_compact_reconstruction.recovery_attempt.v2",
        "written_before_recovery_training": True,
        "resume_exact": False,
        "resume_step": 4_000,
        "first_recovered_step": 4_001,
        "last_recovered_step": 30_000,
        "schedule_iterations": 30_000,
        "remaining_iterations": 26_000,
        "frozen_artifacts": {
            name: {"sha256": recovery_harness._sha256_file(parent / name)} for name in frozen_names
        },
        "target_replay": {
            "reference_sha256": recovery_harness._sha256_file(parent / "compact_targets.json"),
            "deterministic_recovery_replay_verified": True,
            "deterministic_recovery_identity_sha256": deterministic_identity,
            "alpha_replay": {"identity_sha256": alpha_identity},
        },
        "recovery_training_config": recovery_harness.dataclasses.asdict(recovery_trainer),
    }
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    history = {
        "loss": [0.1] * 26_000,
        "psnr": [[30_000, 20.0]],
        "elapsed": [[30_000, 1.0]],
        "n_gaussians": [[30_000, 7]],
        "active_sh_degree": [[30_000, 3]],
        "iteration_offset": 4_000,
        "segment_iterations": 26_000,
        "schedule_iterations": 30_000,
    }
    fit_receipt = {
        "schema": "rtgs.full_compact_reconstruction.fit_complete.v1",
        "final_ply_sha256": recovery_harness._sha256_file(parent / "gaussians_final.ply"),
        "n_final_gaussians": 7,
        "resume_exact": False,
        "recovery": {
            "receipt": "recovery/recovery_attempt_001.json",
            "receipt_sha256": recovery_harness._sha256_file(recovery_path),
        },
    }
    (parent / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
    (parent / "fit_complete.json").write_text(json.dumps(fit_receipt), encoding="utf-8")
    paths = [parent / name for name in recovery_harness.POLISH_PARENT_ARTIFACTS]
    paths.append(recovery_path)
    return {str(path.relative_to(parent)): recovery_harness._sha256_file(path) for path in paths}


def _write_tail_parent(recovery_harness, parent: Path, dataset: Path) -> dict[str, str]:
    parent.mkdir()
    dataset.mkdir()
    manifest = dataset / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    names = list(recovery_harness.EXPECTED_VIEW_NAMES)
    config = {
        "schema": "rtgs.full_compact_reconstruction.config.v1",
        "dataset": str(dataset),
        "fit_mode": "all",
        "fit_indices": list(range(len(names))),
        "validation_indices": list(recovery_harness.V_INDICES),
        "view_names": names,
        "iterations": 30_000,
        "eval_every": 1000,
        "device": "cpu",
        "densify": True,
        "density_strategy": "classic",
        "max_gaussians": 100,
        "seed": 0,
        "smoke": False,
        "structsplat_renderer": "reference",
        "structsplat_chunk": 1,
        "validation_interpretation": "V is fitted-view validation, not held-out evidence.",
    }
    base_trainer = recovery_harness.TrainConfig(
        iterations=26_000,
        iteration_offset=4_000,
        schedule_iterations=30_000,
        device="cpu",
        densify=True,
        density_strategy="classic",
        density=recovery_harness.DensityConfig(max_gaussians=100),
        eval_every=1000,
        target_sh_degree=3,
        sh_degree_interval=1000,
        seed=0,
    )
    trainer = recovery_harness._polish_train_config(base_trainer)
    provenance = {
        "schema": "rtgs.full_compact_reconstruction.provenance.v1",
        "written_before_source_rgb_access": True,
        "manifest": str(manifest),
        "manifest_sha256": recovery_harness._sha256_file(manifest),
        "calibration_sha256": "d" * 64,
        "bounds_hint": [[0.0, 0.0, 0.0], 1.0],
        "fit_indices": list(range(len(names))),
        "expected_component_center_candidates": 130_000,
        "views": [{"view_id": name} for name in names],
        "environment": {"test": True},
    }
    targets = {
        "schema": recovery_harness.DETERMINISTIC_TARGET_SCHEMA,
        "deterministic_algorithms": True,
        "views": [{"global_index": index, "view_id": name} for index, name in enumerate(names)],
    }
    for name, payload in {
        "config.json": config,
        "provenance.json": provenance,
        "training_config.json": recovery_harness.dataclasses.asdict(trainer),
        "compact_targets.json": targets,
        "compact_metrics.json": {},
    }.items():
        (parent / name).write_text(json.dumps(payload), encoding="utf-8")
    (parent / "gaussians_init.ply").write_bytes(b"30k-parent-ply")
    checkpoints = parent / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "gaussians_step_040000.ply"
    checkpoint.write_bytes(b"selected-40k-ply")
    (parent / "gaussians_final.ply").write_bytes(checkpoint.read_bytes())
    deterministic_identity = "e" * 64
    alpha_identity = "f" * 64
    polish_start = {
        "schema": "rtgs.full_compact_reconstruction.polish_start.v1",
        "written_before_polish_training": True,
        "continuation_exact": False,
        "global_steps": {
            "parent_last": 30_000,
            "first_polish": 30_001,
            "last_polish": 40_000,
            "segment_iterations": 10_000,
            "schedule_iterations": 40_000,
        },
        "child_target_receipt": {
            "sha256": recovery_harness._sha256_file(parent / "compact_targets.json")
        },
        "target_replay": {
            "deterministic_recovery_identity_sha256": deterministic_identity,
            "alpha_replay": {"identity_sha256": alpha_identity},
        },
    }
    (parent / "polish_start.json").write_text(json.dumps(polish_start), encoding="utf-8")
    final_sha = recovery_harness._sha256_file(parent / "gaussians_final.ply")
    selection = {
        "schema": recovery_harness.MODEL_SELECTION_SCHEMA,
        "written_before_gaussians_final": True,
        "selected": {
            "global_step": 40_000,
            "artifact": "checkpoints/gaussians_step_040000.ply",
            "sha256": final_sha,
        },
        "convergence": {"joint_status": "still_improving"},
    }
    (parent / "model_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    selection_record = {
        "receipt": "model_selection.json",
        "receipt_sha256": recovery_harness._sha256_file(parent / "model_selection.json"),
        "selected_global_step": 40_000,
        "selected_candidate_sha256": final_sha,
        "joint_convergence_status": "still_improving",
    }
    history = {
        "loss": [0.1] * 10_000,
        "psnr": [[40_000, 20.0]],
        "elapsed": [[40_000, 1.0]],
        "n_gaussians": [[40_000, 7]],
        "active_sh_degree": [[40_000, 3]],
        "iteration_offset": 30_000,
        "segment_iterations": 10_000,
        "schedule_iterations": 40_000,
        "continuation_exact": False,
        "fixed_topology": True,
        "model_selection": selection_record,
    }
    fit = {
        "schema": "rtgs.full_compact_reconstruction.fit_complete.v1",
        "fit_kind": "fixed_topology_polish",
        "continuation_exact": False,
        "fixed_topology": True,
        "source_n_gaussians": 7,
        "n_final_gaussians": 7,
        "final_ply_sha256": final_sha,
        "polish": {
            "receipt": "polish_start.json",
            "receipt_sha256": recovery_harness._sha256_file(parent / "polish_start.json"),
        },
        "model_selection": selection_record,
    }
    (parent / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
    (parent / "fit_complete.json").write_text(json.dumps(fit), encoding="utf-8")
    paths = [parent / name for name in recovery_harness.TAIL_PARENT_ARTIFACTS]
    paths.append(checkpoint)
    return {str(path.relative_to(parent)): recovery_harness._sha256_file(path) for path in paths}


def _write_cooldown_parent(recovery_harness, parent: Path, dataset: Path) -> dict[str, str]:
    _write_tail_parent(recovery_harness, parent, dataset)
    prior_trainer = recovery_harness._load_train_config(parent / "training_config.json")
    trainer = recovery_harness._tail_train_config(prior_trainer)
    (parent / "training_config.json").write_text(
        json.dumps(recovery_harness.dataclasses.asdict(trainer)),
        encoding="utf-8",
    )
    checkpoint = parent / "checkpoints" / "gaussians_step_050000.ply"
    checkpoint.write_bytes(b"selected-50k-ply")
    (parent / "gaussians_final.ply").write_bytes(checkpoint.read_bytes())
    deterministic_identity = "e" * 64
    alpha_identity = "f" * 64
    tail_start = {
        "schema": "rtgs.full_compact_reconstruction.tail_start.v1",
        "written_before_tail_training": True,
        "continuation_exact": False,
        "global_steps": {
            "parent_last": 40_000,
            "first_tail": 40_001,
            "last_tail": 50_000,
            "segment_iterations": 10_000,
            "schedule_iterations": 50_000,
        },
        "child_target_receipt": {
            "sha256": recovery_harness._sha256_file(parent / "compact_targets.json")
        },
        "target_replay": {
            "deterministic_recovery_identity_sha256": deterministic_identity,
            "alpha_replay": {"identity_sha256": alpha_identity},
        },
    }
    (parent / "tail_start.json").write_text(json.dumps(tail_start), encoding="utf-8")
    final_sha = recovery_harness._sha256_file(parent / "gaussians_final.ply")
    selection = {
        "schema": recovery_harness.MODEL_SELECTION_SCHEMA,
        "written_before_gaussians_final": True,
        "selected": {
            "global_step": 50_000,
            "artifact": "checkpoints/gaussians_step_050000.ply",
            "sha256": final_sha,
        },
        "convergence": {"joint_status": "still_improving"},
    }
    (parent / "model_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    selection_record = {
        "receipt": "model_selection.json",
        "receipt_sha256": recovery_harness._sha256_file(parent / "model_selection.json"),
        "selected_global_step": 50_000,
        "selected_candidate_sha256": final_sha,
        "joint_convergence_status": "still_improving",
    }
    history = {
        "loss": [0.1] * 10_000,
        "psnr": [[50_000, 20.0]],
        "elapsed": [[50_000, 1.0]],
        "n_gaussians": [[50_000, 7]],
        "active_sh_degree": [[50_000, 3]],
        "iteration_offset": 40_000,
        "segment_iterations": 10_000,
        "schedule_iterations": 50_000,
        "continuation_exact": False,
        "fixed_topology": True,
        "model_selection": selection_record,
    }
    fit = {
        "schema": "rtgs.full_compact_reconstruction.fit_complete.v1",
        "fit_kind": "fixed_topology_tail",
        "continuation_exact": False,
        "fixed_topology": True,
        "source_n_gaussians": 7,
        "n_final_gaussians": 7,
        "final_ply_sha256": final_sha,
        "tail": {
            "receipt": "tail_start.json",
            "receipt_sha256": recovery_harness._sha256_file(parent / "tail_start.json"),
        },
        "model_selection": selection_record,
    }
    (parent / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
    (parent / "fit_complete.json").write_text(json.dumps(fit), encoding="utf-8")
    paths = [parent / name for name in recovery_harness.COOLDOWN_PARENT_ARTIFACTS]
    paths.append(checkpoint)
    return {str(path.relative_to(parent)): recovery_harness._sha256_file(path) for path in paths}


def _write_settle_parent(recovery_harness, parent: Path, dataset: Path) -> dict[str, str]:
    _write_cooldown_parent(recovery_harness, parent, dataset)
    prior_trainer = recovery_harness._load_train_config(parent / "training_config.json")
    trainer = recovery_harness._cooldown_train_config(prior_trainer)
    (parent / "training_config.json").write_text(
        json.dumps(recovery_harness.dataclasses.asdict(trainer)),
        encoding="utf-8",
    )
    checkpoint = parent / "checkpoints" / "gaussians_step_060000.ply"
    checkpoint.write_bytes(b"selected-60k-ply")
    (parent / "gaussians_final.ply").write_bytes(checkpoint.read_bytes())
    deterministic_identity = "e" * 64
    alpha_identity = "f" * 64
    cooldown_start = {
        "schema": "rtgs.full_compact_reconstruction.cooldown_start.v1",
        "written_before_cooldown_training": True,
        "continuation_exact": False,
        "global_steps": {
            "parent_last": 50_000,
            "first_cooldown": 50_001,
            "last_cooldown": 60_000,
            "segment_iterations": 10_000,
            "schedule_iterations": 60_000,
        },
        "child_target_receipt": {
            "sha256": recovery_harness._sha256_file(parent / "compact_targets.json")
        },
        "target_replay": {
            "deterministic_recovery_identity_sha256": deterministic_identity,
            "alpha_replay": {"identity_sha256": alpha_identity},
        },
    }
    (parent / "cooldown_start.json").write_text(json.dumps(cooldown_start), encoding="utf-8")
    final_sha = recovery_harness._sha256_file(parent / "gaussians_final.ply")
    selection = {
        "schema": recovery_harness.MODEL_SELECTION_SCHEMA,
        "written_before_gaussians_final": True,
        "selected": {
            "global_step": 60_000,
            "artifact": "checkpoints/gaussians_step_060000.ply",
            "sha256": final_sha,
        },
        "convergence": {"joint_status": "still_improving"},
    }
    (parent / "model_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    selection_record = {
        "receipt": "model_selection.json",
        "receipt_sha256": recovery_harness._sha256_file(parent / "model_selection.json"),
        "selected_global_step": 60_000,
        "selected_candidate_sha256": final_sha,
        "joint_convergence_status": "still_improving",
    }
    history = {
        "loss": [0.1] * 10_000,
        "psnr": [[60_000, 20.0]],
        "elapsed": [[60_000, 1.0]],
        "n_gaussians": [[60_000, 7]],
        "active_sh_degree": [[60_000, 3]],
        "iteration_offset": 50_000,
        "segment_iterations": 10_000,
        "schedule_iterations": 60_000,
        "continuation_exact": False,
        "fixed_topology": True,
        "model_selection": selection_record,
    }
    fit = {
        "schema": "rtgs.full_compact_reconstruction.fit_complete.v1",
        "fit_kind": "fixed_topology_cooldown",
        "continuation_exact": False,
        "fixed_topology": True,
        "source_n_gaussians": 7,
        "n_final_gaussians": 7,
        "final_ply_sha256": final_sha,
        "cooldown": {
            "receipt": "cooldown_start.json",
            "receipt_sha256": recovery_harness._sha256_file(parent / "cooldown_start.json"),
        },
        "model_selection": selection_record,
    }
    (parent / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
    (parent / "fit_complete.json").write_text(json.dumps(fit), encoding="utf-8")
    paths = [parent / name for name in recovery_harness.SETTLE_PARENT_ARTIFACTS]
    paths.append(checkpoint)
    return {str(path.relative_to(parent)): recovery_harness._sha256_file(path) for path in paths}


def test_polish_config_is_fixed_topology_terminal_lr_segment(recovery_harness):
    parent = recovery_harness.TrainConfig(
        iterations=26_000,
        iteration_offset=4_000,
        schedule_iterations=30_000,
        lr_means=2.0e-4,
        lr_quats=4.0e-3,
        lr_scales=8.0e-3,
        lr_opacity=4.0e-2,
        lr_sh=2.0e-3,
        lr_sh_rest=1.0e-4,
        densify=True,
        target_sh_degree=3,
        sh_degree_interval=1000,
        eval_every=1000,
        seed=0,
    )
    polish = recovery_harness._polish_train_config(parent)
    assert polish.iterations == 10_000
    assert polish.iteration_offset == 30_000
    assert polish.schedule_iterations == 40_000
    assert polish.densify is False
    assert polish.seed == 1
    assert polish.means_lr_final_factor == 1.0
    assert polish.opacity_logit_epsilon == 1e-6
    assert polish.lr_means == pytest.approx(parent.lr_means * 0.01)
    for name in ("lr_quats", "lr_scales", "lr_opacity", "lr_sh", "lr_sh_rest"):
        assert getattr(polish, name) == pytest.approx(getattr(parent, name) * 0.25)
    assert recovery_harness._resolve_sh_interval(polish) == 1000
    assert min(polish.target_sh_degree, polish.iteration_offset // 1000) == 3


def test_polish_static_preflight_binds_recovered_parent_and_refuses_overwrite(
    recovery_harness,
    tmp_path,
):
    parent = tmp_path / "parent"
    out = tmp_path / "polish"
    _write_polish_parent(recovery_harness, parent, tmp_path / "dataset")
    state = recovery_harness._polish_parent_preflight(parent, out)
    assert state["parent"] == parent.resolve()
    assert state["out"] == out.resolve()
    assert state["n_final"] == 7
    assert state["recovery_trainer_config"].iteration_offset == 4_000
    with pytest.raises(RuntimeError, match="must not be the parent"):
        recovery_harness._polish_parent_preflight(parent, parent)
    out.mkdir()
    with pytest.raises(FileExistsError, match="overwrite polish output"):
        recovery_harness._polish_parent_preflight(parent, out)


def test_polish_static_preflight_rejects_parent_final_hash_drift(recovery_harness, tmp_path):
    parent = tmp_path / "parent"
    _write_polish_parent(recovery_harness, parent, tmp_path / "dataset")
    (parent / "gaussians_final.ply").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs from its completed-fit receipt"):
        recovery_harness._polish_parent_preflight(parent, tmp_path / "polish")


def test_polish_preflight_rejects_unbound_recovery_receipt(recovery_harness, tmp_path):
    parent = tmp_path / "parent"
    _write_polish_parent(recovery_harness, parent, tmp_path / "dataset")
    recovery_path = parent / "recovery/recovery_attempt_001.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery["resume_step"] = 4_001
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    fit = json.loads((parent / "fit_complete.json").read_text(encoding="utf-8"))
    fit["recovery"]["receipt_sha256"] = recovery_harness._sha256_file(recovery_path)
    (parent / "fit_complete.json").write_text(json.dumps(fit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected resume_step"):
        recovery_harness._polish_parent_preflight(parent, tmp_path / "polish")


def test_polish_promotes_v1_targets_and_receipts_opacity_before_training(
    recovery_harness,
    monkeypatch,
    tmp_path,
):
    parent = tmp_path / "parent"
    out = tmp_path / "polish"
    dataset_path = tmp_path / "dataset"
    before = _write_polish_parent(recovery_harness, parent, dataset_path)
    provenance = json.loads((parent / "provenance.json").read_text(encoding="utf-8"))
    target_views = json.loads((parent / "compact_targets.json").read_text(encoding="utf-8"))[
        "views"
    ]
    deterministic_identity = "e" * 64
    alpha_identity = "f" * 64
    dataset = SimpleNamespace(
        path=dataset_path,
        calibration_sha256="d" * 64,
        views=[SimpleNamespace(view_id=name) for name in recovery_harness.EXPECTED_VIEW_NAMES],
    )
    init = SimpleNamespace(
        n=7,
        sh_degree=3,
        opacity=torch.tensor([0.99995, 0.99999, 0.8, 0.5, 0.2, 1e-5, 5e-7]),
    )
    scene = SimpleNamespace()
    called = {}

    monkeypatch.setattr(
        recovery_harness.CompactDataset,
        "load",
        staticmethod(lambda *args, **kwargs: dataset),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_compact_provenance",
        lambda _dataset, _config: copy.deepcopy(provenance),
    )
    monkeypatch.setattr(
        recovery_harness.Gaussians3D,
        "load_ply",
        staticmethod(lambda _path: init),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_materialize_training_scene",
        lambda _dataset, _config: (scene, copy.deepcopy(target_views)),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_independently_replay_training_targets",
        lambda *_args: (
            copy.deepcopy(target_views),
            {"all_alpha_masks_match": True, "identity_sha256": alpha_identity},
        ),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_verify_target_replay",
        lambda *_args: {
            "reference_schema": recovery_harness.LEGACY_NONDETERMINISTIC_TARGET_SCHEMA,
            "original_tensor_equivalence_verified": False,
            "deterministic_recovery_replay_verified": True,
            "deterministic_recovery_identity_sha256": deterministic_identity,
        },
    )

    def fake_train(child_out, config, actual_scene, actual_init, trainer_config, **kwargs):
        receipt_path = kwargs["polish_receipt"]
        assert receipt_path.is_file()
        assert child_out == out.resolve()
        assert actual_scene is scene
        assert actual_init is init
        assert trainer_config.densify is False
        kwargs["initialization_callback"](
            SimpleNamespace(opacity=init.opacity.clamp(1e-6, 1.0 - 1e-6))
        )
        assert kwargs["trainer_entry_record"]["verified_before_optimizer_construction"] is True
        called["receipt"] = json.loads(receipt_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(recovery_harness, "_train_and_finalize", fake_train)
    recovery_harness._polish(SimpleNamespace(parent_out=parent, out=out))

    receipt = called["receipt"]
    assert receipt["written_before_polish_training"] is True
    assert receipt["continuation_exact"] is False
    assert receipt["fixed_topology"] == {
        "enabled": True,
        "densify": False,
        "source_n_gaussians": 7,
    }
    assert receipt["global_steps"] == {
        "parent_last": 30_000,
        "first_polish": 30_001,
        "last_polish": 40_000,
        "segment_iterations": 10_000,
        "schedule_iterations": 40_000,
    }
    assert receipt["parent_recovery"]["resume_exact"] is False
    assert receipt["parent_recovery"]["resume_step"] == 4_000
    assert receipt["parent_target_lineage"]["schema"] == (
        recovery_harness.LEGACY_NONDETERMINISTIC_TARGET_SCHEMA
    )
    assert receipt["parent_target_lineage"]["original_tensor_equivalence_verified"] is False
    assert receipt["child_target_receipt"]["schema"] == (
        recovery_harness.DETERMINISTIC_TARGET_SCHEMA
    )
    assert receipt["opacity_entry_plan"]["source_rows"] == 7
    assert receipt["opacity_entry_plan"]["above_historical_upper_count"] == 2
    assert receipt["opacity_entry_plan"]["polish_epsilon"] == 1e-6
    child_targets = json.loads((out / "compact_targets.json").read_text(encoding="utf-8"))
    assert child_targets["schema"] == recovery_harness.DETERMINISTIC_TARGET_SCHEMA
    assert child_targets["deterministic_algorithms"] is True
    parent_targets = json.loads((parent / "compact_targets.json").read_text(encoding="utf-8"))
    assert parent_targets["schema"] == recovery_harness.LEGACY_NONDETERMINISTIC_TARGET_SCHEMA
    assert (out / "gaussians_init.ply").read_bytes() == b"test-parent-ply"
    for name in recovery_harness.POLISH_COPIED_ARTIFACTS:
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    after = {relative: recovery_harness._sha256_file(parent / relative) for relative in before}
    assert after == before


def test_tail_config_preserves_polish_lrs_and_advances_global_schedule(recovery_harness):
    parent = recovery_harness.TrainConfig(
        iterations=10_000,
        iteration_offset=30_000,
        schedule_iterations=40_000,
        lr_means=2.0e-6,
        lr_quats=1.0e-3,
        lr_scales=2.0e-3,
        lr_opacity=1.0e-2,
        lr_sh=5.0e-4,
        lr_sh_rest=2.5e-5,
        densify=False,
        target_sh_degree=3,
        sh_degree_interval=1000,
        eval_every=1000,
        seed=1,
        means_lr_final_factor=1.0,
        opacity_logit_epsilon=1e-6,
    )
    tail = recovery_harness._tail_train_config(parent)
    assert tail.iterations == 10_000
    assert tail.iteration_offset == 40_000
    assert tail.schedule_iterations == 50_000
    assert tail.densify is False
    assert tail.seed == 2
    assert tail.means_lr_final_factor == 1.0
    assert tail.opacity_logit_epsilon == 1e-6
    for name in ("lr_means", "lr_quats", "lr_scales", "lr_opacity", "lr_sh", "lr_sh_rest"):
        assert getattr(tail, name) == getattr(parent, name)


def test_tail_preflight_binds_selected_40k_child_and_refuses_mutation(
    recovery_harness,
    tmp_path,
):
    parent = tmp_path / "polish"
    out = tmp_path / "tail"
    before = _write_tail_parent(recovery_harness, parent, tmp_path / "dataset")
    state = recovery_harness._tail_parent_preflight(parent, out)
    assert state["parent"] == parent.resolve()
    assert state["out"] == out.resolve()
    assert state["n_final"] == 7
    assert state["selection"]["selected"]["global_step"] == 40_000
    after = {relative: recovery_harness._sha256_file(parent / relative) for relative in before}
    assert after == before
    with pytest.raises(RuntimeError, match="must not be the parent"):
        recovery_harness._tail_parent_preflight(parent, parent)
    out.mkdir()
    with pytest.raises(FileExistsError, match="overwrite tail output"):
        recovery_harness._tail_parent_preflight(parent, out)


def test_tail_preflight_rejects_final_selected_checkpoint_hash_drift(
    recovery_harness,
    tmp_path,
):
    parent = tmp_path / "polish"
    _write_tail_parent(recovery_harness, parent, tmp_path / "dataset")
    (parent / "checkpoints/gaussians_step_040000.ply").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs from the selected 40k checkpoint"):
        recovery_harness._tail_parent_preflight(parent, tmp_path / "tail")


def test_tail_receipts_lineage_before_training_and_never_mutates_parent(
    recovery_harness,
    monkeypatch,
    tmp_path,
):
    parent = tmp_path / "polish"
    out = tmp_path / "tail"
    dataset_path = tmp_path / "dataset"
    before = _write_tail_parent(recovery_harness, parent, dataset_path)
    provenance = json.loads((parent / "provenance.json").read_text(encoding="utf-8"))
    target_views = json.loads((parent / "compact_targets.json").read_text(encoding="utf-8"))[
        "views"
    ]
    deterministic_identity = "e" * 64
    alpha_identity = "f" * 64
    dataset = SimpleNamespace(
        path=dataset_path,
        calibration_sha256="d" * 64,
        views=[SimpleNamespace(view_id=name) for name in recovery_harness.EXPECTED_VIEW_NAMES],
    )
    init = SimpleNamespace(
        n=7,
        sh_degree=3,
        opacity=torch.tensor([0.99999, 0.8, 0.5, 0.2, 0.1, 1e-5, 5e-7]),
    )
    scene = SimpleNamespace()
    called = {}
    monkeypatch.setattr(
        recovery_harness.CompactDataset,
        "load",
        staticmethod(lambda *args, **kwargs: dataset),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_compact_provenance",
        lambda _dataset, _config: copy.deepcopy(provenance),
    )
    monkeypatch.setattr(
        recovery_harness.Gaussians3D,
        "load_ply",
        staticmethod(lambda _path: init),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_materialize_training_scene",
        lambda _dataset, _config: (scene, copy.deepcopy(target_views)),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_independently_replay_training_targets",
        lambda *_args: (
            copy.deepcopy(target_views),
            {"all_alpha_masks_match": True, "identity_sha256": alpha_identity},
        ),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_verify_target_replay",
        lambda *_args: {
            "reference_schema": recovery_harness.DETERMINISTIC_TARGET_SCHEMA,
            "original_tensor_equivalence_verified": True,
            "deterministic_recovery_identity_sha256": deterministic_identity,
        },
    )

    def fake_train(child_out, config, actual_scene, actual_init, trainer_config, **kwargs):
        receipt_path = kwargs["tail_receipt"]
        assert receipt_path.is_file()
        assert child_out == out.resolve()
        assert actual_scene is scene
        assert actual_init is init
        assert trainer_config.iteration_offset == 40_000
        assert trainer_config.schedule_iterations == 50_000
        assert kwargs["selection_baseline_step"] == 40_000
        assert kwargs["selection_final_step"] == 50_000
        kwargs["initialization_callback"](
            SimpleNamespace(opacity=init.opacity.clamp(1e-6, 1.0 - 1e-6))
        )
        assert kwargs["trainer_entry_record"]["verified_before_optimizer_construction"] is True
        called["receipt"] = json.loads(receipt_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(recovery_harness, "_train_and_finalize", fake_train)
    recovery_harness._tail(SimpleNamespace(parent_out=parent, out=out))

    receipt = called["receipt"]
    assert receipt["written_before_tail_training"] is True
    assert receipt["continuation_exact"] is False
    assert receipt["parent_model_selection"]["selected_global_step"] == 40_000
    assert receipt["global_steps"] == {
        "parent_last": 40_000,
        "first_tail": 40_001,
        "last_tail": 50_000,
        "segment_iterations": 10_000,
        "schedule_iterations": 50_000,
    }
    assert receipt["child_target_receipt"]["byte_copied_from_parent"] is True
    assert (out / "compact_targets.json").read_bytes() == (
        parent / "compact_targets.json"
    ).read_bytes()
    assert (out / "gaussians_init.ply").read_bytes() == b"selected-40k-ply"
    after = {relative: recovery_harness._sha256_file(parent / relative) for relative in before}
    assert after == before


def test_cooldown_config_quarters_every_lr_and_advances_global_schedule(recovery_harness):
    parent = recovery_harness.TrainConfig(
        iterations=10_000,
        iteration_offset=40_000,
        schedule_iterations=50_000,
        lr_means=2.0e-6,
        lr_quats=1.0e-3,
        lr_scales=2.0e-3,
        lr_opacity=1.0e-2,
        lr_sh=5.0e-4,
        lr_sh_rest=2.5e-5,
        densify=False,
        target_sh_degree=3,
        sh_degree_interval=1000,
        eval_every=1000,
        seed=2,
        means_lr_final_factor=1.0,
        opacity_logit_epsilon=1e-6,
    )
    cooldown = recovery_harness._cooldown_train_config(parent)
    assert cooldown.iterations == 10_000
    assert cooldown.iteration_offset == 50_000
    assert cooldown.schedule_iterations == 60_000
    assert cooldown.densify is False
    assert cooldown.seed == 3
    assert cooldown.means_lr_final_factor == 1.0
    assert cooldown.opacity_logit_epsilon == 1e-6
    for name in ("lr_means", "lr_quats", "lr_scales", "lr_opacity", "lr_sh", "lr_sh_rest"):
        assert getattr(cooldown, name) == pytest.approx(getattr(parent, name) * 0.25)


def test_cooldown_preflight_binds_selected_50k_tail_and_refuses_mutation(
    recovery_harness,
    tmp_path,
):
    parent = tmp_path / "tail"
    out = tmp_path / "cooldown"
    before = _write_cooldown_parent(recovery_harness, parent, tmp_path / "dataset")
    state = recovery_harness._cooldown_parent_preflight(parent, out)
    assert state["parent"] == parent.resolve()
    assert state["out"] == out.resolve()
    assert state["n_final"] == 7
    assert state["selection"]["selected"]["global_step"] == 50_000
    after = {relative: recovery_harness._sha256_file(parent / relative) for relative in before}
    assert after == before
    with pytest.raises(RuntimeError, match="must not be the parent"):
        recovery_harness._cooldown_parent_preflight(parent, parent)
    out.mkdir()
    with pytest.raises(FileExistsError, match="overwrite cooldown output"):
        recovery_harness._cooldown_parent_preflight(parent, out)


def test_cooldown_preflight_rejects_selected_50k_checkpoint_hash_drift(
    recovery_harness,
    tmp_path,
):
    parent = tmp_path / "tail"
    _write_cooldown_parent(recovery_harness, parent, tmp_path / "dataset")
    (parent / "checkpoints/gaussians_step_050000.ply").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs from the selected 50k checkpoint"):
        recovery_harness._cooldown_parent_preflight(parent, tmp_path / "cooldown")


def test_cooldown_receipts_lineage_before_training_and_never_mutates_parent(
    recovery_harness,
    monkeypatch,
    tmp_path,
):
    parent = tmp_path / "tail"
    out = tmp_path / "cooldown"
    dataset_path = tmp_path / "dataset"
    before = _write_cooldown_parent(recovery_harness, parent, dataset_path)
    provenance = json.loads((parent / "provenance.json").read_text(encoding="utf-8"))
    target_views = json.loads((parent / "compact_targets.json").read_text(encoding="utf-8"))[
        "views"
    ]
    deterministic_identity = "e" * 64
    alpha_identity = "f" * 64
    dataset = SimpleNamespace(
        path=dataset_path,
        calibration_sha256="d" * 64,
        views=[SimpleNamespace(view_id=name) for name in recovery_harness.EXPECTED_VIEW_NAMES],
    )
    init = SimpleNamespace(
        n=7,
        sh_degree=3,
        opacity=torch.tensor([0.99999, 0.8, 0.5, 0.2, 0.1, 1e-5, 5e-7]),
    )
    scene = SimpleNamespace()
    called = {}
    monkeypatch.setattr(
        recovery_harness.CompactDataset,
        "load",
        staticmethod(lambda *args, **kwargs: dataset),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_compact_provenance",
        lambda _dataset, _config: copy.deepcopy(provenance),
    )
    monkeypatch.setattr(
        recovery_harness.Gaussians3D,
        "load_ply",
        staticmethod(lambda _path: init),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_materialize_training_scene",
        lambda _dataset, _config: (scene, copy.deepcopy(target_views)),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_independently_replay_training_targets",
        lambda *_args: (
            copy.deepcopy(target_views),
            {"all_alpha_masks_match": True, "identity_sha256": alpha_identity},
        ),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_verify_target_replay",
        lambda *_args: {
            "reference_schema": recovery_harness.DETERMINISTIC_TARGET_SCHEMA,
            "original_tensor_equivalence_verified": True,
            "deterministic_recovery_identity_sha256": deterministic_identity,
        },
    )

    def fake_train(child_out, config, actual_scene, actual_init, trainer_config, **kwargs):
        receipt_path = kwargs["cooldown_receipt"]
        assert receipt_path.is_file()
        assert child_out == out.resolve()
        assert actual_scene is scene
        assert actual_init is init
        assert trainer_config.iteration_offset == 50_000
        assert trainer_config.schedule_iterations == 60_000
        assert kwargs["selection_baseline_step"] == 50_000
        assert kwargs["selection_final_step"] == 60_000
        kwargs["initialization_callback"](
            SimpleNamespace(opacity=init.opacity.clamp(1e-6, 1.0 - 1e-6))
        )
        assert kwargs["trainer_entry_record"]["verified_before_optimizer_construction"] is True
        called["receipt"] = json.loads(receipt_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(recovery_harness, "_train_and_finalize", fake_train)
    recovery_harness._cooldown(SimpleNamespace(parent_out=parent, out=out))

    receipt = called["receipt"]
    assert receipt["written_before_cooldown_training"] is True
    assert receipt["continuation_exact"] is False
    assert receipt["parent_model_selection"]["selected_global_step"] == 50_000
    assert receipt["global_steps"] == {
        "parent_last": 50_000,
        "first_cooldown": 50_001,
        "last_cooldown": 60_000,
        "segment_iterations": 10_000,
        "schedule_iterations": 60_000,
    }
    assert receipt["learning_rates"]["parent_to_cooldown_factor"] == 0.25
    assert receipt["child_target_receipt"]["byte_copied_from_parent"] is True
    assert (out / "compact_targets.json").read_bytes() == (
        parent / "compact_targets.json"
    ).read_bytes()
    assert (out / "gaussians_init.ply").read_bytes() == b"selected-50k-ply"
    after = {relative: recovery_harness._sha256_file(parent / relative) for relative in before}
    assert after == before


def test_settle_config_quarters_every_lr_and_advances_global_schedule(recovery_harness):
    parent = recovery_harness.TrainConfig(
        iterations=10_000,
        iteration_offset=50_000,
        schedule_iterations=60_000,
        lr_means=5.0e-7,
        lr_quats=2.5e-4,
        lr_scales=5.0e-4,
        lr_opacity=2.5e-3,
        lr_sh=1.25e-4,
        lr_sh_rest=6.25e-6,
        densify=False,
        target_sh_degree=3,
        sh_degree_interval=1000,
        eval_every=1000,
        seed=3,
        means_lr_final_factor=1.0,
        opacity_logit_epsilon=1e-6,
    )
    settle = recovery_harness._settle_train_config(parent)
    assert settle.iterations == 10_000
    assert settle.iteration_offset == 60_000
    assert settle.schedule_iterations == 70_000
    assert settle.densify is False
    assert settle.seed == 4
    assert settle.means_lr_final_factor == 1.0
    assert settle.opacity_logit_epsilon == 1e-6
    for name in ("lr_means", "lr_quats", "lr_scales", "lr_opacity", "lr_sh", "lr_sh_rest"):
        assert getattr(settle, name) == pytest.approx(getattr(parent, name) * 0.25)


def test_settle_preflight_binds_selected_60k_cooldown_and_refuses_mutation(
    recovery_harness,
    tmp_path,
):
    parent = tmp_path / "cooldown"
    out = tmp_path / "settle"
    before = _write_settle_parent(recovery_harness, parent, tmp_path / "dataset")
    state = recovery_harness._settle_parent_preflight(parent, out)
    assert state["parent"] == parent.resolve()
    assert state["out"] == out.resolve()
    assert state["n_final"] == 7
    assert state["selection"]["selected"]["global_step"] == 60_000
    after = {relative: recovery_harness._sha256_file(parent / relative) for relative in before}
    assert after == before
    with pytest.raises(RuntimeError, match="must not be the parent"):
        recovery_harness._settle_parent_preflight(parent, parent)
    out.mkdir()
    with pytest.raises(FileExistsError, match="overwrite settle output"):
        recovery_harness._settle_parent_preflight(parent, out)


def test_settle_preflight_rejects_selected_60k_checkpoint_hash_drift(
    recovery_harness,
    tmp_path,
):
    parent = tmp_path / "cooldown"
    _write_settle_parent(recovery_harness, parent, tmp_path / "dataset")
    (parent / "checkpoints/gaussians_step_060000.ply").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs from the selected 60k checkpoint"):
        recovery_harness._settle_parent_preflight(parent, tmp_path / "settle")


def test_settle_receipts_lineage_before_training_and_never_mutates_parent(
    recovery_harness,
    monkeypatch,
    tmp_path,
):
    parent = tmp_path / "cooldown"
    out = tmp_path / "settle"
    dataset_path = tmp_path / "dataset"
    before = _write_settle_parent(recovery_harness, parent, dataset_path)
    provenance = json.loads((parent / "provenance.json").read_text(encoding="utf-8"))
    target_views = json.loads((parent / "compact_targets.json").read_text(encoding="utf-8"))[
        "views"
    ]
    deterministic_identity = "e" * 64
    alpha_identity = "f" * 64
    dataset = SimpleNamespace(
        path=dataset_path,
        calibration_sha256="d" * 64,
        views=[SimpleNamespace(view_id=name) for name in recovery_harness.EXPECTED_VIEW_NAMES],
    )
    init = SimpleNamespace(
        n=7,
        sh_degree=3,
        opacity=torch.tensor([0.99999, 0.8, 0.5, 0.2, 0.1, 1e-5, 5e-7]),
    )
    scene = SimpleNamespace()
    called = {}
    monkeypatch.setattr(
        recovery_harness.CompactDataset,
        "load",
        staticmethod(lambda *args, **kwargs: dataset),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_compact_provenance",
        lambda _dataset, _config: copy.deepcopy(provenance),
    )
    monkeypatch.setattr(
        recovery_harness.Gaussians3D,
        "load_ply",
        staticmethod(lambda _path: init),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_materialize_training_scene",
        lambda _dataset, _config: (scene, copy.deepcopy(target_views)),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_independently_replay_training_targets",
        lambda *_args: (
            copy.deepcopy(target_views),
            {"all_alpha_masks_match": True, "identity_sha256": alpha_identity},
        ),
    )
    monkeypatch.setattr(
        recovery_harness,
        "_verify_target_replay",
        lambda *_args: {
            "reference_schema": recovery_harness.DETERMINISTIC_TARGET_SCHEMA,
            "original_tensor_equivalence_verified": True,
            "deterministic_recovery_identity_sha256": deterministic_identity,
        },
    )

    def fake_train(child_out, config, actual_scene, actual_init, trainer_config, **kwargs):
        receipt_path = kwargs["settle_receipt"]
        assert receipt_path.is_file()
        assert child_out == out.resolve()
        assert actual_scene is scene
        assert actual_init is init
        assert trainer_config.iteration_offset == 60_000
        assert trainer_config.schedule_iterations == 70_000
        assert kwargs["selection_baseline_step"] == 60_000
        assert kwargs["selection_final_step"] == 70_000
        kwargs["initialization_callback"](
            SimpleNamespace(opacity=init.opacity.clamp(1e-6, 1.0 - 1e-6))
        )
        assert kwargs["trainer_entry_record"]["verified_before_optimizer_construction"] is True
        called["receipt"] = json.loads(receipt_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(recovery_harness, "_train_and_finalize", fake_train)
    recovery_harness._settle(SimpleNamespace(parent_out=parent, out=out))

    receipt = called["receipt"]
    assert receipt["written_before_settle_training"] is True
    assert receipt["continuation_exact"] is False
    assert receipt["parent_model_selection"]["selected_global_step"] == 60_000
    assert receipt["global_steps"] == {
        "parent_last": 60_000,
        "first_settle": 60_001,
        "last_settle": 70_000,
        "segment_iterations": 10_000,
        "schedule_iterations": 70_000,
    }
    assert receipt["learning_rates"]["parent_to_settle_factor"] == 0.25
    assert receipt["child_target_receipt"]["byte_copied_from_parent"] is True
    assert (out / "compact_targets.json").read_bytes() == (
        parent / "compact_targets.json"
    ).read_bytes()
    assert (out / "gaussians_init.ply").read_bytes() == b"selected-60k-ply"
    after = {relative: recovery_harness._sha256_file(parent / relative) for relative in before}
    assert after == before


def test_compact_selection_objective_matches_literal_trainer_formula(recovery_harness):
    generator = torch.Generator().manual_seed(901)
    color = torch.rand(7, 8, 3, generator=generator)
    alpha = torch.rand(7, 8, generator=generator)
    target = torch.rand(7, 8, 3, generator=generator)
    mask = torch.zeros(7, 8)
    mask[1:6, 2:7] = 1.0
    config = recovery_harness.TrainConfig(
        ssim_lambda=0.2,
        outside_alpha_lambda=0.01,
        mask_alpha_lambda=0.05,
        use_masks=True,
        random_background=False,
        density_strategy="gsplat-default",
    )
    terms = recovery_harness._compact_training_objective_terms(color, alpha, target, mask, config)
    target_for_loss = target * mask[..., None]
    weighted_l1 = ((color - target_for_loss).abs() * (0.1 + 0.9 * mask)[..., None]).mean()
    outside_alpha = (alpha * (1.0 - mask)).mean()
    l1 = weighted_l1 + 0.01 * outside_alpha
    alpha_l1 = (alpha - mask).abs().mean()
    crop_ssim = recovery_harness.ssim(
        recovery_harness.masked_crop(color, mask),
        recovery_harness.masked_crop(target_for_loss, mask),
    )
    expected = 0.8 * l1 + 0.05 * alpha_l1 + 0.2 * (1.0 - crop_ssim)
    assert terms["weighted_rgb_l1"] == pytest.approx(float(weighted_l1))
    assert terms["outside_alpha_mean"] == pytest.approx(float(outside_alpha))
    assert terms["mask_alpha_l1"] == pytest.approx(float(alpha_l1))
    assert terms["crop_ssim"] == pytest.approx(float(crop_ssim))
    assert terms["objective"] == pytest.approx(float(expected))


def test_model_selection_uses_earliest_candidate_within_relative_tie(recovery_harness):
    records = [
        {"global_step": 30_000, "objective": 1.0000008},
        {"global_step": 31_000, "objective": 1.0},
        {"global_step": 32_000, "objective": 1.0000002},
    ]
    selected, receipt = recovery_harness._choose_earliest_objective_tie(records)
    assert selected["global_step"] == 30_000
    assert receipt["eligible_global_steps"] == [30_000, 31_000, 32_000]
    records[0]["objective"] = 1.0000012
    selected, receipt = recovery_harness._choose_earliest_objective_tie(records)
    assert selected["global_step"] == 31_000
    assert receipt["eligible_global_steps"] == [31_000, 32_000]


def _trend_records(*, objective_factor: float, psnr_gain: float):
    records = []
    for offset in range(11):
        objective = objective_factor**offset
        records.append(
            {
                "global_step": 30_000 + 1_000 * offset,
                "objective": objective,
                "psnr_fg_db": 30.0 + psnr_gain * offset,
                "per_view": [
                    {"view_name": "a", "objective": objective},
                    {"view_name": "b", "objective": objective * 1.1},
                ],
            }
        )
    return records


def test_material_plateau_requires_five_consecutive_nonmaterial_transitions(
    recovery_harness,
):
    records = _trend_records(objective_factor=0.999, psnr_gain=0.01)
    result = recovery_harness._material_plateau(records)
    assert result["plateau"] is True
    assert result["trailing_nonmaterial_transitions"] == 10
    records[-1]["objective"] = records[-2]["objective"] * 0.996
    result = recovery_harness._material_plateau(records)
    assert result["plateau"] is False
    assert result["trailing_nonmaterial_transitions"] == 0


def test_frozen_last_six_trend_classifies_plateau_improvement_and_regression(
    recovery_harness,
):
    plateau = recovery_harness._frozen_last_six_trend(
        _trend_records(objective_factor=1.0, psnr_gain=0.0)
    )
    assert plateau["status"] == "plateau"
    assert plateau["theil_sen_psnr_slope_db_per_1k"] == pytest.approx(0.0)
    improving = recovery_harness._frozen_last_six_trend(
        _trend_records(objective_factor=0.99, psnr_gain=0.06)
    )
    assert improving["status"] == "still_improving"
    regression_records = _trend_records(objective_factor=1.001, psnr_gain=0.0)
    regression_records[-1]["psnr_fg_db"] -= 0.2
    regression = recovery_harness._frozen_last_six_trend(regression_records)
    assert regression["status"] == "regression"
    assert regression["regression"] is True


def test_polish_candidate_selection_reloads_every_ply_on_cpu(recovery_harness, tmp_path):
    scene = make_synthetic_scene(n_gaussians=4, n_cameras=2, image_size=8, seed=902)
    scene.masks = [torch.ones(image.shape[:2]) for image in scene.images]
    scene.view_names = ["view_0", "view_1"]
    scene.train_indices = [0, 1]
    initial = scene.gt_gaussians.detach().with_sh_degree(3)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    initial.save_ply(tmp_path / "gaussians_init.ply")
    for step in range(31_000, 40_001, 1_000):
        initial.save_ply(checkpoints / f"gaussians_step_{step:06d}.ply")
    config = recovery_harness.TrainConfig(
        rasterizer="torch",
        device="cpu",
        densify=False,
        density_strategy="gsplat-default",
        target_sh_degree=3,
        use_masks=True,
        random_background=False,
        packed=False,
        antialiased=False,
    )

    selected, receipt = recovery_harness._evaluate_polish_candidates(
        tmp_path,
        scene,
        config,
        expected_n=initial.n,
    )

    assert selected == tmp_path / "gaussians_init.ply"
    assert receipt["selected"]["global_step"] == 30_000
    assert len(receipt["candidates"]) == 11
    assert receipt["convergence"]["joint_status"] == "plateau"
    assert all(record["n_gaussians"] == initial.n for record in receipt["candidates"])


def test_tail_candidate_selection_scores_40k_baseline_through_50k_on_cpu(
    recovery_harness,
    tmp_path,
):
    scene = make_synthetic_scene(n_gaussians=3, n_cameras=1, image_size=8, seed=903)
    scene.masks = [torch.ones(scene.images[0].shape[:2])]
    scene.view_names = ["view_0"]
    scene.train_indices = [0]
    initial = scene.gt_gaussians.detach().with_sh_degree(3)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    initial.save_ply(tmp_path / "gaussians_init.ply")
    for step in range(41_000, 50_001, 1_000):
        initial.save_ply(checkpoints / f"gaussians_step_{step:06d}.ply")
    config = recovery_harness.TrainConfig(
        rasterizer="torch",
        device="cpu",
        densify=False,
        density_strategy="gsplat-default",
        target_sh_degree=3,
        use_masks=True,
        random_background=False,
        packed=False,
        antialiased=False,
    )

    selected, receipt = recovery_harness._evaluate_polish_candidates(
        tmp_path,
        scene,
        config,
        expected_n=initial.n,
        baseline_step=40_000,
        final_step=50_000,
    )

    assert selected == tmp_path / "gaussians_init.ply"
    assert receipt["selected"]["global_step"] == 40_000
    assert [record["global_step"] for record in receipt["candidates"]] == list(
        range(40_000, 50_001, 1_000)
    )
    assert receipt["convergence"]["joint_status"] == "plateau"


def test_cooldown_candidate_selection_scores_50k_baseline_through_60k_on_cpu(
    recovery_harness,
    tmp_path,
):
    scene = make_synthetic_scene(n_gaussians=3, n_cameras=1, image_size=8, seed=904)
    scene.masks = [torch.ones(scene.images[0].shape[:2])]
    scene.view_names = ["view_0"]
    scene.train_indices = [0]
    initial = scene.gt_gaussians.detach().with_sh_degree(3)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    initial.save_ply(tmp_path / "gaussians_init.ply")
    for step in range(51_000, 60_001, 1_000):
        initial.save_ply(checkpoints / f"gaussians_step_{step:06d}.ply")
    config = recovery_harness.TrainConfig(
        rasterizer="torch",
        device="cpu",
        densify=False,
        density_strategy="gsplat-default",
        target_sh_degree=3,
        use_masks=True,
        random_background=False,
        packed=False,
        antialiased=False,
    )

    selected, receipt = recovery_harness._evaluate_polish_candidates(
        tmp_path,
        scene,
        config,
        expected_n=initial.n,
        baseline_step=50_000,
        final_step=60_000,
    )

    assert selected == tmp_path / "gaussians_init.ply"
    assert receipt["selected"]["global_step"] == 50_000
    assert [record["global_step"] for record in receipt["candidates"]] == list(
        range(50_000, 60_001, 1_000)
    )
    assert receipt["convergence"]["joint_status"] == "plateau"


def test_settle_candidate_selection_scores_60k_baseline_through_70k_on_cpu(
    recovery_harness,
    tmp_path,
):
    scene = make_synthetic_scene(n_gaussians=3, n_cameras=1, image_size=8, seed=905)
    scene.masks = [torch.ones(scene.images[0].shape[:2])]
    scene.view_names = ["view_0"]
    scene.train_indices = [0]
    initial = scene.gt_gaussians.detach().with_sh_degree(3)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    initial.save_ply(tmp_path / "gaussians_init.ply")
    for step in range(61_000, 70_001, 1_000):
        initial.save_ply(checkpoints / f"gaussians_step_{step:06d}.ply")
    config = recovery_harness.TrainConfig(
        rasterizer="torch",
        device="cpu",
        densify=False,
        density_strategy="gsplat-default",
        target_sh_degree=3,
        use_masks=True,
        random_background=False,
        packed=False,
        antialiased=False,
    )
    selected, receipt = recovery_harness._evaluate_polish_candidates(
        tmp_path,
        scene,
        config,
        expected_n=initial.n,
        baseline_step=60_000,
        final_step=70_000,
    )
    assert selected == tmp_path / "gaussians_init.ply"
    assert receipt["selected"]["global_step"] == 60_000
    assert [record["global_step"] for record in receipt["candidates"]] == list(
        range(60_000, 70_001, 1_000)
    )
    assert receipt["convergence"]["joint_status"] == "plateau"


def test_recovery_cli_requires_checkpoint_only_for_recover(recovery_harness, tmp_path):
    checkpoint = tmp_path / "gaussians_step_004000.ply"
    args = recovery_harness.parse_args(
        [
            "--phase",
            "recover",
            "--out",
            str(tmp_path / "run"),
            "--resume-checkpoint",
            str(checkpoint),
        ]
    )
    assert args.phase == "recover"
    assert args.resume_checkpoint == checkpoint
    with pytest.raises(SystemExit):
        recovery_harness.parse_args(["--phase", "recover", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        recovery_harness.parse_args(
            [
                "--phase",
                "fit",
                "--out",
                str(tmp_path),
                "--resume-checkpoint",
                str(checkpoint),
            ]
        )
    polish = recovery_harness.parse_args(
        [
            "--phase",
            "polish",
            "--out",
            str(tmp_path / "polish"),
            "--parent-out",
            str(tmp_path / "parent"),
        ]
    )
    assert polish.phase == "polish"
    assert polish.parent_out == tmp_path / "parent"
    tail = recovery_harness.parse_args(
        [
            "--phase",
            "tail",
            "--out",
            str(tmp_path / "tail"),
            "--parent-out",
            str(tmp_path / "polish"),
        ]
    )
    assert tail.phase == "tail"
    assert tail.parent_out == tmp_path / "polish"
    cooldown = recovery_harness.parse_args(
        [
            "--phase",
            "cooldown",
            "--out",
            str(tmp_path / "cooldown"),
            "--parent-out",
            str(tmp_path / "tail"),
        ]
    )
    assert cooldown.phase == "cooldown"
    assert cooldown.parent_out == tmp_path / "tail"
    settle = recovery_harness.parse_args(
        [
            "--phase",
            "settle",
            "--out",
            str(tmp_path / "settle"),
            "--parent-out",
            str(tmp_path / "cooldown"),
        ]
    )
    assert settle.phase == "settle"
    assert settle.parent_out == tmp_path / "cooldown"
    with pytest.raises(SystemExit):
        recovery_harness.parse_args(["--phase", "polish", "--out", str(tmp_path / "polish")])
    with pytest.raises(SystemExit):
        recovery_harness.parse_args(["--phase", "tail", "--out", str(tmp_path / "tail")])
    with pytest.raises(SystemExit):
        recovery_harness.parse_args(["--phase", "cooldown", "--out", str(tmp_path / "cooldown")])
    with pytest.raises(SystemExit):
        recovery_harness.parse_args(["--phase", "settle", "--out", str(tmp_path / "settle")])
    with pytest.raises(SystemExit):
        recovery_harness.parse_args(
            [
                "--phase",
                "fit",
                "--out",
                str(tmp_path / "run"),
                "--parent-out",
                str(tmp_path / "parent"),
            ]
        )
