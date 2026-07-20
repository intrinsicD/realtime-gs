"""Outcome-free checks for the sealed compact point-training harness."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARNESS_PATH = ROOT / "benchmarks/compact_point_training.py"


def _record_child_preload(expected: str, result_path_text: str) -> None:
    """Spawn-picklable CPU-only probe for the worker environment contract."""
    actual = os.environ.get("LD_PRELOAD")
    payload = {"status": "PASS" if actual == expected else "FAIL", "ld_preload": actual}
    Path(result_path_text).write_text(json.dumps(payload), encoding="utf-8")
    if actual != expected:
        raise RuntimeError("spawn child did not inherit LD_PRELOAD")


def _record_child_abi(binding: dict[str, str], result_path_text: str) -> None:
    """Spawn-picklable probe for versioned default-namespace symbol resolution."""
    module = _load_harness()
    result_path = Path(result_path_text)
    try:
        resolution = module._default_namespace_abi_resolution(binding)
        module._exclusive_json(result_path, {"status": "PASS", "resolution": resolution})
    except BaseException as error:
        module._exclusive_json(
            result_path,
            {
                "status": "FAIL",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            },
        )
        raise


def _load_harness():
    spec = importlib.util.spec_from_file_location("compact_point_training_harness", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_preload_binding(module, tmp_path: Path) -> dict[str, str]:
    resolved = tmp_path / "libstdc++.so.6.0.test"
    resolved.write_bytes(b"test C++ runtime")
    requested = tmp_path / "libstdc++.so.6"
    requested.symlink_to(resolved.name)
    return module._abi_preload_file_binding(str(requested))


def _fake_abi_resolution(module, binding: dict[str, str]) -> dict[str, str]:
    resolved = binding["resolved_path"]
    return {
        "lookup": "dlvsym(RTLD_DEFAULT)",
        "required_symbol": module.REQUIRED_CXXABI_SYMBOL,
        "required_version": module.REQUIRED_CXXABI_VERSION,
        "symbol_address": "0x123",
        "dladdr_symbol": module.REQUIRED_CXXABI_SYMBOL,
        "dladdr_library_path": binding["requested_path"],
        "dladdr_resolved_library_path": resolved,
        "proc_maps_entry": f"00000100-00000200 r-xp 00000000 00:00 1 {resolved}",
        "proc_maps_library_path": resolved,
        "proc_maps_resolved_library_path": resolved,
        "library_sha256": binding["sha256"],
    }


def _checkpoint(step: int, pixel: float, area: float) -> dict:
    return {"step": step, "risks": {"pixel": pixel, "area": area}}


def _record(module, seed: int, arm: str, risks: tuple[float, float, float, float]) -> dict:
    schedule = [index % 3 for index in range(module.ITERATIONS)]
    invariants = {
        "active_implies_inside_fit_window": True,
        "invalid_zero_joint_density": True,
        "invalid_zero_target_density": True,
        "invalid_zero_proposal_density": True,
        "invalid_zero_importance": True,
        "no_null_resampling": True,
    }
    attempts = module.ITERATIONS * module.ATTEMPTS_PER_STEP
    uniform = arm.endswith("uniform")
    branches = (
        {"uniform": attempts, "gaussian": 0, "gaussian_accepted": 0, "gaussian_rejected": 0}
        if uniform
        else {
            "uniform": attempts // 4,
            "gaussian": 3 * attempts // 4,
            "gaussian_accepted": attempts // 2,
            "gaussian_rejected": attempts // 4,
        }
    )
    checkpoints = [
        _checkpoint(step, risk, risk * 1.1)
        for step, risk in zip(module.CHECKPOINTS, risks, strict=True)
    ]
    return {
        "status": "PASS",
        "seed": seed,
        "arm": arm,
        "risk_measure": "discrete_pixels" if arm.startswith("pixel") else "continuous_area",
        "initialization_hash": f"init-{seed}",
        "teacher_hashes_before": [{"aggregate": f"teacher-{seed}"}],
        "teacher_hashes_after": [{"aggregate": f"teacher-{seed}"}],
        "view_schedule": schedule,
        "view_schedule_sha256": module.canonical_hash(schedule),
        "checkpoints": checkpoints,
        "optimizer_group_clocks": {
            "means": 120,
            "quats": 120,
            "scales": 120,
            "opacities": 120,
            "sh0": 120,
            "shN": 120,
        },
        "optimizer_steps": 120,
        "n_initial": 4,
        "n_final": 4,
        "attempt_count": attempts,
        "maximum_importance": 1.0 if uniform else 3.9,
        "active_fraction": 1.0 if uniform else 0.75,
        "importance_ess_per_attempt": 1.0 if uniform else 0.6,
        "proposal_branch_counts": branches,
        "proposal_invariants": invariants,
        "rgb_access_count": 0,
        "parameter_motion": {
            "means": 1e-3,
            "quaternions": 2e-3,
            "log_scales": 3e-3,
            "opacity_logits": 4e-3,
            "sh0": 5e-3,
        },
        "wall_seconds": 1.0,
        "peak_rss_kib": 1234,
    }


def test_import_is_outcome_free_and_constructor_requires_marker(monkeypatch, tmp_path):
    module = _load_harness()
    assert module.official_fixture_construction_count() == 0
    monkeypatch.setattr(module, "ATTEMPT", tmp_path / "absent-attempt.json")
    with pytest.raises(module.ProtocolInvalid, match="attempt marker"):
        module.official_fixture(0)
    with pytest.raises(module.ProtocolInvalid, match="marker-bound authorization"):
        module._literal_target(authorization=None)
    with pytest.raises(module.ProtocolInvalid, match="marker-bound authorization"):
        module._literal_cameras(authorization=None)
    with pytest.raises(module.ProtocolInvalid, match="marker-bound authorization"):
        module._perturbed_initialization(None, 63840, authorization=None)
    with pytest.raises(module.ProtocolInvalid, match="marker-bound authorization"):
        module._official_config("pixel_uniform", 63840, authorization=None)
    with pytest.raises(module.ProtocolInvalid, match="marker-bound authorization"):
        module._construct_official_fixture(63840, authorization=None)
    with pytest.raises(module.ProtocolInvalid, match="attempt marker"):
        module._execute_seed_arm(63840, "pixel_uniform", "0" * 64)
    assert module.official_fixture_construction_count() == 0


def test_incomplete_fake_marker_cannot_authorize_constructor(monkeypatch, tmp_path):
    module = _load_harness()
    marker_path = tmp_path / "fake-attempt.json"
    module._exclusive_json(
        marker_path,
        {
            "artifact_type": module.ATTEMPT_ARTIFACT_TYPE,
            "preregistration_sha256": module.PREREGISTRATION_SHA256,
        },
    )
    monkeypatch.setattr(module, "ATTEMPT", marker_path)
    monkeypatch.setattr(
        module,
        "load_and_verify_seal",
        lambda: {
            "seal_file_sha256": "a" * 64,
            "seal_payload_sha256": "b" * 64,
            "source_aggregate": "c" * 64,
            "environment": module._environment_metadata(),
        },
    )
    with pytest.raises(module.ProtocolInvalid, match="key set"):
        module.official_fixture(63840)
    assert module.official_fixture_construction_count() == 0


def test_normalized_log_auc_matches_literal_hand_calculation():
    module = _load_harness()
    # log-risk ratios are (0, -0.3, -0.6, -1.2) at normalized times (0,.25,.5,1).
    checkpoints = [
        _checkpoint(0, 1.0, 1.0),
        _checkpoint(30, math.exp(-0.3), math.exp(-0.3)),
        _checkpoint(60, math.exp(-0.6), math.exp(-0.6)),
        _checkpoint(120, math.exp(-1.2), math.exp(-1.2)),
    ]
    assert module.normalized_log_auc(checkpoints, "pixel") == pytest.approx(-0.6, abs=1e-12)


def test_renderer_boundary_helper_uses_actual_near_and_opacity_clamps():
    module = _load_harness()
    import torch

    from rtgs.core.camera import Camera
    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.core.sh import rgb_to_sh

    camera = Camera(
        fx=1.0,
        fy=1.0,
        cx=0.5,
        cy=0.5,
        width=1,
        height=1,
        R=torch.eye(3),
        t=torch.zeros(3),
    )
    gaussians = Gaussians3D(
        means=torch.tensor([[0.0, 0.0, 0.050005], [0.0, 0.0, 0.2]]),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        log_scales=torch.tensor([[-3.0, -3.0, -3.0], [-3.0, -3.0, -3.0]]),
        opacity=torch.tensor([0.0, 0.999]),
        sh=rgb_to_sh(torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]))[:, None],
    )
    distances = module.renderer_boundary_distances(
        gaussians,
        camera,
        torch.tensor([[0.5, 0.5]]),
    )
    assert distances["near_plane_0p05"] < 1e-5
    assert distances["trainer_opacity_logit_clamp_1e_4_1m1e_4"] == 0.0
    assert distances["renderer_alpha_cap_0p999_active_coordinates"] < 1e-7
    assert distances["sh_hard_floor_zero"] < 1e-7


def test_renderer_boundary_helper_covers_depth_support_visibility_and_quaternion():
    module = _load_harness()
    import torch

    from rtgs.core.camera import Camera
    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.core.sh import rgb_to_sh

    support_camera = Camera(1.0, 1.0, 0.5, 0.5, 10, 10, torch.eye(3), torch.zeros(3))
    tied = Gaussians3D(
        means=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        quats=torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        log_scales=torch.full((2, 3), -30.0),
        opacity=torch.full((2,), 0.5),
        sh=rgb_to_sh(torch.full((2, 3), 0.5))[:, None],
    )
    q12_x = 0.5 + math.sqrt(12.0 * 0.3)
    tied_distances = module.renderer_boundary_distances(
        tied,
        support_camera,
        torch.tensor([[0.5, 0.5], [q12_x, 0.5]]),
    )
    assert tied_distances["camera_depth_tie"] == 0.0
    assert tied_distances["student_hard_support_q12_all_score_coordinates"] < 1e-5
    assert tied_distances["quaternion_norm_zero"] == 0.0

    visibility_camera = Camera(
        10.0,
        10.0,
        5.0,
        0.5,
        10,
        10,
        torch.eye(3),
        torch.zeros(3),
    )
    radius = 3.0 * math.sqrt(0.3)
    boundary_u = 0.5 - radius
    boundary_model = Gaussians3D(
        means=torch.tensor([[(boundary_u - 5.0) / 10.0, 0.0, 1.0]]),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        log_scales=torch.full((1, 3), -30.0),
        opacity=torch.tensor([0.5]),
        sh=rgb_to_sh(torch.full((1, 3), 0.5))[:, None],
    )
    visibility_distances = module.renderer_boundary_distances(
        boundary_model,
        visibility_camera,
        torch.tensor([[0.5, 0.5]]),
    )
    assert visibility_distances["visibility_envelope"] < 1e-5


def test_domain_labels_follow_frozen_precedence(monkeypatch):
    module = _load_harness()
    development_seeds = (63841, 63842, 63843)
    monkeypatch.setattr(module, "OFFICIAL_SEEDS", development_seeds)
    uniform = [
        {
            "seed": seed,
            "checkpoints": [
                _checkpoint(0, 1.0, 1.0),
                _checkpoint(30, 0.75, 0.75),
                _checkpoint(60, 0.55, 0.55),
                _checkpoint(120, 0.40, 0.40),
            ],
        }
        for seed in development_seeds
    ]
    winning = [
        {
            "seed": seed,
            "checkpoints": [
                _checkpoint(0, 1.0, 1.0),
                _checkpoint(30, 0.55, 0.55),
                _checkpoint(60, 0.34, 0.34),
                _checkpoint(120, 0.20, 0.20),
            ],
        }
        for seed in development_seeds
    ]
    equal = [{"seed": item["seed"], "checkpoints": item["checkpoints"]} for item in uniform]
    assert module.classify_domain(uniform, winning, risk="pixel")["label"] == (
        "MATERIAL_SAMPLING_WIN"
    )
    assert module.classify_domain(uniform, equal, risk="pixel")["label"] == "NONINFERIOR"
    flat = [
        {
            "seed": seed,
            "checkpoints": [_checkpoint(step, 1.0, 1.0) for step in module.CHECKPOINTS],
        }
        for seed in development_seeds
    ]
    assert module.classify_domain(flat, flat, risk="pixel")["label"] == ("INCONCLUSIVE_TRAINER")
    harmful = [
        {
            "seed": seed,
            "checkpoints": [
                _checkpoint(0, 1.0, 1.0),
                _checkpoint(30, 0.90, 0.90),
                _checkpoint(60, 0.80, 0.80),
                _checkpoint(120, 0.70, 0.70),
            ],
        }
        for seed in development_seeds
    ]
    assert module.classify_domain(uniform, harmful, risk="pixel")["label"] == (
        "NEUTRAL_OR_NEGATIVE"
    )
    mixed_guard = [*winning[:2], harmful[2]]
    assert module.classify_domain(uniform, mixed_guard, risk="pixel")["label"] == (
        "NEUTRAL_OR_NEGATIVE"
    )


def test_result_is_recomputed_from_strict_raw_bindings(monkeypatch):
    module = _load_harness()
    development_seeds = (63841, 63842, 63843)
    monkeypatch.setattr(module, "OFFICIAL_SEEDS", development_seeds)
    seal = {
        "seal_file_sha256": "a" * 64,
        "seal_payload_sha256": "b" * 64,
        "source_aggregate": "c" * 64,
    }
    marker = "d" * 64
    records = []
    for seed in development_seeds:
        for arm in module.ARMS:
            risks = (1.0, 0.75, 0.55, 0.40)
            if arm.endswith("gaussian"):
                risks = (1.0, 0.55, 0.34, 0.20)
            records.append(_record(module, seed, arm, risks))
    raw = {
        "artifact_type": module.RAW_ARTIFACT_TYPE,
        "status": "PASS",
        "timestamp_utc": "2026-07-16T00:00:00+00:00",
        "preregistration_sha256": module.PREREGISTRATION_SHA256,
        "seal_file_sha256": seal["seal_file_sha256"],
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "attempt_sha256": marker,
        "source_aggregate": seal["source_aggregate"],
        "environment_fingerprint": {},
        "records": records,
    }
    result = module.result_from_raw(
        raw,
        raw_sha256="e" * 64,
        marker_sha256=marker,
        seal=seal,
    )
    assert result["status"] == "PASS"
    assert result["decision"] == "SAMPLING_WIN"
    assert result["domains"]["pixel"]["label"] == "MATERIAL_SAMPLING_WIN"
    assert result["domains"]["area"]["label"] == "MATERIAL_SAMPLING_WIN"
    raw["records"][0]["proposal_invariants"]["no_null_resampling"] = False
    with pytest.raises(module.ProtocolInvalid, match="proposal invariant failed"):
        module.result_from_raw(
            raw,
            raw_sha256="e" * 64,
            marker_sha256=marker,
            seal=seal,
        )


def test_audit_authorization_needs_literal_line_and_every_binding():
    module = _load_harness()
    bindings = {"preregistration": "a" * 64, "result": "b" * 64}
    text = (
        "Verdict: QUALIFIED\n\n"
        "CALIBRATED_INTEGRATION_AUTHORIZED: YES\n\n"
        f"preregistration={bindings['preregistration']}\nresult={bindings['result']}\n"
    )
    assert module.validate_audit_authorization(text, bindings=bindings) == "QUALIFIED"
    with pytest.raises(module.ProtocolInvalid, match="literal"):
        module.validate_audit_authorization(
            text.replace("CALIBRATED_INTEGRATION_AUTHORIZED: YES", "authorized"),
            bindings=bindings,
        )


def test_calibrated_mode_has_explicit_no_exhaustive_checkpoint_seam():
    from rtgs.optim.compact_trainer import CompactTrainConfig

    config = CompactTrainConfig(
        iterations=40,
        checkpoints=(0, 10, 20, 40),
        evaluate_checkpoint_risks=False,
    )
    assert config.evaluate_checkpoint_risks is False


def test_calibrated_preflight_rejects_cap_before_carve_index_allocation(monkeypatch):
    module = _load_harness()
    import torch

    import rtgs.lift.compact_carve as carve_module
    from rtgs.core.camera import Camera
    from rtgs.core.observation2d import GaussianObservationField
    from rtgs.data.reconstruction_inputs import ReconstructionInputs
    from rtgs.lift.compact_carve import CompactCarveConfig
    from rtgs.optim.compact_trainer import CompactTrainConfig

    field = GaussianObservationField(
        width=2,
        height=2,
        means=torch.tensor([[1.0, 1.0]]),
        log_scales=torch.zeros(1, 2),
        rotations=torch.zeros(1),
        colors=torch.full((1, 3), 0.5),
        amplitudes=torch.ones(1),
        fit_window=(0, 0, 2, 2),
        view_id="dev-view",
        n_init=1,
        provider="synthetic_fixture",
        producer_source_digest="a" * 64,
        fit_config_digest="b" * 64,
    )
    camera = Camera(1.0, 1.0, 1.0, 1.0, 2, 2, torch.eye(3), torch.zeros(3))
    inputs = ReconstructionInputs([field], [camera], ["dev-view"])
    allocated = False

    class ForbiddenIndex:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            nonlocal allocated
            allocated = True
            raise AssertionError("Carve allocated before preflight")

    monkeypatch.setattr(carve_module, "GaussianObservationIndex", ForbiddenIndex)
    train_config = CompactTrainConfig(max_fitted_pixels_per_view=1)
    carve_config = CompactCarveConfig(n_init_3d=1, min_views=1)
    with pytest.raises(ValueError, match="fitted pixels"):
        module._initialize_compact_carve_after_preflight(
            inputs,
            train_config,
            carve_config,
            bundle_path=None,
        )
    assert not allocated


def test_calibrated_checkpoint_evidence_is_measured_and_tamper_evident():
    module = _load_harness()
    callbacks = [
        {"step": step, "snapshot_sha256": f"hash-{step}", "detached": True}
        for step in (0, 10, 20, 40)
    ]
    history = {
        "checkpoints": [
            {"step": item["step"], "snapshot_sha256": item["snapshot_sha256"], "evaluation": None}
            for item in callbacks
        ],
        "checkpoint_risk_evaluation_enabled": False,
        "checkpoint_risk_evaluation_call_count": 0,
        "checkpoint_snapshot_count": 4,
        "checkpoint_callback_call_count": 4,
    }
    evidence = module.validate_calibrated_checkpoint_evidence(history, callbacks)
    assert evidence["evaluator_call_count"] == 0
    history["checkpoint_risk_evaluation_call_count"] = 1
    with pytest.raises(module.ProtocolInvalid, match="checkpoint evidence differs"):
        module.validate_calibrated_checkpoint_evidence(history, callbacks)


def test_rgb_denial_covers_pathlib_and_allows_non_dataset_reads(tmp_path):
    outside = tmp_path / "bundle-byte"
    outside.write_bytes(b"ok")
    script = f"""
import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location('compact_denial_child', {str(HARNESS_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
counters, _ = module._deny_rgb_access()
assert Path({str(outside)!r}).read_bytes() == b'ok'
print(json.dumps(counters, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    counters = json.loads(completed.stdout.strip().splitlines()[-1])
    assert counters["negative_control_dataset_denials"] == 1
    assert counters["negative_control_os_open_denials"] == 1
    assert counters["negative_control_pil_denials"] == 1
    assert counters["negative_control_calibrated_loader_denials"] == 3
    assert counters["negative_control_scene_data_denials"] == 3
    assert counters["dataset_open"] == 0
    module = _load_harness()
    module._validate_rgb_denial_counters(counters)


def test_source_path_set_rejects_added_startup_source(monkeypatch):
    module = _load_harness()
    original = module._reviewed_paths()
    expected = {str(path): "0" * 64 for path in (*original, module.IMPLEMENTATION_REVIEW)}
    monkeypatch.setattr(
        module,
        "_reviewed_paths",
        lambda: (*original, Path("sitecustomize.py")),
    )
    with pytest.raises(module.ProtocolInvalid, match="source path set"):
        module.verify_source_hashes(expected, "0" * 64)


def test_git_binding_ignores_artifact_status_but_not_tracked_state(monkeypatch):
    module = _load_harness()
    expected = {
        "revision": "a" * 40,
        "tracked_diff_sha256": "b" * 64,
        "status_sha256": "c" * 64,
        "status_lines": ["?? old-artifact"],
    }
    current = {
        **expected,
        "status_sha256": "d" * 64,
        "status_lines": ["?? newly-created-seal"],
    }
    monkeypatch.setattr(module, "_git_metadata", lambda: current)
    module._verify_git_binding(expected)
    current["tracked_diff_sha256"] = "e" * 64
    with pytest.raises(module.ProtocolInvalid, match="tracked_diff"):
        module._verify_git_binding(expected)


def test_collision_scans_include_directories_and_broken_symlinks(monkeypatch, tmp_path):
    module = _load_harness()
    results = tmp_path / "benchmarks" / "results"
    results.mkdir(parents=True)
    directory = results / "20260716_compact_point_training_RAW_directory"
    directory.mkdir()
    broken = results / "20260716_compact_point_training_RESULT_broken"
    broken.symlink_to(results / "missing-target")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    collisions = module._official_namespace_collisions(include_seal=True)
    assert directory in collisions
    assert broken in collisions

    run_link = tmp_path / "broken-run"
    run_link.symlink_to(tmp_path / "missing-run-target")
    monkeypatch.setattr(module, "RUN_DIR", run_link)
    assert module._calibrated_collisions() == [run_link]


def test_failure_payload_distinguishes_pre_and_post_raw_commit():
    module = _load_harness()
    seal = {
        "seal_file_sha256": "a" * 64,
        "seal_payload_sha256": "b" * 64,
        "source_aggregate": "c" * 64,
    }
    error = RuntimeError("injected")
    pre = module._failure_payload(
        artifact_type=module.RESULT_ARTIFACT_TYPE,
        error=error,
        seal=seal,
        marker_sha256="d" * 64,
        records=[],
        failure_phase="pre_raw_commit",
    )
    post = module._failure_payload(
        artifact_type=module.RESULT_ARTIFACT_TYPE,
        error=error,
        seal=seal,
        marker_sha256="d" * 64,
        records=[],
        failure_phase="post_raw_commit",
        raw_sha256="e" * 64,
    )
    assert pre["decision"] == "MECHANISM_FAIL"
    assert "decision" not in post
    assert post["raw_sha256"] == "e" * 64


def test_failure_transition_fault_injection_on_both_sides_of_raw_commit(monkeypatch, tmp_path):
    module = _load_harness()
    seal = {
        "seal_file_sha256": "a" * 64,
        "seal_payload_sha256": "b" * 64,
        "source_aggregate": "c" * 64,
    }
    marker = "d" * 64
    raw_path = tmp_path / "raw.json"
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(module, "RAW", raw_path)
    monkeypatch.setattr(module, "RESULT", result_path)

    phase, raw_digest = module._write_failure_artifacts(
        error=RuntimeError("injected-before-raw"),
        seal=seal,
        marker_sha256=marker,
        records=[],
        raw_commit_sha256=None,
    )
    assert phase == "pre_raw_commit"
    assert module.sha256_file(raw_path) == raw_digest
    pre_raw = module.strict_json_load(raw_path)
    pre_result = module.strict_json_load(result_path)
    assert pre_raw["status"] == pre_result["status"] == "FAIL"
    assert pre_raw["decision"] == pre_result["decision"] == "MECHANISM_FAIL"

    raw_path = tmp_path / "raw-post.json"
    result_path = tmp_path / "result-post.json"
    monkeypatch.setattr(module, "RAW", raw_path)
    monkeypatch.setattr(module, "RESULT", result_path)
    pass_raw = {"artifact_type": module.RAW_ARTIFACT_TYPE, "status": "PASS"}
    pass_digest = module._exclusive_json(raw_path, pass_raw)
    pass_bytes = raw_path.read_bytes()
    phase, raw_digest = module._write_failure_artifacts(
        error=RuntimeError("injected-after-raw"),
        seal=seal,
        marker_sha256=marker,
        records=[],
        raw_commit_sha256=pass_digest,
    )
    assert phase == "post_raw_commit"
    assert raw_digest == pass_digest
    assert raw_path.read_bytes() == pass_bytes
    post_result = module.strict_json_load(result_path)
    assert post_result["status"] == "FAIL"
    assert post_result["failure_phase"] == "post_raw_commit"
    assert post_result["raw_sha256"] == pass_digest
    assert "decision" not in post_result


def test_exclusive_json_removes_caught_partial_write(monkeypatch, tmp_path):
    module = _load_harness()
    destination = tmp_path / "partial.json"
    real_fsync = module.os.fsync
    calls = 0

    def fail_first_fsync(file_descriptor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected fsync failure")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        module._exclusive_json(destination, {"status": "PASS"})
    assert not destination.exists()


def test_bound_strict_reload_rejects_replaced_committed_bytes(tmp_path):
    module = _load_harness()
    path = tmp_path / "raw.json"
    digest = module._exclusive_json(path, {"status": "PASS", "records": []})
    path.write_text('{"status":"PASS","records":["changed"]}\n', encoding="utf-8")
    with pytest.raises(module.ProtocolInvalid, match="digest changed"):
        module.strict_json_load_bound(path, digest)


def test_viewer_probe_rejects_unrelated_listener_content():
    module = _load_harness()
    assert not module.viewer_http_response_is_valid(
        process_alive=True,
        listener_owned=True,
        status=200,
        body=b"unrelated server",
    )
    assert not module.viewer_http_response_is_valid(
        process_alive=True,
        listener_owned=False,
        status=200,
        body=b"<html><title>viser</title></html>",
    )
    assert module.viewer_http_response_is_valid(
        process_alive=True,
        listener_owned=True,
        status=200,
        body=b"<html><title>viser</title></html>",
    )


@pytest.mark.skipif(sys.platform != "linux", reason="listener ownership uses Linux /proc")
def test_linux_listener_binding_requires_the_exact_process():
    import socket

    module = _load_harness()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = int(listener.getsockname()[1])
        owned = module._linux_process_listener_binding(os.getpid(), "127.0.0.1", port)
        assert owned["owned"] is True
        assert owned["owned_target_listener_records"]
        wrong_interface = module._linux_process_listener_binding(os.getpid(), "127.0.0.2", port)
        assert wrong_interface["owned"] is False

        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            not_owned = module._linux_process_listener_binding(
                unrelated.pid,
                "127.0.0.1",
                port,
            )
            assert not_owned["owned"] is False
            assert not not_owned["owned_target_listener_records"]
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=10)


def test_exact_snapshot_worker_captures_preload_and_propagates_failure(
    monkeypatch,
    tmp_path,
):
    module = _load_harness()
    binding = _fake_preload_binding(module, tmp_path)
    preload = binding["requested_path"]
    abi_resolution = _fake_abi_resolution(module, binding)
    plan_sha256 = "a" * 64
    plan = {
        "configuration": {"ld_preload": preload, "ld_preload_binding": binding},
        "gsplat": {"version": "test"},
    }
    acquisition = {"plan_sha256": plan_sha256, "views": []}
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_sha256 = module._exclusive_json(acquisition_path, acquisition)
    initial = tmp_path / "initial.ply"
    final = tmp_path / "final.ply"
    initial.write_bytes(b"initial")
    final.write_bytes(b"final")
    initial_sha256 = module.sha256_file(initial)
    final_sha256 = module.sha256_file(final)
    monkeypatch.setattr(module, "CALIBRATED_ACQUISITION", acquisition_path)
    monkeypatch.setattr(module, "CALIBRATED_INIT", initial)
    monkeypatch.setattr(module, "CALIBRATED_FINAL", final)
    monkeypatch.setattr(module, "_load_calibrated_plan_and_seal", lambda _: (plan, {}))
    monkeypatch.setattr(
        module,
        "_render_programmatic_viewer_snapshots",
        lambda **_: {"rendered": True},
    )
    monkeypatch.setattr(
        module,
        "_default_namespace_abi_resolution",
        lambda _: dict(abi_resolution),
    )

    monkeypatch.setenv("LD_PRELOAD", preload)
    result_path = tmp_path / "worker.json"
    module._exact_snapshot_worker(
        plan_sha256,
        acquisition_sha256,
        initial_sha256,
        final_sha256,
        str(result_path),
    )
    result = module.strict_json_load(result_path)
    assert result["status"] == "PASS"
    assert result["ld_preload_at_startup"] == preload
    assert result["default_namespace_abi"] == abi_resolution
    assert result["exact_gsplat_snapshots"] == {"rendered": True}

    monkeypatch.setenv("LD_PRELOAD", "/wrong/libstdc++.so.6")
    failure_path = tmp_path / "worker-failure.json"
    with pytest.raises(module.ProtocolInvalid, match="did not start"):
        module._exact_snapshot_worker(
            plan_sha256,
            acquisition_sha256,
            initial_sha256,
            final_sha256,
            str(failure_path),
        )
    failure = module.strict_json_load(failure_path)
    assert failure["status"] == "FAIL"
    assert failure["ld_preload_at_startup"] == "/wrong/libstdc++.so.6"


def test_exact_snapshot_spawn_uses_fresh_worker_after_preload(monkeypatch, tmp_path):
    module = _load_harness()
    binding = _fake_preload_binding(module, tmp_path)
    preload = binding["requested_path"]
    plan = {"configuration": {"ld_preload": preload, "ld_preload_binding": binding}}
    acquisition = {"views": []}
    payload = {"status": "PASS", "worker": "bound"}
    captured = {}

    def fake_spawn(target, args, result_path, *, name, timeout):
        captured.update(
            target=target,
            args=args,
            result_path=result_path,
            name=name,
            timeout=timeout,
            preload=os.environ.get("LD_PRELOAD"),
        )
        return payload

    monkeypatch.setenv("LD_PRELOAD", preload)
    monkeypatch.setattr(module, "_spawn_file_worker", fake_spawn)
    monkeypatch.setattr(
        module,
        "_validate_exact_snapshot_worker_result",
        lambda worker_payload, **_: {"worker_payload": worker_payload},
    )
    result_path = tmp_path / "result.json"
    result = module._spawn_exact_snapshot_worker(
        plan=plan,
        acquisition=acquisition,
        plan_sha256="a" * 64,
        acquisition_sha256="b" * 64,
        initial_sha256="c" * 64,
        final_sha256="d" * 64,
        result_path=result_path,
    )
    assert captured == {
        "target": module._exact_snapshot_worker,
        "args": ("a" * 64, "b" * 64, "c" * 64, "d" * 64, str(result_path)),
        "result_path": result_path,
        "name": "compact-exact-gsplat-snapshots",
        "timeout": 1800.0,
        "preload": preload,
    }
    assert result == {"worker_payload": payload}

    monkeypatch.setattr(
        module,
        "_spawn_file_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(module.ProtocolInvalid("child failed")),
    )
    with pytest.raises(module.ProtocolInvalid, match="child failed"):
        module._spawn_exact_snapshot_worker(
            plan=plan,
            acquisition=acquisition,
            plan_sha256="a" * 64,
            acquisition_sha256="b" * 64,
            initial_sha256="c" * 64,
            final_sha256="d" * 64,
            result_path=result_path,
        )


def test_viewer_smoke_routes_native_snapshots_only_through_spawn_worker():
    module = _load_harness()
    tree = ast.parse(textwrap.dedent(inspect.getsource(module._viewer_smoke)))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_spawn_exact_snapshot_worker" in calls
    assert "_linux_process_listener_binding" in calls
    assert "_render_programmatic_viewer_snapshots" not in calls


def test_spawn_child_starts_with_parent_ld_preload(monkeypatch, tmp_path):
    module = _load_harness()
    preload = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    if not Path(preload).is_file():
        pytest.skip("host does not provide the calibrated LD_PRELOAD library")
    monkeypatch.setenv("LD_PRELOAD", preload)
    result_path = tmp_path / "child-environment.json"
    result = module._spawn_file_worker(
        _record_child_preload,
        (preload, str(result_path)),
        result_path,
        name="compact-test-preload-inheritance",
        timeout=30.0,
    )
    assert result == {"status": "PASS", "ld_preload": preload}


@pytest.mark.skipif(sys.platform != "linux", reason="versioned ABI lookup uses Linux dlvsym")
def test_spawn_child_resolves_frozen_versioned_abi_from_preload(monkeypatch, tmp_path):
    module = _load_harness()
    preload = Path(module.CALIBRATED_LD_PRELOAD)
    if not preload.is_file():
        pytest.skip("host does not provide the calibrated LD_PRELOAD library")
    binding = module._abi_preload_file_binding(str(preload))
    monkeypatch.setenv("LD_PRELOAD", str(preload))
    result_path = tmp_path / "child-abi.json"
    result = module._spawn_file_worker(
        _record_child_abi,
        (binding, str(result_path)),
        result_path,
        name="compact-test-versioned-abi-resolution",
        timeout=30.0,
    )
    module._verify_default_namespace_abi_resolution(
        result["resolution"],
        binding=binding,
    )
    assert result["resolution"]["required_symbol"] == "__cxa_call_terminate"
    assert result["resolution"]["required_version"] == "CXXABI_1.3.15"


def test_preload_binding_detects_symlink_and_byte_drift(tmp_path):
    module = _load_harness()
    binding = _fake_preload_binding(module, tmp_path)
    assert module._verify_abi_preload_file_binding(binding) == binding
    assert binding == {
        "requested_path": str(tmp_path / "libstdc++.so.6"),
        "resolved_path": str(tmp_path / "libstdc++.so.6.0.test"),
        "sha256": module.sha256_file(tmp_path / "libstdc++.so.6.0.test"),
        "required_symbol": "__cxa_call_terminate",
        "required_version": "CXXABI_1.3.15",
    }

    alternate = tmp_path / "libstdc++.so.6.0.alternate"
    alternate.write_bytes(b"alternate runtime")
    requested = Path(binding["requested_path"])
    requested.unlink()
    requested.symlink_to(alternate.name)
    with pytest.raises(module.ProtocolInvalid, match="symlink target changed"):
        module._verify_abi_preload_file_binding(binding)

    requested.unlink()
    requested.symlink_to(Path(binding["resolved_path"]).name)
    Path(binding["resolved_path"]).write_bytes(b"changed runtime")
    with pytest.raises(module.ProtocolInvalid, match="bytes changed"):
        module._verify_abi_preload_file_binding(binding)


def test_calibrated_configuration_embeds_versioned_abi_binding():
    module = _load_harness()
    preload = Path(module.CALIBRATED_LD_PRELOAD)
    if not preload.is_file():
        pytest.skip("host does not provide the calibrated LD_PRELOAD library")
    configuration = module._calibrated_configuration()
    assert configuration["ld_preload"] == str(preload)
    assert configuration["ld_preload_binding"] == module._abi_preload_file_binding(str(preload))


def test_exact_snapshot_worker_result_is_strictly_bound(monkeypatch, tmp_path):
    module = _load_harness()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CALIBRATED_INIT", tmp_path / "initial.ply")
    monkeypatch.setattr(module, "CALIBRATED_FINAL", tmp_path / "final.ply")
    monkeypatch.setattr(module, "CALIBRATED_SNAPSHOTS", tmp_path / "snapshots")
    module.CALIBRATED_SNAPSHOTS.mkdir()
    module.CALIBRATED_INIT.write_bytes(b"initial")
    module.CALIBRATED_FINAL.write_bytes(b"final")
    initial_sha256 = module.sha256_file(module.CALIBRATED_INIT)
    final_sha256 = module.sha256_file(module.CALIBRATED_FINAL)
    extension = tmp_path / "gsplat_cuda.so"
    extension.write_bytes(b"extension")
    camera = {"width": 5328, "height": 4608}
    plan_sha256 = "a" * 64
    acquisition_sha256 = "b" * 64
    binding = _fake_preload_binding(module, tmp_path)
    abi_resolution = _fake_abi_resolution(module, binding)
    plan = {
        "configuration": {
            "ld_preload": binding["requested_path"],
            "ld_preload_binding": binding,
        },
        "gsplat": {"version": "test"},
    }
    acquisition = {"views": [{"camera_id": module.CALIBRATED_VIEWS[0], "camera": camera}]}
    models = {}
    for name, ply_sha256 in (("initial", initial_sha256), ("final", final_sha256)):
        snapshot = module.CALIBRATED_SNAPSHOTS / (f"{name}_{module.CALIBRATED_VIEWS[0]}_gsplat.png")
        snapshot.write_bytes(name.encode("ascii"))
        models[name] = {
            "path": snapshot.relative_to(tmp_path).as_posix(),
            "sha256": module.sha256_file(snapshot),
            "bytes": snapshot.stat().st_size,
            "dimensions": [5328, 4608],
            "color_tensor_sha256": "e" * 64,
            "backend": "rtgs.render.gsplat_backend.GsplatRasterizer",
            "device": "cuda:0",
            "gaussians_sha256": ply_sha256,
            "n_gaussians": 835,
            "packed": False,
            "antialiased": False,
        }
    exact = {
        "camera_id": module.CALIBRATED_VIEWS[0],
        "camera": camera,
        "models": models,
        "gsplat_package": plan["gsplat"],
        "loaded_cuda_extension": {
            "path": str(extension.resolve()),
            "sha256": module.sha256_file(extension),
        },
    }
    payload = {
        "artifact_type": "compact_point_training_exact_snapshot_worker_v2",
        "status": "PASS",
        "plan_sha256": plan_sha256,
        "acquisition_sha256": acquisition_sha256,
        "gaussians_init_sha256": initial_sha256,
        "gaussians_final_sha256": final_sha256,
        "ld_preload_at_startup": plan["configuration"]["ld_preload"],
        "default_namespace_abi": abi_resolution,
        "exact_gsplat_snapshots": exact,
    }
    validated = module._validate_exact_snapshot_worker_result(
        payload,
        plan=plan,
        acquisition=acquisition,
        plan_sha256=plan_sha256,
        acquisition_sha256=acquisition_sha256,
        initial_sha256=initial_sha256,
        final_sha256=final_sha256,
    )
    assert validated == exact
    abi_resolution["required_version"] = "CXXABI_1.3.14"
    with pytest.raises(module.ProtocolInvalid, match="differs from the frozen binding"):
        module._validate_exact_snapshot_worker_result(
            payload,
            plan=plan,
            acquisition=acquisition,
            plan_sha256=plan_sha256,
            acquisition_sha256=acquisition_sha256,
            initial_sha256=initial_sha256,
            final_sha256=final_sha256,
        )
    abi_resolution["required_version"] = module.REQUIRED_CXXABI_VERSION
    payload["gaussians_final_sha256"] = "f" * 64
    with pytest.raises(module.ProtocolInvalid, match="wrong artifact bindings"):
        module._validate_exact_snapshot_worker_result(
            payload,
            plan=plan,
            acquisition=acquisition,
            plan_sha256=plan_sha256,
            acquisition_sha256=acquisition_sha256,
            initial_sha256=initial_sha256,
            final_sha256=final_sha256,
        )
    payload["gaussians_final_sha256"] = final_sha256
    models["final"]["n_gaussians"] = 834
    with pytest.raises(module.ProtocolInvalid, match="final artifact binding differs"):
        module._validate_exact_snapshot_worker_result(
            payload,
            plan=plan,
            acquisition=acquisition,
            plan_sha256=plan_sha256,
            acquisition_sha256=acquisition_sha256,
            initial_sha256=initial_sha256,
            final_sha256=final_sha256,
        )
    models["final"]["n_gaussians"] = 835
    models["final"]["color_tensor_sha256"] = "not-a-sha256"
    with pytest.raises(module.ProtocolInvalid, match="final artifact binding differs"):
        module._validate_exact_snapshot_worker_result(
            payload,
            plan=plan,
            acquisition=acquisition,
            plan_sha256=plan_sha256,
            acquisition_sha256=acquisition_sha256,
            initial_sha256=initial_sha256,
            final_sha256=final_sha256,
        )


def test_structsplat_provider_digest_matches_export_adapter():
    pytest.importorskip("structsplat")
    from structsplat.config import FitConfig as StructSplatFitConfig

    from rtgs.image2gs.structsplat_backend import _structsplat_provenance

    module = _load_harness()
    _, expected, _ = _structsplat_provenance(StructSplatFitConfig())
    assert module._external_structsplat_binding()["provider_source_digest"] == expected
