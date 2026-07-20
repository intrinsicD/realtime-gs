"""Synthetic, TUM-value-free checks for the sealed RGB-D validity harness."""

from __future__ import annotations

import json
import math

import pytest
import torch
from benchmarks.tum_rgbd_oriented_validity import (
    CONFIG_ID,
    EXPERIMENT,
    N_GRID_PER_VIEW,
    PREREGISTRATION,
    SOURCES,
    AssociatedFrame,
    SelectedFrame,
    TargetBundle,
    TimedPath,
    TimedPose,
    TumDepthBackend,
    _associate,
    _audit_targets,
    _camera_from_pose,
    _create_confirmatory_seal_once,
    _gate,
    _interpolate_pose,
    _linear_quantile,
    _parse_timed_paths,
    _run_audit,
    _select_frames,
    _sha256_file,
    _target_composite_hash,
    _thresholds,
    _validate_payload_access,
    _validate_threshold_manifest,
)


def _pose(timestamp_ns: int, center=(0.0, 0.0, 0.0), angle_deg: float = 0.0) -> TimedPose:
    half = math.radians(angle_deg) / 2
    return TimedPose(
        timestamp_ns=timestamp_ns,
        timestamp_token=str(timestamp_ns),
        center=torch.tensor(center, dtype=torch.float64),
        quaternion_xyzw=torch.tensor(
            [0.0, math.sin(half), 0.0, math.cos(half)], dtype=torch.float64
        ),
    )


def _frame(index: int, pose: TimedPose) -> AssociatedFrame:
    rgb = TimedPath(index * 1_000_000, str(index), f"rgb/{index}.png")
    depth = TimedPath(index * 1_000_000 + 1, str(index), f"depth/{index}.png")
    return AssociatedFrame(rgb, depth, pose, rgb_depth_delta_ns=1)


def test_association_is_strict_globally_greedy_and_unique():
    left = [TimedPath(0, "0", "a"), TimedPath(10, "10", "b")]
    right = [TimedPath(6, "6", "x"), TimedPath(20_000_000, "boundary", "y")]
    matches = _associate(left, right)
    assert [(a.timestamp_ns, b.timestamp_ns) for a, b in matches] == [(10, 6)]
    assert _associate([TimedPath(0, "0", "a")], [right[1]]) == []


def test_manifest_payload_paths_must_be_safe_and_unique():
    with pytest.raises(ValueError, match="duplicate manifest payload path"):
        _parse_timed_paths("0 depth/shared.png\n1 depth/shared.png\n")
    with pytest.raises(ValueError, match="unsafe archive-relative path"):
        _parse_timed_paths("0 ../escape.png\n")


def test_pose_interpolation_uses_shortest_slerp_and_exact_endpoints():
    poses = [_pose(0, angle_deg=0), _pose(10_000_000, center=(1, 0, 0), angle_deg=90)]
    middle = _interpolate_pose(poses, 5_000_000)
    assert middle is not None
    assert torch.allclose(middle.center, torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64))
    expected = torch.tensor(
        [0.0, math.sin(math.radians(22.5)), 0.0, math.cos(math.radians(22.5))],
        dtype=torch.float64,
    )
    assert torch.allclose(middle.quaternion_xyzw, expected, atol=1e-12, rtol=0)
    exact = _interpolate_pose(poses, 0)
    assert exact is not None
    assert torch.equal(exact.quaternion_xyzw, poses[0].quaternion_xyzw)
    assert _interpolate_pose(poses, -1) is None


def test_pose_only_keyframes_half_up_select_exact_roles():
    frames = [_frame(index, _pose(index, center=(0.09 * index, 0.0, 0.0))) for index in range(65)]
    selected, diagnostics = _select_frames(frames)
    assert len(selected) == 64
    assert diagnostics["uniform_source_indices"][0] == 0
    assert diagnostics["uniform_source_indices"][-1] == 64
    assert [item.role for item in selected].count("T") == 48
    assert [item.role for item in selected].count("V") == 8
    assert [item.role for item in selected].count("H") == 8
    assert [item.role_ordinal for item in selected if item.role == "T"] == list(range(48))


def test_camera_pose_conversion_preserves_center_and_projection():
    pose = _pose(0, center=(1.0, -2.0, 3.0), angle_deg=35.0)
    camera = _camera_from_pose(pose)
    assert camera.R.dtype == torch.float64
    assert camera.t.dtype == torch.float64
    assert torch.allclose(camera.position, pose.center, atol=1e-12, rtol=0)


def test_linear_quantile_averages_even_median_and_interpolates_tails():
    values = torch.tensor([4.0, 1.0, 3.0, 2.0], dtype=torch.float64)
    assert _linear_quantile(values, 0.5) == 2.5
    assert math.isclose(_linear_quantile(values, 0.1), 1.3, abs_tol=1e-15)
    assert math.isclose(_linear_quantile(values, 0.9), 3.7, abs_tol=1e-15)


def test_audit_keeps_depth_pairs_when_oriented_stencil_is_invalid():
    pose = _pose(0)
    camera = _camera_from_pose(pose)
    depths = {}
    cameras = {}
    validation = []
    for ordinal in range(2):
        frame = _frame(ordinal, pose)
        selected = SelectedFrame(
            frame, selected_ordinal=3 + 8 * ordinal, role="V", role_ordinal=ordinal
        )
        validation.append(selected)
        depth = torch.full((480, 640), 2.0)
        # Break only the normal stencil at target 0; its center depth remains valid.
        depth[100, 102] = 2.2
        depths[frame.view_id] = depth
        cameras[frame.view_id] = camera

    pixels = torch.tensor([[100.5, 100.5], [200.5, 100.5]])
    target_depth = torch.tensor([1.0, 2.0])
    points = camera.unproject(pixels, target_depth)
    targets = TargetBundle(
        points_world=points,
        normals_world=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        source_t_ordinals=torch.tensor([0, 0]),
        source_selected_ordinals=torch.tensor([0, 0]),
        rows=torch.tensor([100, 100]),
        columns=torch.tensor([100, 200]),
        per_t_counts=(2,),
        eligibility_mask=torch.zeros((1, N_GRID_PER_VIEW), dtype=torch.bool),
    )
    targets.eligibility_mask[0, :2] = True
    metrics, diagnostics, raw = _audit_targets(
        targets, validation, cameras, depths, TumDepthBackend(depths)
    )
    assert diagnostics["depth_valid_pairs"] == 4
    assert diagnostics["oriented_valid_pairs"] == 2
    assert diagnostics["depth_supported_targets"] == 2
    assert diagnostics["oriented_supported_targets"] == 1
    assert metrics["S"] == 0.5
    assert metrics["F"] == 0.5
    assert metrics["R90"] == 0.0
    assert metrics["C50"] == 1.0
    assert int(raw["depth_pair_mask"].sum()) == 4
    assert int(raw["oriented_pair_mask"].sum()) == 2
    assert raw["depth_residuals"].dtype == torch.float64
    assert raw["surface_residuals"].dtype == torch.float64
    assert diagnostics["ineligible_targets"] == 48 * N_GRID_PER_VIEW - 2
    assert diagnostics["per_construction_view"][0]["depth_valid_pairs"] == 4
    assert diagnostics["per_construction_view"][0]["oriented_valid_pairs"] == 2
    for view in diagnostics["per_validation_view"]:
        assert (
            view["out_of_frame_targets"]
            + view["construction_invisible_targets"]
            + view["invalid_center_depth_pairs"]
            + view["depth_valid_pairs"]
            == 2
        )


def test_threshold_formulas_and_gate_directions_are_exact():
    development = {
        "A": 0.8,
        "A_min": 0.5,
        "S": 0.6,
        "S_10": 0.4,
        "R90": 0.01,
        "D90": 0.04,
        "C50": 0.9,
        "C10": 0.7,
        "F": 0.04,
    }
    thresholds = _thresholds(development)
    assert thresholds == {
        "A": 0.5599999999999999,
        "A_min": 0.25,
        "S": 0.36,
        "S_10": 0.2,
        "R90": 0.02,
        "D90": 0.05,
        "C50": 0.75,
        "C10": 0.5499999999999999,
        "F": 0.06999999999999999,
    }
    assert _gate(thresholds, thresholds)["all"]
    assert not _gate({**thresholds, "A": thresholds["A"] - 1e-6}, thresholds)["A"]
    assert not _gate({**thresholds, "R90": thresholds["R90"] + 1e-6}, thresholds)["R90"]


def test_target_composite_hash_binds_tensors_manifest_and_config():
    baseline = _target_composite_hash("tensor", "manifest")
    assert baseline == _target_composite_hash("tensor", "manifest")
    assert baseline != _target_composite_hash("tensor-edited", "manifest")
    assert baseline != _target_composite_hash("tensor", "manifest-edited")
    assert baseline != _target_composite_hash("tensor", "manifest", config_id="edited")


def test_payload_audit_requires_attempted_decoded_and_allowlist_identity():
    class FakeArchive:
        attempted_depth_members = ["a", "b"]
        decoded_depth_members = ["a", "b"]

    archive = FakeArchive()
    _validate_payload_access(archive, {"a", "b"})  # type: ignore[arg-type]
    archive.decoded_depth_members = ["a"]
    with pytest.raises(AssertionError, match="attempted/decoded"):
        _validate_payload_access(archive, {"a", "b"})  # type: ignore[arg-type]


def test_confirmatory_seal_is_atomic_and_once_only(tmp_path):
    path = tmp_path / "seal.json"
    _create_confirmatory_seal_once(path, {"status": "started"})
    assert json.loads(path.read_text()) == {"status": "started"}
    with pytest.raises(FileExistsError):
        _create_confirmatory_seal_once(path, {"status": "second"})


def test_direct_audit_api_fails_closed_on_attempt_callback(tmp_path):
    missing = tmp_path / "missing.tgz"
    with pytest.raises(ValueError, match="requires the atomic attempt-seal callback"):
        _run_audit(missing, "confirmatory")
    with pytest.raises(ValueError, match="must not receive"):
        _run_audit(missing, "development", before_payload_decode=lambda: None)


def test_threshold_manifest_recomputes_values_and_rejects_tampering(tmp_path):
    metrics = {
        "A": 0.8,
        "A_min": 0.5,
        "S": 0.6,
        "S_10": 0.4,
        "R90": 0.01,
        "D90": 0.02,
        "C50": 0.9,
        "C10": 0.7,
        "F": 0.04,
    }
    implementation = {
        "aggregate_sha256": "implementation",
        "files": {str(PREREGISTRATION): "preregistration"},
    }
    development = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "phase": "development",
        "config_id": CONFIG_ID,
        "source": {
            "sha256": SOURCES["development"]["sha256"],
            "archive_sha256": SOURCES["development"]["sha256"],
        },
        "implementation": implementation,
        "metrics": metrics,
        "derived_confirmatory_thresholds": _thresholds(metrics),
    }
    development_path = tmp_path / "development.json"
    development_path.write_text(json.dumps(development))
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "config_id": CONFIG_ID,
        "development_archive_sha256": SOURCES["development"]["sha256"],
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
        "preregistration_sha256": implementation["files"][str(PREREGISTRATION)],
        "development_artifact": str(development_path),
        "development_artifact_sha256": _sha256_file(development_path),
        "development_metrics": metrics,
        "thresholds": _thresholds(metrics),
    }
    manifest_path = tmp_path / "thresholds.json"
    manifest_path.write_text(json.dumps(manifest))
    _, validated, validated_sha256 = _validate_threshold_manifest(
        manifest_path, tmp_path, implementation
    )
    assert validated == _thresholds(metrics)
    assert validated_sha256 == _sha256_file(manifest_path)

    manifest["thresholds"]["R90"] += 1e-6
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mechanical formulas"):
        _validate_threshold_manifest(manifest_path, tmp_path, implementation)

    manifest_path.write_text('{"schema_version":1,"experiment":NaN}')
    with pytest.raises(ValueError, match="nonfinite JSON"):
        _validate_threshold_manifest(manifest_path, tmp_path, implementation)
