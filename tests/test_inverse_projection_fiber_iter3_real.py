"""Focused CPU tests for the staged Iteration 3 calibrated interaction.

No test opens the repository's frozen real payload or executes an official optimizer outcome.
Synthetic compact bundles exercise access staging, exact camera-z intervals, frozen selection,
validation-camera use, and evidence completeness.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from benchmarks import inverse_projection_fiber_iter3_real as real

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.fiber_correspondence import (
    CorrespondencePlan,
    FiberFitConfig,
    ObservationGaussians,
)
from rtgs.lift.inverse_projection_fiber import InverseProjectionFiber


def _camera(index: int = 0) -> Camera:
    position = torch.tensor([0.15 * index, -0.05 * index, 0.0])
    return Camera(
        fx=60.0,
        fy=62.0,
        cx=40.0,
        cy=40.0,
        width=80,
        height=80,
        R=torch.eye(3),
        t=-position,
    )


def _field(view_id: str, *, count: int = 640) -> GaussianObservationField:
    if count != 640:
        raise ValueError("focused fixture intentionally matches the frozen cardinality")
    means = []
    for cell_y in range(8):
        for cell_x in range(8):
            for local in range(10):
                means.append(
                    [
                        cell_x * 10.0 + 1.0 + 0.7 * local,
                        cell_y * 10.0 + 1.0 + 0.6 * (local % 5),
                    ]
                )
    return GaussianObservationField(
        width=80,
        height=80,
        means=torch.tensor(means, dtype=torch.float32),
        log_scales=torch.zeros(count, 2),
        rotations=torch.zeros(count),
        colors=torch.linspace(0.1, 0.9, count)[:, None].expand(-1, 3).clone(),
        amplitudes=torch.ones(count),
        blend_mode="normalized",
        aa_dilation=0.3,
        view_id=view_id,
        fit_window=(0, 0, 80, 80),
        n_init=count,
        provider="synthetic_fixture",
        producer_version="focused-test",
    )


def _inputs(view_ids: tuple[str, ...]) -> ReconstructionInputs:
    return ReconstructionInputs(
        observations=[_field(view_id) for view_id in view_ids],
        cameras=[_camera(index) for index in range(len(view_ids))],
        view_names=list(view_ids),
        name="iter3-real-focused",
    )


@pytest.fixture(scope="module")
def compact_bundle(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("iter3_real_bundle") / "reconstruction_inputs"
    _inputs(real.EXPECTED_BUNDLE_IDS).save(directory)
    return directory


def _wide_bounds() -> real.Bounds:
    center = torch.tensor([0.0, 0.0, 2.5], dtype=torch.float64)
    lower = torch.tensor([-100.0, -100.0, 1.0], dtype=torch.float64)
    upper = torch.tensor([100.0, 100.0, 4.0], dtype=torch.float64)
    return real.Bounds(
        center=center,
        extent=200.0,
        lower=lower,
        upper=upper,
        semantic_sha256="0" * 64,
    )


def test_official_config_matches_frozen_schedule_and_semantics():
    config = real.official_fit_config()
    assert len(config.temperatures) == len(config.residual_variances) == 20
    assert config.temperatures[0] == 2.0
    assert config.temperatures[-1] == 0.10
    assert config.residual_variances[0] == 1.0
    assert config.residual_variances[-1] == 0.05
    assert config.geometry_steps == 2
    assert config.learning_rate == 0.025
    assert config.dustbin_cost == 4.0
    assert config.sinkhorn_iterations == 50
    assert config.sinkhorn_tolerance == 0.0
    assert config.marginal_penalty == 1.0


def test_exact_world_ray_scalar_reprojects_as_camera_z():
    camera = Camera(
        fx=73.0,
        fy=69.0,
        cx=31.0,
        cy=29.0,
        width=64,
        height=60,
        R=torch.tensor(
            [
                [1.0, 1.0e-4, 0.0],
                [0.0, 1.0, -2.0e-4],
                [1.0e-4, 0.0, 1.0],
            ]
        ),
        t=torch.tensor([0.3, -0.2, 0.4]),
    )
    xy = torch.tensor([[12.25, 24.75], [45.5, 37.25]], dtype=torch.float64)
    origins, directions = real._exact_camera_z_world_rays(camera, xy)
    depths = torch.tensor([0.3, 1.7], dtype=torch.float64)
    world = origins + depths[:, None] * directions
    camera_points = camera.world_to_cam(world)
    projected, projected_depths = camera.project(world)

    assert torch.allclose(camera_points[:, 2], depths, atol=1e-12, rtol=0)
    assert torch.allclose(projected_depths, depths, atol=1e-12, rtol=0)
    assert torch.allclose(projected, xy, atol=1e-11, rtol=0)


def test_ray_box_intervals_use_frozen_near_and_reject_pre_near_cube():
    camera = _camera()
    xy = torch.tensor([[40.0, 40.0]], dtype=torch.float64)
    near, far = real.ray_box_camera_z_intervals(
        camera,
        xy,
        torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0, 3.0], dtype=torch.float64),
    )
    assert torch.equal(near, torch.tensor([1.0], dtype=torch.float64))
    assert torch.equal(far, torch.tensor([3.0], dtype=torch.float64))

    invalid_near, invalid_far, valid = real._ray_box_camera_z_intervals_with_validity(
        camera,
        xy,
        torch.tensor([-1.0, -1.0, 0.01], dtype=torch.float64),
        torch.tensor([1.0, 1.0, 0.04], dtype=torch.float64),
    )
    assert torch.equal(invalid_near, torch.tensor([real.NEAR_DEPTH], dtype=torch.float64))
    assert float(invalid_far[0]) == pytest.approx(0.04)
    assert not bool(valid[0])
    with pytest.raises(RuntimeError, match="invalid or empty"):
        real.ray_box_camera_z_intervals(
            camera,
            xy,
            torch.tensor([-1.0, -1.0, 0.01], dtype=torch.float64),
            torch.tensor([1.0, 1.0, 0.04], dtype=torch.float64),
        )


def test_grid_selection_is_two_per_cell_stable_and_eligibility_first():
    field = _field(real.DEVELOPMENT_IDS[0])
    eligible = torch.ones(field.n, dtype=torch.bool)
    eligible[0] = False
    selected = real.select_grid_anchors(field, eligible=eligible)
    assert selected.shape == (128,)
    assert int(torch.unique(selected).numel()) == 128
    assert selected[:2].tolist() == [1, 2]
    for cell in range(1, 64):
        assert selected[2 * cell : 2 * cell + 2].tolist() == [
            cell * 10,
            cell * 10 + 1,
        ]
    assert bool(eligible[selected].all())

    too_few = torch.zeros(field.n, dtype=torch.bool)
    too_few[:127] = True
    with pytest.raises(RuntimeError, match="fewer than 128"):
        real.select_grid_anchors(field, eligible=too_few)


def test_problem_has_exact_640_fibers_and_retains_all_ray_addendum_evidence():
    inputs = _inputs(real.DEVELOPMENT_IDS)
    problem = real.build_real_problem(inputs, _wide_bounds())
    assert problem.anchors.source_view_indices.shape == (640,)
    assert problem.anchors.source_component_indices.shape == (640,)
    assert torch.bincount(problem.anchors.source_view_indices).tolist() == [128] * 5
    assert len(problem.anchors.all_ray_valid) == 5
    assert all(value.shape == (640,) for value in problem.anchors.all_ray_valid)
    assert all(value.shape == (640,) for value in problem.anchors.all_depth_lower)
    assert all(value.shape == (640,) for value in problem.anchors.all_depth_upper)
    assert all(bool(value.all()) for value in problem.anchors.all_ray_valid)

    arrays = real._input_arrays(problem)
    assert torch.equal(arrays["near_depth"], torch.tensor([0.05], dtype=torch.float64))
    for view_index in range(5):
        prefix = f"view_{view_index:02d}"
        assert arrays[f"{prefix}_all_ray_valid"].shape == (640,)
        assert arrays[f"{prefix}_all_depth_lower"].shape == (640,)
        assert arrays[f"{prefix}_all_depth_upper"].shape == (640,)
        assert arrays[f"{prefix}_selected_components"].shape == (128,)

    summary = real._anchor_evidence_summary(
        problem,
        {"semantic_sha256": "a" * 64},
    )
    assert summary["evidence_npz_semantic_sha256"] == "a" * 64
    assert all(item["valid_interval_fraction"] == 1.0 for item in summary["per_view"])
    assert all(item["non_intersecting_target_fraction"] == 0.0 for item in summary["per_view"])

    fiber = real.fresh_fiber(problem)
    source_means, source_covariances, source_depths = fiber.source_projection()
    assert torch.allclose(source_means, fiber.source_means2d, atol=1e-8, rtol=0)
    relative = (source_covariances - fiber.source_covariances2d).flatten(1).norm(
        dim=-1
    ) / fiber.source_covariances2d.flatten(1).norm(dim=-1)
    assert float(relative.detach().max()) <= 1e-8
    assert bool(((source_depths > fiber.depth_lower) & (source_depths < fiber.depth_upper)).all())


def test_symmetric_set_cost_requires_and_uses_explicit_validation_cameras():
    development_cameras = tuple(_camera(index) for index in range(5))
    fiber = InverseProjectionFiber(
        cameras=development_cameras,
        source_view_indices=torch.tensor([0], dtype=torch.long),
        source_component_indices=torch.tensor([0], dtype=torch.long),
        source_means2d=torch.tensor([[40.0, 40.0]], dtype=torch.float64),
        source_covariances2d=torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float64),
        initial_depths=torch.tensor([2.0], dtype=torch.float64),
        depth_lower=1.0,
        depth_upper=3.0,
        dilation=0.3,
    )
    validation_cameras = (
        Camera(60, 62, 40, 40, 80, 80, torch.eye(3), torch.tensor([-0.4, 0.0, 0.0])),
        Camera(60, 62, 40, 40, 80, 80, torch.eye(3), torch.tensor([0.3, 0.0, 0.0])),
    )
    observations = []
    for camera in validation_cameras:
        projected = fiber.project(camera)
        observations.append(
            ObservationGaussians(
                means=projected.means2d.detach(),
                covariances=projected.covariances2d.detach(),
                capacities=torch.ones(1, dtype=torch.float64),
                dilation=0.3,
            )
        )
    with pytest.raises(ValueError, match="equal length"):
        real.symmetric_set_costs(fiber, tuple(observations))
    costs = real.symmetric_set_costs(
        fiber,
        tuple(observations),
        cameras=validation_cameras,
    )
    assert costs == pytest.approx((0.0, 0.0), abs=1e-12)


def _source_excluded_test_plan(source_index: int) -> CorrespondencePlan:
    active_index = 1 - source_index
    real_mass = torch.zeros(2, 1, dtype=torch.float64)
    track_dustbin_mass = torch.zeros(2, dtype=torch.float64)
    track_capacities = torch.zeros(2, dtype=torch.float64)
    real_mass[active_index, 0] = 0.3
    track_dustbin_mass[active_index] = 0.1
    track_capacities[active_index] = 1.0
    return CorrespondencePlan(
        real_mass=real_mass,
        track_dustbin_mass=track_dustbin_mass,
        observation_dustbin_mass=torch.tensor([0.2], dtype=torch.float64),
        dustbin_dustbin_mass=torch.tensor(0.4, dtype=torch.float64),
        track_capacities=track_capacities,
        observation_capacities=torch.ones(1, dtype=torch.float64),
        method="unbalanced_sinkhorn",
        iterations=50,
        fixed_point_residual=0.025,
    )


def test_uot_marginal_diagnostics_save_declared_realized_errors_and_retention():
    plan = _source_excluded_test_plan(0)
    summary, arrays = real._uot_marginal_diagnostics(plan)

    assert torch.equal(
        arrays["declared_augmented_row_target"],
        torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64),
    )
    assert torch.equal(
        arrays["declared_augmented_column_target"],
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    assert torch.allclose(
        arrays["realized_augmented_row_total"],
        torch.tensor([0.0, 0.4, 0.6], dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )
    assert torch.allclose(
        arrays["realized_augmented_column_total"],
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )
    assert summary["fixed_point_residual"] == pytest.approx(0.025)
    assert summary["max_absolute_marginal_error"] == pytest.approx(0.1)
    assert summary["max_relative_marginal_error"] == pytest.approx(0.2)
    assert summary["total_augmented_retention"] == pytest.approx(1.0)

    persisted = real._plan_arrays((plan,))
    for name in (
        "declared_augmented_row_target",
        "declared_augmented_column_target",
        "realized_augmented_row_total",
        "realized_augmented_column_total",
        "augmented_row_absolute_error",
        "augmented_column_absolute_error",
        "augmented_row_relative_error",
        "augmented_column_relative_error",
        "fixed_point_residual",
        "total_augmented_retention",
    ):
        assert f"view_00_{name}" in persisted

    missing_residual = CorrespondencePlan(
        real_mass=plan.real_mass,
        track_dustbin_mass=plan.track_dustbin_mass,
        observation_dustbin_mass=plan.observation_dustbin_mass,
        dustbin_dustbin_mass=plan.dustbin_dustbin_mass,
        track_capacities=plan.track_capacities,
        observation_capacities=plan.observation_capacities,
        method=plan.method,
        iterations=plan.iterations,
    )
    with pytest.raises(RuntimeError, match="fixed-point residual"):
        real._uot_marginal_diagnostics(missing_residual)


def test_final_arm_validation_accepts_structural_uot_and_rejects_invalid_state():
    cameras = (_camera(0), _camera(1))
    fiber = InverseProjectionFiber(
        cameras=cameras,
        source_view_indices=torch.tensor([0, 1], dtype=torch.long),
        source_component_indices=torch.tensor([0, 0], dtype=torch.long),
        source_means2d=torch.tensor([[40.0, 40.0], [40.0, 40.0]], dtype=torch.float64),
        source_covariances2d=torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 2.0]]],
            dtype=torch.float64,
        ),
        initial_depths=torch.tensor([2.0, 2.0], dtype=torch.float64),
        depth_lower=1.0,
        depth_upper=3.0,
        dilation=0.3,
    )
    observations = []
    for view_index, camera in enumerate(cameras):
        projected = fiber.project(camera)
        active_index = 1 - view_index
        observations.append(
            ObservationGaussians(
                means=projected.means2d[active_index : active_index + 1].detach(),
                covariances=projected.covariances2d[active_index : active_index + 1].detach(),
                capacities=torch.ones(1, dtype=torch.float64),
                dilation=0.3,
            )
        )
    config = FiberFitConfig(
        temperatures=(1.0,),
        residual_variances=(0.1,),
        geometry_steps=1,
        assignment="unbalanced_sinkhorn",
        sinkhorn_iterations=50,
        sinkhorn_tolerance=0.0,
    )
    validation = real.validate_final_arm_state(
        fiber,
        tuple(observations),
        (_source_excluded_test_plan(0), _source_excluded_test_plan(1)),
        track_capacities=torch.ones(2, dtype=torch.float64),
        config=config,
    )
    assert validation["passed"] is True
    assert validation["world_covariances_spd"] is True
    assert validation["supported_projection_counts"] == [1, 1]

    with torch.no_grad():
        fiber.depth_logits[0] = torch.inf
    with pytest.raises(RuntimeError, match="non-finite raw parameters"):
        real.validate_final_arm_state(
            fiber,
            tuple(observations),
            (_source_excluded_test_plan(0), _source_excluded_test_plan(1)),
            track_capacities=torch.ones(2, dtype=torch.float64),
            config=config,
        )


def test_development_loader_opens_only_five_then_validation_release_opens_all(
    compact_bundle,
    tmp_path,
    monkeypatch,
):
    opened: list[str] = []
    original = GaussianObservationField.load_npz

    def tracked(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(GaussianObservationField, "load_npz", staticmethod(tracked))
    development = real.load_development_inputs(
        compact_bundle,
        expected_manifest_sha256=None,
    )
    assert tuple(development.inputs.view_names) == real.DEVELOPMENT_IDS
    assert opened == [f"{index:04d}.teacher.npz" for index in range(5)]
    assert development.receipt["validation_geometry_opened"] is False
    assert development.receipt["manifest_metadata_ids_read"] == list(real.EXPECTED_BUNDLE_IDS)
    assert development.receipt["materialized_camera_ids"] == list(real.DEVELOPMENT_IDS)
    assert development.receipt["teacher_geometry_ids_opened"] == list(real.DEVELOPMENT_IDS)
    assert development.receipt["report_rgb_or_mask_opened"] is False

    output = tmp_path / "freeze"
    output.mkdir()
    geometry_artifact = real._write_npz_exclusive(
        output / "geometry.npz",
        {"value": torch.tensor([1.0])},
    )
    geometry = real._freeze_receipt(
        output,
        "geometry",
        {"geometry": geometry_artifact},
    )
    appearance_artifact = real._write_npz_exclusive(
        output / "appearance.npz",
        {"value": torch.tensor([2.0])},
    )
    appearance = real._freeze_receipt(
        output,
        "appearance",
        {"appearance": appearance_artifact},
        parent_receipt=geometry,
    )
    validation = real.release_validation_inputs(
        compact_bundle,
        appearance,
        expected_manifest_sha256=None,
    )
    assert tuple(validation.inputs.view_names) == real.VALIDATION_IDS
    assert opened[-7:] == [f"{index:04d}.teacher.npz" for index in range(7)]
    assert validation.receipt["all_opened_ids"] == list(real.EXPECTED_BUNDLE_IDS)
    assert validation.receipt["returned_ids"] == list(real.VALIDATION_IDS)


def test_recursive_freeze_detects_geometry_tamper_before_validation(tmp_path):
    output = tmp_path / "tamper"
    output.mkdir()
    state_path = output / "geometry.npz"
    geometry_artifact = real._write_npz_exclusive(
        state_path,
        {"value": torch.tensor([1.0])},
    )
    geometry = real._freeze_receipt(
        output,
        "geometry",
        {"geometry": geometry_artifact},
    )
    appearance_artifact = real._write_npz_exclusive(
        output / "appearance.npz",
        {"value": torch.tensor([2.0])},
    )
    appearance = real._freeze_receipt(
        output,
        "appearance",
        {"appearance": appearance_artifact},
        parent_receipt=geometry,
    )
    state_path.write_bytes(state_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        real.verify_freeze_receipt(appearance, expected_kind="appearance")


def _canonical_json(value: dict) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _synthetic_descriptor(path: Path, *, members: bool = False) -> dict:
    descriptor = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if members:
        descriptor["members"] = {}
    return descriptor


def _write_fixture_json(path: Path, value: dict) -> None:
    path.write_bytes(_canonical_json(value))


def _resign_synthetic_root(result: dict, root_index: int = 0) -> None:
    root = result["root_results"][root_index]
    summary_path = Path(root["artifacts"]["root_summary_json"]["path"])
    summary = copy.deepcopy(root)
    summary_artifacts = dict(summary["artifacts"])
    summary_artifacts.pop("root_summary_json")
    summary["artifacts"] = summary_artifacts
    _write_fixture_json(summary_path, summary)
    root["artifacts"]["root_summary_json"] = _synthetic_descriptor(summary_path)


def _complete_signed_synthetic_release(tmp_path, monkeypatch) -> Path:
    protocol_directory = tmp_path / "protocol"
    protocol_directory.mkdir()
    documents = []
    for index, payload in enumerate((b"base", b"addendum-1", b"addendum-2", b"addendum-3")):
        path = protocol_directory / f"document_{index}.md"
        path.write_bytes(payload)
        documents.append((path, hashlib.sha256(payload).hexdigest()))
    monkeypatch.setattr(real, "PREREG_DOCUMENTS", tuple(documents))

    source_root = tmp_path / "synthetic_sources"
    source_root.mkdir()
    source_paths = (Path("runner.py"), Path("core.py"))
    for index, relative in enumerate(source_paths):
        (source_root / relative).write_text(f"source-{index}\n")
    monkeypatch.setattr(real, "SYNTHETIC_SOURCE_ROOT", source_root)
    monkeypatch.setattr(real, "SYNTHETIC_EXECUTED_SOURCE_PATHS", source_paths)

    result_path = tmp_path / "SYNTHETIC_RESULT.json"
    attempt_path = tmp_path / "SYNTHETIC_ATTEMPT.json"
    artifacts = tmp_path / "synthetic_artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(real, "SYNTHETIC_RESULT", result_path)
    monkeypatch.setattr(real, "SYNTHETIC_ATTEMPT", attempt_path)
    monkeypatch.setattr(real, "SYNTHETIC_ARTIFACTS", artifacts)

    protocol = real._protocol_receipt()
    source_hashes = real._synthetic_source_hashes()
    roots_list = [list(item) for item in real.SYNTHETIC_OFFICIAL_ROOT_TUPLES]
    attempt_receipt = {
        "namespace": real.SYNTHETIC_NAMESPACE,
        "status": "ATTEMPTED",
        "mode": "official",
        "root_tuples": roots_list,
        "all_nine_roots": [
            value for roots in real.SYNTHETIC_OFFICIAL_ROOT_TUPLES for value in roots
        ],
        "scene_roots": [roots[0] for roots in real.SYNTHETIC_OFFICIAL_ROOT_TUPLES],
        "depth_roots": [roots[1] for roots in real.SYNTHETIC_OFFICIAL_ROOT_TUPLES],
        "order_roots": [roots[2] for roots in real.SYNTHETIC_OFFICIAL_ROOT_TUPLES],
        "config": dict(real.SYNTHETIC_FROZEN_CONFIG),
        "arms": list(real.SYNTHETIC_ARM_NAMES),
        "protocol": protocol,
        "source_hashes": source_hashes,
        "artifacts_directory": str(artifacts),
        "result_path": str(result_path),
        "environment_at_reservation": {"fixture": True},
    }
    _write_fixture_json(attempt_path, attempt_receipt)
    official_attempt = {
        "receipt": attempt_receipt,
        "descriptor": _synthetic_descriptor(attempt_path),
    }

    root_results = []
    for root_index, roots in enumerate(real.SYNTHETIC_OFFICIAL_ROOT_TUPLES):
        root_directory = artifacts / f"root_{root_index}"
        root_directory.mkdir()
        arm_summaries = {}
        for arm_index, name in enumerate(real.SYNTHETIC_ARM_NAMES):
            arm_summaries[name] = {
                "association": {
                    "parent_purity": 0.99,
                    "parent_completeness": 0.99,
                    "outlier_track_dust_recall": 0.90,
                    "outlier_observation_dust_recall": 0.90,
                    "inlier_track_dust_false_positive_rate": 0.10,
                    "inlier_observation_dust_false_positive_rate": 0.10,
                },
                "geometry": {
                    "finite_spd": True,
                    "source_center_max_px": 0.0,
                    "source_covariance_relative_max": 0.0,
                    "depth_bound_incidence": 0,
                },
                "transport_mass_diagnostics": {"pass": True},
                "numerical_validity": {"pass": True},
                "initial_state_sha256": f"{root_index + arm_index + 1:064x}",
                "initial_state_matches_common": True,
                "history": [],
                "wall_time_seconds": 0.01,
                "heldout": {},
            }
        artifacts_payload = {}
        artifact_paths = {
            "initial_ply": root_directory / "gaussians_init.ply",
            "evidence_npz": root_directory / "evidence.npz",
            **{f"{name}_ply": root_directory / f"{name}.ply" for name in real.SYNTHETIC_ARM_NAMES},
        }
        for name, path in artifact_paths.items():
            path.write_bytes(f"{root_index}:{name}\n".encode())
            artifacts_payload[name] = _synthetic_descriptor(
                path,
                members=name == "evidence_npz",
            )
        nonoracle = set(real.SYNTHETIC_ARM_NAMES) - {"oracle"}
        root_summary = {
            "roots": list(roots),
            "input_receipt": {
                "roots": list(roots),
                "view_receipts": [
                    {
                        "moment_mean_max_abs": 0.0,
                        "moment_covariance_max_abs": 0.0,
                    }
                    for _view in range(5)
                ],
            },
            "common_initial_state_sha256": f"{root_index + 1:064x}",
            "nonoracle_state_hashes_before_evaluator": {
                name: f"{root_index + 10:064x}" for name in nonoracle
            },
            "nonoracle_plan_hashes_before_evaluator": {
                name: f"{root_index + 20:064x}" for name in nonoracle
            },
            "frozen_state_hashes_before_heldout": {
                name: f"{root_index + 30:064x}" for name in real.SYNTHETIC_ARM_NAMES
            },
            "frozen_plan_hashes_before_heldout": {
                name: f"{root_index + 40:064x}" for name in real.SYNTHETIC_ARM_NAMES
            },
            "projection_validity_arm_semantics": {"shared_mask": True},
            "heldout_release": {"materialized": True},
            "arms": arm_summaries,
            "artifacts": artifacts_payload,
            "peak_rss_kib": 1,
        }
        root_summary_path = root_directory / "ROOT_RESULT.json"
        _write_fixture_json(root_summary_path, root_summary)
        root_summary["artifacts"] = {
            **artifacts_payload,
            "root_summary_json": _synthetic_descriptor(root_summary_path),
        }
        root_results.append(root_summary)

    validity_checks = {
        "all_arms_finite_spd": True,
        "all_arms_share_common_initial_state": True,
        "uot_area_source_center_le_1e-8": True,
        "uot_area_source_covariance_le_1e-8": True,
        "all_depths_strictly_bounded": True,
        "moment_splits_rederive": True,
        "oracle_purity_ge_0_99": True,
        "oracle_completeness_ge_0_99": True,
    }
    result = {
        "namespace": real.SYNTHETIC_NAMESPACE,
        "status": "FAIL",
        "mode": "official",
        "roots": roots_list,
        "config": dict(real.SYNTHETIC_FROZEN_CONFIG),
        "arms": list(real.SYNTHETIC_ARM_NAMES),
        "root_results": root_results,
        "synthetic_gates": {
            "eligible": True,
            "validity": {"checks": validity_checks, "pass": True},
            "transport_mass_validity": {"checks": {"fixture": True}, "pass": True},
            "absolute_mechanism": {"checks": {"fixture": False}, "pass": False},
            "soft_assignment_gain": {"checks": {"fixture": False}, "pass": False},
            "capacity_attribution": {"checks": {"fixture": False}, "pass": False},
            "negative_control": {"checks": {"fixture": False}, "pass": False},
            "overall": {"pass": False},
        },
        "real_release": {
            "permitted": True,
            "primary_transport_arm": "uot_area",
            "paired_diagnostic_arms": ["uot_uniform", "uot_area"],
            "reason": "synthetic validity passed and uot_area was scientifically accepted",
        },
        "environment": {"fixture": True},
        "wall_time_seconds": 0.1,
        "peak_rss_kib": 1,
        "protocol": protocol,
        "source_hashes": source_hashes,
        "official_attempt": official_attempt,
    }
    _write_fixture_json(result_path, result)
    return result_path


def test_signed_synthetic_release_selects_primary_and_rejects_adversarial_mutations(
    tmp_path,
    monkeypatch,
):
    path = _complete_signed_synthetic_release(tmp_path, monkeypatch)
    release = real._assert_synthetic_release(path)
    assert release["eligible"] is True
    assert release["validity_pass"] is True
    assert release["primary_transport_arm"] == "uot_area"
    assert real._resolve_primary_arm(release, None) == "uot_area"
    with pytest.raises(RuntimeError, match="conflicts"):
        real._resolve_primary_arm(release, "uot_uniform")

    baseline_result = path.read_bytes()
    baseline_attempt = real.SYNTHETIC_ATTEMPT.read_bytes()
    copied = tmp_path / "forged_copy.json"
    copied.write_bytes(baseline_result)
    with pytest.raises(RuntimeError, match="exact frozen synthetic result path"):
        real._assert_synthetic_release(copied)

    forged_uniform = json.loads(baseline_result)
    forged_uniform["real_release"].update(
        {
            "primary_transport_arm": "uot_uniform",
            "reason": ("synthetic validity passed and uot_uniform was scientifically accepted"),
        }
    )
    _write_fixture_json(path, forged_uniform)
    with pytest.raises(RuntimeError, match="area-first"):
        real._assert_synthetic_release(path)

    mutations = {
        "root": lambda value: value["roots"][0].__setitem__(0, -1),
        "config": lambda value: value["config"].__setitem__("outer_steps", 19),
        "protocol": lambda value: value["protocol"]["base"].__setitem__("sha256", "0" * 64),
        "source": lambda value: value["source_hashes"].__setitem__("runner.py", "0" * 64),
        "arms": lambda value: value["arms"].pop(),
        "root_arm": lambda value: value["root_results"][0]["arms"].pop("row"),
    }
    for _label, mutate in mutations.items():
        path.write_bytes(baseline_result)
        value = json.loads(baseline_result)
        mutate(value)
        _write_fixture_json(path, value)
        with pytest.raises(RuntimeError, match="mismatch|complete frozen arm suite"):
            real._assert_synthetic_release(path)

    validity_mutations = {
        "finite_spd": lambda root: root["arms"]["hardmin"]["geometry"].__setitem__(
            "finite_spd", False
        ),
        "common_initial_state": lambda root: root["arms"]["row"].__setitem__(
            "initial_state_matches_common", False
        ),
        "area_source_center": lambda root: root["arms"]["uot_area"]["geometry"].__setitem__(
            "source_center_max_px", 2e-8
        ),
        "area_source_covariance": lambda root: root["arms"]["uot_area"]["geometry"].__setitem__(
            "source_covariance_relative_max", 2e-8
        ),
        "depth_incidence": lambda root: root["arms"]["oracle"]["geometry"].__setitem__(
            "depth_bound_incidence", 1
        ),
        "moment_split": lambda root: root["input_receipt"]["view_receipts"][0].__setitem__(
            "moment_mean_max_abs", 2e-10
        ),
        "oracle_purity": lambda root: root["arms"]["oracle"]["association"].__setitem__(
            "parent_purity", 0.98
        ),
        "oracle_completeness": lambda root: root["arms"]["oracle"]["association"].__setitem__(
            "parent_completeness", 0.98
        ),
    }
    for _label, mutate in validity_mutations.items():
        value = json.loads(baseline_result)
        mutate(value["root_results"][0])
        _resign_synthetic_root(value)
        _write_fixture_json(path, value)
        with pytest.raises(RuntimeError, match="validity checks disagree"):
            real._assert_synthetic_release(path)

    restored = json.loads(baseline_result)
    _resign_synthetic_root(restored)
    path.write_bytes(baseline_result)

    path.write_bytes(baseline_result)
    attempt = json.loads(baseline_attempt)
    attempt["root_tuples"][0][0] = -1
    _write_fixture_json(real.SYNTHETIC_ATTEMPT, attempt)
    result = json.loads(baseline_result)
    result["official_attempt"] = {
        "receipt": attempt,
        "descriptor": _synthetic_descriptor(real.SYNTHETIC_ATTEMPT),
    }
    _write_fixture_json(path, result)
    with pytest.raises(RuntimeError, match="attempt receipt is inconsistent"):
        real._assert_synthetic_release(path)

    real.SYNTHETIC_ATTEMPT.write_bytes(baseline_attempt)
    _write_fixture_json(
        path,
        {
            "namespace": real.SYNTHETIC_NAMESPACE,
            "mode": "official",
            "synthetic_gates": {"eligible": True, "validity": {"pass": True}},
        },
    )
    with pytest.raises(RuntimeError, match="keys mismatch"):
        real._assert_synthetic_release(path)


def test_official_real_invocation_requires_confirmation_and_exact_paths(
    tmp_path,
    monkeypatch,
):
    bundle = tmp_path / "bundle"
    output = tmp_path / "output"
    result = tmp_path / "result.json"
    synthetic = tmp_path / "synthetic.json"
    scene = tmp_path / "scene"
    monkeypatch.setattr(real, "BUNDLE", bundle)
    monkeypatch.setattr(real, "DEFAULT_OUT", output)
    monkeypatch.setattr(real, "DEFAULT_RESULT", result)
    monkeypatch.setattr(real, "SYNTHETIC_RESULT", synthetic)
    monkeypatch.setattr(real, "SCENE", scene)

    valid = {
        "mode": "official",
        "confirm_official": True,
        "bundle": bundle,
        "output": output,
        "result_path": result,
        "synthetic_result": synthetic,
        "scene": scene,
        "expected_manifest_sha256": real.FROZEN_MANIFEST_SHA256,
    }
    real._require_official_real_invocation(**valid)

    for field in ("bundle", "output", "result_path", "synthetic_result", "scene"):
        invalid = dict(valid)
        invalid[field] = tmp_path / f"wrong_{field}"
        with pytest.raises(RuntimeError, match="path mismatch"):
            real._require_official_real_invocation(**invalid)

    invalid = dict(valid)
    invalid["mode"] = "development"
    with pytest.raises(RuntimeError, match="official mode and confirmation"):
        real._require_official_real_invocation(**invalid)
    invalid = dict(valid)
    invalid["confirm_official"] = False
    with pytest.raises(RuntimeError, match="official mode and confirmation"):
        real._require_official_real_invocation(**invalid)
    invalid = dict(valid)
    invalid["expected_manifest_sha256"] = None
    with pytest.raises(RuntimeError, match="manifest"):
        real._require_official_real_invocation(**invalid)


def test_real_attempt_reservation_refuses_any_occupied_namespace(
    tmp_path,
    monkeypatch,
):
    synthetic_release = {
        "path": "synthetic",
        "bytes": 1,
        "sha256": "a" * 64,
        "official_attempt_path": "synthetic-attempt",
        "official_attempt_sha256": "b" * 64,
    }
    for occupied in ("attempt", "result", "output"):
        domain = tmp_path / occupied
        domain.mkdir()
        attempt = domain / "ATTEMPT.json"
        result = domain / "RESULT.json"
        output = domain / "output"
        monkeypatch.setattr(real, "OFFICIAL_ATTEMPT", attempt)
        if occupied == "attempt":
            attempt.write_text("{}", encoding="utf-8")
        elif occupied == "result":
            result.write_text("{}", encoding="utf-8")
        else:
            output.mkdir()

        with pytest.raises(RuntimeError, match="already exists"):
            real._reserve_real_attempt(
                bundle=domain / "bundle",
                output=output,
                result_path=result,
                synthetic_release=synthetic_release,
                primary_arm="uot_uniform",
                release_report=False,
                protocol={"base": {}, "addenda": []},
                source_hashes={"runner.py": "c" * 64},
            )


def test_real_official_crash_after_attempt_is_durable_and_loader_is_not_reused(
    tmp_path,
    monkeypatch,
):
    synthetic_result = _complete_signed_synthetic_release(tmp_path, monkeypatch)
    bundle = tmp_path / "frozen_bundle"
    output = tmp_path / "real_output"
    result = tmp_path / "REAL_RESULT.json"
    attempt = tmp_path / "REAL_ATTEMPT.json"
    scene = tmp_path / "frozen_scene"
    monkeypatch.setattr(real, "BUNDLE", bundle)
    monkeypatch.setattr(real, "DEFAULT_OUT", output)
    monkeypatch.setattr(real, "DEFAULT_RESULT", result)
    monkeypatch.setattr(real, "OFFICIAL_ATTEMPT", attempt)
    monkeypatch.setattr(real, "SCENE", scene)

    source_root = tmp_path / "real_sources"
    source_root.mkdir()
    source_paths = (Path("real_runner.py"), Path("fiber.py"))
    for index, relative in enumerate(source_paths):
        (source_root / relative).write_text(f"real-source-{index}\n")
    monkeypatch.setattr(real, "REAL_SOURCE_ROOT", source_root)
    monkeypatch.setattr(real, "REAL_EXECUTED_SOURCE_PATHS", source_paths)

    loader_calls = 0

    def crash_after_attempt(*_args, **_kwargs):
        nonlocal loader_calls
        loader_calls += 1
        assert attempt.is_file()
        assert output.is_dir()
        raise RuntimeError("injected crash after durable real attempt")

    monkeypatch.setattr(real, "load_development_inputs", crash_after_attempt)
    kwargs = {
        "mode": "official",
        "confirm_official": True,
        "bundle": bundle,
        "output": output,
        "result_path": result,
        "synthetic_result": synthetic_result,
        "scene": scene,
    }
    with pytest.raises(RuntimeError, match="injected crash"):
        real.run_real_interaction(**kwargs)
    receipt = json.loads(attempt.read_bytes())
    assert receipt["status"] == "ATTEMPTED"
    assert receipt["bundle"] == str(bundle)
    assert receipt["output_directory"] == str(output)
    assert receipt["result_path"] == str(result)
    assert (
        receipt["synthetic_result"]["sha256"]
        == hashlib.sha256(synthetic_result.read_bytes()).hexdigest()
    )
    assert receipt["executed_source_hashes"] == real._real_source_hashes()
    assert loader_calls == 1
    assert not result.exists()

    with pytest.raises(RuntimeError, match="attempt/result/output already exists"):
        real.run_real_interaction(**kwargs)
    assert loader_calls == 1
    assert attempt.is_file()


def test_protocol_receipt_binds_base_and_all_frozen_addenda():
    receipt = real._protocol_receipt()
    assert receipt["base"]["sha256"] == real.PREREG_DOCUMENTS[0][1]
    assert [item["sha256"] for item in receipt["addenda"]] == [
        expected for _path, expected in real.PREREG_DOCUMENTS[1:]
    ]
