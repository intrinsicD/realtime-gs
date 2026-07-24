"""CPU-only contract tests for the frozen G²SR correspondence diagnostic."""

from __future__ import annotations

import copy
import inspect
import json
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from benchmarks import g2sr_correspondence_diagnostic as diagnostic

from rtgs.core.camera import Camera


def _skip_without_teacher_bundle() -> None:
    """Skip when the sealed teacher evidence bundle is absent.

    The plan/replay diagnostics bind to ``runs/.../reconstruction_inputs/manifest.json``,
    which lives under the gitignored ``runs/`` tree and only exists in the local
    reproduction environment. On a clean checkout (CI, a fresh clone) the bundle is
    missing, so these tests skip rather than hard-fail on the missing manifest —
    matching the CPU-first testability rule and the existing precondition-skip idiom in
    ``test_compact_occupancy_refinement_factorial``.
    """
    manifest = diagnostic.TEACHER_BUNDLE / "manifest.json"
    if not manifest.exists():
        pytest.skip(f"sealed teacher evidence bundle absent: {manifest} (runs/ is gitignored)")


def _parallel_camera(center_x: float, *, width: int = 96, height: int = 72) -> Camera:
    return Camera(
        fx=80.0,
        fy=78.0,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
        R=torch.eye(3),
        t=torch.tensor([-center_x, 0.0, 0.0]),
    )


def _constant_flow(height: int, width: int, offset: tuple[float, float]) -> torch.Tensor:
    result = torch.zeros(height, width, 2)
    result[..., 0] = offset[0]
    result[..., 1] = offset[1]
    return result


def test_crop_coordinate_and_covariance_round_trip_is_exact() -> None:
    diagnostic.assert_crop_coordinate_convention()
    means_native = torch.tensor(
        [[1489.5, 1777.5], [2000.25, 2400.75], [5327.5, 3455.5]],
        dtype=torch.float64,
    )
    covariance_native = torch.tensor(
        [[[9.0, 1.5], [1.5, 36.0]], [[4.5, -0.25], [-0.25, 18.0]]],
        dtype=torch.float64,
    )
    means_flow = diagnostic.native_to_flow_points(means_native)
    covariance_flow = diagnostic.native_to_flow_covariances(covariance_native)

    assert torch.equal(diagnostic.flow_to_native_points(means_flow), means_native)
    assert torch.equal(
        diagnostic.flow_to_native_covariances(covariance_flow),
        covariance_native,
    )
    assert torch.equal(means_flow[0], torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert torch.equal(covariance_flow, covariance_native / 9.0)


def test_flow_camera_projects_crop_resized_pixels_consistently() -> None:
    camera = _parallel_camera(-0.5, width=5328, height=4608)
    camera.fx = 4566.0
    camera.fy = 4565.5
    camera.cx = 2649.5
    camera.cy = 2318.5
    points = torch.tensor([[0.1, -0.2, 3.0], [-0.15, 0.1, 4.0]])
    native, _ = camera.project(points)
    transformed = diagnostic.native_to_flow_points(native)
    projected, _ = diagnostic.flow_camera(camera).project(points)
    assert torch.allclose(projected, transformed, atol=1e-4, rtol=0)
    assert (diagnostic.flow_camera(camera).width, diagnostic.flow_camera(camera).height) == (
        1280,
        560,
    )


def test_pair_protocol_is_frozen_and_heldout_is_rejected() -> None:
    pairs = diagnostic.validate_pair_protocol()
    assert tuple(pair.pair_id for pair in pairs) == (
        "C0014_to_C0005",
        "C0014_to_C0026",
    )
    assert diagnostic.HELDOUT_VIEW not in {
        diagnostic.REFERENCE_VIEW,
        *diagnostic.TARGET_VIEWS,
        *diagnostic.BOUNDS_VIEWS,
        *diagnostic.VIEWER_VIEWS,
    }
    with pytest.raises(ValueError, match="held-out view C1004"):
        diagnostic.validate_pair_protocol("C1004", diagnostic.TARGET_VIEWS)
    with pytest.raises(ValueError, match="held-out view C1004"):
        diagnostic.validate_pair_protocol(
            diagnostic.REFERENCE_VIEW,
            ("C0005", "C1004"),
        )


def test_plan_preregisters_all_required_protocol_axes_and_memory() -> None:
    _skip_without_teacher_bundle()
    plan = diagnostic.plan_payload()
    assert plan["status"] == "PREREGISTERED_NOT_RUN"
    assert plan["reference"] == "C0014"
    assert plan["targets"] == ["C0005", "C0026"]
    assert plan["heldout"]["view"] == "C1004"
    assert plan["crop"]["native_xywh"] == [1488, 1776, 3840, 1680]
    assert plan["crop"]["flow_size"] == [1280, 560]
    assert plan["flow"]["all_five_sigma_points_required"]
    assert plan["flow"]["device"] == "cuda"
    assert plan["fb_sensitivity_px"] == [1.0, 3.0, 5.0]
    assert plan["bounds"]["views"] == list(diagnostic.BOUNDS_VIEWS)
    assert str(list(diagnostic.VIEWER_VIEWS)) in plan["artifacts"]["viewer"]
    assert plan["estimated_memory"]["conservative_peak_gpu_gib"] == 5.5
    assert "--allow-download" not in plan["commands"]["acquire_local_weights_only"]
    assert "--allow-download" in plan["commands"]["acquire_allow_official_weight_download"]
    _, calibration_path = diagnostic._calibration_records()
    assert calibration_path.is_file()


def test_replay_binding_covers_every_local_rtgs_python_source(
    tmp_path,
    monkeypatch,
) -> None:
    _skip_without_teacher_bundle()
    replay_paths = set(diagnostic.REPLAY_CODE_PATHS)
    expected_rtgs = {
        path.relative_to(diagnostic.ROOT)
        for path in (diagnostic.ROOT / "src" / "rtgs").rglob("*.py")
        if path.is_file()
    }
    assert expected_rtgs <= replay_paths
    assert diagnostic.Path("src/rtgs/lift/base.py") in replay_paths
    assert diagnostic.Path("src/rtgs/core/sh.py") in replay_paths
    diagnostic._assert_rtgs_import_origins()

    plan = diagnostic.plan_payload(output=tmp_path / "canonical-run")
    label = "src/rtgs/lift/base.py"
    wrong_label = "src/rtgs/core/sh.py"
    misbound = copy.deepcopy(plan)
    for key in ("path", "sha256", "bytes", "mtime_ns"):
        misbound["code_sources"][label][key] = plan["code_sources"][wrong_label][key]
    with pytest.raises(RuntimeError, match="misbound"):
        diagnostic._validate_plan_schema(misbound)

    escaped_replay = copy.deepcopy(plan)
    escaped_replay["code_sources"][label]["replay_path"] = str((tmp_path / "outside.py").resolve())
    with pytest.raises(RuntimeError, match="misbound"):
        diagnostic._validate_plan_schema(escaped_replay)

    shadow = SimpleNamespace(__file__=str(tmp_path / "shadowed.py"))
    monkeypatch.setitem(sys.modules, "rtgs.shadow_probe", shadow)
    with pytest.raises(RuntimeError, match="shadowed outside"):
        diagnostic._assert_rtgs_import_origins()


def test_cuda_device_is_frozen_in_plan_factory_and_cli() -> None:
    _skip_without_teacher_bundle()
    assert diagnostic.plan_payload()["flow"]["device"] == diagnostic.ACQUISITION_DEVICE
    with pytest.raises(ValueError, match="frozen acquisition device"):
        diagnostic.create_flow_backend(
            diagnostic.FLOW_BACKEND,
            device="cpu",
            allow_download=False,
        )
    with pytest.raises(SystemExit):
        diagnostic._parser().parse_args(["acquire", "--device", "cpu"])


def test_decorated_raft_and_transform_runtime_sources_are_bound() -> None:
    def numerical_factory() -> None:
        return None

    def decorated_entrypoint() -> None:
        return numerical_factory()

    decorated_entrypoint.__wrapped__ = numerical_factory  # type: ignore[attr-defined]

    class RuntimeTransforms:
        pass

    records = diagnostic._torchvision_runtime_source_records(
        decorated_entrypoint,
        RuntimeTransforms(),
    )
    assert set(records) == {
        "raft_entrypoint",
        "raft_unwrapped",
        "raft_module",
        "transforms_module",
    }
    assert records["raft_entrypoint"]["qualname"].endswith("decorated_entrypoint")
    assert records["raft_unwrapped"]["qualname"].endswith("numerical_factory")
    assert records["transforms_module"]["qualname"].endswith("RuntimeTransforms")
    for record in records.values():
        source = diagnostic.Path(record["path"])
        assert source.is_file()
        assert record["sha256"] == diagnostic.sha256_file(source)


def test_import_does_not_eagerly_import_torchvision() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import benchmarks.g2sr_correspondence_diagnostic; "
            "assert 'torchvision' not in sys.modules"
        ),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_rgb_free_source_verification_uses_only_compact_view_bytes(
    tmp_path,
    monkeypatch,
) -> None:
    _skip_without_teacher_bundle()
    output = tmp_path / "rgb-free-plan"
    plan_path = diagnostic.write_plan(output)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    original_open = diagnostic.Path.open

    def deny_raster_open(path, *args, **kwargs):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            raise AssertionError(f"raster opened during RGB-free verification: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(diagnostic.Path, "open", deny_raster_open)
    hashes = diagnostic._current_source_hashes(
        plan,
        verify_source_color_bytes=False,
    )
    assert all(
        hashes[f"source_colors/{view}"]
        == plan["scientific_inputs"]["source_colors"][view]["sha256"]
        for view in (diagnostic.REFERENCE_VIEW, *diagnostic.TARGET_VIEWS)
    )
    verified = diagnostic._current_source_hashes(
        plan,
        verify_source_color_bytes=True,
    )
    assert verified == hashes


def test_post_flow_npz_audit_rejects_source_rgb_payloads(tmp_path) -> None:
    clean = tmp_path / "clean.npz"
    np.savez_compressed(clean, forward=np.zeros((2, 3, 2), dtype=np.float32))
    assert diagnostic.audit_npz_no_source_rgb(clean) == ("forward",)
    leaked = tmp_path / "leaked.npz"
    np.savez_compressed(leaked, source_rgb=np.zeros((2, 3, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="forbidden source RGB"):
        diagnostic.audit_npz_no_source_rgb(leaked)


def test_epipolar_residual_and_dlt_are_exact_for_synthetic_pair() -> None:
    first = _parallel_camera(-0.6)
    second = _parallel_camera(0.5)
    points_world = torch.tensor(
        [[-0.2, -0.1, 3.0], [0.25, 0.2, 4.5], [0.1, -0.15, 5.0]],
        dtype=torch.float64,
    )
    first_points = first.project(points_world)[0]
    second_points = second.project(points_world)[0]
    residual = diagnostic.calibrated_epipolar_residual(
        first,
        second,
        first_points,
        second_points,
    )
    triangulated = diagnostic.triangulate_centers_dlt(
        [first, second],
        torch.stack([first_points, second_points], dim=1),
    )
    angles = diagnostic.ray_angle_degrees(first, second, first_points, second_points)

    assert torch.all(residual < 1e-10)
    assert bool(triangulated.valid.all())
    assert torch.allclose(triangulated.points_world, points_world, atol=2e-6, rtol=2e-6)
    assert torch.all(angles > 1.0)


def test_pure_pair_analysis_accepts_synthetic_constant_flow_and_reports_sensitivity(
    tmp_path,
) -> None:
    # Use the production 1280x560 flow frame and cameras whose native projections are related by
    # exactly a -11 px native x displacement (-11/3 flow px).
    first = Camera(
        fx=300.0,
        fy=300.0,
        cx=1488.0 + 3.0 * 640.0,
        cy=1776.0 + 3.0 * 280.0,
        width=5328,
        height=4608,
        R=torch.eye(3, dtype=torch.float64),
        t=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
    )
    second = Camera(
        fx=300.0,
        fy=300.0,
        cx=first.cx,
        cy=first.cy,
        width=5328,
        height=4608,
        R=torch.eye(3, dtype=torch.float64),
        t=torch.tensor([-0.11, 0.0, 0.0], dtype=torch.float64),
    )
    world = torch.tensor(
        [[0.0, 0.0, 3.0], [0.2, 0.1, 3.0]],
        dtype=torch.float64,
    )
    reference_means = first.project(world)[0]
    target_means = second.project(world)[0]
    displacement_flow = (
        diagnostic.native_to_flow_points(target_means)
        - diagnostic.native_to_flow_points(reference_means)
    )[0]
    forward = _constant_flow(560, 1280, tuple(displacement_flow.tolist())).double()
    backward = -forward
    covariances = torch.eye(2, dtype=torch.float64)[None].repeat(2, 1, 1) * 36.0
    mask = torch.ones(560, 1280, dtype=torch.float64)
    foreground = torch.tensor([True, False])
    gates = diagnostic.GateConfig(
        affine_max_rms_px=1e-3,
        epipolar_max_error_native_px=1e-3,
        dlt_max_reprojection_native_px=1e-3,
        min_ray_angle_deg=0.1,
    )

    arrays, summary = diagnostic.analyze_pair_tensors(
        pair=diagnostic.PairSpec("C0014", "C0005"),
        reference_means_native=reference_means,
        reference_covariances_native=covariances,
        reference_foreground=foreground,
        target_mask_flow=mask,
        forward_flow=forward,
        backward_flow=backward,
        reference_camera=first,
        target_camera=second,
        bounds_lower=torch.tensor([-2.0, -2.0, 1.0], dtype=torch.float64),
        bounds_upper=torch.tensor([2.0, 2.0, 5.0], dtype=torch.float64),
        gates=gates,
    )

    assert bool(arrays["all_sigma_samples_valid"].all())
    assert bool(arrays["accepted_geometry"].all())
    assert torch.equal(arrays["accepted_foreground"], foreground)
    assert summary["counts"]["marginal"]["accepted_geometry"] == 2
    assert summary["counts"]["marginal"]["accepted_foreground"] == 1
    assert set(summary["stage_distributions"]) == {
        "pretri",
        "dlt_valid",
        "accepted_geometry",
        "accepted_foreground",
    }
    assert summary["dlt_distribution_policy"]["minimum_observation_count"] == 2
    assert summary["archive_nonfinite_policy"]["encoding"] == "uint8_sidecar"
    assert (
        summary["archive_nonfinite_policy"]["encoded_float_arrays"]
        == diagnostic._encoded_correspondence_float_names()
    )
    for name in diagnostic._encoded_correspondence_float_names():
        assert f"{name}{diagnostic.NONFINITE_CODE_SUFFIX}" in arrays
    assert summary["counts"]["sequential_order"][-1] == "accepted_foreground"
    assert set(summary["fb_sensitivity"]) == {"1.0", "3.0", "5.0"}
    assert all(cell["accepted_geometry_count"] == 2 for cell in summary["fb_sensitivity"].values())

    # Pad the two active rows into the frozen 640-lineage archive. Inactive rows have false stage
    # masks, so replay must exactly recover the runtime distributions and DLT marginal gates.
    schema = diagnostic._correspondence_schema()
    archived = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in schema.items()}
    archived["component_id"] = np.arange(
        diagnostic.REFERENCE_COMPONENT_COUNT,
        dtype=np.int64,
    )
    for name, value in arrays.items():
        archived[name][: value.shape[0]] = value.detach().cpu().numpy()
    archived["dlt_observation_valid"][2:, 0] = True
    archived["dlt_observation_count"][2:] = 1
    path = tmp_path / "synthetic_correspondence.npz"
    np.savez_compressed(path, **archived)
    assert diagnostic.reproduce_stage_distributions(path) == summary["stage_distributions"]
    diagnostic.verify_correspondence_evidence(path, summary, gates=gates)
    first_distribution = summary["stage_distributions"]["dlt_valid"]["dlt_nullspace_gap"]
    assert set(first_distribution) == {"finite_quantiles", "value_class_counts"}
    assert (
        sum(
            first_distribution["value_class_counts"][name]
            for name in (
                "finite",
                "nan",
                "positive_infinity",
                "negative_infinity",
            )
        )
        == first_distribution["value_class_counts"]["total"]
    )

    invented_distribution_counts = copy.deepcopy(summary)
    invented_distribution_counts["stage_distributions"]["dlt_valid"]["dlt_nullspace_gap"][
        "value_class_counts"
    ]["positive_infinity"] += 1
    with pytest.raises(RuntimeError, match="stage distributions"):
        diagnostic.verify_correspondence_evidence(
            path,
            invented_distribution_counts,
            gates=gates,
        )

    inconsistent = copy.deepcopy(archived)
    inconsistent["dlt_observation_count"][2] = 0
    inconsistent_path = tmp_path / "inconsistent_dlt_observations.npz"
    np.savez_compressed(inconsistent_path, **inconsistent)
    with pytest.raises(RuntimeError, match="observation counts"):
        diagnostic.verify_correspondence_evidence(
            inconsistent_path,
            summary,
            gates=gates,
        )

    invented_valid = copy.deepcopy(archived)
    invented_valid["dlt_valid"][2] = True
    invented_valid_path = tmp_path / "invented_dlt_valid.npz"
    np.savez_compressed(invented_valid_path, **invented_valid)
    with pytest.raises(RuntimeError, match="DLT-valid mask"):
        diagnostic.verify_correspondence_evidence(
            invented_valid_path,
            summary,
            gates=gates,
        )


def test_native_means_api_is_preferred_with_compatibility_fallback() -> None:
    class Native:
        means = torch.tensor([[1.0, 2.0]])

        def native_means(self, *, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor([[1.25, 2.5]], dtype=dtype)

    class Legacy:
        means = torch.tensor([[1.0, 2.0]])

    assert torch.equal(
        diagnostic.observation_native_means(Native(), dtype=torch.float64),
        torch.tensor([[1.25, 2.5]], dtype=torch.float64),
    )
    assert torch.equal(
        diagnostic.observation_native_means(Legacy(), dtype=torch.float64),
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
    )


def test_plan_write_is_immutable(tmp_path) -> None:
    _skip_without_teacher_bundle()
    path = diagnostic.write_plan(tmp_path)
    before = path.read_bytes()
    payload = json.loads(before)
    assert payload["artifact_type"] == diagnostic.PLAN_ARTIFACT_TYPE
    assert payload["decision_rule"]["binary_mechanism_success"] is False
    assert payload["teacher_bundle"]["reference_teacher"]["sha256"]
    assert payload["git"]["head"]
    for label, binding in payload["code_sources"].items():
        replay = diagnostic.ROOT / binding["replay_path"]
        assert replay.is_file(), label
        assert diagnostic.sha256_file(replay) == binding["sha256"]
    with pytest.raises(FileExistsError):
        diagnostic.write_plan(tmp_path)
    assert path.read_bytes() == before


def test_empty_mechanism_ply_is_valid_and_does_not_invent_a_marker(tmp_path) -> None:
    gaussians = diagnostic._mechanism_gaussians(
        torch.empty(0, 3),
        torch.empty(0, 3),
        extent=2.0,
    )
    path = tmp_path / "empty.ply"
    diagnostic._save_mechanism_ply(path, gaussians)
    header = path.read_bytes().decode("ascii")
    assert "element vertex 0" in header


def test_plan_schema_is_fail_closed_and_rejects_heldout_leakage() -> None:
    _skip_without_teacher_bundle()
    plan = diagnostic.plan_payload()
    diagnostic._validate_plan_schema(plan)

    extra = copy.deepcopy(plan)
    extra["unregistered_override"] = True
    with pytest.raises(RuntimeError, match="key set mismatch"):
        diagnostic._validate_plan_schema(extra)

    leaked = copy.deepcopy(plan)
    leaked["scientific_inputs"]["source_colors"]["C0014"]["path"] = (
        "dataset/frame/gaussians2d/C1004.rtgsv"
    )
    with pytest.raises(RuntimeError, match="exactly once"):
        diagnostic._validate_plan_schema(leaked)

    for key in ("question", "scope"):
        changed = copy.deepcopy(plan)
        changed[key] += " changed"
        with pytest.raises(RuntimeError, match="question/scope"):
            diagnostic._validate_plan_schema(changed)
    changed_policy = copy.deepcopy(plan)
    changed_policy["decision_rule"]["policy"] += " changed"
    with pytest.raises(RuntimeError, match="decision policy"):
        diagnostic._validate_plan_schema(changed_policy)

    result_contract = {
        "decision_rule": diagnostic._frozen_decision_rule(),
        "claim_scope": diagnostic.EXPLORATORY_CLAIM_SCOPE,
    }
    diagnostic._validate_exploratory_contract(result_contract, result=True)
    changed_claim = copy.deepcopy(result_contract)
    changed_claim["claim_scope"] += " changed"
    with pytest.raises(RuntimeError, match="claim scope"):
        diagnostic._validate_exploratory_contract(changed_claim, result=True)


def test_analyze_has_no_custom_gate_bypass_and_plan_names_both_affine_conditions() -> None:
    _skip_without_teacher_bundle()
    assert list(inspect.signature(diagnostic.analyze).parameters) == ["output"]
    source = inspect.getsource(diagnostic.analyze)
    assert source.index("_verify_analysis_directory(output, temporary)") < source.index(
        "os.replace(temporary, final)"
    )
    gates = diagnostic.plan_payload()["gates"]
    assert "affine_source_design_max_condition" in gates
    assert "affine_recovered_max_condition" in gates
    assert "affine_max_condition" not in gates


def test_crop_rejects_non_native_or_nonfinite_inputs(monkeypatch) -> None:
    with pytest.raises(ValueError, match="exact native HWC"):
        diagnostic.crop_area_resize(torch.zeros(10, 10, 3))
    monkeypatch.setattr(diagnostic, "NATIVE_HEIGHT", 9)
    monkeypatch.setattr(diagnostic, "NATIVE_WIDTH", 12)
    monkeypatch.setattr(diagnostic, "FLOW_CROP", (0, 0, 12, 9))
    monkeypatch.setattr(diagnostic, "FLOW_SHAPE", (3, 4))
    malformed = torch.zeros(9, 12)
    malformed[0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        diagnostic.crop_area_resize(malformed)


def test_exact_flow_and_analysis_input_npz_schemas_reject_malformed_payloads(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(diagnostic, "FLOW_HEIGHT", 3)
    monkeypatch.setattr(diagnostic, "FLOW_WIDTH", 4)
    monkeypatch.setattr(diagnostic, "FLOW_SHAPE", (3, 4))
    forward = np.zeros(
        (diagnostic.FLOW_HEIGHT, diagnostic.FLOW_WIDTH, 2),
        dtype=np.float32,
    )
    flow = tmp_path / "flow.npz"
    np.savez_compressed(flow, forward=forward, backward=forward)
    assert diagnostic.validate_flow_npz(flow) == ("forward", "backward")

    wrong_dtype = tmp_path / "wrong_dtype.npz"
    np.savez_compressed(
        wrong_dtype,
        forward=forward.astype(np.float64),
        backward=forward,
    )
    with pytest.raises(RuntimeError, match="dtype"):
        diagnostic.validate_flow_npz(wrong_dtype)

    extra = tmp_path / "extra.npz"
    np.savez_compressed(extra, forward=forward, backward=forward, extra=np.zeros(1))
    with pytest.raises(RuntimeError, match="key set"):
        diagnostic.validate_flow_npz(extra)

    schema = diagnostic._analysis_input_schema()
    arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in schema.items()}
    compact = tmp_path / "analysis_input.npz"
    np.savez_compressed(compact, **arrays)
    assert set(diagnostic.validate_analysis_input_npz(compact)) == set(schema)
    arrays["reference_means_native"][0, 0] = np.nan
    nonfinite = tmp_path / "analysis_nonfinite.npz"
    np.savez_compressed(nonfinite, **arrays)
    with pytest.raises(RuntimeError, match="non-finite"):
        diagnostic.validate_analysis_input_npz(nonfinite)


def test_mask_hashes_compact_masks_and_bounds_are_replayed(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(diagnostic, "NATIVE_HEIGHT", 2)
    monkeypatch.setattr(diagnostic, "NATIVE_WIDTH", 2)
    monkeypatch.setattr(diagnostic, "FLOW_SHAPE", (2, 2))
    monkeypatch.setattr(diagnostic, "crop_area_resize", lambda value: value)

    cameras = [_parallel_camera(float(index), width=2, height=2) for index in range(7)]
    field = SimpleNamespace(
        native_means=lambda *, dtype: torch.tensor([[0.5, 0.5]], dtype=dtype),
        effective_variances=lambda: torch.tensor([[1.0, 2.0]]),
        rotations=torch.tensor([0.0]),
        colors=torch.tensor([[0.2, 0.4, 0.6]]),
        amplitudes=torch.tensor([0.75]),
    )
    fake_inputs = SimpleNamespace(
        view_names=list(diagnostic.BOUNDS_VIEWS),
        cameras=cameras,
        observations=[
            field if view == diagnostic.REFERENCE_VIEW else SimpleNamespace()
            for view in diagnostic.BOUNDS_VIEWS
        ],
    )
    monkeypatch.setattr(
        diagnostic.ReconstructionInputs,
        "load",
        lambda *args, **kwargs: fake_inputs,
    )
    records = {
        view: {}
        for view in {
            diagnostic.REFERENCE_VIEW,
            *diagnostic.TARGET_VIEWS,
            *diagnostic.BOUNDS_VIEWS,
        }
    }
    monkeypatch.setattr(
        diagnostic,
        "_calibration_records",
        lambda: (records, tmp_path / "calibration.json"),
    )
    masks = {
        view: torch.ones(2, 2)
        for view in {
            diagnostic.REFERENCE_VIEW,
            *diagnostic.TARGET_VIEWS,
            *diagnostic.BOUNDS_VIEWS,
        }
    }
    monkeypatch.setattr(
        diagnostic,
        "_load_undistorted_mask",
        lambda view, camera, record: masks[view].clone(),
    )

    def forbid_rgb_open(*args, **kwargs):
        raise AssertionError("RGB/JPEG access is forbidden during analysis verification")

    monkeypatch.setattr(diagnostic, "_camera_for_native_view", forbid_rgb_open)
    monkeypatch.setattr(diagnostic, "_load_undistorted_rgb", forbid_rgb_open)
    dimension_camera_calls = []

    def frozen_dimension_camera(record, *, width, height):
        dimension_camera_calls.append((width, height))
        return _parallel_camera(0.0, width=width, height=height)

    monkeypatch.setattr(diagnostic, "_camera_for_dimensions", frozen_dimension_camera)
    from PIL import Image as PILImage

    monkeypatch.setattr(PILImage, "open", forbid_rgb_open)
    replay_center = torch.tensor([0.25, -0.5, 2.0])
    replay_extent = 4.0
    import rtgs.data.calibrated as calibrated

    monkeypatch.setattr(
        calibrated,
        "_object_bounds",
        lambda replay_cameras, replay_masks: (replay_center, replay_extent),
    )

    mask_bindings = {}
    planned_masks = {}
    for view in records:
        path = tmp_path / f"mask_{view}.png"
        path.write_bytes(f"mask-{view}".encode())
        planned_masks[view] = diagnostic._file_binding(path)
        if view in diagnostic.BOUNDS_VIEWS:
            mask = masks[view]
            mask_bindings[view] = {
                "path": diagnostic.display_path(path),
                "source_sha256": diagnostic.sha256_file(path),
                "undistorted_tensor_sha256": diagnostic.tensor_sha256(mask),
                "foreground_fraction": float((mask > 0.5).float().mean()),
            }
    lower = replay_center - replay_extent * diagnostic.BOUNDS_SCALE
    upper = replay_center + replay_extent * diagnostic.BOUNDS_SCALE
    bounds = {
        "source": "seven_training_masks",
        "views": list(diagnostic.BOUNDS_VIEWS),
        "center": replay_center.tolist(),
        "extent": replay_extent,
        "bounds_scale": diagnostic.BOUNDS_SCALE,
        "aabb_lower": lower.tolist(),
        "aabb_upper": upper.tolist(),
        "masks": mask_bindings,
    }
    compact = tmp_path / "analysis_input.npz"
    np.savez_compressed(
        compact,
        reference_means_native=np.array([[0.5, 0.5]], dtype=np.float64),
        reference_covariances_native=np.array(
            [[[1.0, 0.0], [0.0, 2.0]]],
            dtype=np.float64,
        ),
        reference_colors=np.array([[0.2, 0.4, 0.6]], dtype=np.float32),
        reference_amplitudes=np.array([0.75], dtype=np.float32),
        reference_foreground=np.array([True]),
        **{
            f"target_mask_flow_{target}": masks[target].numpy()
            for target in diagnostic.TARGET_VIEWS
        },
    )
    plan = {"scientific_inputs": {"source_masks": planned_masks}}
    manifest = {"compact_analysis_input": {"path": compact.name}}
    diagnostic._verify_mask_and_bounds_replay(
        plan=plan,
        acquisition=tmp_path,
        manifest=manifest,
        bounds=bounds,
    )
    assert dimension_camera_calls == [(2, 2)]

    bad_hashes = copy.deepcopy(bounds)
    bad_hashes["masks"][diagnostic.BOUNDS_VIEWS[0]]["undistorted_tensor_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="undistorted mask tensor digest"):
        diagnostic._verify_mask_and_bounds_replay(
            plan=plan,
            acquisition=tmp_path,
            manifest=manifest,
            bounds=bad_hashes,
        )
    bad_bounds = copy.deepcopy(bounds)
    bad_bounds["aabb_upper"][0] += 0.25
    with pytest.raises(RuntimeError, match="AABB"):
        diagnostic._verify_mask_and_bounds_replay(
            plan=plan,
            acquisition=tmp_path,
            manifest=manifest,
            bounds=bad_bounds,
        )

    with np.load(compact, allow_pickle=False) as archive:
        original_arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    for name in (
        "reference_means_native",
        "reference_covariances_native",
        "reference_colors",
        "reference_amplitudes",
    ):
        tampered_arrays = copy.deepcopy(original_arrays)
        tampered_arrays[name].flat[0] += 0.1
        np.savez_compressed(compact, **tampered_arrays)
        with pytest.raises(RuntimeError, match=f"compact {name}.*strict hash-bound"):
            diagnostic._verify_mask_and_bounds_replay(
                plan=plan,
                acquisition=tmp_path,
                manifest=manifest,
                bounds=bounds,
            )
    np.savez_compressed(compact, **original_arrays)


def test_correspondence_npz_schema_preserves_nonfinite_semantics_explicitly(tmp_path) -> None:
    schema = diagnostic._correspondence_schema()
    arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in schema.items()}
    arrays["component_id"] = np.arange(
        diagnostic.REFERENCE_COMPONENT_COUNT,
        dtype=np.int64,
    )
    path = tmp_path / "correspondence.npz"
    np.savez_compressed(path, **arrays)
    assert set(diagnostic.validate_correspondence_npz(path)) == set(schema)

    condition_code = f"dlt_condition{diagnostic.NONFINITE_CODE_SUFFIX}"
    arrays[condition_code][4] = 2
    encoded = tmp_path / "correspondence_encoded_inf.npz"
    np.savez_compressed(encoded, **arrays)
    diagnostic.validate_correspondence_npz(encoded)
    restored = diagnostic.restore_archived_nonfinite(
        torch.from_numpy(arrays["dlt_condition"]),
        torch.from_numpy(arrays[condition_code]),
    )
    assert torch.isposinf(restored[4])

    unknown_code = copy.deepcopy(arrays)
    unknown_code[condition_code][4] = 4
    bad_code = tmp_path / "correspondence_bad_code.npz"
    np.savez_compressed(bad_code, **unknown_code)
    with pytest.raises(RuntimeError, match="unknown code"):
        diagnostic.validate_correspondence_npz(bad_code)

    wrong_replacement = copy.deepcopy(arrays)
    wrong_replacement["dlt_condition"][4] = 7.0
    bad_replacement = tmp_path / "correspondence_bad_replacement.npz"
    np.savez_compressed(bad_replacement, **wrong_replacement)
    with pytest.raises(RuntimeError, match="replacement zeros"):
        diagnostic.validate_correspondence_npz(bad_replacement)


def test_nonfinite_codec_distinguishes_nan_and_signed_infinities() -> None:
    raw = torch.tensor([1.5, torch.nan, torch.inf, -torch.inf], dtype=torch.float64)
    finite, code = diagnostic._encode_archive_float(raw)
    assert torch.equal(code, torch.tensor([0, 1, 2, 3], dtype=torch.uint8))
    assert torch.equal(finite, torch.tensor([1.5, 0.0, 0.0, 0.0], dtype=torch.float64))
    restored = diagnostic.restore_archived_nonfinite(finite, code)
    assert restored[0] == 1.5
    assert torch.isnan(restored[1])
    assert torch.isposinf(restored[2])
    assert torch.isneginf(restored[3])


def test_distribution_reports_each_nonfinite_class_without_hiding_positive_infinity() -> None:
    distribution = diagnostic._masked_distribution(
        torch.tensor([2.5, torch.nan, torch.inf, -torch.inf], dtype=torch.float64),
        torch.ones(4, dtype=torch.bool),
    )
    assert distribution["value_class_counts"] == {
        "total": 4,
        "finite": 1,
        "nan": 1,
        "positive_infinity": 1,
        "negative_infinity": 1,
    }
    assert distribution["finite_quantiles"] == {
        str(q): 2.5 for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def test_recovered_affine_condition_is_a_separate_fail_closed_gate() -> None:
    singular_values, condition = diagnostic.recovered_affine_condition(
        torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0e-3]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
    )
    assert torch.allclose(singular_values[0], torch.tensor([1.0, 1.0e-3]))
    assert condition[0] == pytest.approx(1000.0)
    gates = diagnostic.GateConfig(affine_recovered_max_condition=100.0)
    recovered_gate = torch.isfinite(condition) & (condition <= gates.affine_recovered_max_condition)
    assert recovered_gate.tolist() == [False, True]


def test_cross_pair_counts_overlap_and_disagreement_are_explicit() -> None:
    result = diagnostic.cross_pair_disagreement(
        torch.tensor([1, 2, 4], dtype=torch.int64),
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor([2, 3, 4], dtype=torch.int64),
        torch.tensor(
            [[1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    assert result["component_id"].tolist() == [2, 4]
    assert result["overlap_unique_component_count"] == 2
    assert result["union_unique_component_count"] == 4
    assert torch.equal(
        result["distance_world"],
        torch.tensor([0.5, 0.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="duplicate"):
        diagnostic.cross_pair_disagreement(
            torch.tensor([1, 1], dtype=torch.int64),
            torch.zeros(2, 3),
            torch.tensor([], dtype=torch.int64),
            torch.zeros(0, 3),
        )


def test_full_replay_verifier_rejects_impossible_geometry_and_output_tampering(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "run"
    acquisition = output / "acquisition"
    analysis = output / "analysis"
    correspondences = analysis / "correspondences"
    pair_plys = analysis / "pair_plys"
    acquisition.mkdir(parents=True)
    correspondences.mkdir(parents=True)
    pair_plys.mkdir()

    schema = diagnostic._correspondence_schema()
    archive_arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in schema.items()}
    archive_arrays["component_id"] = np.arange(
        diagnostic.REFERENCE_COMPONENT_COUNT,
        dtype=np.int64,
    )
    replay_arrays = {name: torch.from_numpy(value.copy()) for name, value in archive_arrays.items()}
    replay_summary = {
        "counts": {
            "marginal": {"accepted_geometry": 0},
            "sequential": {"dlt_valid": 0},
        },
        "fb_sensitivity": {
            "1.0": {"accepted_geometry_count": 0},
            "3.0": {"accepted_geometry_count": 0},
            "5.0": {"accepted_geometry_count": 0},
        },
    }
    empty_points = torch.empty((0, 3), dtype=torch.float64)
    empty_colors = torch.empty((0, 3), dtype=torch.float64)
    bounds_path = acquisition / "bounds.json"
    bounds_path.write_text(json.dumps({"extent": 2.0}))
    manifest = {
        "bounds": {"path": bounds_path.name},
        "pairs": {},
    }
    pair_results = {}
    for pair in diagnostic.PAIRS:
        flow_path = acquisition / f"{pair.pair_id}.npz"
        np.savez_compressed(
            flow_path,
            forward=np.zeros((1, 1, 2), dtype=np.float32),
            backward=np.zeros((1, 1, 2), dtype=np.float32),
        )
        manifest["pairs"][pair.pair_id] = {"flow_path": flow_path.name}
        correspondence_path = correspondences / f"{pair.pair_id}.npz"
        np.savez_compressed(correspondence_path, **archive_arrays)
        pair_ply_path = pair_plys / f"{pair.pair_id}.ply"
        diagnostic._save_mechanism_ply(
            pair_ply_path,
            diagnostic._mechanism_gaussians(
                empty_points,
                empty_colors,
                extent=2.0,
            ),
        )
        pair_results[pair.pair_id] = copy.deepcopy(replay_summary) | {
            "artifacts": {
                "correspondence": {
                    "path": f"correspondences/{pair.pair_id}.npz",
                },
                "pair_colored_ply": {
                    "path": f"pair_plys/{pair.pair_id}.ply",
                    "sample_count": 0,
                    "display_rgb": diagnostic.PAIR_DISPLAY_COLORS[pair.pair_id].tolist(),
                },
            }
        }

    cross_path = analysis / "cross_pair_disagreement.npz"
    np.savez_compressed(
        cross_path,
        component_id=np.empty((0,), dtype=np.int64),
        point_C0014_to_C0005=np.empty((0, 3), dtype=np.float64),
        point_C0014_to_C0026=np.empty((0, 3), dtype=np.float64),
        distance_world=np.empty((0,), dtype=np.float64),
    )
    lineage_path = analysis / "accepted_lineage.json"
    lineage = {
        "pair_sample_count": 0,
        "unique_component_count": 0,
        "rows": [],
    }
    lineage_path.write_text(json.dumps(lineage))
    combined_path = analysis / "accepted_pair_samples.ply"
    diagnostic._save_mechanism_ply(
        combined_path,
        diagnostic._mechanism_gaussians(empty_points, empty_colors, extent=2.0),
    )
    empty_quantiles = {str(q): None for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)}
    result = {
        "pairs": pair_results,
        "accepted_counts": {
            "pair_sample_count": 0,
            "unique_component_count": 0,
        },
        "pair_overlap": {
            "accepted_component_counts": {pair.pair_id: 0 for pair in diagnostic.PAIRS},
            "overlap_unique_component_count": 0,
            "union_unique_component_count": 0,
            "cross_pair_3d_disagreement_world": empty_quantiles,
            "artifact": {"path": cross_path.name},
        },
        "accepted_lineage": {"path": lineage_path.name},
        "combined_pair_samples": {"path": combined_path.name},
    }
    plan = {
        "gates": diagnostic.dataclasses.asdict(diagnostic.GateConfig()),
        "fb_sensitivity_px": list(diagnostic.FB_SENSITIVITY_PX),
    }
    monkeypatch.setattr(
        diagnostic,
        "_load_replay_inputs",
        lambda *args, **kwargs: (
            {
                diagnostic.REFERENCE_VIEW: None,
                **{target: None for target in diagnostic.TARGET_VIEWS},
            },
            {target: torch.empty(0) for target in diagnostic.TARGET_VIEWS},
            torch.empty(0),
            torch.empty(0),
            torch.empty(0, dtype=torch.bool),
            torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float64,
            ),
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "analyze_pair_tensors",
        lambda **kwargs: (
            {name: value.clone() for name, value in replay_arrays.items()},
            copy.deepcopy(replay_summary),
        ),
    )

    diagnostic._verify_full_analysis_replay(
        output,
        analysis=analysis,
        result=result,
        manifest=manifest,
        plan=plan,
    )

    first_pair = diagnostic.PAIRS[0].pair_id
    first_correspondence = correspondences / f"{first_pair}.npz"
    impossible = copy.deepcopy(archive_arrays)
    impossible["accepted_geometry"][0] = True
    np.savez_compressed(first_correspondence, **impossible)
    with pytest.raises(RuntimeError, match="without a DLT-valid"):
        diagnostic._verify_full_analysis_replay(
            output,
            analysis=analysis,
            result=result,
            manifest=manifest,
            plan=plan,
        )
    np.savez_compressed(first_correspondence, **archive_arrays)

    wrong_summary = copy.deepcopy(result)
    wrong_summary["pairs"][first_pair]["fb_sensitivity"]["3.0"]["accepted_geometry_count"] = 1
    with pytest.raises(RuntimeError, match="FB sensitivity"):
        diagnostic._verify_full_analysis_replay(
            output,
            analysis=analysis,
            result=wrong_summary,
            manifest=manifest,
            plan=plan,
        )

    lineage_path.write_text(
        json.dumps(
            {
                "pair_sample_count": 1,
                "unique_component_count": 1,
                "rows": [
                    {
                        "pair_sample_index": 0,
                        "pair_id": first_pair,
                        "component_id": 0,
                    }
                ],
            }
        )
    )
    with pytest.raises(RuntimeError, match="accepted lineage"):
        diagnostic._verify_full_analysis_replay(
            output,
            analysis=analysis,
            result=result,
            manifest=manifest,
            plan=plan,
        )
    lineage_path.write_text(json.dumps(lineage))

    original_pair_ply = (pair_plys / f"{first_pair}.ply").read_bytes()
    (pair_plys / f"{first_pair}.ply").write_bytes(original_pair_ply + b"tampered")
    with pytest.raises(RuntimeError, match="PLY does not replay"):
        diagnostic._verify_full_analysis_replay(
            output,
            analysis=analysis,
            result=result,
            manifest=manifest,
            plan=plan,
        )
    (pair_plys / f"{first_pair}.ply").write_bytes(original_pair_ply)

    np.savez_compressed(
        cross_path,
        component_id=np.array([0], dtype=np.int64),
        point_C0014_to_C0005=np.zeros((1, 3), dtype=np.float64),
        point_C0014_to_C0026=np.zeros((1, 3), dtype=np.float64),
        distance_world=np.zeros((1,), dtype=np.float64),
    )
    with pytest.raises(RuntimeError, match="shape"):
        diagnostic._verify_full_analysis_replay(
            output,
            analysis=analysis,
            result=result,
            manifest=manifest,
            plan=plan,
        )


def test_failure_phases_leave_receipts_without_partial_final_artifacts(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        diagnostic.acquire(tmp_path, device="cpu")
    receipts = list(tmp_path.glob("acquisition_failure_attempt_*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "FAILED"
    assert not (tmp_path / "acquisition").exists()

    with pytest.raises(FileNotFoundError):
        diagnostic.analyze(tmp_path)
    receipts = list(tmp_path.glob("analysis_failure_attempt_*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "FAILED"
    assert not (tmp_path / "analysis").exists()


def test_snapshot_receipt_is_output_aware_and_fail_closed(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "run"
    acquisition = output / "acquisition"
    analysis = output / "analysis"
    viewer_dir = output / "viewer"
    snapshot_dir = viewer_dir / "calibrated_snapshots"
    acquisition.mkdir(parents=True)
    analysis.mkdir()
    snapshot_dir.mkdir(parents=True)

    scene = tmp_path / "reporting_scene"
    rgb_dir = scene / "rgb"
    rgb_dir.mkdir(parents=True)
    for view in diagnostic.VIEWER_VIEWS:
        (rgb_dir / f"{view}.jpg").write_bytes(f"source-{view}".encode())
    monkeypatch.setattr(diagnostic, "SCENE", scene)

    source_calibration = tmp_path / "calibration_dome.json"
    source_calibration.write_text('{"cameras":[]}\n')
    monkeypatch.setattr(
        diagnostic,
        "_calibration_records",
        lambda: ({}, source_calibration),
    )

    camera = Camera(
        fx=4000.0,
        fy=4000.0,
        cx=diagnostic.NATIVE_WIDTH / 2,
        cy=diagnostic.NATIVE_HEIGHT / 2,
        width=diagnostic.NATIVE_WIDTH,
        height=diagnostic.NATIVE_HEIGHT,
        R=torch.eye(3),
        t=torch.zeros(3),
    )
    camera_payload = diagnostic.camera_record(camera)
    calibration_path = acquisition / "calibration.json"
    calibration_path.write_text(json.dumps({"native": {"C0014": camera_payload}}))
    (acquisition / "manifest.json").write_text(
        json.dumps({"calibration": {"path": calibration_path.name}})
    )
    (output / "plan.json").write_text('{"fixture":"plan"}\n')
    ply = analysis / "accepted_pair_samples.ply"
    ply.write_bytes(b"synthetic-ply-fixture")
    result = {
        "combined_pair_samples": {
            "path": ply.name,
            "sha256": diagnostic.sha256_file(ply),
        },
        "viewer": {"view_allowlist": list(diagnostic.VIEWER_VIEWS)},
    }
    (analysis / "result.json").write_text('{"fixture":"result"}\n')
    monkeypatch.setattr(diagnostic, "_verify_analysis", lambda candidate: result)

    snapshot = snapshot_dir / "C0014_native_gsplat.png"
    snapshot.write_bytes(b"synthetic-png-fixture")
    record = {
        "camera_view": "C0014",
        "camera_record_sha256": diagnostic.canonical_hash(camera_payload),
        "calibration_file_sha256": diagnostic.sha256_file(calibration_path),
        "source_ply_sha256": diagnostic.sha256_file(ply),
        "backend_class": "rtgs.render.gsplat_backend.GsplatRasterizer",
        "device": "cuda:0",
        "width": diagnostic.NATIVE_WIDTH,
        "height": diagnostic.NATIVE_HEIGHT,
        "packed": False,
        "antialiased": False,
        "started_utc": "2026-07-17T00:00:00.000000Z",
        "completed_utc": "2026-07-17T00:00:01.000000Z",
        "path": snapshot.name,
        "sha256": diagnostic.sha256_file(snapshot),
        "bytes": snapshot.stat().st_size,
    }
    diagnostic._validate_snapshot_record(record, directory=snapshot_dir)

    bad_backend = record | {"backend_class": "rtgs.render.torch_ref.TorchRasterizer"}
    with pytest.raises(RuntimeError, match="CUDA gsplat"):
        diagnostic._validate_snapshot_record(bad_backend, directory=snapshot_dir)
    bad_hash = record | {"sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="binding changed"):
        diagnostic._validate_snapshot_record(bad_hash, directory=snapshot_dir)

    token = "a" * 64
    process_pid = 1234
    marker = diagnostic._viewer_handshake_marker(
        token,
        pid=process_pid,
        port=8890,
    )
    command = diagnostic.viewer_command(output, handshake_token=token)
    shell = diagnostic.viewer_shell(command)
    assert shell.startswith(f"LD_PRELOAD={diagnostic.LOCAL_LIBSTDCXX_PRELOAD}")
    assert ",".join(diagnostic.VIEWER_VIEWS) in command
    assert diagnostic.HELDOUT_VIEW not in command
    receipt = {
        "artifact_type": diagnostic.VIEWER_ARTIFACT_TYPE,
        "status": "PASS",
        "completed_utc": "2026-07-17T00:00:02.000000Z",
        "plan_file_sha256": diagnostic.sha256_file(output / "plan.json"),
        "acquisition_manifest_sha256": diagnostic.sha256_file(acquisition / "manifest.json"),
        "analysis_result_sha256": diagnostic.sha256_file(analysis / "result.json"),
        "url": "http://127.0.0.1:8890",
        "http_status": 200,
        "last_error": None,
        "command": command,
        "shell": shell,
        "environment": {
            "LD_PRELOAD": str(diagnostic.LOCAL_LIBSTDCXX_PRELOAD),
        },
        "reporting_inputs": {
            "view_allowlist": list(diagnostic.VIEWER_VIEWS),
            "source_colors": diagnostic._viewer_source_bindings(),
            "calibration": diagnostic._file_binding(source_calibration),
        },
        "full_resolution": True,
        "rasterizer": "gsplat",
        "packed": False,
        "antialiased": False,
        "mandatory_native_snapshot": record,
        "process_handshake": {
            "token": token,
            "token_sha256": diagnostic.hashlib.sha256(token.encode()).hexdigest(),
            "pid": process_pid,
            "port": 8890,
            "marker": marker,
            "marker_observed": True,
            "listener_owner_verified": True,
        },
        "process_output_tail": marker,
    }
    diagnostic._validate_viewer_receipt(
        receipt,
        output=output,
        directory=viewer_dir,
    )

    with pytest.raises(RuntimeError, match="requires HTTP 200"):
        diagnostic._validate_viewer_receipt(
            receipt | {"http_status": 500},
            output=output,
            directory=viewer_dir,
        )
    with pytest.raises(RuntimeError, match="upstream hash chain"):
        diagnostic._validate_viewer_receipt(
            receipt | {"plan_file_sha256": "0" * 64},
            output=output,
            directory=viewer_dir,
        )
    with pytest.raises(RuntimeError, match="command/view binding"):
        diagnostic._validate_viewer_receipt(
            receipt | {"command": [*command, "--unexpected"]},
            output=output,
            directory=viewer_dir,
        )
    wrong_process = copy.deepcopy(receipt)
    wrong_process["process_handshake"]["listener_owner_verified"] = False
    with pytest.raises(RuntimeError, match="listening-socket ownership"):
        diagnostic._validate_viewer_receipt(
            wrong_process,
            output=output,
            directory=viewer_dir,
        )
    leaked_inputs = copy.deepcopy(receipt)
    leaked_inputs["reporting_inputs"]["view_allowlist"][-1] = diagnostic.HELDOUT_VIEW
    with pytest.raises(ValueError, match="frozen"):
        diagnostic._validate_viewer_receipt(
            leaked_inputs,
            output=output,
            directory=viewer_dir,
        )
    wrong_camera = copy.deepcopy(receipt)
    wrong_camera["mandatory_native_snapshot"]["camera_record_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="camera/calibration/PLY binding"):
        diagnostic._validate_viewer_receipt(
            wrong_camera,
            output=output,
            directory=viewer_dir,
        )

    (viewer_dir / "viewer_receipt.json").write_text(json.dumps(receipt))
    assert diagnostic._verify_viewer(output) == receipt

    # Even a source-byte change after receipt publication invalidates reporting provenance.
    (rgb_dir / f"{diagnostic.VIEWER_VIEWS[0]}.jpg").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="source-color bindings"):
        diagnostic._verify_viewer(output)


def test_viewer_handshake_rejects_occupied_port_and_binds_listener_pid() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = int(listener.getsockname()[1])
        with pytest.raises(RuntimeError, match="already occupied"):
            diagnostic._assert_port_available(port)
        assert diagnostic._process_owns_listening_port(diagnostic.os.getpid(), port)
        assert not diagnostic._process_owns_listening_port(diagnostic.os.getpid() + 1_000_000, port)


def test_reporting_viewer_loads_only_the_exact_training_allowlist(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "run"
    acquisition = output / "acquisition"
    acquisition.mkdir(parents=True)
    (acquisition / "manifest.json").write_text(json.dumps({"bounds": {"path": "bounds.json"}}))
    (acquisition / "bounds.json").write_text(json.dumps({"center": [0.0, 0.0, 0.0], "extent": 2.0}))
    monkeypatch.setattr(diagnostic, "_verify_analysis", lambda candidate: {})
    monkeypatch.setattr(diagnostic, "NATIVE_WIDTH", 2)
    monkeypatch.setattr(diagnostic, "NATIVE_HEIGHT", 2)
    records = {view: {} for view in diagnostic.VIEWER_VIEWS}
    monkeypatch.setattr(
        diagnostic,
        "_calibration_records",
        lambda: (records, tmp_path / "calibration.json"),
    )
    loaded: list[str] = []

    def camera_for_view(view, record):
        assert record is records[view]
        return Camera(
            fx=2.0,
            fy=2.0,
            cx=1.0,
            cy=1.0,
            width=2,
            height=2,
            R=torch.eye(3),
            t=torch.zeros(3),
        )

    def load_rgb(view, camera, record):
        loaded.append(view)
        assert record is records[view]
        return torch.zeros(camera.height, camera.width, 3)

    monkeypatch.setattr(diagnostic, "_camera_for_native_view", camera_for_view)
    monkeypatch.setattr(diagnostic, "_load_undistorted_rgb", load_rgb)
    scene = diagnostic._load_reporting_viewer_scene(output, diagnostic.VIEWER_VIEWS)
    assert tuple(scene.view_names) == diagnostic.VIEWER_VIEWS
    assert scene.training_views == list(range(len(diagnostic.VIEWER_VIEWS)))
    assert scene.testing_views == []
    assert tuple(loaded) == diagnostic.VIEWER_VIEWS
    assert diagnostic.HELDOUT_VIEW not in loaded

    with pytest.raises(ValueError, match="frozen"):
        diagnostic._load_reporting_viewer_scene(
            output,
            (*diagnostic.VIEWER_VIEWS[:-1], diagnostic.HELDOUT_VIEW),
        )
