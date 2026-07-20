"""Fail-closed toy tests for the stage-1 weight/color gauge audit."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians2d import Gaussians2D
from rtgs.data.scene import SceneData
from rtgs.data.synthetic import make_ring_cameras

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "stage1_weight_gauge_audit.py"
_SPEC = importlib.util.spec_from_file_location("stage1_weight_gauge_audit", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def _toy_gaussians2d() -> Gaussians2D:
    return Gaussians2D(
        xy=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
        chol=torch.tensor([[1.0, 0.0, 1.0], [1.1, 0.1, 0.9], [0.8, -0.1, 1.2], [1.0, 0.0, 1.0]]),
        color=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.4, 0.8], [0.0, 0.5, 1.0], [1.0, 0.5, 0.25]]),
        weight=torch.tensor([0.0, 0.5, 0.2, 1.0]),
    )


def _prepared(
    *,
    seed: int,
    scene: SceneData,
    gauges: dict[str, list[Gaussians2D]],
    changed: int,
) -> bench.PreparedSeed:
    count = sum(view.n for view in gauges["identity"])
    aggregate = {
        gauge: {
            "component_count": count,
            "joint_weight_color_changed_count": 0 if gauge == "identity" else changed,
        }
        for gauge in bench.GAUGES
    }
    return bench.PreparedSeed(
        seed=seed,
        scene=scene,
        fitted=gauges["identity"],
        gauges=gauges,
        preparation={},
        transformations={"aggregate": aggregate},
    )


def test_harness_exposes_only_outcome_free_seal_and_scientific_audit_actions():
    completed = subprocess.run(
        [sys.executable, str(_PATH), "--help"],
        cwd=_PATH.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{seal,audit}" in completed.stdout
    assert [list(command) for command in bench.VERIFICATION_COMMANDS] == [
        [".venv/bin/python", "-m", "ruff", "check", "."],
        [".venv/bin/python", "-m", "ruff", "format", "--check", "."],
        [".venv/bin/python", "-m", "pytest", "-q", "-m", "not slow"],
        [".venv/bin/python", "scripts/docs_sync.py"],
        ["git", "diff", "--check"],
    ]
    assert list(bench.VERIFICATION_LITERAL_COMMANDS) == [
        ".venv/bin/python -m ruff check .",
        ".venv/bin/python -m ruff format --check .",
        '.venv/bin/python -m pytest -q -m "not slow"',
        ".venv/bin/python scripts/docs_sync.py",
        "git diff --check",
    ]


def test_exact_gauges_preserve_bounds_product_order_and_zero_amplitude():
    source = _toy_gaussians2d()
    gauges, evidence = bench.construct_gauges([source], seed=91)
    amplitude = source.weight[:, None] * source.color
    for name in bench.GAUGES:
        gauge = gauges[name][0]
        assert torch.equal(gauge.xy, source.xy)
        assert torch.equal(gauge.chol, source.chol)
        assert bool(((gauge.weight >= 0) & (gauge.weight <= 1)).all())
        assert bool(((gauge.color >= 0) & (gauge.color <= 1)).all())
        assert torch.allclose(gauge.weight[:, None] * gauge.color, amplitude, atol=1e-7, rtol=1e-6)
        assert sum(evidence["per_view"][name][0]["weight_bin_counts"]) == source.n
        assert sum(evidence["per_view"][name][0]["peak_color_bin_counts"]) == source.n
    assert gauges["unit_weight"][0].weight[0] == 1
    assert torch.equal(gauges["unit_weight"][0].color[0], torch.zeros(3))
    assert gauges["peak_color"][0].weight[0] == 0
    assert torch.equal(gauges["peak_color"][0].color[0], torch.zeros(3))
    assert torch.equal(gauges["peak_color"][0].color[2], torch.tensor([0.0, 0.5, 1.0]))


def test_fitted_shape_validation_explicitly_rejects_noncontract_xy():
    malformed = Gaussians2D(
        xy=torch.zeros(150, 3),
        chol=torch.tensor([[1.0, 0.0, 1.0]]).repeat(150, 1),
        color=torch.zeros(150, 3),
        weight=torch.zeros(150),
    )
    with pytest.raises(bench.AuditInvalid, match="xy shape differs"):
        bench.validate_fitted_views([malformed] * len(bench.TRAIN_INDICES))


def test_source_equivalence_failure_never_invokes_downstream(monkeypatch):
    gaussian = _toy_gaussians2d()
    gauges = {name: [gaussian.detach() for _ in bench.TRAIN_INDICES] for name in bench.GAUGES}
    item = bench.PreparedSeed(
        seed=0,
        scene=None,
        fitted=gauges["identity"],
        gauges=gauges,
        preparation={},
        transformations={},
    )
    calls = 0

    def unequal_render(*args, **kwargs):
        nonlocal calls
        calls += 1
        return torch.ones(48, 48, 3) if calls == 1 else torch.zeros(48, 48, 3)

    monkeypatch.setattr(bench, "render_gaussians_2d", unequal_render)
    reached_downstream = False

    def downstream(_equivalence):
        nonlocal reached_downstream
        reached_downstream = True
        bench.render_gaussian_coverage_2d(gaussian, 48, 48)
        return {}

    with pytest.raises(bench.AuditInvalid, match="source equivalence failed"):
        bench.run_after_source_equivalence([item], downstream)
    assert not reached_downstream
    assert calls == 2
    payload = bench.invalid_payload(
        stage="source_equivalence",
        reason="toy failure",
        preparations=[{}],
        transformations=[{}],
        equivalence={"passed": False},
        evidence={"completed_checks": 0},
    )
    assert not ({"coverage_and_retention", "depth", "carve", "materiality"} & payload.keys())
    assert payload["decision"] is None


def test_source_keys_and_set_comparisons_are_exact_and_seed_tagged():
    identity = [bench.source_key(0, 0, 1), bench.source_key(0, 0, 2)]
    transformed = [bench.source_key(0, 0, 2), bench.source_key(0, 1, 1)]
    comparison = bench.key_set_comparison(identity, transformed)
    assert comparison["intersection_keys"] == [[0, 0, 2]]
    assert comparison["symmetric_difference_keys"] == [[0, 0, 1], [0, 1, 1]]
    assert comparison["union_count"] == 3
    assert comparison["set_disagreement"] == pytest.approx(2 / 3)
    assert bench.source_key(1, 0, 1) != bench.source_key(0, 0, 1)


def test_ordered_raw_sum_pool_and_frozen_materiality_boundaries():
    assert bench.ordered_float64_sum([0.1] * 9) == 0.8999999999999999
    input_gate = bench.input_materiality_from_raw(
        joint_changed_count=10,
        source_component_count=100,
        retention_symmetric_difference_count=10,
        retention_union_count=1_000,
        coverage_delta_l1=0.0,
        coverage_reference_l1=1.0,
        coverage_crossing_count=0,
        coverage_pixel_count=1_000,
    )
    assert input_gate["input_consumption_material"]
    coverage_gate = bench.input_materiality_from_raw(
        joint_changed_count=10,
        source_component_count=100,
        retention_symmetric_difference_count=0,
        retention_union_count=1_000,
        coverage_delta_l1=1.0,
        coverage_reference_l1=100.0,
        coverage_crossing_count=1,
        coverage_pixel_count=1_000,
    )
    assert coverage_gate["input_consumption_material"]
    below = bench.input_materiality_from_raw(
        joint_changed_count=9,
        source_component_count=100,
        retention_symmetric_difference_count=10,
        retention_union_count=1_000,
        coverage_delta_l1=0.01,
        coverage_reference_l1=1.0,
        coverage_crossing_count=1,
        coverage_pixel_count=1_000,
    )
    assert not below["input_consumption_material"]
    lift_gate = bench.lift_materiality_from_raw(
        joint_changed_count=10,
        source_component_count=100,
        set_symmetric_difference_count=10,
        set_union_count=1_000,
        render_delta_l1=0.0,
        identity_signal_l1=1.0,
        identity_residual_l1=1.0,
    )
    assert lift_gate["lift_material"]
    render_gate = bench.lift_materiality_from_raw(
        joint_changed_count=10,
        source_component_count=100,
        set_symmetric_difference_count=0,
        set_union_count=1_000,
        render_delta_l1=1.0,
        identity_signal_l1=1_000.0,
        identity_residual_l1=100.0,
    )
    assert render_gate["lift_material"]


def _fake_seed_record(seed: int, unit_pass: bool, peak_pass: bool) -> dict:
    identity = [[seed, 0, index] for index in range(20)]

    def comparison(passes: bool) -> dict:
        transformed = identity[10:] if passes else list(identity)
        return {
            "output_key_sets": {
                "identity_keys": identity,
                "transformed_keys": transformed,
            },
            "render_comparison": {
                "render_delta_l1": 1.0 if passes else 0.0,
                "identity_signal_l1": 10.0,
                "identity_residual_l1": 10.0,
                "alpha_delta_l1": 0.0,
                "accumulated_depth_delta_l1": 0.0,
            },
            "materiality": {"lift_material": passes},
        }

    return {
        "seed": seed,
        "comparisons": {
            "unit_weight": comparison(unit_pass),
            "peak_color": comparison(peak_pass),
        },
    }


def _fake_prepared(seed: int) -> bench.PreparedSeed:
    aggregate = {
        transform: {
            "joint_weight_color_changed_count": 100,
            "component_count": 100,
        }
        for transform in bench.TRANSFORMS
    }
    return bench.PreparedSeed(
        seed=seed,
        scene=None,
        fitted=[],
        gauges={},
        preparation={},
        transformations={"aggregate": aggregate},
    )


def test_backend_decision_requires_same_transform_in_two_of_three_seeds():
    prepared = [_fake_prepared(seed) for seed in bench.SEEDS]
    split_wins = [
        _fake_seed_record(0, True, False),
        _fake_seed_record(1, False, True),
        _fake_seed_record(2, False, False),
    ]
    assert not bench.pool_lift_backend(split_wins, prepared)["backend_materially_gauge_dependent"]
    same_transform = [
        _fake_seed_record(0, True, False),
        _fake_seed_record(1, True, False),
        _fake_seed_record(2, False, True),
    ]
    decision = bench.pool_lift_backend(same_transform, prepared)
    assert decision["backend_materially_gauge_dependent"]
    assert decision["qualifying_transforms"] == ["unit_weight"]


def _one_camera_depth_fixture() -> bench.PreparedSeed:
    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, -2.0]),
        torch.zeros(3),
        width=8,
        height=8,
        fov_x_deg=45,
    )
    scene = SceneData(
        images=[torch.full((8, 8, 3), 0.3)],
        cameras=[camera],
        gt_depths=[torch.ones(8, 8)],
        gt_gaussians=None,
        bounds_hint=(torch.zeros(3), 1.0),
        train_indices=[0],
        test_indices=[],
        name="toy-depth-gauge",
    )
    identity = Gaussians2D(
        xy=torch.tensor([[4.0, 4.0], [5.0, 4.0]]),
        chol=torch.tensor([[0.8, 0.0, 0.8], [0.8, 0.0, 0.8]]),
        color=torch.tensor([[0.4, 0.3, 0.2], [0.2, 0.2, 0.2]]),
        weight=torch.tensor([0.2, 0.05]),
    )
    gauges, _ = bench.construct_gauges([identity], seed=99)
    return _prepared(seed=99, scene=scene, gauges=gauges, changed=2)


def test_depth_masks_bind_exact_keys_and_shared_geometry_control():
    item = _one_camera_depth_fixture()
    identity = bench.run_depth_gauge(item, "identity")
    unit = bench.run_depth_gauge(item, "unit_weight")
    assert identity.keys == [(99, 0, 0)]
    assert unit.keys == [(99, 0, 0), (99, 0, 1)]
    comparison = bench.compare_lifts(
        identity,
        unit,
        scene=item.scene,
        changed_counts=item.transformations["aggregate"]["unit_weight"],
        require_shared_geometry=True,
    )
    assert comparison["shared_geometry_control_passed"]
    assert comparison["shared_key_deltas"]["maximum_center_displacement_over_extent"] == 0.0
    assert comparison["shared_key_deltas"]["maximum_relative_covariance_frobenius_delta"] == 0.0


def _carve_fixture() -> tuple[bench.PreparedSeed, dict]:
    cameras = make_ring_cameras(n_cameras=3, distance=2.0, image_size=12)
    color = torch.tensor([0.4, 0.3, 0.2])
    scene = SceneData(
        images=[color[None, None, :].expand(12, 12, 3).clone() for _ in cameras],
        cameras=cameras,
        masks=[torch.ones(12, 12) for _ in cameras],
        gt_gaussians=None,
        bounds_hint=(torch.zeros(3), 1.0),
        train_indices=[0, 1, 2],
        test_indices=[],
        name="toy-carve-sidecar",
    )
    views = [
        Gaussians2D(
            xy=torch.tensor([[6.0, 6.0]]),
            chol=torch.tensor([[2.0, 0.0, 2.0]]),
            color=color[None, :].clone(),
            weight=torch.tensor([0.8]),
        )
        for _ in cameras
    ]
    gauges = {name: [view.detach() for view in views] for name in bench.GAUGES}
    config = {
        **bench.carve_kwargs(),
        "grid_res": 8,
        "samples_per_ray": 16,
    }
    return _prepared(seed=77, scene=scene, gauges=gauges, changed=0), config


def test_carve_sidecar_matches_one_ordinary_lifter_call_on_toy_fixture():
    item, config = _carve_fixture()
    runtime = bench.run_carve_gauge(item, "identity", config=config)
    assert runtime.output.n == 3
    assert runtime.keys == [(77, 0, 0), (77, 1, 0), (77, 2, 0)]
    assert runtime.evidence["ordinary_lifter_call_count"] == 1
    assert max(runtime.evidence["sidecar_parity_errors"].values()) <= 2e-6
    sidecar = runtime.evidence["sidecar"]
    assert sidecar["source_counts"] == {"total": 3, "keep": 3, "valid_ray": 3, "placed": 3}
    assert set(sidecar["volume_hashes"]) >= {
        "n_seen",
        "n_covered",
        "hull",
        "consistency",
    }


def test_output_preflight_and_attempt_marker_refuse_overwrite(tmp_path):
    prospective = tmp_path / "20260716T000000Z_cpu_stage1_weight_gauge_audit.json"
    invalid = bench.invalid_output_path(prospective)
    bench.preflight_audit_outputs(prospective)
    assert invalid.name.endswith("_invalid.json")
    invalid.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to start"):
        bench.preflight_audit_outputs(prospective)

    marker = tmp_path / "attempt.json"
    invalid.unlink()
    bench.claim_attempt(
        marker,
        prospective_output=prospective,
        invalid_output=invalid,
        inputs={"seal_sha256": "a" * 64},
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["prospective_valid_output"] == str(prospective)
    assert payload["derived_invalid_output"] == str(invalid)
    with pytest.raises(RuntimeError, match="already been claimed"):
        bench.claim_attempt(
            marker,
            prospective_output=prospective,
            invalid_output=invalid,
            inputs={"seal_sha256": "a" * 64},
        )
    bench.validate_official_output_path(
        Path("benchmarks/results/20260716T000000Z_cpu_stage1_weight_gauge_audit.json")
    )
    with pytest.raises(ValueError, match="UTC audit filename"):
        bench.validate_official_output_path(Path("benchmarks/results/gauge_audit.json"))


def test_seal_covers_protocol_harness_source_and_focused_tests_without_attempt():
    sealed = {str(path) for path in bench.SEALED_PATHS}
    assert str(bench.PREREGISTRATION) in sealed
    assert "benchmarks/stage1_weight_gauge_audit.py" in sealed
    assert "src/rtgs/lift/carve.py" in sealed
    assert "src/rtgs/lift/depth.py" in sealed
    assert "tests/test_stage1_weight_gauge_audit.py" in sealed
    assert str(bench.ATTEMPT.relative_to(bench.ROOT)) not in sealed
