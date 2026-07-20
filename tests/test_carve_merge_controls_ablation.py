"""Fail-closed tests for the preregistered Carve merge-controls harness."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat
from rtgs.data.synthetic import make_synthetic_scene

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "carve_merge_controls_ablation.py"
_SPEC = importlib.util.spec_from_file_location("carve_merge_controls_ablation", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def _raw_with_scales(means: torch.Tensor, scales: torch.Tensor) -> Gaussians3D:
    count = means.shape[0]
    generator = torch.Generator().manual_seed(91)
    return Gaussians3D(
        means=means,
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        log_scales=scales[:, None].repeat(1, 3).log(),
        opacity=torch.full((count,), 0.5),
        sh=torch.randn(count, 1, 3, generator=generator) * 0.1,
    )


def _small_phase_a_record() -> dict:
    raw = _raw_with_scales(
        torch.tensor(
            [
                [-0.9, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ]
        ),
        torch.tensor([0.1, 0.1, 0.4, 0.2, 0.3, 0.5]),
    )
    arms, construction = bench.construct_arms(raw, 1.0)
    audit = bench.audit_arm_construction(raw, arms, construction, parity=arms["moment"].detach())
    views = [
        {
            "view": view,
            "raw_residual_l1": 100.0,
            "moment_control_color_l1": {
                "voxel_representative": 1.0,
                "global_budget_prune": 1.0,
            },
        }
        for view in bench.TRAIN_INDICES
    ]
    denominator, numerators = bench.materiality_totals_from_views(views)
    return {
        "preparation": {"raw_count": raw.n},
        "construction": construction,
        "construction_audit": audit,
        "render_materiality": {
            "views": views,
            "raw_residual_l1": denominator,
            "moment_control_color_l1": numerators,
            "ratios": {
                control: numerator / denominator for control, numerator in numerators.items()
            },
        },
    }


def test_harness_supports_preregistered_direct_cli_invocation():
    completed = subprocess.run(
        [sys.executable, str(_PATH), "--help"],
        cwd=_PATH.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{seal,audit,ablate}" in completed.stdout


def test_controls_use_world_origin_groups_exact_counts_and_frozen_ties():
    raw = _raw_with_scales(
        torch.tensor(
            [
                [-0.9, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ]
        ),
        torch.tensor([0.1, 0.1, 0.4, 0.2, 0.3, 0.5]),
    )
    arms, construction = bench.construct_arms(raw, 1.0)
    assert construction["unique_keys"] == [[-1, 0, 0], [0, 0, 0], [1, 0, 0]]
    assert construction["representative_indices"] == [0, 2, 5]
    assert construction["global_budget_indices"] == [2, 4, 5]
    assert {arms[name].n for name in bench.ARMS} == {3}
    assert bench.gaussians_hash(arms["moment"]) == bench.gaussians_hash(
        bench.merge_by_voxel(raw, 1.0, opacity_mode="mean")
    )
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        assert torch.equal(
            getattr(arms["voxel_representative"], field),
            getattr(raw, field)[torch.tensor([0, 2, 5])],
        )
        assert torch.equal(
            getattr(arms["global_budget_prune"], field),
            getattr(raw, field)[torch.tensor([2, 4, 5])],
        )


def test_float64_moment_audit_reconstructs_native_moments_and_structure():
    cells = torch.arange(300, dtype=torch.float32)
    means = torch.stack(
        [
            torch.stack([cells + 0.1, torch.zeros_like(cells), torch.zeros_like(cells)], dim=-1),
            torch.stack([cells + 0.2, torch.zeros_like(cells), torch.zeros_like(cells)], dim=-1),
        ],
        dim=1,
    ).reshape(-1, 3)
    raw = _raw_with_scales(means, torch.full((600,), 0.05))
    arms, construction = bench.construct_arms(raw, 1.0)
    audit = bench.audit_arm_construction(raw, arms, construction, parity=arms["moment"].detach())
    assert audit["structural_valid"]
    assert audit["moment_identity_pass"]
    assert audit["group_count"] == 300
    assert audit["group_count_distribution"]["multi_member_groups"] == 300
    assert audit["covariance_minimum_eigenvalue"] > 0.0


def test_raw_covariance_expectation_promotes_parameters_before_math():
    raw = Gaussians3D(
        means=torch.zeros(2, 3),
        quats=torch.tensor(
            [[0.91, 0.13, -0.27, 0.31], [0.47, -0.22, 0.71, 0.09]],
            dtype=torch.float32,
        ),
        log_scales=torch.tensor(
            [[-3.17, -2.03, -1.11], [-4.21, -2.72, -0.83]], dtype=torch.float32
        ),
        opacity=torch.tensor([0.2, 0.7]),
        sh=torch.zeros(2, 1, 3),
    )
    actual = bench.covariance_float64_from_raw(raw)
    rotations = quat_to_rotmat(raw.quats.double())
    variance = torch.diag_embed(raw.log_scales.double().exp().square())
    expected = rotations @ variance @ rotations.transpose(-1, -2)
    assert actual.dtype == torch.float64
    assert torch.allclose(actual, expected, atol=1e-15, rtol=1e-13)
    assert not torch.equal(actual, raw.covariance().double())


def test_structural_size_floors_become_serializable_gate_failures_not_abortions():
    record = _small_phase_a_record()
    gate = bench.phase_a_seed_gate_from_raw_evidence(record)
    assert not gate["pass"]
    assert not gate["criteria"]["raw_count_at_least_500"]
    assert not gate["criteria"]["group_count_at_least_100"]
    assert gate["criteria"]["structural_and_moment_identities_pass"]

    for view in record["render_materiality"]["views"]:
        view["moment_control_color_l1"]["global_budget_prune"] = 0.4
    denominator, numerators = bench.materiality_totals_from_views(
        record["render_materiality"]["views"]
    )
    record["render_materiality"]["moment_control_color_l1"] = numerators
    record["render_materiality"]["ratios"] = {
        control: numerator / denominator for control, numerator in numerators.items()
    }
    failed_materiality = bench.phase_a_seed_gate_from_raw_evidence(record)
    assert not failed_materiality["criteria"][
        "moment_vs_global_budget_prune_render_ratio_at_least_0_005"
    ]


def test_positive_left_fold_discriminator_reproduces_all_materiality_reductions_exactly():
    values = [0.1] * len(bench.TRAIN_INDICES)
    folded = bench.left_fold_float64(values)
    assert folded == 0.8999999999999999
    if sys.version_info[:2] == (3, 12):
        assert sum(values) == 0.9
        assert sum(values) != folded

    views = [
        {
            "view": view,
            "raw_residual_l1": value,
            "moment_control_color_l1": {
                "voxel_representative": value,
                "global_budget_prune": value,
            },
        }
        for view, value in zip(bench.TRAIN_INDICES, values)
    ]
    denominator, numerators = bench.materiality_totals_from_views(views)
    assert denominator == folded
    assert numerators == {
        "voxel_representative": folded,
        "global_budget_prune": folded,
    }

    record = _small_phase_a_record()
    record["render_materiality"] = {
        "views": views,
        "raw_residual_l1": denominator,
        "moment_control_color_l1": numerators,
        "ratios": {control: numerator / denominator for control, numerator in numerators.items()},
    }
    gate = bench.phase_a_seed_gate_from_raw_evidence(record)
    assert gate["criteria"]["moment_vs_voxel_representative_render_ratio_at_least_0_005"]
    assert gate["criteria"]["moment_vs_global_budget_prune_render_ratio_at_least_0_005"]


def test_phase_a_materiality_rejects_changed_per_view_numerator():
    record = _small_phase_a_record()
    record["render_materiality"]["views"][0]["moment_control_color_l1"]["global_budget_prune"] = 0.0
    with pytest.raises(RuntimeError, match="control numerator differs"):
        bench.phase_a_seed_gate_from_raw_evidence(record)


def test_phase_a_materiality_rejects_changed_reported_denominator_aggregate():
    record = _small_phase_a_record()
    record["render_materiality"]["raw_residual_l1"] += 1.0
    with pytest.raises(RuntimeError, match="reported residual differs"):
        bench.phase_a_seed_gate_from_raw_evidence(record)


def test_phase_a_materiality_rejects_changed_reported_control_aggregate():
    record = _small_phase_a_record()
    record["render_materiality"]["moment_control_color_l1"]["voxel_representative"] += 1.0
    with pytest.raises(RuntimeError, match="control numerator differs"):
        bench.phase_a_seed_gate_from_raw_evidence(record)


def test_phase_a_materiality_rejects_changed_reported_ratio():
    record = _small_phase_a_record()
    record["render_materiality"]["ratios"]["voxel_representative"] += 1.0
    with pytest.raises(RuntimeError, match="render ratio differs"):
        bench.phase_a_seed_gate_from_raw_evidence(record)


def test_checkpoint_auc_uses_frozen_zero_through_120_trapezoid():
    checkpoints = [
        {"step": step, "metrics": {"mean_psnr_fg": value}}
        for step, value in zip(bench.CHECKPOINT_STEPS, [10.0, 11.0, 12.0, 13.0, 14.0])
    ]
    assert bench.checkpoint_auc(checkpoints) == pytest.approx(12.0)
    checkpoints[-1]["step"] = 119
    with pytest.raises(ValueError, match="checkpoint steps differ"):
        bench.checkpoint_auc(checkpoints)


def test_heldout_metrics_serialize_recomputable_raw_numerators():
    scene = make_synthetic_scene(n_gaussians=5, n_cameras=12, image_size=10, seed=92)
    truths = bench.truth_cache(scene, bench.TEST_INDICES)
    inputs = bench.heldout_checkpoint_inputs(scene, truths)
    checkpoint = bench.evaluate_checkpoint(scene.gt_gaussians, inputs)
    for view in checkpoint["per_view"]:
        raw = view["raw_numerators"]
        foreground_mse = raw["foreground_rgb_squared_error_sum"] / raw["foreground_rgb_value_count"]
        full_mse = raw["full_rgb_squared_error_sum"] / raw["full_rgb_value_count"]
        assert view["psnr_fg"] == pytest.approx(
            -10.0 * math.log10(max(foreground_mse, 1e-12)), abs=2e-5
        )
        assert view["psnr_full"] == pytest.approx(
            -10.0 * math.log10(max(full_mse, 1e-12)), abs=2e-5
        )

    metrics = bench.evaluate_views(scene, scene.gt_gaussians, bench.TEST_INDICES, truths)
    for view in metrics["per_view"]:
        raw = view["raw_numerators"]
        expected_depth = (
            math.sqrt(raw["depth_squared_error_sum"] / raw["depth_intersection_pixel_count"])
            / raw["scene_extent"]
        )
        assert view["depth_rmse_over_extent"] == pytest.approx(expected_depth, abs=1e-8)
        assert view["alpha_iou"] == pytest.approx(
            raw["alpha_intersection_pixel_count"] / raw["alpha_union_pixel_count"]
        )
        assert view["foreground_coverage"] == pytest.approx(
            raw["alpha_intersection_pixel_count"] / raw["truth_support_pixel_count"]
        )


def test_isolated_schedule_probe_is_bound_but_arm_independent():
    hashes = {arm: str(index) * 64 for index, arm in enumerate(bench.ARMS, start=1)}
    probe = bench.isolated_schedule_probe(2, hashes)
    assert probe["all_arm_schedules_equal"]
    schedules = [probe["arms"][arm]["local_schedule"] for arm in bench.ARMS]
    assert schedules[0] == schedules[1] == schedules[2]
    assert len(schedules[0]) == 120
    assert probe["arms"]["moment"]["arm_hash"] == hashes["moment"]


def _passing_summary() -> dict:
    summary = {}
    for arm in bench.ARMS:
        auc = 20.2 if arm == "moment" else 20.0
        metrics = {}
        for metric in bench.METRICS:
            value = 20.0 if metric.startswith("psnr") else 0.8
            if metric == "depth_rmse_over_extent":
                value = 0.1
            metrics[metric] = {"samples": [value] * 3, "mean": value, "stdev": 0.0}
        summary[arm] = {
            "checkpoint_auc_psnr_fg_db": {
                "samples": [auc] * 3,
                "mean": auc,
                "stdev": 0.0,
            },
            "final_metrics": metrics,
        }
    return summary


def test_ablation_decision_requires_auc_superiority_over_each_control_and_safety():
    decision = bench.ablation_decision(_passing_summary())
    assert decision["primary_hypothesis_pass"]
    failed = _passing_summary()
    failed["global_budget_prune"]["checkpoint_auc_psnr_fg_db"] = {
        "samples": [20.2] * 3,
        "mean": 20.2,
        "stdev": 0.0,
    }
    assert not bench.ablation_decision(failed)["primary_hypothesis_pass"]
    unsafe = _passing_summary()
    unsafe["moment"]["final_metrics"]["ssim_crop"] = {
        "samples": [0.79] * 3,
        "mean": 0.79,
        "stdev": 0.0,
    }
    assert not bench.ablation_decision(unsafe)["primary_hypothesis_pass"]


def test_construction_record_raw_hashes_fail_closed_on_tampering():
    raw = _raw_with_scales(
        torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )
    _, construction = bench.construct_arms(raw, 1.0)
    bench._validate_construction_record(construction)
    construction["group_ids"][0] = 1
    with pytest.raises(RuntimeError, match="hashes differ"):
        bench._validate_construction_record(construction)


def test_phase_a_review_and_attempt_are_strictly_bound_and_once_only(tmp_path):
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")
    seal = {"sha256": "a" * 64, "source_aggregate": "b" * 64}
    review_path = tmp_path / "review.json"
    payload = {
        "artifact_type": "carve_merge_controls_iter2_phase_a_scientist_review",
        "verdict": "pass",
        "phase_b_execution_clearance": True,
        "audit_sha256": bench.sha256_file(audit_path),
        "seal_sha256": seal["sha256"],
        "source_aggregate": seal["source_aggregate"],
    }
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    assert bench.verify_phase_a_review(review_path, audit_path=audit_path, seal=seal)["sha256"]
    payload["extra"] = True
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected keys"):
        bench.verify_phase_a_review(review_path, audit_path=audit_path, seal=seal)

    marker = tmp_path / "attempt.json"
    output = tmp_path / "result.json"
    bench.claim_attempt(marker, phase="phase_a", output=output, inputs={"seal_sha256": "c" * 64})
    assert json.loads(marker.read_text(encoding="utf-8"))["artifact_type"] == (
        "carve_merge_controls_iter2_once_only_attempt"
    )
    with pytest.raises(RuntimeError, match="already been claimed"):
        bench.claim_attempt(
            marker, phase="phase_a", output=output, inputs={"seal_sha256": "c" * 64}
        )


def test_seal_covers_protocol_harness_all_source_and_tests_without_attempts():
    sealed = {str(path) for path in bench.SEALED_PATHS}
    assert "benchmarks/carve_merge_controls_ablation.py" in sealed
    assert "benchmarks/results/20260715_carve_merge_controls_PREREG.md" in sealed
    assert "benchmarks/results/20260716_carve_merge_controls_iter2_PREREG.md" in sealed
    assert "benchmarks/results/20260715_carve_merge_controls_SEAL.json" in sealed
    assert "benchmarks/results/20260715_carve_merge_controls_PHASE_A_ATTEMPT.json" in sealed
    assert "benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_FAILURE_AUDIT.md" in sealed
    assert "benchmarks/results/20260716_carve_merge_controls_iter2_repair.patch" in sealed
    assert "src/rtgs/optim/trainer.py" in sealed
    assert "tests/test_carve_merge_controls_ablation.py" in sealed
    assert str(bench.PHASE_A_ATTEMPT.relative_to(bench.ROOT)) not in sealed
    assert str(bench.PHASE_B_ATTEMPT.relative_to(bench.ROOT)) not in sealed
    assert all(path.is_file() for path in (bench.ROOT / item for item in bench.SEALED_PATHS))


def test_retry_provenance_is_hash_bound_and_failed_targets_remain_absent():
    provenance = bench.verify_retry_provenance()
    assert provenance["fixed_hashes"][str(bench.PREREGISTRATION)] == (bench.PREREGISTRATION_SHA256)
    assert provenance["fixed_hashes"][str(bench.FAILURE_AUDIT)] == bench.FAILURE_AUDIT_SHA256
    assert provenance["failed_targets_absent"] == [
        str(bench.FAILED_PHASE_A_OUTPUT),
        str(bench.FAILED_PHASE_A_NOTE),
    ]
    assert not (bench.ROOT / bench.FAILED_PHASE_A_OUTPUT).exists()
    assert not (bench.ROOT / bench.FAILED_PHASE_A_NOTE).exists()
    assert bench.DEFAULT_SEAL.name == "20260716_carve_merge_controls_iter2_SEAL.json"


def test_training_orders_are_exactly_counterbalanced():
    flat_positions = {
        arm: [bench.TRAINING_ORDER[seed].index(arm) for seed in bench.SEEDS] for arm in bench.ARMS
    }
    assert all(sorted(positions) == [0, 1, 2] for positions in flat_positions.values())
    assert math.prod(len(order) for order in bench.TRAINING_ORDER.values()) == 27
