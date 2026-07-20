"""Fail-closed tests for the preregistered visibility-margin harness."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from rtgs.data.synthetic import make_synthetic_scene

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "visibility_margin_ablation.py"
_SPEC = importlib.util.spec_from_file_location("visibility_margin_ablation", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


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


def _view_record(view: int, *, missed: int = 20, support: int = 20_000) -> dict:
    included = support - missed
    missed_mass = float(missed)
    included_mass = float(included)
    return {
        "view": view,
        "missed_gaussian_indices": [view],
        "included_pair_count": included,
        "missed_pair_count": missed,
        "support_pair_count": support,
        "included_effective_mass": included_mass,
        "missed_effective_mass": missed_mass,
        "total_effective_mass": included_mass + missed_mass,
        "residual_l1": 100.0,
        "render_delta_l1": 0.2,
        "current_objective": 1.0,
        "support_safe_objective": 0.999,
        "objective_delta": -0.001,
        "q_bins": [
            {"low": 9.0, "high": 10.0, "count": missed, "effective_mass": missed_mass},
            {"low": 10.0, "high": 11.0, "count": 0, "effective_mass": 0.0},
            {"low": 11.0, "high": 12.0, "count": 0, "effective_mass": 0.0},
        ],
    }


def _checkpoint(*, support: int = 20_000) -> dict:
    views = [_view_record(view, support=support) for view in bench.TRAIN_INDICES]
    return {"views": views, "summary": bench.aggregate_checkpoint_views(views)}


def test_checkpoint_aggregation_pools_raw_counts_mass_and_render_delta():
    checkpoint = _checkpoint()
    summary = checkpoint["summary"]
    assert summary["support_pair_count"] == 180_000
    assert summary["missed_pair_fraction"] == pytest.approx(0.001)
    assert summary["missed_effective_mass_fraction"] == pytest.approx(0.001)
    assert summary["render_delta_over_residual"] == pytest.approx(0.002)
    assert summary["distinct_missed_gaussian_view_exposures"] == 9
    assert summary["all_training_views_audited"]
    assert bench.audit_gate(summary)


def test_checkpoint_aggregation_rejects_invalid_partition_and_exposure_ids():
    views = [_view_record(view) for view in bench.TRAIN_INDICES]
    views[0]["q_bins"][0]["count"] -= 1
    with pytest.raises(AssertionError, match="partitioned"):
        bench.aggregate_checkpoint_views(views)
    views = [_view_record(view) for view in bench.TRAIN_INDICES]
    views[0]["missed_gaussian_indices"] = [2, 2]
    with pytest.raises(AssertionError, match="identities"):
        bench.aggregate_checkpoint_views(views)


def test_decision_uses_only_final_diffuse_and_requires_two_seeds_plus_pool():
    runs = []
    for condition in bench.CONDITIONS:
        for seed in bench.SEEDS:
            checkpoint = _checkpoint()
            if condition == "view_dependent":
                checkpoint["summary"]["missed_pair_fraction"] = 0.0
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "initial_incidence": _checkpoint(),
                    "final_incidence": checkpoint,
                }
            )
    decision = bench.audit_decision(runs)
    assert decision["seed_passes"] == [True, True, True]
    assert decision["pooled_pass"]
    assert decision["phase_b_authorized"]


def test_any_seed_validity_failure_blocks_phase_b_even_when_material_gates_pass():
    runs = []
    for condition in bench.CONDITIONS:
        for seed in bench.SEEDS:
            checkpoint = _checkpoint(
                support=10_000 if condition == "diffuse" and seed == 2 else 20_000
            )
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "initial_incidence": _checkpoint(),
                    "final_incidence": checkpoint,
                }
            )
    decision = bench.audit_decision(runs)
    assert decision["seed_material_passes"] == [True, True, True]
    assert decision["seed_validity_passes"] == [True, True, False]
    assert not decision["phase_b_authorized"]


def _passing_ablation_summary() -> dict:
    result = {}
    for condition in bench.CONDITIONS:
        result[condition] = {}
        for arm in (
            "current",
            "current_forward_safe",
            bench.CANDIDATE_ARM,
            "support_safe_common_current",
        ):
            result[condition][arm] = {}
            for metric in bench.METRICS:
                base = 20.0 if metric.startswith("psnr") else 0.8
                if metric == "depth_rmse_over_extent":
                    base = 0.1
                gain = (
                    0.2
                    if condition == "diffuse"
                    and arm == bench.CANDIDATE_ARM
                    and metric.startswith("psnr")
                    else 0.0
                )
                samples = [base + gain] * 3
                result[condition][arm][metric] = {
                    "samples": samples,
                    "mean": sum(samples) / 3,
                    "stdev": 0.0,
                }
    return result


def test_ablation_decision_requires_primary_utility_and_all_guardrails():
    decision = bench.ablation_decision(_passing_ablation_summary())
    assert decision["primary_hypothesis_pass"]
    failed = _passing_ablation_summary()
    failed["view_dependent"][bench.CANDIDATE_ARM]["psnr_fg"] = {
        "samples": [19.5] * 3,
        "mean": 19.5,
        "stdev": 0.0,
    }
    assert not bench.ablation_decision(failed)["primary_hypothesis_pass"]


def test_phase_a_review_is_strictly_bound(tmp_path):
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")
    seal = {"sha256": "a" * 64, "source_aggregate": "b" * 64}
    review_path = tmp_path / "review.json"
    payload = {
        "artifact_type": "visibility_margin_iter2_phase_a_scientist_review",
        "verdict": "pass",
        "phase_b_execution_clearance": True,
        "audit_sha256": bench.sha256_file(audit_path),
        "seal_sha256": seal["sha256"],
        "source_aggregate": seal["source_aggregate"],
    }
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    assert bench.verify_phase_a_review(review_path, audit_path=audit_path, seal=seal)["sha256"]
    payload["phase_b_execution_clearance"] = False
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid or unbound"):
        bench.verify_phase_a_review(review_path, audit_path=audit_path, seal=seal)


def test_phase_b_recomputes_raw_checkpoint_summaries_and_gate():
    schedule = [bench.TRAIN_INDICES[index % len(bench.TRAIN_INDICES)] for index in range(120)]
    runs = []
    for condition in bench.CONDITIONS:
        for seed in bench.SEEDS:
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "train_config": asdict(bench.train_config(seed, bench.CURRENT_MARGIN_SIGMA)),
                    "schedule_hash": bench.canonical_json_hash(schedule),
                    "history": {"sampled_train_views": schedule},
                    "target_margin_invariance": {
                        "all_twelve_views_bit_exact": True,
                        "views": [
                            {
                                "view": index,
                                "target_sha256": "a" * 64,
                                "target_depth_sha256": "b" * 64,
                                "render_sha256": "c" * 64,
                                "maximum_absolute_errors": {
                                    "current_safe_color": 0.0,
                                    "current_safe_alpha": 0.0,
                                    "current_safe_depth": 0.0,
                                    "stored_color": 0.0,
                                    "stored_depth": 0.0,
                                },
                            }
                            for index in range(12)
                        ],
                    },
                    "initial_incidence": _checkpoint(),
                    "final_incidence": _checkpoint(),
                }
            )
    environment = bench.environment_metadata()
    seal = {
        "path": "seal.json",
        "sha256": "a" * 64,
        "source_aggregate": "b" * 64,
        "verification_sha256": "c" * 64,
        "environment_fingerprint": bench.environment_fingerprint(environment),
    }
    audit = {
        "artifact_type": "visibility_margin_iter2_phase_a_audit",
        "preregistration": {
            "path": str(bench.PREREGISTRATION),
            "sha256": bench.sha256_file(bench.ROOT / bench.PREREGISTRATION),
        },
        "split": {"train": bench.TRAIN_INDICES, "held_out": bench.TEST_INDICES},
        "seal": seal,
        "environment": environment,
        "runs": runs,
        "decision": bench.audit_decision(runs),
    }
    bench.validate_phase_a_audit(audit, seal)
    audit["runs"][0]["final_incidence"]["summary"]["missed_pair_count"] += 1
    with pytest.raises(RuntimeError, match="summary differs"):
        bench.validate_phase_a_audit(audit, seal)


def test_attempt_marker_and_output_namespace_are_once_only(tmp_path):
    marker = tmp_path / "attempt.json"
    output = tmp_path / "result.json"
    bench.claim_attempt(marker, phase="phase_a", output=output, inputs={"seal_sha256": "a" * 64})
    with pytest.raises(RuntimeError, match="already been claimed"):
        bench.claim_attempt(
            marker, phase="phase_a", output=output, inputs={"seal_sha256": "a" * 64}
        )
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to start"):
        bench.preflight_output(output)


def test_frozen_synthetic_targets_are_margin_invariant():
    scene = make_synthetic_scene(n_gaussians=5, n_cameras=12, image_size=10, seed=47)
    result = bench.verify_synthetic_target_margin_invariance(scene)
    assert result["all_twelve_views_bit_exact"]
    assert len(result["views"]) == 12


def test_checkpoint_audit_fails_closed_on_nonfinite_projection(monkeypatch):
    scene = make_synthetic_scene(n_gaussians=5, n_cameras=12, image_size=10, seed=48)
    original = bench._projection

    def nonfinite_projection(gaussians, camera):
        z, in_front, uv, cov2d, sigma = original(gaussians, camera)
        z = z.clone()
        z[0] = torch.nan
        return z, in_front, uv, cov2d, sigma

    monkeypatch.setattr(bench, "_projection", nonfinite_projection)
    with pytest.raises(AssertionError, match="projected depth is non-finite"):
        bench.audit_checkpoint(scene, scene.gt_gaussians)


def test_seal_covers_protocol_harness_all_source_and_tests():
    sealed = {str(path) for path in bench.SEALED_PATHS}
    assert "benchmarks/visibility_margin_ablation.py" in sealed
    assert "benchmarks/kernel_support_taper_ablation.py" in sealed
    assert "benchmarks/results/20260715_visibility_margin_PREREG.md" in sealed
    assert "benchmarks/results/20260715_visibility_margin_iter2_PREREG.md" in sealed
    assert "benchmarks/results/20260715_visibility_margin_SEAL.json" in sealed
    assert "benchmarks/results/20260715_visibility_margin_PHASE_A_ATTEMPT.json" in sealed
    assert "tests/test_visibility_margin.py" in sealed
    assert "tests/test_visibility_margin_ablation.py" in sealed
    assert all(path.is_file() for path in (bench.ROOT / item for item in bench.SEALED_PATHS))


def test_retry_provenance_binds_consumed_attempt_and_absent_output():
    provenance = bench.verify_retry_provenance()
    assert provenance["failed_outputs_absent"]
    assert provenance["incorporated_hashes"][str(bench.PREVIOUS_SEAL)] == (
        bench.PREVIOUS_SEAL_SHA256
    )
    assert "iter2" in bench.DEFAULT_SEAL.name
    assert "iter2" in bench.PHASE_A_ATTEMPT.name
    assert "iter2" in bench.PHASE_B_ATTEMPT.name
