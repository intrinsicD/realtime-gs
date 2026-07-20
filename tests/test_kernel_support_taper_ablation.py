"""Focused fail-closed checks for the kernel-support experiment harness."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rtgs.optim.trainer import _empty_kernel_support_summary

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "kernel_support_taper_ablation.py"
_SPEC = importlib.util.spec_from_file_location("kernel_support_taper_ablation", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def _record(view: int) -> dict:
    record = _empty_kernel_support_summary(iteration=1, view=view, active_sh_degree=0)
    record.update(
        {
            "visible_gaussian_count": 20,
            "chunk_count": 1,
            "observation_count": 25_000,
            "eligible_count": 20_000,
            "interior_count": 15_000,
            "boundary_count": 3_000,
            "annulus_count": 5_000,
            "local_upstream_l1": 100.0,
            "annulus_upstream_l1": 5.0,
            "recoverable_annulus_upstream_l1": 2.0,
            "active_hard_qgrad_l1": 100.0,
            "boundary_hard_qgrad_l1": 10.0,
            "annulus_candidate_qgrad_l1": 1.0,
            "recoverable_annulus_candidate_qgrad_l1": 0.6,
            "annulus_incidence": 0.25,
            "annulus_upstream_fraction": 0.05,
            "recoverable_annulus_fraction": 0.4,
            "recovered_total_ratio": 0.006,
            "recovered_boundary_ratio": 0.06,
            "ratio_denominators_valid": True,
            "q_min": 0.0,
            "q_max": 50.0,
        }
    )
    return record


def test_audit_aggregation_pools_raw_masses_and_applies_diffuse_gate():
    records = [_record(view) for view in bench.TRAIN_INDICES]
    summary = bench.aggregate_diagnostics(records)
    assert summary["eligible_count"] == 180_000
    assert summary["annulus_upstream_fraction"] == pytest.approx(0.05)
    assert summary["recoverable_annulus_fraction"] == pytest.approx(0.4)
    assert summary["recovered_total_ratio"] == pytest.approx(0.006)
    assert summary["recovered_boundary_ratio"] == pytest.approx(0.06)
    assert summary["all_training_views_sampled"]
    assert bench.audit_gate(summary)
    pooled = bench.pool_audit_summaries([summary, summary, summary])
    assert pooled["recovered_total_ratio"] == pytest.approx(summary["recovered_total_ratio"])


def test_audit_aggregation_rejects_hard_gradient_invariant_violation():
    record = _record(0)
    record["hard_outside_qgrad_nonzero_count"] = 1
    with pytest.raises(AssertionError, match="hard_outside_qgrad_nonzero_count"):
        bench.aggregate_diagnostics([record])


def _passing_ablation_summary() -> dict:
    metrics = (
        "psnr_fg",
        "psnr_full",
        "psnr_crop",
        "ssim_crop",
        "depth_rmse_over_extent",
        "alpha_iou",
        "foreground_coverage",
    )
    result = {}
    for condition in bench.CONDITIONS:
        result[condition] = {}
        for arm in ("hard", *bench.CANDIDATE_ARMS):
            result[condition][arm] = {}
            for metric in metrics:
                base = 20.0 if metric.startswith("psnr") else 0.8
                if metric == "depth_rmse_over_extent":
                    base = 0.1
                gain = 0.0
                if condition == "diffuse" and arm == "c1_taper" and metric.startswith("psnr"):
                    gain = 0.2
                if (
                    condition == "diffuse"
                    and arm == "hard_forward_c1_taper_gradient"
                    and metric.startswith("psnr")
                ):
                    gain = 0.12
                samples = [base + gain] * 3
                result[condition][arm][metric] = {
                    "samples": samples,
                    "mean": sum(samples) / 3,
                    "stdev": 0.0,
                }
    return result


def test_ablation_decision_requires_utility_guardrails_and_attribution():
    decision = bench.ablation_decision(_passing_ablation_summary())
    assert decision["primary_hypothesis_pass"]
    assert decision["taper_mean_psnr_gain_db"] == pytest.approx(0.2)
    failed = _passing_ablation_summary()
    failed["diffuse"]["hard_forward_c1_taper_gradient"]["psnr_fg"] = {
        "samples": [19.9] * 3,
        "mean": 19.9,
        "stdev": 0.0,
    }
    assert not bench.ablation_decision(failed)["primary_hypothesis_pass"]


def test_phase_a_review_is_strictly_bound(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    seal = {"sha256": "a" * 64, "source_aggregate": "b" * 64}
    review = tmp_path / "review.json"
    payload = {
        "artifact_type": "kernel_support_taper_phase_a_scientist_review",
        "verdict": "pass",
        "phase_b_execution_clearance": True,
        "audit_sha256": bench.sha256_file(audit),
        "seal_sha256": seal["sha256"],
        "source_aggregate": seal["source_aggregate"],
    }
    review.write_text(json.dumps(payload), encoding="utf-8")
    assert bench.verify_phase_a_review(review, audit_path=audit, seal=seal)["sha256"]
    payload["audit_sha256"] = "0" * 64
    review.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid or unbound"):
        bench.verify_phase_a_review(review, audit_path=audit, seal=seal)


def test_phase_b_recomputes_phase_a_decision_and_rejects_tampering(tmp_path):
    records = [_record(view) for view in bench.TRAIN_INDICES]
    schedule = [bench.TRAIN_INDICES[index % len(bench.TRAIN_INDICES)] for index in range(120)]
    runs = []
    for condition in bench.CONDITIONS:
        for seed in bench.SEEDS:
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "train_config": asdict(bench.train_config(seed, "hard", diagnostics=True)),
                    "schedule_hash": bench.canonical_json_hash(schedule),
                    "diagnostic_summary": bench.aggregate_diagnostics(records),
                    "history": {
                        "sampled_train_views": schedule,
                        "kernel_support_diagnostics": records,
                    },
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
        "artifact_type": "kernel_support_taper_phase_a_audit",
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
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    bench.validate_phase_a_audit(audit, path, seal)
    audit["decision"]["phase_b_authorized"] = False
    with pytest.raises(RuntimeError, match="decision does not match"):
        bench.validate_phase_a_audit(audit, path, seal)


def test_seal_covers_outcome_dependency_tree():
    sealed = {str(path) for path in bench.SEALED_PATHS}
    assert "benchmarks/kernel_support_taper_ablation.py" in sealed
    assert "benchmarks/results/20260715_kernel_support_taper_PREREG.md" in sealed
    assert "benchmarks/results/20260715_kernel_support_taper_iter2_PREREG.md" in sealed
    assert "tests/test_kernel_support_taper.py" in sealed
    assert "tests/test_kernel_support_taper_ablation.py" in sealed
    assert all(path.is_file() for path in (bench.ROOT / item for item in bench.SEALED_PATHS))


def test_retry_history_and_json_schedule_container_equivalence():
    verified = bench.verify_retry_history()
    assert len(verified) == len(bench.HISTORICAL_ARTIFACT_HASHES) + 1
    tuple_schedule = [(30, 0), (60, 1), (90, 2), (120, 3)]
    list_schedule = [[30, 0], [60, 1], [90, 2], [120, 3]]
    assert bench.canonical_json_hash(tuple_schedule) == bench.canonical_json_hash(list_schedule)
