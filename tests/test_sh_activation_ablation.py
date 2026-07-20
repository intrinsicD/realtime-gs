"""Focused tests for the preregistered SH activation experiment harness."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "sh_activation_ablation.py"
_SPEC = importlib.util.spec_from_file_location("sh_activation_ablation", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
sh_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sh_bench)


def test_view_dependent_condition_changes_only_target_sh_and_rendered_data():
    diffuse, _ = sh_bench.make_condition_scene(0, "diffuse")
    view_dependent, target_range = sh_bench.make_condition_scene(0, "view_dependent")
    assert torch.equal(diffuse.gt_gaussians.means, view_dependent.gt_gaussians.means)
    assert torch.equal(diffuse.gt_gaussians.quats, view_dependent.gt_gaussians.quats)
    assert torch.equal(diffuse.gt_gaussians.log_scales, view_dependent.gt_gaussians.log_scales)
    assert torch.equal(diffuse.gt_gaussians.opacity, view_dependent.gt_gaussians.opacity)
    assert torch.equal(diffuse.gt_gaussians.sh[:, 0], view_dependent.gt_gaussians.sh[:, 0])
    assert float(view_dependent.gt_gaussians.sh[:, 1:].abs().max()) > 0.0
    assert target_range["minimum"] >= 0.03
    assert target_range["maximum"] <= 0.97
    assert diffuse.training_views == sh_bench.TRAIN_INDICES
    assert diffuse.testing_views == sh_bench.TEST_INDICES


def _diagnostic_record(
    *,
    view: int,
    observations: int = 4_000,
    negative_fraction: float = 0.02,
    upstream: float = 10.0,
    recoverable: float = 0.6,
    recovered: float = 0.1,
) -> dict:
    negative_count = round(observations * negative_fraction)
    negative_bins = [
        {
            "low": low,
            "high": high,
            "count": negative_count if index == 4 else 0,
            "upstream_l1": 0.0,
            "recoverable_l1": recoverable if index == 4 else 0.0,
            "smu1_retained_l1": recovered if index == 4 else 0.0,
        }
        for index, (low, high) in enumerate(
            ((None, -0.1), (-0.1, -0.05), (-0.05, -0.02), (-0.02, -0.01), (-0.01, 0.0))
        )
    ]
    positive_bins = [
        {
            "low": low,
            "high": high,
            "count": observations - negative_count if index == 4 else 0,
            "upstream_l1": upstream if index == 4 else 0.0,
            "recoverable_l1": 0.0,
            "smu1_retained_l1": 0.95 * upstream if index == 4 else 0.0,
        }
        for index, (low, high) in enumerate(
            ((0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, None))
        )
    ]
    channels = [
        {
            "channel": channel,
            "observation_count": observations // 3,
            "negative_count": negative_count // 3,
            "upstream_l1": upstream / 3,
            "blocked_l1": 0.3,
            "recoverable_l1": recoverable / 3,
            "smu1_recovered_l1": recovered / 3,
        }
        for channel in range(3)
    ]
    return {
        "iteration": 31,
        "view": view,
        "active_sh_degree": 1,
        "observation_count": observations,
        "negative_count": negative_count,
        "upstream_l1": upstream,
        "blocked_l1": 0.9,
        "recoverable_l1": recoverable,
        "smu1_recovered_l1": recovered,
        "positive_upstream_l1": upstream,
        "positive_smu1_retained_l1": 0.95 * upstream,
        "negative_raw_gradient_nonzero_count": 0,
        "negative_raw_gradient_max_abs": 0.0,
        "positive_raw_upstream_max_abs_error": 0.0,
        "negative_margin_bins": negative_bins,
        "positive_margin_bins": positive_bins,
        "channels": channels,
    }


def test_audit_aggregation_uses_raw_sums_and_frozen_gate():
    records = [_diagnostic_record(view=view) for view in sh_bench.TRAIN_INDICES]
    summary = sh_bench.aggregate_diagnostics(records)
    assert summary["observation_count"] == 36_000
    assert summary["negative_fraction"] == pytest.approx(0.02)
    assert summary["recoverable_fraction"] == pytest.approx(0.06)
    assert summary["smu1_recovered_fraction"] == pytest.approx(0.01)
    assert summary["all_training_views_sampled"]
    assert sh_bench.audit_gate(summary)
    pooled = sh_bench.pool_audit_summaries([summary, summary, summary])
    assert pooled["recoverable_fraction"] == pytest.approx(summary["recoverable_fraction"])


def test_audit_aggregation_rejects_hard_gradient_invariant_violation():
    record = _diagnostic_record(view=0)
    record["negative_raw_gradient_nonzero_count"] = 1
    with pytest.raises(AssertionError, match="nonzero negative"):
        sh_bench.aggregate_diagnostics([record])


def _passing_summary() -> dict:
    result = {}
    metrics = (
        "psnr_fg",
        "psnr_full",
        "psnr_crop",
        "ssim_crop",
        "depth_rmse_over_extent",
        "alpha_iou",
        "foreground_coverage",
    )
    for condition in sh_bench.CONDITIONS:
        result[condition] = {}
        for arm in ("hard", *sh_bench.CANDIDATE_ARMS):
            result[condition][arm] = {}
            for metric in metrics:
                if metric.startswith("psnr"):
                    base = 20.0
                elif metric == "ssim_crop":
                    base = 0.8
                elif metric == "depth_rmse_over_extent":
                    base = 0.1
                else:
                    base = 0.75
                gain = 0.0
                if condition == "view_dependent" and arm == "smu1" and metric.startswith("psnr"):
                    gain = 0.3
                if (
                    condition == "view_dependent"
                    and arm == "hard_forward_smu1_negative_gradient"
                    and metric.startswith("psnr")
                ):
                    gain = 0.2
                samples = [base + gain] * 3
                result[condition][arm][metric] = {
                    "samples": samples,
                    "mean": sum(samples) / 3,
                    "stdev": 0.0,
                }
    return result


def test_ablation_decision_requires_utility_guardrails_and_attribution():
    decision = sh_bench.ablation_decision(_passing_summary())
    assert decision["primary_hypothesis_pass"]
    assert decision["smu1_mean_psnr_gain_db"] == pytest.approx(0.3)
    failed = _passing_summary()
    failed["view_dependent"]["hard_forward_smu1_negative_gradient"]["psnr_fg"] = {
        "samples": [19.9, 19.9, 19.9],
        "mean": 19.9,
        "stdev": 0.0,
    }
    assert not sh_bench.ablation_decision(failed)["primary_hypothesis_pass"]


def test_phase_a_review_requires_structured_audit_and_seal_binding(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text('{"decision": false}\n', encoding="utf-8")
    seal = {"sha256": "a" * 64, "source_aggregate": "b" * 64}
    review = tmp_path / "review.json"
    payload = {
        "artifact_type": "sh_activation_phase_a_scientist_review",
        "verdict": "pass",
        "phase_b_execution_clearance": True,
        "audit_sha256": sh_bench.sha256_file(audit),
        "seal_sha256": seal["sha256"],
        "source_aggregate": seal["source_aggregate"],
    }
    review.write_text(json.dumps(payload), encoding="utf-8")
    assert sh_bench.verify_phase_a_review(review, audit_path=audit, seal=seal)["sha256"]
    payload["audit_sha256"] = "0" * 64
    review.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid or unbound"):
        sh_bench.verify_phase_a_review(review, audit_path=audit, seal=seal)


def test_strict_json_and_output_preflight_fail_closed(tmp_path):
    malformed = tmp_path / "nan.json"
    malformed.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON"):
        sh_bench.strict_json_load(malformed)
    output = tmp_path / "result.json"
    sh_bench.preflight_output(output)
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to start"):
        sh_bench.preflight_output(output)


def test_once_only_attempt_marker_is_atomic(tmp_path):
    assert sh_bench.PHASE_A_ATTEMPT.is_absolute()
    assert sh_bench.PHASE_B_ATTEMPT.is_absolute()
    marker = tmp_path / "attempt.json"
    sh_bench.claim_attempt(marker, phase="phase_a", output=tmp_path / "out.json", inputs={})
    with pytest.raises(RuntimeError, match="already been claimed"):
        sh_bench.claim_attempt(marker, phase="phase_a", output=tmp_path / "out2.json", inputs={})


def test_loaded_source_verifier_uses_runtime_hashes_and_fails_on_mismatch(tmp_path, monkeypatch):
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"source_hashes": {"src/rtgs/a.py": "good"}}), encoding="utf-8")
    monkeypatch.setattr(
        sh_bench,
        "loaded_source_hashes",
        lambda: ({"src/rtgs/a.py": "good"}, "loaded-aggregate"),
    )
    assert sh_bench.verify_loaded_sources_against_seal(seal)[1] == "loaded-aggregate"
    monkeypatch.setattr(
        sh_bench,
        "loaded_source_hashes",
        lambda: ({"src/rtgs/a.py": "bad"}, "loaded-aggregate"),
    )
    with pytest.raises(RuntimeError, match="outside/different"):
        sh_bench.verify_loaded_sources_against_seal(seal)


def test_seal_covers_outcome_dependency_tree():
    sealed = {str(path) for path in sh_bench.SEALED_PATHS}
    required = {
        "src/rtgs/core/metrics.py",
        "src/rtgs/data/synthetic.py",
        "src/rtgs/image2gs/fit.py",
        "src/rtgs/image2gs/renderer2d.py",
        "src/rtgs/lift/depth.py",
        "src/rtgs/lift/merge.py",
        "src/rtgs/optim/trainer.py",
        "src/rtgs/render/torch_ref.py",
        "benchmarks/sh_activation_ablation.py",
        "tests/test_sh_activation_ablation.py",
    }
    assert required <= sealed
    assert all(path.is_file() for path in (sh_bench.ROOT / item for item in sh_bench.SEALED_PATHS))


def test_phase_b_recomputes_phase_a_decision_and_rejects_tampering(tmp_path):
    runs = []
    for condition in sh_bench.CONDITIONS:
        for seed in sh_bench.SEEDS:
            records = [_diagnostic_record(view=view) for view in sh_bench.TRAIN_INDICES]
            schedule = [sh_bench.TRAIN_INDICES[index % 9] for index in range(120)]
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "train_config": asdict(sh_bench.train_config(seed, "hard", diagnostics=True)),
                    "schedule_hash": sh_bench.canonical_json_hash(schedule),
                    "diagnostic_summary": sh_bench.aggregate_diagnostics(records),
                    "history": {
                        "sampled_train_views": schedule,
                        "sh_color_diagnostics": records,
                    },
                }
            )
    environment = sh_bench.environment_metadata()
    seal = {
        "path": "seal.json",
        "sha256": "a" * 64,
        "source_aggregate": "b" * 64,
        "verification_sha256": "c" * 64,
        "environment_fingerprint": sh_bench.environment_fingerprint(environment),
    }
    audit = {
        "artifact_type": "sh_activation_phase_a_audit",
        "split": {"train": sh_bench.TRAIN_INDICES, "held_out": sh_bench.TEST_INDICES},
        "seal": seal,
        "environment": environment,
        "runs": runs,
        "decision": sh_bench.audit_decision(runs),
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    sh_bench.validate_phase_a_audit(audit, audit_path, seal)
    audit["decision"]["phase_b_authorized"] = not audit["decision"]["phase_b_authorized"]
    with pytest.raises(RuntimeError, match="decision does not match"):
        sh_bench.validate_phase_a_audit(audit, audit_path, seal)
