"""Toy-only tests for quaternion-gauge protocol math, reductions, and artifact guards."""

from __future__ import annotations

import copy
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from benchmarks import quaternion_gauge_ablation as gauge

from rtgs.core.gaussians3d import Gaussians3D


def _toy_gaussians(rows: int = 130) -> Gaussians3D:
    return Gaussians3D(
        means=torch.zeros(rows, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(rows, 1),
        log_scales=torch.tensor([[0.0, math.log(2.0), math.log(3.0)]]).repeat(rows, 1),
        opacity=torch.full((rows,), 0.2),
        sh=torch.zeros(rows, 1, 3),
    )


def test_frozen_protocol_hash_paths_and_cli_contract():
    assert str(gauge.IMPLEMENTATION_REVIEW_ADDENDUM) == (
        "benchmarks/results/20260716_quaternion_gauge_iter2_IMPLEMENTATION_REVIEW_ADDENDUM_1.md"
    )
    assert gauge.sha256_file(gauge.ROOT / gauge.PREREGISTRATION) == (gauge.PREREGISTRATION_SHA256)
    assert gauge.sha256_file(gauge.ROOT / gauge.ORIGINAL_PREREGISTRATION) == (
        gauge.ORIGINAL_PREREGISTRATION_SHA256
    )
    sealed = gauge._sealed_paths()
    assert gauge.ORIGINAL_PREREGISTRATION in sealed
    assert gauge.PREREGISTRATION in sealed
    assert gauge.IMPLEMENTATION_REVIEW in sealed
    assert gauge.IMPLEMENTATION_REVIEW_ADDENDUM in sealed
    assert gauge.HARNESS_PATH in sealed
    assert set(gauge.FOCUSED_TEST_PATHS).issubset(sealed)
    assert Path("src/rtgs/optim/trainer.py") in sealed
    assert {
        "artifact_type",
        "verdict",
        "phase_b_execution_clearance",
        "phase_a_sha256",
        "human_audit_sha256",
        "seal_sha256",
        "phase_a_attempt_sha256",
        "source_aggregate",
    } == gauge.SCIENTIST_REVIEW_KEYS
    assert gauge.parse_args(["seal", "--output", "seal.json"]).action == "seal"
    assert (
        gauge.parse_args(["audit", "--seal", "seal.json", "--output-prefix", "prefix"]).action
        == "audit"
    )
    assert (
        gauge.parse_args(
            [
                "run",
                "--seal",
                "seal.json",
                "--phase-a",
                "phase-a.json",
                "--review",
                "review.json",
                "--output",
                "out.json",
            ]
        ).action
        == "run"
    )


def test_hamilton_product_radial_and_antipodal_covariance_gauge():
    base = _toy_gaussians(2)
    axis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    angle = math.radians(20.0) / 2.0
    delta = torch.cat([torch.full((2, 1), math.cos(angle)), axis * math.sin(angle)], dim=-1)
    perturbed = torch.nn.functional.normalize(gauge.hamilton_product(delta, base.quats), dim=-1)
    reference = gauge.covariance_float64(gauge.gaussian_with_quaternions(base, perturbed))
    for quaternions in (0.25 * perturbed, 4.0 * perturbed, -perturbed):
        candidate = gauge.covariance_float64(gauge.gaussian_with_quaternions(base, quaternions))
        assert torch.max((candidate - reference).abs()) <= 2e-12


def test_diagnostic_selection_breaks_ties_by_low_index_then_emits_sorted():
    selection = gauge.diagnostic_selection(_toy_gaussians())
    assert selection["eligible_indices"] == list(range(130))
    assert selection["selected_indices"] == list(range(128))


def test_linear_quantile_auc_and_psnr_raw_reductions():
    assert gauge.linear_quantile([0.0, 10.0], 0.9) == pytest.approx(9.0)
    assert gauge.normalized_auc([0, 10, 20], [0.0, 10.0, 0.0]) == pytest.approx(5.0)
    assert gauge.psnr_from_sse(1.0, 100) == pytest.approx(20.0)


def test_canonical_seal_digest_omits_only_its_own_field():
    payload = {"artifact_type": "seal", "value": [1, 2, 3]}
    digest = gauge.seal_payload_digest(payload)
    signed = {**payload, "sha256": digest}
    assert gauge.seal_payload_digest(signed) == digest
    signed["value"].append(4)
    assert gauge.seal_payload_digest(signed) != digest


def test_seal_refuses_source_drift_across_verification(monkeypatch):
    monkeypatch.setattr(gauge, "verify_preregistration", lambda: None)
    monkeypatch.setattr(
        gauge,
        "verify_implementation_review",
        lambda: {"path": "review", "sha256": "review-hash"},
    )
    monkeypatch.setattr(
        gauge,
        "verify_implementation_review_addendum",
        lambda: {"path": "addendum", "sha256": "addendum-hash"},
    )
    monkeypatch.setattr(gauge, "environment_metadata", lambda: {"environment": "toy"})
    monkeypatch.setattr(gauge, "assert_official_environment", lambda _metadata: None)
    monkeypatch.setattr(gauge, "run_verification", lambda: {"passed": True})
    snapshots = iter(
        [
            ({"source.py": "before"}, "before-aggregate"),
            ({"source.py": "after"}, "after-aggregate"),
        ]
    )
    monkeypatch.setattr(gauge, "source_hashes", lambda: next(snapshots))
    with pytest.raises(RuntimeError, match="changed during full verification"):
        gauge.create_seal()


def test_consumed_attempt_provenance_binds_hashes_and_absence():
    provenance = gauge.verify_consumed_attempt_provenance()
    assert provenance["original_preregistration"] == {
        "path": str(gauge.ORIGINAL_PREREGISTRATION),
        "sha256": gauge.ORIGINAL_PREREGISTRATION_SHA256,
    }
    assert provenance["consumed_artifacts"] == {
        str(path): digest for path, digest in gauge.CONSUMED_ATTEMPT_BINDINGS.items()
    }
    assert provenance["verified_absent"] is True


def test_consumed_attempt_provenance_rejects_hash_drift(monkeypatch):
    first_path = next(iter(gauge.CONSUMED_ATTEMPT_BINDINGS))
    real_sha256_file = gauge.sha256_file

    def drift_one(path):
        if Path(path).resolve() == (gauge.ROOT / first_path).resolve():
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(gauge, "sha256_file", drift_one)
    with pytest.raises(RuntimeError, match="consumed-attempt artifact hash differs"):
        gauge.verify_consumed_attempt_provenance()


def test_retry_seal_payload_binds_original_protocol_and_consumed_attempt(monkeypatch):
    provenance = gauge.verify_consumed_attempt_provenance()
    monkeypatch.setattr(gauge, "verify_preregistration", lambda: None)
    monkeypatch.setattr(
        gauge,
        "verify_implementation_review",
        lambda: {"path": str(gauge.IMPLEMENTATION_REVIEW), "sha256": "b" * 64},
    )
    monkeypatch.setattr(
        gauge,
        "verify_implementation_review_addendum",
        lambda: {"path": str(gauge.IMPLEMENTATION_REVIEW_ADDENDUM), "sha256": "e" * 64},
    )
    monkeypatch.setattr(gauge, "environment_metadata", lambda: {"environment": "toy"})
    monkeypatch.setattr(gauge, "assert_official_environment", lambda _metadata: None)
    monkeypatch.setattr(gauge, "run_verification", lambda: {"passed": True})
    monkeypatch.setattr(gauge, "source_hashes", lambda: ({"source.py": "c" * 64}, "d" * 64))
    monkeypatch.setattr(gauge, "git_metadata", lambda: {"revision": "toy"})
    payload = gauge.create_seal()
    assert payload["artifact_type"] == "quaternion_gauge_iter2_implementation_seal"
    assert payload["preregistration"] == {
        "path": str(gauge.PREREGISTRATION),
        "sha256": gauge.PREREGISTRATION_SHA256,
    }
    assert payload["original_preregistration"] == provenance["original_preregistration"]
    assert payload["retry_provenance"] == provenance
    assert payload["implementation_review_addendum"] == {
        "path": str(gauge.IMPLEMENTATION_REVIEW_ADDENDUM),
        "sha256": "e" * 64,
    }
    assert payload["sha256"] == gauge.seal_payload_digest(payload)


def test_implementation_review_addendum_is_mandatory_and_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(gauge, "ROOT", tmp_path)
    path = tmp_path / gauge.IMPLEMENTATION_REVIEW_ADDENDUM
    with pytest.raises(FileNotFoundError, match="addendum is missing"):
        gauge.verify_implementation_review_addendum()

    path.parent.mkdir(parents=True)
    path.write_text("Verdict: FAIL\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lacks exact 'Verdict: PASS'"):
        gauge.verify_implementation_review_addendum()

    path.write_text("# Independent addendum\n\nVerdict: PASS\n", encoding="utf-8")
    record = gauge.verify_implementation_review_addendum()
    assert record == {
        "path": str(gauge.IMPLEMENTATION_REVIEW_ADDENDUM),
        "sha256": gauge.sha256_file(path),
    }
    path.write_text("# Independent addendum\n\nVerdict: PASS\n\nTampered.\n", encoding="utf-8")
    assert gauge.verify_implementation_review_addendum()["sha256"] != record["sha256"]


def test_seal_review_addendum_binding_rejects_missing_path_hash_and_tamper():
    review = {"path": str(gauge.IMPLEMENTATION_REVIEW), "sha256": "1" * 64}
    addendum = {
        "path": str(gauge.IMPLEMENTATION_REVIEW_ADDENDUM),
        "sha256": "2" * 64,
    }
    payload = {
        "implementation_review": copy.deepcopy(review),
        "implementation_review_addendum": copy.deepcopy(addendum),
    }
    gauge.validate_implementation_review_bindings(payload, review, addendum)

    missing = copy.deepcopy(payload)
    del missing["implementation_review_addendum"]
    with pytest.raises(RuntimeError, match="addendum binding differs"):
        gauge.validate_implementation_review_bindings(missing, review, addendum)

    wrong_path = copy.deepcopy(payload)
    wrong_path["implementation_review_addendum"]["path"] = "wrong.md"
    with pytest.raises(RuntimeError, match="addendum binding differs"):
        gauge.validate_implementation_review_bindings(wrong_path, review, addendum)

    wrong_hash = copy.deepcopy(payload)
    wrong_hash["implementation_review_addendum"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="addendum binding differs"):
        gauge.validate_implementation_review_bindings(wrong_hash, review, addendum)

    tampered_current = copy.deepcopy(addendum)
    tampered_current["sha256"] = "3" * 64
    with pytest.raises(RuntimeError, match="addendum binding differs"):
        gauge.validate_implementation_review_bindings(payload, review, tampered_current)


def test_phase_a_artifact_binding_schema_rejects_seal_file_tamper_and_key_drift():
    seal = {
        "path": str(gauge.DEFAULT_SEAL),
        "sha256": "1" * 64,
        "file_sha256": "2" * 64,
        "source_aggregate": "3" * 64,
        "implementation_review": {
            "path": str(gauge.IMPLEMENTATION_REVIEW),
            "sha256": "4" * 64,
        },
        "retry_provenance": {"verified": True},
    }
    marker = {"path": str(gauge.PHASE_A_ATTEMPT), "sha256": "5" * 64}
    bindings = gauge.artifact_bindings(seal, marker)
    gauge.validate_artifact_bindings(bindings, seal)

    tampered = copy.deepcopy(bindings)
    tampered["seal_file_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="seal_file_sha256"):
        gauge.validate_artifact_bindings(tampered, seal)

    missing = copy.deepcopy(bindings)
    del missing["seal_file_sha256"]
    with pytest.raises(RuntimeError, match="key set"):
        gauge.validate_artifact_bindings(missing, seal)

    redundant = copy.deepcopy(bindings)
    redundant["unexpected"] = "6" * 64
    with pytest.raises(RuntimeError, match="key set"):
        gauge.validate_artifact_bindings(redundant, seal)


@pytest.mark.parametrize(
    "forbidden",
    [
        *gauge.CONSUMED_ATTEMPT_FORBIDDEN_PATHS,
        Path("benchmarks/results/20260716T050000Z_cpu_quaternion_gauge_ablation.json"),
        Path("benchmarks/results/20260716T050000Z_cpu_quaternion_gauge_ablation_RESULT.md"),
    ],
)
def test_consumed_attempt_provenance_rejects_each_forbidden_sibling(
    tmp_path, monkeypatch, forbidden
):
    for path in gauge.CONSUMED_ATTEMPT_BINDINGS:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bound")
    blocked = tmp_path / forbidden
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_bytes(b"forbidden")
    expected_by_name = {
        str((tmp_path / path).resolve()): digest
        for path, digest in gauge.CONSUMED_ATTEMPT_BINDINGS.items()
    }
    monkeypatch.setattr(gauge, "ROOT", tmp_path)
    monkeypatch.setattr(
        gauge,
        "sha256_file",
        lambda path: expected_by_name[str(Path(path).resolve())],
    )
    with pytest.raises(RuntimeError, match="unexpectedly exposes"):
        gauge.verify_consumed_attempt_provenance()


def test_official_actions_require_the_fixed_seal_path():
    assert (
        gauge.require_default_seal_path(gauge.DEFAULT_SEAL)
        == (gauge.ROOT / gauge.DEFAULT_SEAL).resolve()
    )
    with pytest.raises(ValueError, match="sole implementation seal"):
        gauge.require_default_seal_path(Path("copied-seal.json"))


@pytest.mark.parametrize(
    "content,match",
    [
        ('{"a":1,"a":2}', "duplicate JSON key"),
        ('{"a":NaN}', "non-standard JSON"),
        ('{"a":Infinity}', "non-standard JSON"),
    ],
)
def test_strict_json_rejects_duplicates_and_nonfinite(tmp_path, content, match):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        gauge.strict_json_load(path)


def test_append_only_write_and_output_namespace_guards(tmp_path):
    path = tmp_path / "append-only.txt"
    gauge.exclusive_atomic_write(path, "first")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gauge.exclusive_atomic_write(path, "second")
    assert path.read_text() == "first"
    with pytest.raises(ValueError, match="benchmarks/results"):
        gauge.phase_a_output_paths(tmp_path / "20260716T040000Z_cpu_quaternion_gauge_iter2")
    prefix = gauge.ROOT / "benchmarks/results/20260716T040000Z_cpu_quaternion_gauge_iter2"
    paths = gauge.phase_a_output_paths(prefix)
    assert paths["audit_json"].name.endswith("_audit.json")
    assert paths["invalid_note"].name.endswith("_invalid_RESULT.md")
    with pytest.raises(ValueError, match="output-prefix"):
        gauge.phase_a_output_paths(
            gauge.ROOT / "benchmarks/results/20260716T040000Z_cpu_quaternion_gauge"
        )
    phase_b = gauge.phase_b_output_paths(
        gauge.ROOT / "benchmarks/results/20260716T050000Z_cpu_quaternion_gauge_iter2_ablation.json"
    )
    assert phase_b["note"].name.endswith("_ablation_RESULT.md")
    with pytest.raises(ValueError, match="Phase-B output basename"):
        gauge.phase_b_output_paths(
            gauge.ROOT / "benchmarks/results/20260716T050000Z_cpu_quaternion_gauge_ablation.json"
        )


def test_phase_a_attempt_payload_binds_real_full_and_loaded_hash_domains(tmp_path, monkeypatch):
    prospective_addendum_sha256 = "a" * 64
    if (gauge.ROOT / gauge.IMPLEMENTATION_REVIEW_ADDENDUM).is_file():
        full_hashes, full_aggregate = gauge.source_hashes()
    else:
        existing = tuple(path for path in gauge._sealed_paths() if (gauge.ROOT / path).is_file())
        full_hashes, _ = gauge.source_hashes(existing)
        full_hashes[str(gauge.IMPLEMENTATION_REVIEW_ADDENDUM)] = prospective_addendum_sha256
        full_aggregate = gauge.canonical_json_hash(full_hashes)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json\n"
                "from pathlib import Path\n"
                "from benchmarks import quaternion_gauge_ablation as gauge\n"
                "addendum = (gauge.ROOT / gauge.IMPLEMENTATION_REVIEW_ADDENDUM).resolve()\n"
                "real_sha256_file = gauge.sha256_file\n"
                "if not addendum.is_file():\n"
                f"    prospective = {prospective_addendum_sha256!r}\n"
                "    gauge.sha256_file = lambda path: (\n"
                "        prospective\n"
                "        if Path(path).resolve() == addendum\n"
                "        else real_sha256_file(path)\n"
                "    )\n"
                "hashes, aggregate = gauge.loaded_source_hashes()\n"
                "print(json.dumps({'hashes': hashes, 'aggregate': aggregate}, sort_keys=True))\n"
            ),
        ],
        cwd=gauge.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded_payload = json.loads(probe.stdout)
    loaded_hashes = loaded_payload["hashes"]
    loaded_aggregate = loaded_payload["aggregate"]
    assert loaded_aggregate == gauge.canonical_json_hash(loaded_hashes)
    assert set(loaded_hashes) < set(full_hashes)
    assert loaded_aggregate != full_aggregate

    monkeypatch.setattr(gauge, "ROOT", tmp_path)
    monkeypatch.setattr(
        gauge,
        "verify_loaded_sources_against_seal",
        lambda _path: (loaded_hashes, loaded_aggregate),
    )
    phase_a = tmp_path / "benchmarks/results/20260716T040000Z_cpu_quaternion_gauge_iter2_audit.json"
    prefix = phase_a.with_name(phase_a.name.removesuffix("_audit.json"))
    seal = {
        "path": "unused-seal.json",
        "source_hashes": full_hashes,
        "source_aggregate": full_aggregate,
        "sha256": "seal-hash",
    }
    marker = gauge.attempt_marker(
        artifact_type="quaternion_gauge_iter2_phase_a_attempt",
        marker_path=tmp_path / "phase-a-attempt.json",
        seal=seal,
        outputs=gauge.phase_a_output_paths(prefix),
        inputs={"output_prefix": str(prefix)},
    )["payload"]
    gauge.validate_phase_a_attempt_payload(marker, seal, phase_a)

    old_type = copy.deepcopy(marker)
    old_type["artifact_type"] = "quaternion_gauge_phase_a_attempt"
    with pytest.raises(ValueError, match="artifact type differs"):
        gauge.validate_phase_a_attempt_payload(old_type, seal, phase_a)

    bad_aggregate = copy.deepcopy(marker)
    bad_aggregate["loaded_source_aggregate"] = "0" * 64
    with pytest.raises(RuntimeError, match="loaded-source aggregate"):
        gauge.validate_phase_a_attempt_payload(bad_aggregate, seal, phase_a)

    missing_mandatory = copy.deepcopy(marker)
    missing_mandatory["loaded_source_hashes"].pop(str(gauge.HARNESS_PATH))
    missing_mandatory["loaded_source_aggregate"] = gauge.canonical_json_hash(
        missing_mandatory["loaded_source_hashes"]
    )
    with pytest.raises(RuntimeError, match="mandatory loaded-source"):
        gauge.validate_phase_a_attempt_payload(missing_mandatory, seal, phase_a)

    missing_addendum = copy.deepcopy(marker)
    missing_addendum["loaded_source_hashes"].pop(str(gauge.IMPLEMENTATION_REVIEW_ADDENDUM))
    missing_addendum["loaded_source_aggregate"] = gauge.canonical_json_hash(
        missing_addendum["loaded_source_hashes"]
    )
    with pytest.raises(RuntimeError, match="mandatory loaded-source"):
        gauge.validate_phase_a_attempt_payload(missing_addendum, seal, phase_a)

    wrong_output = copy.deepcopy(marker)
    wrong_output["outputs"]["audit_json"] = str(phase_a.with_name("wrong_audit.json"))
    with pytest.raises(RuntimeError, match="output binding"):
        gauge.validate_phase_a_attempt_payload(wrong_output, seal, phase_a)


def _optimizer_records(q_star: torch.Tensor, q_new: torch.Tensor) -> tuple[dict, dict]:
    parameter = torch.nn.Parameter(q_star.detach().clone())
    optimizer = torch.optim.Adam(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-15,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
    )
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    before = gauge.optimizer_state_record(optimizer, parameter)
    with torch.no_grad():
        parameter.copy_(q_new)
    return before, gauge.optimizer_state_record(optimizer, parameter)


def _phase_a_raw_step() -> dict:
    q_old = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    gradient = torch.tensor([[0.01, 0.02, 0.0, 0.0], [0.03, 0.01, 0.0, 0.0]], dtype=torch.float32)
    q_star = torch.tensor([[1.02, 0.01, 0.0, 0.0], [1.03, 0.02, 0.0, 0.0]], dtype=torch.float32)
    before, after = _optimizer_records(q_star, q_star)
    return gauge._phase_a_step_record(
        q_old,
        gradient,
        q_star,
        q_star,
        step=1,
        view=0,
        loss=0.2,
        policy="current",
        optimizer_before_policy=before,
        optimizer_after_policy=after,
        projection=None,
    )


def _adversarial_projection_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    q_old = torch.tensor(
        [
            [0.12345679, -0.33333334, 0.7777778, 0.5151515],
            [0.9876543, 0.1111111, -0.2222222, 0.3333333],
        ],
        dtype=torch.float32,
    )
    gradient = torch.tensor(
        [
            [0.03141593, -0.02718282, 0.01618034, -0.01414214],
            [-0.00987654, 0.01234567, -0.02345678, 0.03456789],
        ],
        dtype=torch.float32,
    )
    return q_old, gradient


def _projection_phase_a_step() -> dict:
    q_old, gradient = _adversarial_projection_inputs()
    parameter = torch.nn.Parameter(q_old.clone())
    optimizer = torch.optim.Adam(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-15,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
    )
    unit = torch.nn.functional.normalize(q_old, dim=-1)
    projected = gradient - unit * (unit * gradient).sum(dim=-1, keepdim=True)
    parameter.grad = projected.clone()
    optimizer.step()
    q_star = parameter.detach().clone()
    optimizer_record = gauge.optimizer_state_record(optimizer, parameter)
    diagnostics = gauge.removed_gradient_diagnostics_float64(q_old, gradient)
    projection = {
        "removed_numerator": gauge.tensor_values(diagnostics["numerator"]),
        "removed_denominator": gauge.tensor_values(diagnostics["denominator"]),
        "removed_fraction": gauge.tensor_values(diagnostics["fraction"]),
        "diagnostic_sha256": gauge.tensor_collection_hash(
            [
                ("removed_numerator", diagnostics["numerator"]),
                ("removed_denominator", diagnostics["denominator"]),
                ("removed_fraction", diagnostics["fraction"]),
            ]
        ),
    }
    return gauge._phase_a_step_record(
        q_old,
        gradient,
        q_star,
        q_star,
        step=1,
        view=0,
        loss=0.2,
        policy="gradient_projection_current",
        optimizer_before_policy=optimizer_record,
        optimizer_after_policy=copy.deepcopy(optimizer_record),
        projection=projection,
    )


def _phase_a_current_arm(scale: str) -> dict:
    multiplier = {"0.25": 1.0, "1": 1.002, "4": 1.02}[scale]
    covariance = (torch.eye(3, dtype=torch.float64) * multiplier)[None].tolist()
    sse = {"0.25": 0.010, "1": 0.009, "4": 0.008}[scale]
    checkpoints = {}
    for step in gauge.PHASE_A_CHECKPOINTS:
        view_sse = sse / 9
        per_view = [
            {
                "view": view,
                "color_sse": view_sse,
                "color_count": 3,
                "color_mse": view_sse / 3,
                "color_psnr": gauge.psnr_from_sse(view_sse, 3),
                "loss": view_sse,
                "color_sha256": "1" * 64,
                "alpha_sha256": "2" * 64,
                "depth_sha256": "3" * 64,
            }
            for view in range(9)
        ]
        pooled_count = 27
        pooled_psnr = gauge.psnr_from_sse(sse, pooled_count)
        checkpoints[str(step)] = {
            "step": step,
            "covariance": covariance,
            "per_view": per_view,
            "pooled_color_sse": sse,
            "pooled_color_count": pooled_count,
            "pooled_color_mse": sse / pooled_count,
            "pooled_color_psnr": pooled_psnr,
            "mean_loss": view_sse,
        }
    return {
        "self_target_auc_db": gauge.psnr_from_sse(sse, 27),
        "checkpoints": checkpoints,
        "steps": [_phase_a_raw_step()],
    }


def test_phase_a_seed_and_pooled_materiality_use_frozen_raw_formulas():
    seeds = [
        {
            "seed": seed,
            "arms": {
                "current": {scale: _phase_a_current_arm(scale) for scale in ("0.25", "1", "4")}
            },
        }
        for seed in gauge.SEEDS
    ]
    per_seed = gauge._phase_a_seed_decision(seeds[0])
    assert per_seed["gauge_auc_spread_db"] > 0.05
    assert per_seed["unit_effective_lr_p90"] > 0.01
    assert per_seed["unit_radial_fraction_median"] > 0.1
    assert per_seed["ambient_gauge_material"]
    pooled = gauge._pooled_phase_a_decision(seeds)
    assert pooled["unit_effective_lr_p90"] > 0.01
    assert pooled["unit_radial_fraction_median"] > 0.1
    assert pooled["ambient_gauge_material"]


def test_phase_a_raw_recomputation_rejects_tampered_summaries():
    arm = _phase_a_current_arm("1")
    arm["self_target_auc_db"] += 1.0
    with pytest.raises(ValueError, match="AUC differs"):
        gauge.derive_phase_a_auc(arm)

    step = _phase_a_raw_step()
    step["effective_lr_scale"][0] += 0.1
    with pytest.raises(ValueError, match="effective_lr_scale differs"):
        gauge.derive_phase_a_step_metrics(step)

    count_tamper = _phase_a_raw_step()
    count_tamper["active_gradient_count"] -= 1
    with pytest.raises(ValueError, match="active_gradient_count differs"):
        gauge.derive_phase_a_step_metrics(count_tamper)

    optimizer_tamper = _phase_a_raw_step()
    optimizer_tamper["optimizer_before_policy"]["state_entries"].reverse()
    with pytest.raises(ValueError, match="optimizer state ordering differs"):
        gauge.derive_phase_a_step_metrics(optimizer_tamper)

    order_tamper = _phase_a_raw_step()
    with pytest.raises(ValueError, match="step order differs"):
        gauge.derive_phase_a_step_metrics(order_tamper, expected_step=2)

    policy_tamper = _phase_a_raw_step()
    policy_tamper["q_new"][0][0] += 0.01
    with pytest.raises(ValueError, match="q_new"):
        gauge.derive_phase_a_step_metrics(policy_tamper)


def test_nontrivial_phase_a_step_diagnostics_are_float64_recomputable():
    generator = torch.Generator().manual_seed(9182)
    q_old = torch.nn.functional.normalize(
        torch.randn(7, 4, generator=generator, dtype=torch.float32), dim=-1
    )
    gradient = torch.randn(7, 4, generator=generator, dtype=torch.float32) * 0.003
    q_star = q_old + torch.randn(7, 4, generator=generator, dtype=torch.float32) * 0.002
    q_new = torch.nn.functional.normalize(q_star, dim=-1)
    optimizer_before, optimizer_after = _optimizer_records(q_star, q_new)
    record = gauge._phase_a_step_record(
        q_old,
        gradient,
        q_star,
        q_new,
        step=1,
        view=2,
        loss=0.3,
        policy="unit_retraction",
        optimizer_before_policy=optimizer_before,
        optimizer_after_policy=optimizer_after,
        projection=None,
    )
    derived = gauge.derive_phase_a_step_metrics(record)
    q_star64 = q_star.to(torch.float64)
    expected = (1.0 / torch.linalg.vector_norm(q_star64, dim=-1) - 1.0).abs()
    assert derived["effective_lr_scale"] == expected.tolist()
    float32_norm = torch.linalg.vector_norm(q_star, dim=-1).to(torch.float64)
    assert bool((expected != (1.0 / float32_norm - 1.0).abs()).any())


def test_removed_gradient_helper_promotes_before_normalization_against_adversarial_order():
    q_old, gradient = _adversarial_projection_inputs()
    normalize32_then_promote = torch.nn.functional.normalize(q_old, dim=-1).to(torch.float64)

    q64 = q_old.to(torch.float64)
    g64 = gradient.to(torch.float64)
    q_norm = torch.linalg.vector_norm(q64, dim=-1, keepdim=True)
    unit_reference = q64 / q_norm
    signed_dot = (unit_reference * g64).sum(dim=-1)
    numerator = signed_dot.abs()
    denominator = torch.linalg.vector_norm(g64, dim=-1).clamp_min(gauge.ACTIVE_NORM)
    fraction = numerator / denominator
    projected = g64 - unit_reference * signed_dot.unsqueeze(-1)

    assert not torch.equal(normalize32_then_promote, unit_reference)
    diagnostics = gauge.removed_gradient_diagnostics_float64(q_old, gradient)
    for key, expected in (
        ("numerator", numerator),
        ("denominator", denominator),
        ("fraction", fraction),
        ("projected_gradient", projected),
    ):
        assert torch.equal(diagnostics[key], expected)
        assert diagnostics[key].dtype == torch.float64

    legacy_numerator = (normalize32_then_promote * g64).sum(dim=-1).abs()
    legacy_fraction = legacy_numerator / denominator
    assert not torch.equal(diagnostics["numerator"], legacy_numerator)
    assert not torch.equal(diagnostics["fraction"], legacy_fraction)


@pytest.mark.parametrize(
    "q_old,gradient,match",
    [
        (torch.ones(2, 4, dtype=torch.float64), torch.ones(2, 4), "raw float32"),
        (torch.ones(2, 4), torch.ones(2, 4, dtype=torch.float64), "raw float32"),
        (torch.ones(2, 3), torch.ones(2, 3), "matching \\(N,4\\)"),
        (torch.ones(2, 4), torch.ones(3, 4), "matching \\(N,4\\)"),
        (
            torch.tensor([[float("nan"), 0.0, 0.0, 0.0]], dtype=torch.float32),
            torch.ones(1, 4),
            "finite",
        ),
        (
            torch.ones(1, 4),
            torch.tensor([[float("inf"), 0.0, 0.0, 0.0]], dtype=torch.float32),
            "finite",
        ),
        (torch.zeros(1, 4), torch.ones(1, 4), "zero/near-zero"),
    ],
)
def test_removed_gradient_helper_rejects_invalid_raw_inputs(q_old, gradient, match):
    with pytest.raises(ValueError, match=match):
        gauge.removed_gradient_diagnostics_float64(q_old, gradient)


def test_projection_diagnostic_json_roundtrip_and_tampering_fail_closed():
    record = _projection_phase_a_step()
    restored = json.loads(json.dumps(record, allow_nan=False))
    gauge.derive_phase_a_step_metrics(
        restored,
        expected_policy="gradient_projection_current",
        expected_step=1,
        expected_view=0,
    )
    fractions = gauge.derive_projection_removed_fractions(restored)
    assert fractions == restored["projection"]["removed_fraction"]

    numerator_tamper = copy.deepcopy(restored)
    numerator_tamper["projection"]["removed_numerator"][0] += 1e-12
    with pytest.raises(ValueError, match="stored projected-gradient fractions"):
        gauge.derive_projection_removed_fractions(numerator_tamper)

    fraction_tamper = copy.deepcopy(restored)
    fraction_tamper["projection"]["removed_fraction"][0] += 1e-12
    with pytest.raises(ValueError, match="stored projected-gradient fractions"):
        gauge.derive_projection_removed_fractions(fraction_tamper)

    hash_tamper = copy.deepcopy(restored)
    hash_tamper["projection"]["diagnostic_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="diagnostic hash"):
        gauge.derive_projection_removed_fractions(hash_tamper)


def _native_projection_adam_branch(
    q_old: torch.Tensor, gradient: torch.Tensor, *, call_diagnostic: bool
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    parameter = torch.nn.Parameter(q_old.clone())
    optimizer = torch.optim.Adam(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-15,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
    )
    if call_diagnostic:
        gauge.removed_gradient_diagnostics_float64(q_old, gradient)
    unit = torch.nn.functional.normalize(parameter.detach(), dim=-1)
    optimizer_gradient = gradient - unit * (unit * gradient).sum(dim=-1, keepdim=True)
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = optimizer_gradient.detach().clone()
    optimizer.step()
    return (
        optimizer_gradient,
        parameter.detach().clone(),
        gauge.optimizer_state_record(optimizer, parameter),
    )


def test_removed_gradient_diagnostic_is_observationally_pure_for_native_adam():
    q_old, gradient = _adversarial_projection_inputs()
    q_before = q_old.clone()
    gradient_before = gradient.clone()
    baseline = _native_projection_adam_branch(q_old, gradient, call_diagnostic=False)
    diagnosed = _native_projection_adam_branch(q_old, gradient, call_diagnostic=True)
    assert torch.equal(q_old, q_before)
    assert torch.equal(gradient, gradient_before)
    assert torch.equal(baseline[0], diagnosed[0])
    assert torch.equal(baseline[1], diagnosed[1])
    assert baseline[2] == diagnosed[2]


def test_real_projection_validator_and_native_float32_replay_remain_exact():
    step = _projection_phase_a_step()
    gauge.derive_phase_a_step_metrics(
        step,
        expected_policy="gradient_projection_current",
        expected_step=1,
        expected_view=0,
    )
    replay_source = inspect.getsource(gauge._phase_a_invariants)
    assert 'raw_gradient = torch.tensor(step["gradient"], dtype=torch.float32)' in replay_source
    assert "unit = F.normalize(simulated_q.detach(), dim=-1)" in replay_source
    assert "removed_gradient_diagnostics_float64" not in replay_source

    simulated_q = torch.nn.Parameter(torch.tensor(step["q_old"], dtype=torch.float32))
    simulated_optimizer = torch.optim.Adam(
        [simulated_q],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-15,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
    )
    raw_gradient = torch.tensor(step["gradient"], dtype=torch.float32)
    unit = torch.nn.functional.normalize(simulated_q.detach(), dim=-1)
    optimizer_gradient = raw_gradient - unit * (unit * raw_gradient).sum(dim=-1, keepdim=True)
    assert optimizer_gradient.dtype == torch.float32
    simulated_optimizer.zero_grad(set_to_none=True)
    simulated_q.grad = optimizer_gradient.detach().clone()
    simulated_optimizer.step()
    assert torch.equal(simulated_q.detach(), torch.tensor(step["q_star"], dtype=torch.float32))
    assert (
        gauge.optimizer_state_record(simulated_optimizer, simulated_q)
        == (step["optimizer_before_policy"])
    )


def test_all_removed_gradient_evidence_sites_call_the_shared_helper():
    for function in (
        gauge.gradient_prerequisites,
        gauge.run_phase_a_arm,
        gauge.derive_projection_removed_fractions,
        gauge.recompute_prerequisite_validity,
    ):
        assert inspect.getsource(function).count("removed_gradient_diagnostics_float64") == 1


def _phase_a_preparation(seed: int = 0) -> dict:
    initialization = _toy_gaussians(256)
    selection = gauge.diagnostic_selection(initialization)
    target = initialization.subset(torch.tensor(selection["selected_indices"])).detach()
    _, perturbation = gauge.perturb_diagnostic(target, seed)
    training_hashes = {
        "scene": {"images": "1" * 64},
        "fit_fields": "2" * 64,
        "fit_histories": "3" * 64,
        "fit_order": "4" * 64,
        "initialization_fields": {
            field: gauge.tensor_collection_hash([(field, getattr(initialization, field))])
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "initialization": gauge.gaussians_hash(initialization),
        "initialization_covariance": gauge.tensor_collection_hash(
            [("covariance_float64", gauge.covariance_float64(initialization))]
        ),
    }
    training_hashes["aggregate"] = gauge.canonical_json_hash(training_hashes)
    schedule = gauge.phase_a_schedule(seed)
    return {
        "seed": seed,
        "training_hashes": training_hashes,
        "fit_config": gauge.asdict(gauge.fit_config()),
        "depth_lifter_config": gauge.depth_lifter_config(),
        "initialization_n": initialization.n,
        "initialization_fields": {
            field: gauge.tensor_values(getattr(initialization, field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "training_original_indices": list(gauge.TRAIN_INDICES),
        "selection": selection,
        "perturbation": perturbation,
        "target_hashes": {
            "fields": gauge.gaussians_hash(target),
            "covariance": gauge.tensor_collection_hash(
                [("covariance", gauge.covariance_float64(target))]
            ),
            "renders": "5" * 64,
        },
        "target_fields": {
            field: gauge.tensor_values(getattr(target, field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "target_covariance": gauge.tensor_values(gauge.covariance_float64(target)),
        "schedule": schedule,
        "schedule_sha256": gauge.canonical_json_hash(schedule),
    }


def test_phase_a_selection_schedule_and_construction_summaries_fail_closed():
    preparation = _phase_a_preparation()
    gauge.validate_phase_a_preparation(preparation, 0)

    bad_selection = copy.deepcopy(preparation)
    bad_selection["selection"]["selected_indices"][-1] = 200
    with pytest.raises(ValueError, match="diagnostic selection differs"):
        gauge.validate_phase_a_preparation(bad_selection, 0)

    bad_schedule = copy.deepcopy(preparation)
    bad_schedule["schedule"][0] = (bad_schedule["schedule"][0] + 1) % 9
    bad_schedule["schedule_sha256"] = gauge.canonical_json_hash(bad_schedule["schedule"])
    with pytest.raises(ValueError, match="preparation schedule differs"):
        gauge.validate_phase_a_preparation(bad_schedule, 0)

    candidate = _toy_gaussians(128)
    covariance = gauge._equivalence_record(
        gauge.covariance_float64(candidate), gauge.covariance_float64(candidate)
    )
    construction = {
        "passed": True,
        "covariance": covariance,
        "views": [
            {
                "view": view,
                "color_max_absolute_error": 0.0,
                "alpha_max_absolute_error": 0.0,
                "depth_max_absolute_error": 0.0,
                "color_l1_numerator": 0.0,
                "color_l1_denominator": 1.0,
                "color_l1_relative_error": 0.0,
            }
            for view in range(9)
        ],
        "pooled_color_l1_numerator": 0.0,
        "pooled_color_l1_denominator": 9.0,
        "pooled_color_l1_relative_error": 0.0,
        "candidate_covariance_sha256": gauge.tensor_collection_hash(
            [("covariance", gauge.covariance_float64(candidate))]
        ),
        "reference_covariance_sha256": gauge.tensor_collection_hash(
            [("covariance", gauge.covariance_float64(candidate))]
        ),
        "candidate_render_sha256": "6" * 64,
        "reference_render_sha256": "7" * 64,
    }
    assert gauge.derive_physical_equivalence(
        construction, candidate=candidate, reference=candidate, context="toy"
    )
    construction["pooled_color_l1_numerator"] = 0.1
    with pytest.raises(ValueError, match="pooled numerator differs"):
        gauge.derive_physical_equivalence(
            construction, candidate=candidate, reference=candidate, context="toy"
        )


def test_phase_a_pooled_reduction_rejects_per_view_contradiction():
    arm = _phase_a_current_arm("1")
    arm["checkpoints"]["20"]["pooled_color_sse"] += 0.1
    with pytest.raises(ValueError, match="pooled color SSE differs"):
        gauge.derive_phase_a_auc(arm)


def _phase_b_truth() -> list[dict]:
    truth = []
    for local_view, original_view in enumerate(gauge.HELD_OUT_INDICES):
        color = torch.full((4, 4, 3), 0.25, dtype=torch.float32)
        alpha = torch.ones(4, 4, dtype=torch.float32)
        depth = torch.ones(4, 4, dtype=torch.float32)
        expected_depth = depth / alpha.clamp_min(1e-6)
        support = alpha > 0.05
        truth.append(
            {
                "local_view": local_view,
                "original_view": original_view,
                "color": gauge.tensor_values(color),
                "alpha": gauge.tensor_values(alpha),
                "accumulated_depth": gauge.tensor_values(depth),
                "expected_depth": gauge.tensor_values(expected_depth),
                "support": gauge.tensor_values(support),
                "support_count": int(support.sum()),
                "sha256": gauge.tensor_collection_hash(
                    [
                        ("color", color),
                        ("alpha", alpha),
                        ("accumulated_depth", depth),
                        ("expected_depth", expected_depth),
                        ("support", support),
                    ]
                ),
            }
        )
    return truth


def _phase_b_gaussians() -> Gaussians3D:
    n = 256
    return Gaussians3D(
        means=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        log_scales=torch.zeros(n, 3),
        opacity=torch.full((n,), 0.2),
        sh=torch.zeros(n, 16, 3),
    )


def _phase_b_checkpoint(step: int, foreground_psnr: float, truth: list[dict]) -> dict:
    gaussians = _phase_b_gaussians()
    delta = torch.tensor(10 ** (-foreground_psnr / 20.0), dtype=torch.float32)
    per_view = []
    render_tensors = []
    pooled = {
        "full_color_sse": 0.0,
        "full_color_count": 0,
        "foreground_color_sse": 0.0,
        "foreground_color_count": 0,
        "ssim_sum": 0.0,
        "ssim_count": 0,
        "depth_squared_error": 0.0,
        "depth_count": 0,
        "alpha_intersection_count": 0,
        "alpha_union_count": 0,
        "covered_truth_count": 0,
        "truth_support_count": 0,
    }
    for index, target in enumerate(truth):
        truth_color = torch.tensor(target["color"], dtype=torch.float32)
        truth_alpha = torch.tensor(target["alpha"], dtype=torch.float32)
        truth_depth = torch.tensor(target["accumulated_depth"], dtype=torch.float32)
        truth_expected = torch.tensor(target["expected_depth"], dtype=torch.float32)
        support = torch.tensor(target["support"], dtype=torch.bool)
        predicted_color = truth_color + delta
        predicted_alpha = torch.ones_like(truth_alpha)
        predicted_depth = torch.full_like(truth_depth, 1.1)
        predicted_expected = predicted_depth / predicted_alpha.clamp_min(1e-6)
        error = predicted_color.to(torch.float64) - truth_color.to(torch.float64)
        full_sse = float(error.square().sum())
        full_count = error.numel()
        foreground_error = error[support]
        foreground_sse = float(foreground_error.square().sum())
        foreground_count = foreground_error.numel()
        view_ssim = float(gauge.ssim(predicted_color, truth_color))
        depth_error = (predicted_expected.to(torch.float64) - truth_expected.to(torch.float64))[
            support
        ]
        depth_sse = float(depth_error.square().sum())
        depth_count = depth_error.numel()
        intersection = int(support.sum())
        union = intersection
        numeric = {
            "full_color_sse": full_sse,
            "full_color_count": full_count,
            "foreground_color_sse": foreground_sse,
            "foreground_color_count": foreground_count,
            "ssim_sum": view_ssim,
            "ssim_count": 1,
            "depth_squared_error": depth_sse,
            "depth_count": depth_count,
            "alpha_intersection_count": intersection,
            "alpha_union_count": union,
            "covered_truth_count": intersection,
            "truth_support_count": int(support.sum()),
        }
        for key, value in numeric.items():
            pooled[key] += value
        per_view.append(
            {
                "local_view": index,
                "original_view": gauge.HELD_OUT_INDICES[index],
                "full_color_sse": full_sse,
                "full_color_count": full_count,
                "full_color_psnr": gauge.psnr_from_sse(full_sse, full_count),
                "foreground_color_sse": foreground_sse,
                "foreground_color_count": foreground_count,
                "foreground_color_psnr": gauge.psnr_from_sse(foreground_sse, foreground_count),
                "ssim": view_ssim,
                "predicted_color": gauge.tensor_values(predicted_color),
                "predicted_alpha": gauge.tensor_values(predicted_alpha),
                "predicted_accumulated_depth": gauge.tensor_values(predicted_depth),
                "predicted_expected_depth": gauge.tensor_values(predicted_expected),
                "truth_color": target["color"],
                "truth_alpha": target["alpha"],
                "truth_accumulated_depth": target["accumulated_depth"],
                "truth_expected_depth": target["expected_depth"],
                "truth_support": target["support"],
                "depth_squared_error": depth_sse,
                "depth_count": depth_count,
                "alpha_intersection_count": intersection,
                "alpha_union_count": union,
                "covered_truth_count": intersection,
                "truth_support_count": int(support.sum()),
                "predicted_sha256": gauge.tensor_collection_hash(
                    [
                        ("color", predicted_color),
                        ("alpha", predicted_alpha),
                        ("accumulated_depth", predicted_depth),
                        ("expected_depth", predicted_expected),
                    ]
                ),
                "truth_sha256": target["sha256"],
            }
        )
        render_tensors.extend(
            [
                (f"view_{index}/color", predicted_color),
                (f"view_{index}/alpha", predicted_alpha),
                (f"view_{index}/depth", predicted_depth),
            ]
        )
    covariance = gauge.covariance_float64(gaussians)
    scene_extent = 1.0
    pooled_record = {
        **pooled,
        "full_color_psnr": gauge.psnr_from_sse(
            pooled["full_color_sse"], pooled["full_color_count"]
        ),
        "foreground_color_psnr": gauge.psnr_from_sse(
            pooled["foreground_color_sse"], pooled["foreground_color_count"]
        ),
        "ssim": pooled["ssim_sum"] / pooled["ssim_count"],
        "scene_extent": scene_extent,
        "normalized_depth_rmse": math.sqrt(pooled["depth_squared_error"] / pooled["depth_count"])
        / scene_extent,
        "alpha_iou": pooled["alpha_intersection_count"] / pooled["alpha_union_count"],
        "truth_support_coverage": pooled["covered_truth_count"] / pooled["truth_support_count"],
    }
    return {
        "step": step,
        "wall_seconds": float(step) + 0.1,
        "per_view": per_view,
        "pooled": pooled_record,
        "raw_quaternion_norms": gauge.tensor_values(
            torch.linalg.vector_norm(gaussians.quats.to(torch.float64), dim=-1)
        ),
        "effective_quaternions": gauge.tensor_values(gaussians.quats),
        "fields": {
            field: gauge.tensor_values(getattr(gaussians, field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "covariance": gauge.tensor_values(covariance),
        "field_hash": gauge.gaussians_hash(gaussians),
        "field_shapes": {
            field: list(getattr(gaussians, field).shape)
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "non_quaternion_hash": gauge.non_quaternion_hash(gaussians),
        "covariance_hash": gauge.tensor_collection_hash([("covariance", covariance)]),
        "render_hash": gauge.tensor_collection_hash(render_tensors),
    }


def _phase_b_arm(
    seed: int, arm: str, foreground_psnr: float, truth: list[dict], raw_hash: str
) -> dict:
    checkpoints = {
        str(step): _phase_b_checkpoint(step, 20.0 if step == 0 else foreground_psnr, truth)
        for step in gauge.PHASE_B_CHECKPOINTS
    }
    q = torch.tensor(checkpoints["0"]["effective_quaternions"], dtype=torch.float32)
    quaternion_steps = [
        {
            "step": step,
            "q_old": gauge.tensor_values(q),
            "q_star": gauge.tensor_values(q),
            "q_new": gauge.tensor_values(q),
            "q_old_norm": gauge.tensor_values(
                torch.linalg.vector_norm(q.to(torch.float64), dim=-1)
            ),
            "q_star_norm": gauge.tensor_values(
                torch.linalg.vector_norm(q.to(torch.float64), dim=-1)
            ),
            "q_new_norm": gauge.tensor_values(
                torch.linalg.vector_norm(q.to(torch.float64), dim=-1)
            ),
            "q_old_sha256": gauge.tensor_collection_hash([("q_old", q)]),
            "q_star_sha256": gauge.tensor_collection_hash([("q_star", q)]),
            "q_new_sha256": gauge.tensor_collection_hash([("q_new", q)]),
        }
        for step in range(1, 121)
    ]
    schedule = gauge.phase_b_schedule(seed)
    n = q.shape[0]
    eval_steps = [30, 60, 90, 120]
    history = {
        "loss": [0.2] * 120,
        "loss_terms": [
            {"l1": 0.2, "alpha": 0.0, "opacity_reg": 0.0, "scale_reg": 0.0} for _ in range(120)
        ],
        "psnr": [[step, 20.0] for step in eval_steps],
        "elapsed": [[step, float(step)] for step in eval_steps],
        "n_gaussians": [[step, n] for step in eval_steps],
        "active_sh_degree": [[step, degree] for step, degree in zip(eval_steps, range(4))],
        "sampled_train_views": schedule["local_positions"],
        "sh_color_diagnostics": [],
        "kernel_support_diagnostics": [],
        "density_stats": None,
        "density_strategy": "none",
        "resolved_sh_degree_interval": 30,
        "peak_vram_gb": 0.0,
    }
    foreground = [
        checkpoints[str(step)]["pooled"]["foreground_color_psnr"]
        for step in gauge.PHASE_B_CHECKPOINTS
    ]
    return {
        "arm": arm,
        "config": gauge.asdict(gauge.train_config(seed, arm)),
        "common_input_hash": checkpoints["0"]["field_hash"],
        "common_input_raw_initialization_hash": raw_hash,
        "effective_parameter_hash": checkpoints["0"]["field_hash"],
        "effective_quaternion_hash": gauge.tensor_collection_hash([("effective_quaternions", q)]),
        "schedule_probe": schedule,
        "sampled_train_views": schedule["local_positions"],
        "sampled_original_views": schedule["original_view_indices"],
        "quaternion_steps": quaternion_steps,
        "checkpoints": checkpoints,
        "heldout_foreground_auc_db": gauge.normalized_auc(
            list(gauge.PHASE_B_CHECKPOINTS), foreground
        ),
        "history": history,
        "final_field_hash": checkpoints["120"]["field_hash"],
        "final_n": n,
        "validity": {
            "finite": True,
            "complete_checkpoints": True,
            "exact_schedule": True,
            "fixed_topology": True,
            "candidate_norm_invariant": True,
        },
    }


def _phase_b_payload() -> dict:
    truth = _phase_b_truth()
    seeds = []
    for seed in gauge.SEEDS:
        raw_hash = gauge.canonical_json_hash({"seed": seed, "kind": "raw-init"})
        training_hashes = {
            "scene": {"images": gauge.canonical_json_hash({"seed": seed, "scene": True})},
            "fit_fields": gauge.canonical_json_hash({"seed": seed, "fit": "fields"}),
            "fit_histories": gauge.canonical_json_hash({"seed": seed, "fit": "history"}),
            "fit_order": gauge.canonical_json_hash(list(range(9))),
            "initialization_fields": {
                field: gauge.canonical_json_hash({"seed": seed, "field": field})
                for field in ("means", "quats", "log_scales", "opacity", "sh")
            },
            "initialization": raw_hash,
            "initialization_covariance": gauge.canonical_json_hash(
                {"seed": seed, "covariance": True}
            ),
        }
        training_hashes["aggregate"] = gauge.canonical_json_hash(training_hashes)
        arms = {
            "current": _phase_b_arm(seed, "current", 20.0, truth, raw_hash),
            "unit_retraction": _phase_b_arm(seed, "unit_retraction", 20.1, truth, raw_hash),
            "tangent_displacement_retraction": _phase_b_arm(
                seed, "tangent_displacement_retraction", 20.15, truth, raw_hash
            ),
        }
        seeds.append(
            {
                "seed": seed,
                "training_order": list(gauge.TRAINING_ORDER[seed]),
                "training_hashes": training_hashes,
                "scene_extent": 1.0,
                "truth": copy.deepcopy(truth),
                "truth_aggregate_sha256": gauge.canonical_json_hash(truth),
                "step_zero_invariant": gauge.phase_b_step_zero_invariant(arms),
                "arms": arms,
            }
        )
    return {"seeds": seeds}


def test_phase_b_utility_safety_and_preference_recompute_from_raw_counts():
    decision = gauge.recompute_phase_b_decision(_phase_b_payload())
    assert decision["candidates"]["unit_retraction"]["passed"]
    assert decision["candidates"]["tangent_displacement_retraction"]["passed"]
    assert decision["preference"]["tangent_preferred"]
    assert decision["confirmatory_candidate"] == "tangent_displacement_retraction"
    assert not decision["production_default_change_authorized"]


def test_phase_b_raw_evidence_layers_reject_adversarial_tampering():
    payload = _phase_b_payload()
    truth = payload["seeds"][0]["truth"]
    checkpoint = payload["seeds"][0]["arms"]["current"]["checkpoints"]["30"]

    pooled = copy.deepcopy(checkpoint)
    pooled["pooled"]["foreground_color_sse"] += 1.0
    with pytest.raises(ValueError, match="pooled foreground_color_sse differs"):
        gauge.derive_phase_b_checkpoint(
            pooled,
            expected_step=30,
            expected_truth=truth,
            expected_scene_extent=1.0,
        )

    raw_view = copy.deepcopy(checkpoint)
    raw_view["per_view"][0]["full_color_sse"] += 1.0
    with pytest.raises(ValueError, match="view 0 full_color_sse differs"):
        gauge.derive_phase_b_checkpoint(
            raw_view,
            expected_step=30,
            expected_truth=truth,
            expected_scene_extent=1.0,
        )

    raw_prediction = copy.deepcopy(checkpoint)
    raw_prediction["per_view"][0]["predicted_color"][0][0][0] += 0.01
    with pytest.raises(ValueError, match="view 0 full_color_sse differs"):
        gauge.derive_phase_b_checkpoint(
            raw_prediction,
            expected_step=30,
            expected_truth=truth,
            expected_scene_extent=1.0,
        )

    raw_fields = copy.deepcopy(checkpoint)
    raw_fields["fields"]["means"][0][0] = 0.1
    with pytest.raises(ValueError, match="field hash differs"):
        gauge.derive_phase_b_checkpoint(
            raw_fields,
            expected_step=30,
            expected_truth=truth,
            expected_scene_extent=1.0,
        )

    wrong_extent = copy.deepcopy(checkpoint)
    wrong_extent["pooled"]["scene_extent"] = 2.0
    with pytest.raises(ValueError, match="scene extent differs"):
        gauge.derive_phase_b_checkpoint(
            wrong_extent,
            expected_step=30,
            expected_truth=truth,
            expected_scene_extent=1.0,
        )

    arm = copy.deepcopy(payload["seeds"][0]["arms"]["unit_retraction"])
    arm["validity"]["finite"] = False
    with pytest.raises(ValueError, match="stored validity summary differs"):
        gauge._phase_b_arm_metrics(
            arm,
            expected_arm="unit_retraction",
            seed=0,
            expected_truth=truth,
            expected_scene_extent=1.0,
        )

    q_policy = copy.deepcopy(payload["seeds"][0]["arms"]["unit_retraction"])
    q_policy["quaternion_steps"][0]["q_new"][0][0] = 0.9
    with pytest.raises(ValueError, match="q_new norms"):
        gauge._phase_b_arm_metrics(
            q_policy,
            expected_arm="unit_retraction",
            seed=0,
            expected_truth=truth,
            expected_scene_extent=1.0,
        )

    stored_step_zero = copy.deepcopy(payload)
    stored_step_zero["seeds"][0]["step_zero_invariant"]["arms"]["unit_retraction"]["passed"] = False
    with pytest.raises(ValueError, match="stored step-zero invariant differs"):
        gauge.recompute_phase_b_decision(stored_step_zero)

    common_input = copy.deepcopy(payload)
    common_input["seeds"][0]["arms"]["unit_retraction"]["common_input_hash"] = "f" * 64
    with pytest.raises(ValueError, match="common effective input hash differs"):
        gauge.recompute_phase_b_decision(common_input)

    bad_truth = copy.deepcopy(truth)
    bad_truth[0]["support_count"] -= 1
    with pytest.raises(ValueError, match="support count differs"):
        gauge.validate_phase_b_truth(bad_truth)

    bad_identity = copy.deepcopy(payload["seeds"][0]["arms"]["current"])
    bad_identity["arm"] = "unit_retraction"
    with pytest.raises(ValueError, match="arm identity differs"):
        gauge._phase_b_arm_metrics(
            bad_identity,
            expected_arm="current",
            seed=0,
            expected_truth=truth,
            expected_scene_extent=1.0,
        )


def test_train_config_has_only_frozen_arm_difference():
    configs = [gauge.train_config(0, arm) for arm in gauge.PHASE_B_ARMS]
    payloads = [config.__dict__.copy() for config in configs]
    for payload in payloads:
        payload.pop("quaternion_update_policy")
    assert payloads[0] == payloads[1] == payloads[2]
    assert [config.quaternion_update_policy for config in configs] == list(gauge.PHASE_B_ARMS)


def test_canonical_json_refuses_nan():
    with pytest.raises(ValueError):
        json.loads(gauge.canonical_json_bytes({"value": float("nan")}))
