"""Outcome-free toy tests for the Stage-1 fit-parameterization protocol engine."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
from benchmarks import stage1_fit_parameterization as protocol


def test_complete_status_and_in_process_cli_calls_are_side_effect_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert protocol.IMPLEMENTATION_COMPLETE is True
    assert protocol.IMPLEMENTATION_GAPS == ()
    assert protocol.implementation_status() == {
        "ready": True,
        "missing_clause_count": 0,
        "missing_clauses": [],
        "seal_authorized": True,
        "scientific_run_authorized": True,
    }

    monkeypatch.setattr(
        protocol,
        "run_verification",
        lambda: pytest.fail("in-process CLI must not run verification"),
    )
    for arguments in (
        ["seal", "--output", str(tmp_path / "seal.json")],
        [
            "run",
            "--seal",
            str(tmp_path / "seal.json"),
            "--output",
            str(tmp_path / "result.json"),
        ],
    ):
        with pytest.raises(RuntimeError, match="actual preregistered process CLI"):
            protocol.main(arguments)
    assert list(tmp_path.iterdir()) == []


def test_semantic_array_hash_matches_frozen_little_endian_contract_and_ignores_name() -> None:
    value = np.asarray([[1.25, -2.5], [3.0, 4.5]], dtype=">f4")
    normalized = np.ascontiguousarray(value.astype("<f4"))
    token = np.dtype("<f4").str.encode("ascii")
    shape = np.asarray(value.shape, dtype="<i8").tobytes(order="C")
    expected = hashlib.sha256(
        token + b"\0" + shape + b"\0" + normalized.tobytes(order="C")
    ).hexdigest()
    assert protocol.array_content_sha256(value) == expected
    first_manifest, first_collection = protocol.array_manifest({"a/value": value})
    second_manifest, second_collection = protocol.array_manifest({"b/value": value})
    assert first_manifest[0]["content_sha256"] == second_manifest[0]["content_sha256"]
    assert first_collection != second_collection


def test_little_endian_normalization_preserves_zero_dimensional_scalars() -> None:
    scalar = np.asarray(7, dtype=">i8")
    normalized = protocol.little_endian_array(scalar)
    assert normalized.dtype == np.dtype("<i8")
    assert normalized.shape == ()
    assert int(normalized) == 7

    archive = protocol.RawArchive()
    stored = archive.add("scalar/value", scalar)
    assert stored.shape == ()
    assert protocol.array_manifest(archive.arrays)[0][0]["shape"] == []


def test_fit_config_contract_matches_literal_block_execution() -> None:
    contract = protocol.fit_config_contract()
    assert set(contract) == set(protocol.BLOCKS)
    for block in protocol.BLOCKS:
        assert set(contract[block]) == set(protocol.ARMS)
        for arm in protocol.ARMS:
            assert contract[block][arm]["appearance_parameterization"] == arm
            assert contract[block][arm]["freeze_geometry"] is (block == "appearance_only")
    assert contract["joint"][protocol.ARMS[0]] == {
        **protocol.frozen_fit_config(protocol.ARMS[0]).__dict__,
        "freeze_geometry": False,
    }


def test_protocol_invalid_sanitizes_nonfinite_structured_evidence() -> None:
    error = protocol.ProtocolInvalid(
        "equivalence",
        "non-finite diagnostic",
        {
            "tensor": torch.tensor([float("nan"), float("inf"), -float("inf"), 2.0]),
            "array": np.asarray([np.float32("nan")], dtype=np.float32),
            "path": Path("raw/evidence"),
        },
    )
    assert error.phase == "equivalence"
    assert error.reason == str(error) == "non-finite diagnostic"
    assert error.evidence == {
        "tensor": [
            {"value": 0.0, "nonfinite_classification": 1},
            {"value": 0.0, "nonfinite_classification": 2},
            {"value": 0.0, "nonfinite_classification": 3},
            2.0,
        ],
        "array": [{"value": 0.0, "nonfinite_classification": 1}],
        "path": "raw/evidence",
    }
    assert "NaN" not in protocol.canonical_json(error.evidence)
    assert "Infinity" not in protocol.canonical_json(error.evidence)


@pytest.mark.parametrize(
    "payload",
    (
        '{"artifact_type":"first","artifact_type":"second"}',
        '{"outer":{"sha256":"first","sha256":"second"}}',
    ),
)
def test_strict_json_loader_rejects_duplicate_keys_at_every_depth(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        protocol.strict_json_load(path)


def test_raw_sidecar_is_uncompressed_pickle_free_and_tamper_evident(tmp_path: Path) -> None:
    arrays = {
        "identity/seed_indices": np.asarray([0, 1, 2], dtype=np.int64),
        "checkpoint/render": np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        "nullable/defined": np.asarray([True, False], dtype=np.bool_),
    }
    path = tmp_path / "raw.npz"
    binding = protocol.write_raw_sidecar(path, arrays)
    loaded = protocol.validate_raw_sidecar(path, binding)
    assert sorted(loaded) == sorted(arrays)
    for name in arrays:
        assert np.array_equal(loaded[name], arrays[name])

    corrupt = dict(binding)
    corrupt["collection_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="semantic manifest"):
        protocol.validate_raw_sidecar(path, corrupt)

    numeric_alias = json.loads(json.dumps(binding))
    numeric_alias["manifest"][0]["byte_length"] = float(numeric_alias["manifest"][0]["byte_length"])
    with pytest.raises(ValueError, match="semantic manifest"):
        protocol.validate_raw_sidecar(path, numeric_alias)

    for key, value, message in (
        ("path", str(tmp_path / "same-content-wrong-path.npz"), "path differs"),
        ("format", "numpy.savez/compressed", "format or allow_pickle"),
        ("allow_pickle", True, "format or allow_pickle"),
        ("array_count", False, "nonnegative integer"),
        ("array_count", 1.0, "nonnegative integer"),
    ):
        malformed_binding = dict(binding)
        malformed_binding[key] = value
        with pytest.raises(ValueError, match=message):
            protocol.validate_raw_sidecar(path, malformed_binding)

    malformed_names = (
        7,
        "",
        "/x",
        "x/",
        "x//y",
        "file",
        ".",
        "..",
        "x/./y",
        "x/../y",
        "x\\y",
        "x\x00y",
        "x\ny",
        "café",
    )
    for index, malformed_name in enumerate(malformed_names):
        malformed_path = tmp_path / f"malformed-{index}.npz"
        with pytest.raises(ValueError, match="invalid raw-array logical name"):
            protocol.write_raw_sidecar(
                malformed_path,
                {malformed_name: np.asarray([0], dtype=np.int64)},
            )
        assert not malformed_path.exists()
    with pytest.raises(ValueError, match="invalid_evidence must be boolean"):
        protocol.write_raw_sidecar(
            tmp_path / "nonboolean-mode.npz",
            {"value": np.asarray([0], dtype=np.int64)},
            invalid_evidence=1,
        )

    hostile_path = tmp_path / "self-consistent-hostile-name.npz"
    hostile_arrays = {"x/../y": np.asarray([1], dtype=np.int64)}
    with hostile_path.open("xb") as handle:
        np.savez(handle, **hostile_arrays)
    hostile_manifest, hostile_collection = protocol.array_manifest(hostile_arrays)
    hostile_binding = {
        "path": str(hostile_path.resolve()),
        "sha256": protocol.sha256_file(hostile_path),
        "collection_sha256": hostile_collection,
        "array_count": 1,
        "manifest": hostile_manifest,
        "format": "numpy.savez/uncompressed",
        "allow_pickle": False,
        "invalid_evidence": False,
    }
    with pytest.raises(ValueError, match="invalid raw-array logical name"):
        protocol.validate_raw_sidecar(hostile_path, hostile_binding)


def test_raw_archive_rejects_duplicate_names_and_preserves_nonfinite_classification() -> None:
    archive = protocol.RawArchive()
    archive.phase = "appearance_only"
    archive.add("seed/view/value", torch.tensor([1.0]))
    with pytest.raises(protocol.ProtocolInvalid, match="name reused"):
        archive.add("seed/view/value", torch.tensor([2.0]))

    with pytest.raises(protocol.ProtocolInvalid, match="non-finite raw array"):
        archive.add("seed/view/offending", np.asarray([0.0, np.nan, np.inf, -np.inf]))
    assert np.array_equal(
        archive.arrays["seed/view/offending/nonfinite_classification"],
        np.asarray([0, 1, 2, 3], dtype=np.uint8),
    )
    assert np.isnan(archive.arrays["seed/view/offending"][1])
    with pytest.raises(protocol.ProtocolInvalid, match="classification names are reserved"):
        protocol.RawArchive().add("x/nonfinite_classification", np.asarray([0], dtype=np.uint8))
    for malformed_name in (
        None,
        7,
        "",
        "/x",
        "x/",
        "x//y",
        "file",
        ".",
        "..",
        "x/./y",
        "x/../y",
        "x\\y",
        "x\x00y",
        "x\ny",
        "café",
    ):
        with pytest.raises(protocol.ProtocolInvalid, match="invalid raw-array logical name"):
            protocol.RawArchive().add(malformed_name, np.asarray([0], dtype=np.int64))
    with pytest.raises(protocol.ProtocolInvalid, match="defined flag must be boolean"):
        protocol.RawArchive().add_nullable("nullable", 1.0, defined=1)
    for malformed_nullable_name in (None, 7):
        with pytest.raises(protocol.ProtocolInvalid, match="invalid raw-array logical name"):
            protocol.RawArchive().add_nullable(malformed_nullable_name, 1.0, defined=True)
    with pytest.raises(protocol.ProtocolInvalid, match="unsupported raw array string_value"):
        protocol.RawArchive().add("string_value", np.asarray(["forbidden"]))


def test_invalid_raw_sidecar_preserves_nonfinite_bytes_and_requires_exact_mask(
    tmp_path: Path,
) -> None:
    archive = protocol.RawArchive()
    with pytest.raises(protocol.ProtocolInvalid, match="non-finite raw array"):
        archive.add("offending", np.asarray([np.nan, np.inf, -np.inf], dtype="<f8"))
    path = tmp_path / "invalid_RAW.npz"
    binding = protocol.write_raw_sidecar(path, archive.arrays, invalid_evidence=True)
    loaded = protocol.validate_raw_sidecar(path, binding, invalid_evidence=True)
    assert loaded["offending"].tobytes() == archive.arrays["offending"].tobytes()
    assert np.array_equal(
        loaded["offending/nonfinite_classification"], np.asarray([1, 2, 3], dtype=np.uint8)
    )

    with pytest.raises(ValueError, match="valid raw sidecar contains non-finite"):
        protocol.write_raw_sidecar(
            tmp_path / "wrong_mode.npz", archive.arrays, invalid_evidence=False
        )
    malformed = dict(archive.arrays)
    malformed["offending/nonfinite_classification"] = np.asarray([1, 3, 2], dtype=np.uint8)
    with pytest.raises(ValueError, match="classification differs"):
        protocol.write_raw_sidecar(tmp_path / "wrong_mask.npz", malformed, invalid_evidence=True)
    with pytest.raises(ValueError, match="lacks classification"):
        protocol.write_raw_sidecar(
            tmp_path / "missing_mask.npz",
            {"offending": np.asarray([np.nan], dtype=np.float32)},
            invalid_evidence=True,
        )
    with pytest.raises(ValueError, match="orphan non-finite classification"):
        protocol.write_raw_sidecar(
            tmp_path / "orphan_mask.npz",
            {"orphan/nonfinite_classification": np.asarray([1], dtype=np.uint8)},
            invalid_evidence=True,
        )
    with pytest.raises(ValueError, match="classification sibling|no offending value"):
        protocol.write_raw_sidecar(
            tmp_path / "finite_mask.npz",
            {
                "finite": np.asarray([1.0], dtype=np.float32),
                "finite/nonfinite_classification": np.asarray([0], dtype=np.uint8),
            },
            invalid_evidence=True,
        )
    with pytest.raises(ValueError, match="invalid-evidence mode differs"):
        protocol.validate_raw_sidecar(path, binding, invalid_evidence=False)


def test_current_and_candidate_jacobians_match_autograd_and_null_identity() -> None:
    torch.manual_seed(19)
    s = torch.randn(5, dtype=torch.float64, requires_grad=True)
    u = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.sigmoid(s)
    color = torch.sigmoid(u)
    expected_current = torch.stack(
        [
            torch.autograd.functional.jacobian(
                lambda theta: torch.sigmoid(theta[3]) * torch.sigmoid(theta[:3]),
                torch.cat([u[index], s[index, None]]),
            )
            for index in range(5)
        ]
    )
    actual_current = protocol.current_jacobian(weight, color)
    assert torch.allclose(actual_current, expected_current, atol=1e-14, rtol=1e-14)
    diagnostics = protocol.jacobian_diagnostics(actual_current, current=True)
    assert torch.equal(diagnostics["rank"], torch.full((5,), 3, dtype=torch.int64))
    assert torch.max(diagnostics["null_residual"]) < 1e-14
    analytic = protocol.analytic_current_null(
        weight, color, diagnostics["null_vector"], diagnostics["rank"]
    )
    assert bool(analytic["defined"].all())
    assert torch.allclose(analytic["alignment"], torch.ones(5, dtype=torch.float64), atol=1e-12)

    r = torch.randn(5, 3, dtype=torch.float64)
    candidate = torch.sigmoid(r)
    expected_candidate = torch.diag_embed(candidate * (1.0 - candidate))
    assert torch.equal(protocol.candidate_jacobian(candidate), expected_candidate)


@pytest.mark.parametrize("arm", protocol.ARMS)
def test_chain_rule_equations_match_autograd(arm: str) -> None:
    torch.manual_seed(23)
    grad_amplitude = torch.randn(4, 3)
    if arm == "weight_color_9p":
        s = torch.randn(4, requires_grad=True)
        u = torch.randn(4, 3, requires_grad=True)
        weight = torch.sigmoid(s)
        color = torch.sigmoid(u)
        loss = (weight[:, None] * color * grad_amplitude).sum()
        actual = torch.autograd.grad(loss, (u, s))
        expected = protocol.chain_rule_expected(arm, grad_amplitude, weight, color)
        assert torch.allclose(actual[0], expected["u"], atol=1e-7, rtol=1e-6)
        assert torch.allclose(actual[1], expected["s"], atol=1e-7, rtol=1e-6)
    else:
        r = torch.randn(4, 3, requires_grad=True)
        color = torch.sigmoid(r)
        loss = (color * grad_amplitude).sum()
        (actual,) = torch.autograd.grad(loss, (r,))
        expected = protocol.chain_rule_expected(arm, grad_amplitude, torch.ones(4), color)["r"]
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_adam_reconstruction_matches_torch_for_multiple_steps() -> None:
    torch.manual_seed(29)
    parameter = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = torch.optim.Adam(
        [parameter],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
        decoupled_weight_decay=False,
    )
    exp_avg = torch.zeros_like(parameter)
    exp_avg_sq = torch.zeros_like(parameter)
    for step in range(3):
        before = parameter.detach().clone()
        gradient = torch.randn_like(parameter)
        expected = protocol.adam_reconstruct(before, gradient, exp_avg, exp_avg_sq, step, 0.01)
        parameter.grad = gradient.clone()
        optimizer.step()
        state = optimizer.state[parameter]
        assert torch.allclose(parameter, expected["parameter_after"], atol=2e-7, rtol=2e-5)
        assert torch.equal(state["exp_avg"], expected["exp_avg_after"])
        assert torch.equal(state["exp_avg_sq"], expected["exp_avg_sq_after"])
        assert int(state["step"]) == expected["step_after"]
        exp_avg = state["exp_avg"].clone()
        exp_avg_sq = state["exp_avg_sq"].clone()


def test_histogram_edges_and_auc_use_the_exact_frozen_conventions() -> None:
    edges = torch.tensor(protocol.HISTOGRAM_EDGES, dtype=torch.float64)
    counts = protocol.histogram_counts(edges)
    assert int(counts.sum()) == len(edges)
    assert counts[0] == 1
    assert torch.equal(counts[1:-1], torch.ones_like(counts[1:-1]))
    assert counts[-1] == 2  # .999 enters the last bin and the terminal edge 1 is included.

    linear = [float(step) for step in protocol.CHECKPOINTS]
    assert protocol.normalized_trapezoid_auc(linear) == pytest.approx(60.0)


def test_checkpoint_diagnostic_reductions_preserve_exact_populations() -> None:
    fit_rows = protocol._expected_fit_rows()
    fit = {
        "block_index": np.asarray([row[1] for row in fit_rows], dtype=np.int64),
        "seed_position": np.asarray([row[2] for row in fit_rows], dtype=np.int64),
        "seed": np.asarray([row[3] for row in fit_rows], dtype=np.int64),
        "local_view": np.asarray([row[4] for row in fit_rows], dtype=np.int64),
        "original_view": np.asarray([row[5] for row in fit_rows], dtype=np.int64),
        "arm_code": np.asarray([row[6] for row in fit_rows], dtype=np.int64),
    }
    checkpoint_rows = protocol._expected_checkpoint_rows()
    checkpoint = {
        "checkpoint_index": np.asarray([row[0] for row in checkpoint_rows], dtype=np.int64),
        "fit_index": np.asarray([row[1] for row in checkpoint_rows], dtype=np.int64),
        "step": np.asarray([row[2] for row in checkpoint_rows], dtype=np.int64),
        "channel_count": np.full(len(checkpoint_rows), 48 * 48 * 3, dtype=np.int64),
        "below_count": np.zeros(len(checkpoint_rows), dtype=np.int64),
        "above_count": np.zeros(len(checkpoint_rows), dtype=np.int64),
    }

    def detail_table(arm_code: int) -> dict[str, np.ndarray]:
        identities = protocol._expected_checkpoint_arm_rows(arm_code)
        rows = len(identities)
        fields = 3 if arm_code == 0 else 1
        counts = np.zeros((rows, fields, 6), dtype=np.int64)
        output_counts = [450, 150, 450] if arm_code == 0 else [450]
        for field_index, output_count in enumerate(output_counts):
            counts[:, field_index, 1] = output_count
            counts[:, field_index, 4] = output_count
        count_defined = np.ones((rows, fields, 6), dtype=np.bool_)
        fraction_defined = np.ones((rows, fields, 4), dtype=np.bool_)
        if arm_code == 0:
            count_defined[:, 2] = np.asarray(
                [False, False, True, True, True, False], dtype=np.bool_
            )
            fraction_defined[:, 2] = np.asarray([False, True, True, False], dtype=np.bool_)
        histograms = np.zeros((rows, fields, len(protocol.HISTOGRAM_EDGES) - 1), dtype=np.int64)
        for field_index, output_count in enumerate(output_counts):
            histograms[:, field_index, 0] = output_count
        return {
            "checkpoint_index": np.asarray([row[0] for row in identities], dtype=np.int64),
            "fit_index": np.asarray([row[1] for row in identities], dtype=np.int64),
            "step": np.asarray([row[2] for row in identities], dtype=np.int64),
            "rank": np.full((rows, protocol.COMPONENTS), 3, dtype=np.int64),
            "smallest_positive_defined": np.ones((rows, protocol.COMPONENTS), dtype=np.bool_),
            "condition_defined": np.ones((rows, protocol.COMPONENTS), dtype=np.bool_),
            "weakly_responsive": np.zeros((rows, protocol.COMPONENTS), dtype=np.bool_),
            "saturation_counts": counts,
            "saturation_count_defined": count_defined,
            "saturation_fractions": np.zeros((rows, fields, 4), dtype=np.float64),
            "saturation_fraction_defined": fraction_defined,
            "histograms": histograms,
        }

    reduced = protocol._checkpoint_diagnostic_reductions(
        {
            "fit": fit,
            "checkpoint": checkpoint,
            "checkpoint_current": detail_table(0),
            "checkpoint_candidate": detail_table(1),
        }
    )
    assert len(reduced["checkpoint_cells"]) == 864
    assert len(reduced["seed_step_pools"]) == 96
    assert len(reduced["seed_arm_pools"]) == 12
    assert len(reduced["block_arm_pools"]) == 4
    mechanism_current = next(
        row
        for row in reduced["block_arm_pools"]
        if row["block"] == "appearance_only" and row["arm"] == protocol.ARMS[0]
    )
    assert mechanism_current["component_count"] == 32_400
    assert mechanism_current["rank_counts"] == [0, 0, 0, 32_400]
    assert mechanism_current["weak_count"] == 0
    assert mechanism_current["clamp"]["channel_count"] == 216 * 48 * 48 * 3
    assert [sum(histogram) for histogram in mechanism_current["histograms"]] == [
        216 * 450,
        216 * 150,
        216 * 450,
    ]


def test_frozen_decisions_recompute_all_threshold_boundaries() -> None:
    summary = {
        "appearance_only": {
            "delta_auc_by_seed": [0.05, 0.05, 0.20],
            "delta_final_psnr_by_seed": [-0.049, -0.049, -0.049],
            "delta_final_ssim_by_seed": [-0.0019, -0.0019, -0.0019],
            "null_global": {
                "defined": True,
                "null_energy_ratio": 0.01,
                "null_large_fraction": 0.10,
            },
            "null_by_seed": [
                {"defined": True, "null_energy_ratio": 0.005},
                {"defined": True, "null_energy_ratio": 0.005},
                {"defined": False, "null_energy_ratio": 0.0},
            ],
            "weak_fraction_delta_global": 0.05,
            "weak_fraction_delta_by_seed": [0.10, 0.0, -0.1],
        },
        "joint": {
            "delta_final_psnr_by_seed": [0.05, 0.05, 0.20],
            "delta_final_ssim_by_seed": [0.0, 0.0, 0.0],
            "delta_auc_by_seed": [0.10, 0.10, 0.10],
        },
    }
    decisions = protocol.frozen_decisions(summary, global_validity_passed=True)
    assert decisions == {
        "appearance_curve_improved": True,
        "null_update_material": True,
        "candidate_saturation_guard_passed": True,
        "fit_time_redundant_coordinate_interference_consistent": True,
        "joint_stage1_noninferior": True,
        "joint_stage1_material_improvement": True,
    }

    changed = json.loads(json.dumps(summary))
    changed["joint"]["delta_final_psnr_by_seed"][2] = -0.3000001
    changed_decisions = protocol.frozen_decisions(changed, global_validity_passed=True)
    assert changed_decisions["joint_stage1_noninferior"] is False
    assert changed_decisions["joint_stage1_material_improvement"] is False

    invalid = protocol.frozen_decisions(summary, global_validity_passed=False)
    assert not any(invalid.values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["joint"].__setitem__("delta_auc_by_seed", [0.1, 0.1]),
            "exactly three",
        ),
        (
            lambda value: value["appearance_only"].__setitem__(
                "delta_final_psnr_by_seed", [0.0, float("nan"), 0.0]
            ),
            "non-finite",
        ),
        (
            lambda value: value["appearance_only"]["null_by_seed"][0].__setitem__("defined", "yes"),
            "must be boolean",
        ),
        (
            lambda value: value["joint"].__setitem__("delta_auc_by_seed", [0.0, "0.0", 0.0]),
            "non-numeric",
        ),
        (
            lambda value: value["appearance_only"]["null_global"].__setitem__(
                "null_large_fraction", 1.01
            ),
            "must lie in",
        ),
        (
            lambda value: value["appearance_only"]["null_global"].update(
                {"defined": False, "null_energy_ratio": 0.01, "null_large_fraction": 0.0}
            ),
            "exact zero placeholders",
        ),
        (
            lambda value: value["appearance_only"]["null_by_seed"][1].update(
                {"defined": False, "null_energy_ratio": 0.005}
            ),
            "exact zero placeholder",
        ),
        (
            lambda value: value["appearance_only"].__setitem__(
                "weak_fraction_delta_by_seed", [0.0, -1.01, 0.0]
            ),
            r"must lie in \[-1,1\]",
        ),
    ],
)
def test_frozen_decisions_reject_malformed_or_nonfinite_summaries(mutation, message: str) -> None:
    summary = {
        "appearance_only": {
            "delta_auc_by_seed": [0.0, 0.0, 0.0],
            "delta_final_psnr_by_seed": [0.0, 0.0, 0.0],
            "delta_final_ssim_by_seed": [0.0, 0.0, 0.0],
            "null_global": {
                "defined": True,
                "null_energy_ratio": 0.0,
                "null_large_fraction": 0.0,
            },
            "null_by_seed": [
                {"defined": True, "null_energy_ratio": 0.0},
                {"defined": True, "null_energy_ratio": 0.0},
                {"defined": True, "null_energy_ratio": 0.0},
            ],
            "weak_fraction_delta_global": 0.0,
            "weak_fraction_delta_by_seed": [0.0, 0.0, 0.0],
        },
        "joint": {
            "delta_final_psnr_by_seed": [0.0, 0.0, 0.0],
            "delta_final_ssim_by_seed": [0.0, 0.0, 0.0],
            "delta_auc_by_seed": [0.0, 0.0, 0.0],
        },
    }
    mutation(summary)
    with pytest.raises(ValueError, match=message):
        protocol.frozen_decisions(summary, global_validity_passed=True)


def test_render_metrics_rejects_broadcast_nonfinite_and_out_of_range_targets() -> None:
    valid = torch.zeros(4, 5, 3, dtype=torch.float32)
    result = protocol.render_metrics(valid, valid)
    assert result["channel_count"] == 60
    with pytest.raises(protocol.ProtocolInvalid, match="identical nonempty"):
        protocol.render_metrics(valid, torch.zeros(1, 5, 3, dtype=torch.float32))
    nonfinite = valid.clone()
    nonfinite[0, 0, 0] = float("inf")
    with pytest.raises(protocol.ProtocolInvalid, match="must be finite") as default_phase:
        protocol.render_metrics(nonfinite, valid)
    assert default_phase.value.phase == "metrics"
    out_of_range = valid.clone()
    out_of_range[0, 0, 0] = 1.01
    with pytest.raises(protocol.ProtocolInvalid, match=r"target must lie in \[0,1\]"):
        protocol.render_metrics(valid, out_of_range)
    with pytest.raises(protocol.ProtocolInvalid, match="must be float32"):
        protocol.render_metrics(valid.to(torch.float64), valid.to(torch.float64))
    with pytest.raises(protocol.ProtocolInvalid, match="must be nonempty"):
        protocol.render_metrics(torch.empty(0, 5, 3), torch.empty(0, 5, 3))
    nonfinite_target = valid.clone()
    nonfinite_target[0, 0, 0] = float("nan")
    with pytest.raises(protocol.ProtocolInvalid, match="must be finite"):
        protocol.render_metrics(valid, nonfinite_target)

    assert protocol._phase_code("metrics") >= 0
    for phase in ("appearance_only", "joint", "reduction"):
        assert protocol._phase_code(phase) >= 0
        with pytest.raises(protocol.ProtocolInvalid, match="must be finite") as raised:
            protocol.render_metrics(nonfinite, valid, phase=phase)
        assert raised.value.phase == phase


def test_optimizer_environment_and_generated_path_contracts_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = protocol.optimizer_scheduler_contract()
    assert contract["optimizer_class"] == "torch.optim.Adam"
    assert contract["optimizer_defaults"] == {
        "lr": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }
    assert json.loads(json.dumps(contract)) == contract
    scheduler = contract["scheduler_state_at_construction"]
    assert scheduler["T_max"] == 120
    assert scheduler["eta_min"] == 0.001
    assert scheduler["base_lrs"] == [0.01]
    assert scheduler["last_epoch"] == 0

    environment = {
        "python_executable": protocol.EXPECTED_PYTHON_EXECUTABLE,
        "python_executable_resolved": protocol.EXPECTED_PYTHON_EXECUTABLE_RESOLVED,
        "python_prefix": protocol.EXPECTED_PYTHON_PREFIX,
        "torch": protocol.EXPECTED_TORCH_VERSION,
        "cuda_visible_devices": "",
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "omp_num_threads": "4",
        "mkl_num_threads": "4",
        "torch_num_threads": 4,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "optional_backend_modules_loaded": [],
    }
    protocol.assert_official_environment(environment)
    for key, invalid in (
        ("python_executable", "/usr/bin/python"),
        ("python_executable_resolved", "/usr/bin/python3"),
        ("python_prefix", "/usr"),
        ("torch", "wrong"),
        ("torch_cuda_available", True),
        ("torch_cuda_available", 0),
        ("torch_cuda_device_count", False),
        ("torch_num_threads", 4.0),
        ("deterministic_algorithms", 1),
        ("deterministic_warn_only", True),
        ("optional_backend_modules_loaded", ["gsplat"]),
    ):
        malformed = dict(environment)
        malformed[key] = invalid
        with pytest.raises(RuntimeError, match="official environment mismatch"):
            protocol.assert_official_environment(malformed)

    observed_environment = protocol.environment_metadata()
    assert observed_environment["python_executable"] == protocol.EXPECTED_PYTHON_EXECUTABLE
    assert (
        observed_environment["python_executable_resolved"]
        == protocol.EXPECTED_PYTHON_EXECUTABLE_RESOLVED
    )
    assert observed_environment["python_prefix"] == protocol.EXPECTED_PYTHON_PREFIX
    assert protocol.seal_command_record() == [
        protocol.EXPECTED_PYTHON_EXECUTABLE,
        str((protocol.ROOT / protocol.HARNESS).resolve()),
        "seal",
        "--output",
        str(protocol.DEFAULT_SEAL.resolve()),
    ]
    prospective_output = protocol.ROOT / (
        "benchmarks/results/20990101T010203Z_cpu_stage1_fit_parameterization.json"
    )
    assert protocol.run_command_record(prospective_output) == [
        protocol.EXPECTED_PYTHON_EXECUTABLE,
        str((protocol.ROOT / protocol.HARNESS).resolve()),
        "run",
        "--seal",
        str(protocol.DEFAULT_SEAL.resolve()),
        "--output",
        str(prospective_output.resolve()),
    ]

    generated_name = "20990101T010203Z_cpu_stage1_fit_parameterization_RESULT.md"
    assert protocol._is_generated_protocol_path(f"benchmarks/results/{generated_name}")
    assert not protocol._is_generated_protocol_path(f"scratch/{generated_name}")
    assert not protocol._is_generated_protocol_path(generated_name)

    monkeypatch.setitem(protocol.sys.modules, "structsplat.synthetic_test", object())
    assert (
        "structsplat.synthetic_test"
        in protocol.environment_metadata()["optional_backend_modules_loaded"]
    )


def test_seal_loader_rejects_a_noncanonical_recorded_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(
        json.dumps(
            {
                "artifact_type": protocol.SEAL_ARTIFACT_TYPE,
                "command": ["/usr/bin/python", "unbound-harness.py", "seal"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(protocol, "DEFAULT_SEAL", seal_path)
    with pytest.raises(RuntimeError, match="seal top-level schema differs"):
        protocol.load_and_verify_seal(seal_path)


def test_implementation_review_machine_binds_every_reviewed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        "prereg": Path("benchmarks/results/prereg.md"),
        "review": Path("benchmarks/results/review.md"),
        "harness": Path("benchmarks/harness.py"),
        "focused": Path("tests/focused.py"),
        "fit": Path("src/rtgs/image2gs/fit.py"),
        "seam": Path("tests/seam.py"),
        "project": Path("pyproject.toml"),
    }
    for index, path in enumerate(paths.values()):
        absolute = tmp_path / path
        absolute.parent.mkdir(parents=True, exist_ok=True)
        if path != paths["review"]:
            absolute.write_text(f"source-{index}\n", encoding="utf-8")
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    monkeypatch.setattr(protocol, "PREREGISTRATION", paths["prereg"])
    monkeypatch.setattr(protocol, "IMPLEMENTATION_REVIEW", paths["review"])
    monkeypatch.setattr(protocol, "HARNESS", paths["harness"])
    monkeypatch.setattr(protocol, "FOCUSED_TESTS", paths["focused"])
    monkeypatch.setattr(protocol, "FIT_SEAM", paths["fit"])
    monkeypatch.setattr(protocol, "SEAM_TESTS", paths["seam"])
    monkeypatch.setattr(protocol, "DEFAULT_SEAL", tmp_path / "seal.json")
    monkeypatch.setattr(protocol, "ATTEMPT", tmp_path / "attempt.json")
    frozen_map = {str(paths["fit"]): "frozen-fit"}
    monkeypatch.setattr(protocol, "FROZEN_SOURCE_HASHES", frozen_map)
    prereg_hash = protocol.sha256_file(tmp_path / paths["prereg"])
    monkeypatch.setattr(protocol, "PREREGISTRATION_SHA256", prereg_hash)
    reviewed_paths = tuple(path for key, path in paths.items() if key != "review")
    monkeypatch.setattr(
        protocol,
        "sealed_paths",
        lambda: (*reviewed_paths, paths["review"]),
    )
    snapshot = protocol.source_snapshot(reviewed_paths)
    review_text = "\n".join(
        (
            f"Preregistration-SHA256: {prereg_hash}",
            f"Frozen-Expected-Map-SHA256: {protocol.canonical_json_hash(frozen_map)}",
            f"Reviewed-Source-Collection-SHA256: {snapshot['collection_sha256']}",
            f"Harness-SHA256: {protocol.sha256_file(tmp_path / paths['harness'])}",
            f"Focused-Tests-SHA256: {protocol.sha256_file(tmp_path / paths['focused'])}",
            f"Fit-Seam-SHA256: {protocol.sha256_file(tmp_path / paths['fit'])}",
            f"Seam-Tests-SHA256: {protocol.sha256_file(tmp_path / paths['seam'])}",
            "Official-Seeds-Touched: none",
            "Official-Artifact-State: seal=absent; attempt=absent; result=absent",
            "Reviewer: independent-test-reviewer",
            "Reviewed-At-UTC: 2026-07-16T08:45:23Z",
            "Verdict: PASS",
            "",
        )
    )
    (tmp_path / paths["review"]).write_text(review_text, encoding="utf-8")
    receipt = protocol.verify_implementation_review()
    assert receipt["reviewer"] == "independent-test-reviewer"
    assert receipt["reviewed_source_collection_sha256"] == snapshot["collection_sha256"]

    protocol.DEFAULT_SEAL.write_text("sealed later\n", encoding="utf-8")
    protocol.verify_implementation_review()
    with pytest.raises(RuntimeError, match="absence binding"):
        protocol.verify_implementation_review(require_artifacts_absent=True)
    protocol.DEFAULT_SEAL.unlink()

    future_review = review_text.replace(
        "Reviewed-At-UTC: 2026-07-16T08:45:23Z",
        "Reviewed-At-UTC: 2999-01-01T00:00:00Z",
    )
    (tmp_path / paths["review"]).write_text(future_review, encoding="utf-8")
    with pytest.raises(RuntimeError, match="review chronology"):
        protocol.verify_implementation_review()
    (tmp_path / paths["review"]).write_text(review_text, encoding="utf-8")

    (tmp_path / paths["harness"]).write_text("post-review mutation\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Reviewed-Source-Collection"):
        protocol.verify_implementation_review()


def test_seal_loader_is_type_strict_for_snapshots_environment_and_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_path = tmp_path / "seal.json"
    monkeypatch.setattr(protocol, "DEFAULT_SEAL", seal_path)
    reviewed_at = "2026-07-16T08:45:23Z"
    created_at = "2026-07-16T09:00:00+00:00"
    now = datetime(2026, 7, 16, 10, 0, 0, tzinfo=timezone.utc)
    snapshot = {
        "preregistration": {"sha256": "0" * 64},
        "implementation_review": {
            "sha256": "2" * 64,
            "reviewed_at_utc": reviewed_at,
        },
        "preimplementation_sources": {"passed": True},
        "sealed_sources": {"collection_sha256": "1" * 64},
        "loaded_repository_sources": {"collection_sha256": "3" * 64},
        "git": {"revision": "test"},
        "fit_configs": {},
        "optimizer_scheduler_contract": {},
        "raw_schema": {},
        "command_templates": {},
    }
    environment = protocol.environment_metadata()
    monkeypatch.setattr(protocol, "seal_snapshot", lambda: snapshot)
    monkeypatch.setattr(protocol, "environment_metadata", lambda: environment)
    monkeypatch.setattr(protocol, "assert_official_environment", lambda value: None)
    monkeypatch.setattr(protocol, "_utc_now", lambda: now)
    monkeypatch.setattr(protocol, "verification_commands", lambda: (("verify",),))
    monkeypatch.setattr(protocol, "VERIFICATION_LITERAL_COMMANDS", ("verify",))
    baseline = {
        "artifact_type": protocol.SEAL_ARTIFACT_TYPE,
        "created_at_utc": created_at,
        "command": protocol.seal_command_record(),
        **snapshot,
        "verification_snapshot_sha256": protocol.canonical_json_hash(snapshot),
        "verification": {
            "passed": True,
            "commands": [
                {
                    "command": ["verify"],
                    "literal_command": "verify",
                    "returncode": 0,
                    "seconds": 0.25,
                    "stdout": "",
                    "stderr": "",
                    "stdout_sha256": protocol.sha256_bytes(b""),
                    "stderr_sha256": protocol.sha256_bytes(b""),
                }
            ],
        },
        "environment": environment,
    }

    def assert_rejected(mutation, message: str) -> None:
        payload = json.loads(json.dumps(baseline))
        mutation(payload)
        seal_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match=message):
            protocol.load_and_verify_seal(seal_path)

    seal_path.write_text(json.dumps(baseline), encoding="utf-8")
    protocol.load_and_verify_seal(seal_path)
    assert_rejected(lambda value: value.__setitem__("extra", 0), "top-level schema")
    assert_rejected(
        lambda value: value["environment"].__setitem__("torch_cuda_available", 0),
        "runtime environment differs",
    )
    assert_rejected(
        lambda value: value["verification"].__setitem__("passed", 1),
        "exact passing verification sequence",
    )
    assert_rejected(
        lambda value: value["verification"]["commands"][0].__setitem__("returncode", False),
        "malformed verification return code",
    )
    assert_rejected(
        lambda value: value["verification"]["commands"][0].__setitem__("seconds", -1),
        "malformed verification duration",
    )
    assert_rejected(
        lambda value: value["verification"]["commands"][0].__setitem__("stdout", 0),
        "malformed verification output",
    )


def test_source_snapshot_and_attempt_binding_are_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"first\n")
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    first = protocol.source_snapshot((Path("source.py"),))
    source.write_bytes(b"second\n")
    second = protocol.source_snapshot((Path("source.py"),))
    assert first["sha256"] != second["sha256"]
    assert first["collection_sha256"] != second["collection_sha256"]

    marker_path = tmp_path / "ATTEMPT.json"
    monkeypatch.setattr(protocol, "ATTEMPT", marker_path)
    seal_path = tmp_path / "seal.json"
    now = datetime(2099, 1, 1, 1, 2, 3, tzinfo=timezone.utc)
    result_path = tmp_path / "20990101T010203Z_cpu_stage1_fit_parameterization.json"
    monkeypatch.setattr(protocol, "DEFAULT_SEAL", seal_path)
    monkeypatch.setattr(protocol, "IMPLEMENTATION_COMPLETE", True)
    monkeypatch.setattr(protocol, "IMPLEMENTATION_GAPS", ())
    monkeypatch.setattr(protocol, "_utc_now", lambda: now)
    monkeypatch.setattr(
        protocol,
        "_CLI_AUTHORIZATION",
        ("run", seal_path.resolve(), result_path.resolve()),
    )
    marker = protocol.claim_attempt(
        marker_path,
        {"valid_json": result_path},
        {
            "sha256": "1" * 64,
            "source_collection_sha256": "2" * 64,
            "payload": {"created_at_utc": "2099-01-01T01:00:00+00:00"},
        },
    )
    assert marker["payload"]["command"] == protocol.run_command_record(result_path)
    protocol.validate_attempt_binding(marker)
    monkeypatch.setattr(protocol, "environment_metadata", lambda: {"drifted": True})
    protocol.validate_attempt_binding(marker)
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["resume_permitted"] = True
    marker_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="attempt marker changed"):
        protocol.validate_attempt_binding(marker)


def test_output_routing_derives_disjoint_valid_and_invalid_triples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        protocol,
        "_utc_now",
        lambda: datetime(2099, 1, 1, 1, 2, 3, tzinfo=timezone.utc),
    )
    output = (
        protocol.ROOT / "benchmarks/results/20990101T010203Z_cpu_stage1_fit_parameterization.json"
    )
    paths = protocol.official_output_paths(output)
    assert len(set(paths.values())) == 6
    assert paths["valid_raw"].name.endswith("_RAW.npz")
    assert paths["valid_note"].name.endswith("_RESULT.md")
    assert paths["invalid_json"].name.endswith("_invalid.json")
    assert paths["invalid_raw"].name.endswith("_invalid_RAW.npz")
    assert paths["invalid_note"].name.endswith("_invalid_RESULT.md")
    with pytest.raises(ValueError, match="official output"):
        protocol.official_output_paths(output.with_name("result.json"))
    with pytest.raises(ValueError, match="not fresh"):
        protocol.official_output_paths(
            output.with_name("20981231T230000Z_cpu_stage1_fit_parameterization.json")
        )
    with pytest.raises(ValueError, match="not fresh"):
        protocol.official_output_paths(
            output.with_name("20990101T020000Z_cpu_stage1_fit_parameterization.json")
        )
    with pytest.raises(ValueError, match="real UTC instant"):
        protocol.official_output_paths(
            output.with_name("20991340T010203Z_cpu_stage1_fit_parameterization.json")
        )


def test_preflight_and_exclusive_writer_reject_broken_symlink(
    tmp_path: Path,
) -> None:
    redirected = tmp_path / "redirected.json"
    fixed = tmp_path / "fixed.json"
    fixed.symlink_to(redirected)
    with pytest.raises(FileExistsError, match="prospective paths"):
        protocol.preflight_absent((fixed,))
    with pytest.raises(FileExistsError):
        protocol.exclusive_write_bytes(fixed, b"sealed\n")
    assert not redirected.exists()


def test_ready_mutation_helpers_still_require_actual_process_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(protocol, "IMPLEMENTATION_COMPLETE", True)
    monkeypatch.setattr(protocol, "IMPLEMENTATION_GAPS", ())
    monkeypatch.setattr(protocol, "_CLI_AUTHORIZATION", None)
    with pytest.raises(RuntimeError, match="exact preregistered process CLI"):
        protocol.create_seal()
    with pytest.raises(RuntimeError, match="exact preregistered process CLI"):
        protocol.run_bound_scientific_experiment(tmp_path / "seal.json", tmp_path / "result.json")
    with pytest.raises(RuntimeError, match="exact preregistered process CLI"):
        protocol.claim_attempt(
            tmp_path / "attempt.json",
            {"valid_json": tmp_path / "result.json"},
            {"sha256": "1" * 64, "source_collection_sha256": "2" * 64},
        )
    assert not any(tmp_path.iterdir())


def test_main_run_dispatches_only_after_process_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seal = tmp_path / "seal.json"
    output = tmp_path / "20990101T010203Z_cpu_stage1_fit_parameterization.json"
    authorization = ("run", seal.absolute(), output.absolute())
    monkeypatch.setattr(
        protocol,
        "_authorize_actual_process_cli",
        lambda _args, _argv: authorization,
    )
    monkeypatch.setattr(protocol.torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(protocol.torch, "use_deterministic_algorithms", lambda _value: None)
    observed: dict[str, Path] = {}

    def run(seal_path: Path, output_path: Path) -> dict[str, str]:
        assert authorization == protocol._CLI_AUTHORIZATION
        observed.update(seal=seal_path, output=output_path)
        return {
            "json_path": str(output_path),
            "json_sha256": "1" * 64,
        }

    monkeypatch.setattr(protocol, "run_bound_scientific_experiment", run)
    assert protocol.main(["run", "--seal", str(seal), "--output", str(output)]) == 0
    assert observed == {"seal": seal.absolute(), "output": output.absolute()}
    assert protocol._CLI_AUTHORIZATION is None
    assert "saved" in capsys.readouterr().out


def _invalid_prefix_fixture(*, phase: str, prerequisite_rows: int = 0) -> dict[str, np.ndarray]:
    archive = protocol.RawArchive()
    evidence = protocol.ScientificEvidence(archive)
    expected_rows = protocol._expected_prerequisite_rows()
    table = evidence.tables["prerequisite"]
    for row_index in range(prerequisite_rows):
        identity = expected_rows[row_index]
        values = {
            name: np.zeros(field.shape, dtype=np.dtype(field.dtype))
            for name, field in table.spec.fields.items()
        }
        for name, value in zip(
            (
                "block_index",
                "seed_position",
                "seed",
                "local_view",
                "original_view",
                "initializer_seed",
            ),
            identity,
            strict=True,
        ):
            values[name] = value
        table.append(**values)
    evidence.completed_scenes = int(np.ceil(prerequisite_rows / len(protocol.SELECTED_VIEWS)))
    evidence.completed_initializers = prerequisite_rows
    archive.completed_initializers = prerequisite_rows
    archive.completed_paired_views = prerequisite_rows
    archive.phase = phase
    evidence.flush(require_complete=False, include_completion=False)
    evidence.completion_arrays(phase=phase)
    archive.completion_arrays(phase=phase)
    return dict(archive.arrays)


def test_invalid_prefix_requires_fixed_evidence_phase_and_exact_names() -> None:
    with pytest.raises(protocol.ProtocolInvalid, match="lacks fixed evidence"):
        protocol.validate_invalid_scientific_prefix({}, expected_phase="preflight")

    arrays = _invalid_prefix_fixture(phase="preflight")
    validated = protocol.validate_invalid_scientific_prefix(arrays, expected_phase="preflight")
    assert validated["phase"] == "preflight"
    assert not any(validated["table_row_counts"].values())
    assert validated["scientific_decisions_present"] is False

    missing = dict(arrays)
    missing.pop("identity/checkpoints")
    with pytest.raises(protocol.ProtocolInvalid, match="lacks fixed evidence"):
        protocol.validate_invalid_scientific_prefix(missing, expected_phase="preflight")

    wrong_phase = dict(arrays)
    wrong_phase["completion/phase_code"] = np.asarray(protocol._phase_code("scene"), dtype=np.int64)
    with pytest.raises(protocol.ProtocolInvalid, match="completion phase"):
        protocol.validate_invalid_scientific_prefix(wrong_phase, expected_phase="preflight")

    extra = dict(arrays)
    extra["arbitrary/value"] = np.asarray(1, dtype=np.int64)
    with pytest.raises(protocol.ProtocolInvalid, match="unexpected invalid raw name"):
        protocol.validate_invalid_scientific_prefix(extra, expected_phase="preflight")

    fake_nonfinite_failure = dict(arrays)
    fake_nonfinite_failure["failure/raw_parameter/color_raw"] = np.zeros(
        (protocol.COMPONENTS, 3), dtype=np.float32
    )
    with pytest.raises(protocol.ProtocolInvalid, match="unexpected invalid raw name"):
        protocol.validate_invalid_scientific_prefix(
            fake_nonfinite_failure, expected_phase="preflight"
        )

    with pytest.raises(protocol.ProtocolInvalid, match="not reachable"):
        protocol.validate_invalid_scientific_prefix(arrays, expected_phase="complete")


def test_finite_initialization_failure_preserves_reached_target_and_initializer() -> None:
    archive = protocol.RawArchive()
    evidence = protocol.ScientificEvidence(archive)
    archive.phase = "initialization"
    evidence.completed_scenes = 1
    bad_target = torch.zeros(protocol.IMAGE_SIZE - 1, protocol.IMAGE_SIZE, 3, dtype=torch.float32)
    g0 = protocol.Gaussians2D(
        xy=torch.zeros(protocol.COMPONENTS, 2, dtype=torch.float32),
        chol=torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32).repeat(protocol.COMPONENTS, 1),
        color=torch.full((protocol.COMPONENTS, 3), 0.5, dtype=torch.float32),
        weight=torch.full((protocol.COMPONENTS,), 0.5, dtype=torch.float32),
    )
    with pytest.raises(protocol.ProtocolInvalid) as caught:
        protocol._validate_target_and_initializer(bad_target, g0)
    failure = protocol._scientific_failure_record(caught.value, archive)
    assert failure["phase"] == "initialization"
    evidence.flush(require_complete=False, include_completion=False)
    evidence.completion_arrays(phase="initialization")
    archive.completion_arrays(phase="initialization")
    assert archive.arrays["failure/protocol/target"].shape == bad_target.shape
    for name in ("xy", "chol", "color", "weight"):
        assert f"failure/protocol/g0_{name}" in archive.arrays
    validated = protocol.validate_invalid_scientific_prefix(
        archive.arrays,
        expected_phase="initialization",
        expected_failure=failure,
    )
    assert validated["phase"] == "initialization"
    assert validated["scientific_decisions_present"] is False

    incomplete = dict(archive.arrays)
    incomplete.pop("failure/protocol/g0_color")
    with pytest.raises(protocol.ProtocolInvalid, match="incomplete|receipt binding"):
        protocol.validate_invalid_scientific_prefix(
            incomplete,
            expected_phase="initialization",
            expected_failure=failure,
        )

    fabricated = {
        name: value for name, value in archive.arrays.items() if not name.startswith("failure/")
    }
    fabricated["failure/protocol/snapshot_fabricated"] = np.asarray([1], dtype=np.int64)
    with pytest.raises(protocol.ProtocolInvalid, match="unexpected invalid raw name"):
        protocol.validate_invalid_scientific_prefix(
            fabricated,
            expected_phase="initialization",
            expected_failure=failure,
        )


def test_nonfinite_table_row_failure_receipt_includes_preexisting_boundary_array() -> None:
    archive = protocol.RawArchive()
    evidence = protocol.ScientificEvidence(archive)
    archive.phase = "equivalence"
    evidence.completed_scenes = 1
    table = evidence.tables["prerequisite"]
    values = {
        name: np.zeros(field.shape, dtype=np.dtype(field.dtype))
        for name, field in table.spec.fields.items()
    }
    identity = protocol._expected_prerequisite_rows()[0]
    for name, value in zip(
        (
            "block_index",
            "seed_position",
            "seed",
            "local_view",
            "original_view",
            "initializer_seed",
        ),
        identity,
        strict=True,
    ):
        values[name] = value
    values["current_loss"] = np.asarray(np.nan, dtype=np.float32)
    with pytest.raises(protocol.ProtocolInvalid) as caught:
        table.append(**values)
    failure = protocol._scientific_failure_record(caught.value, archive)
    base = "failure/prerequisite/current_loss/000000"
    assert failure["raw_evidence"]["names"] == [
        base,
        f"{base}/nonfinite_classification",
    ]
    evidence.flush(require_complete=False, include_completion=False)
    evidence.completion_arrays(phase="equivalence")
    archive.completion_arrays(phase="equivalence")
    validated = protocol.validate_invalid_scientific_prefix(
        archive.arrays,
        expected_phase="equivalence",
        expected_failure=failure,
    )
    assert validated["phase"] == "equivalence"


def test_invalid_prerequisite_prefix_is_canonical_and_counter_bound() -> None:
    arrays = _invalid_prefix_fixture(phase="equivalence", prerequisite_rows=2)
    validated = protocol.validate_invalid_scientific_prefix(arrays, expected_phase="equivalence")
    assert validated["table_row_counts"]["prerequisite"] == 2
    assert validated["scientific_completion"]["initializers"] == 2

    reordered = {name: value.copy() for name, value in arrays.items()}
    reordered["scientific/prerequisite/local_view"][[0, 1]] = reordered[
        "scientific/prerequisite/local_view"
    ][[1, 0]]
    with pytest.raises(protocol.ProtocolInvalid, match="canonical prefix"):
        protocol.validate_invalid_scientific_prefix(reordered, expected_phase="equivalence")

    wrong_counter = {name: value.copy() for name, value in arrays.items()}
    names = protocol._decode_utf8_rows(
        wrong_counter["scientific_completion/count_name_bytes"],
        wrong_counter["scientific_completion/count_name_lengths"],
    )
    position = names.index("initializers")
    wrong_counter["scientific_completion/count_values"][position] += 1
    with pytest.raises(protocol.ProtocolInvalid, match="completion count"):
        protocol.validate_invalid_scientific_prefix(wrong_counter, expected_phase="equivalence")


def test_invalid_prefix_rejects_explicit_zero_row_table_and_nonfinite_scientific_field() -> None:
    arrays = _invalid_prefix_fixture(phase="preflight")
    arrays["scientific/fit/row_count"] = np.asarray(0, dtype=np.int64)
    with pytest.raises(protocol.ProtocolInvalid, match="row count differs"):
        protocol.validate_invalid_scientific_prefix(arrays, expected_phase="preflight")

    nonfinite = _invalid_prefix_fixture(phase="equivalence", prerequisite_rows=1)
    nonfinite["scientific/prerequisite/current_loss"][0] = np.nan
    nonfinite["scientific/prerequisite/current_loss/nonfinite_classification"] = np.asarray(
        [1], dtype=np.uint8
    )
    with pytest.raises(protocol.ProtocolInvalid, match="field set|non-finite"):
        protocol.validate_invalid_scientific_prefix(nonfinite, expected_phase="equivalence")


def test_invalid_prefix_rejects_future_arm_detail_at_reached_checkpoint() -> None:
    arrays = _invalid_prefix_fixture(
        phase="appearance_only", prerequisite_rows=protocol.SCIENTIFIC_PLAN.initializers
    )

    def insert_row(table_name: str, identity: dict[str, int | bool]) -> None:
        spec = protocol.SCIENTIFIC_TABLE_SPECS[table_name]
        prefix = f"scientific/{table_name}"
        arrays[f"{prefix}/row_count"] = np.asarray(1, dtype=np.int64)
        for field_name, field_spec in spec.fields.items():
            value = np.zeros((1, *field_spec.shape), dtype=np.dtype(field_spec.dtype))
            if field_name in identity:
                value[0] = identity[field_name]
            arrays[f"{prefix}/{field_name}"] = value

    insert_row(
        "fit_event",
        {"fit_index": 0, "event_code": protocol.EVENT_CODES["initial"], "step": 0},
    )
    insert_row("checkpoint", {"checkpoint_index": 0, "fit_index": 0, "step": 0})
    future_checkpoint, future_fit, future_step = protocol._expected_checkpoint_arm_rows(1)[0]
    insert_row(
        "checkpoint_candidate",
        {
            "checkpoint_index": future_checkpoint,
            "fit_index": future_fit,
            "step": future_step,
        },
    )
    count_names = protocol._decode_utf8_rows(
        arrays["scientific_completion/count_name_bytes"],
        arrays["scientific_completion/count_name_lengths"],
    )
    for name, value in {"callback_events": 1, "checkpoints": 1}.items():
        arrays["scientific_completion/count_values"][count_names.index(name)] = value
    for name, value in {
        "completed_callback_events": 1,
        "completed_checkpoints": 1,
        "completed_checkpoint_component_rows": protocol.COMPONENTS,
    }.items():
        arrays[f"completion/{name}"] = np.asarray(value, dtype=np.int64)

    with pytest.raises(protocol.ProtocolInvalid, match="detail union"):
        protocol.validate_invalid_scientific_prefix(arrays, expected_phase="appearance_only")


def test_joint_trajectory_replay_rejects_unanchored_step5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    target = torch.rand(
        (protocol.IMAGE_SIZE, protocol.IMAGE_SIZE, 3),
        generator=torch.Generator(device="cpu").manual_seed(424_242),
    )
    g0 = protocol.init_gaussians_2d(
        target,
        n=protocol.COMPONENTS,
        grad_mix=0.7,
        generator=torch.Generator(device="cpu").manual_seed(314_159),
    )
    (
        current_raw,
        candidate_raw,
        current,
        candidate,
        current_render,
        candidate_render,
        current_loss,
        candidate_loss,
        _gate_values,
    ) = protocol._prepare_step_zero(target, g0)
    cell = protocol.PreparedFitCell(
        prerequisite_index=0,
        block_index=1,
        seed_position=0,
        seed=424_242,
        local_view=0,
        original_view=0,
        target=target,
        g0=g0,
        current_raw=current_raw,
        candidate_raw=candidate_raw,
        current_built=current,
        candidate_built=candidate,
        current_render=current_render,
        candidate_render=candidate_render,
        current_loss=current_loss,
        candidate_loss=candidate_loss,
    )
    evidence = protocol.ScientificEvidence(protocol.RawArchive())
    evidence.archive.phase = "joint"
    recorder = protocol.NativeEvidenceRecorder(
        evidence,
        cell,
        fit_index=1,
        arm=protocol.ARMS[1],
        geometry_frozen=False,
    )
    config = protocol.frozen_fit_config(protocol.ARMS[1])
    config.freeze_geometry = False
    result, history = protocol.fit_image_from_initialization(
        target,
        g0,
        config,
        mask=None,
        diagnostic_callback=recorder,
        diagnostic_steps=protocol.POSITIVE_CHECKPOINTS,
    )
    recorder.finish()

    def stacked(table_name: str) -> dict[str, np.ndarray]:
        table = evidence.tables[table_name]
        return {
            name: np.stack([row[name] for row in table.rows], axis=0) for name in table.spec.fields
        }

    def mechanism_placeholder_and_joint(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        return np.stack([np.zeros_like(value), value], axis=0)

    fit = {
        "block_index": np.asarray([0, 1], dtype=np.int64),
        "arm_code": np.asarray([0, 1], dtype=np.int64),
        "seed_position": np.asarray([0, 0], dtype=np.int64),
        "local_view": np.asarray([0, 0], dtype=np.int64),
        "result_xy": mechanism_placeholder_and_joint(result.xy.detach().numpy()),
        "result_chol": mechanism_placeholder_and_joint(result.chol.detach().numpy()),
        "result_color": mechanism_placeholder_and_joint(result.color.detach().numpy()),
        "result_weight": mechanism_placeholder_and_joint(result.weight.detach().numpy()),
        "final_psnr_full": mechanism_placeholder_and_joint(
            np.asarray(history["final_psnr_full"], dtype=np.float32)
        ),
        "final_psnr": mechanism_placeholder_and_joint(
            np.asarray(history["final_psnr"], dtype=np.float32)
        ),
    }
    empty_identity = {
        "fit_index": np.empty(0, dtype=np.int64),
        "step": np.empty(0, dtype=np.int64),
    }
    tables = {
        "fit": fit,
        "prerequisite": {
            "block_index": np.asarray([1], dtype=np.int64),
            "seed_position": np.asarray([0], dtype=np.int64),
            "local_view": np.asarray([0], dtype=np.int64),
            "target": target.detach().numpy()[None],
            "g0_xy": g0.xy.detach().numpy()[None],
            "g0_chol": g0.chol.detach().numpy()[None],
            "g0_color": g0.color.detach().numpy()[None],
            "g0_weight": g0.weight.detach().numpy()[None],
        },
        "checkpoint": stacked("checkpoint"),
        "checkpoint_current": dict(empty_identity),
        "checkpoint_candidate": stacked("checkpoint_candidate"),
        "joint_current": dict(empty_identity),
        "joint_candidate": stacked("joint_candidate"),
    }
    monkeypatch.setattr(
        protocol,
        "SCIENTIFIC_PLAN",
        replace(protocol.SCIENTIFIC_PLAN, fits=2, initializers=1),
    )

    protocol._replay_joint_trajectories(tables)
    transition = tables["joint_candidate"]
    step5 = int(np.flatnonzero(transition["step"] == 5)[0])
    transition["pre_xy_raw"][step5, 0, 0] += np.float32(0.25)
    with pytest.raises(
        protocol.ProtocolInvalid,
        match="joint_replay/transition/pre_xy_raw",
    ):
        protocol._replay_joint_trajectories(tables)
    torch.set_num_threads(previous_threads)
    torch.use_deterministic_algorithms(previous_deterministic)


def _source_replay_fixture(
    seed: int, images: list[torch.Tensor]
) -> tuple[dict[str, np.ndarray], list[tuple[int, int, int, int, int, int]]]:
    expected = [
        (0, 0, seed, local_view, original_view, seed + local_view)
        for local_view, original_view in enumerate(protocol.SELECTED_VIEWS)
    ]
    targets = np.stack(
        [
            images[original_view].detach().contiguous().cpu().numpy()
            for original_view in protocol.SELECTED_VIEWS
        ]
    ).astype("<f4", copy=False)
    return (
        {
            "block_index": np.asarray([row[0] for row in expected], dtype=np.int64),
            "seed_position": np.asarray([row[1] for row in expected], dtype=np.int64),
            "seed": np.asarray([row[2] for row in expected], dtype=np.int64),
            "local_view": np.asarray([row[3] for row in expected], dtype=np.int64),
            "original_view": np.asarray([row[4] for row in expected], dtype=np.int64),
            "initializer_seed": np.asarray([row[5] for row in expected], dtype=np.int64),
            "target": targets,
            "target_sha256": np.stack(
                [
                    protocol._sha256_bytes(protocol.array_content_sha256(target))
                    for target in targets
                ]
            ),
        },
        expected,
    )


def test_reviewer_source_replay_uses_one_scene_and_selected_views_only() -> None:
    seed = 424_242
    images = [
        torch.full(
            (protocol.IMAGE_SIZE, protocol.IMAGE_SIZE, 3),
            view / 12.0,
            dtype=torch.float32,
        )
        for view in range(12)
    ]
    prerequisite, expected = _source_replay_fixture(seed, images)
    accesses: list[int] = []
    calls: list[dict[str, int]] = []

    class LoggingImages:
        def __len__(self) -> int:
            return len(images)

        def __getitem__(self, index: int) -> torch.Tensor:
            accesses.append(index)
            return images[index]

    class Scene:
        def __init__(self) -> None:
            self.images = LoggingImages()

    def factory(**kwargs):
        calls.append(kwargs)
        return Scene()

    receipt = protocol._replay_source_target_rows(prerequisite, expected, scene_factory=factory)
    assert receipt["passed"] is True
    assert receipt["scene_call_count"] == 1
    assert receipt["target_count"] == len(protocol.SELECTED_VIEWS)
    assert len(calls) == 1 and calls[0]["seed"] == seed
    assert accesses == list(protocol.SELECTED_VIEWS)

    identity_mutation = {name: value.copy() for name, value in prerequisite.items()}
    identity_mutation["original_view"][0] = 11
    calls.clear()
    with pytest.raises(protocol.ProtocolInvalid, match="identity rows"):
        protocol._replay_source_target_rows(identity_mutation, expected, scene_factory=factory)
    assert calls == []

    self_consistent_target_mutation = {name: value.copy() for name, value in prerequisite.items()}
    self_consistent_target_mutation["target"][0, 0, 0, 0] += np.float32(0.01)
    self_consistent_target_mutation["target_sha256"][0] = protocol._sha256_bytes(
        protocol.array_content_sha256(self_consistent_target_mutation["target"][0])
    )
    with pytest.raises(protocol.ProtocolInvalid, match="replayed source target differs"):
        protocol._replay_source_target_rows(
            self_consistent_target_mutation, expected, scene_factory=factory
        )


def test_reviewer_source_replay_matches_real_nonofficial_scene() -> None:
    seed = 314_159
    scene = protocol.make_synthetic_scene(
        n_gaussians=40,
        n_cameras=12,
        image_size=protocol.IMAGE_SIZE,
        seed=seed,
    )
    prerequisite, expected = _source_replay_fixture(seed, scene.images)
    del scene
    receipt = protocol._replay_source_target_rows(
        prerequisite, expected, scene_factory=protocol.make_synthetic_scene
    )
    assert receipt["passed"] is True
    assert receipt["scene_call_count"] == 1
    assert receipt["target_count"] == 9
    assert receipt["torch_rng_unchanged"] is True


def test_public_reviewer_source_replay_binds_raw_sources_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = {"review/value": np.asarray([1, 2, 3], dtype=np.int64)}
    _manifest, collection = protocol.array_manifest(arrays)
    monkeypatch.setattr(protocol, "_scientific_required_names", lambda: set(arrays))
    monkeypatch.setattr(protocol, "_validate_scientific_constants", lambda _arrays: None)
    monkeypatch.setattr(protocol, "_validate_scientific_completion", lambda _arrays: None)
    monkeypatch.setattr(protocol, "_validate_archive_completion", lambda _arrays: None)
    monkeypatch.setattr(
        protocol,
        "_raw_table",
        lambda _arrays, name, require_complete: {"table": name},
    )
    monkeypatch.setattr(protocol, "_validate_scientific_order", lambda _tables: None)
    environment = {"stable": True}
    monkeypatch.setattr(protocol, "environment_metadata", lambda: dict(environment))
    monkeypatch.setattr(protocol, "assert_official_environment", lambda _value: None)
    monkeypatch.setattr(protocol, "official_environment_fingerprint", lambda value: value)
    source = {"collection_sha256": "a" * 64, "sha256": {"source.py": "b" * 64}}
    monkeypatch.setattr(protocol, "source_snapshot", lambda _paths: dict(source))
    monkeypatch.setattr(
        protocol,
        "_replay_source_target_rows",
        lambda *_args, **_kwargs: {
            "passed": True,
            "reviewer_only": True,
            "scene_call_count": 6,
            "target_count": 54,
        },
    )
    receipt = protocol.reviewer_replay_source_targets(
        arrays, expected_raw_collection_sha256=collection
    )
    assert receipt["raw_collection_sha256"] == collection
    assert receipt["target_generator_source_collection_sha256"] == "a" * 64
    assert receipt["environment_fingerprint_sha256"] == protocol.canonical_json_hash(environment)
    with pytest.raises(protocol.ProtocolInvalid, match="raw binding differs"):
        protocol.reviewer_replay_source_targets(arrays, expected_raw_collection_sha256="0" * 64)


def test_global_prerequisite_builds_each_nonofficial_scene_once_before_unlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development_seeds = {
        "appearance_only": (424_242, 314_159, 271_828),
        "joint": (161_803, 8_675_309, 123_456),
    }
    monkeypatch.setattr(protocol, "BLOCK_SEEDS", development_seeds)
    archive = protocol.RawArchive()
    evidence = protocol.ScientificEvidence(archive)
    calls: list[int] = []
    accesses: dict[int, list[int]] = {}

    class LoggingImages:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.values = [
                torch.full(
                    (protocol.IMAGE_SIZE, protocol.IMAGE_SIZE, 3),
                    ((seed % 17) + view) / 32.0,
                    dtype=torch.float32,
                )
                for view in range(12)
            ]

        def __len__(self) -> int:
            return 12

        def __getitem__(self, index: int) -> torch.Tensor:
            assert evidence.optimization_unlocked is False
            accesses[self.seed].append(index)
            return self.values[index]

    class Scene:
        def __init__(self, seed: int) -> None:
            self.images = LoggingImages(seed)

    def scene_factory(*, seed: int, **_kwargs):
        assert evidence.optimization_unlocked is False
        calls.append(seed)
        accesses[seed] = []
        return Scene(seed)

    xy = torch.zeros(protocol.COMPONENTS, 2, dtype=torch.float32)
    chol = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32).repeat(protocol.COMPONENTS, 1)
    color = torch.full((protocol.COMPONENTS, 3), 0.5, dtype=torch.float32)
    weight = torch.full((protocol.COMPONENTS,), 0.5, dtype=torch.float32)
    g0 = protocol.Gaussians2D(xy=xy, chol=chol, color=color, weight=weight)

    def initializer(*_args, **_kwargs):
        assert evidence.optimization_unlocked is False
        return g0

    def prepare(target: torch.Tensor, initial: protocol.Gaussians2D):
        assert evidence.optimization_unlocked is False
        common_geometry = {
            "xy_raw": xy.clone(),
            "diag_raw": torch.zeros(protocol.COMPONENTS, 2),
            "off_raw": torch.zeros(protocol.COMPONENTS),
        }
        current_raw = {
            **common_geometry,
            "u": torch.zeros(protocol.COMPONENTS, 3),
            "s": torch.zeros(protocol.COMPONENTS),
        }
        candidate_raw = {
            **{name: value.clone() for name, value in common_geometry.items()},
            "r": torch.zeros(protocol.COMPONENTS, 3),
        }
        candidate = protocol.Gaussians2D(
            xy=initial.xy,
            chol=initial.chol,
            color=initial.weight[:, None] * initial.color,
            weight=torch.ones_like(initial.weight),
        )
        render = torch.zeros_like(target)
        loss = torch.asarray(1.0, dtype=torch.float32)
        return (
            current_raw,
            candidate_raw,
            initial,
            candidate,
            render,
            render.clone(),
            loss,
            loss.clone(),
            np.zeros(10, dtype=np.float64),
        )

    monkeypatch.setattr(protocol, "make_synthetic_scene", scene_factory)
    monkeypatch.setattr(protocol, "init_gaussians_2d", initializer)
    monkeypatch.setattr(protocol, "_validate_target_and_initializer", lambda *_args: None)
    monkeypatch.setattr(protocol, "_prepare_step_zero", prepare)
    protocol.build_global_prerequisite(evidence)

    expected_calls = [seed for block in protocol.BLOCKS for seed in development_seeds[block]]
    assert calls == expected_calls
    assert all(accesses[seed] == list(protocol.SELECTED_VIEWS) for seed in expected_calls)
    assert evidence.completed_scenes == 6
    assert evidence.completed_initializers == 54
    assert len(evidence.tables["prerequisite"].rows) == 54
    assert len(evidence.prepared) == 54
    for row in evidence.tables["prerequisite"].rows:
        seed = int(row["seed"])
        original_view = int(row["original_view"])
        expected_target = torch.full(
            (protocol.IMAGE_SIZE, protocol.IMAGE_SIZE, 3),
            ((seed % 17) + original_view) / 32.0,
            dtype=torch.float32,
        )
        assert torch.equal(torch.from_numpy(row["target"]), expected_target)
        key = (
            int(row["block_index"]),
            int(row["seed_position"]),
            int(row["local_view"]),
        )
        assert torch.equal(evidence.prepared[key].target, expected_target)
    assert evidence.optimization_unlocked is True
