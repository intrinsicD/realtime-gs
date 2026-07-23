"""Toy-only contract tests for the preregistered Stage-1 semantic factorial."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rtgs.core.gaussians2d import Gaussians2D

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "stage1_semantic_factorial.py"
_SPEC = importlib.util.spec_from_file_location("stage1_semantic_factorial", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def _toy_gaussians2d() -> Gaussians2D:
    return Gaussians2D(
        xy=torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.0, 1.0]]),
        chol=torch.tensor([[1.0, 0.0, 1.0], [1.1, 0.1, 0.9], [0.8, -0.1, 1.2], [1.0, 0.0, 1.0]]),
        color=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.4, 0.8], [0.0, 0.5, 1.0], [1.0, 0.5, 0.25]]),
        weight=torch.tensor([0.0, 0.5, 0.2, 1.0]),
    )


def _independent_raw_digest(array: np.ndarray) -> str:
    value = np.asarray(array)
    little_dtype = value.dtype.newbyteorder("<")
    little = np.ascontiguousarray(value.astype(little_dtype, copy=False))
    shape = np.asarray(value.shape, dtype="<i8").tobytes(order="C")
    payload = little_dtype.str.encode("ascii") + b"\0" + shape + b"\0" + little.tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_cli_exposes_only_the_three_preregistered_actions() -> None:
    completed = subprocess.run(
        [sys.executable, str(_PATH), "--help"],
        cwd=_PATH.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{seal,mechanism,utility}" in completed.stdout


@pytest.mark.parametrize("action", ["seal", "mechanism", "utility"])
def test_cli_action_help_never_crosses_into_execution(action: str, tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(_PATH), action, "--help"],
        cwd=_PATH.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert list(tmp_path.iterdir()) == []


def test_implementation_readiness_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    review = tmp_path / "IMPLEMENTATION_REVIEW.md"
    review.write_text("# Review\n\nVerdict: PASS\n", encoding="utf-8")
    monkeypatch.setattr(bench, "IMPLEMENTATION_REVIEW", review)
    monkeypatch.setattr(bench, "IMPLEMENTATION_COMPLETE", False)
    monkeypatch.setattr(bench, "IMPLEMENTATION_GAPS", ("toy gap",))
    with pytest.raises(RuntimeError, match="incomplete|gap"):
        bench.assert_implementation_ready()

    monkeypatch.setattr(bench, "IMPLEMENTATION_COMPLETE", True)
    monkeypatch.setattr(bench, "IMPLEMENTATION_GAPS", ())
    review.write_text("# Review\n\nVerdict: FAIL\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PASS"):
        bench.assert_implementation_ready()


def test_seal_covers_protocol_harness_tests_and_every_direct_scientific_source() -> None:
    sealed = {str(path) for path in bench.SEALED_PATHS}
    required = {
        "benchmarks/results/20260716_stage1_semantic_factorial_PREREG.md",
        "benchmarks/results/20260716_stage1_semantic_factorial_PREREG_REVIEW.md",
        "benchmarks/results/20260716_stage1_semantic_factorial_IMPLEMENTATION_REVIEW.md",
        "benchmarks/stage1_semantic_factorial.py",
        "tests/test_stage1_semantic_factorial.py",
        "src/rtgs/core/gaussians2d.py",
        "src/rtgs/core/gaussians3d.py",
        "src/rtgs/core/metrics.py",
        "src/rtgs/data/scene.py",
        "src/rtgs/data/synthetic.py",
        "src/rtgs/depth/mock.py",
        "src/rtgs/image2gs/fit.py",
        "src/rtgs/image2gs/renderer2d.py",
        "src/rtgs/lift/base.py",
        "src/rtgs/lift/carve.py",
        "src/rtgs/lift/depth.py",
        "src/rtgs/optim/trainer.py",
        "src/rtgs/render/torch_ref.py",
    }
    assert required <= sealed
    assert len(bench.SEALED_PATHS) == len(sealed)
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


def test_default_fit_config_is_the_frozen_default_preserving_9p_path() -> None:
    config = bench.fit_config()
    assert asdict(config) == {
        "n_gaussians": 150,
        "max_gaussians": 5_000,
        "iterations": 120,
        "backend": "native",
        "adaptive_density": True,
        "growth_waves": 5,
        "relocate_fraction": 0.0,
        "structsplat_renderer": "auto",
        # Stage-1 acceleration seams stay off in the frozen harness: the exact serial
        # torch path is the preregistered semantics.
        "native_renderer": "torch",
        "batch_views": False,
        "lr": 0.01,
        "grad_init_mix": 0.7,
        "row_chunk": 64,
        "log_every": 50,
        "convergence_patience": 0,
        "convergence_tol": 0.05,
        "convergence_check_every": 25,
        "appearance_parameterization": "weight_color_9p",
        "freeze_geometry": False,
        # The opt-in pooled allocation seam is off; the frozen harness fits the fixed native set.
        "pool": False,
        "pool_capacity": None,
        "pool_triage_every": 50,
        "pool_prune_count": 32,
        "pool_spawn_count": 32,
        "pool_min_live": 1,
    }


def test_gauges_preserve_geometry_order_bounds_amplitude_and_zero_branch() -> None:
    source = _toy_gaussians2d()
    gauges = bench.construct_gauges(source)
    amplitude = source.weight[:, None] * source.color
    assert tuple(gauges) == ("identity", "unit_weight", "peak_color")
    for gauge in gauges.values():
        assert torch.equal(gauge.xy, source.xy)
        assert torch.equal(gauge.chol, source.chol)
        assert bool(torch.isfinite(gauge.weight).all())
        assert bool(torch.isfinite(gauge.color).all())
        assert bool(((gauge.weight >= 0) & (gauge.weight <= 1)).all())
        assert bool(((gauge.color >= 0) & (gauge.color <= 1)).all())
        assert torch.allclose(
            gauge.weight[:, None] * gauge.color,
            amplitude,
            atol=1e-7,
            rtol=1e-6,
        )
    peak = gauges["peak_color"]
    assert peak.weight[0].item() == 0.0
    assert torch.equal(peak.color[0], torch.zeros(3))


def test_candidate_fields_use_exact_zero_branch_and_bilinear_pixel_centers() -> None:
    source = _toy_gaussians2d()
    image = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]],
            [[0.9, 0.8, 0.7], [0.3, 0.2, 0.1]],
        ]
    )
    fields = bench.candidate_fields(source, image)
    amplitude = source.weight[:, None] * source.color
    expected_peak = amplitude.max(dim=-1).values
    assert torch.equal(fields["a"], amplitude)
    assert torch.equal(fields["m"], expected_peak)
    assert torch.equal(fields["h"][0], torch.zeros(3))
    assert torch.equal(fields["o"][:3], image.reshape(-1, 3)[:3])
    assert torch.allclose(fields["o"][3], image.mean(dim=(0, 1)))
    assert bool(torch.isfinite(fields["h"]).all())
    assert bool(((fields["h"] >= 0) & (fields["h"] <= 1)).all())
    assert torch.allclose(fields["m"][:, None] * fields["h"], amplitude)


def test_mechanism_representations_route_the_frozen_scalar_and_color_fields() -> None:
    gauge = _toy_gaussians2d()
    image = torch.rand(2, 2, 3, generator=torch.Generator().manual_seed(9))
    fields = bench.candidate_fields(gauge, image)
    representations = bench.mechanism_representations(gauge, fields)
    assert tuple(representations) == bench.MECHANISM_REPRESENTATIONS
    observed = representations["m_amp__rgb_obs"]
    normalized = representations["m_amp__h_norm"]
    unit = representations["unit_weight__a_amp"]
    assert torch.equal(observed.weight, fields["m"])
    assert torch.equal(observed.color, fields["o"])
    assert torch.equal(normalized.weight, fields["m"])
    assert torch.equal(normalized.color, fields["h"])
    assert torch.equal(unit.weight, torch.ones_like(gauge.weight))
    assert torch.equal(unit.color, fields["a"])
    for representation in representations.values():
        assert torch.equal(representation.xy, gauge.xy)
        assert torch.equal(representation.chol, gauge.chol)
        assert bool(torch.isfinite(representation.weight).all())
        assert bool(torch.isfinite(representation.color).all())
        assert bool(((representation.weight >= 0) & (representation.weight <= 1)).all())
        assert bool(((representation.color >= 0) & (representation.color <= 1)).all())


def test_candidate_semantics_are_gauge_invariant_on_the_toy_view() -> None:
    source = _toy_gaussians2d()
    image = torch.rand(2, 2, 3, generator=torch.Generator().manual_seed(11))
    gauges = bench.construct_gauges(source)
    fields = {name: bench.candidate_fields(gauge, image) for name, gauge in gauges.items()}
    identity = fields["identity"]
    for name in bench.GAUGES:
        candidate = fields[name]
        assert torch.equal(candidate["o"], identity["o"])
        assert torch.allclose(candidate["m"], identity["m"], atol=1e-7, rtol=1e-6)
        assert torch.allclose(candidate["h"], identity["h"], atol=2e-6, rtol=2e-5)
        assert torch.allclose(
            candidate["m"][:, None] * candidate["h"],
            identity["a"],
            atol=2e-7,
            rtol=2e-6,
        )


def test_candidate_fields_reject_nonfinite_or_out_of_range_observations() -> None:
    source = _toy_gaussians2d()
    nonfinite = torch.zeros(2, 2, 3)
    nonfinite[0, 0, 0] = torch.nan
    with pytest.raises(bench.ProtocolInvalid, match="non-finite"):
        bench.candidate_fields(source, nonfinite)
    out_of_range = torch.full((2, 2, 3), 1.1)
    with pytest.raises(bench.ProtocolInvalid, match="range"):
        bench.candidate_fields(source, out_of_range)


def test_raw_content_hash_uses_exact_little_endian_semantics() -> None:
    native = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    big_endian = native.astype(">i4")
    expected = _independent_raw_digest(native)
    assert bench.raw_content_sha256(native) == expected
    assert bench.raw_content_sha256(big_endian) == expected
    assert bench.raw_content_sha256(native[:, ::-1]) != expected


def test_raw_archive_rejects_object_string_and_nonfinite_arrays() -> None:
    archive = bench.RawArchive()
    with pytest.raises((TypeError, ValueError), match="object|numeric|boolean"):
        archive.add("bad/object", np.asarray([{"x": 1}], dtype=object))
    with pytest.raises((TypeError, ValueError), match="string|numeric|boolean"):
        archive.add("bad/string", np.asarray(["not raw evidence"]))
    with pytest.raises((TypeError, ValueError), match="finite"):
        archive.add("bad/nan", np.asarray([np.nan], dtype=np.float32))
    with pytest.raises((TypeError, ValueError), match="finite"):
        archive.add("bad/inf", np.asarray([np.inf], dtype=np.float32))


def test_invalid_nonfinite_evidence_uses_finite_nullable_encoding() -> None:
    archive = bench.RawArchive()
    archive.phase = "candidate_fields"
    with pytest.raises(bench.ProtocolInvalid, match="non-finite"):
        archive.add("failure/nonfinite/value", np.asarray([np.nan, np.inf, -np.inf]))
    archive.add_nullable("failure/optional", 123.0, defined=False)
    assert np.array_equal(archive.arrays["failure/nonfinite/value"], [0.0, 0.0, 0.0])
    assert np.array_equal(
        archive.arrays["failure/nonfinite/value/nonfinite_classification"], [1, 2, 3]
    )
    assert archive.arrays["failure/optional/value"].item() == 0
    assert not bool(archive.arrays["failure/optional/defined"].item())
    manifest = archive.manifest()
    assert all(entry["name"].startswith("failure/") for entry in manifest)


def test_protocol_invalid_sanitizes_nonfinite_structured_evidence() -> None:
    error = bench.ProtocolInvalid(
        "mechanism_lifts",
        "toy parity failure",
        {"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")},
    )
    assert error.evidence == {
        "nan": {"value": 0.0, "nonfinite_classification": 1},
        "positive": {"value": 0.0, "nonfinite_classification": 2},
        "negative": {"value": 0.0, "nonfinite_classification": 3},
    }
    assert json.loads(bench.canonical_json(error.evidence)) == error.evidence


def test_uncompressed_pickle_free_sidecar_roundtrip(tmp_path: Path) -> None:
    archive = bench.RawArchive()
    archive.add("fit/seed=17/view=0/xy", np.arange(8, dtype=np.float32).reshape(4, 2))
    archive.add("fit/seed=17/view=0/source_keys", np.arange(12, dtype=np.int64).reshape(4, 3))
    archive.add("fit/seed=17/view=0/valid", np.asarray([True, False, True, True]))
    path = tmp_path / "toy_RAW.npz"
    metadata = archive.write(path)
    assert metadata["npz_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        archive.write(path)

    with zipfile.ZipFile(path) as zipped:
        assert zipped.namelist()
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zipped.infolist())
    with np.load(path, allow_pickle=False) as loaded:
        assert sorted(loaded.files) == sorted(entry["name"] for entry in archive.manifest())
        assert np.array_equal(loaded["fit/seed=17/view=0/source_keys"], np.arange(12).reshape(4, 3))
        assert all(loaded[name].dtype.kind in "biufc" for name in loaded.files)

    assert archive.collection_sha256() == bench.raw_collection_sha256(archive.manifest())


def test_raw_sidecar_validator_independently_rejects_nonfinite_arrays(tmp_path: Path) -> None:
    path = tmp_path / "corrupt_RAW.npz"
    arrays = {"bad/value": np.asarray([np.nan], dtype=np.float32)}
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
    manifest, collection = bench.array_manifest(arrays)
    binding = {
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest": manifest,
        "collection_sha256": collection,
        "array_count": 1,
    }
    with pytest.raises(ValueError, match="non-finite"):
        bench.validate_raw_sidecar(path, binding)


def test_raw_manifest_is_sorted_complete_and_name_sensitive() -> None:
    archive = bench.RawArchive()
    archive.add("z/value", np.asarray([2.0], dtype=np.float64))
    archive.add("a/value", np.asarray([1.0], dtype=np.float32))
    manifest = archive.manifest()
    assert [entry["name"] for entry in manifest] == ["a/value", "z/value"]
    for entry in manifest:
        assert set(entry) == {"name", "dtype", "shape", "byte_length", "raw_content_sha256"}
    pairs = [[entry["name"], entry["raw_content_sha256"]] for entry in manifest]
    expected_collection = hashlib.sha256(
        json.dumps(pairs, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    assert archive.collection_sha256() == expected_collection
    renamed = [dict(entry) for entry in manifest]
    renamed[0]["name"] = "b/value"
    assert bench.raw_collection_sha256(manifest) != bench.raw_collection_sha256(renamed)


def test_configs_and_fit_histories_are_decodable_numeric_raw_evidence() -> None:
    archive = bench.RawArchive()
    bench.add_config_raw(archive, "fit/config", bench.fit_config())
    histories = [
        {"iteration": [0, 50, 100, 119], "loss": [1.0, 0.7, 0.5, 0.4]},
        {"iteration": [0, 50, 100, 119], "loss": [1.1, 0.8, 0.6, 0.45]},
    ]
    bench.add_fit_histories_raw(archive, "fit/histories", histories)
    arrays = archive.arrays
    assert arrays["fit/config/n_gaussians"].item() == 150
    assert arrays["fit/config/iterations"].item() == 120
    assert arrays["fit/config/freeze_geometry"].item() == 0
    assert bytes(arrays["fit/config/backend/utf8"]).decode() == "native"
    assert (
        bytes(arrays["fit/config/appearance_parameterization/utf8"]).decode() == "weight_color_9p"
    )
    assert np.array_equal(arrays["fit/histories/view=0/iteration"], [0, 50, 100, 119])
    assert np.allclose(arrays["fit/histories/view=1/loss"], [1.1, 0.8, 0.6, 0.45])
    assert all(value.dtype.kind in "biufc" for value in arrays.values())


def test_utility_arms_have_the_exact_two_by_two_factorial_fields() -> None:
    fitted = _toy_gaussians2d()
    image = torch.tensor(
        [
            [[0.9, 0.1, 0.2], [0.8, 0.2, 0.3]],
            [[0.7, 0.3, 0.4], [0.6, 0.4, 0.5]],
        ]
    )
    fields = bench.candidate_fields(fitted, image)
    arms = bench.construct_utility_arms(fitted, fields)
    bench.validate_factorial_integrity(fitted, arms)

    arm00 = arms["w_fit__c_fit"]
    arm10 = arms["m_amp__c_fit"]
    arm01 = arms["w_fit__rgb_obs"]
    arm11 = arms["m_amp__rgb_obs"]
    for arm in arms.values():
        assert torch.equal(arm.xy, fitted.xy)
        assert torch.equal(arm.chol, fitted.chol)
    assert torch.equal(arm00.weight, fitted.weight)
    assert torch.equal(arm00.color, fitted.color)
    assert torch.equal(arm10.weight, fields["m"])
    assert torch.equal(arm10.color, fitted.color)
    assert torch.equal(arm01.weight, fitted.weight)
    assert torch.equal(arm01.color, fields["o"])
    assert torch.equal(arm11.weight, arm10.weight)
    assert torch.equal(arm11.color, arm01.color)


def test_factorial_validator_rejects_a_cross_contaminated_arm() -> None:
    fitted = _toy_gaussians2d()
    fields = bench.candidate_fields(fitted, torch.rand(2, 2, 3))
    arms = bench.construct_utility_arms(fitted, fields)
    arms["m_amp__c_fit"] = Gaussians2D(
        arms["m_amp__c_fit"].xy,
        arms["m_amp__c_fit"].chol,
        fields["o"],
        arms["m_amp__c_fit"].weight,
    )
    with pytest.raises(bench.ProtocolInvalid, match="factorial|color"):
        bench.validate_factorial_integrity(fitted, arms)


def test_factorial_validator_requires_both_treatments_to_change_ten_percent() -> None:
    fitted = _toy_gaussians2d()
    no_treatment = {
        "m": fitted.weight.detach().clone(),
        "o": fitted.color.detach().clone(),
    }
    arms = bench.construct_utility_arms(fitted, no_treatment)
    with pytest.raises(bench.ProtocolInvalid, match="10%|separation"):
        bench.validate_factorial_integrity(fitted, arms)


def test_coverage_name_is_the_single_canonical_content_domain() -> None:
    name = bench.coverage_array_name(1103, "unit_weight", "m_amp__rgb_obs", 4)
    assert name == "coverage/seed=1103/gauge=unit_weight/arm=m_amp__rgb_obs/view=4"
    utility_name = bench.coverage_array_name(4409, "identity", "w_fit__c_fit", 4)
    assert utility_name == "coverage/seed=4409/gauge=identity/arm=w_fit__c_fit/view=4"
    archive = bench.RawArchive()
    coverage = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    archive.add(name, coverage)
    digest = next(entry for entry in archive.manifest() if entry["name"] == name)[
        "raw_content_sha256"
    ]
    sidecar_reference = {"raw_name": name, "raw_content_sha256": digest}
    assert sidecar_reference["raw_name"] in archive.arrays
    assert sidecar_reference["raw_content_sha256"] == bench.raw_content_sha256(coverage)
    with pytest.raises(bench.ProtocolInvalid, match="reused"):
        archive.add(name, coverage.copy())


def test_official_output_paths_are_complete_disjoint_and_strict() -> None:
    result_dir = bench.ROOT / "benchmarks/results"
    output = result_dir / "20990101T010203Z_cpu_stage1_semantic_factorial_mechanism.json"
    paths = bench.official_output_paths(output, "mechanism")
    assert set(paths) == {
        "valid_json",
        "valid_raw",
        "valid_note",
        "invalid_json",
        "invalid_raw",
        "invalid_note",
    }
    assert len(set(paths.values())) == 6
    assert paths["valid_raw"].name.endswith("_mechanism_RAW.npz")
    assert paths["invalid_raw"].name.endswith("_mechanism_invalid_RAW.npz")
    utility_output = result_dir / "20990101T010204Z_cpu_stage1_semantic_factorial_utility.json"
    utility_paths = bench.official_output_paths(utility_output, "utility")
    assert utility_paths["valid_raw"].name.endswith("_utility_RAW.npz")
    assert utility_paths["invalid_note"].name.endswith("_utility_invalid_RESULT.md")
    with pytest.raises(ValueError, match="official output"):
        bench.official_output_paths(result_dir / "result.json", "mechanism")
    with pytest.raises(ValueError, match="phase"):
        bench.official_output_paths(output, "not-a-phase")


def test_once_only_marker_requires_every_prospective_sibling_to_be_fresh(tmp_path: Path) -> None:
    marker = tmp_path / "attempt.json"
    prospective = [tmp_path / f"result-{index}" for index in range(6)]
    bench.claim_attempt_marker(marker, {"artifact_type": "toy"}, prospective)
    assert json.loads(marker.read_text(encoding="utf-8"))["artifact_type"] == "toy"
    with pytest.raises((FileExistsError, bench.ProtocolInvalid), match="exist|claimed"):
        bench.claim_attempt_marker(marker, {"artifact_type": "toy"}, prospective)

    marker.unlink()
    prospective[2].write_text("occupied", encoding="utf-8")
    with pytest.raises((FileExistsError, bench.ProtocolInvalid), match="exist|occupied"):
        bench.claim_attempt_marker(marker, {"artifact_type": "toy"}, prospective)
    assert not marker.exists()


def test_full_recorded_environment_participates_in_runtime_fingerprint() -> None:
    baseline = bench.environment_metadata()
    for key in ("python", "platform", "processor", "torch", "numpy"):
        changed = dict(baseline)
        changed[key] = f"{baseline[key]}-drift"
        assert bench._environment_fingerprint(changed) != bench._environment_fingerprint(baseline)


def test_offline_network_guard_blocks_dns_and_restores_socket_module() -> None:
    archive = bench.RawArchive()
    archive.phase = "candidate_fields"
    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    with bench.offline_network_guard(archive):
        with pytest.raises(bench.ProtocolInvalid, match="network access"):
            socket.getaddrinfo("example.invalid", 443)
        with pytest.raises(bench.ProtocolInvalid, match="network access"):
            socket.socket().connect(("127.0.0.1", 1))
        with pytest.raises(bench.ProtocolInvalid, match="child process"):
            subprocess.run(["true"], check=False)
        for event in ("os.fork", "os.forkpty", "os.exec", "pty.spawn"):
            with pytest.raises(bench.ProtocolInvalid, match="child process"):
                bench._scientific_runtime_audit_hook(event, ())
    assert socket.socket is original_socket
    assert socket.create_connection is original_create_connection
    assert socket.getaddrinfo is original_getaddrinfo
    assert not bench._NETWORK_GUARD_ACTIVE
    assert bench._NETWORK_GUARD_ARCHIVE is None
    assert subprocess.run(["true"], check=False).returncode == 0


def test_phase_a_review_schema_requires_raw_and_every_decisive_gate() -> None:
    raw_binding = {"array_count": 17, "collection_sha256": "c" * 64}
    review = {
        "raw_archive_recomputation": {
            "loaded_with_allow_pickle_false": True,
            "npz_sha256_recomputed": True,
            "semantic_manifest_recomputed": True,
            "collection_sha256_recomputed": True,
            "all_raw_floating_arrays_finite": True,
            "array_count": 17,
            "collection_sha256": "c" * 64,
        },
        "decisive_gate_recomputation": {
            gate: {"recomputed": True, "passed": True, "evidence": {"toy_count": 1}}
            for gate in bench.PHASE_A_REVIEW_GATES
        },
    }
    bench._validate_phase_a_review_recomputation(review, raw_binding)
    missing = dict(review)
    missing["decisive_gate_recomputation"] = dict(review["decisive_gate_recomputation"])
    missing["decisive_gate_recomputation"].pop(bench.PHASE_A_REVIEW_GATES[-1])
    with pytest.raises(ValueError, match="gate schema"):
        bench._validate_phase_a_review_recomputation(missing, raw_binding)
    assert bench._require_utc_timestamp("2026-07-16T12:00:00Z", "reviewed_at_utc")
    with pytest.raises(ValueError, match="UTC"):
        bench._require_utc_timestamp("2026-07-16T12:00:00+02:00", "reviewed_at_utc")


def _toy_artifact_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        f"{label}_{suffix}": tmp_path / f"{label}.{suffix}"
        for label in ("valid", "invalid")
        for suffix in ("json", "raw", "note")
    }


def test_artifact_writer_validates_strict_json_before_creating_raw(tmp_path: Path) -> None:
    archive = bench.RawArchive()
    archive.add("toy/value", np.asarray([1.0], dtype=np.float32))
    paths = _toy_artifact_paths(tmp_path)
    seal = {"path": "toy-seal", "sha256": "s" * 64, "source_collection_sha256": "c" * 64}
    marker = {"path": "toy-marker", "sha256": "m" * 64}
    with pytest.raises(ValueError, match="JSON|range|compliant"):
        bench._write_phase_artifact(
            phase="mechanism",
            valid=False,
            paths=paths,
            archive=archive,
            scientific_payload={"bad": float("nan"), "decision": {}},
            seal=seal,
            marker=marker,
        )
    assert not any(path.exists() for path in paths.values())


def test_bound_nonfinite_invalidation_writes_only_complete_invalid_triple(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _toy_artifact_paths(tmp_path)
    seal = {"path": "toy-seal", "sha256": "s" * 64, "source_collection_sha256": "c" * 64}
    marker = {"path": "toy-marker", "sha256": "m" * 64}

    monkeypatch.setattr(bench, "official_output_paths", lambda _output, _phase: paths)
    monkeypatch.setattr(bench, "load_and_verify_seal", lambda _path: seal)
    monkeypatch.setattr(bench, "claim_attempt_marker", lambda *_args, **_kwargs: marker)
    monkeypatch.setattr(bench, "_validate_marker", lambda _marker: None)

    def fail(archive: bench.RawArchive) -> None:
        archive.add("toy/before_failure", np.asarray([1.0], dtype=np.float32))
        raise bench.ProtocolInvalid("mechanism_lifts", "toy failure", {"means": float("nan")})

    monkeypatch.setattr(bench, "execute_mechanism", fail)
    written = bench.run_bound_mechanism(tmp_path / "seal.json", tmp_path / "result.json")
    assert not written["valid"]
    assert all(paths[f"invalid_{suffix}"].is_file() for suffix in ("json", "raw", "note"))
    assert not any(paths[f"valid_{suffix}"].exists() for suffix in ("json", "raw", "note"))
    payload = json.loads(paths["invalid_json"].read_text(encoding="utf-8"))
    assert payload["evidence_reached_before_invalidation"]["means"] == {
        "value": 0.0,
        "nonfinite_classification": 1,
    }


def test_heldout_guard_raises_for_every_field_before_unlock() -> None:
    images = [torch.full((2, 2, 3), float(index)) for index in range(3)]
    cameras = [object(), object(), object()]
    depths = [torch.full((2, 2), float(index + 1)) for index in range(3)]
    guard = bench.HeldOutGuard(images, cameras, depths, original_indices=(3, 7, 11))
    assert not guard.unlocked
    for field in ("images", "cameras", "depths", "original_indices"):
        with pytest.raises(RuntimeError, match="locked"):
            getattr(guard, field)


def test_heldout_guard_has_one_global_unlock_and_no_mutable_aliases() -> None:
    images = [torch.full((2, 2, 3), float(index)) for index in range(3)]
    cameras = [object(), object(), object()]
    depths = [torch.full((2, 2), float(index + 1)) for index in range(3)]
    guard = bench.HeldOutGuard(images, cameras, depths)
    images.clear()
    cameras.clear()
    depths.clear()
    guard.unlock()
    assert guard.unlocked
    assert len(guard.images) == len(guard.cameras) == len(guard.depths) == 3
    assert guard.original_indices == (3, 7, 11)
    with pytest.raises(RuntimeError, match="only once"):
        guard.unlock()


def test_rho_uses_float64_peak_amplitude_and_cholesky_diagonal_product() -> None:
    fitted = _toy_gaussians2d()
    amplitude = fitted.weight[:, None].double() * fitted.color.double()
    peak = amplitude.max(dim=-1).values
    rho = bench.integrated_mass_rho(fitted, peak)
    expected = peak * fitted.chol[:, 0].double() * fitted.chol[:, 2].double()
    assert rho.dtype == torch.float64
    assert torch.equal(rho, expected)


@pytest.mark.parametrize("backend", ["Depth", "Carve"])
def test_rho_tie_break_uses_exact_utf8_domain_and_literal_backend_token(backend: str) -> None:
    payload = f"stage1-semantic-factorial-v1|4409|{backend}|2|17".encode()
    assert bench.rho_tie_break(4409, backend, 2, 17) == hashlib.sha256(payload).digest()
    with pytest.raises(ValueError, match="literal"):
        bench.rho_tie_break(4409, backend.lower(), 2, 17)


def test_rank_available_keys_uses_rho_then_digest_then_component() -> None:
    rho = np.zeros(bench.COMPONENTS_PER_VIEW, dtype=np.float64)
    rho[[2, 3]] = 4.0
    rho[7] = 5.0
    tied = sorted(
        [2, 3], key=lambda component: (bench.rho_tie_break(4409, "Depth", 1, component), component)
    )
    ranked = bench.rank_available_keys(
        seed=4409,
        backend="Depth",
        local_view=1,
        available_components=[3, 7, 2],
        rho=rho,
    )
    assert ranked == [7, *tied]


def test_exact_capacity_uses_minimum_per_view_and_canonical_output_order() -> None:
    availability = {
        arm: [list(range(30 + index)) for _view in range(9)]
        for index, arm in enumerate(bench.UTILITY_ARMS)
    }
    rho = [
        np.linspace(float(view), float(view + 1), bench.COMPONENTS_PER_VIEW, dtype=np.float64)
        for view in range(9)
    ]
    matched = bench.match_exact_capacity(
        seed=4409,
        backend="Depth",
        availability_by_arm_view=availability,
        rho_by_view=rho,
    )
    assert matched["per_view_quotas"] == [30] * 9
    assert matched["total_quota"] == 270
    for arm in bench.UTILITY_ARMS:
        selected = matched["selected_components"][arm]
        assert [len(view) for view in selected] == [30] * 9
        assert all(view == sorted(view) for view in selected)


def test_exact_capacity_defaults_enforce_per_view_and_total_floors() -> None:
    availability = {arm: {view: list(range(20)) for view in range(9)} for arm in bench.UTILITY_ARMS}
    rho = {view: np.arange(bench.COMPONENTS_PER_VIEW, dtype=np.float64) for view in range(9)}
    with pytest.raises(bench.ProtocolInvalid, match="270|total|capacity"):
        bench.match_exact_capacity(
            seed=4409,
            backend="Depth",
            availability_by_arm_view=availability,
            rho_by_view=rho,
        )


def test_all_capacity_cells_pass_before_any_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = (101, 202)
    monkeypatch.setattr(bench, "PHASE_B_SEEDS", seeds)
    monkeypatch.setattr(bench, "TRAIN_INDICES", (0,))
    monkeypatch.setattr(bench, "COMPONENTS_PER_VIEW", 8)
    fitted = _toy_gaussians2d()
    prepared = [
        SimpleNamespace(
            seed=seed, fitted=[fitted], scene=object(), heldout=SimpleNamespace(unlocked=False)
        )
        for seed in seeds
    ]
    fields = {seed: [{"m": torch.ones(fitted.n)}] for seed in seeds}
    arms = {seed: {arm: [fitted] for arm in bench.UTILITY_ARMS} for seed in seeds}
    dummy_gaussian = SimpleNamespace(n=8)
    refinement_calls: list[tuple[int, str, str]] = []
    capacity_calls = 0

    monkeypatch.setattr(bench, "integrated_mass_rho", lambda *_args: torch.ones(8))
    monkeypatch.setattr(
        bench, "render_gaussian_coverage_2d", lambda *_args, **_kwargs: torch.ones(2, 2)
    )

    def lift(*, seed: int, backend: str, archive: bench.RawArchive, **_kwargs):
        archive.completed_lifts += 1
        keys = [(seed, 0, component) for component in range(8)]
        return bench.LiftResult(dummy_gaussian, keys, {"backend": backend}, {})

    monkeypatch.setattr(bench, "run_lift", lift)
    monkeypatch.setattr(bench.RawArchive, "add_gaussians3d", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bench,
        "_matched_initialization",
        lambda lift, seed, selected: (
            dummy_gaussian,
            [(seed, 0, component) for component in selected[0]],
        ),
    )

    def capacity(**_kwargs):
        nonlocal capacity_calls
        capacity_calls += 1
        if capacity_calls == len(seeds) * len(bench.BACKENDS):
            raise bench.ProtocolInvalid("capacity", "toy final-cell floor failure")
        selected = {arm: [list(range(8))] for arm in bench.UTILITY_ARMS}
        return {
            "per_view_quotas": [8],
            "total_quota": 8,
            "selected_components": selected,
            "ranked_components": selected,
        }

    monkeypatch.setattr(bench, "match_exact_capacity", capacity)

    def refine(**kwargs):
        refinement_calls.append((kwargs["seed"], kwargs["backend"], kwargs["arm"]))
        raise AssertionError("refinement crossed the global capacity gate")

    monkeypatch.setattr(bench, "refine_utility_model", refine)
    with pytest.raises(bench.ProtocolInvalid, match="final-cell"):
        bench.run_utility_preunlock(prepared, fields, arms, bench.RawArchive())
    assert capacity_calls == len(seeds) * len(bench.BACKENDS)
    assert refinement_calls == []


@pytest.mark.parametrize("backend", ["Depth", "Carve"])
def test_expected_schedule_is_the_exact_cpu_torch_randint_stream(backend: str) -> None:
    schedule = bench.expected_schedule(4409, backend)
    generator = torch.Generator(device="cpu").manual_seed(bench.trainer_seed(4409, backend))
    independent = np.asarray(
        [int(torch.randint(0, 9, (1,), generator=generator)) for _ in range(120)],
        dtype=np.int64,
    )
    assert schedule.dtype == np.dtype(np.int64)
    assert np.array_equal(schedule, independent)
    assert set(schedule.tolist()) == set(range(9))


def test_factorial_estimands_are_exact() -> None:
    values = {
        "w_fit__c_fit": 10.0,
        "m_amp__c_fit": 12.0,
        "w_fit__rgb_obs": 13.0,
        "m_amp__rgb_obs": 17.0,
    }
    assert bench.factorial_estimands(values) == {
        "scalar_main_effect": 3.0,
        "color_main_effect": 4.0,
        "interaction": 2.0,
        "full_candidate_difference": 7.0,
    }


def test_utility_reduction_summarizes_every_effect_for_both_metrics() -> None:
    records = {}
    for seed_index, seed in enumerate(bench.PHASE_B_SEEDS):
        records[seed] = {}
        for backend_index, backend in enumerate(bench.BACKENDS):
            base = 20.0 + seed_index + backend_index
            records[seed][backend] = {}
            for arm_index, arm in enumerate(bench.UTILITY_ARMS):
                records[seed][backend][arm] = {
                    "final": {
                        "mean_psnr": base + arm_index,
                        "mean_ssim": 0.5 + 0.01 * (seed_index + backend_index + arm_index),
                    }
                }
    reduction = bench.reduce_utility_results({"seeds": records}, validity_gates_pass=True)
    names = {
        "scalar_main_effect",
        "color_main_effect",
        "interaction",
        "full_candidate_difference",
    }
    for backend in bench.BACKENDS:
        for metric in ("psnr", "ssim"):
            summaries = reduction["paired_summaries"][backend][metric]
            assert set(summaries) == names
            for summary in summaries.values():
                assert len(summary["seed_effects"]) == 3
                assert summary["minimum"] <= summary["mean"] <= summary["maximum"]


def test_backend_decision_threshold_boundaries_are_inclusive() -> None:
    noninferior = bench.backend_decision(
        [-0.75, 0.0, 0.0],
        [-0.020, 0.0025, 0.0025],
        validity_gates_pass=True,
    )
    assert noninferior["noninferior"]
    assert not noninferior["material_improvement"]
    improved = bench.backend_decision(
        [-0.25, 0.5, 0.5],
        [0.0, 0.0, 0.0],
        validity_gates_pass=True,
    )
    assert improved["noninferior"]
    assert improved["material_improvement"]
    assert not bench.backend_decision([1.0, 1.0, 1.0], [0.1, 0.1, 0.1], validity_gates_pass=False)[
        "noninferior"
    ]


def test_cross_backend_decisions_require_both_backends() -> None:
    psnr = {"Depth": [0.3, 0.3, 0.3], "Carve": [0.3, 0.3, 0.3]}
    ssim = {"Depth": [0.0, 0.0, 0.0], "Carve": [0.0, 0.0, 0.0]}
    both = bench.frozen_decisions(psnr, ssim, validity_gates_pass=True)
    assert both["repair_utility_survives"]
    assert both["cross_backend_material_improvement"]
    psnr["Carve"] = [-0.8, 0.0, 0.0]
    one = bench.frozen_decisions(psnr, ssim, validity_gates_pass=True)
    assert one["by_backend"]["Depth"]["material_improvement"]
    assert not one["by_backend"]["Carve"]["noninferior"]
    assert not one["repair_utility_survives"]
    assert not one["cross_backend_material_improvement"]


def test_material_driver_requires_magnitude_and_two_matching_seed_signs() -> None:
    assert bench.material_driver([0.5, 0.5, -0.2])
    assert bench.material_driver([-0.5, -0.5, 0.2])
    assert not bench.material_driver([0.2, 0.2, 0.2])
    assert not bench.material_driver([1.0, -1.0, 0.0])
