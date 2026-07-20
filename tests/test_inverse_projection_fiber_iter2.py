"""CPU tests for the frozen Iteration 2 inverse-projection-fiber harness."""

from __future__ import annotations

import inspect
import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from benchmarks import inverse_projection_fiber_iter2 as iter2


@pytest.fixture(scope="module")
def relabeling_sentinel() -> dict[str, object]:
    return iter2._run_relabeling_sentinel()


def test_candidate_boundary_permutations_and_relabeling_are_deterministic(
    relabeling_sentinel: dict[str, object],
) -> None:
    candidate, evaluator, heldout_recipe, first = iter2._build_root_inputs(91_001, 91_101, 91_201)
    _candidate_again, _evaluator_again, _heldout_again, second = iter2._build_root_inputs(
        91_001,
        91_101,
        91_201,
    )
    assert first == second
    assert sorted(candidate.__dataclass_fields__) == first["candidate_api_fields"]
    assert all(
        "label" not in field and not field.startswith("gt")
        for field in first["candidate_api_fields"]
    )
    assert all("heldout" not in field for field in evaluator.__dataclass_fields__)
    released = iter2._materialize_heldout(evaluator, heldout_recipe)
    assert len(released.cameras) == iter2.N_HELDOUT_VIEWS
    assert released.receipt["materialized"]
    assert first["scene_generator"]["before_sha256"]
    assert first["scene_generator"]["after_sha256"]
    assert relabeling_sentinel["pass"]
    assert all(relabeling_sentinel["checks"].values())
    signature = inspect.signature(iter2._relabeling_sentinel)
    assert "evaluator" not in signature.parameters
    sentinel_source = inspect.getsource(iter2._relabeling_sentinel)
    assert "heldout" not in sentinel_source
    assert "fit_labels" not in sentinel_source


def test_root_inputs_are_constructed_exactly_once(monkeypatch) -> None:
    candidate, evaluator, heldout_recipe, _receipt = iter2._build_root_inputs(
        92_001, 92_101, 92_201
    )
    calls: list[tuple[int, int, int]] = []

    def fake_builder(scene: int, depth: int, order: int):
        roots = (scene, depth, order)
        calls.append(roots)
        return candidate, evaluator, heldout_recipe, {"roots": list(roots)}

    monkeypatch.setattr(iter2, "_build_root_inputs", fake_builder)
    requested = [(93_001, 93_101, 93_201), (93_002, 93_102, 93_202)]
    bundles = iter2._construct_root_inputs_once(requested)
    assert calls == requested
    assert [bundle.roots for bundle in bundles] == requested
    assert all(bundle.candidate is candidate for bundle in bundles)
    assert "_build_root_inputs" not in inspect.getsource(iter2._run_root)


def test_fit_costs_reject_nonpositive_loss_view_depth(monkeypatch) -> None:
    candidate, _evaluator, _heldout_recipe, _receipt = iter2._build_root_inputs(
        94_001, 94_101, 94_201
    )
    model = iter2._new_fiber(candidate)
    costs, depths = iter2._fit_costs(model, candidate)
    assert len(costs) == iter2.N_FIT_VIEWS
    assert torch.all(depths > 0)
    original_project = iter2.InverseProjectionFiber.project

    def negative_depth(self, camera):
        projection = original_project(self, camera)
        return SimpleNamespace(
            means2d=projection.means2d,
            covariances2d=projection.covariances2d,
            depth=-projection.depth.abs(),
        )

    monkeypatch.setattr(iter2.InverseProjectionFiber, "project", negative_depth)
    with pytest.raises(RuntimeError, match="non-positive projected loss-view depth"):
        iter2._fit_costs(model, candidate)


def test_npz_and_json_descriptors_are_recaptured_from_disk(tmp_path) -> None:
    npz_path = tmp_path / "evidence.npz"
    npz_descriptor = iter2._write_npz_exclusive(
        npz_path,
        {"values": torch.arange(6, dtype=torch.float64).reshape(2, 3)},
    )
    valid_npz = iter2._validate_npz_descriptor(
        npz_descriptor,
        expected_path=npz_path,
        expected_specs={"values": iter2.MemberSpec("<f8", (2, 3))},
    )
    assert valid_npz["pass"]
    with npz_path.open("r+b") as stream:
        stream.seek(0)
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 0x01]))
        stream.flush()
    assert not iter2._validate_npz_descriptor(
        npz_descriptor,
        expected_path=npz_path,
        expected_specs={"values": iter2.MemberSpec("<f8", (2, 3))},
    )["pass"]

    json_path = tmp_path / "receipt.json"
    payload = {"finite": 1.0, "status": "COMMITTED"}
    descriptor = iter2._write_json_exclusive(json_path, payload)
    assert iter2._validate_json_descriptor(
        descriptor,
        expected_path=json_path,
        expected_payload=payload,
    )["pass"]
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(iter2._canonical_bytes(payload))
    replacement.replace(json_path)
    assert not iter2._validate_json_descriptor(
        descriptor,
        expected_path=json_path,
        expected_payload=payload,
    )["pass"]


def test_npz_schema_rejects_wrong_dtype_shape_and_byte_count(tmp_path) -> None:
    expected = {"values": iter2.MemberSpec("<f8", (2, 3))}

    wrong_dtype_path = tmp_path / "wrong_dtype.npz"
    wrong_dtype = iter2._write_npz_exclusive(
        wrong_dtype_path,
        {"values": np.zeros((2, 3), dtype=np.float32)},
    )
    wrong_dtype_report = iter2._validate_npz_descriptor(
        wrong_dtype,
        expected_path=wrong_dtype_path,
        expected_specs=expected,
    )
    assert "dtype_mismatch:values" in wrong_dtype_report["errors"]

    wrong_shape_path = tmp_path / "wrong_shape.npz"
    wrong_shape = iter2._write_npz_exclusive(
        wrong_shape_path,
        {"values": np.zeros((3, 2), dtype=np.float64)},
    )
    wrong_shape_report = iter2._validate_npz_descriptor(
        wrong_shape,
        expected_path=wrong_shape_path,
        expected_specs=expected,
    )
    assert "shape_mismatch:values" in wrong_shape_report["errors"]

    payload = io.BytesIO()
    np.save(payload, np.zeros((2, 3), dtype=np.float64), allow_pickle=False)
    wrong_count_path = tmp_path / "wrong_count.npz"
    with zipfile.ZipFile(wrong_count_path, mode="x", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("values.npy", payload.getvalue() + b"x")
    wrong_count_report = iter2._validate_npz_descriptor(
        {},
        expected_path=wrong_count_path,
        expected_specs=expected,
    )
    assert "byte_count_mismatch:values" in wrong_count_report["errors"]


def test_npz_schema_rejects_duplicate_unsafe_fortran_and_oversized_members(
    tmp_path,
    monkeypatch,
) -> None:
    expected = {"values": iter2.MemberSpec("<f8", (2, 3))}
    npy = io.BytesIO()
    np.save(npy, np.zeros((2, 3), dtype=np.float64), allow_pickle=False)
    duplicate_path = tmp_path / "duplicate.npz"
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(
            duplicate_path,
            mode="x",
            compression=zipfile.ZIP_STORED,
        ) as archive,
    ):
        archive.writestr("values.npy", npy.getvalue())
        archive.writestr("values.npy", npy.getvalue())
    duplicate_report = iter2._validate_npz_descriptor(
        {},
        expected_path=duplicate_path,
        expected_specs=expected,
    )
    assert "duplicate_zip_member" in duplicate_report["errors"]

    unsafe_cases = {
        "object": np.array([[object(), object(), object()], [object(), object(), object()]]),
        "structured": np.zeros((2, 3), dtype=[("value", "<f8")]),
        "fortran": np.asfortranarray(np.zeros((2, 3), dtype=np.float64)),
    }
    expected_error = {
        "object": "object_dtype:values",
        "structured": "structured_dtype:values",
        "fortran": "fortran_order:values",
    }
    for case, values in unsafe_cases.items():
        path = tmp_path / f"{case}.npz"
        with path.open("xb") as stream:
            np.savez(stream, values=values)
        report = iter2._validate_npz_descriptor(
            {},
            expected_path=path,
            expected_specs=expected,
        )
        assert expected_error[case] in report["errors"]

    subdtype_npy = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        subdtype_npy,
        {"descr": ("<f8", (2,)), "fortran_order": False, "shape": (1,)},
    )
    subdtype_npy.write(bytes(16))
    subdtype_path = tmp_path / "subdtype.npz"
    with zipfile.ZipFile(
        subdtype_path,
        mode="x",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr("values.npy", subdtype_npy.getvalue())
    subdtype_report = iter2._validate_npz_descriptor(
        {},
        expected_path=subdtype_path,
        expected_specs={"values": iter2.MemberSpec("<f8", (1, 2))},
    )
    assert "subdtype:values" in subdtype_report["errors"]

    oversized_path = tmp_path / "oversized.npz"
    with oversized_path.open("xb") as stream:
        np.savez_compressed(stream, values=np.zeros(100_000, dtype=np.float64))

    def allocation_forbidden(*_args, **_kwargs):
        raise AssertionError("oversized member reached array allocation")

    monkeypatch.setattr(iter2.np, "frombuffer", allocation_forbidden)
    oversized_report = iter2._validate_npz_descriptor(
        {},
        expected_path=oversized_path,
        expected_specs={"values": iter2.MemberSpec("<f8", (1,))},
    )
    assert "member_size_bound_exceeded:values" in oversized_report["errors"]


def test_npz_capture_rejects_during_read_mutation_before_parsing(tmp_path, monkeypatch) -> None:
    path = tmp_path / "mutating.npz"
    descriptor = iter2._write_npz_exclusive(
        path,
        {"values": np.zeros((2, 3), dtype=np.float64)},
    )
    original_pread = iter2.os.pread
    mutated = False

    def mutating_pread(fd: int, count: int, offset: int) -> bytes:
        nonlocal mutated
        block = original_pread(fd, count, offset)
        if not mutated:
            mutated = True
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([value[0] ^ 1]))
                stream.flush()
        return block

    monkeypatch.setattr(iter2.os, "pread", mutating_pread)

    def parsing_forbidden(*_args, **_kwargs):
        raise AssertionError("mutating capture reached NPZ parsing")

    monkeypatch.setattr(iter2, "_load_npz_payload_arrays", parsing_forbidden)
    report = iter2._validate_npz_descriptor(
        descriptor,
        expected_path=path,
        expected_specs={"values": iter2.MemberSpec("<f8", (2, 3))},
    )
    assert not report["pass"]
    assert any("evidence metadata changed during capture" in error for error in report["errors"])


def _zero_arrays(specs: dict[str, iter2.MemberSpec]) -> dict[str, np.ndarray]:
    return {name: np.zeros(spec.shape, dtype=np.dtype(spec.dtype)) for name, spec in specs.items()}


def _add_decision_arrays(
    arrays: dict[str, np.ndarray],
    prefix: str,
    *,
    accepted: bool,
) -> None:
    arrays[f"{prefix}_scores"] = np.zeros(iter2.N_HYPOTHESES, dtype=np.float64)
    retained = np.zeros(iter2.N_HYPOTHESES, dtype=np.bool_)
    component_ids = np.full(iter2.N_HYPOTHESES, -1, dtype=np.int64)
    representative_map = np.full(iter2.N_HYPOTHESES, -1, dtype=np.int64)
    codes = np.zeros(iter2.N_HYPOTHESES, dtype=np.int8)
    if accepted:
        representatives = np.arange(iter2.N_GAUSSIANS, dtype=np.int64)
        retained[representatives] = True
        component_ids[representatives] = representatives
        representative_map[representatives] = representatives
        codes[representatives] = 2
        assignments = np.tile(representatives, (iter2.N_FIT_VIEWS, 1))
    else:
        representatives = np.empty((0,), dtype=np.int64)
        assignments = np.empty((0, 0), dtype=np.int64)
    arrays[f"{prefix}_retained"] = retained
    arrays[f"{prefix}_component_ids"] = component_ids
    arrays[f"{prefix}_representative_for_hypothesis"] = representative_map
    arrays[f"{prefix}_decision_codes"] = codes
    arrays[f"{prefix}_representatives"] = representatives
    arrays[f"{prefix}_assignments"] = assignments


def test_fit_schema_conditions_final_members_on_reopened_decisions(tmp_path) -> None:
    specs = iter2._base_fit_member_specs()
    arrays = _zero_arrays(specs)
    _add_decision_arrays(arrays, "proposed", accepted=False)
    _add_decision_arrays(arrays, "shuffled", accepted=True)
    for builder in (
        iter2._model_member_specs,
        iter2._fit_member_specs,
        iter2._geometry_member_specs,
    ):
        conditional_specs = builder("shuffled_final", iter2.N_GAUSSIANS)
        specs.update(conditional_specs)
        arrays.update(_zero_arrays(conditional_specs))
    checkpoint_specs = iter2._checkpoint_member_specs("shuffled", iter2.RECOVERY_UPDATES)
    specs.update(checkpoint_specs)
    arrays.update(_zero_arrays(checkpoint_specs))

    path = tmp_path / "conditional_fit.npz"
    descriptor = iter2._write_npz_exclusive(path, arrays)
    report, reopened, acceptance = iter2._load_fit_evidence_arrays(
        descriptor,
        expected_path=path,
    )
    assert report["pass"], report["errors"]
    assert acceptance == {"proposed": False, "shuffled": True}
    assert "proposed_final_means" not in report["required_members"]
    assert "shuffled_final_means" in report["required_members"]
    assert reopened["shuffled_final_means"].shape == (iter2.N_GAUSSIANS, 3)


def test_fit_input_evidence_includes_exact_camera_arrays() -> None:
    candidate, evaluator, _heldout_recipe, _receipt = iter2._build_root_inputs(
        95_001,
        95_101,
        95_201,
    )
    arrays = iter2._fit_input_arrays(candidate, evaluator)
    specs = iter2._fit_input_member_specs()
    for name in (
        "input_fit_camera_R",
        "input_fit_camera_t",
        "input_fit_camera_intrinsics",
        "input_fit_camera_image_sizes",
    ):
        array = arrays[name].detach().cpu().numpy()
        assert array.dtype.str == specs[name].dtype
        assert array.shape == specs[name].shape


def test_fit_input_duplicates_are_exactly_bound_to_committed_input() -> None:
    candidate, evaluator, _heldout_recipe, _receipt = iter2._build_root_inputs(
        95_002,
        95_102,
        95_202,
    )
    input_arrays = {
        name: iter2._as_numpy(value).copy()
        for name, value in iter2._fit_input_arrays(candidate, evaluator).items()
    }
    fit_arrays = {name: value.copy() for name, value in input_arrays.items()}
    assert all(iter2._fit_input_binding_checks(input_arrays, fit_arrays).values())

    fit_arrays["input_initial_depths"][0] += 0.25
    checks = iter2._fit_input_binding_checks(input_arrays, fit_arrays)
    assert not checks["fit_input_exactly_bound:input_initial_depths"]


def test_common_initialization_is_derived_from_committed_depths() -> None:
    candidate, _evaluator, _heldout_recipe, _receipt = iter2._build_root_inputs(
        95_003,
        95_103,
        95_203,
    )
    common = iter2._new_fiber(candidate)
    oracle = common.subset(torch.arange(iter2.N_GAUSSIANS, dtype=torch.long))
    arrays = {
        name: iter2._as_numpy(value).copy()
        for name, value in {
            **iter2._model_arrays("common_initial", common),
            **iter2._model_arrays("oracle_initial", oracle),
        }.items()
    }
    assert all(iter2._canonical_common_initial_checks(candidate, arrays).values())

    tampered_common = iter2._new_fiber(candidate)
    with torch.no_grad():
        tampered_common.depth_logits.add_(0.25)
    tampered_oracle = tampered_common.subset(torch.arange(iter2.N_GAUSSIANS, dtype=torch.long))
    arrays.update(
        {
            name: iter2._as_numpy(value).copy()
            for name, value in {
                **iter2._model_arrays("common_initial", tampered_common),
                **iter2._model_arrays("oracle_initial", tampered_oracle),
            }.items()
        }
    )
    reopened_common, common_checks = iter2._model_from_evidence("common_initial", arrays, candidate)
    reopened_oracle, oracle_checks = iter2._model_from_evidence("oracle_initial", arrays, candidate)
    pairing_still_self_consistent = all(
        torch.equal(left, right)
        for left, right in zip(
            reopened_common.subset(torch.arange(iter2.N_GAUSSIANS, dtype=torch.long))
            .state_dict()
            .values(),
            reopened_oracle.state_dict().values(),
            strict=True,
        )
    )
    assert all(common_checks.values())
    assert all(oracle_checks.values())
    assert pairing_still_self_consistent
    canonical = iter2._canonical_common_initial_checks(candidate, arrays)
    assert not canonical["canonical_input_initialization:common_initial_depth_logits"]


def test_input_commit_descriptor_is_bound_to_receipt_and_roots(tmp_path) -> None:
    roots = (95_004, 95_104, 95_204)
    candidate, evaluator, heldout_recipe, receipt = iter2._build_root_inputs(*roots)
    bundle = iter2.RootInputs(roots, candidate, evaluator, heldout_recipe, receipt)
    root_directory = tmp_path / f"scene_{roots[0]}"
    root_directory.mkdir()
    commit = iter2._commit_root_input(
        bundle,
        index=0,
        root_directory=root_directory,
        artifacts_directory=tmp_path,
        development=True,
        transaction_id=None,
        artifact_fd=None,
    )
    descriptor, checks = iter2._committed_input_binding(commit, roots=roots)
    assert all(checks.values())
    report, _arrays = iter2._load_input_evidence_arrays(
        descriptor,
        expected_path=root_directory / "input_evidence.npz",
    )
    assert report["pass"], report["errors"]

    bad_receipt = {
        **commit["receipt"],
        "input_evidence": {**commit["receipt"]["input_evidence"], "sha256": "0" * 64},
    }
    _descriptor, tampered_checks = iter2._committed_input_binding(
        {**commit, "receipt": bad_receipt},
        roots=roots,
    )
    assert not tampered_checks["input_commit_evidence_descriptor_exact"]
    _descriptor, swapped_checks = iter2._committed_input_binding(
        commit,
        roots=(roots[0] + 1, roots[1], roots[2]),
    )
    assert not swapped_checks["input_commit_root_bundle_exact"]
    assert not swapped_checks["input_commit_relative_path_exact"]


def test_owned_result_exchange_preserves_displaced_reservation(tmp_path) -> None:
    out = tmp_path / "result.json"
    reservation = {
        "schema": "reservation",
        "transaction_id": "a" * 32,
    }
    descriptor = iter2._write_json_exclusive(out, reservation)
    attempt = {
        "transaction_id": "a" * 32,
        "reservation_payload": reservation,
        "reservation_descriptor": descriptor,
    }
    result = {"schema": "result", "status": "FAIL"}
    result_descriptor, exchange = iter2._publish_reserved_result(
        out,
        result,
        attempt=attempt,
    )
    assert json.loads(out.read_bytes()) == result
    assert result_descriptor["sha256"] == iter2._sha256_bytes(iter2._canonical_bytes(result))
    assert exchange["accepted"]
    assert (
        json.loads((tmp_path / f".result.json.recovery.{'a' * 32}.prepared").read_bytes())
        == reservation
    )


def test_official_worker_requires_preimport_launcher(monkeypatch) -> None:
    monkeypatch.delenv("RTGS_ITER2_OFFICIAL_ATTEMPT", raising=False)
    monkeypatch.delenv("RTGS_ITER2_OFFICIAL_ATTEMPT_SHA256", raising=False)
    monkeypatch.delenv("RTGS_ITER2_OFFICIAL_ATTEMPT_PUBLIC_JSON", raising=False)
    with pytest.raises(RuntimeError, match="pre-import launcher context"):
        iter2._load_official_attempt()


def test_rejected_topology_keeps_component_representatives_and_reason_codes() -> None:
    decision = iter2.TopologyDecision(
        arm="test",
        scores=torch.arange(iter2.N_HYPOTHESES, dtype=torch.float64),
        retained=torch.tensor([True, True, True, False] + [False] * (iter2.N_HYPOTHESES - 4)),
        components=((0, 1), (2,)),
        representatives=torch.tensor([0, 2]),
        assignments=(),
        child=None,
        rejected=True,
        rejection_reason="component_count=2 expected=8",
    )
    arrays = iter2._decision_arrays("test", decision)
    assert arrays["test_component_ids"].tolist()[:4] == [0, 0, 1, -1]
    assert arrays["test_representative_for_hypothesis"].tolist()[:4] == [0, 0, 2, -1]
    assert arrays["test_decision_codes"].tolist()[:4] == [2, 1, 2, 0]
    assert iter2._decision_integrity(decision)["complete_rejection_receipt"]
