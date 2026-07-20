"""Focused CPU tests for the Iteration 3 unequal-decomposition harness."""

from __future__ import annotations

import dataclasses
import inspect
import json

import numpy as np
import pytest
import torch
from benchmarks import inverse_projection_fiber_iter3 as iter3


def _arm_state(
    model: iter3.InverseProjectionFiber,
    plans: tuple[iter3.CorrespondencePlan, ...],
    *,
    name: str = "uot_area",
) -> iter3.ArmState:
    return iter3.ArmState(
        name=name,
        model=model,
        plans=plans,
        history=(),
        wall_time_seconds=0.0,
        initial_state_sha256="test",
    )


def _uot_plan_from_augmented(
    augmented: torch.Tensor,
    track_capacities: torch.Tensor,
    observation_capacities: torch.Tensor,
) -> iter3.CorrespondencePlan:
    return iter3.CorrespondencePlan(
        real_mass=augmented[:-1, :-1],
        track_dustbin_mass=augmented[:-1, -1],
        observation_dustbin_mass=augmented[-1, :-1],
        dustbin_dustbin_mass=augmented[-1, -1],
        track_capacities=track_capacities,
        observation_capacities=observation_capacities,
        method="unbalanced_sinkhorn",
        iterations=50,
        fixed_point_residual=0.0,
    )


@pytest.mark.parametrize("count", [1, 2, 3])
def test_moment_split_preserves_first_and_second_moments(count: int) -> None:
    mean = torch.tensor([17.25, 29.75], dtype=torch.float64)
    covariance = torch.tensor([[5.0, 1.1], [1.1, 3.4]], dtype=torch.float64)
    means, covariances, check = iter3._moment_split(mean, covariance, count)
    recovered_mean = means.mean(dim=0)
    centered = means - recovered_mean
    recovered_covariance = covariances.mean(dim=0) + centered.T @ centered / count
    assert torch.allclose(recovered_mean, mean, atol=1e-12, rtol=0)
    assert torch.allclose(recovered_covariance, covariance, atol=1e-12, rtol=0)
    intrinsic = covariances - iter3.DILATION * torch.eye(2, dtype=torch.float64)
    assert torch.all(torch.linalg.eigvalsh(intrinsic) > 0)
    assert check["mean_max_abs"] <= 1e-12
    assert check["covariance_max_abs"] <= 1e-12


def test_mass_split_duplication_preserves_weighted_objective() -> None:
    single_cost = torch.tensor([[2.75]], dtype=torch.float64)
    single_mass = torch.tensor([[1.0]], dtype=torch.float64)
    duplicated_cost = torch.tensor([[2.75, 2.75]], dtype=torch.float64)
    duplicated_mass = torch.tensor([[0.35, 0.65]], dtype=torch.float64)
    single = iter3._mass_normalized_expected_cost(single_mass, single_cost)
    duplicated = iter3._mass_normalized_expected_cost(duplicated_mass, duplicated_cost)
    assert single is not None and duplicated is not None
    assert torch.equal(single, duplicated)


def test_uot_mass_diagnostics_accept_exact_1x1_and_reject_mass_destruction() -> None:
    root = iter3._build_root_inputs(90_001, 90_101, 90_201)
    model = iter3._new_fiber(root.candidate)
    track = torch.ones(1, dtype=torch.float64)
    observation = torch.ones(1, dtype=torch.float64)
    exact_augmented = torch.full((2, 2), 0.25, dtype=torch.float64)
    exact = _arm_state(
        model,
        (_uot_plan_from_augmented(exact_augmented, track, observation),),
    )
    exact_diagnostic = iter3._uot_mass_diagnostics(exact)
    assert exact_diagnostic["pass"] is True
    assert exact_diagnostic["views"][0]["realized_to_declared_ratios"] == {
        "track_row": 1.0,
        "observation_column": 1.0,
        "augmented": 1.0,
    }

    destroyed = _arm_state(
        model,
        (_uot_plan_from_augmented(0.01 * exact_augmented, track, observation),),
    )
    destroyed_diagnostic = iter3._uot_mass_diagnostics(destroyed)
    assert destroyed_diagnostic["pass"] is False
    assert destroyed_diagnostic["views"][0]["checks"]["mass_ratios_in_range"] is False


def test_root_construction_is_deterministic_unequal_and_label_free() -> None:
    first = iter3._build_root_inputs(91_001, 91_101, 91_201)
    second = iter3._build_root_inputs(91_001, 91_101, 91_201)
    assert [item.n for item in first.candidate.observations] == [17, 19, 18, 17, 19]
    assert first.candidate.source_means2d.shape == (90, 2)
    assert sorted(first.candidate.__dataclass_fields__) == first.receipt["candidate_fields"]
    assert all(
        "label" not in field and not field.startswith("gt")
        for field in first.candidate.__dataclass_fields__
    )
    assert first.receipt == second.receipt
    assert torch.equal(first.candidate.initial_depths, second.candidate.initial_depths)
    assert all(
        torch.equal(left.means, right.means) and torch.equal(left.covariances, right.covariances)
        for left, right in zip(
            first.candidate.observations, second.candidate.observations, strict=True
        )
    )
    assert all(
        int((labels < 0).sum()) == iter3.OUTLIERS_PER_VIEW
        for labels in first.evaluator.fit_parent_labels
    )
    assert (
        max(receipt["moment_covariance_max_abs"] for receipt in first.receipt["view_receipts"])
        <= 1e-10
    )
    prefit_arrays = iter3._candidate_input_arrays(first.candidate)
    assert all("label" not in name and not name.startswith("gt") for name in prefit_arrays)
    assert "root" not in inspect.signature(iter3._run_arm).parameters


def test_common_initialization_and_source_invariant_are_exact_across_arms() -> None:
    root = iter3._build_root_inputs(92_001, 92_101, 92_201)
    models = [iter3._new_fiber(root.candidate) for _ in iter3.ARM_NAMES]
    reference = models[0].state_dict()
    for model in models[1:]:
        assert all(torch.equal(reference[name], model.state_dict()[name]) for name in reference)
    center, covariance = iter3._source_errors(models[0])
    assert center <= 1e-8
    assert covariance <= 1e-8
    depths = models[0].depths()
    assert torch.all(depths > iter3.DEPTH_LOWER)
    assert torch.all(depths < iter3.DEPTH_UPPER)


def test_hard_row_and_transport_exclude_each_tracks_source_view() -> None:
    root = iter3._build_root_inputs(93_001, 93_101, 93_201)
    config = iter3.ScientificConfig(
        outer_steps=1,
        geometry_steps=1,
        sinkhorn_iterations=4,
    )
    for arm in ("hardmin", "row", "uot_uniform", "uot_area"):
        state = iter3._run_arm(arm, root.candidate, config)
        for view, plan in enumerate(state.plans):
            own_source = state.model.source_view_indices == view
            assert torch.count_nonzero(plan.real_mass[own_source]) == 0
            assert torch.all(plan.track_support[own_source] == 0)
        center, covariance = iter3._source_errors(state.model)
        assert center <= 1e-8
        assert covariance <= 1e-8
        assert torch.all(state.model.depths() > iter3.DEPTH_LOWER)
        assert torch.all(state.model.depths() < iter3.DEPTH_UPPER)


def test_completeness_rejects_epsilon_correct_support() -> None:
    root = iter3._build_root_inputs(93_003, 93_103, 93_203)
    model = iter3._new_fiber(root.candidate)
    source_labels = root.evaluator.source_parent_labels
    epsilon = 1e-6
    plans: list[iter3.CorrespondencePlan] = []
    for view, target_labels in enumerate(root.evaluator.fit_parent_labels):
        active = model.source_view_indices != view
        capacities = active.to(dtype=torch.float64)
        real = torch.zeros((model.n, target_labels.numel()), dtype=torch.float64)
        for row in (active & (source_labels >= 0)).nonzero(as_tuple=True)[0].tolist():
            columns = target_labels == int(source_labels[row])
            real[row, columns] = epsilon / int(columns.sum())
        plans.append(
            iter3.CorrespondencePlan(
                real_mass=real,
                track_dustbin_mass=capacities - real.sum(dim=1),
                observation_dustbin_mass=None,
                dustbin_dustbin_mass=None,
                track_capacities=capacities,
                observation_capacities=None,
                method="row_softmax",
                iterations=1,
                fixed_point_residual=0.0,
            )
        )
    metrics = iter3._association_metrics(
        _arm_state(model, tuple(plans), name="row"),
        root,
        root.evaluator.fit_parent_labels,
    )
    assert metrics["parent_purity"] == pytest.approx(1.0)
    assert metrics["parent_completeness"] == 0.0
    support = np.asarray(metrics["capacity_normalized_correct_support_by_parent_view"])
    assert np.max(support) == pytest.approx(epsilon)


def test_declared_capacity_dust_recall_rejects_annihilated_outlier_mass() -> None:
    root = iter3._build_root_inputs(93_004, 93_104, 93_204)
    model = iter3._new_fiber(root.candidate)
    source_labels = root.evaluator.source_parent_labels
    epsilon = 1e-6
    plans: list[iter3.CorrespondencePlan] = []
    for view, target_labels in enumerate(root.evaluator.fit_parent_labels):
        track = (model.source_view_indices != view).to(dtype=torch.float64)
        observation = torch.ones(target_labels.numel(), dtype=torch.float64)
        normalizer = track.sum() + observation.sum()
        row_target = torch.cat([track, observation.sum().reshape(1)]) / normalizer
        column_target = torch.cat([observation, track.sum().reshape(1)]) / normalizer
        augmented = row_target[:, None] * column_target[None, :]
        outlier_rows = (source_labels < 0) & (track > 0)
        outlier_row_indices = outlier_rows.nonzero(as_tuple=True)[0]
        augmented[outlier_row_indices] = 0.0
        augmented[outlier_row_indices, -1] = epsilon * row_target[outlier_row_indices]
        outlier_columns = target_labels < 0
        outlier_column_indices = outlier_columns.nonzero(as_tuple=True)[0]
        augmented[:, outlier_column_indices] = 0.0
        augmented[-1, outlier_column_indices] = epsilon * column_target[outlier_column_indices]
        plans.append(_uot_plan_from_augmented(augmented, track, observation))
    state = _arm_state(model, tuple(plans))
    diagnostics = iter3._uot_mass_diagnostics(state)
    metrics = iter3._association_metrics(
        state,
        root,
        root.evaluator.fit_parent_labels,
    )
    assert diagnostics["pass"] is True
    assert metrics["outlier_track_dust_recall"] == pytest.approx(epsilon)
    assert metrics["outlier_observation_dust_recall"] == pytest.approx(epsilon)
    assert metrics["realized_conditional_outlier_track_dust_fraction_by_view"] == pytest.approx(
        [1.0] * iter3.N_FIT_VIEWS
    )
    assert metrics[
        "realized_conditional_outlier_observation_dust_fraction_by_view"
    ] == pytest.approx([1.0] * iter3.N_FIT_VIEWS)


def test_transport_release_checks_each_dust_route_on_every_root() -> None:
    def results_with(**overrides: float) -> list[dict[str, object]]:
        association = {
            "parent_purity": 0.95,
            "parent_completeness": 0.95,
            "outlier_track_dust_recall": 0.90,
            "outlier_observation_dust_recall": 0.90,
            "inlier_track_dust_false_positive_rate": 0.10,
            "inlier_observation_dust_false_positive_rate": 0.10,
        }
        roots: list[dict[str, object]] = []
        for index in range(3):
            values = dict(association)
            if index == 1:
                values.update(overrides)
            roots.append(
                {
                    "arms": {
                        "uot_area": {
                            "association": values,
                            "transport_mass_diagnostics": {"pass": True},
                        }
                    }
                }
            )
        return roots

    assert iter3._transport_arm_accepted(results_with(), "uot_area") is True
    assert (
        iter3._transport_arm_accepted(results_with(outlier_track_dust_recall=0.79), "uot_area")
        is False
    )
    assert (
        iter3._transport_arm_accepted(
            results_with(outlier_observation_dust_recall=0.79), "uot_area"
        )
        is False
    )
    assert (
        iter3._transport_arm_accepted(
            results_with(inlier_track_dust_false_positive_rate=0.21), "uot_area"
        )
        is False
    )
    assert (
        iter3._transport_arm_accepted(
            results_with(inlier_observation_dust_false_positive_rate=0.21), "uot_area"
        )
        is False
    )


def test_all_arms_share_projection_validity_domain(monkeypatch) -> None:
    root = iter3._build_root_inputs(93_002, 93_102, 93_202)
    config = iter3.ScientificConfig(
        outer_steps=1,
        geometry_steps=1,
        sinkhorn_iterations=4,
    )
    original_project = iter3.InverseProjectionFiber.project

    def invalidate_first_row(self, camera):
        projection = original_project(self, camera)
        means = projection.means2d.clone()
        means[0] = torch.nan
        return dataclasses.replace(projection, means2d=means)

    monkeypatch.setattr(iter3.InverseProjectionFiber, "project", invalidate_first_row)
    for arm in iter3.ARM_NAMES:
        state = iter3._run_arm(
            arm,
            root.candidate,
            config,
            oracle_evaluator=root.evaluator if arm == "oracle" else None,
        )
        for view, plan in enumerate(state.plans):
            assert torch.count_nonzero(plan.real_mass[0]) == 0
            if view != int(state.model.source_view_indices[0]) and arm in {
                "row",
                "uot_uniform",
                "uot_area",
                "oracle",
                "shuffled_view",
            }:
                assert plan.track_dustbin_mass[0] > 0


def test_custom_optimizer_validates_every_step_and_final_state(monkeypatch) -> None:
    root = iter3._build_root_inputs(93_005, 93_105, 93_205)
    config = iter3.ScientificConfig(
        outer_steps=1,
        geometry_steps=2,
        sinkhorn_iterations=4,
    )
    original_validate = iter3.validate_fiber_state
    validated_state_hashes: list[str] = []

    def record_validation(model):
        original_validate(model)
        validated_state_hashes.append(iter3._tensor_mapping_hash(dict(model.state_dict())))

    monkeypatch.setattr(iter3, "validate_fiber_state", record_validation)
    iter3._run_arm("hardmin", root.candidate, config)
    assert len(validated_state_hashes) >= config.geometry_steps + 2
    assert validated_state_hashes[-1] == validated_state_hashes[-2]


def test_oracle_is_exact_and_shuffled_control_consumes_next_view() -> None:
    root = iter3._build_root_inputs(94_001, 94_101, 94_201)
    config = iter3.ScientificConfig(
        outer_steps=1,
        geometry_steps=1,
        sinkhorn_iterations=4,
    )
    oracle = iter3._run_arm(
        "oracle",
        root.candidate,
        config,
        oracle_evaluator=root.evaluator,
    )
    metrics = iter3._association_metrics(
        oracle,
        root,
        root.evaluator.fit_parent_labels,
    )
    assert metrics["parent_purity"] == pytest.approx(1.0)
    assert metrics["parent_completeness"] == pytest.approx(1.0)
    assert metrics["outlier_dust_recall"] == pytest.approx(1.0)

    shuffled = iter3._run_arm("shuffled_view", root.candidate, config)
    shuffled_labels = iter3._released_target_labels("shuffled_view", root.evaluator)
    for view in range(iter3.N_FIT_VIEWS):
        expected = root.evaluator.fit_parent_labels[(view + 1) % iter3.N_FIT_VIEWS]
        assert torch.equal(shuffled_labels[view], expected)
        assert shuffled.plans[view].real_mass.shape[1] == expected.numel()


def test_heldout_materialization_is_late_and_deterministic() -> None:
    root = iter3._build_root_inputs(95_001, 95_101, 95_201)
    assert root.receipt["heldout_materialized"] is False
    assert all("heldout" not in field for field in root.evaluator.__dataclass_fields__)
    source = inspect.getsource(iter3._build_root_inputs)
    assert "_materialize_heldout" not in source
    first = iter3._materialize_heldout(root)
    second = iter3._materialize_heldout(root)
    assert first.receipt == second.receipt
    assert len(first.cameras) == iter3.N_HELDOUT_VIEWS
    assert all(
        torch.equal(left.means, right.means)
        for left, right in zip(first.observations, second.observations, strict=True)
    )


def test_heldout_invalid_projection_is_excluded_not_assigned(monkeypatch) -> None:
    root = iter3._build_root_inputs(95_002, 95_102, 95_202)
    model = iter3._new_fiber(root.candidate)
    state = _arm_state(model, (), name="hardmin")
    heldout = iter3._materialize_heldout(root)
    invalid_row = int((root.evaluator.source_parent_labels >= 0).nonzero(as_tuple=True)[0][0])
    original_project = iter3.InverseProjectionFiber.project
    baseline_metrics, _baseline_arrays = iter3._heldout_metrics(state, root, heldout)
    for view, camera in enumerate(heldout.cameras):
        projection = original_project(model, camera)
        _means, _covariances, valid = iter3.safe_projection_geometry(camera, projection)
        assert bool(valid[invalid_row]), f"chosen row is naturally invalid in held-out view {view}"

    def invalidate_one_row(self, camera):
        projection = original_project(self, camera)
        means = projection.means2d.clone()
        means[invalid_row] = torch.nan
        return dataclasses.replace(projection, means2d=means)

    monkeypatch.setattr(iter3.InverseProjectionFiber, "project", invalidate_one_row)
    metrics, arrays = iter3._heldout_metrics(state, root, heldout)
    assert metrics["heldout_invalid_projection_incidence_by_view"] == [
        value + 1 for value in baseline_metrics["heldout_invalid_projection_incidence_by_view"]
    ]
    assert metrics["heldout_invalid_inlier_projection_incidence_by_view"] == [
        value + 1
        for value in baseline_metrics["heldout_invalid_inlier_projection_incidence_by_view"]
    ]
    assert metrics["heldout_assignment_denominator_by_view"] == [
        value - 1 for value in baseline_metrics["heldout_assignment_denominator_by_view"]
    ]
    for view in range(iter3.N_HELDOUT_VIEWS):
        assert int(arrays[f"heldout_view_{view}_assignment"][invalid_row]) == -1
        assert not bool(arrays[f"heldout_view_{view}_projection_valid"][invalid_row])


def test_nonoracle_frozen_hashes_are_invariant_to_poisoned_labels(tmp_path) -> None:
    root = iter3._build_root_inputs(95_003, 95_103, 95_203)

    def permute(labels: torch.Tensor) -> torch.Tensor:
        return torch.where(labels >= 0, (labels + 3) % iter3.N_PARENTS, labels)

    poisoned_evaluator = dataclasses.replace(
        root.evaluator,
        fit_parent_labels=tuple(permute(labels) for labels in root.evaluator.fit_parent_labels),
        source_parent_labels=permute(root.evaluator.source_parent_labels),
    )
    poisoned_root = dataclasses.replace(root, evaluator=poisoned_evaluator)
    config = iter3.ScientificConfig(
        outer_steps=1,
        geometry_steps=1,
        sinkhorn_iterations=4,
    )
    clean = iter3._run_root(root, tmp_path / "clean", config, ("hardmin",))
    poisoned = iter3._run_root(
        poisoned_root,
        tmp_path / "poisoned",
        config,
        ("hardmin",),
    )
    assert (
        clean["nonoracle_state_hashes_before_evaluator"]
        == poisoned["nonoracle_state_hashes_before_evaluator"]
    )
    assert (
        clean["nonoracle_plan_hashes_before_evaluator"]
        == poisoned["nonoracle_plan_hashes_before_evaluator"]
    )


def test_development_smoke_writes_json_npz_and_viewer_ready_plys(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    result = tmp_path / "result.json"
    config = iter3.ScientificConfig(
        outer_steps=1,
        geometry_steps=1,
        sinkhorn_iterations=4,
    )
    payload = iter3.run_experiment(
        mode="development",
        artifacts_dir=artifacts,
        result_path=result,
        config=config,
        roots=((96_001, 96_101, 96_201),),
    )
    on_disk = json.loads(result.read_bytes())
    assert on_disk["namespace"] == iter3.NAMESPACE
    assert on_disk["mode"] == "development"
    assert payload["synthetic_gates"]["eligible"] is False
    assert payload["real_release"]["permitted"] is False
    root_directory = artifacts / "root_0"
    expected_plys = {"gaussians_init.ply", *(f"{arm}.ply" for arm in iter3.ARM_NAMES)}
    assert expected_plys.issubset({path.name for path in root_directory.glob("*.ply")})
    evidence_path = root_directory / "evidence.npz"
    with np.load(evidence_path, allow_pickle=False) as evidence:
        assert "common_initial_means" in evidence
        assert "uot_area_view_0_real_mass" in evidence
        assert "uot_area_view_0_augmented_target_row_marginal" in evidence
        assert "uot_area_view_0_augmented_target_column_marginal" in evidence
        assert "heldout_view_0_parent_labels" in evidence
        assert int(evidence["heldout_release_marker"][0]) == 1
    assert payload["environment"]["torch_num_threads"] > 0
    assert payload["environment"]["torch_num_interop_threads"] > 0
    assert len(payload["environment"]["load_average_1m_5m_15m"]) == 3
    for arm in iter3.ARM_NAMES:
        assert payload["root_results"][0]["arms"][arm]["geometry"]["finite_spd"]


def test_official_mode_is_fail_closed_and_development_refuses_official_roots(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="confirm-official-roots"):
        iter3.run_experiment(
            mode="official",
            artifacts_dir=iter3.OFFICIAL_ARTIFACTS,
            result_path=iter3.OFFICIAL_RESULT,
        )
    with pytest.raises(RuntimeError, match="development mode refuses official roots"):
        iter3.run_experiment(
            mode="development",
            artifacts_dir=tmp_path / "artifacts",
            result_path=tmp_path / "result.json",
            roots=(iter3.OFFICIAL_ROOT_TUPLES[0],),
        )
    assert not (tmp_path / "artifacts").exists()


def test_official_crash_after_attempt_is_durable_and_nonreusable(tmp_path, monkeypatch) -> None:
    attempt = tmp_path / "ATTEMPT.json"
    result = tmp_path / "RESULT.json"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(iter3, "OFFICIAL_ATTEMPT", attempt)
    monkeypatch.setattr(iter3, "OFFICIAL_RESULT", result)
    monkeypatch.setattr(iter3, "OFFICIAL_ARTIFACTS", artifacts)
    build_calls = 0

    def crash_before_root_construction(*_args):
        nonlocal build_calls
        build_calls += 1
        raise RuntimeError("injected crash after attempt reservation")

    monkeypatch.setattr(iter3, "_build_root_inputs", crash_before_root_construction)
    with pytest.raises(RuntimeError, match="injected crash"):
        iter3.run_experiment(
            mode="official",
            artifacts_dir=artifacts,
            result_path=result,
            confirm_official_roots=True,
        )
    receipt = json.loads(attempt.read_bytes())
    assert receipt["status"] == "ATTEMPTED"
    assert receipt["all_nine_roots"] == [
        value for roots in iter3.OFFICIAL_ROOT_TUPLES for value in roots
    ]
    assert receipt["artifacts_directory"] == str(artifacts)
    assert receipt["result_path"] == str(result)
    assert set(receipt["source_hashes"]) == {str(path) for path in iter3.EXECUTED_SOURCE_PATHS}
    assert not result.exists()
    assert artifacts.is_dir()
    assert build_calls == 1

    with pytest.raises(RuntimeError, match="attempt/result already exists"):
        iter3.run_experiment(
            mode="official",
            artifacts_dir=artifacts,
            result_path=result,
            confirm_official_roots=True,
        )
    assert attempt.exists()
    assert build_calls == 1


def test_run_refuses_executed_source_change(tmp_path, monkeypatch) -> None:
    calls = 0
    original = iter3._source_hashes

    def changing_hashes() -> dict[str, str]:
        nonlocal calls
        calls += 1
        hashes = original()
        if calls > 1:
            hashes = dict(hashes)
            hashes["benchmarks/inverse_projection_fiber_iter3.py"] = "0" * 64
        return hashes

    monkeypatch.setattr(iter3, "_source_hashes", changing_hashes)
    monkeypatch.setattr(iter3, "_build_root_inputs", lambda *_roots: object())
    monkeypatch.setattr(iter3, "_run_root", lambda *_args: {"arms": {}})
    monkeypatch.setattr(iter3, "_synthetic_gates", lambda _results: {"eligible": False})
    monkeypatch.setattr(iter3, "_real_release", lambda _gates, _results: {"permitted": False})

    result = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="sources changed"):
        iter3.run_experiment(
            mode="development",
            artifacts_dir=tmp_path / "artifacts",
            result_path=result,
            roots=((101, 102, 103),),
        )

    assert not result.exists()
