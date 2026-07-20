"""CPU mechanism tests for RGB-free compact-field 2D-to-3D initialization."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    _GroupedObservationIndexReference,
)
from rtgs.core.sh import sh_to_rgb
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import (
    CompactCandidateAudit,
    CompactCarveConfig,
    CompactCarveInitializer,
    CompactRayDepthAuditBatch,
    _bounds_source,
    _solve_projector_center,
    score_world_points,
)

_TARGETS = torch.tensor(
    [
        [-0.18, -0.16, 0.0],
        [0.18, -0.16, 0.0],
        [-0.18, 0.16, 0.0],
        [0.18, 0.16, 0.0],
    ]
)
_COLORS = torch.tensor(
    [
        [0.85, 0.15, 0.15],
        [0.15, 0.80, 0.20],
        [0.15, 0.25, 0.90],
        [0.80, 0.75, 0.15],
    ]
)


def _camera(x: float) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([x, 0.0, -3.0]),
        target=torch.zeros(3),
        width=32,
        height=32,
        fov_x_deg=55.0,
    )


def _field(camera: Camera, view_id: str, *, n_init: int) -> GaussianObservationField:
    means, depth = camera.project(_TARGETS)
    assert bool((depth > 0).all())
    return GaussianObservationField(
        width=camera.width,
        height=camera.height,
        means=means,
        log_scales=torch.log(torch.full((len(_TARGETS), 2), 0.85)),
        rotations=torch.tensor([0.0, 0.2, -0.15, 0.35]),
        colors=_COLORS,
        amplitudes=torch.tensor([0.9, 0.8, 0.7, 1.0]),
        view_id=view_id,
        n_init=n_init,
    )


def _inputs(*, n_init: tuple[int, int] = (2, 3)) -> ReconstructionInputs:
    cameras = [_camera(-0.75), _camera(0.75)]
    return ReconstructionInputs(
        observations=[
            _field(cameras[0], "left", n_init=n_init[0]),
            _field(cameras[1], "right", n_init=n_init[1]),
        ],
        cameras=cameras,
        view_names=["left", "right"],
        bounds_hint=(torch.zeros(3), 1.2),
        name="compact-plane",
    )


def _config(n_init_3d: int = 4, *, query_batch_size: int = 256) -> CompactCarveConfig:
    return CompactCarveConfig(
        n_init_3d=n_init_3d,
        candidate_multiplier=8,
        samples_per_ray=32,
        query_batch_size=query_batch_size,
        seed=17,
        min_views=2,
        hull_fraction=1.0,
        coverage_scale=4.0,
        coverage_threshold=0.20,
        color_std_sigma=0.25,
        min_score=0.01,
    )


def _duplicate_component(
    field: GaussianObservationField,
    component_id: int,
) -> GaussianObservationField:
    def split(values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                values[:component_id],
                values[component_id : component_id + 1].repeat((2,) + (1,) * (values.ndim - 1)),
                values[component_id + 1 :],
            ]
        )

    amplitudes = split(field.amplitudes)
    amplitudes[component_id : component_id + 2] *= 0.5
    return replace(
        field,
        means=split(field.means),
        log_scales=split(field.log_scales),
        rotations=split(field.rotations),
        colors=split(field.colors),
        amplitudes=amplitudes,
    )


def _replace_observations(
    inputs: ReconstructionInputs,
    observations: list[GaussianObservationField],
) -> ReconstructionInputs:
    return ReconstructionInputs(
        observations=observations,
        cameras=inputs.cameras,
        view_names=inputs.view_names,
        points=inputs.points,
        point_visibility=inputs.point_visibility,
        bounds_hint=inputs.bounds_hint,
        name=inputs.name,
    )


def _residual_bearing_inputs() -> ReconstructionInputs:
    inputs = _inputs()
    observations = []
    for view_index, field in enumerate(inputs.observations):
        sign = 1.0 if view_index == 0 else -1.0
        residuals = torch.tensor(
            [
                [0.03125, -0.015625],
                [-0.046875, 0.0234375],
                [0.015625, 0.0390625],
                [-0.0234375, -0.03125],
            ],
            dtype=torch.float32,
        )
        observations.append(
            replace(
                field,
                fit_window=(1, 1, field.width - 2, field.height - 2),
                mean_residuals=sign * residuals,
            )
        )
    return _replace_observations(inputs, observations)


def test_exact_3d_budget_is_deterministic_and_independent_of_2d_init_metadata():
    inputs = _inputs()
    config = _config()
    first = CompactCarveInitializer(config).initialize(inputs)
    repeated = CompactCarveInitializer(config).initialize(inputs)
    assert first.n_init_3d == config.n_init_3d
    assert torch.equal(first.gaussians.means, repeated.gaussians.means)
    assert torch.equal(first.gaussians.log_scales, repeated.gaussians.log_scales)
    assert torch.equal(first.scores, repeated.scores)
    assert torch.equal(
        first.lineage.source_component_indices,
        repeated.lineage.source_component_indices,
    )
    assert first.diagnostics["bounds_source"] == "explicit_hint"
    assert first.diagnostics["bounds_center"] == [0.0, 0.0, 0.0]
    assert first.diagnostics["bounds_extent"] == 1.2
    assert first.diagnostics["search_aabb_lower"] == pytest.approx([-0.6, -0.6, -0.6])
    assert first.diagnostics["search_aabb_upper"] == pytest.approx([0.6, 0.6, 0.6])

    metadata_only = _replace_observations(
        inputs,
        [replace(inputs.observations[0], n_init=41), replace(inputs.observations[1], n_init=73)],
    )
    changed_metadata = CompactCarveInitializer(config).initialize(metadata_only)
    assert torch.equal(first.gaussians.means, changed_metadata.gaussians.means)
    assert torch.equal(first.scores, changed_metadata.scores)
    assert changed_metadata.diagnostics["n_init_2d"] == [41, 73]
    assert changed_metadata.diagnostics["n_opt_2d"] == [4, 4]


def test_bounds_source_reports_the_geometry_path_actually_used():
    inputs = _inputs()
    assert _bounds_source(inputs) == "explicit_hint"
    assert _bounds_source(replace(inputs, bounds_hint=None)) == "camera_axis_fallback"
    sparse = replace(
        inputs,
        bounds_hint=None,
        points=torch.tensor(
            [
                [-0.1, -0.1, 0.0],
                [0.1, -0.1, 0.0],
                [-0.1, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ]
        ),
    )
    assert _bounds_source(sparse) == "sparse_points"
    assert _bounds_source(replace(sparse, points=sparse.points[:3])) == "camera_axis_fallback"

    smaller = CompactCarveInitializer(_config(n_init_3d=2)).initialize(inputs)
    assert smaller.n_init_3d == 2
    assert inputs.n_opt_2d == [4, 4]


def test_projector_center_full_rank_is_repeatable_and_gels_equivalent():
    matrix = torch.tensor(
        [
            [6.2393332, 0.38290277, -0.49555126],
            [0.38290277, 3.9799309, -0.17785272],
            [-0.49555126, -0.17785272, 3.7807350],
        ]
    )
    vector = torch.tensor([-1.3665650, 0.09661019, 9.058019])
    expected = torch.linalg.lstsq(matrix, vector, driver="gels").solution

    results = [_solve_projector_center(matrix, vector) for _ in range(32)]

    assert all(torch.equal(result, expected) for result in results)


def test_projector_center_parallel_rank_deficiency_uses_minimum_norm_solution():
    matrix = torch.diag(torch.tensor([7.0, 7.0, 0.0]))
    vector = torch.tensor([0.0, 7.0, 0.0])

    center = _solve_projector_center(matrix, vector)

    assert torch.equal(center, torch.tensor([0.0, 1.0, 0.0]))


def test_projector_center_near_degenerate_solution_is_bounded_and_repeatable():
    matrix = torch.tensor(
        [
            [7.0, 0.0, 0.0],
            [0.0, 7.0, 3.6379788e-12],
            [0.0, 3.6379788e-12, 0.0],
        ]
    )
    vector = torch.tensor([0.7, -1.4, 1.1200063e-7])

    results = [_solve_projector_center(matrix, vector) for _ in range(32)]

    assert all(torch.equal(result, results[0]) for result in results)
    assert bool(torch.isfinite(results[0]).all())
    assert float(results[0].abs().max()) < 1.0
    assert torch.allclose(results[0][:2], torch.tensor([0.1, -0.2]))
    assert abs(float(results[0][2])) < 1e-10


def test_projector_center_lstsq_drivers_are_always_explicit(monkeypatch):
    original = torch.linalg.lstsq
    drivers: list[str | None] = []

    def tracked_lstsq(*args, driver=None, **kwargs):
        drivers.append(driver)
        return original(*args, driver=driver, **kwargs)

    monkeypatch.setattr(torch.linalg, "lstsq", tracked_lstsq)
    full_rank = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    _solve_projector_center(full_rank, torch.ones(3))
    rank_deficient = torch.diag(torch.tensor([3.0, 2.0, 0.0]))
    _solve_projector_center(rank_deficient, torch.ones(3))

    assert drivers == ["gelsd", "gels", "gelsd"]


def test_component_center_anchors_use_every_variable_2d_identity_once():
    inputs = _inputs()
    inputs = _replace_observations(
        inputs,
        [_duplicate_component(inputs.observations[0], 1), inputs.observations[1]],
    )
    config = replace(_config(), anchor_mode="component_centers")
    result = CompactCarveInitializer(config).initialize(inputs)

    assert result.diagnostics["anchor_mode"] == "component_centers"
    assert result.diagnostics["candidate_count"] == 9
    assert result.diagnostics["proposed_candidates_per_view"] == [5, 4]
    assert result.diagnostics["anchor_attempt_count"] == 9
    for output_id in range(result.n_init_3d):
        view_id = int(result.lineage.source_view_indices[output_id])
        component_id = int(result.lineage.source_component_indices[output_id])
        assert torch.equal(
            result.lineage.source_xy[output_id],
            inputs.observations[view_id].means[component_id],
        )


def test_candidate_audit_records_every_component_center_once_and_exact_selection():
    inputs = _inputs()
    inputs = _replace_observations(
        inputs,
        [_duplicate_component(inputs.observations[0], 1), inputs.observations[1]],
    )
    config = replace(_config(), anchor_mode="component_centers")
    audits: list[CompactCandidateAudit] = []

    result = CompactCarveInitializer(config).initialize(
        inputs,
        candidate_audit_callback=audits.append,
    )

    assert len(audits) == 1
    audit = audits[0]
    expected_identities = [
        (view_index, component_index)
        for view_index, field in enumerate(inputs.observations)
        for component_index in range(field.n)
    ]
    actual_identities = list(
        zip(
            audit.candidate_source_view_indices.tolist(),
            audit.candidate_source_component_indices.tolist(),
            strict=True,
        )
    )
    assert actual_identities == expected_identities
    expected_xy = torch.cat(
        [field.native_means(dtype=torch.float64) for field in inputs.observations]
    )
    assert torch.equal(audit.candidate_source_xy, expected_xy)

    selected = audit.selected_candidate_indices
    assert torch.equal(
        audit.candidate_source_view_indices[selected],
        result.lineage.source_view_indices,
    )
    assert torch.equal(
        audit.candidate_source_component_indices[selected],
        result.lineage.source_component_indices,
    )
    assert torch.equal(audit.candidate_source_xy[selected], result.lineage.source_xy)
    assert torch.equal(audit.candidate_best_depths[selected], result.depths)
    assert torch.equal(audit.candidate_depth_sigmas[selected], result.depth_sigmas)
    assert torch.equal(audit.candidate_best_means[selected], result.gaussians.means)
    assert torch.equal(audit.candidate_best_scores[selected], result.scores)
    assert torch.equal(
        audit.candidate_consensus_colors[selected].clamp(0.0, 1.0),
        sh_to_rgb(result.gaussians.sh[:, 0]),
    )
    assert bool(audit.candidate_valid_mask[selected].all())
    assert bool(audit.candidate_eligible_mask[selected].all())


def test_component_center_anchors_preserve_residual_corrected_native_means():
    inputs = _residual_bearing_inputs()
    audits: list[CompactCandidateAudit] = []

    result = CompactCarveInitializer(
        replace(_config(), anchor_mode="component_centers")
    ).initialize(
        inputs,
        candidate_audit_callback=audits.append,
    )

    stored = torch.cat([field.means.double() for field in inputs.observations])
    native = torch.cat([field.native_means(dtype=torch.float64) for field in inputs.observations])
    assert not torch.equal(stored, native)
    assert torch.equal(audits[0].candidate_source_xy, native)
    selected = audits[0].selected_candidate_indices
    assert torch.equal(result.lineage.source_xy, native[selected])


def test_ray_depth_audit_streams_exact_order_shapes_and_winner_statistics():
    inputs = _residual_bearing_inputs()
    config = replace(
        _config(query_batch_size=64),
        anchor_mode="component_centers",
    )
    depth_batches: list[CompactRayDepthAuditBatch] = []
    candidate_audits: list[CompactCandidateAudit] = []

    result = CompactCarveInitializer(config).initialize(
        inputs,
        candidate_audit_callback=candidate_audits.append,
        ray_depth_audit_callback=depth_batches.append,
    )

    assert len(candidate_audits) == 1
    assert len(depth_batches) == 4
    expected_start = 0
    for batch in depth_batches:
        assert batch.candidate_start == expected_start
        assert batch.candidate_end > batch.candidate_start
        batch_size = batch.candidate_end - batch.candidate_start
        assert batch.candidate_source_view_indices.shape == (batch_size,)
        assert batch.candidate_source_component_indices.shape == (batch_size,)
        assert batch.candidate_source_xy.shape == (batch_size, 2)
        assert batch.candidate_valid_mask.shape == (batch_size,)
        assert batch.depths.shape == (batch_size, config.samples_per_ray)
        assert batch.scores.shape == batch.depths.shape
        assert batch.coverages.shape == batch.depths.shape
        assert batch.color_variances.shape == batch.depths.shape
        assert batch.n_seen.shape == batch.depths.shape
        assert batch.n_covered.shape == batch.depths.shape
        assert batch.consensus_colors.shape == (
            batch_size,
            config.samples_per_ray,
            3,
        )
        assert batch.n_seen.dtype == batch.n_covered.dtype == torch.long
        expected_start = batch.candidate_end
    audit = candidate_audits[0]
    assert expected_start == audit.candidate_source_view_indices.numel()

    view_ids = torch.cat([batch.candidate_source_view_indices for batch in depth_batches])
    component_ids = torch.cat([batch.candidate_source_component_indices for batch in depth_batches])
    source_xy = torch.cat([batch.candidate_source_xy for batch in depth_batches])
    valid = torch.cat([batch.candidate_valid_mask for batch in depth_batches])
    depths = torch.cat([batch.depths for batch in depth_batches])
    scores = torch.cat([batch.scores for batch in depth_batches])
    coverages = torch.cat([batch.coverages for batch in depth_batches])
    variances = torch.cat([batch.color_variances for batch in depth_batches])
    n_seen = torch.cat([batch.n_seen for batch in depth_batches])
    n_covered = torch.cat([batch.n_covered for batch in depth_batches])
    colors = torch.cat([batch.consensus_colors for batch in depth_batches])

    assert torch.equal(view_ids, audit.candidate_source_view_indices)
    assert torch.equal(component_ids, audit.candidate_source_component_indices)
    assert torch.equal(source_xy, audit.candidate_source_xy)
    assert torch.equal(valid, audit.candidate_valid_mask)
    best_indices = scores.argmax(dim=1)
    row = torch.arange(scores.shape[0])
    assert torch.equal(best_indices, audit.candidate_best_depth_indices)
    assert torch.equal(depths[row, best_indices], audit.candidate_best_depths)
    assert torch.equal(scores[row, best_indices], audit.candidate_best_scores)
    assert torch.equal(coverages[row, best_indices], audit.candidate_best_coverages)
    assert torch.equal(
        variances[row, best_indices],
        audit.candidate_best_color_variances,
    )
    assert torch.equal(n_seen[row, best_indices], audit.candidate_best_n_seen)
    assert torch.equal(n_covered[row, best_indices], audit.candidate_best_n_covered)
    assert torch.equal(colors[row, best_indices], audit.candidate_consensus_colors)
    without_best = scores.clone()
    without_best[row, best_indices] = -torch.inf
    second_best = without_best.max(dim=1).values
    assert torch.equal(second_best, audit.candidate_second_best_scores)
    assert torch.equal(
        audit.candidate_best_scores - second_best,
        audit.candidate_score_margins,
    )
    assert bool((audit.candidate_half_max_widths >= 0).all())

    selected = audit.selected_candidate_indices
    assert torch.equal(best_indices[selected], audit.candidate_best_depth_indices[selected])
    assert torch.equal(depths[selected, best_indices[selected]], result.depths)
    assert torch.equal(scores[selected, best_indices[selected]], result.scores)


def _assert_initializations_equal(
    actual,
    expected,
):
    assert torch.equal(actual.gaussians.means, expected.gaussians.means)
    assert torch.equal(actual.gaussians.quats, expected.gaussians.quats)
    assert torch.equal(actual.gaussians.log_scales, expected.gaussians.log_scales)
    assert torch.equal(actual.gaussians.opacity, expected.gaussians.opacity)
    assert torch.equal(actual.gaussians.sh, expected.gaussians.sh)
    assert torch.equal(
        actual.lineage.source_view_indices,
        expected.lineage.source_view_indices,
    )
    assert torch.equal(
        actual.lineage.source_component_indices,
        expected.lineage.source_component_indices,
    )
    assert torch.equal(actual.lineage.source_xy, expected.lineage.source_xy)
    assert torch.equal(actual.depths, expected.depths)
    assert torch.equal(actual.depth_sigmas, expected.depth_sigmas)
    assert torch.equal(actual.ray_sigmas, expected.ray_sigmas)
    assert torch.equal(actual.scores, expected.scores)
    assert actual.diagnostics == expected.diagnostics


def test_candidate_audit_callback_mutation_cannot_change_initialization():
    inputs = _inputs()
    inputs = _replace_observations(
        inputs,
        [
            replace(field, means=field.means.detach().clone().requires_grad_())
            for field in inputs.observations
        ],
    )
    config = replace(_config(), anchor_mode="component_centers")
    expected = CompactCarveInitializer(config).initialize(inputs)
    observed: list[CompactCandidateAudit] = []

    def mutate_audit(audit: CompactCandidateAudit) -> None:
        observed.append(audit)
        assert all(not value.requires_grad for value in audit.__dict__.values())
        for value in audit.__dict__.values():
            value.zero_()

    actual = CompactCarveInitializer(config).initialize(
        inputs,
        candidate_audit_callback=mutate_audit,
    )

    assert len(observed) == 1
    assert all(not bool(value.any()) for value in observed[0].__dict__.values())
    _assert_initializations_equal(actual, expected)


def test_absent_candidate_audit_callback_preserves_behavior():
    inputs = _inputs()
    config = _config()

    omitted = CompactCarveInitializer(config).initialize(inputs)
    explicit_none = CompactCarveInitializer(config).initialize(
        inputs,
        candidate_audit_callback=None,
    )

    _assert_initializations_equal(explicit_none, omitted)


def test_ray_depth_audit_callback_mutation_cannot_change_initialization():
    inputs = _residual_bearing_inputs()
    config = replace(_config(query_batch_size=64), anchor_mode="component_centers")
    expected = CompactCarveInitializer(config).initialize(inputs)
    observed: list[CompactRayDepthAuditBatch] = []

    def mutate_batch(batch: CompactRayDepthAuditBatch) -> None:
        observed.append(batch)
        for value in batch.__dict__.values():
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad
                value.zero_()

    actual = CompactCarveInitializer(config).initialize(
        inputs,
        ray_depth_audit_callback=mutate_batch,
    )

    assert observed
    for batch in observed:
        for value in batch.__dict__.values():
            if isinstance(value, torch.Tensor):
                assert not bool(value.any())
    _assert_initializations_equal(actual, expected)


def test_absent_ray_depth_audit_callback_preserves_behavior():
    inputs = _inputs()
    config = _config()

    omitted = CompactCarveInitializer(config).initialize(inputs)
    explicit_none = CompactCarveInitializer(config).initialize(
        inputs,
        ray_depth_audit_callback=None,
    )

    _assert_initializations_equal(explicit_none, omitted)


def test_component_center_anchor_cap_fails_before_scoring():
    config = replace(
        _config(),
        anchor_mode="component_centers",
        max_anchor_candidates=7,
    )
    with pytest.raises(ValueError, match="exceeds max_anchor_candidates"):
        CompactCarveInitializer(config).initialize(_inputs())


def test_world_score_is_exact_colocated_split_invariant_and_uses_all_views():
    inputs = _inputs()
    points = torch.cat([_TARGETS, torch.tensor([[0.0, 0.0, 0.35]])])
    config = _config()
    baseline = score_world_points(inputs, points, config)

    fragmented = _replace_observations(
        inputs,
        [_duplicate_component(inputs.observations[0], 1), inputs.observations[1]],
    )
    split_score = score_world_points(fragmented, points, config)
    assert fragmented.n_opt_2d == [5, 4]
    assert torch.allclose(split_score.score, baseline.score, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        split_score.consensus_color,
        baseline.consensus_color,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(split_score.coverage, baseline.coverage, atol=2e-6, rtol=2e-6)

    shifted_colors = inputs.observations[1].colors + torch.tensor([0.7, -0.4, 0.5])
    changed = _replace_observations(
        inputs,
        [inputs.observations[0], replace(inputs.observations[1], colors=shifted_colors)],
    )
    changed_score = score_world_points(changed, points, config)
    assert bool((changed_score.score[: len(_TARGETS)] < baseline.score[: len(_TARGETS)]).all())

    # Only relative-density coverage is invariant to a global amplitude rescale. Exact teacher
    # color deliberately retains normalized epsilon (and additive fields retain amplitude).
    rescaled = _replace_observations(
        inputs,
        [
            replace(
                inputs.observations[0],
                amplitudes=0.25 * inputs.observations[0].amplitudes,
            ),
            inputs.observations[1],
        ],
    )
    rescaled_score = score_world_points(rescaled, points, config)
    assert torch.allclose(rescaled_score.coverage, baseline.coverage, atol=2e-6, rtol=2e-6)
    assert torch.equal(rescaled_score.n_covered, baseline.n_covered)


def test_exact_colocated_split_preserves_initialized_geometry():
    inputs = _inputs()
    config = _config()
    baseline = CompactCarveInitializer(config).initialize(inputs)
    fragmented_inputs = _replace_observations(
        inputs,
        [_duplicate_component(inputs.observations[0], 1), inputs.observations[1]],
    )
    fragmented = CompactCarveInitializer(config).initialize(fragmented_inputs)
    assert fragmented_inputs.n_opt_2d == [5, 4]
    assert torch.allclose(
        fragmented.gaussians.means, baseline.gaussians.means, atol=2e-6, rtol=2e-6
    )
    assert torch.allclose(
        fragmented.gaussians.covariance(),
        baseline.gaussians.covariance(),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(fragmented.scores, baseline.scores, atol=2e-6, rtol=2e-6)


def test_initializer_loads_bundle_without_any_rgb_decoder(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    path = tmp_path / "compact-inputs"
    _inputs().save(path)

    def forbidden_image_open(*_args, **_kwargs):
        raise AssertionError("RGB image decoding is forbidden after the compact boundary")

    monkeypatch.setattr(PILImage, "open", forbidden_image_open)
    loaded = ReconstructionInputs.load(path)
    result = CompactCarveInitializer(_config(n_init_3d=2)).initialize(loaded)
    assert result.n_init_3d == 2
    assert not hasattr(loaded, "images")


def test_sparse_batches_match_and_never_exceed_query_limit(monkeypatch):
    inputs = _inputs()
    observed_batch_sizes: list[int] = []
    observed_pair_counts: list[int] = []
    original_query = GaussianObservationIndex.query
    original_paired_values = GaussianObservationField._paired_values

    def tracked_query(self, xy, component_chunk=4096):
        observed_batch_sizes.append(xy.shape[0])
        return original_query(self, xy, component_chunk=component_chunk)

    def tracked_paired_values(self, xy, component_ids):
        observed_pair_counts.append(component_ids.numel())
        return original_paired_values(self, xy, component_ids)

    # The CSR query streams bounded (point, component) pairs through paired evaluation, so the
    # per-call pair budget is enforced there, not in the old per-tile Cartesian product.
    monkeypatch.setattr(GaussianObservationIndex, "query", tracked_query)
    monkeypatch.setattr(GaussianObservationField, "_paired_values", tracked_paired_values)
    small_config = replace(
        _config(query_batch_size=64),
        query_component_chunk=2,
        max_query_pairs=64,
    )
    small = CompactCarveInitializer(small_config).initialize(inputs)
    assert max(observed_batch_sizes) <= small_config.query_batch_size
    assert max(observed_pair_counts) <= small_config.max_query_pairs

    observed_batch_sizes.clear()
    observed_pair_counts.clear()
    large_config = replace(
        _config(query_batch_size=256),
        query_component_chunk=3,
        max_query_pairs=256,
    )
    large = CompactCarveInitializer(large_config).initialize(inputs)
    assert max(observed_batch_sizes) <= 256
    assert max(observed_pair_counts) <= large_config.max_query_pairs
    assert torch.allclose(small.gaussians.means, large.gaussians.means, atol=1e-6, rtol=1e-6)
    assert torch.allclose(small.scores, large.scores, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("indexed", [False, True])
def test_builtin_query_backends_must_match_ordered_teachers(indexed):
    inputs = _inputs()
    fields = inputs.observations
    backends = (
        [GaussianObservationIndex(fields[1]), GaussianObservationIndex(fields[0])]
        if indexed
        else [fields[1], fields[0]]
    )
    with pytest.raises(ValueError, match="must correspond to their ordered teacher"):
        score_world_points(inputs, _TARGETS, _config(), backends=backends)
    with pytest.raises(ValueError, match="must correspond to their ordered teacher"):
        CompactCarveInitializer(_config()).initialize(inputs, backends=backends)


def test_initializer_accepts_nonindexed_query_backends():
    inputs = _inputs()
    baseline = CompactCarveInitializer(_config()).initialize(inputs)
    fields = list(inputs.observations)
    direct = CompactCarveInitializer(_config()).initialize(inputs, backends=fields)

    assert torch.allclose(direct.gaussians.means, baseline.gaussians.means)
    assert direct.diagnostics["teacher_backend_kinds"] == [
        "GaussianObservationField",
        "GaussianObservationField",
    ]
    assert direct.diagnostics["teacher_index_entries"] == [None, None]


def test_injected_indexes_must_match_carve_tile_and_observed_caps():
    inputs = _inputs()
    backends = [GaussianObservationIndex(field, tile_size=8) for field in inputs.observations]
    with pytest.raises(ValueError, match="differs from compact Carve caps"):
        score_world_points(inputs, _TARGETS, _config(), backends=backends)
    with pytest.raises(ValueError, match="differs from compact Carve caps"):
        CompactCarveInitializer(_config()).initialize(inputs, backends=backends)


def test_selected_geometry_reprojects_parent_effective_covariance():
    inputs = _inputs()
    result = CompactCarveInitializer(_config()).initialize(inputs)
    distance = torch.cdist(result.gaussians.means, _TARGETS).min(dim=1).values
    assert float(distance.median()) < 0.22

    output_id = 0
    view_id = int(result.lineage.source_view_indices[output_id])
    component_id = int(result.lineage.source_component_indices[output_id])
    camera = inputs.cameras[view_id]
    mean = result.gaussians.means[output_id : output_id + 1]
    point_camera = camera.world_to_cam(mean)
    x, y, z = point_camera[0]
    jacobian = torch.tensor(
        [
            [camera.fx / z, 0.0, -camera.fx * x / z.square()],
            [0.0, camera.fy / z, -camera.fy * y / z.square()],
        ]
    )
    covariance_camera = camera.R @ result.gaussians.covariance()[output_id] @ camera.R.T
    projected = jacobian @ covariance_camera @ jacobian.T

    field = inputs.observations[view_id]
    variance = field.effective_variances()[component_id]
    theta = field.rotations[component_id]
    rotation = torch.stack(
        [
            torch.stack([torch.cos(theta), -torch.sin(theta)]),
            torch.stack([torch.sin(theta), torch.cos(theta)]),
        ]
    )
    expected = rotation @ torch.diag(variance) @ rotation.T
    assert torch.allclose(projected, expected, atol=2e-4, rtol=2e-4)

    source_xy = result.lineage.source_xy[output_id : output_id + 1]
    _, direction = camera.pixel_rays(source_xy)
    expected_ray_sigma = result.depth_sigmas[output_id] * direction.norm(dim=-1)[0]
    assert torch.allclose(result.ray_sigmas[output_id], expected_ray_sigma, atol=1e-7, rtol=1e-7)
    unit_ray = torch.nn.functional.normalize(direction, dim=-1)[0]
    world_covariance = result.gaussians.covariance()[output_id]
    ray_variance = unit_ray @ world_covariance @ unit_ray
    assert torch.allclose(
        ray_variance,
        expected_ray_sigma.square(),
        atol=2e-6,
        rtol=2e-5,
    )


def test_initializer_fails_closed_without_required_multiview_support():
    inputs = _inputs()
    config = replace(_config(), min_views=3)
    with pytest.raises(ValueError, match="min_views exceeds"):
        CompactCarveInitializer(config).initialize(inputs)


def test_initializer_fails_closed_when_too_few_candidates_pass_thresholds():
    inputs = _inputs()
    config = replace(_config(), min_score=2.0)
    with pytest.raises(ValueError, match="fewer globally supported"):
        CompactCarveInitializer(config).initialize(inputs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_init_3d", True),
        ("n_init_3d", 2.5),
        ("candidate_multiplier", 1.5),
        ("max_anchor_candidates", 8.0),
        ("samples_per_ray", 32.0),
        ("query_batch_size", False),
        ("query_component_chunk", 2.0),
        ("max_query_pairs", 256.0),
        ("tile_size", 16.0),
        ("seed", 0.0),
        ("min_views", 2.0),
        ("sh_degree", 0.0),
        ("max_anchor_rounds", 8.0),
    ],
)
def test_config_rejects_non_integer_discrete_controls(name, value):
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        replace(_config(), **{name: value})


def test_candidate_budget_may_be_smaller_than_view_count():
    cameras = [_camera(x) for x in (-0.8, -0.4, 0.0, 0.4, 0.8)]
    names = [f"view-{index}" for index in range(len(cameras))]
    inputs = ReconstructionInputs(
        observations=[
            _field(camera, name, n_init=2) for camera, name in zip(cameras, names, strict=True)
        ],
        cameras=cameras,
        view_names=names,
        bounds_hint=(torch.zeros(3), 1.2),
        name="many-view-compact-plane",
    )
    config = replace(
        _config(n_init_3d=1),
        candidate_multiplier=1,
        hull_fraction=0.4,
    )
    result = CompactCarveInitializer(config).initialize(inputs)
    assert result.n_init_3d == 1


def test_config_rejects_unknown_anchor_mode():
    with pytest.raises(ValueError, match="anchor_mode"):
        replace(_config(), anchor_mode="unknown")


def test_score_world_points_csr_matches_grouped_reference_every_field():
    inputs = _inputs()
    config = _config()
    points = torch.cat([_TARGETS, torch.tensor([[0.0, 0.0, 0.35], [0.05, -0.1, -0.2]])])
    csr = score_world_points(inputs, points, config)
    grouped_backends = [
        _GroupedObservationIndexReference(field, tile_size=config.tile_size)
        for field in inputs.observations
    ]
    grouped = score_world_points(inputs, points, config, backends=grouped_backends)

    # Discrete consensus statistics must be exactly equal; continuous fields within the contract.
    assert torch.equal(csr.n_seen, grouped.n_seen)
    assert torch.equal(csr.n_covered, grouped.n_covered)
    torch.testing.assert_close(csr.score, grouped.score, atol=1e-6, rtol=2e-6)
    torch.testing.assert_close(csr.consensus_color, grouped.consensus_color, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(csr.color_variance, grouped.color_variance, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(csr.coverage, grouped.coverage, atol=2e-6, rtol=2e-6)


def test_initializer_csr_preserves_grouped_reference_selection_and_geometry():
    inputs = _inputs()
    config = _config()
    csr_audits: list[CompactCandidateAudit] = []
    grouped_audits: list[CompactCandidateAudit] = []

    csr = CompactCarveInitializer(config).initialize(
        inputs, candidate_audit_callback=csr_audits.append
    )
    grouped_backends = [
        _GroupedObservationIndexReference(field, tile_size=config.tile_size)
        for field in inputs.observations
    ]
    grouped = CompactCarveInitializer(config).initialize(
        inputs,
        backends=grouped_backends,
        candidate_audit_callback=grouped_audits.append,
    )

    # Exact discrete selection identity: winning depth, selected candidate indices, lineage.
    assert torch.equal(
        csr_audits[0].candidate_best_depth_indices,
        grouped_audits[0].candidate_best_depth_indices,
    )
    assert torch.equal(
        csr_audits[0].selected_candidate_indices,
        grouped_audits[0].selected_candidate_indices,
    )
    assert torch.equal(csr.lineage.source_view_indices, grouped.lineage.source_view_indices)
    assert torch.equal(
        csr.lineage.source_component_indices, grouped.lineage.source_component_indices
    )
    assert torch.equal(
        csr_audits[0].candidate_eligible_mask, grouped_audits[0].candidate_eligible_mask
    )

    # Lifted geometry and scores agree within the float32 numerical contract.
    torch.testing.assert_close(csr.gaussians.means, grouped.gaussians.means, atol=1e-6, rtol=2e-6)
    torch.testing.assert_close(
        csr.gaussians.covariance(), grouped.gaussians.covariance(), atol=1e-6, rtol=2e-6
    )
    torch.testing.assert_close(csr.depths, grouped.depths, atol=1e-6, rtol=2e-6)
    torch.testing.assert_close(csr.scores, grouped.scores, atol=1e-6, rtol=2e-6)


def test_select_all_eligible_lifts_every_supported_2d_gaussian():
    inputs = _inputs()
    config = replace(
        _config(n_init_3d=1), anchor_mode="component_centers", select_all_eligible=True
    )
    result = CompactCarveInitializer(config).initialize(inputs)

    # Every proposed component-center candidate that passes support is retained (no top-K).
    assert result.diagnostics["selection_mode"] == "all_eligible"
    assert result.n_init_3d == result.diagnostics["eligible_candidate_count"]
    assert result.n_init_3d > _config().n_init_3d  # denser than the sparse 5k-style default
    # Lineage/geometry stay 1:1 with the lifted set and every view contributes.
    assert result.lineage.source_view_indices.numel() == result.n_init_3d
    assert result.depths.numel() == result.n_init_3d
    assert result.gaussians.covariance().shape == (result.n_init_3d, 3, 3)
    assert set(result.lineage.source_view_indices.tolist()) == set(range(inputs.n_views))

    # Deterministic.
    repeated = CompactCarveInitializer(config).initialize(inputs)
    assert torch.equal(result.gaussians.means, repeated.gaussians.means)
    assert torch.equal(result.scores, repeated.scores)

    # Default (top-K) selection is untouched.
    assert (
        CompactCarveInitializer(_config()).initialize(inputs).diagnostics["selection_mode"]
        == "balanced_topk"
    )


def test_config_rejects_non_bool_select_all_eligible():
    with pytest.raises(TypeError, match="select_all_eligible must be a bool"):
        replace(_config(), select_all_eligible=1)


def test_all_gaussian_lift_then_voxel_merge_recovers_cross_view_correspondence():
    from rtgs.lift.merge import merge_by_voxel

    cameras = [_camera(x) for x in (-0.75, 0.0, 0.75)]
    names = [f"v{i}" for i in range(3)]
    inputs = ReconstructionInputs(
        observations=[
            _field(camera, name, n_init=2) for camera, name in zip(cameras, names, strict=True)
        ],
        cameras=cameras,
        view_names=names,
        bounds_hint=(torch.zeros(3), 1.2),
        name="tri-view-plane",
    )
    config = replace(
        _config(n_init_3d=1), anchor_mode="component_centers", select_all_eligible=True
    )
    lifted = CompactCarveInitializer(config).initialize(inputs)
    assert lifted.n_init_3d >= len(_TARGETS) * inputs.n_views - 2  # ~one lift per (view, target)

    merged, group = merge_by_voxel(
        lifted.gaussians,
        voxel_size=0.08,
        component_weights=lifted.scores.clamp_min(1e-6),
        return_group=True,
    )
    assert group.shape == (lifted.n_init_3d,)
    assert merged.n < lifted.n_init_3d  # redundant per-view lifts are deduplicated

    # The group map is the correspondence byproduct: at least one cluster fuses lifts from
    # more than one source view (the same 3D surface patch seen from different cameras).
    view_ids = lifted.lineage.source_view_indices
    cross_view = 0
    for cluster in group.unique().tolist():
        member_views = view_ids[group == cluster].unique()
        if member_views.numel() > 1:
            cross_view += 1
    assert cross_view >= 1


def test_placement_progress_is_silent_by_default_and_persists_counters():
    inputs = _inputs()
    config = _config()
    records: list[object] = []
    result = CompactCarveInitializer(config).initialize(inputs, progress_callback=records.append)
    phases = [record.phase for record in records]
    assert phases[0] == "index_built"
    assert phases[-1] == "complete"
    assert "ray_batch" in phases
    # Every emitted chunk respected the configured pair cap.
    assert all(record.peak_pair_chunk <= config.max_query_pairs for record in records)

    diagnostics = result.diagnostics
    assert diagnostics["placement_evaluated_pairs"] == records[-1].evaluated_pairs
    assert diagnostics["placement_evaluated_pairs"] > 0
    assert diagnostics["placement_peak_pair_chunk"] == records[-1].peak_pair_chunk
    assert diagnostics["placement_sampled_points"] == records[-1].sampled_points
    assert len(diagnostics["placement_index_payload_bytes"]) == inputs.n_views
    assert diagnostics["teacher_component_id_dtypes"] == ["int32", "int32"]

    # Silent by default: no callback means no progress side effects, identical result.
    silent = CompactCarveInitializer(config).initialize(inputs)
    assert silent.diagnostics == diagnostics
