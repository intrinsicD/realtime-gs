"""CPU tests for the preregistered compact correspondence-confidence gate."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import (
    CompactCandidateAudit,
    CompactInitializationResult,
    CompactLineage,
)
from rtgs.lift.compact_confidence_gate import (
    ClusterConfidenceConfig,
    gate_merged_initialization,
)


def _camera(x: float) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([x, 0.0, -3.0]),
        target=torch.zeros(3),
        width=48,
        height=48,
        fov_x_deg=55.0,
    )


def _gaussians(means: torch.Tensor) -> Gaussians3D:
    count = means.shape[0]
    return Gaussians3D(
        means=means.clone(),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        log_scales=torch.full((count, 3), -3.0),
        opacity=torch.full((count,), 0.5),
        sh=torch.zeros(count, 1, 3),
    )


def _fixture() -> tuple[
    ReconstructionInputs,
    CompactInitializationResult,
    CompactCandidateAudit,
    Gaussians3D,
    torch.Tensor,
]:
    cameras = [_camera(x) for x in (-0.75, 0.0, 0.75)]
    target = torch.tensor([0.0, 0.0, 0.0])
    decoy = torch.tensor([0.45, 0.0, 0.0])
    observations = []
    for index, camera in enumerate(cameras):
        projected, depth = camera.project(torch.stack([target, decoy]))
        assert bool((depth > 0).all())
        observations.append(
            GaussianObservationField(
                width=48,
                height=48,
                means=projected,
                log_scales=torch.zeros(2, 2),
                rotations=torch.zeros(2),
                colors=torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.8, 0.2]]),
                amplitudes=torch.ones(2),
                view_id=f"v{index}",
                n_init=2,
            )
        )
    inputs = ReconstructionInputs(
        observations=observations,
        cameras=cameras,
        view_names=[f"v{index}" for index in range(3)],
        bounds_hint=(torch.zeros(3), 1.0),
        name="confidence-gate-fixture",
    )

    dense_means = torch.stack([target, target, target, decoy])
    source_views = torch.tensor([0, 1, 2, 0])
    source_xy = torch.stack(
        [
            cameras[int(view)].project(dense_means[row : row + 1])[0][0]
            for row, view in enumerate(source_views)
        ]
    )
    dense = CompactInitializationResult(
        gaussians=_gaussians(dense_means),
        lineage=CompactLineage(
            source_view_indices=source_views,
            source_component_indices=torch.tensor([0, 0, 0, 1]),
            source_xy=source_xy,
        ),
        depths=torch.ones(4),
        depth_sigmas=torch.full((4,), 0.05),
        ray_sigmas=torch.full((4,), 0.05),
        scores=torch.full((4,), 0.8),
        diagnostics={},
    )
    audit = CompactCandidateAudit(
        candidate_source_view_indices=source_views.clone(),
        candidate_source_component_indices=torch.tensor([0, 0, 0, 1]),
        candidate_source_xy=source_xy.clone(),
        candidate_best_depths=torch.ones(4),
        candidate_depth_sigmas=torch.full((4,), 0.05),
        candidate_best_means=dense_means.clone(),
        candidate_best_scores=torch.full((4,), 0.8),
        candidate_best_depth_indices=torch.zeros(4, dtype=torch.long),
        candidate_best_coverages=torch.full((4,), 0.9),
        candidate_best_color_variances=torch.full((4,), 0.01),
        candidate_best_n_seen=torch.full((4,), 3, dtype=torch.long),
        candidate_best_n_covered=torch.tensor([3, 3, 3, 2]),
        candidate_second_best_scores=torch.full((4,), 0.6),
        candidate_score_margins=torch.full((4,), 0.2),
        candidate_half_max_widths=torch.full((4,), 0.1),
        candidate_consensus_colors=torch.tensor(
            [
                [0.80, 0.20, 0.10],
                [0.81, 0.20, 0.10],
                [0.79, 0.20, 0.10],
                [0.10, 0.80, 0.20],
            ]
        ),
        candidate_valid_mask=torch.ones(4, dtype=torch.bool),
        candidate_eligible_mask=torch.ones(4, dtype=torch.bool),
        selected_candidate_indices=torch.arange(4),
    )
    merged = _gaussians(torch.stack([target, decoy]))
    group = torch.tensor([0, 0, 0, 1])
    return inputs, dense, audit, merged, group


def test_confidence_gate_keeps_three_view_target_and_drops_single_view_decoy():
    inputs, dense, audit, merged, group = _fixture()

    result = gate_merged_initialization(
        inputs,
        dense,
        audit,
        merged,
        group,
        merge_voxel_size=0.06,
    )

    assert result.keep_mask.tolist() == [True, False]
    assert result.kept_count == 1
    assert result.dropped_count == 1
    assert result.gaussians.n == 1
    assert result.records[0].view_multiplicity == 3
    assert result.records[0].reprojection_max_px == pytest.approx(0.0, abs=1e-6)
    assert result.records[0].failures == ()
    assert result.records[1].failures == ("view_multiplicity",)
    payload = result.as_dict()
    assert payload["failure_histogram"] == {"view_multiplicity": 1}
    assert json.loads(json.dumps(payload))["kept_count"] == 1


def test_confidence_gate_enforces_every_frozen_threshold():
    inputs, dense, audit, merged, group = _fixture()
    audit = replace(
        audit,
        candidate_half_max_widths=torch.tensor([0.1, 0.1, 0.21, 0.1]),
        candidate_best_n_covered=torch.tensor([3, 1, 3, 2]),
    )

    result = gate_merged_initialization(
        inputs,
        dense,
        audit,
        merged,
        group,
        merge_voxel_size=0.06,
        config=ClusterConfidenceConfig(max_reprojection_residual_px=0.0),
    )

    assert not result.records[0].kept
    assert result.records[0].failures == ("half_max_width", "best_n_covered")


def test_confidence_gate_rejects_mismatched_audit_lineage_and_group():
    inputs, dense, audit, merged, group = _fixture()
    mismatched = replace(
        audit,
        candidate_source_component_indices=torch.tensor([1, 0, 0, 1]),
    )
    with pytest.raises(ValueError, match="source component"):
        gate_merged_initialization(
            inputs,
            dense,
            mismatched,
            merged,
            group,
            merge_voxel_size=0.06,
        )
    with pytest.raises(ValueError, match="canonical merged cluster"):
        gate_merged_initialization(
            inputs,
            dense,
            audit,
            merged,
            torch.tensor([0, 0, 0, 0]),
            merge_voxel_size=0.06,
        )


@pytest.mark.parametrize("merge_voxel_size", [0.0, -0.1, float("inf")])
def test_confidence_gate_rejects_invalid_merge_voxel(merge_voxel_size):
    inputs, dense, audit, merged, group = _fixture()
    with pytest.raises(ValueError, match="finite and positive"):
        gate_merged_initialization(
            inputs,
            dense,
            audit,
            merged,
            group,
            merge_voxel_size=merge_voxel_size,
        )
