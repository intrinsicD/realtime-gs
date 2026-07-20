"""Synthetic, TUM-value-free checks for the signed attribution harness."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from benchmarks import tum_rgbd_oriented_validity as base
from benchmarks import tum_rgbd_signed_attribution as audit


def _camera() -> object:
    pose = base.TimedPose(
        timestamp_ns=0,
        timestamp_token="0",
        center=torch.zeros(3, dtype=torch.float64),
        quaternion_xyzw=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
    )
    return base._camera_from_pose(pose)


def test_signed_classes_have_frozen_camera_z_direction_and_neutral_band():
    predicted = torch.tensor([2.0, 1.0, 1.04, 0.96, 2.0], dtype=torch.float64)
    observed = torch.tensor([1.0, 2.0, 1.0, 1.0, float("nan")], dtype=torch.float64)
    signed, relative, tolerance, positive, negative = audit._classify_signed(predicted, observed)
    assert signed[:4].tolist() == [1.0, -1.0, 0.040000000000000036, -0.040000000000000036]
    assert relative[:2].tolist() == [1.0, -0.5]
    assert tolerance[:4].tolist() == [0.05, 0.06, 0.05, 0.05]
    assert positive.tolist() == [True, False, False, False, False]
    assert negative.tolist() == [False, True, False, False, False]


def test_dense_t_only_foreground_hides_sparse_target_and_preserves_nesting():
    camera = _camera()
    pixel = torch.tensor([[320.5, 240.5]], dtype=torch.float64)
    target = camera.unproject(pixel, torch.tensor([2.0], dtype=torch.float64))
    foreground = camera.unproject(pixel, torch.tensor([1.0], dtype=torch.float64))
    _, sparse_visible, sparse_depth, sparse_pixels, _ = audit._construction_visibility(
        target, target, camera
    )
    _, dense_visible, dense_depth, dense_pixels, _ = audit._construction_visibility(
        target, torch.cat([target, foreground]), camera
    )
    assert sparse_visible.tolist() == [True]
    assert dense_visible.tolist() == [False]
    assert not bool((dense_visible & ~sparse_visible).any())
    assert torch.equal(sparse_depth, dense_depth)
    assert torch.equal(sparse_pixels, dense_pixels)


def test_validation_observation_can_change_labels_but_not_construction_visibility():
    camera = _camera()
    target = camera.unproject(
        torch.tensor([[320.5, 240.5]], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
    )
    visibility_before = audit._construction_visibility(target, target, camera)[1]
    positive_near = audit._classify_signed(torch.tensor([2.0]), torch.tensor([1.0]))[3]
    positive_far = audit._classify_signed(torch.tensor([2.0]), torch.tensor([3.0]))[3]
    visibility_after = audit._construction_visibility(target, target, camera)[1]
    assert torch.equal(visibility_before, visibility_after)
    assert positive_near.tolist() == [True]
    assert positive_far.tolist() == [False]


def test_target_balancing_requires_two_views_and_uses_linear_median():
    valid = torch.tensor([[True, True, False], [True, False, False]])
    positive = torch.tensor([[True, False, False], [True, False, False]])
    negative = torch.tensor([[False, True, False], [False, False, False]])
    relative = torch.tensor([[0.1, -0.3, float("nan")], [0.8, float("nan"), float("nan")]])
    vectors = audit._target_vectors(valid, positive, negative, relative)
    assert vectors["supported"].tolist() == [True, False]
    assert vectors["positive_rate"][0] == 0.5
    assert vectors["negative_rate"][0] == 0.5
    assert vectors["contradiction_rate"][0] == 1.0
    assert vectors["absolute_relative_median"][0] == pytest.approx(0.2)
    assert math.isnan(float(vectors["positive_rate"][1]))


def test_occlusion_gate_directions_and_bootstrap_sign_requirement_are_exact():
    arms = {
        "sparse": {"P_plus": 0.30, "P_minus": 0.10, "D90": 0.20},
        "dense_T": {
            "P_plus": 0.20,
            "P_minus": 0.10,
            "D90": 0.18,
            "supported_targets": 2_000,
        },
    }
    attribution = {
        "sparse_depth_valid_pairs": 100_000,
        "dense_depth_valid_pairs": 80_000,
        "removed_depth_valid_pairs": 20_000,
        "dense_retention": 0.80,
        "removed_fraction": 0.20,
        "paired_removed_retained_targets": 1_000,
        "E_plus": 0.20,
        "E_minus": 0.05,
        "positive_selectivity": 0.15,
        "positive_risk_ratio": 3.0,
        "P_plus_reduction": 0.10,
        "P_minus_change_dense_minus_sparse": 0.0,
        "D90_ratio_dense_over_sparse": 0.90,
    }
    bootstrap = {"estimable": True, "all_asserted_positive_signs": True}
    gate = audit._occlusion_gate(attribution, arms, bootstrap)
    assert gate["comparisons"]["all"]
    failed = copy.deepcopy(attribution)
    failed["positive_selectivity"] = 0.049
    assert not audit._occlusion_gate(failed, arms, bootstrap)["comparisons"]["positive_selectivity"]
    assert not audit._occlusion_gate(attribution, arms, {"estimable": False})["comparisons"][
        "bootstrap_signs"
    ]


def test_temporal_boundaries_are_inclusive_and_middle_is_excluded():
    valid = torch.ones((2, 4), dtype=torch.bool)
    contradiction = torch.tensor(
        [[False, True, True, True], [True, True, False, False]], dtype=torch.bool
    )
    gap = torch.tensor([[0.20, 0.40, 0.60, 0.80]] * 2, dtype=torch.float64)
    translation = torch.zeros_like(gap)
    rotation = torch.zeros_like(gap)
    summary, target_delta = audit._temporal_summary(
        valid, contradiction, gap, translation, rotation
    )
    assert summary["near_pairs"] == 2
    assert summary["far_pairs"] == 4
    assert summary["paired_targets"] == 2
    assert summary["temporal_delta"] == 0.0
    assert target_delta.tolist() == [1.0, -1.0]
    assert not summary["pose_conditioned"]["estimable"]


def test_pose_weight_cap_requires_four_cells_and_sums_exactly():
    weights = audit._capped_weights([1.0, 1.0, 1.0, 1.0], 0.25)
    assert weights == [0.25, 0.25, 0.25, 0.25]
    with pytest.raises(ValueError, match="too few pose cells"):
        audit._capped_weights([1.0, 1.0, 1.0], 0.25)


def test_tensor_decoder_roundtrips_and_rejects_hash_tampering():
    tensor = torch.tensor([[1.0, float("nan")], [3.0, 4.0]], dtype=torch.float64)
    encoded = base._encoded_tensor(tensor)
    decoded = audit._decode_tensor(encoded, label="fixture")
    assert torch.equal(torch.isnan(decoded), torch.isnan(tensor))
    assert torch.equal(torch.nan_to_num(decoded), torch.nan_to_num(tensor))
    tampered = dict(encoded)
    tampered["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw hash mismatch"):
        audit._decode_tensor(tampered, label="fixture")
