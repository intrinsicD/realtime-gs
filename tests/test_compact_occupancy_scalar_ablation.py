"""Outcome-free checks for the compact occupancy-scalar ablation protocol."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from benchmarks import compact_occupancy_scalar_ablation as protocol

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs


def _variant_metrics(precision: float, recall: float, iou: float, auc: float) -> dict:
    return {"pooled": {"precision": precision, "recall": recall, "iou": iou, "auc": auc}}


def test_antithetic_sobol_normal_samples_are_frozen_and_deterministic() -> None:
    first = protocol.footprint_standard_samples(torch.float32)
    second = protocol.footprint_standard_samples(torch.float32)
    assert first.shape == (protocol.FOOTPRINT_SAMPLES, 2)
    assert torch.equal(first, second)
    assert torch.equal(first[:16], -first[16:])
    assert protocol.base.tensor_hash(first) == (
        "7dbfe44552126414cfb54eaf5c366f5aff2f4ef8834771aded7068fcb5783667"
    )


def test_normalized_lse_is_bounded_and_preserves_constant_endpoints() -> None:
    zeros = torch.zeros(3, 32)
    ones = torch.ones(3, 32)
    mixed = torch.tensor([[0.0, 0.25, 0.75, 1.0]])
    for beta in protocol.LSE_BETAS:
        assert torch.allclose(protocol.normalized_lse(zeros, beta), torch.zeros(3), atol=1e-7)
        assert torch.allclose(protocol.normalized_lse(ones, beta), torch.ones(3), atol=1e-7)
        smooth = protocol.normalized_lse(mixed, beta)
        assert bool((smooth >= mixed.mean(dim=-1) - 1e-7).all())
        assert bool((smooth <= mixed.max(dim=-1).values + 1e-7).all())
    assert protocol.normalized_lse(mixed, 16) >= protocol.normalized_lse(mixed, 2)
    with pytest.raises(ValueError, match="positive"):
        protocol.normalized_lse(mixed, 0)
    with pytest.raises(ValueError, match="non-empty"):
        protocol.normalized_lse(torch.empty(2, 0), 2)


def test_selector_enforces_precision_guard_and_uses_lower_beta_for_exact_tie() -> None:
    tuning = {
        "center": _variant_metrics(0.99, 0.70, 0.69, 0.90),
        "mean": _variant_metrics(0.99, 0.71, 0.70, 0.91),
        "lse_beta_2": _variant_metrics(0.985, 0.82, 0.80, 0.94),
        "lse_beta_4": _variant_metrics(0.979, 0.99, 0.97, 0.99),
        "lse_beta_8": _variant_metrics(0.982, 0.85, 0.83, 0.96),
        "lse_beta_16": _variant_metrics(0.982, 0.85, 0.83, 0.96),
        "hard_max": _variant_metrics(0.982, 0.88, 0.86, 0.97),
    }
    selected = protocol.select_stage_b_variants(tuning)
    assert selected["precision_floor"] == pytest.approx(0.98)
    assert selected["selected_lse"]["variant"] == "lse_beta_8"
    assert {item["variant"] for item in selected["rejected_lse_candidates"]} == {"lse_beta_4"}
    assert selected["hard_max_rule"]["included"] is True
    assert selected["stage_b_variants"] == ["center", "mean", "lse_beta_8", "hard_max"]


def test_selector_has_no_report_metric_input_and_fails_closed_without_guard_candidate() -> None:
    assert tuple(inspect.signature(protocol.select_stage_b_variants).parameters) == ("tuning",)
    tuning = {
        "center": _variant_metrics(0.99, 0.70, 0.69, 0.90),
        "mean": _variant_metrics(0.99, 0.71, 0.70, 0.91),
        "hard_max": _variant_metrics(0.70, 1.0, 0.70, 0.90),
        **{
            f"lse_beta_{beta}": _variant_metrics(0.95, 1.0, 0.95, 0.99)
            for beta in protocol.LSE_BETAS
        },
    }
    with pytest.raises(RuntimeError, match="precision guard"):
        protocol.select_stage_b_variants(tuning)


def test_run_seals_before_mask_decode_and_selects_before_report_query() -> None:
    source = inspect.getsource(protocol.run)
    seal = source.index('base.write_json(output / "plan.json", plan)')
    mask_decode = source.index("masks, mask_bindings = base.load_masks(exact_inputs)")
    scalar_construction = source.index("variant_fields, scalar_records, observed_standard")
    selection = source.index("selection = select_stage_b_variants(tuning)")
    report_query = source.index("selected_views=REPORT_VIEWS")
    assert seal < mask_decode
    assert seal < scalar_construction
    assert selection < report_query


def test_stage_b_center_extent_is_evaluated_once_then_injected_for_every_arm() -> None:
    cameras = [
        Camera.look_at(torch.tensor(eye), torch.zeros(3), width=8, height=8)
        for eye in ((1.0, 0.0, 0.2), (0.0, 1.0, 0.3), (-1.0, 0.0, 0.4))
    ]
    names = [f"view-{index}" for index in range(3)]
    fields = [
        GaussianObservationField(
            width=8,
            height=8,
            means=torch.tensor([[4.0, 4.0]]),
            log_scales=torch.zeros(1, 2),
            rotations=torch.zeros(1),
            colors=torch.ones(1, 3),
            amplitudes=torch.ones(1),
            view_id=name,
            provider="synthetic_fixture",
        )
        for name in names
    ]
    inputs = ReconstructionInputs(fields, cameras, names, name="bounds_test")
    frozen, binding = protocol.freeze_stage_b_center_extent(
        inputs,
        SimpleNamespace(bounds_scale=0.5),
    )
    assert frozen.bounds_hint is not None
    injected_center, injected_extent = frozen.bounds_hint
    for _ in range(4):
        center, extent = protocol.base._center_and_extent(frozen, torch.float32)
        assert torch.equal(center, injected_center)
        assert extent == injected_extent
        assert protocol.base.tensor_hash(center) == binding["center_sha256"]

    source = inspect.getsource(protocol.run)
    assert source.count("freeze_stage_b_center_extent(") == 1
    assert "inputs=stage_b_inputs" in source
    assert "build_single_variant_backends(\n                stage_b_inputs" in source
