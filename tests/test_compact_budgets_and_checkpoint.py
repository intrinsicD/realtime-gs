"""Aggregate observation-index budgets and checkpointed pair-chunk queries.

Both are opt-in production-scale controls from the ROADMAP compact-scaling bullet: the
aggregate entry/byte budgets fail fast when the sum of per-view indexes exceeds a configured
total (the per-view caps alone let a many-view scene multiply unbounded), and
``checkpoint_pair_chunks`` bounds backward activation memory for gradient-carrying queries
without changing values or gradients.
"""

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import (
    CompactCarveConfig,
    build_query_backends,
    score_world_points,
)


def _field(n: int = 60, canvas: int = 64, seed: int = 0) -> GaussianObservationField:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return GaussianObservationField(
        width=canvas,
        height=canvas,
        means=torch.rand(n, 2, generator=gen) * (canvas - 2.0) + 1.0,
        log_scales=torch.log(torch.rand(n, 2, generator=gen) * 4.5 + 1.0),
        rotations=(torch.rand(n, generator=gen) - 0.5) * 2.0,
        colors=torch.rand(n, 3, generator=gen) * 1.4 - 0.2,
        amplitudes=torch.rand(n, generator=gen) * 0.9 + 0.1,
        color_grads=torch.randn(n, 2, 3, generator=gen) / 8.0,
        support_fade_alpha=0.4,
        fit_window=(1, 1, canvas - 2, canvas - 2),
    )


def _total_estimated_entries(fields, tile_size: int) -> int:
    return sum(GaussianObservationIndex.estimate_entries(f, tile_size) for f in fields)


def test_aggregate_entry_budget_preflights_before_allocation():
    fields = [_field(seed=0), _field(seed=1)]
    total = _total_estimated_entries(fields, 16)
    assert total > 1
    config = CompactCarveConfig(n_init_3d=8, max_index_entries_total=total - 1)
    with pytest.raises(ValueError, match="before allocation"):
        build_query_backends(fields, config)
    # At the exact budget the build succeeds.
    fits = CompactCarveConfig(n_init_3d=8, max_index_entries_total=total)
    assert len(build_query_backends(fields, fits)) == 2


def test_aggregate_byte_budget_enforced_on_built_backends():
    fields = [_field(seed=0), _field(seed=1)]
    config = CompactCarveConfig(n_init_3d=8, max_index_bytes_total=64)
    with pytest.raises(ValueError, match="byte budget"):
        build_query_backends(fields, config)


def test_aggregate_budget_config_validation():
    with pytest.raises(ValueError, match="max_index_entries_total"):
        CompactCarveConfig(n_init_3d=8, max_index_entries_total=0)
    with pytest.raises(ValueError, match="max_index_bytes_total"):
        CompactCarveConfig(n_init_3d=8, max_index_bytes_total=-4)


def test_score_world_points_enforces_aggregate_budget_for_supplied_backends():
    cameras = [
        Camera.look_at(
            eye=torch.tensor([x, 0.0, -3.0]),
            target=torch.zeros(3),
            width=64,
            height=64,
            fov_x_deg=55.0,
        )
        for x in (-0.5, 0.5)
    ]
    fields = [_field(seed=0), _field(seed=1)]
    inputs = ReconstructionInputs(
        observations=fields,
        cameras=cameras,
        view_names=["a", "b"],
    )
    base = CompactCarveConfig(n_init_3d=8)
    backends = build_query_backends(fields, base)
    tight = CompactCarveConfig(n_init_3d=8, max_index_entries_total=1)
    points = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="entry budget exceeded"):
        score_world_points(inputs, points, tight, backends=backends)
    # Unbounded defaults keep the existing behavior.
    scores = score_world_points(inputs, points, base, backends=backends)
    assert scores.score.shape == (4,)


def _multi_chunk_index(field: GaussianObservationField) -> GaussianObservationIndex:
    # A small pair budget forces the query to stream several chunks.
    return GaussianObservationIndex(field, tile_size=16, max_query_pairs=512)


def _grad_query(index, xy_base, *, checkpointed: bool):
    xy = xy_base.clone().requires_grad_(True)
    result = index.query(xy, component_chunk=64, checkpoint_pair_chunks=checkpointed)
    result.color.square().sum().backward()
    return result, xy.grad


def test_checkpointed_query_matches_baseline_values_and_grads():
    field = _field()
    xy_base = torch.rand(256, 2, generator=torch.Generator().manual_seed(2)) * field.width
    baseline, grad_baseline = _grad_query(_multi_chunk_index(field), xy_base, checkpointed=False)
    checked, grad_checked = _grad_query(_multi_chunk_index(field), xy_base, checkpointed=True)
    assert torch.equal(checked.color, baseline.color)
    assert torch.equal(checked.weight_sum, baseline.weight_sum)
    assert torch.allclose(grad_checked, grad_baseline, atol=1e-7, rtol=1e-6)

    index = _multi_chunk_index(field)
    xy = xy_base.clone().requires_grad_(True)
    plain = index.query_weight_sum(xy, component_chunk=64)
    xy_ck = xy_base.clone().requires_grad_(True)
    ck = index.query_weight_sum(xy_ck, component_chunk=64, checkpoint_pair_chunks=True)
    assert torch.equal(ck, plain)
    plain.sum().backward()
    ck.sum().backward()
    assert torch.allclose(xy_ck.grad, xy.grad, atol=1e-7, rtol=1e-6)


def test_checkpointing_bounds_saved_activation_bytes():
    field = _field()
    xy_base = torch.rand(256, 2, generator=torch.Generator().manual_seed(3)) * field.width

    def saved_bytes(checkpointed: bool) -> int:
        index = _multi_chunk_index(field)
        total = 0

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            nonlocal total
            total += tensor.numel() * tensor.element_size()
            return tensor

        xy = xy_base.clone().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            index.query(xy, component_chunk=64, checkpoint_pair_chunks=checkpointed)
        # The pair budget must actually have split the stream, or the bound is untested.
        assert index.peak_pair_chunk < index.total_pairs_evaluated
        return total

    baseline = saved_bytes(False)
    checkpointed = saved_bytes(True)
    assert checkpointed < baseline, (checkpointed, baseline)


def test_checkpoint_flag_is_inert_without_gradients():
    field = _field()
    index = _multi_chunk_index(field)
    xy = torch.rand(64, 2, generator=torch.Generator().manual_seed(4)) * field.width
    with torch.no_grad():
        plain = index.query(xy, checkpoint_pair_chunks=False)
        flagged = index.query(xy, checkpoint_pair_chunks=True)
    assert torch.equal(plain.color, flagged.color)
    assert torch.equal(plain.weight_sum, flagged.weight_sum)
