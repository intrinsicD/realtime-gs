"""Adam-moment inheritance for arena clone/split children (protocol
20260731_topology_moment_inheritance): child moments must equal gamma * parent moment,
the default zero profile must reproduce the historical cold start exactly, and the
inheritance layout must follow the transaction's survivor/clone/split row order."""

from __future__ import annotations

import pytest
import torch

from rtgs.optim.arena import (
    GeometricParameterArena,
    MomentInheritance,
    apply_default_topology_transaction,
)


def _make_arena(n: int = 4) -> tuple[dict, dict, GeometricParameterArena]:
    gen = torch.Generator().manual_seed(11)
    params = {
        "means": torch.nn.Parameter(torch.randn(n, 3, generator=gen)),
        "scales": torch.nn.Parameter(torch.randn(n, 3, generator=gen) * 0.1 - 2.0),
        "quats": torch.nn.Parameter(torch.randn(n, 4, generator=gen)),
        "opacities": torch.nn.Parameter(torch.full((n,), 4.0)),
    }
    optimizers = {
        name: torch.optim.Adam([parameter], lr=1e-3) for name, parameter in params.items()
    }
    arena = GeometricParameterArena(params, optimizers, max_capacity=64)
    gen_state = torch.Generator().manual_seed(23)
    for name in params:
        avg = arena.active_state(name, "exp_avg")
        avg.copy_(torch.randn(avg.shape, generator=gen_state))
        avg_sq = arena.active_state(name, "exp_avg_sq")
        avg_sq.copy_(torch.rand(avg_sq.shape, generator=gen_state))
    return params, optimizers, arena


def _transaction(arena, *, inheritance=None, n: int = 4):
    clone_mask = torch.zeros(n, dtype=torch.bool)
    split_mask = torch.zeros(n, dtype=torch.bool)
    clone_mask[1] = True
    split_mask[3] = True
    offsets = torch.randn(2, 1, 3, generator=torch.Generator().manual_seed(5)) * 0.01
    return apply_default_topology_transaction(
        arena,
        clone_mask=clone_mask,
        split_mask=split_mask,
        split_offsets=offsets,
        split_factor=1.6,
        revised_opacity=False,
        prune_opacity=0.005,
        prune_large_scale=None,
        max_gaussians=64,
        iteration=1,
        moment_inheritance=inheritance,
    )


def _moments_before(arena) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: (
            arena.active_state(name, "exp_avg").clone(),
            arena.active_state(name, "exp_avg_sq").clone(),
        )
        for name in arena.params
    }


def test_default_children_are_exact_zero():
    params, _, arena = _make_arena()
    before = _moments_before(arena)
    receipt = _transaction(arena)
    assert receipt.n_after == 6
    for name in params:
        for key, index in (("exp_avg", 0), ("exp_avg_sq", 1)):
            state = arena.active_state(name, key)
            # Rows: survivors 0,1,2 then clone(1), split0(3), split1(3).
            assert torch.equal(state[:3], before[name][index][:3])
            assert torch.equal(state[3:], torch.zeros_like(state[3:]))


def test_exact_inheritance_copies_parent_moments():
    params, _, arena = _make_arena()
    before = _moments_before(arena)
    _transaction(arena, inheritance=MomentInheritance(1.0, 1.0, 1.0, 1.0))
    for name in params:
        for key, index in (("exp_avg", 0), ("exp_avg_sq", 1)):
            state = arena.active_state(name, key)
            parent = before[name][index]
            assert torch.equal(state[:3], parent[:3])
            assert torch.equal(state[3], parent[1])
            assert torch.equal(state[4], parent[3])
            assert torch.equal(state[5], parent[3])


def test_attenuated_inheritance_scales_by_gamma():
    params, _, arena = _make_arena()
    before = _moments_before(arena)
    inheritance = MomentInheritance(
        clone_gamma_m=0.5, clone_gamma_v=0.25, split_gamma_m=0.75, split_gamma_v=1.0
    )
    _transaction(arena, inheritance=inheritance)
    for name in params:
        avg = arena.active_state(name, "exp_avg")
        avg_sq = arena.active_state(name, "exp_avg_sq")
        assert torch.allclose(avg[3], before[name][0][1] * 0.5)
        assert torch.allclose(avg_sq[3], before[name][1][1] * 0.25)
        assert torch.allclose(avg[4], before[name][0][3] * 0.75)
        assert torch.allclose(avg[5], before[name][0][3] * 0.75)
        assert torch.equal(avg_sq[4], before[name][1][3])


def test_field_override_takes_precedence():
    params, _, arena = _make_arena()
    before = _moments_before(arena)
    inheritance = MomentInheritance(
        clone_gamma_m=0.5,
        clone_gamma_v=0.5,
        split_gamma_m=0.5,
        split_gamma_v=0.5,
        field_overrides={"means": (0.25, 1.0, 0.25, 1.0)},
    )
    _transaction(arena, inheritance=inheritance)
    means_avg = arena.active_state("means", "exp_avg")
    assert torch.allclose(means_avg[3], before["means"][0][1] * 0.25)
    assert torch.equal(arena.active_state("means", "exp_avg_sq")[3], before["means"][1][1])
    scales_avg = arena.active_state("scales", "exp_avg")
    assert torch.allclose(scales_avg[3], before["scales"][0][1] * 0.5)


def test_zero_gamma_never_propagates_non_finite_parent_state():
    params, _, arena = _make_arena()
    arena.active_state("means", "exp_avg")[1] = float("inf")
    _transaction(arena, inheritance=MomentInheritance(0.0, 1.0, 0.0, 1.0))
    means_avg = arena.active_state("means", "exp_avg")
    assert torch.equal(means_avg[3], torch.zeros_like(means_avg[3]))
    assert bool(torch.isfinite(means_avg[3:]).all())


def test_prune_keeps_inherited_moments_aligned():
    params, _, arena = _make_arena()
    with torch.no_grad():
        params["opacities"][0] = -10.0  # survivor row 0 prunes away
    before = _moments_before(arena)
    receipt = _transaction(arena, inheritance=MomentInheritance(1.0, 1.0, 1.0, 1.0))
    assert receipt.n_prune == 1
    assert receipt.n_after == 5
    avg = arena.active_state("means", "exp_avg")
    # Row 0 was pruned: layout is survivors 1,2 then clone(1), split0(3), split1(3).
    assert torch.equal(avg[0], before["means"][0][1])
    assert torch.equal(avg[1], before["means"][0][2])
    assert torch.equal(avg[2], before["means"][0][1])
    assert torch.equal(avg[3], before["means"][0][3])
    assert torch.equal(avg[4], before["means"][0][3])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clone_gamma_m": -0.1},
        {"clone_gamma_v": 1.5},
        {"split_gamma_m": float("nan")},
        {"split_gamma_v": True},
    ],
)
def test_invalid_gammas_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        MomentInheritance(**kwargs)


def test_invalid_field_overrides_are_rejected():
    with pytest.raises(ValueError):
        MomentInheritance(field_overrides={"means": (1.0, 1.0, 1.0)})
    with pytest.raises(ValueError):
        MomentInheritance(field_overrides={"": (1.0, 1.0, 1.0, 1.0)})
    with pytest.raises(ValueError):
        MomentInheritance(field_overrides={"means": (1.0, 1.0, 1.0, 2.0)})


def test_trainer_rejects_inheritance_without_geometric_storage():
    from rtgs.data.synthetic import make_synthetic_scene
    from rtgs.optim.trainer import TrainConfig, Trainer

    scene = make_synthetic_scene(n_gaussians=6, n_cameras=2, image_size=16, seed=3)
    cfg = TrainConfig(
        iterations=2,
        rasterizer="torch",
        densify=False,
        topology_moment_gammas=(1.0, 1.0, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="geometric"):
        Trainer(cfg).train(scene, scene.gt_gaussians.detach())
