from __future__ import annotations

import math

import torch

from rtgs.optim.arena import GeometricParameterArena, apply_default_topology_transaction


def _adam(params: dict[str, torch.nn.Parameter]) -> dict[str, torch.optim.Optimizer]:
    return {
        name: torch.optim.Adam(
            [{"params": [parameter], "lr": 1e-2, "name": name}],
            eps=1e-15,
        )
        for name, parameter in params.items()
    }


def test_geometric_arena_keeps_live_shaped_adam_across_capacity_growth():
    params = {
        "means": torch.nn.Parameter(torch.arange(9, dtype=torch.float32).reshape(3, 3)),
        "opacity": torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3])),
    }
    optimizers = _adam(params)
    arena = GeometricParameterArena(
        params,
        optimizers,
        max_capacity=16,
        growth_factor=2.0,
    )
    assert arena.active_n == 3
    assert arena.capacity == 4

    sum(parameter.sum() for parameter in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()
    values_before = {name: value.detach().clone() for name, value in params.items()}
    moments_before = {name: arena.active_state(name, "exp_avg").clone() for name in params}
    steps_before = {name: optimizers[name].state[params[name]]["step"].clone() for name in params}

    assert arena.reserve(6, iteration=7)
    assert arena.capacity == 8
    assert len(arena.migrations) == 1
    for name in params:
        assert params[name].shape[0] == 3
        assert torch.equal(params[name], values_before[name])
        assert torch.equal(arena.active_state(name, "exp_avg"), moments_before[name])
        assert torch.equal(optimizers[name].state[params[name]]["step"], steps_before[name])
        assert optimizers[name].state[params[name]]["exp_avg"].shape == params[name].shape

    sum(parameter.square().sum() for parameter in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()
    assert all(int(optimizers[name].state[params[name]]["step"].item()) == 2 for name in params)


def test_default_topology_transaction_preserves_gsplat_row_order_and_moments():
    params = {
        "means": torch.nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                ]
            )
        ),
        "quats": torch.nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            )
        ),
        "scales": torch.nn.Parameter(torch.zeros(3, 3)),
        "opacities": torch.nn.Parameter(torch.full((3,), 2.0)),
        "sh0": torch.nn.Parameter(torch.arange(9, dtype=torch.float32).reshape(3, 1, 3)),
        "shN": torch.nn.Parameter(torch.zeros(3, 15, 3)),
    }
    optimizers = _adam(params)
    arena = GeometricParameterArena(params, optimizers, max_capacity=8)
    sum(parameter.sum() for parameter in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()
    old_means = params["means"].detach().clone()
    old_scales = params["scales"].detach().clone()
    old_moment = arena.active_state("means", "exp_avg").clone()

    receipt = apply_default_topology_transaction(
        arena,
        clone_mask=torch.tensor([True, False, False]),
        split_mask=torch.tensor([False, True, False]),
        split_offsets=torch.tensor(
            [
                [[1.0, 2.0, 3.0]],
                [[-1.0, -2.0, -3.0]],
            ]
        ),
        split_factor=1.6,
        revised_opacity=False,
        prune_opacity=0.0,
        prune_large_scale=None,
        max_gaussians=8,
        iteration=5,
    )

    assert receipt.n_before == 3
    assert receipt.n_after == 5
    assert receipt.capacity_before == 4
    assert receipt.capacity_after == 8
    # Dynamic gsplat order is: unsplit originals, clones, first split children, second children.
    expected_means = torch.stack(
        (
            old_means[0],
            old_means[2],
            old_means[0],
            old_means[1] + torch.tensor([1.0, 2.0, 3.0]),
            old_means[1] - torch.tensor([1.0, 2.0, 3.0]),
        )
    )
    assert torch.equal(params["means"], expected_means)
    expected_moment = torch.stack(
        (
            old_moment[0],
            old_moment[2],
            torch.zeros_like(old_moment[0]),
            torch.zeros_like(old_moment[0]),
            torch.zeros_like(old_moment[0]),
        )
    )
    assert torch.equal(arena.active_state("means", "exp_avg"), expected_moment)
    assert torch.allclose(
        params["scales"][-2:],
        old_scales[1].expand(2, -1) + math.log(1.0 / 1.6),
    )
    assert arena.diagnostics()["migration_count"] == 1
