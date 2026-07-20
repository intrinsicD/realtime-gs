"""Focused CPU checks for the frozen responsibility-birth mechanism."""

from __future__ import annotations

import hashlib

import pytest
import torch
from benchmarks import compact_responsibility_birth_allocation as birth

from rtgs.render.point_base import PointRenderOutput


def _selection_fixture() -> dict[str, torch.Tensor | float | int]:
    n = 40
    ids = torch.arange(n, dtype=torch.long)
    # Twenty small and twenty large rows; the support median creates ten rows in
    # each of the four strata, enough for the frozen quota of eight.
    scale = torch.cat(
        [
            torch.full((20,), 0.005),
            torch.full((20,), 0.02),
        ]
    )
    support = torch.arange(1, n + 1, dtype=torch.float64)
    residual = torch.arange(n, 0, -1, dtype=torch.float64).square()
    gradient = torch.roll(residual.to(torch.float32), 7)
    support_by_view = support[None, :].repeat(3, 1) / 3.0
    return {
        "gradient_score": gradient,
        "residual_score": residual,
        "support_score": support,
        "support_by_view": support_by_view,
        "visible_step_count": torch.ones(n, dtype=torch.int64),
        "scale_max": scale,
        "persistent_ids": ids,
        "extent": 1.0,
        "shuffle_root": 993003,
    }


def test_domain_seed_exact_encoding_and_validation() -> None:
    payload = b"rtgs.compact-responsibility-birth.v1\0evaluation_bank\0993101\0C0001\0uniform"
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    expected &= (1 << 63) - 1
    assert birth.domain_seed("evaluation_bank", 993101, "C0001", "uniform") == expected
    assert birth.evaluation_bank_seed(993101, "C0001", "uniform") == expected
    assert birth.encode_atom("µ") == "µ".encode()
    for value in (-1, True, 1.0, b"x", None):
        with pytest.raises(TypeError):
            birth.encode_atom(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        birth.evaluation_bank_seed(993101, "C0001", "Uniform")


def test_matched_strata_ranking_and_shuffle_are_deterministic() -> None:
    fixture = _selection_fixture()
    first = birth.build_matched_selections(**fixture)
    second = birth.build_matched_selections(**fixture)
    assert first == second
    assert first["semantic_sha256"] == second["semantic_sha256"]
    eligible = set(first["eligible_rows"])
    seen: set[int] = set()
    for stratum in birth.STRATA:
        record = first["strata"][stratum]
        members = set(record["members"])
        assert len(members) == 10
        assert not seen & members
        seen |= members
        for arm in birth.ARMS:
            assert len(record["selected"][arm]) == 8
        shuffle = first["shuffle"][stratum]
        assert sorted(shuffle["permutation"]) == list(range(10))
        assert len(shuffle["assignments"]) == 10
    assert seen == eligible
    for arm in birth.ARMS:
        selected = first["selected_rows"][arm]
        assert len(selected) == 32
        assert len(set(selected)) == 32
        assert set(selected) <= eligible
        assert selected == sorted(selected)


def _params(n: int) -> dict[str, torch.nn.Parameter]:
    scales = torch.empty(n, 3)
    scales[: n // 2] = torch.log(torch.tensor(0.005))
    scales[n // 2 :] = torch.log(torch.tensor(0.02))
    return {
        "means": torch.nn.Parameter(torch.zeros(n, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)),
        "scales": torch.nn.Parameter(scales),
        "opacities": torch.nn.Parameter(torch.zeros(n)),
        "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3)),
        "shN": torch.nn.Parameter(torch.empty(n, 0, 3)),
    }


def _optimizers(
    params: dict[str, torch.nn.Parameter],
) -> dict[str, torch.optim.Optimizer]:
    return {
        name: torch.optim.Adam(
            [{"params": [parameter], "lr": 1e-3, "name": name}],
            foreach=False,
            fused=False,
            eps=1e-15,
        )
        for name, parameter in params.items()
    }


def _observe_artificial_step(
    controller: birth.ResponsibilityBirthController,
    *,
    step: int,
    view: int,
    n: int,
) -> None:
    attempts = 6
    means2d = torch.nn.Parameter(
        torch.stack([torch.arange(n, dtype=torch.float32), torch.ones(n)], dim=-1)
    )
    basis = torch.nn.Parameter(
        torch.stack(
            [
                torch.linspace(0.1, 0.8, n),
                torch.linspace(0.2, 0.9, n),
                torch.linspace(0.3, 1.0, n),
            ],
            dim=-1,
        )
    )
    raw = torch.arange(attempts * n, dtype=torch.float32).reshape(attempts, n) + 1 + view
    weights = raw / (raw.sum(dim=1, keepdim=True) * 2.0)
    position_term = (weights @ means2d[:, :1]).expand(-1, 3) * 1e-3
    color = weights @ basis + position_term
    alpha = weights.sum(dim=1)
    output = PointRenderOutput(
        color=color,
        alpha=alpha,
        depth=torch.zeros(attempts),
        visible=torch.arange(n),
        means2d=means2d,
        compositing_color_basis=basis,
    )
    point_loss = (color.detach() - 0.25).square().mean(dim=-1)
    active = torch.tensor([True, False, True, True, False, True])
    controller.observe_pre_backward(
        step=step,
        view_index=view,
        output=output,
        point_loss=point_loss,
        active=active,
        attempts=attempts,
    )
    assert output.compositing_color_basis is None
    color.square().mean().backward()
    controller.observe_post_backward(
        step=step,
        view_index=view,
        output=output,
        width=64,
        height=48,
    )


def test_controller_vjp_accumulation_and_smoke_surgery() -> None:
    n = 8
    params = _params(n)
    optimizers = _optimizers(params)
    pre: list[int] = []
    post: list[int] = []
    controller = birth.ResponsibilityBirthController(
        arm="R",
        split_root=993002,
        shuffle_root=993003,
        score_steps=2,
        quota_per_stratum=1,
        on_pre_surgery=lambda snapshot, _: pre.append(snapshot.n),
        on_post_surgery=lambda snapshot, _: post.append(snapshot.n),
    )
    controller.bind(
        params,
        optimizers,
        extent=1.0,
        n_views=2,
        attempts_per_step=6,
    )
    _observe_artificial_step(controller, step=1, view=0, n=n)
    unchanged = controller.after_step(
        step=1,
        params=params,
        optimizers=optimizers,
        snapshot=birth._params_to_gaussians(params).detach(),
    )
    assert unchanged is params
    _observe_artificial_step(controller, step=2, view=1, n=n)
    changed = controller.after_step(
        step=2,
        params=params,
        optimizers=optimizers,
        snapshot=birth._params_to_gaussians(params).detach(),
    )
    history = controller.history_record()
    assert pre == [8]
    assert post == [12]
    assert changed["means"].shape[0] == 12
    assert controller.persistent_ids.shape == (12,)
    assert len(history["selection"]["selected_rows"]["R"]) == 4
    assert history["surgery"]["receipt"]["n_before"] == 8
    assert history["surgery"]["receipt"]["n_after"] == 12
    assert len(history["lineage"]) == 6
    assert len(history["step_evidence"]) == 2
    for evidence in history["step_evidence"]:
        assert evidence["native_residual_sum"] == pytest.approx(
            evidence["alpha_residual_sum"], rel=2e-5, abs=2e-6
        )
        assert evidence["native_support_sum"] == pytest.approx(
            evidence["alpha_support_sum"], rel=2e-5, abs=2e-6
        )


def test_terminal_decision_ordering() -> None:
    def arm(q: float, u: float) -> dict[str, object]:
        return {
            "checkpoint_metrics": {key: {"J_Q": q, "J_U": u} for key in birth.RECOVERY_CHECKPOINTS}
        }

    records = {
        root: {"R": arm(0.8, 1.0), "G": arm(1.0, 1.0), "U": arm(1.0, 1.0)} for root in (1, 2, 3)
    }
    result = birth.compute_terminal_decision(
        records,
        [0.97] * 21,
        roots=(1, 2, 3),
    )
    assert result["scientific_decision"] == "RESIDUAL_RESPONSIBILITY_ALLOCATION_PROMISING"
    unavailable = birth.compute_terminal_decision(
        records,
        [0.94] * 21,
        roots=(1, 2, 3),
    )
    assert unavailable["scientific_decision"] == "UNAVAILABLE"
