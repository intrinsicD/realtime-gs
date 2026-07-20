"""CPU tests for transactional exact-objective field topology moves."""

from __future__ import annotations

import math
import pickle
from collections.abc import Sequence

import pytest

from rtgs.lift.field_topology import (
    DeterministicTopologyScheduler,
    FieldComponent,
    FieldComponentPayload,
    FieldTopologyState,
    MoveProposal,
    SourceAnchor,
    SourceLineage,
    TopologyOps,
    propose_birth,
    propose_merge,
    propose_prune,
    propose_split,
)


def _payload(
    *,
    row: int,
    x: float,
    mass: float = 1.0,
    depth: float = 2.0,
    opacity: float = 0.4,
    color: tuple[float, float, float] = (0.8, 0.2, 0.1),
) -> FieldComponentPayload:
    source = SourceLineage(0, row)
    return FieldComponentPayload(
        source_lineage=(source,),
        source_anchor=SourceAnchor(source, (x, 0.0)),
        depth=depth,
        cross=(0.1, -0.2),
        log_ray_scale=math.log(0.3),
        density_mass=mass,
        source_color=color,
        render_opacity=opacity,
    )


def _component(stable_id: int, **kwargs: object) -> FieldComponent:
    return FieldComponent(stable_id, _payload(**kwargs))


_SAMPLES = tuple(-1.5 + index * 0.125 for index in range(25))
_WIDTH = 0.24


def _field(state: FieldTopologyState) -> tuple[float, ...]:
    """Tiny exact additive field fixture, intentionally independent of field_loss.py."""

    return tuple(
        sum(
            component.density_mass
            * component.source_color[0]
            * math.exp(-0.5 * ((sample - component.source_anchor.xy[0]) / _WIDTH) ** 2)
            for component in state.components
        )
        for sample in _SAMPLES
    )


def _objective(target: tuple[float, ...]):
    def evaluate(state: FieldTopologyState) -> float:
        return sum(
            (predicted - expected) ** 2 for predicted, expected in zip(_field(state), target)
        )

    return evaluate


def test_duplicate_merge_preserves_field_and_uses_mass_not_opacity() -> None:
    duplicate = FieldTopologyState(
        (
            _component(4, row=4, x=0.0, mass=3.0, depth=2.0, opacity=0.01),
            _component(9, row=9, x=0.0, mass=1.0, depth=2.0, opacity=0.99),
        ),
        next_stable_id=10,
    )
    target = _field(duplicate)
    proposal = propose_merge(duplicate, 9, 4, tag="duplicate")
    merged = proposal.add_components[0]

    assert merged.stable_id == 4
    assert merged.source_anchor == duplicate.component(4).source_anchor
    assert merged.source_color == duplicate.component(4).source_color
    assert merged.density_mass == pytest.approx(4.0)
    assert merged.render_opacity == pytest.approx(0.9901)
    assert set(merged.source_lineage) == {
        duplicate.component(4).source_anchor.source,
        duplicate.component(9).source_anchor.source,
    }

    result = DeterministicTopologyScheduler(
        _objective(target),
        parsimony_per_component=0.05,
    ).try_move(duplicate, proposal)
    assert result.receipts[0].exact_candidate == pytest.approx(0.0, abs=1e-28)
    assert result.receipts[0].accepted
    assert result.state.stable_ids == (4,)
    assert _field(result.state) == pytest.approx(target)

    unequal = FieldTopologyState(
        (
            _component(1, row=1, x=0.0, mass=9.0, depth=1.0, opacity=0.01),
            _component(2, row=2, x=0.0, mass=1.0, depth=5.0, opacity=0.99),
        )
    )
    mass_merge = propose_merge(unequal, 1, 2).add_components[0]
    assert mass_merge.depth == pytest.approx(1.0)
    assert mass_merge.cross == unequal.component(1).cross
    assert mass_merge.log_ray_scale == unequal.component(1).log_ray_scale
    assert mass_merge.source_anchor == unequal.component(1).source_anchor


def test_planted_birth_is_accepted_from_explicit_unexplained_candidate() -> None:
    planted = FieldTopologyState(
        (
            _component(0, row=0, x=-0.65, mass=1.0),
            _component(1, row=1, x=0.70, mass=0.8, color=(0.5, 0.1, 0.0)),
        )
    )
    initial = FieldTopologyState((planted.component(0),), next_stable_id=1)
    proposal = propose_birth(initial, planted.component(1).payload, tag="unexplained-peak")
    assert proposal.add_ids == (1,)

    result = DeterministicTopologyScheduler(
        _objective(_field(planted)),
        parsimony_per_component=1e-4,
    ).try_move(initial, proposal)
    assert result.receipts[0].accepted
    assert result.receipts[0].reason == "strict_decrease"
    assert result.state == planted


def test_prune_acceptance_and_rejection_are_transactional() -> None:
    desired = _component(0, row=0, x=-0.4)
    spurious = _component(1, row=1, x=0.8, mass=0.7)
    state = FieldTopologyState((desired, spurious))
    scheduler = DeterministicTopologyScheduler(
        _objective(_field(FieldTopologyState((desired,)))),
        parsimony_per_component=1e-3,
    )

    accepted = scheduler.try_move(state, propose_prune(state, 1, tag="spurious"))
    assert accepted.receipts[0].accepted
    assert accepted.state.components == (desired,)

    before = pickle.dumps(accepted.state, protocol=5)
    rejected = scheduler.try_move(
        accepted.state,
        propose_prune(accepted.state, 0, tag="supported"),
    )
    assert not rejected.receipts[0].accepted
    assert rejected.receipts[0].reason == "not_improving"
    assert rejected.state is accepted.state
    assert pickle.dumps(rejected.state, protocol=5) == before


def test_colocated_split_rejects_but_residual_directed_split_accepts() -> None:
    parent = _component(0, row=0, x=0.0, mass=2.0, opacity=0.64)
    state = FieldTopologyState((parent,))
    same_field_scheduler = DeterministicTopologyScheduler(
        _objective(_field(state)),
        parsimony_per_component=0.02,
    )
    before = pickle.dumps(state, protocol=5)
    co_located = propose_split(state, 0, mass_fraction=0.25, tag="co-located")
    left, right = co_located.add_components
    assert left.density_mass + right.density_mass == pytest.approx(parent.density_mass)
    assert 1.0 - (1.0 - left.render_opacity) * (1.0 - right.render_opacity) == pytest.approx(
        parent.render_opacity
    )
    rejected = same_field_scheduler.try_move(state, co_located)
    assert not rejected.receipts[0].accepted
    assert rejected.receipts[0].exact_candidate == pytest.approx(0.0, abs=1e-28)
    assert rejected.state is state
    assert pickle.dumps(rejected.state, protocol=5) == before

    left_source = SourceLineage(0, 10)
    right_source = SourceLineage(0, 11)
    anchors = (
        SourceAnchor(left_source, (-0.55, 0.0)),
        SourceAnchor(right_source, (0.55, 0.0)),
    )
    split_target = FieldTopologyState(
        (
            FieldComponent(
                0,
                _payload(row=10, x=-0.55, mass=1.0, opacity=left.render_opacity),
            ),
            FieldComponent(
                1,
                _payload(row=11, x=0.55, mass=1.0, opacity=right.render_opacity),
            ),
        )
    )
    directed = propose_split(
        state,
        0,
        mass_fraction=0.5,
        child_anchors=anchors,
        tag="residual-peaks",
    )
    accepted = DeterministicTopologyScheduler(
        _objective(_field(split_target)),
        parsimony_per_component=1e-4,
    ).try_move(state, directed)
    assert accepted.receipts[0].accepted
    assert tuple(component.source_anchor for component in accepted.state.components) == anchors


class _StaticOps:
    def __init__(self, proposals: Sequence[MoveProposal]) -> None:
        self._proposals = tuple(proposals)

    def proposals(self, state: FieldTopologyState) -> Sequence[MoveProposal]:
        return self._proposals


def test_scheduler_receipts_are_deterministic_across_proposal_order() -> None:
    desired = _component(0, row=0, x=0.0)
    weak = _component(1, row=1, x=-0.8, mass=0.2)
    strong = _component(2, row=2, x=0.8, mass=0.8)
    state = FieldTopologyState((desired, weak, strong))
    target = _field(FieldTopologyState((desired,)))
    proposals = (
        propose_prune(state, 1, tag="weak"),
        propose_prune(state, 2, tag="strong"),
    )
    scheduler = DeterministicTopologyScheduler(
        _objective(target),
        parsimony_per_component=1e-3,
    )
    assert isinstance(_StaticOps(proposals), TopologyOps)

    forward = scheduler.run(state, _StaticOps(proposals), max_rounds=1)
    reverse = scheduler.run(state, _StaticOps(tuple(reversed(proposals))), max_rounds=1)
    assert forward == reverse
    assert forward.state.stable_ids == (0, 1)
    assert [receipt.proposal.tag for receipt in forward.receipts] == ["weak", "strong"]
    assert [receipt.reason for receipt in forward.receipts] == ["superseded", "strict_decrease"]
