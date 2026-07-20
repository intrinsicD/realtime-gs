"""Deterministic exact-objective topology moves for field-based Gaussian lifting.

This module deliberately does not define a field loss.  Callers inject the exact objective
that matches their observation provider, while this module owns immutable topology state,
mass-preserving move construction, parsimony, and deterministic accept/reject receipts.

``density_mass`` and ``render_opacity`` are separate quantities throughout.  Density mass is
the capacity used for moment matching and source-representative selection.  Opacity is only a
rendering parameter; it is never used as a correspondence confidence or merge weight.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol, runtime_checkable

MoveKind = Literal["prune", "merge", "split", "birth"]
ExactObjective = Callable[["FieldTopologyState"], float]

_MOVE_ORDER: dict[MoveKind, int] = {
    "prune": 0,
    "merge": 1,
    "split": 2,
    "birth": 3,
}


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite_tuple(name: str, value: Sequence[float], length: int) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True, order=True)
class SourceLineage:
    """Stable identifier of one source-view observation component."""

    view_index: int
    component_index: int

    def __post_init__(self) -> None:
        _require_int("view_index", self.view_index)
        _require_int("component_index", self.component_index)


@dataclass(frozen=True, order=True)
class SourceAnchor:
    """The source component and image-plane position whose fiber anchors a track."""

    source: SourceLineage
    xy: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceLineage):
            raise TypeError("source must be SourceLineage")
        object.__setattr__(self, "xy", _finite_tuple("xy", self.xy, 2))


@dataclass(frozen=True)
class FieldComponentPayload:
    """Immutable field-lift parameters carried by one persistent component.

    ``depth``, ``cross``, and ``log_ray_scale`` are the four source-fiber coordinates.
    ``density_mass`` participates in topology moments. ``render_opacity`` does not.
    """

    source_lineage: tuple[SourceLineage, ...]
    source_anchor: SourceAnchor
    depth: float
    cross: tuple[float, float]
    log_ray_scale: float
    density_mass: float
    source_color: tuple[float, float, float]
    render_opacity: float

    def __post_init__(self) -> None:
        lineage = tuple(sorted(set(self.source_lineage)))
        if not lineage or any(not isinstance(item, SourceLineage) for item in lineage):
            raise ValueError("source_lineage must contain at least one SourceLineage")
        if not isinstance(self.source_anchor, SourceAnchor):
            raise TypeError("source_anchor must be SourceAnchor")
        if self.source_anchor.source not in lineage:
            raise ValueError("source anchor must be represented in source_lineage")
        depth = float(self.depth)
        log_ray_scale = float(self.log_ray_scale)
        density_mass = float(self.density_mass)
        render_opacity = float(self.render_opacity)
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError("depth must be finite and positive")
        if not math.isfinite(log_ray_scale):
            raise ValueError("log_ray_scale must be finite")
        if not math.isfinite(density_mass) or density_mass <= 0.0:
            raise ValueError("density_mass must be finite and positive")
        if not math.isfinite(render_opacity) or not 0.0 <= render_opacity <= 1.0:
            raise ValueError("render_opacity must be finite and in [0,1]")
        object.__setattr__(self, "source_lineage", lineage)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "cross", _finite_tuple("cross", self.cross, 2))
        object.__setattr__(self, "log_ray_scale", log_ray_scale)
        object.__setattr__(self, "density_mass", density_mass)
        object.__setattr__(
            self, "source_color", _finite_tuple("source_color", self.source_color, 3)
        )
        object.__setattr__(self, "render_opacity", render_opacity)

    def canonical_key(self) -> tuple[object, ...]:
        """Return a total, deterministic ordering key."""

        return (
            tuple((item.view_index, item.component_index) for item in self.source_lineage),
            self.source_anchor.source.view_index,
            self.source_anchor.source.component_index,
            self.source_anchor.xy,
            self.depth,
            self.cross,
            self.log_ray_scale,
            self.density_mass,
            self.source_color,
            self.render_opacity,
        )


@dataclass(frozen=True)
class FieldComponent:
    """One persistent component and its immutable field-lift payload."""

    stable_id: int
    payload: FieldComponentPayload

    def __post_init__(self) -> None:
        _require_int("stable_id", self.stable_id)
        if not isinstance(self.payload, FieldComponentPayload):
            raise TypeError("payload must be FieldComponentPayload")

    @property
    def source_lineage(self) -> tuple[SourceLineage, ...]:
        return self.payload.source_lineage

    @property
    def source_anchor(self) -> SourceAnchor:
        return self.payload.source_anchor

    @property
    def depth(self) -> float:
        return self.payload.depth

    @property
    def cross(self) -> tuple[float, float]:
        return self.payload.cross

    @property
    def log_ray_scale(self) -> float:
        return self.payload.log_ray_scale

    @property
    def density_mass(self) -> float:
        return self.payload.density_mass

    @property
    def source_color(self) -> tuple[float, float, float]:
        return self.payload.source_color

    @property
    def render_opacity(self) -> float:
        return self.payload.render_opacity

    def canonical_key(self) -> tuple[object, ...]:
        return (self.stable_id, *self.payload.canonical_key())


@dataclass(frozen=True)
class FieldTopologyState:
    """Canonical immutable collection of components and a monotonic ID allocator."""

    components: tuple[FieldComponent, ...]
    next_stable_id: int | None = None

    def __post_init__(self) -> None:
        components = tuple(sorted(self.components, key=lambda item: item.stable_id))
        if any(not isinstance(item, FieldComponent) for item in components):
            raise TypeError("components must contain FieldComponent values")
        ids = tuple(item.stable_id for item in components)
        if len(set(ids)) != len(ids):
            raise ValueError("component stable IDs must be unique")
        minimum_next = max(ids, default=-1) + 1
        next_stable_id = minimum_next if self.next_stable_id is None else self.next_stable_id
        _require_int("next_stable_id", next_stable_id)
        if next_stable_id < minimum_next:
            raise ValueError("next_stable_id must exceed every existing stable ID")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "next_stable_id", next_stable_id)

    @property
    def stable_ids(self) -> tuple[int, ...]:
        return tuple(component.stable_id for component in self.components)

    def component(self, stable_id: int) -> FieldComponent:
        _require_int("stable_id", stable_id)
        for component in self.components:
            if component.stable_id == stable_id:
                return component
        raise KeyError(f"unknown stable ID {stable_id}")


@dataclass(frozen=True)
class MoveProposal:
    """A transactional replacement of stable IDs by immutable candidate components."""

    kind: MoveKind
    remove_ids: tuple[int, ...]
    add_components: tuple[FieldComponent, ...]
    tag: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _MOVE_ORDER:
            raise ValueError(f"unknown move kind {self.kind!r}")
        remove_ids = tuple(sorted(self.remove_ids))
        if any(type(item) is not int or item < 0 for item in remove_ids):
            raise ValueError("remove_ids must contain non-negative integers")
        if len(set(remove_ids)) != len(remove_ids):
            raise ValueError("remove_ids must be unique")
        additions = tuple(sorted(self.add_components, key=lambda item: item.stable_id))
        if any(not isinstance(item, FieldComponent) for item in additions):
            raise TypeError("add_components must contain FieldComponent values")
        addition_ids = tuple(item.stable_id for item in additions)
        if len(set(addition_ids)) != len(addition_ids):
            raise ValueError("added component stable IDs must be unique")
        if not isinstance(self.tag, str):
            raise TypeError("tag must be a string")
        object.__setattr__(self, "remove_ids", remove_ids)
        object.__setattr__(self, "add_components", additions)

    @property
    def add_ids(self) -> tuple[int, ...]:
        return tuple(component.stable_id for component in self.add_components)

    def canonical_key(self) -> tuple[object, ...]:
        return (
            _MOVE_ORDER[self.kind],
            self.remove_ids,
            tuple(component.canonical_key() for component in self.add_components),
            self.tag,
        )


@dataclass(frozen=True)
class MoveReceipt:
    """Deterministic evidence for one exact objective comparison."""

    proposal: MoveProposal
    accepted: bool
    reason: Literal["strict_decrease", "not_improving", "superseded"]
    exact_before: float
    exact_candidate: float
    penalized_before: float
    penalized_candidate: float
    result_ids: tuple[int, ...]


@dataclass(frozen=True)
class TopologyScheduleResult:
    """Final state and canonical receipts from one or more scheduler rounds."""

    state: FieldTopologyState
    receipts: tuple[MoveReceipt, ...]


@runtime_checkable
class TopologyOps(Protocol):
    """Candidate generator consumed by :class:`DeterministicTopologyScheduler`."""

    def proposals(self, state: FieldTopologyState) -> Sequence[MoveProposal]:
        """Return topology candidates for the current immutable state."""
        ...


def _apply_proposal(state: FieldTopologyState, proposal: MoveProposal) -> FieldTopologyState:
    existing_ids = set(state.stable_ids)
    remove_ids = set(proposal.remove_ids)
    missing = remove_ids - existing_ids
    if missing:
        raise KeyError(f"proposal removes unknown stable IDs {sorted(missing)}")
    survivor_ids = existing_ids - remove_ids
    collisions = survivor_ids.intersection(proposal.add_ids)
    if collisions:
        raise ValueError(f"proposal additions collide with surviving IDs {sorted(collisions)}")
    survivors = tuple(
        component for component in state.components if component.stable_id not in remove_ids
    )
    maximum_added = max(proposal.add_ids, default=-1) + 1
    return FieldTopologyState(
        survivors + proposal.add_components,
        next_stable_id=max(int(state.next_stable_id), maximum_added),
    )


def propose_prune(
    state: FieldTopologyState,
    stable_id: int,
    *,
    tag: str = "",
) -> MoveProposal:
    """Propose deleting one component."""

    state.component(stable_id)
    return MoveProposal("prune", (stable_id,), (), tag)


def propose_merge(
    state: FieldTopologyState,
    left_id: int,
    right_id: int,
    *,
    tag: str = "",
) -> MoveProposal:
    """Propose a mass merge on one deterministic representative source fiber.

    The greater-mass component supplies the persistent ID and source anchor. Exact mass ties
    use source lineage and then stable ID. Fiber coordinates are not averaged because depths,
    shears, and ray scales from different source cameras live in different bases. The exact
    objective decides whether retaining the representative geometry is acceptable. Render
    opacity never chooses the representative; coincident coverage uses the union formula.
    """

    if left_id == right_id:
        raise ValueError("merge requires two distinct stable IDs")
    left = state.component(left_id)
    right = state.component(right_id)
    representative = min(
        (left, right),
        key=lambda item: (
            -item.density_mass,
            item.source_anchor,
            item.stable_id,
        ),
    )
    total_mass = left.density_mass + right.density_mass

    opacity = 1.0 - (1.0 - left.render_opacity) * (1.0 - right.render_opacity)
    payload = FieldComponentPayload(
        source_lineage=tuple(sorted(set(left.source_lineage + right.source_lineage))),
        source_anchor=representative.source_anchor,
        depth=representative.depth,
        cross=representative.cross,
        log_ray_scale=representative.log_ray_scale,
        density_mass=total_mass,
        # The persistent source fiber keeps one *observed* directional color exactly. Other
        # lineage colors remain cross-view supervision; averaging them here would invent an
        # unobserved anchor and break the source-ground-truth invariant.
        source_color=representative.source_color,
        render_opacity=min(max(opacity, 0.0), 1.0),
    )
    merged = FieldComponent(representative.stable_id, payload)
    return MoveProposal("merge", (left_id, right_id), (merged,), tag)


def _partition_opacity(opacity: float, fraction: float) -> tuple[float, float]:
    if opacity <= 0.0:
        return 0.0, 0.0
    if opacity >= 1.0:
        return 1.0, 1.0
    log_transmittance = math.log1p(-opacity)
    left = -math.expm1(fraction * log_transmittance)
    right = -math.expm1((1.0 - fraction) * log_transmittance)
    return left, right


def propose_split(
    state: FieldTopologyState,
    stable_id: int,
    *,
    mass_fraction: float = 0.5,
    child_anchors: tuple[SourceAnchor, SourceAnchor] | None = None,
    tag: str = "",
) -> MoveProposal:
    """Propose a two-child mass partition.

    Without ``child_anchors`` this is an exact co-located partition of additive density mass;
    opacity is partitioned in optical-thickness space so the two-child alpha union also equals
    the parent. Residual-directed splitting supplies two explicit source anchors.
    """

    parent = state.component(stable_id)
    fraction = float(mass_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("mass_fraction must be finite and strictly between zero and one")
    if child_anchors is None:
        anchors = (parent.source_anchor, parent.source_anchor)
    else:
        if len(child_anchors) != 2 or any(
            not isinstance(anchor, SourceAnchor) for anchor in child_anchors
        ):
            raise ValueError("child_anchors must contain exactly two SourceAnchor values")
        anchors = child_anchors
    opacities = _partition_opacity(parent.render_opacity, fraction)
    fractions = (fraction, 1.0 - fraction)
    child_ids = (parent.stable_id, int(state.next_stable_id))
    children = []
    for child_id, child_fraction, opacity, anchor in zip(
        child_ids,
        fractions,
        opacities,
        anchors,
    ):
        lineage = tuple(sorted(set(parent.source_lineage + (anchor.source,))))
        payload = replace(
            parent.payload,
            source_lineage=lineage,
            source_anchor=anchor,
            density_mass=parent.density_mass * child_fraction,
            render_opacity=opacity,
        )
        children.append(FieldComponent(child_id, payload))
    return MoveProposal("split", (stable_id,), tuple(children), tag)


def propose_birth(
    state: FieldTopologyState,
    unexplained_candidate: FieldComponentPayload,
    *,
    tag: str = "",
) -> MoveProposal:
    """Propose one explicit unexplained candidate using the next persistent ID."""

    if not isinstance(unexplained_candidate, FieldComponentPayload):
        raise TypeError("unexplained_candidate must be FieldComponentPayload")
    component = FieldComponent(int(state.next_stable_id), unexplained_candidate)
    return MoveProposal("birth", (), (component,), tag)


class DeterministicTopologyScheduler:
    """Evaluate exact move deltas and accept at most one canonical best move per round."""

    def __init__(
        self,
        exact_objective: ExactObjective,
        *,
        parsimony_per_component: float,
    ) -> None:
        if not callable(exact_objective):
            raise TypeError("exact_objective must be callable")
        parsimony = float(parsimony_per_component)
        if not math.isfinite(parsimony) or parsimony < 0.0:
            raise ValueError("parsimony_per_component must be finite and non-negative")
        self._exact_objective = exact_objective
        self.parsimony_per_component = parsimony

    def _objective(self, state: FieldTopologyState) -> float:
        value = float(self._exact_objective(state))
        if not math.isfinite(value):
            raise ValueError("exact objective callback must return a finite scalar")
        return value

    def step(
        self,
        state: FieldTopologyState,
        proposals: Sequence[MoveProposal],
    ) -> TopologyScheduleResult:
        """Evaluate a proposal set in canonical order and accept its best strict decrease."""

        ordered = tuple(sorted(proposals, key=lambda proposal: proposal.canonical_key()))
        if any(not isinstance(proposal, MoveProposal) for proposal in ordered):
            raise TypeError("proposals must contain MoveProposal values")
        if not ordered:
            return TopologyScheduleResult(state, ())

        exact_before = self._objective(state)
        penalized_before = exact_before + self.parsimony_per_component * len(state.components)
        candidates: list[tuple[MoveProposal, FieldTopologyState, float, float]] = []
        for proposal in ordered:
            candidate = _apply_proposal(state, proposal)
            exact_candidate = self._objective(candidate)
            penalized_candidate = exact_candidate + self.parsimony_per_component * len(
                candidate.components
            )
            candidates.append((proposal, candidate, exact_candidate, penalized_candidate))

        best_index = min(
            range(len(candidates)),
            key=lambda index: (candidates[index][3], candidates[index][0].canonical_key()),
        )
        best_improves = candidates[best_index][3] < penalized_before
        result_state = candidates[best_index][1] if best_improves else state
        receipts = []
        for index, (proposal, candidate, exact_candidate, penalized_candidate) in enumerate(
            candidates
        ):
            accepted = best_improves and index == best_index
            if accepted:
                reason: Literal["strict_decrease", "not_improving", "superseded"] = (
                    "strict_decrease"
                )
            elif penalized_candidate >= penalized_before:
                reason = "not_improving"
            else:
                reason = "superseded"
            receipts.append(
                MoveReceipt(
                    proposal=proposal,
                    accepted=accepted,
                    reason=reason,
                    exact_before=exact_before,
                    exact_candidate=exact_candidate,
                    penalized_before=penalized_before,
                    penalized_candidate=penalized_candidate,
                    result_ids=candidate.stable_ids if accepted else state.stable_ids,
                )
            )
        return TopologyScheduleResult(result_state, tuple(receipts))

    def try_move(
        self,
        state: FieldTopologyState,
        proposal: MoveProposal,
    ) -> TopologyScheduleResult:
        """Evaluate one transactional move."""

        return self.step(state, (proposal,))

    def run(
        self,
        state: FieldTopologyState,
        operations: TopologyOps,
        *,
        max_rounds: int = 1,
    ) -> TopologyScheduleResult:
        """Regenerate candidates after every accepted move, stopping at a local optimum."""

        _require_int("max_rounds", max_rounds, minimum=1)
        current = state
        receipts: list[MoveReceipt] = []
        for _round in range(max_rounds):
            proposals = tuple(operations.proposals(current))
            result = self.step(current, proposals)
            receipts.extend(result.receipts)
            if result.state is current:
                break
            current = result.state
        return TopologyScheduleResult(current, tuple(receipts))


__all__ = [
    "DeterministicTopologyScheduler",
    "ExactObjective",
    "FieldComponent",
    "FieldComponentPayload",
    "FieldTopologyState",
    "MoveKind",
    "MoveProposal",
    "MoveReceipt",
    "SourceAnchor",
    "SourceLineage",
    "TopologyOps",
    "TopologyScheduleResult",
    "propose_birth",
    "propose_merge",
    "propose_prune",
    "propose_split",
]
