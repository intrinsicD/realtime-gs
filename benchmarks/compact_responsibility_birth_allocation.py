"""Sealed compact residual-responsibility birth-allocation experiment.

The public ``seal``, ``phase-a`` and ``phase-b`` commands implement the once-only
lifecycle frozen in
``20260717_compact_responsibility_birth_allocation_PREREG.md``.  The useful mechanism
code is intentionally kept importable: focused CPU tests exercise the exact seed
derivation, score contraction, matched-stratum allocation, and topology controller
without consuming an official seed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import tarfile
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from benchmarks import compact_occupancy_refinement_factorial as factorial

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianPointProposal,
    ObservationSamples,
)
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer
from rtgs.optim.density import SelectedBirthReceipt, apply_selected_birth_surgery
from rtgs.render.point_base import PointRenderOutput
from rtgs.render.torch_points import TorchPointRasterizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
PREREGISTRATION = RESULTS / "20260717_compact_responsibility_birth_allocation_PREREG.md"
PREREGISTRATION_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_PREREG_REVIEW.md"
)
PREREGISTRATION_INITIAL_FAIL = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_PREREG_REVIEW_INITIAL_FAIL.md"
)
PREREGISTRATION_ADDENDUM_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_PREREG_REVIEW_ADDENDUM_1.md"
)
IMPLEMENTATION_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_IMPLEMENTATION_REVIEW.md"
)
SEAL = RESULTS / "20260717_compact_responsibility_birth_allocation_SEAL.json"
PHASE_A_ATTEMPT = RESULTS / "20260717_compact_responsibility_birth_allocation_PHASE_A_ATTEMPT.json"
PHASE_A_RESULT = RESULTS / "20260717_compact_responsibility_birth_allocation_PHASE_A_RESULT.json"
PHASE_A_AUDIT = RESULTS / "20260717_compact_responsibility_birth_allocation_PHASE_A_AUDIT.json"
PHASE_B_ATTEMPT = RESULTS / "20260717_compact_responsibility_birth_allocation_PHASE_B_ATTEMPT.json"
RESULT = RESULTS / "20260717_compact_responsibility_birth_allocation_RESULT.json"
EXECUTED_SOURCES = RESULTS / "20260717_compact_responsibility_birth_allocation_EXECUTED_SOURCES.tar"
RUN_DIR = ROOT / "runs/compact_responsibility_birth_allocation_20260717"

TEACHER_BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
PROXY_BUNDLE = ROOT / "runs/compact_occupancy_scalar_ablation_20260717/proxy_bundles/center"
INIT_PLY = ROOT / "runs/compact_occupancy_scalar_ablation_20260717/stage_b/center/gaussians.ply"
EXPECTED_INIT_PLY_SHA256 = "0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e"
EXPECTED_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
EXPLICIT_EXTENT = 1.5469313859939577

TRAIN_ROOTS = (77001, 77002, 77003)
EVALUATION_ROOTS = (77101, 77102, 77103)
SPLIT_ROOTS = (77201, 77202, 77203)
SHUFFLE_ROOTS = (77301, 77302, 77303)
FOCUSED_TRAIN_ROOTS = (993001, 993002, 993003)
FOCUSED_EVALUATION_ROOTS = (993101, 993102)
OFFICIAL_ROOTS = frozenset(TRAIN_ROOTS + EVALUATION_ROOTS + SPLIT_ROOTS + SHUFFLE_ROOTS)
FOCUSED_ROOTS = frozenset(FOCUSED_TRAIN_ROOTS + FOCUSED_EVALUATION_ROOTS)

ARMS = ("G", "R", "U")
ARM_ORDER = {
    77001: ("G", "R", "U"),
    77002: ("R", "U", "G"),
    77003: ("U", "G", "R"),
}
STRATA = ("small_low", "small_high", "large_low", "large_high")
STRATUM_CODES = {name: index for index, name in enumerate(STRATA)}
SCORE_STEPS = 35
TRAIN_ITERATIONS = 140
TRAIN_ATTEMPTS = 128
BANK_ATTEMPTS = 4096
QUOTA_PER_STRATUM = 8
CHECKPOINTS = (0, 35, 70, 105, 140)
RECOVERY_CHECKPOINTS = ("35_post", "70", "105", "140")
EVALUATION_SEED_DOMAIN = "rtgs.compact-responsibility-birth.eval.v1"
DOMAIN_PREFIX = b"rtgs.compact-responsibility-birth.v1\0"
FOCUSED_TEST_ENV = "RTGS_RESPONSIBILITY_BIRTH_FOCUSED_TEST"
PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33")
WORKER_TIMEOUT_SECONDS = 900

SOURCE_PATHS = (
    Path("benchmarks/compact_responsibility_birth_allocation.py"),
    Path("benchmarks/compact_occupancy_refinement_factorial.py"),
    Path("tests/test_compact_responsibility_birth_allocation.py"),
    Path("tests/test_compact_trainer.py"),
    Path("tests/test_point_render.py"),
    Path("src/rtgs/optim/compact_trainer.py"),
    Path("src/rtgs/optim/density.py"),
    Path("src/rtgs/render/point_base.py"),
    Path("src/rtgs/render/torch_points.py"),
    Path("src/rtgs/core/observation2d.py"),
    Path("src/rtgs/core/gaussians3d.py"),
    Path("src/rtgs/data/reconstruction_inputs.py"),
    PREREGISTRATION.relative_to(ROOT),
    PREREGISTRATION_REVIEW.relative_to(ROOT),
    PREREGISTRATION_INITIAL_FAIL.relative_to(ROOT),
    PREREGISTRATION_ADDENDUM_REVIEW.relative_to(ROOT),
    IMPLEMENTATION_REVIEW.relative_to(ROOT),
)

# The frozen protocol explicitly reuses the independently exercised iter3 denial
# layer.  Keeping the alias public also makes the boundary easy to audit.
RGBAccessGuard = factorial.RGBAccessGuard


class ProtocolInvalid(RuntimeError):
    """The executable state differs from the frozen protocol."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    detached = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(canonical_bytes(list(detached.shape)))
    digest.update(detached.numpy().tobytes(order="C"))
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(canonical_bytes(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def encode_atom(value: int | str) -> bytes:
    """Encode one frozen domain-seed atom, rejecting bools and integer aliases."""
    if type(value) is int and value >= 0:
        return str(value).encode("ascii")
    if type(value) is str:
        return value.encode("utf-8")
    raise TypeError("domain-seed atoms must be non-negative built-in ints or strings")


def domain_seed(label: str, root: int, *parts: int | str) -> int:
    payload = (
        DOMAIN_PREFIX
        + encode_atom(label)
        + b"\0"
        + encode_atom(root)
        + b"".join(b"\0" + encode_atom(part) for part in parts)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def evaluation_bank_seed(root: int, view_name: str, measure_name: str) -> int:
    if measure_name not in {"uniform", "proposal"}:
        raise ValueError("measure_name must be 'uniform' or 'proposal'")
    if not view_name:
        raise ValueError("view_name must be non-empty")
    return domain_seed("evaluation_bank", root, view_name, measure_name)


def split_seed(root: int) -> int:
    return domain_seed("split", root, 1)


def shuffle_seed(root: int, stratum: str) -> int:
    if stratum not in STRATUM_CODES:
        raise ValueError(f"unknown stratum {stratum!r}")
    return domain_seed("shuffle", root, 1, STRATUM_CODES[stratum])


def _reject_roots_for_mode(*roots: int, focused: bool) -> None:
    values = set(roots)
    forbidden = OFFICIAL_ROOTS if focused else FOCUSED_ROOTS
    overlap = sorted(values & forbidden)
    if overlap:
        mode = "focused" if focused else "official"
        raise ProtocolInvalid(f"{mode} mode rejected root(s): {overlap}")


def frozen_config(
    training_root: int,
    *,
    focused: bool = False,
    device: str = "cuda:0",
) -> CompactTrainConfig:
    allowed = FOCUSED_TRAIN_ROOTS if focused else TRAIN_ROOTS
    if training_root not in allowed:
        raise ValueError("training root is not in the selected frozen domain")
    _reject_roots_for_mode(training_root, focused=focused)
    return CompactTrainConfig(
        iterations=TRAIN_ITERATIONS,
        attempts_per_step=TRAIN_ATTEMPTS,
        proposal_mode="area_gaussian",
        schedule_mode="balanced_cycle",
        target_mode="proposal_attempt",
        uniform_fraction=0.25,
        seed=training_root,
        extent=EXPLICIT_EXTENT,
        device=device,
        lr_means=1.6e-4,
        lr_quats=1e-3,
        lr_scales=5e-3,
        lr_opacity=5e-2,
        lr_sh=2.5e-3,
        lr_sh_rest=1.25e-4,
        point_chunk=256,
        gaussian_chunk=256,
        outer_microbatch=128,
        query_component_chunk=640,
        teacher_tile_size=16,
        evaluation_chunk=256,
        checkpoints=CHECKPOINTS,
        evaluate_checkpoint_risks=False,
        sh_degree=0,
        sh_color_activation="hard",
        sh_smu1_mu=0.00784313725490196,
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
    )


def _finite_list(values: torch.Tensor) -> list[float]:
    if not bool(torch.isfinite(values).all()):
        raise ProtocolInvalid("score evidence contains a non-finite value")
    return [float(value) for value in values.detach().cpu().tolist()]


def _score_order(
    members: Sequence[int],
    score: torch.Tensor,
    persistent_ids: torch.Tensor,
) -> list[int]:
    return sorted(
        members,
        key=lambda row: (-float(score[row]), int(persistent_ids[row])),
    )


def _selection_hash(selected: Mapping[str, Sequence[int]]) -> str:
    return canonical_hash({key: list(selected[key]) for key in ARMS})


def build_matched_selections(
    *,
    gradient_score: torch.Tensor,
    residual_score: torch.Tensor,
    support_score: torch.Tensor,
    support_by_view: torch.Tensor,
    visible_step_count: torch.Tensor,
    scale_max: torch.Tensor,
    persistent_ids: torch.Tensor,
    extent: float,
    shuffle_root: int,
    quota_per_stratum: int = QUOTA_PER_STRATUM,
) -> dict[str, Any]:
    """Construct the frozen eligibility, strata and G/R/U parent allocations."""
    vectors = (
        gradient_score,
        residual_score,
        support_score,
        visible_step_count,
        scale_max,
        persistent_ids,
    )
    n = int(persistent_ids.numel())
    if any(value.ndim != 1 or value.numel() != n for value in vectors):
        raise ValueError("selection vectors must be aligned one-dimensional tensors")
    if support_by_view.ndim != 2 or support_by_view.shape[1] != n:
        raise ValueError("support_by_view must have shape (V,N)")
    if persistent_ids.dtype != torch.long or persistent_ids.unique().numel() != n:
        raise ValueError("persistent_ids must be unique int64 values")
    if quota_per_stratum <= 0:
        raise ValueError("quota_per_stratum must be positive")
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("extent must be finite and positive")

    finite = (
        torch.isfinite(gradient_score)
        & torch.isfinite(residual_score)
        & torch.isfinite(support_score)
    )
    positive_views = (support_by_view > 0).sum(dim=0)
    eligible_mask = finite & (support_score > 0) & (positive_views >= 2) & (visible_step_count > 0)
    eligible = eligible_mask.nonzero(as_tuple=True)[0].detach().cpu().tolist()
    boundary = 0.01 * extent
    small = {row for row in eligible if float(scale_max[row]) <= boundary}
    large = set(eligible) - small

    stratum_rows: dict[str, list[int]] = {}
    for family, members in (("small", small), ("large", large)):
        ordered = sorted(
            members,
            key=lambda row: (float(support_score[row]), int(persistent_ids[row])),
        )
        cut = len(ordered) // 2
        stratum_rows[f"{family}_low"] = ordered[:cut]
        stratum_rows[f"{family}_high"] = ordered[cut:]

    if set().union(*(set(rows) for rows in stratum_rows.values())) != set(eligible):
        raise ProtocolInvalid("matched strata are not exhaustive")
    if sum(len(rows) for rows in stratum_rows.values()) != len(eligible):
        raise ProtocolInvalid("matched strata overlap")

    selected: dict[str, list[int]] = {arm: [] for arm in ARMS}
    shuffle_records: dict[str, Any] = {}
    stratum_records: dict[str, Any] = {}
    for stratum in STRATA:
        members = stratum_rows[stratum]
        if len(members) < quota_per_stratum:
            raise ProtocolInvalid(
                f"stratum {stratum} has {len(members)} eligible rows, requires {quota_per_stratum}"
            )
        g_order = _score_order(members, gradient_score, persistent_ids)
        r_order = _score_order(members, residual_score, persistent_ids)
        selected["G"].extend(g_order[:quota_per_stratum])
        selected["R"].extend(r_order[:quota_per_stratum])

        recipients = sorted(members, key=lambda row: int(persistent_ids[row]))
        seed = shuffle_seed(shuffle_root, stratum)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = [
            int(value) for value in torch.randperm(len(recipients), generator=generator).tolist()
        ]
        # ``r_order`` is the canonical descending-R, ascending-ID rank.  Assigning
        # source rank indices rather than comparing the recipient ID preserves complete
        # tied rank labels and makes U a uniform quota subset.
        source_rank = {row: rank for rank, row in enumerate(r_order)}
        assignments = [
            {
                "recipient_row": recipient,
                "recipient_id": int(persistent_ids[recipient]),
                "source_row": recipients[source_index],
                "source_id": int(persistent_ids[recipients[source_index]]),
                "source_R": float(residual_score[recipients[source_index]]),
                "source_rank": source_rank[recipients[source_index]],
            }
            for recipient, source_index in zip(recipients, permutation, strict=True)
        ]
        u_winners = sorted(assignments, key=lambda item: item["source_rank"])[:quota_per_stratum]
        selected["U"].extend(item["recipient_row"] for item in u_winners)
        moved = sum(item["recipient_row"] != item["source_row"] for item in assignments)
        shuffle_records[stratum] = {
            "seed": seed,
            "recipients": recipients,
            "permutation": permutation,
            "assignments": assignments,
            "fixed_points": len(assignments) - moved,
            "moved": moved,
            "selected_rows": [item["recipient_row"] for item in u_winners],
        }
        stratum_records[stratum] = {
            "members": members,
            "member_ids": [int(persistent_ids[row]) for row in members],
            "gradient_order": g_order,
            "residual_order": r_order,
            "selected": {
                arm: [row for row in selected[arm] if row in set(members)] for arm in ARMS
            },
        }

    expected_count = len(STRATA) * quota_per_stratum
    for arm, rows in selected.items():
        if len(rows) != expected_count or len(set(rows)) != expected_count:
            raise ProtocolInvalid(f"arm {arm} selection count/uniqueness changed")
        if not set(rows) <= set(eligible):
            raise ProtocolInvalid(f"arm {arm} selected an ineligible row")

    # Materialization order is ascending persistent identity, independent of score order.
    selected = {
        arm: sorted(rows, key=lambda row: int(persistent_ids[row]))
        for arm, rows in selected.items()
    }
    overlaps = {
        f"{left}_{right}": {
            "intersection": len(set(selected[left]) & set(selected[right])),
            "jaccard": len(set(selected[left]) & set(selected[right]))
            / len(set(selected[left]) | set(selected[right])),
        }
        for left, right in (("R", "G"), ("R", "U"), ("G", "U"))
    }
    moved = sum(record["moved"] for record in shuffle_records.values())
    assigned = sum(len(record["assignments"]) for record in shuffle_records.values())
    score_sums = {
        arm: float(
            residual_score[torch.tensor(rows, device=residual_score.device, dtype=torch.long)].sum(
                dtype=torch.float64
            )
        )
        for arm, rows in selected.items()
    }
    gates = {
        "all_strata_have_quota": all(
            len(stratum_rows[name]) >= quota_per_stratum for name in STRATA
        ),
        "selection_count_and_match": all(
            len(rows) == expected_count and len(set(rows)) == expected_count
            for rows in selected.values()
        ),
        "G_distinct_positive_ge_16": len(
            {float(gradient_score[row]) for row in eligible if gradient_score[row] > 0}
        )
        >= 16,
        "R_distinct_positive_ge_16": len(
            {float(residual_score[row]) for row in eligible if residual_score[row] > 0}
        )
        >= 16,
        "jaccard_R_G_lt_0_80": overlaps["R_G"]["jaccard"] < 0.80,
        "jaccard_R_U_lt_0_80": overlaps["R_U"]["jaccard"] < 0.80,
        "shuffle_moved_fraction_ge_0_75": assigned > 0 and moved / assigned >= 0.75,
        "R_score_sum_ge_1_01_G": score_sums["G"] > 0 and score_sums["R"] >= 1.01 * score_sums["G"],
        "R_score_sum_ge_1_01_U": score_sums["U"] > 0 and score_sums["R"] >= 1.01 * score_sums["U"],
    }
    return {
        "schema": "rtgs.compact_responsibility_birth_selection.v1",
        "scale_boundary": boundary,
        "eligible_rows": eligible,
        "eligible_ids": [int(persistent_ids[row]) for row in eligible],
        "positive_support_view_count": [
            int(value) for value in positive_views.detach().cpu().tolist()
        ],
        "strata": stratum_records,
        "shuffle": shuffle_records,
        "selected_rows": selected,
        "selected_ids": {
            arm: [int(persistent_ids[row]) for row in rows] for arm, rows in selected.items()
        },
        "overlaps": overlaps,
        "shuffle_moved_fraction": moved / assigned,
        "residual_score_sums": score_sums,
        "gates_without_assigned_fraction": gates,
        "semantic_sha256": _selection_hash(selected),
    }


def _params_to_gaussians(params: Mapping[str, torch.Tensor]) -> Gaussians3D:
    return Gaussians3D(
        means=params["means"],
        quats=params["quats"],
        log_scales=params["scales"],
        opacity=torch.sigmoid(params["opacities"]),
        sh=torch.cat([params["sh0"], params["shN"]], dim=1),
    )


def _receipt_record(receipt: SelectedBirthReceipt) -> dict[str, Any]:
    return dataclasses.asdict(receipt)


class ResponsibilityBirthController:
    """Collect frozen G/R/S scores and optionally apply one matched birth wave."""

    def __init__(
        self,
        *,
        arm: Literal["G", "R", "U"] | None,
        split_root: int,
        shuffle_root: int,
        score_steps: int = SCORE_STEPS,
        quota_per_stratum: int = QUOTA_PER_STRATUM,
        expected_selection_sha256: str | None = None,
        on_pre_surgery: Callable[[Gaussians3D, dict[str, Any]], None] | None = None,
        on_post_surgery: Callable[[Gaussians3D, dict[str, Any]], None] | None = None,
    ) -> None:
        if arm not in (*ARMS, None):
            raise ValueError("arm must be G, R, U, or None")
        if score_steps <= 0 or quota_per_stratum <= 0:
            raise ValueError("score_steps and quota_per_stratum must be positive")
        self.arm = arm
        self.split_root = split_root
        self.shuffle_root = shuffle_root
        self.score_steps = score_steps
        self.quota_per_stratum = quota_per_stratum
        self.expected_selection_sha256 = expected_selection_sha256
        self.on_pre_surgery = on_pre_surgery
        self.on_post_surgery = on_post_surgery
        self._ids: torch.Tensor | None = None
        self._extent: float | None = None
        self._n_views: int | None = None
        self._attempts: int | None = None
        self._r_by_view: torch.Tensor | None = None
        self._s_by_view: torch.Tensor | None = None
        self._view_step_count: torch.Tensor | None = None
        self._g_sum: torch.Tensor | None = None
        self._g_count: torch.Tensor | None = None
        self._step_evidence: list[dict[str, Any]] = []
        self._assigned_numerator = 0.0
        self._assigned_denominator = 0.0
        self._pre_calls: dict[int, int] = {}
        self._post_calls: dict[int, int] = {}
        self._selection: dict[str, Any] | None = None
        self._surgery: dict[str, Any] | None = None
        self._lineage: list[dict[str, Any]] = []

    @property
    def persistent_ids(self) -> torch.Tensor:
        if self._ids is None:
            raise RuntimeError("controller is not bound")
        return self._ids

    def bind(
        self,
        params: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        extent: float,
        n_views: int,
        attempts_per_step: int,
    ) -> None:
        del optimizers
        if self._ids is not None:
            raise RuntimeError("controller cannot be rebound")
        n = int(params["means"].shape[0])
        device = params["means"].device
        self._ids = torch.arange(n, device=device, dtype=torch.long)
        self._extent = float(extent)
        self._n_views = int(n_views)
        self._attempts = int(attempts_per_step)
        self._r_by_view = torch.zeros(n_views, n, dtype=torch.float64, device=device)
        self._s_by_view = torch.zeros(n_views, n, dtype=torch.float64, device=device)
        self._view_step_count = torch.zeros(n_views, dtype=torch.int64, device=device)
        self._g_sum = torch.zeros(n, dtype=torch.float32, device=device)
        self._g_count = torch.zeros(n, dtype=torch.int64, device=device)

    def needs_compositing_color_basis(self, step: int) -> bool:
        return step <= self.score_steps

    def observe_pre_backward(
        self,
        *,
        step: int,
        view_index: int,
        output: PointRenderOutput,
        point_loss: torch.Tensor,
        active: torch.Tensor,
        attempts: int,
    ) -> None:
        if step > self.score_steps:
            if output.compositing_color_basis is not None:
                raise ProtocolInvalid("compositing basis remained enabled after score window")
            return
        if self._r_by_view is None or self._attempts is None:
            raise RuntimeError("controller is not bound")
        if attempts != self._attempts or point_loss.shape != active.shape:
            raise ProtocolInvalid("responsibility attempt shape/count changed")
        self._pre_calls[step] = self._pre_calls.get(step, 0) + 1
        if self._pre_calls[step] != 1:
            raise ProtocolInvalid("score step used more than one outer microbatch")

        native_error = point_loss.detach().to(dtype=torch.float32)
        native_active = active.detach().to(dtype=torch.float32)
        basis = output.compositing_color_basis
        visible = output.visible
        global_r = torch.zeros(
            self.persistent_ids.numel(),
            dtype=torch.float64,
            device=self.persistent_ids.device,
        )
        global_s = torch.zeros_like(global_r)
        if basis is not None:
            if (
                visible is None
                or basis.ndim != 2
                or basis.shape != (visible.numel(), 3)
                or not basis.requires_grad
            ):
                raise ProtocolInvalid("invalid compositor basis/visibility mapping")
            grad_outputs = torch.zeros_like(output.color)
            grad_outputs[:, 0] = native_active * native_error
            grad_outputs[:, 1] = native_active
            contracted = torch.autograd.grad(
                output.color,
                basis,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].detach()
            if contracted.requires_grad or not bool(torch.isfinite(contracted).all()):
                raise ProtocolInvalid("VJP result is not finite and detached")
            native_r = contracted[:, 0].to(dtype=torch.float32)
            native_s = contracted[:, 1].to(dtype=torch.float32)
            global_r.index_add_(0, visible, native_r.to(dtype=torch.float64) / attempts)
            global_s.index_add_(0, visible, native_s.to(dtype=torch.float64) / attempts)
        elif visible is not None and visible.numel() != 0:
            raise ProtocolInvalid("requested compositor basis is missing for visible rows")
        output.compositing_color_basis = None

        self._r_by_view[view_index].add_(global_r)
        self._s_by_view[view_index].add_(global_s)
        if self._view_step_count is None:
            raise RuntimeError("controller is not bound")
        self._view_step_count[view_index] += 1
        assigned_native = float(global_r.sum() * attempts)
        alpha_native = float(
            (native_active * native_error * output.alpha.detach().to(torch.float32)).sum()
        )
        support_native = float(global_s.sum() * attempts)
        alpha_support_native = float(
            (native_active * output.alpha.detach().to(torch.float32)).sum()
        )
        if not math.isclose(assigned_native, alpha_native, abs_tol=2e-6, rel_tol=2e-5):
            raise ProtocolInvalid("residual VJP/alpha contraction identity failed")
        if not math.isclose(support_native, alpha_support_native, abs_tol=2e-6, rel_tol=2e-5):
            raise ProtocolInvalid("support VJP/alpha contraction identity failed")
        error_denominator = float((native_active * native_error).sum())
        self._assigned_numerator += assigned_native
        self._assigned_denominator += error_denominator
        self._step_evidence.append(
            {
                "step": step,
                "view_index": view_index,
                "visible_rows": ([] if visible is None else visible.detach().cpu().tolist()),
                "residual_global_float64": _finite_list(global_r),
                "support_global_float64": _finite_list(global_s),
                "native_residual_sum": assigned_native,
                "alpha_residual_sum": alpha_native,
                "native_support_sum": support_native,
                "alpha_support_sum": alpha_support_native,
                "active_error_sum": error_denominator,
                "residual_sha256": tensor_hash(global_r),
                "support_sha256": tensor_hash(global_s),
            }
        )

    def observe_post_backward(
        self,
        *,
        step: int,
        view_index: int,
        output: PointRenderOutput,
        width: int,
        height: int,
    ) -> None:
        del view_index
        if step > self.score_steps:
            return
        if self._g_sum is None or self._g_count is None:
            raise RuntimeError("controller is not bound")
        self._post_calls[step] = self._post_calls.get(step, 0) + 1
        if self._post_calls[step] != 1:
            raise ProtocolInvalid("gradient score step used more than one microbatch")
        visible = output.visible
        means2d = output.means2d
        if visible is None or visible.numel() == 0:
            return
        if means2d is None or means2d.grad is None:
            raise ProtocolInvalid("visible render lacks retained screen gradients")
        native = torch.linalg.vector_norm(means2d.grad.detach(), dim=-1)
        native = native * (max(width, height) * 0.5)
        native = native.to(dtype=torch.float32)
        if not bool(torch.isfinite(native).all()):
            raise ProtocolInvalid("ordinary screen-gradient score is non-finite")
        self._g_sum.index_add_(0, visible, native)
        self._g_count.index_add_(0, visible, torch.ones_like(visible, dtype=torch.int64))
        self._step_evidence[-1]["gradient_visible_float32"] = _finite_list(native)
        self._step_evidence[-1]["gradient_visible_sha256"] = tensor_hash(native)

    def _finalize_selection(self, params: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        if any(
            value is None
            for value in (
                self._r_by_view,
                self._s_by_view,
                self._view_step_count,
                self._g_sum,
                self._g_count,
                self._extent,
            )
        ):
            raise RuntimeError("controller is not bound")
        assert self._r_by_view is not None
        assert self._s_by_view is not None
        assert self._view_step_count is not None
        assert self._g_sum is not None
        assert self._g_count is not None
        assert self._extent is not None
        if bool((self._view_step_count <= 0).any()):
            raise ProtocolInvalid("score window omitted a view")
        r_view = self._r_by_view / self._view_step_count[:, None]
        s_view = self._s_by_view / self._view_step_count[:, None]
        r = r_view.mean(dim=0)
        s = s_view.mean(dim=0)
        g = self._g_sum / self._g_count.clamp_min(1).to(torch.float32)
        scale_max = params["scales"].detach().exp().amax(dim=-1)
        record = build_matched_selections(
            gradient_score=g,
            residual_score=r,
            support_score=s,
            support_by_view=s_view,
            visible_step_count=self._g_count,
            scale_max=scale_max,
            persistent_ids=self.persistent_ids,
            extent=self._extent,
            shuffle_root=self.shuffle_root,
            quota_per_stratum=self.quota_per_stratum,
        )
        assigned_fraction = (
            self._assigned_numerator / self._assigned_denominator
            if self._assigned_denominator > 0
            else math.nan
        )
        record["scores"] = {
            "G_float32": _finite_list(g),
            "R_float64": _finite_list(r),
            "S_float64": _finite_list(s),
            "R_by_view_float64": [_finite_list(row) for row in r_view],
            "S_by_view_float64": [_finite_list(row) for row in s_view],
            "visible_step_count": self._g_count.detach().cpu().tolist(),
            "view_step_count": self._view_step_count.detach().cpu().tolist(),
            "scale_max": _finite_list(scale_max),
        }
        record["assigned_residual"] = {
            "numerator": self._assigned_numerator,
            "denominator": self._assigned_denominator,
            "fraction": assigned_fraction,
        }
        record["gates_without_assigned_fraction"]["assigned_fraction_ge_0_10"] = (
            math.isfinite(assigned_fraction) and assigned_fraction >= 0.10
        )
        record["all_phase_a_gates_pass"] = all(record["gates_without_assigned_fraction"].values())
        record["step_evidence_sha256"] = canonical_hash(self._step_evidence)
        record["semantic_sha256"] = canonical_hash(
            {
                "selected_rows": record["selected_rows"],
                "scores": record["scores"],
                "assigned_residual": record["assigned_residual"],
                "gates": record["gates_without_assigned_fraction"],
            }
        )
        return record

    def after_step(
        self,
        *,
        step: int,
        params: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        snapshot: Gaussians3D,
    ) -> dict[str, torch.Tensor]:
        if step != self.score_steps:
            return params
        self._selection = self._finalize_selection(params)
        if (
            self.expected_selection_sha256 is not None
            and self._selection["semantic_sha256"] != self.expected_selection_sha256
        ):
            raise ProtocolInvalid("Phase-B score/selection replay differs from Phase A")
        if self.on_pre_surgery is not None:
            self.on_pre_surgery(snapshot.detach(), self._selection)
        if self.arm is None:
            return params
        selected_rows = self._selection["selected_rows"][self.arm]
        generator = torch.Generator(device=params["means"].device).manual_seed(
            split_seed(self.split_root)
        )
        new_params, receipt = apply_selected_birth_surgery(
            params,
            optimizers,
            selected_rows,
            scene_extent=float(self._extent),
            generator=generator,
            split_scale_frac=0.01,
            split_factor=1.6,
            revised_opacity=True,
            max_gaussians=867,
        )
        old_ids = self.persistent_ids
        survivor_ids = old_ids[
            torch.tensor(
                receipt.survivor_old_rows,
                dtype=torch.long,
                device=old_ids.device,
            )
        ]
        next_id = int(old_ids.max()) + 1
        newborn_ids = torch.arange(
            next_id,
            next_id + len(receipt.newborns),
            dtype=torch.long,
            device=old_ids.device,
        )
        self._ids = torch.cat([survivor_ids, newborn_ids])
        self._lineage = [
            {
                "birth_id": int(birth_id),
                "parent_id": int(old_ids[item.parent_row]),
                "operator": item.operator,
                "child_ordinal": item.child_ordinal,
                "physical_row": item.new_row,
            }
            for birth_id, item in zip(
                newborn_ids.detach().cpu().tolist(), receipt.newborns, strict=True
            )
        ]
        self._surgery = {
            "arm": self.arm,
            "split_seed": split_seed(self.split_root),
            "receipt": _receipt_record(receipt),
            "persistent_ids_after": self._ids.detach().cpu().tolist(),
            "lineage": self._lineage,
        }
        post = _params_to_gaussians(new_params).detach()
        if self.on_post_surgery is not None:
            self.on_post_surgery(post, self._surgery)
        return new_params

    def history_record(self) -> dict[str, Any]:
        return {
            "schema": "rtgs.compact_responsibility_birth_controller.v1",
            "arm": self.arm,
            "score_steps": self.score_steps,
            "split_root": self.split_root,
            "shuffle_root": self.shuffle_root,
            "split_seed": split_seed(self.split_root),
            "selection": self._selection,
            "surgery": self._surgery,
            "persistent_ids": (None if self._ids is None else self._ids.detach().cpu().tolist()),
            "lineage": self._lineage,
            "step_evidence": self._step_evidence,
        }


def _sample_arrays(samples: ObservationSamples, color: torch.Tensor) -> dict[str, np.ndarray]:
    values = {
        "xy": samples.xy,
        "active": samples.active,
        "inside_fit_window": samples.inside_fit_window,
        "proposal_component_ids": samples.proposal_component_ids,
        "proposal_density": samples.proposal_density,
        "joint_density": samples.joint_density,
        "color": color,
    }
    return {name: value.detach().cpu().contiguous().numpy() for name, value in values.items()}


def _validate_bank(
    *,
    samples: ObservationSamples,
    color: torch.Tensor,
    product: GaussianObservationField,
    measure: str,
    attempts: int,
) -> None:
    expected = {
        "xy": (attempts, 2),
        "active": (attempts,),
        "inside_fit_window": (attempts,),
        "proposal_component_ids": (attempts,),
        "proposal_density": (attempts,),
        "joint_density": (attempts,),
        "target_density": (attempts,),
        "importance": (attempts,),
        "color": (attempts, 3),
    }
    values: dict[str, torch.Tensor] = {
        "xy": samples.xy,
        "active": samples.active,
        "inside_fit_window": samples.inside_fit_window,
        "proposal_component_ids": samples.proposal_component_ids,
        "proposal_density": samples.proposal_density,
        "joint_density": samples.joint_density,
        "target_density": samples.target_density,
        "importance": samples.importance,
        "color": color,
    }
    for name, shape in expected.items():
        if tuple(values[name].shape) != shape:
            raise ProtocolInvalid(f"evaluation bank {name} shape changed")
    for name in (
        "xy",
        "proposal_density",
        "joint_density",
        "target_density",
        "importance",
        "color",
    ):
        if not bool(torch.isfinite(values[name]).all()):
            raise ProtocolInvalid(f"evaluation bank {name} is non-finite")
    if measure == "uniform":
        x, y, width, height = product.fit_window
        direct = (
            samples.active.all()
            and samples.inside_fit_window.all()
            and (samples.proposal_component_ids == -1).all()
            and (samples.xy[:, 0] >= x).all()
            and (samples.xy[:, 0] < x + width).all()
            and (samples.xy[:, 1] >= y).all()
            and (samples.xy[:, 1] < y + height).all()
        )
        if not bool(direct):
            raise ProtocolInvalid("uniform evaluation bank is not direct/half-open")
    elif measure == "proposal":
        inactive = ~samples.active
        if bool((samples.active & ~samples.inside_fit_window).any()):
            raise ProtocolInvalid("active proposal-bank row lies outside fit window")
        for name in ("joint_density", "target_density", "importance"):
            if bool((values[name][inactive] != 0).any()):
                raise ProtocolInvalid(f"inactive proposal-bank {name} is nonzero")
    else:
        raise ValueError("unknown evaluation measure")


def _exclusive_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(path)


def generate_evaluation_bank(
    *,
    evaluation_root: int,
    teachers: ReconstructionInputs,
    product_fields: Sequence[GaussianObservationField],
    path: Path,
    focused: bool = False,
    attempts: int = BANK_ATTEMPTS,
) -> dict[str, Any]:
    """Materialize one complete fixed bank under the amended seed domain."""
    allowed = FOCUSED_EVALUATION_ROOTS if focused else EVALUATION_ROOTS
    if evaluation_root not in allowed:
        raise ValueError("evaluation root is not in the selected frozen domain")
    _reject_roots_for_mode(evaluation_root, focused=focused)
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if len(teachers.observations) != len(product_fields):
        raise ValueError("teacher/product view counts differ")

    arrays: dict[str, np.ndarray] = {}
    views: list[dict[str, Any]] = []
    for view_index, (view_name, teacher, product) in enumerate(
        zip(
            teachers.view_names,
            teachers.observations,
            product_fields,
            strict=True,
        )
    ):
        proposal = GaussianPointProposal(product, product)
        bank_records: dict[str, Any] = {}
        for measure, uniform_fraction in (("uniform", 1.0), ("proposal", 0.25)):
            derived = evaluation_bank_seed(evaluation_root, view_name, measure)
            generator = torch.Generator(device=product.device).manual_seed(derived)
            samples = proposal.sample(
                attempts,
                uniform_fraction=uniform_fraction,
                generator=generator,
            )
            color = teacher.query(samples.xy, component_chunk=640).color
            _validate_bank(
                samples=samples,
                color=color,
                product=product,
                measure=measure,
                attempts=attempts,
            )
            values = _sample_arrays(samples, color)
            descriptors = {}
            for name, value in values.items():
                key = f"v{view_index}_{measure}_{name}"
                arrays[key] = value
                descriptors[name] = {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": array_hash(value),
                }
            active_count = int(values["active"].sum())
            bank_records[measure] = {
                "seed_domain": EVALUATION_SEED_DOMAIN,
                "root": evaluation_root,
                "generator_seed": derived,
                "view_name": view_name,
                "measure": measure,
                "attempts": attempts,
                "active_count": active_count,
                "null_count": attempts - active_count,
                "active_fraction": active_count / attempts,
                "tensors": descriptors,
            }
        views.append(
            {
                "view_index": view_index,
                "view_name": view_name,
                "m_opt_2d": teacher.n,
                "banks": bank_records,
            }
        )
    metadata: dict[str, Any] = {
        "schema": "rtgs.compact_responsibility_birth_banks.v1",
        "seed_domain": EVALUATION_SEED_DOMAIN,
        "evaluation_root": evaluation_root,
        "attempts_per_bank": attempts,
        "views": views,
    }
    metadata["semantic_sha256"] = canonical_hash(metadata)
    arrays["metadata_utf8"] = np.frombuffer(canonical_bytes(metadata), dtype=np.uint8)
    file_sha = _exclusive_npz(path, arrays)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": file_sha,
        "bytes": path.stat().st_size,
        "metadata": metadata,
    }


def load_evaluation_bank(
    path: Path,
    *,
    expected_root: int,
    attempts: int = BANK_ATTEMPTS,
    expected_views: Sequence[str] = EXPECTED_VIEWS,
) -> tuple[list[dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_utf8" not in archive.files:
            raise ProtocolInvalid("evaluation archive lacks metadata")
        metadata = json.loads(np.asarray(archive["metadata_utf8"], dtype=np.uint8).tobytes())
        digest = dict(metadata)
        stored = digest.pop("semantic_sha256", None)
        if stored != canonical_hash(digest):
            raise ProtocolInvalid("evaluation-bank metadata digest changed")
        if (
            metadata.get("schema") != "rtgs.compact_responsibility_birth_banks.v1"
            or metadata.get("seed_domain") != EVALUATION_SEED_DOMAIN
            or metadata.get("evaluation_root") != expected_root
            or metadata.get("attempts_per_bank") != attempts
        ):
            raise ProtocolInvalid("evaluation-bank metadata differs from contract")
        expected_keys = {"metadata_utf8"}
        loaded = []
        view_records = metadata.get("views")
        if not isinstance(view_records, list) or len(view_records) != len(expected_views):
            raise ProtocolInvalid("evaluation-bank view count changed")
        for view_index, (view_name, record) in enumerate(
            zip(expected_views, view_records, strict=True)
        ):
            if record.get("view_index") != view_index or record.get("view_name") != view_name:
                raise ProtocolInvalid("evaluation-bank view ordering changed")
            measures = {}
            for measure in ("uniform", "proposal"):
                measure_record = record["banks"][measure]
                expected_seed = evaluation_bank_seed(expected_root, view_name, measure)
                if (
                    measure_record.get("seed_domain") != EVALUATION_SEED_DOMAIN
                    or measure_record.get("root") != expected_root
                    or measure_record.get("generator_seed") != expected_seed
                    or measure_record.get("measure") != measure
                ):
                    raise ProtocolInvalid("evaluation-bank seed binding changed")
                values = {}
                for name, descriptor in measure_record["tensors"].items():
                    key = f"v{view_index}_{measure}_{name}"
                    expected_keys.add(key)
                    value = np.asarray(archive[key]).copy()
                    actual = {
                        "dtype": value.dtype.str,
                        "shape": list(value.shape),
                        "sha256": array_hash(value),
                    }
                    if actual != descriptor:
                        raise ProtocolInvalid(f"bank tensor changed: {key}")
                    values[name] = value
                if int(values["active"].sum()) != measure_record["active_count"]:
                    raise ProtocolInvalid("evaluation-bank active count changed")
                measures[measure] = values
            loaded.append(measures)
        if set(archive.files) != expected_keys:
            raise ProtocolInvalid("evaluation archive has unexpected arrays")
    return loaded, metadata


def evaluate_snapshot(
    snapshot: Gaussians3D,
    inputs: ReconstructionInputs,
    banks: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
    *,
    device: str = "cuda:0",
    point_chunk: int = 256,
    gaussian_chunk: int = 256,
) -> dict[str, Any]:
    """Evaluate frozen J_U/J_Q banks in detached float64 arithmetic."""
    target_device = torch.device(device)
    model = snapshot.to(target_device)
    renderer = TorchPointRasterizer(point_chunk=point_chunk, gaussian_chunk=gaussian_chunk)
    background = torch.zeros(3, device=target_device, dtype=model.means.dtype)
    per_view = []
    with torch.no_grad():
        for view_index, (name, camera, view_banks) in enumerate(
            zip(inputs.view_names, inputs.cameras, banks, strict=True)
        ):
            values: dict[str, float] = {}
            diagnostics: dict[str, Any] = {}
            for measure, risk_name in (("uniform", "J_U"), ("proposal", "J_Q")):
                bank = view_banks[measure]
                xy = torch.from_numpy(bank["xy"]).to(target_device)
                target = torch.from_numpy(bank["color"]).to(target_device)
                active = torch.from_numpy(bank["active"]).to(device=target_device, dtype=torch.bool)
                prediction = renderer.render_points(
                    model,
                    camera.to(target_device),
                    xy,
                    background=background,
                    sh_degree=0,
                ).color
                if not bool(torch.isfinite(prediction).all()):
                    raise ProtocolInvalid("evaluation point render is non-finite")
                point_loss = (
                    (prediction.detach().to(torch.float64) - target.detach().to(torch.float64))
                    .square()
                    .mean(dim=-1)
                )
                weighted = (
                    point_loss if measure == "uniform" else point_loss * active.to(torch.float64)
                )
                attempts = int(xy.shape[0])
                values[risk_name] = float(weighted.sum(dtype=torch.float64) / attempts)
                diagnostics[measure] = {
                    "attempts": attempts,
                    "active_count": int(active.sum()),
                    "loss_sum": float(weighted.sum(dtype=torch.float64)),
                }
            per_view.append(
                {
                    "view_index": view_index,
                    "view_name": name,
                    **values,
                    "banks": diagnostics,
                }
            )
    return {
        "J_U": sum(item["J_U"] for item in per_view) / len(per_view),
        "J_Q": sum(item["J_Q"] for item in per_view) / len(per_view),
        "worst_view_J_U": max(item["J_U"] for item in per_view),
        "worst_view_J_Q": max(item["J_Q"] for item in per_view),
        "per_view": per_view,
    }


def recovery_log_auc(risks: Sequence[float]) -> float:
    if len(risks) != 4:
        raise ValueError("recovery curve needs 35_post,70,105,140")
    if any(not math.isfinite(value) or value <= 0 for value in risks):
        raise ValueError("recovery risks must be finite and positive")
    x = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    logs = [math.log(max(value, 1e-12)) for value in risks]
    return sum(
        (x[index + 1] - x[index]) * (logs[index + 1] + logs[index]) * 0.5 for index in range(3)
    )


def geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def compute_terminal_decision(
    records: Mapping[int, Mapping[str, Mapping[str, Any]]],
    active_fractions: Sequence[float],
    *,
    roots: Sequence[int] = TRAIN_ROOTS,
    structural_passed: bool = True,
) -> dict[str, Any]:
    """Apply the amended exhaustive ordered decision map."""
    comparisons: dict[str, Any] = {}
    for comparator in ("G", "U"):
        final_q_ratios = []
        auc_q_ratios = []
        final_u_ratios = []
        wins = 0
        for root in roots:
            r_metrics = records[root]["R"]["checkpoint_metrics"]
            c_metrics = records[root][comparator]["checkpoint_metrics"]
            r_q = [float(r_metrics[key]["J_Q"]) for key in RECOVERY_CHECKPOINTS]
            c_q = [float(c_metrics[key]["J_Q"]) for key in RECOVERY_CHECKPOINTS]
            final_q_ratios.append(r_q[-1] / c_q[-1])
            auc_q_ratios.append(math.exp(recovery_log_auc(r_q) - recovery_log_auc(c_q)))
            r_u = float(r_metrics["140"]["J_U"])
            c_u = float(c_metrics["140"]["J_U"])
            final_u_ratios.append(r_u / c_u)
            wins += int(r_q[-1] < c_q[-1])
        summary = {
            "final_J_Q_ratios": final_q_ratios,
            "recovery_auc_J_Q_ratios": auc_q_ratios,
            "final_J_U_ratios": final_u_ratios,
            "geometric_final_J_Q_ratio": geometric_mean(final_q_ratios),
            "geometric_recovery_auc_ratio": geometric_mean(auc_q_ratios),
            "strict_final_J_Q_wins": wins,
            "geometric_final_J_U_ratio": geometric_mean(final_u_ratios),
            "max_root_final_J_U_ratio": max(final_u_ratios),
        }
        summary["primary"] = (
            summary["geometric_final_J_Q_ratio"] <= 0.97
            and summary["geometric_recovery_auc_ratio"] <= 0.98
            and summary["strict_final_J_Q_wins"] >= 2
        )
        summary["safety"] = (
            summary["geometric_final_J_U_ratio"] <= 1.05
            and summary["max_root_final_J_U_ratio"] <= 1.10
        )
        comparisons[comparator] = summary

    active_guard = (
        len(active_fractions) == 21
        and min(active_fractions) >= 0.95
        and max(active_fractions) / min(active_fractions) <= 1.03
    )
    mandatory = structural_passed and active_guard
    primary_g = comparisons["G"]["primary"]
    primary_u = comparisons["U"]["primary"]
    safety_g = comparisons["G"]["safety"]
    safety_u = comparisons["U"]["safety"]
    if not mandatory:
        decision = "UNAVAILABLE"
    elif primary_g and primary_u and safety_g and safety_u:
        decision = "RESIDUAL_RESPONSIBILITY_ALLOCATION_PROMISING"
    elif primary_g and primary_u and not (safety_g and safety_u):
        decision = "UNIFORM_RISK_TRADEOFF"
    elif primary_g and not primary_u:
        decision = "MAPPING_NOT_ISOLATED"
    elif primary_u and not primary_g:
        decision = "NOT_BETTER_THAN_GRADIENT"
    else:
        decision = "NO_PARENT_ALLOCATION_WIN"
    return {
        "comparisons": comparisons,
        "population_guards": {
            "count_21": len(active_fractions) == 21,
            "minimum_ge_0_95": bool(active_fractions) and min(active_fractions) >= 0.95,
            "max_over_min_le_1_03": bool(active_fractions)
            and min(active_fractions) > 0
            and max(active_fractions) / min(active_fractions) <= 1.03,
            "passed": active_guard,
        },
        "structural_passed": structural_passed,
        "scientific_decision": decision,
    }


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    data = canonical_bytes(payload) + b"\n"
    _exclusive_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ProtocolInvalid(f"{path} is not a JSON object")
    if canonical_bytes(parsed) + b"\n" != raw:
        raise ProtocolInvalid(f"{path} is not canonical JSON")
    return parsed


def _source_hashes() -> tuple[dict[str, str], str]:
    records = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise ProtocolInvalid(f"sealed source is missing: {relative}")
        records[relative.as_posix()] = sha256_file(path)
    return records, canonical_hash(records)


def _review_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return "Verdict: PASS" in text and "Unresolved findings: none" in text


def _config_record() -> dict[str, Any]:
    # Literal values only: seal creation must not pass an official root to a
    # scheduler, generator, sampler, or trainer.
    return {
        "schema": "rtgs.compact_responsibility_birth_config.v1",
        "training_roots": list(TRAIN_ROOTS),
        "evaluation_roots": list(EVALUATION_ROOTS),
        "split_roots": list(SPLIT_ROOTS),
        "shuffle_roots": list(SHUFFLE_ROOTS),
        "focused_training_roots": list(FOCUSED_TRAIN_ROOTS),
        "focused_evaluation_roots": list(FOCUSED_EVALUATION_ROOTS),
        "root_sets_pairwise_disjoint": len(
            set(
                TRAIN_ROOTS
                + EVALUATION_ROOTS
                + SPLIT_ROOTS
                + SHUFFLE_ROOTS
                + FOCUSED_TRAIN_ROOTS
                + FOCUSED_EVALUATION_ROOTS
            )
        )
        == 16,
        "iterations": TRAIN_ITERATIONS,
        "score_steps": SCORE_STEPS,
        "attempts_per_step": TRAIN_ATTEMPTS,
        "bank_attempts": BANK_ATTEMPTS,
        "checkpoints": list(CHECKPOINTS),
        "recovery_checkpoints": list(RECOVERY_CHECKPOINTS),
        "quota_per_stratum": QUOTA_PER_STRATUM,
        "strata": list(STRATA),
        "arm_order": {str(root): list(order) for root, order in ARM_ORDER.items()},
        "domain_prefix_hex": DOMAIN_PREFIX.hex(),
        "evaluation_seed_domain": EVALUATION_SEED_DOMAIN,
        "extent": EXPLICIT_EXTENT,
        "device": "cuda:0",
        "optimizer": {
            "order": ["means", "quats", "scales", "opacities", "sh0", "shN"],
            "algorithm": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-15,
            "foreach": False,
            "fused": False,
        },
    }


def _binding_state() -> dict[str, Any]:
    sources, aggregate = _source_hashes()
    return {
        "source_hashes": sources,
        "source_aggregate_sha256": aggregate,
        "inputs": factorial.input_bindings(),
        "runtime": factorial.runtime_binding(),
        "config": _config_record(),
        "prerequisite_artifacts": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_PREREG.md",
                RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json",
                RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_AUDIT.md",
                RESULTS / "20260716_residual_responsibility_density_PREREG.md",
            )
        },
    }


def _run_command(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=None if env is None else dict(env),
        text=True,
        capture_output=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _create_source_tar(path: Path) -> str:
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with tarfile.open(temporary, "w") as archive:
            for relative in SOURCE_PATHS:
                source = ROOT / relative
                info = archive.gettarinfo(source, arcname=relative.as_posix())
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _official_namespace_absent_for_seal() -> bool:
    return not any(
        path.exists()
        for path in (
            SEAL,
            PHASE_A_ATTEMPT,
            PHASE_A_RESULT,
            PHASE_A_AUDIT,
            PHASE_B_ATTEMPT,
            RESULT,
            EXECUTED_SOURCES,
            RUN_DIR,
        )
    )


def create_seal() -> dict[str, Any]:
    if not _official_namespace_absent_for_seal():
        raise ProtocolInvalid("official namespace is not pristine")
    if not _review_passed(PREREGISTRATION_ADDENDUM_REVIEW):
        raise ProtocolInvalid("amended preregistration lacks an independent PASS")
    if not _review_passed(IMPLEMENTATION_REVIEW):
        raise ProtocolInvalid("implementation review lacks an independent PASS")
    before = _binding_state()
    env = dict(os.environ)
    env[FOCUSED_TEST_ENV] = "1"
    verification = [
        _run_command(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "pytest",
                "-q",
                "tests/test_compact_responsibility_birth_allocation.py",
                "tests/test_point_render.py",
                "tests/test_compact_trainer.py",
            ],
            env=env,
        ),
        _run_command([str(ROOT / "scripts/verify.sh")], env=env),
        _run_command(["git", "diff", "--check"], env=env),
    ]
    if any(record["returncode"] != 0 for record in verification):
        raise ProtocolInvalid("seal verification failed")
    after = _binding_state()
    if before != after:
        raise ProtocolInvalid("source/input/runtime drifted during seal verification")
    executed_sources_sha = _create_source_tar(EXECUTED_SOURCES)
    payload = {
        "artifact_type": "compact_responsibility_birth_seal_v1",
        "timestamp_utc": factorial.timestamp_utc(),
        "status": "PASS",
        "bindings": after,
        "bindings_sha256": canonical_hash(after),
        "reviews": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                PREREGISTRATION_REVIEW,
                PREREGISTRATION_INITIAL_FAIL,
                PREREGISTRATION_ADDENDUM_REVIEW,
                IMPLEMENTATION_REVIEW,
            )
        },
        "verification": verification,
        "executed_sources": {
            "path": str(EXECUTED_SOURCES.relative_to(ROOT)),
            "sha256": executed_sources_sha,
        },
    }
    exclusive_json(SEAL, payload)
    return strict_json(SEAL)


def verify_seal() -> dict[str, Any]:
    sealed = strict_json(SEAL)
    if (
        sealed.get("artifact_type") != "compact_responsibility_birth_seal_v1"
        or sealed.get("status") != "PASS"
        or sealed.get("bindings_sha256") != canonical_hash(sealed.get("bindings"))
    ):
        raise ProtocolInvalid("seal schema/status/digest is invalid")
    if sealed["bindings"] != _binding_state():
        raise ProtocolInvalid("current source/input/runtime differs from seal")
    executed = sealed["executed_sources"]
    path = ROOT / executed["path"]
    if sha256_file(path) != executed["sha256"]:
        raise ProtocolInvalid("executed-source archive changed")
    return sealed


def _save_npz_snapshot(path: Path, snapshot: Gaussians3D) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_npz(path)
    replay = Gaussians3D.load_npz(path)
    semantic = factorial.gaussians_hash(snapshot)
    if factorial.gaussians_hash(replay) != semantic:
        raise ProtocolInvalid("snapshot NPZ round trip changed")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "semantic_sha256": semantic,
        "n_gaussians": snapshot.n,
    }


def _save_final_ply(path: Path, snapshot: Gaussians3D) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    snapshot.save_ply(path)
    replay = Gaussians3D.load_ply(path)
    if replay.n != 867:
        raise ProtocolInvalid("final PLY count differs from 867")
    fields = {}
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        loaded = getattr(replay, name).detach().to(dtype=torch.float32)
        source = getattr(snapshot, name).detach().cpu().to(dtype=torch.float32)
        if loaded.shape != source.shape or not bool(torch.isfinite(loaded).all()):
            raise ProtocolInvalid(f"final PLY field {name} is invalid")
        absolute = (loaded - source).abs()
        tolerance = 1e-6 + 1e-6 * source.abs()
        excess = (absolute - tolerance).clamp_min(0)
        if bool((absolute > tolerance).any()):
            raise ProtocolInvalid(f"final PLY field {name} exceeds tolerance")
        fields[name] = {
            "max_abs_error": float(absolute.max()) if absolute.numel() else 0.0,
            "max_normalized_excess": (float((excess / tolerance).max()) if excess.numel() else 0.0),
        }
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "n_gaussians": replay.n,
        "source_semantic_sha256": factorial.gaussians_hash(snapshot),
        "roundtrip": fields,
    }


def _load_frozen_inputs() -> tuple[
    ReconstructionInputs,
    list[GaussianObservationField],
    Gaussians3D,
    list[dict[str, Any]],
]:
    inputs = ReconstructionInputs.load(TEACHER_BUNDLE, strict=True)
    proxies = ReconstructionInputs.load(PROXY_BUNDLE, strict=True)
    product, alignment = factorial.build_product_fields(inputs, proxies)
    init = Gaussians3D.load_ply(INIT_PLY)
    if (
        tuple(inputs.view_names) != EXPECTED_VIEWS
        or init.n != 835
        or sha256_file(INIT_PLY) != EXPECTED_INIT_PLY_SHA256
    ):
        raise ProtocolInvalid("frozen inputs changed")
    return inputs, product, init, alignment


def _phase_a_worker(
    *,
    training_root: int,
    split_root_value: int,
    shuffle_root_value: int,
    marker_sha256: str,
    seal_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    if sha256_file(PHASE_A_ATTEMPT) != marker_sha256 or sha256_file(SEAL) != seal_sha256:
        raise ProtocolInvalid("Phase-A worker marker/seal mismatch")
    verify_seal()
    replicate = TRAIN_ROOTS.index(training_root)
    if SPLIT_ROOTS[replicate] != split_root_value or SHUFFLE_ROOTS[replicate] != shuffle_root_value:
        raise ProtocolInvalid("Phase-A root pairing changed")
    worker_dir = RUN_DIR / f"seed_{training_root}" / "phase_a"
    worker_dir.mkdir(parents=True, exist_ok=False)
    guard = RGBAccessGuard()
    with guard:
        inputs, product, init, alignment = _load_frozen_inputs()
        pre_snapshot: list[Gaussians3D] = []
        controller = ResponsibilityBirthController(
            arm=None,
            split_root=split_root_value,
            shuffle_root=shuffle_root_value,
            on_pre_surgery=lambda snapshot, _: pre_snapshot.append(snapshot.to("cpu")),
        )
        final, history = CompactTrainer(frozen_config(training_root)).train(
            inputs,
            init,
            proposal_fields=product,
            bundle_path=TEACHER_BUNDLE,
            topology_controller=controller,
            stop_after_step=SCORE_STEPS,
        )
        if (
            final.n != 835
            or len(pre_snapshot) != 1
            or factorial.gaussians_hash(final.to("cpu"))
            != factorial.gaussians_hash(pre_snapshot[0])
        ):
            raise ProtocolInvalid("Phase-A prefix/snapshot count changed")
        controller_record = controller.history_record()
        snapshot_record = _save_npz_snapshot(worker_dir / "gaussians_35_pre.npz", pre_snapshot[0])
        history_path = worker_dir / "history.json"
        history_sha = exclusive_json(history_path, history)
    denial = guard.record()
    if not denial["passed"]:
        raise ProtocolInvalid("Phase-A worker crossed RGB denial boundary")
    payload = {
        "artifact_type": "compact_responsibility_birth_phase_a_worker_v1",
        "status": "PASS",
        "training_root": training_root,
        "split_root": split_root_value,
        "shuffle_root": shuffle_root_value,
        "n_init_3d": 835,
        "n_opt_3d": 835,
        "n_init_2d": inputs.n_init_2d,
        "n_opt_2d": inputs.n_opt_2d,
        "sum_m_opt_2d": sum(inputs.n_opt_2d),
        "alignment": alignment,
        "rgb_denial": denial,
        "snapshot_35_pre": snapshot_record,
        "selection": controller_record["selection"],
        "history": {
            "path": str(history_path.relative_to(ROOT)),
            "sha256": history_sha,
            "view_schedule_sha256": history["view_schedule_sha256"],
            "steps": history["steps"],
            "controller": controller_record,
        },
    }
    exclusive_json(output_path, payload)
    return strict_json(output_path)


def _worker_command(*arguments: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        *arguments,
    ]


def _worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(PRELOAD)
    env.pop(FOCUSED_TEST_ENV, None)
    return env


def _run_worker(command: Sequence[str]) -> dict[str, Any]:
    record = _run_command(command, env=_worker_environment())
    if record["returncode"] != 0:
        raise ProtocolInvalid(
            "official worker failed:\n" + record["stderr"][-4000:] + "\n" + record["stdout"][-4000:]
        )
    return record


def run_phase_a() -> dict[str, Any]:
    sealed = verify_seal()
    if any(
        path.exists()
        for path in (
            PHASE_A_ATTEMPT,
            PHASE_A_RESULT,
            PHASE_A_AUDIT,
            PHASE_B_ATTEMPT,
            RESULT,
            RUN_DIR,
        )
    ):
        raise ProtocolInvalid("Phase-A namespace is not pristine")
    marker = {
        "artifact_type": "compact_responsibility_birth_phase_a_attempt_v1",
        "timestamp_utc": factorial.timestamp_utc(),
        "seal_sha256": sha256_file(SEAL),
        "bindings_sha256": sealed["bindings_sha256"],
    }
    marker_sha = exclusive_json(PHASE_A_ATTEMPT, marker)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    workers = []
    commands = []
    try:
        for training_root, split_root_value, shuffle_root_value in zip(
            TRAIN_ROOTS, SPLIT_ROOTS, SHUFFLE_ROOTS, strict=True
        ):
            output = RUN_DIR / f"seed_{training_root}" / "phase_a_worker_result.json"
            command = _worker_command(
                "_phase-a-worker",
                "--training-root",
                str(training_root),
                "--split-root",
                str(split_root_value),
                "--shuffle-root",
                str(shuffle_root_value),
                "--attempt-sha256",
                marker_sha,
                "--seal-sha256",
                sha256_file(SEAL),
                "--worker-output",
                str(output),
            )
            commands.append(_run_worker(command))
            workers.append(strict_json(output))
        gates = {
            str(record["training_root"]): record["selection"]["gates_without_assigned_fraction"]
            for record in workers
        }
        all_gates = all(
            all(value for value in root_gates.values()) for root_gates in gates.values()
        )
        payload = {
            "artifact_type": "compact_responsibility_birth_phase_a_result_v1",
            "timestamp_utc": factorial.timestamp_utc(),
            "status": "PASS",
            "phase_a_decision": ("AUTHORIZE_AUDIT" if all_gates else "STOP_PHASE_B"),
            "seal_sha256": sha256_file(SEAL),
            "phase_a_attempt_sha256": marker_sha,
            "workers": workers,
            "gates": gates,
            "all_phase_a_gates_pass": all_gates,
            "commands": commands,
        }
    except BaseException as error:
        payload = {
            "artifact_type": "compact_responsibility_birth_phase_a_result_v1",
            "timestamp_utc": factorial.timestamp_utc(),
            "status": "FAIL",
            "phase_a_decision": "STOP_PHASE_B",
            "scientific_decision": "UNAVAILABLE",
            "seal_sha256": sha256_file(SEAL),
            "phase_a_attempt_sha256": marker_sha,
            "error": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "workers": workers,
            "commands": commands,
        }
    exclusive_json(PHASE_A_RESULT, payload)
    return strict_json(PHASE_A_RESULT)
