"""Fresh, fail-closed compact responsibility-birth allocation experiment.

This module implements the once-only lifecycle frozen in
``20260717_compact_responsibility_birth_allocation_iter3_PREREG.md``.  The
failed first namespace is never imported: outcome-neutral mechanism code is
reproduced here under fresh roots and domains so lifecycle and source closure
remain independently auditable.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import subprocess
import sys
import tarfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

if __package__:
    from benchmarks import compact_occupancy_refinement_factorial as factorial
else:
    import compact_occupancy_refinement_factorial as factorial

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianPointProposal,
    ObservationSamples,
)
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.optim import compact_trainer as compact_trainer_module
from rtgs.optim.compact_trainer import (
    CompactTrainConfig,
    CompactTrainer,
    build_view_schedule,
    step_sample_seed,
)
from rtgs.optim.density import SelectedBirthReceipt, apply_selected_birth_surgery
from rtgs.render.point_base import PointRenderOutput
from rtgs.render.torch_points import TorchPointRasterizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
PREREGISTRATION = RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_PREREG.md"
PREREGISTRATION_ADDENDUM = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_PREREG_ADDENDUM_1.md"
)
PREREGISTRATION_INITIAL_FAIL_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_PREREG_REVIEW_INITIAL_FAIL.md"
)
PREREGISTRATION_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_PREREG_REVIEW.md"
)
IMPLEMENTATION_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_IMPLEMENTATION_REVIEW.md"
)
SEAL_ATTEMPT = RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_SEAL_ATTEMPT.json"
SEAL_FAILURE = RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_SEAL_FAILURE.json"
SEAL = RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_SEAL.json"
PHASE_A_ATTEMPT = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_PHASE_A_ATTEMPT.json"
)
PHASE_A_RESULT = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_PHASE_A_RESULT.json"
)
PHASE_A_AUDIT = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_PHASE_A_AUDIT.json"
)
PHASE_B_ATTEMPT = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_PHASE_B_ATTEMPT.json"
)
RESULT = RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_RESULT.json"
EXECUTED_SOURCES = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_EXECUTED_SOURCES.tar"
)
RUN_DIR = ROOT / "runs/compact_responsibility_birth_allocation_iter3_20260717"
VISUALIZER = ROOT / "benchmarks/visualize_compact_responsibility_birth_allocation_iter3.py"
ITER3_FAILURE_AUDIT = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter3_FAILURE_AUDIT.md"
)

IMPORTED_PREREGISTRATION = RESULTS / "20260717_compact_responsibility_birth_allocation_PREREG.md"
IMPORTED_PREMATURE_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_PREREG_REVIEW.md"
)
IMPORTED_INITIAL_FAIL_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_PREREG_REVIEW_INITIAL_FAIL.md"
)
IMPORTED_ADDENDUM_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_PREREG_REVIEW_ADDENDUM_1.md"
)
FAILURE_AUDIT = RESULTS / "20260717_compact_responsibility_birth_allocation_FAILURE_AUDIT.md"
ITER2_FAILURE_AUDIT = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter2_FAILURE_AUDIT.md"
)
ITER2_FAILURE_AUDIT_ADDENDUM = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter2_FAILURE_AUDIT_ADDENDUM_1.md"
)
ITER2_PREREGISTRATION = RESULTS / "20260717_compact_responsibility_birth_allocation_iter2_PREREG.md"
ITER2_PREREGISTRATION_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter2_PREREG_REVIEW.md"
)
ITER2_IMPLEMENTATION_REVIEW = (
    RESULTS / "20260717_compact_responsibility_birth_allocation_iter2_IMPLEMENTATION_REVIEW.md"
)

EXPECTED_LIFECYCLE_SHA256 = {
    PREREGISTRATION: "352133e2830d921af272c472cfe41b3d7114643627fd7d585b4bef8ac2613f81",
    PREREGISTRATION_ADDENDUM: ("b96dfbe572563c18fd319e665f7adf7bad0408a4347585c7118a1c4b9277ec8b"),
    PREREGISTRATION_INITIAL_FAIL_REVIEW: (
        "dab05011c2531a837873ca2f286ac86a2a580c688951d61688a466cc0a3e76ac"
    ),
    PREREGISTRATION_REVIEW: ("39d388acd16178fffe9fb80a8c531fc3efec6fce42501351b5f09d0ce472277b"),
    ITER2_FAILURE_AUDIT: "b0992cf6a190b9ac9f9bde5701b09abb05af8617c0a6234182355cf49f80b0fa",
    ITER2_FAILURE_AUDIT_ADDENDUM: (
        "f75b7943b4bf29b38d27599839e5c174ee9bf1ee98174f0695a56638feecb386"
    ),
    ITER2_PREREGISTRATION: ("e0be823718b1b074d0c720d1cccf8800a18bd72580877fb1e1f44c30dcb5806c"),
    ITER2_PREREGISTRATION_REVIEW: (
        "59b60d6516ee3547978bb41cf5faa51fc2353f262c136feec71fc6a14def22a5"
    ),
    ITER2_IMPLEMENTATION_REVIEW: (
        "217c443e4a2f17291653b7a742d9702b79dfc492875930bc101fbb56d6e96e52"
    ),
}

TEACHER_BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
PROXY_BUNDLE = ROOT / "runs/compact_occupancy_scalar_ablation_20260717/proxy_bundles/center"
INIT_PLY = ROOT / "runs/compact_occupancy_scalar_ablation_20260717/stage_b/center/gaussians.ply"
EXPECTED_INIT_PLY_SHA256 = "0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e"
EXPECTED_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
EXPLICIT_EXTENT = 1.5469313859939577
EXPECTED_PREREQUISITE_SHA256 = {
    "benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_PREREG.md": (
        "5b3f721307b2f85446a1862584406ec9383ea63a75ff93585b7840f244861ef8"
    ),
    "benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json": (
        "c0a278a8cc41f12632be121b14937f9fc2a2a03cd03716bae96b5bd9d6510116"
    ),
    "benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_AUDIT.md": (
        "448369940fe376bd72547165d63171d622c9a766782fb6b9ce205c1c8120d16c"
    ),
    "benchmarks/results/20260716_residual_responsibility_density_PREREG.md": (
        "f65b4afecc09532dd2113c353043afebc8607e11cbdc714cee326ceec8e3e368"
    ),
}
EXPECTED_FROZEN_INPUT_BINDINGS_SHA256 = (
    "682ea803dd928abcf27e4ddca5367b2bf9518365914e8dbb1a13528e5ba23f1d"
)
EXPECTED_FROZEN_RUNTIME_SHA256 = "165f117243cbc93fa5374cc2b58c257d96ee5d62ca0e49b57f49dc7902772489"
FROZEN_RUNTIME_PROJECTION_FIELDS = (
    "python",
    "executable",
    "numpy",
    "torch",
    "torch_git_version",
    "torch_cuda",
    "gsplat",
    "cuda_device",
    "cuda_capability",
    "cuda_total_memory",
    "cuda_multiprocessor_count",
    "cuda_uuid",
    "cuda_pci_bus_id",
    "nvidia_smi_driver_device",
    "cuda_matmul_fp32_precision",
    "deterministic_algorithms",
    "sys_path",
    "torch_generated_import_path",
    "pythonpath",
    "preload",
    "preload_sha256",
    "effective_ld_preload",
)

TRAIN_ROOTS = (80101, 80102, 80103)
EVALUATION_ROOTS = (80201, 80202, 80203)
SPLIT_ROOTS = (80301, 80302, 80303)
SHUFFLE_ROOTS = (80401, 80402, 80403)
FOCUSED_TRAIN_ROOTS = (996001, 996002, 996003)
FOCUSED_EVALUATION_ROOTS = (996101, 996102)
FOCUSED_SPLIT_ROOTS = (996201, 996202, 996203)
FOCUSED_SHUFFLE_ROOTS = (996301, 996302, 996303)
OFFICIAL_ROOTS = frozenset(TRAIN_ROOTS + EVALUATION_ROOTS + SPLIT_ROOTS + SHUFFLE_ROOTS)
FOCUSED_ROOTS = frozenset(
    FOCUSED_TRAIN_ROOTS + FOCUSED_EVALUATION_ROOTS + FOCUSED_SPLIT_ROOTS + FOCUSED_SHUFFLE_ROOTS
)
FIRST_FAILED_ROOTS = frozenset(
    (
        77001,
        77002,
        77003,
        77101,
        77102,
        77103,
        77201,
        77202,
        77203,
        77301,
        77302,
        77303,
        993001,
        993002,
        993003,
        993101,
        993102,
    )
)
ITER2_RETIRED_ROOTS = frozenset(
    (
        78101,
        78102,
        78103,
        78201,
        78202,
        78203,
        78301,
        78302,
        78303,
        78401,
        78402,
        78403,
        994001,
        994002,
        994003,
        994101,
        994102,
        994201,
        994202,
        994203,
        994301,
        994302,
        994303,
    )
)
CONTAMINATED_CANDIDATE_ROOTS = frozenset(
    (
        79101,
        79102,
        79103,
        79201,
        79202,
        79203,
        79301,
        79302,
        79303,
        79401,
        79402,
        79403,
        995001,
        995002,
        995003,
        995101,
        995102,
        995201,
        995202,
        995203,
        995301,
        995302,
        995303,
    )
)
FAILED_ROOTS = FIRST_FAILED_ROOTS | ITER2_RETIRED_ROOTS | CONTAMINATED_CANDIDATE_ROOTS

ARMS = ("G", "R", "U")
ARM_ORDER = {
    TRAIN_ROOTS[0]: ("G", "R", "U"),
    TRAIN_ROOTS[1]: ("R", "U", "G"),
    TRAIN_ROOTS[2]: ("U", "G", "R"),
}
STRATA = ("small_low", "small_high", "large_low", "large_high")
STRATUM_CODES = {name: index for index, name in enumerate(STRATA)}
GROUP_ORDER = ("means", "quats", "scales", "opacities", "sh0", "shN")
SCORE_STEPS = 35
TRAIN_ITERATIONS = 140
TRAIN_ATTEMPTS = 128
BANK_ATTEMPTS = 4096
QUOTA_PER_STRATUM = 8
CHECKPOINTS = (0, 35, 70, 105, 140)
RECOVERY_CHECKPOINTS = ("35_post", "70", "105", "140")
EVALUATION_SEED_DOMAIN = "rtgs.compact-responsibility-birth.iter3.eval.v1"
DOMAIN_PREFIX = b"rtgs.compact-responsibility-birth.iter3.v1\0"
FOCUSED_TEST_ENV = "RTGS_RESPONSIBILITY_BIRTH_ITER3_FOCUSED_TEST"
PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33")
WORKER_TIMEOUT_SECONDS = 1800
SEAL_TRANSCRIPT_TAIL_BYTES = 8192
SEAL_FAILURE_STAGES = (
    "post_claim_entry_validation",
    "post_claim_namespace_validation",
    "lifecycle_document_validation",
    "review_validation",
    "static_root_proof",
    "dynamic_root_proof",
    "binding_before_verification",
    "verification_1",
    "verification_2",
    "verification_3",
    "verification_gate",
    "binding_after_verification",
    "archive_creation",
    "archive_validation",
    "binding_after_archive",
    "seal_publication",
    "post_publication_verification",
)
SEAL_LITERAL_COMMAND = (
    "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 "
    ".venv/bin/python benchmarks/compact_responsibility_birth_allocation_iter3.py seal"
)
ALPHA_EVIDENCE_UPPER_TOLERANCE = 2e-6

RGBAccessGuard = factorial.RGBAccessGuard


class ProtocolInvalid(RuntimeError):
    """The executable state differs from the frozen iter3 protocol."""


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


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def generator_state_hash(generator: torch.Generator) -> str:
    return tensor_hash(generator.get_state())


def encode_atom(value: int | str) -> bytes:
    if type(value) is int and value >= 0:
        return str(value).encode("ascii")
    if type(value) is str:
        return value.encode("utf-8")
    raise TypeError("seed atoms must be non-negative built-in ints or strings")


_AUTHORIZED_PHASES: set[str] = set()


def _required_phase_for_root(root: int) -> frozenset[str]:
    if root in TRAIN_ROOTS or root in SHUFFLE_ROOTS:
        return frozenset(("phase-a", "phase-b"))
    if root in EVALUATION_ROOTS or root in SPLIT_ROOTS:
        return frozenset(("phase-b",))
    return frozenset()


def _require_root_authorized(
    root: int,
    *,
    domain: str | None = None,
    focused: bool | None = None,
    marker_path: Path | None = None,
    marker_artifact_type: str | None = None,
) -> None:
    """Reject failed roots and unmarked official roots before any mechanism.

    The optional arguments make the guard directly testable without constructing a
    generator.  Production callers authorize a marker once and then use the process-local
    authorization set; a supplied marker is always strictly re-read first.
    """
    del domain
    if root in FAILED_ROOTS:
        raise ProtocolInvalid("the failed namespace can never be reused")
    if focused is True and root in OFFICIAL_ROOTS:
        raise ProtocolInvalid("focused mode rejected an official iter3 root")
    if focused is False and root in FOCUSED_ROOTS:
        raise ProtocolInvalid("official mode rejected a focused-only root")
    if marker_path is not None:
        if marker_artifact_type is None:
            raise ValueError("marker_artifact_type is required with marker_path")
        marker = strict_json(marker_path)
        if marker.get("artifact_type") != marker_artifact_type:
            raise ProtocolInvalid("root authorization marker type changed")
    required = _required_phase_for_root(root)
    if required and not (_AUTHORIZED_PHASES & required):
        raise ProtocolInvalid(f"official iter3 root {root} reached a mechanism before its marker")


def domain_seed(label: str, root: int, *parts: int | str) -> int:
    _require_root_authorized(root)
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
    if values & FAILED_ROOTS:
        raise ProtocolInvalid("failed-namespace roots are forbidden")
    forbidden = OFFICIAL_ROOTS if focused else FOCUSED_ROOTS
    overlap = sorted(values & forbidden)
    if overlap:
        mode = "focused" if focused else "official"
        raise ProtocolInvalid(f"{mode} mode rejected root(s): {overlap}")
    for root in roots:
        _require_root_authorized(root)


def frozen_config(
    training_root: int,
    *,
    focused: bool = False,
    device: str = "cuda:0",
    iterations: int = TRAIN_ITERATIONS,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
) -> CompactTrainConfig:
    allowed = FOCUSED_TRAIN_ROOTS if focused else TRAIN_ROOTS
    if training_root not in allowed:
        raise ValueError("training root is not in the selected frozen domain")
    _reject_roots_for_mode(training_root, focused=focused)
    return CompactTrainConfig(
        iterations=iterations,
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
        checkpoints=checkpoints,
        evaluate_checkpoint_risks=False,
        sh_degree=0,
        sh_color_activation="hard",
        sh_smu1_mu=0.00784313725490196,
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
    )


def _finite_list(values: torch.Tensor) -> list[float]:
    if not bool(torch.isfinite(values).all()):
        raise ProtocolInvalid("non-finite tensor cannot enter evidence")
    return [float(value) for value in values.detach().cpu().reshape(-1).tolist()]


def _tensor_receipt(value: torch.Tensor, *, include_values: bool = False) -> dict[str, Any]:
    detached = value.detach()
    record: dict[str, Any] = {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "device": str(detached.device),
        "finite": bool(torch.isfinite(detached).all())
        if detached.is_floating_point() or detached.is_complex()
        else True,
        "sha256": tensor_hash(detached),
    }
    if include_values:
        record["values"] = detached.cpu().tolist()
    return record


def _group_fields(group: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in (
        "name",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "fused",
        "capturable",
        "differentiable",
    ):
        if name in group:
            value = group[name]
            fields[name] = list(value) if isinstance(value, tuple) else value
    return fields


def optimizer_state_record(
    params: Mapping[str, torch.Tensor],
    optimizers: Mapping[str, torch.optim.Optimizer],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name in GROUP_ORDER:
        parameter = params[name]
        optimizer = optimizers[name]
        if len(optimizer.param_groups) != 1:
            raise ProtocolInvalid("optimizer group count changed")
        group = optimizer.param_groups[0]
        if len(group["params"]) != 1 or group["params"][0] is not parameter:
            raise ProtocolInvalid("optimizer parameter binding changed")
        state = optimizer.state.get(parameter, {})
        step_value = state.get("step", 0)
        step = int(step_value.item()) if isinstance(step_value, torch.Tensor) else int(step_value)
        moments: dict[str, Any] = {}
        for moment_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if moment_name in state:
                moment = state[moment_name]
                if moment.shape != parameter.shape or not bool(torch.isfinite(moment).all()):
                    raise ProtocolInvalid("optimizer moment is invalid")
                moments[moment_name] = _tensor_receipt(moment)
        groups[name] = {
            "parameter": _tensor_receipt(parameter),
            "step": step,
            "group": _group_fields(group),
            "moments": moments,
        }
    record = {"order": list(GROUP_ORDER), "groups": groups}
    record["semantic_sha256"] = canonical_hash(record)
    return record


def _lineage_family(item: Mapping[str, Any]) -> str:
    if item["operator"] == "clone":
        return "clone"
    return f"split_child_{int(item['child_ordinal'])}"


def _summary(values: torch.Tensor) -> dict[str, Any]:
    detached = values.detach()
    if detached.numel() == 0:
        return {"count": 0, "finite": True}
    if not bool(torch.isfinite(detached).all()):
        raise ProtocolInvalid("lineage summary encountered non-finite values")
    flat = detached.to(torch.float64).reshape(-1)
    return {
        "count": int(detached.shape[0]),
        "elements": int(detached.numel()),
        "mean": float(flat.mean()),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "l2": float(torch.linalg.vector_norm(flat)),
        "sha256": tensor_hash(detached),
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    x = left.detach().to(torch.float64)
    y = right.detach().to(torch.float64)
    if x.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return None
    value = float((x * y).sum() / denominator)
    return value if math.isfinite(value) else None


def _stable_ordinal_ranks(
    values: torch.Tensor, rows: Sequence[int], persistent_ids: torch.Tensor
) -> torch.Tensor:
    ordered = sorted(
        rows,
        key=lambda row: (float(values[row]), int(persistent_ids[row])),
    )
    rank_by_row = {row: rank for rank, row in enumerate(ordered)}
    return torch.tensor([rank_by_row[row] for row in rows], dtype=torch.float64)


class ResponsibilityBirthController:
    """Collect exact score evidence and optionally execute one matched birth wave."""

    def __init__(
        self,
        *,
        arm: Literal["G", "R", "U"] | None,
        shuffle_root: int,
        split_root: int | None = None,
        score_steps: int = SCORE_STEPS,
        quota_per_stratum: int = QUOTA_PER_STRATUM,
        expected_view_visits: int = 5,
        expected_n_before: int = 835,
        expected_clone_count: int = 16,
        expected_split_count: int = 16,
        expected_selection_sha256: str | None = None,
        expected_pre_state_sha256: str | None = None,
        on_pre_surgery: Callable[[Gaussians3D, dict[str, Any]], None] | None = None,
        on_post_surgery: Callable[[Gaussians3D, dict[str, Any]], None] | None = None,
    ) -> None:
        if arm not in (*ARMS, None):
            raise ValueError("arm must be G, R, U, or None")
        if arm is not None and split_root is None:
            raise ValueError("a birth arm requires a split root")
        if (
            min(
                score_steps,
                quota_per_stratum,
                expected_view_visits,
                expected_n_before,
            )
            <= 0
        ):
            raise ValueError("controller dimensions must be positive")
        self.arm = arm
        self.shuffle_root = shuffle_root
        self.split_root = split_root
        self.score_steps = score_steps
        self.quota_per_stratum = quota_per_stratum
        self.expected_view_visits = expected_view_visits
        self.expected_n_before = expected_n_before
        self.expected_clone_count = expected_clone_count
        self.expected_split_count = expected_split_count
        self.expected_selection_sha256 = expected_selection_sha256
        self.expected_pre_state_sha256 = expected_pre_state_sha256
        self.on_pre_surgery = on_pre_surgery
        self.on_post_surgery = on_post_surgery
        self._ids: torch.Tensor | None = None
        self._extent: float | None = None
        self._n_views: int | None = None
        self._attempts: int | None = None
        self._params: dict[str, torch.Tensor] | None = None
        self._optimizers: dict[str, torch.optim.Optimizer] | None = None
        self._initial_params: dict[str, torch.Tensor] = {}
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
        self._state_checkpoints: dict[str, Any] = {}
        self._raw_state_arrays: dict[str, np.ndarray] = {}
        self._raw_state_metadata: dict[str, Any] = {}

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
        if self._ids is not None:
            raise RuntimeError("controller cannot be rebound")
        n = int(params["means"].shape[0])
        if n != self.expected_n_before:
            raise ProtocolInvalid(
                f"controller expected {self.expected_n_before} initial rows, got {n}"
            )
        device = params["means"].device
        self._ids = torch.arange(n, device=device, dtype=torch.long)
        self._extent = float(extent)
        self._n_views = int(n_views)
        self._attempts = int(attempts_per_step)
        self._params = params
        self._optimizers = optimizers
        self._initial_params = {name: params[name].detach().clone() for name in GROUP_ORDER}
        self._r_by_view = torch.zeros(n_views, n, dtype=torch.float64, device=device)
        self._s_by_view = torch.zeros(n_views, n, dtype=torch.float64, device=device)
        self._view_step_count = torch.zeros(n_views, dtype=torch.int64, device=device)
        self._g_sum = torch.zeros(n, dtype=torch.float32, device=device)
        self._g_count = torch.zeros(n, dtype=torch.int64, device=device)
        self._state_checkpoints["0"] = self._state_record(params, optimizers)
        self._capture_raw_state("0", params, optimizers)

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
                raise ProtocolInvalid("compositor basis remained enabled after score window")
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
        if visible is None:
            visible = torch.empty(0, dtype=torch.long, device=self.persistent_ids.device)
        if basis is None:
            if visible.numel():
                raise ProtocolInvalid("requested compositor basis is missing")
            native_r = torch.empty(0, dtype=torch.float32, device=self.persistent_ids.device)
            native_s = torch.empty_like(native_r)
        else:
            if basis.ndim != 2 or basis.shape != (visible.numel(), 3) or not basis.requires_grad:
                raise ProtocolInvalid("invalid compositor basis/visible mapping")
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
            if contracted.requires_grad or contracted.dtype != torch.float32:
                raise ProtocolInvalid("VJP precision or graph contract changed")
            if not bool(torch.isfinite(contracted).all()):
                raise ProtocolInvalid("VJP contains non-finite values")
            native_r = contracted[:, 0].contiguous()
            native_s = contracted[:, 1].contiguous()
            if not bool(torch.equal(contracted[:, 2], torch.zeros_like(contracted[:, 2]))):
                raise ProtocolInvalid("unused VJP channel is nonzero")
        output.compositing_color_basis = None

        native_r_sum = native_r.sum(dtype=torch.float32)
        native_s_sum = native_s.sum(dtype=torch.float32)
        global_r = torch.zeros(
            self.persistent_ids.numel(),
            dtype=torch.float64,
            device=self.persistent_ids.device,
        )
        global_s = torch.zeros_like(global_r)
        if visible.numel():
            global_r.index_add_(0, visible, native_r.to(dtype=torch.float64) / attempts)
            global_s.index_add_(0, visible, native_s.to(dtype=torch.float64) / attempts)
        self._r_by_view[view_index].add_(global_r)
        self._s_by_view[view_index].add_(global_s)
        assert self._view_step_count is not None
        self._view_step_count[view_index] += 1

        alpha_residual = (
            native_active * native_error * output.alpha.detach().to(dtype=torch.float32)
        ).sum(dtype=torch.float32)
        alpha_support = (native_active * output.alpha.detach().to(dtype=torch.float32)).sum(
            dtype=torch.float32
        )
        native_alpha = output.alpha.detach().to(dtype=torch.float32).contiguous()
        if not torch.allclose(native_r_sum, alpha_residual, atol=2e-6, rtol=2e-5):
            raise ProtocolInvalid("residual VJP/alpha identity failed")
        if not torch.allclose(native_s_sum, alpha_support, atol=2e-6, rtol=2e-5):
            raise ProtocolInvalid("support VJP/alpha identity failed")
        active_error_sum = (native_active * native_error).sum(dtype=torch.float32)
        self._assigned_numerator += float(native_r_sum)
        self._assigned_denominator += float(active_error_sum)
        self._step_evidence.append(
            {
                "step": step,
                "view_index": view_index,
                "attempts": attempts,
                "visible_to_global": visible.detach().cpu().tolist(),
                "visible_to_global_sha256": tensor_hash(visible),
                "native_residual_visible_float32": _finite_list(native_r),
                "native_support_visible_float32": _finite_list(native_s),
                "native_residual_visible_sha256": tensor_hash(native_r),
                "native_support_visible_sha256": tensor_hash(native_s),
                "native_residual_parent_sum_float32": float(native_r_sum),
                "native_support_parent_sum_float32": float(native_s_sum),
                "alpha_residual_sum_float32": float(alpha_residual),
                "alpha_support_sum_float32": float(alpha_support),
                "active_error_sum_float32": float(active_error_sum),
                "residual_global_divided_float64": _finite_list(global_r),
                "support_global_divided_float64": _finite_list(global_s),
                "residual_global_divided_sha256": tensor_hash(global_r),
                "support_global_divided_sha256": tensor_hash(global_s),
                "point_loss_float32_sha256": tensor_hash(native_error),
                "active_sha256": tensor_hash(active),
                "active_float32": _finite_list(native_active),
                "active_float32_sha256": tensor_hash(native_active),
                "error_float32": _finite_list(native_error),
                "error_float32_sha256": tensor_hash(native_error),
                "alpha_float32": _finite_list(native_alpha),
                "alpha_float32_sha256": tensor_hash(native_alpha),
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
        if self._g_sum is None or self._g_count is None or self._params is None:
            raise RuntimeError("controller is not bound")
        self._post_calls[step] = self._post_calls.get(step, 0) + 1
        if self._post_calls[step] != 1:
            raise ProtocolInvalid("gradient score step used more than one microbatch")
        evidence = self._step_evidence[-1]
        gradients: dict[str, Any] = {}
        for name in GROUP_ORDER:
            gradient = self._params[name].grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise ProtocolInvalid(f"missing/non-finite {name} gradient")
            gradients[name] = _tensor_receipt(gradient)
        evidence["six_parameter_gradients"] = gradients
        evidence["six_parameter_gradients_sha256"] = canonical_hash(gradients)

        visible = output.visible
        means2d = output.means2d
        if visible is None:
            visible = torch.empty(0, dtype=torch.long, device=self.persistent_ids.device)
        evidence["screen_gradient_scale"] = {
            "width": width,
            "height": height,
            "factor": max(width, height) * 0.5,
        }
        if visible.numel() == 0:
            evidence["gradient_visible_float32"] = []
            evidence["gradient_visible_sha256"] = tensor_hash(torch.empty(0, dtype=torch.float32))
            evidence["means2d_gradient"] = None
            return
        if means2d is None or means2d.grad is None:
            raise ProtocolInvalid("visible render lacks screen gradients")
        if not bool(torch.isfinite(means2d.grad).all()):
            raise ProtocolInvalid("screen gradient is non-finite")
        native = torch.linalg.vector_norm(means2d.grad.detach(), dim=-1)
        native = (native * (max(width, height) * 0.5)).to(dtype=torch.float32)
        self._g_sum.index_add_(0, visible, native)
        self._g_count.index_add_(0, visible, torch.ones_like(visible, dtype=torch.int64))
        evidence["gradient_visible_float32"] = _finite_list(native)
        evidence["gradient_visible_sha256"] = tensor_hash(native)
        evidence["means2d_gradient"] = _tensor_receipt(means2d.grad.detach(), include_values=True)

    def _state_record(
        self,
        params: Mapping[str, torch.Tensor],
        optimizers: Mapping[str, torch.optim.Optimizer],
    ) -> dict[str, Any]:
        record = {
            "parameters": parameter_state_record(params),
            "optimizers": optimizer_state_record(params, optimizers),
            "persistent_ids": _tensor_receipt(self.persistent_ids, include_values=True),
        }
        record["semantic_sha256"] = canonical_hash(record)
        return record

    def _capture_raw_state(
        self,
        label: str,
        params: Mapping[str, torch.Tensor],
        optimizers: Mapping[str, torch.optim.Optimizer],
    ) -> None:
        if label in self._raw_state_metadata:
            raise ProtocolInvalid(f"raw state label {label} was captured twice")
        groups: dict[str, Any] = {}
        for name in GROUP_ORDER:
            parameter = params[name]
            optimizer = optimizers[name]
            state = optimizer.state.get(parameter, {})
            entries: dict[str, torch.Tensor] = {"parameter": parameter.detach()}
            for moment_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if moment_name in state:
                    entries[moment_name] = state[moment_name].detach()
            descriptors: dict[str, Any] = {}
            for entry_name, tensor in entries.items():
                key = f"{label}__{name}__{entry_name}"
                array = tensor.cpu().contiguous().numpy().copy()
                self._raw_state_arrays[key] = array
                descriptors[entry_name] = {
                    "key": key,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "sha256": array_hash(array),
                }
            step_value = state.get("step", 0)
            step = (
                int(step_value.item()) if isinstance(step_value, torch.Tensor) else int(step_value)
            )
            groups[name] = {
                "step": step,
                "group": _group_fields(optimizer.param_groups[0]),
                "tensors": descriptors,
            }
        ids_key = f"{label}__persistent_ids"
        ids = self.persistent_ids.detach().cpu().contiguous().numpy().copy()
        self._raw_state_arrays[ids_key] = ids
        state_metadata: dict[str, Any] = {
            "label": label,
            "group_order": list(GROUP_ORDER),
            "groups": groups,
            "persistent_ids": {
                "key": ids_key,
                "dtype": ids.dtype.str,
                "shape": list(ids.shape),
                "sha256": array_hash(ids),
            },
        }
        state_metadata["semantic_sha256"] = canonical_hash(state_metadata)
        self._raw_state_metadata[label] = state_metadata

    def state_archive_payload(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        metadata: dict[str, Any] = {
            "schema": "rtgs.compact_responsibility_birth_iter3.states.v1",
            "labels": list(self._raw_state_metadata),
            "states": self._raw_state_metadata,
        }
        metadata["semantic_sha256"] = canonical_hash(metadata)
        return dict(self._raw_state_arrays), metadata

    def _finalize_selection(
        self,
        params: Mapping[str, torch.Tensor],
        optimizers: Mapping[str, torch.optim.Optimizer],
    ) -> dict[str, Any]:
        assert self._r_by_view is not None
        assert self._s_by_view is not None
        assert self._view_step_count is not None
        assert self._g_sum is not None
        assert self._g_count is not None
        assert self._extent is not None
        expected_visits = torch.full_like(self._view_step_count, self.expected_view_visits)
        if not torch.equal(self._view_step_count, expected_visits):
            raise ProtocolInvalid("score window is not exact visits-per-view")
        if len(self._step_evidence) != self.score_steps:
            raise ProtocolInvalid("score evidence does not contain every step")
        if set(self._pre_calls) != set(range(1, self.score_steps + 1)):
            raise ProtocolInvalid("pre-backward score call coverage changed")
        if set(self._post_calls) != set(range(1, self.score_steps + 1)):
            raise ProtocolInvalid("post-backward score call coverage changed")
        if int(params["means"].shape[0]) != self.expected_n_before:
            raise ProtocolInvalid("pre-surgery cardinality changed")
        expected_ids = torch.arange(
            self.expected_n_before,
            dtype=torch.long,
            device=self.persistent_ids.device,
        )
        if not torch.equal(self.persistent_ids, expected_ids):
            raise ProtocolInvalid("pre-surgery persistent IDs changed")
        scale_max = params["scales"].detach().exp().amax(dim=-1)
        if not bool(torch.isfinite(scale_max).all()):
            raise ProtocolInvalid("step-35 scales are non-finite")

        r_view = self._r_by_view / self._view_step_count[:, None]
        s_view = self._s_by_view / self._view_step_count[:, None]
        residual = r_view.mean(dim=0)
        support = s_view.mean(dim=0)
        gradient = self._g_sum / self._g_count.clamp_min(1).to(torch.float32)
        record = build_matched_selections(
            gradient_score=gradient,
            residual_score=residual,
            support_score=support,
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
            else None
        )
        if assigned_fraction is None or not math.isfinite(assigned_fraction):
            raise ProtocolInvalid("assigned-residual denominator is not finite positive")
        record["scores"] = {
            "G_float32": _finite_list(gradient),
            "R_float64": _finite_list(residual),
            "S_float64": _finite_list(support),
            "R_by_view_float64": [_finite_list(row) for row in r_view],
            "S_by_view_float64": [_finite_list(row) for row in s_view],
            "visible_step_count": self._g_count.cpu().tolist(),
            "view_step_count": self._view_step_count.cpu().tolist(),
            "scale_max": _finite_list(scale_max),
        }
        record["assigned_residual"] = {
            "native_float32_numerator_reduced_before_division": self._assigned_numerator,
            "native_float32_denominator": self._assigned_denominator,
            "fraction": assigned_fraction,
        }
        eligible_rows = record["eligible_rows"]
        eligible_index = torch.tensor(eligible_rows, dtype=torch.long, device=residual.device)
        g_eligible = gradient[eligible_index]
        r_eligible = residual[eligible_index]
        stable_g_rank = _stable_ordinal_ranks(gradient, eligible_rows, self.persistent_ids)
        stable_r_rank = _stable_ordinal_ranks(residual, eligible_rows, self.persistent_ids)
        positive_views = (s_view > 0).sum(dim=0)
        selected_one_view_fraction: dict[str, float] = {}
        selection_lineage: dict[str, list[dict[str, Any]]] = {}
        selected_distributions: dict[str, Any] = {}
        opacity = torch.sigmoid(params["opacities"].detach())
        for arm, rows in record["selected_rows"].items():
            index = torch.tensor(rows, dtype=torch.long, device=residual.device)
            one_view = int((positive_views[index] == 1).sum())
            if one_view != 0:
                raise ProtocolInvalid("eligible selection contains a one-view parent")
            selected_one_view_fraction[arm] = one_view / len(rows)
            selection_lineage[arm] = [
                {
                    "persistent_id": int(self.persistent_ids[row]),
                    "pre_surgery_row": int(row),
                    "lineage": "original",
                    "operator": (
                        "clone" if float(scale_max[row]) <= 0.01 * self._extent else "split"
                    ),
                }
                for row in rows
            ]
            selected_distributions[arm] = {
                "scale_max": _summary(scale_max[index]),
                "opacity": _summary(opacity[index]),
                "residual": _summary(residual[index]),
                "support": _summary(support[index]),
            }
        concentration_count = max(1, math.ceil(0.1 * len(eligible_rows)))

        def concentration(values: torch.Tensor) -> dict[str, Any]:
            candidate = values[eligible_index].clamp_min(0)
            total = candidate.sum(dtype=torch.float64)
            top = torch.topk(candidate, concentration_count).values.sum(dtype=torch.float64)
            return {
                "eligible_total": float(total),
                "top_10pct_count": concentration_count,
                "top_10pct_fraction": (float(top / total) if float(total) > 0 else None),
            }

        record["non_gating_diagnostics"] = {
            "G_R_pearson_eligible": _pearson(g_eligible, r_eligible),
            "G_R_stable_rank_spearman_eligible": _pearson(stable_g_rank, stable_r_rank),
            "R_by_view_float64": record["scores"]["R_by_view_float64"],
            "R_max_view_per_parent_float64": _finite_list(r_view.max(dim=0).values),
            "support_concentration": concentration(support),
            "residual_concentration": concentration(residual),
            "eligible_distributions": {
                "scale_max": _summary(scale_max[eligible_index]),
                "opacity": _summary(opacity[eligible_index]),
                "residual": _summary(r_eligible),
                "support": _summary(support[eligible_index]),
            },
            "selected_distributions": selected_distributions,
            "positive_support_view_count": positive_views.cpu().tolist(),
            "selected_one_view_fraction": selected_one_view_fraction,
            "selection_lineage": selection_lineage,
            "changes_no_gate": True,
        }
        record["gates_without_assigned_fraction"]["assigned_fraction_ge_0_10"] = (
            math.isfinite(self._assigned_numerator)
            and math.isfinite(self._assigned_denominator)
            and self._assigned_denominator > 0
            and math.isfinite(assigned_fraction)
            and assigned_fraction >= 0.10
        )
        record["all_phase_a_gates_pass"] = all(record["gates_without_assigned_fraction"].values())
        record["step_evidence_sha256"] = canonical_hash(self._step_evidence)
        pre_state = self._state_record(params, optimizers)
        record["pre_surgery_state"] = pre_state
        record["semantic_sha256"] = canonical_hash(
            {
                "selection": {
                    key: record[key]
                    for key in (
                        "eligible_rows",
                        "strata",
                        "shuffle",
                        "selected_rows",
                        "overlaps",
                    )
                },
                "scores": record["scores"],
                "assigned_residual": record["assigned_residual"],
                "gates": record["gates_without_assigned_fraction"],
                "step_evidence_sha256": record["step_evidence_sha256"],
                "pre_surgery_state_sha256": pre_state["semantic_sha256"],
            }
        )
        return record

    def _validate_surgery(
        self,
        *,
        old_state: Mapping[str, Mapping[str, torch.Tensor]],
        old_groups: Mapping[str, Mapping[str, Any]],
        new_params: Mapping[str, torch.Tensor],
        optimizers: Mapping[str, torch.optim.Optimizer],
        receipt: SelectedBirthReceipt,
    ) -> None:
        expected_after = self.expected_n_before + (
            self.expected_clone_count + self.expected_split_count
        )
        if receipt.n_before != self.expected_n_before or receipt.n_after != expected_after:
            raise ProtocolInvalid("selected-birth count arithmetic changed")
        if (
            len(receipt.clone_parent_rows) != self.expected_clone_count
            or len(receipt.split_parent_rows) != self.expected_split_count
        ):
            raise ProtocolInvalid("clone/split operator quotas changed")
        if len(receipt.newborns) != (self.expected_clone_count + 2 * self.expected_split_count):
            raise ProtocolInvalid("newborn count changed")
        for name in GROUP_ORDER:
            parameter = new_params[name]
            optimizer = optimizers[name]
            group = optimizer.param_groups[0]
            if _group_fields(group) != old_groups[name]:
                raise ProtocolInvalid("optimizer group fields changed through surgery")
            state = optimizer.state[parameter]
            step = state["step"]
            clock = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
            if clock != self.score_steps:
                raise ProtocolInvalid("Adam clock changed through surgery")
            survivors = torch.tensor(
                receipt.survivor_old_rows,
                dtype=torch.long,
                device=parameter.device,
            )
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = state[moment_name]
                before = old_state[name][moment_name]
                if moment.shape != parameter.shape or not bool(torch.isfinite(moment).all()):
                    raise ProtocolInvalid("post-surgery optimizer moment invalid")
                if not torch.equal(moment[: survivors.numel()], before[survivors]):
                    raise ProtocolInvalid("surviving Adam moments changed")
                if not torch.equal(
                    moment[survivors.numel() :],
                    torch.zeros_like(moment[survivors.numel() :]),
                ):
                    raise ProtocolInvalid("newborn Adam moments are not exact zero")
            if not bool(torch.isfinite(parameter).all()):
                raise ProtocolInvalid("post-surgery parameter is non-finite")

    def after_step(
        self,
        *,
        step: int,
        params: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        snapshot: Gaussians3D,
    ) -> dict[str, torch.Tensor]:
        self._params = params
        self._optimizers = optimizers
        if step == self.score_steps:
            self._selection = self._finalize_selection(params, optimizers)
            for counts in self._selection["operator_counts"].values():
                if counts != {
                    "clone": self.expected_clone_count,
                    "split": self.expected_split_count,
                }:
                    raise ProtocolInvalid(
                        "selection operator quota differs from controller contract"
                    )
            if (
                self.expected_selection_sha256 is not None
                and self._selection["semantic_sha256"] != self.expected_selection_sha256
            ):
                raise ProtocolInvalid("Phase-B selection replay differs from Phase A")
            actual_pre = self._selection["pre_surgery_state"]["semantic_sha256"]
            if (
                self.expected_pre_state_sha256 is not None
                and actual_pre != self.expected_pre_state_sha256
            ):
                raise ProtocolInvalid("Phase-B step-35 state replay differs from Phase A")
            self._state_checkpoints["35_pre"] = self._selection["pre_surgery_state"]
            self._capture_raw_state("35_pre", params, optimizers)
            if self.on_pre_surgery is not None:
                self.on_pre_surgery(snapshot.detach(), self._selection)
            if self.arm is None:
                return params
            assert self.split_root is not None
            selected_rows = self._selection["selected_rows"][self.arm]
            old_groups = {
                name: _group_fields(optimizers[name].param_groups[0]) for name in GROUP_ORDER
            }
            old_state = {
                name: {
                    key: value.detach().clone()
                    for key, value in optimizers[name].state[params[name]].items()
                    if key in {"exp_avg", "exp_avg_sq"}
                }
                for name in GROUP_ORDER
            }
            derived = split_seed(self.split_root)
            generator = torch.Generator(device=params["means"].device)
            generator.manual_seed(derived)
            new_params, receipt = apply_selected_birth_surgery(
                params,
                optimizers,
                selected_rows,
                scene_extent=float(self._extent),
                generator=generator,
                split_scale_frac=0.01,
                split_factor=1.6,
                revised_opacity=True,
                max_gaussians=(
                    self.expected_n_before + self.expected_clone_count + self.expected_split_count
                ),
            )
            self._validate_surgery(
                old_state=old_state,
                old_groups=old_groups,
                new_params=new_params,
                optimizers=optimizers,
                receipt=receipt,
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
                    "family": _lineage_family(dataclasses.asdict(item)),
                    "physical_row": item.new_row,
                }
                for birth_id, item in zip(newborn_ids.cpu().tolist(), receipt.newborns, strict=True)
            ]
            self._params = new_params
            post_state = self._state_record(new_params, optimizers)
            self._state_checkpoints["35_post"] = post_state
            self._capture_raw_state("35_post", new_params, optimizers)
            self._surgery = {
                "arm": self.arm,
                "split_root": self.split_root,
                "split_seed": derived,
                "receipt": dataclasses.asdict(receipt),
                "persistent_ids_after": self._ids.cpu().tolist(),
                "lineage": self._lineage,
                "optimizer_state_after": post_state["optimizers"],
                "state_sha256": post_state["semantic_sha256"],
                "accounting": {
                    "n_before": receipt.n_before,
                    "removed_split_parents": len(receipt.split_parent_rows),
                    "survivors": len(receipt.survivor_old_rows),
                    "clone_children": len(receipt.clone_new_rows),
                    "split_child_0": len(receipt.split_child0_new_rows),
                    "split_child_1": len(receipt.split_child1_new_rows),
                    "newborn_rows": len(receipt.newborns),
                    "net_growth": receipt.net_growth,
                    "n_after": receipt.n_after,
                    "pruned": 0,
                },
            }
            if self.on_post_surgery is not None:
                self.on_post_surgery(_params_to_gaussians(new_params).detach(), self._surgery)
            return new_params
        if step in (70, 105, 140):
            self._state_checkpoints[str(step)] = self._state_record(params, optimizers)
            self._capture_raw_state(str(step), params, optimizers)
        return params

    def _variable_cardinality_summary(self) -> dict[str, Any] | None:
        if self._params is None or self._ids is None:
            return None
        current_ids = self._ids
        original_mask = current_ids < self.expected_n_before
        original_ids = current_ids[original_mask]
        removed_ids = sorted(
            set(range(self.expected_n_before))
            - set(int(value) for value in original_ids.cpu().tolist())
        )
        survivor_rows = original_mask.nonzero(as_tuple=True)[0]
        motion: dict[str, Any] = {}
        for name in GROUP_ORDER:
            initial_rows = self._initial_params[name][original_ids]
            delta = self._params[name].detach()[survivor_rows] - initial_rows
            motion[name] = _summary(delta)
        newborn: dict[str, Any] = {}
        for family in ("clone", "split_child_0", "split_child_1"):
            rows = [int(item["physical_row"]) for item in self._lineage if item["family"] == family]
            index = torch.tensor(rows, dtype=torch.long, device=current_ids.device)
            newborn[family] = {
                name: _summary(self._params[name].detach()[index]) for name in GROUP_ORDER
            }
        return {
            "surviving_original_ids": original_ids.cpu().tolist(),
            "removed_original_ids": removed_ids,
            "survivor_motion": motion,
            "newborn_by_lineage": newborn,
        }

    def history_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": "rtgs.compact_responsibility_birth_iter3.controller.v1",
            "arm": self.arm,
            "score_steps": self.score_steps,
            "shuffle_root": self.shuffle_root,
            # Deliberately do not derive or serialize a split seed for Phase A.
            "split_root": self.split_root,
            "selection": self._selection,
            "surgery": self._surgery,
            "persistent_ids": (None if self._ids is None else self._ids.detach().cpu().tolist()),
            "lineage": self._lineage,
            "step_evidence": self._step_evidence,
            "state_checkpoints": self._state_checkpoints,
            "variable_cardinality": self._variable_cardinality_summary(),
        }
        record["semantic_sha256"] = canonical_hash(record)
        return record


BANK_TENSOR_NAMES = (
    "xy",
    "active",
    "inside_fit_window",
    "proposal_component_ids",
    "proposal_density",
    "joint_density",
    "target_density",
    "importance",
    "color",
)


def _sample_arrays(samples: ObservationSamples, color: torch.Tensor) -> dict[str, np.ndarray]:
    values = {
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
    if tuple(values) != BANK_TENSOR_NAMES:
        raise ProtocolInvalid("evaluation-bank tensor schema changed")
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
    arrays = {
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
    if tuple(arrays) != BANK_TENSOR_NAMES:
        raise ProtocolInvalid("evaluation-bank tensor names changed")
    expected_dtypes = {
        "xy": torch.float32,
        "active": torch.bool,
        "inside_fit_window": torch.bool,
        "proposal_component_ids": torch.int64,
        "proposal_density": torch.float32,
        "joint_density": torch.float32,
        "target_density": torch.float32,
        "importance": torch.float32,
        "color": torch.float32,
    }
    for name, shape in expected.items():
        if tuple(arrays[name].shape) != shape or arrays[name].dtype != expected_dtypes[name]:
            raise ProtocolInvalid(f"evaluation-bank {name} shape/dtype changed")
    for name in (
        "xy",
        "proposal_density",
        "joint_density",
        "target_density",
        "importance",
        "color",
    ):
        if not bool(torch.isfinite(arrays[name]).all()):
            raise ProtocolInvalid(f"evaluation-bank {name} is non-finite")
    for name in (
        "proposal_density",
        "joint_density",
        "target_density",
        "importance",
    ):
        if bool((arrays[name] < 0).any()):
            raise ProtocolInvalid(f"evaluation-bank {name} is negative")
    if measure == "uniform":
        x, y, width, height = product.fit_window
        direct = (
            bool(samples.active.all())
            and bool(samples.inside_fit_window.all())
            and bool((samples.proposal_component_ids == -1).all())
            and bool((samples.xy[:, 0] >= x).all())
            and bool((samples.xy[:, 0] < x + width).all())
            and bool((samples.xy[:, 1] >= y).all())
            and bool((samples.xy[:, 1] < y + height).all())
        )
        if not direct:
            raise ProtocolInvalid("uniform bank is not direct, active, and half-open")
    elif measure == "proposal":
        inactive = ~samples.active
        if bool((samples.active & ~samples.inside_fit_window).any()):
            raise ProtocolInvalid("active proposal-bank row is outside the fit window")
        for name in ("joint_density", "target_density", "importance"):
            if bool((arrays[name][inactive] != 0).any()):
                raise ProtocolInvalid(f"inactive proposal-bank {name} is nonzero")
    else:
        raise ValueError("unknown evaluation measure")


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    failed = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        failed = False
    finally:
        if failed:
            path.unlink(missing_ok=True)


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    data = canonical_bytes(payload) + b"\n"
    _exclusive_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def _fsync_parent(path: Path) -> None:
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _durable_exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish a lifecycle claim without removing a partial file after failure."""
    data = canonical_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_parent(path)
    return hashlib.sha256(data).hexdigest()


def _strict_json_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise ProtocolInvalid(f"required JSON artifact is missing: {path}") from error

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed_items: dict[str, Any] = {}
        for key, value in items:
            if key in parsed_items:
                raise ProtocolInvalid(f"{path} has duplicate JSON key {key!r}")
            parsed_items[key] = value
        return parsed_items

    parsed = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(parsed, dict):
        raise ProtocolInvalid(f"{path} is not a JSON object")
    if canonical_bytes(parsed) + b"\n" != raw:
        raise ProtocolInvalid(f"{path} is not canonical JSON")
    return parsed, hashlib.sha256(raw).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    return _strict_json_with_sha256(path)[0]


def _load_hash_frozen_json(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load a legacy pretty-printed prerequisite after exact byte-hash binding."""
    if sha256_file(path) != expected_sha256:
        raise ProtocolInvalid(f"hash-frozen JSON prerequisite changed: {path}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in items:
            if key in parsed:
                raise ProtocolInvalid(f"hash-frozen JSON has duplicate key {key!r}")
            parsed[key] = value
        return parsed

    parsed = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(parsed, dict):
        raise ProtocolInvalid("hash-frozen JSON prerequisite is not an object")
    return parsed


def _strict_metadata_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in items:
            if key in parsed:
                raise ProtocolInvalid(f"{label} has duplicate JSON key {key!r}")
            parsed[key] = value
        return parsed

    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise ProtocolInvalid(f"{label} metadata is not canonical JSON")
    return value


def _strict_reread_attempt(
    path: Path,
    *,
    artifact_type: str,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Strictly re-read, schema-check and optionally byte-bind an attempt marker."""
    marker, digest = _strict_json_with_sha256(path)
    if marker.get("artifact_type") != artifact_type:
        raise ProtocolInvalid("attempt marker artifact type changed")
    if expected is not None and marker != dict(expected):
        raise ProtocolInvalid("attempt marker differs from the just-published payload")
    if digest != hashlib.sha256(canonical_bytes(marker) + b"\n").hexdigest():
        raise ProtocolInvalid("attempt marker digest changed during strict reread")
    return marker, digest


def _authorize_marker(
    phase: Literal["phase-a", "phase-b"],
    path: Path,
    *,
    artifact_type: str,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    marker, digest = _strict_reread_attempt(path, artifact_type=artifact_type, expected=expected)
    sealed = verify_seal()
    common = {
        "artifact_type",
        "timestamp_utc",
        "seal_sha256",
        "seal_attempt_sha256",
        "bindings_sha256",
    }
    required = (
        common
        if phase == "phase-a"
        else common
        | {
            "phase_a_attempt_sha256",
            "phase_a_result_sha256",
            "phase_a_audit_sha256",
        }
    )
    if (
        set(marker) != required
        or not _is_rfc3339_utc(marker.get("timestamp_utc"))
        or marker.get("seal_sha256") != sha256_file(SEAL)
        or marker.get("seal_attempt_sha256") != sealed["seal_attempt"]["sha256"]
        or marker.get("bindings_sha256") != sealed["bindings_sha256"]
    ):
        raise ProtocolInvalid(f"{phase} marker seal binding changed")
    if phase == "phase-b" and (
        marker.get("phase_a_attempt_sha256") != sha256_file(PHASE_A_ATTEMPT)
        or marker.get("phase_a_result_sha256") != sha256_file(PHASE_A_RESULT)
        or marker.get("phase_a_audit_sha256") != sha256_file(PHASE_A_AUDIT)
    ):
        raise ProtocolInvalid("phase-b marker Phase-A binding changed")
    _AUTHORIZED_PHASES.add(phase)
    return marker, digest


def _exclusive_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    failed = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        failed = False
    finally:
        if failed:
            path.unlink(missing_ok=True)
    return sha256_file(path)


def save_state_archive(path: Path, controller: ResponsibilityBirthController) -> dict[str, Any]:
    arrays, metadata = controller.state_archive_payload()
    payload = dict(arrays)
    payload["metadata_utf8"] = np.frombuffer(canonical_bytes(metadata), dtype=np.uint8)
    digest = _exclusive_npz(path, payload)
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "metadata": metadata,
    }


def load_state_archive(
    path: Path,
    *,
    expected_labels: Sequence[str] | None = None,
    expected_clocks: Mapping[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Strictly load raw parameters/moments and recompute every bound digest."""
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ProtocolInvalid("state archive has duplicate ZIP members")
        if "metadata_utf8" not in archive.files:
            raise ProtocolInvalid("state archive lacks metadata")
        metadata = _strict_metadata_json(
            np.asarray(archive["metadata_utf8"], dtype=np.uint8).tobytes(),
            label="state archive",
        )
        digest = dict(metadata)
        stored = digest.pop("semantic_sha256", None)
        if stored != canonical_hash(digest):
            raise ProtocolInvalid("state archive metadata digest changed")
        if (
            metadata.get("schema") != "rtgs.compact_responsibility_birth_iter3.states.v1"
            or not isinstance(metadata.get("labels"), list)
            or not isinstance(metadata.get("states"), dict)
            or len(metadata["labels"]) != len(set(metadata["labels"]))
            or set(metadata["labels"]) != set(metadata["states"])
        ):
            raise ProtocolInvalid("state archive metadata schema changed")
        if expected_labels is not None and metadata["labels"] != list(expected_labels):
            raise ProtocolInvalid("state archive label order changed")
        if expected_clocks is not None and set(expected_clocks) != set(metadata["labels"]):
            raise ProtocolInvalid("expected state-clock labels changed")
        expected_keys = {"metadata_utf8"}
        arrays: dict[str, np.ndarray] = {}
        for label in metadata["labels"]:
            state = metadata["states"][label]
            state_digest = dict(state)
            state_stored = state_digest.pop("semantic_sha256", None)
            if state_stored != canonical_hash(state_digest):
                raise ProtocolInvalid("state entry metadata digest changed")
            if (
                state.get("label") != label
                or state.get("group_order") != list(GROUP_ORDER)
                or set(state.get("groups", {})) != set(GROUP_ORDER)
            ):
                raise ProtocolInvalid("state entry group schema changed")
            descriptors = [state["persistent_ids"]]
            for name in GROUP_ORDER:
                group = state["groups"][name]
                if not isinstance(group.get("step"), int):
                    raise ProtocolInvalid("state optimizer clock schema changed")
                if expected_clocks is not None and group["step"] != expected_clocks[label]:
                    raise ProtocolInvalid("state optimizer clock changed")
                tensors = group.get("tensors", {})
                exact_tensor_names = (
                    {"parameter"} if group["step"] == 0 else {"parameter", "exp_avg", "exp_avg_sq"}
                )
                if set(tensors) != exact_tensor_names:
                    raise ProtocolInvalid("state tensor-name schema changed")
                descriptors.extend(tensors.values())
            for descriptor in descriptors:
                key = descriptor["key"]
                expected_keys.add(key)
                if key not in archive.files:
                    raise ProtocolInvalid(f"state tensor is missing: {key}")
                value = np.asarray(archive[key]).copy()
                actual = {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": array_hash(value),
                }
                expected = {
                    "dtype": descriptor["dtype"],
                    "shape": descriptor["shape"],
                    "sha256": descriptor["sha256"],
                }
                if actual != expected:
                    raise ProtocolInvalid(f"state tensor changed: {key}")
                arrays[key] = value
        if set(archive.files) != expected_keys:
            raise ProtocolInvalid("state archive has unexpected arrays")
    return arrays, metadata


def generate_evaluation_bank(
    *,
    evaluation_root: int,
    teachers: ReconstructionInputs,
    product_fields: Sequence[GaussianObservationField],
    path: Path,
    focused: bool = False,
    attempts: int = BANK_ATTEMPTS,
) -> dict[str, Any]:
    """Materialize one complete fixed bank with exact draw/state evidence."""
    allowed = FOCUSED_EVALUATION_ROOTS if focused else EVALUATION_ROOTS
    if evaluation_root not in allowed:
        raise ValueError("evaluation root is outside the selected frozen domain")
    _reject_roots_for_mode(evaluation_root, focused=focused)
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if len(teachers.observations) != len(product_fields) or len(teachers.view_names) != len(
        product_fields
    ):
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
        banks: dict[str, Any] = {}
        for measure, uniform_fraction in (("uniform", 1.0), ("proposal", 0.25)):
            derived = evaluation_bank_seed(evaluation_root, view_name, measure)
            generator = torch.Generator(device=product.device)
            generator.manual_seed(derived)
            state_before = generator_state_hash(generator)
            samples = proposal.sample(
                attempts,
                uniform_fraction=uniform_fraction,
                generator=generator,
            )
            state_after = generator_state_hash(generator)
            color = teacher.query(samples.xy, component_chunk=640).color
            _validate_bank(
                samples=samples,
                color=color,
                product=product,
                measure=measure,
                attempts=attempts,
            )
            values = _sample_arrays(samples, color)
            descriptors: dict[str, Any] = {}
            for name in BANK_TENSOR_NAMES:
                value = values[name]
                key = f"v{view_index}_{measure}_{name}"
                arrays[key] = value
                descriptors[name] = {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": array_hash(value),
                }
            draw_sha = canonical_hash(
                {name: descriptors[name]["sha256"] for name in BANK_TENSOR_NAMES}
            )
            active_count = int(values["active"].sum())
            banks[measure] = {
                "seed_domain": EVALUATION_SEED_DOMAIN,
                "root": evaluation_root,
                "derived_seed": derived,
                "view_name": view_name,
                "measure": measure,
                "attempts": attempts,
                "generator_state_before_sha256": state_before,
                "generator_state_after_sha256": state_after,
                "draw_sha256": draw_sha,
                "active_count": active_count,
                "null_count": attempts - active_count,
                "active_fraction": active_count / attempts,
                "tensor_names": list(BANK_TENSOR_NAMES),
                "tensors": descriptors,
            }
        views.append(
            {
                "view_index": view_index,
                "view_name": view_name,
                "m_opt_2d": teacher.n,
                "banks": banks,
            }
        )
    metadata: dict[str, Any] = {
        "schema": "rtgs.compact_responsibility_birth_iter3.banks.v1",
        "seed_domain": EVALUATION_SEED_DOMAIN,
        "evaluation_root": evaluation_root,
        "attempts_per_bank": attempts,
        "tensor_names": list(BANK_TENSOR_NAMES),
        "views": views,
    }
    metadata["semantic_sha256"] = canonical_hash(metadata)
    arrays["metadata_utf8"] = np.frombuffer(canonical_bytes(metadata), dtype=np.uint8)
    file_sha = _exclusive_npz(path, arrays)
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
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
    product_fields: Sequence[GaussianObservationField] | None = None,
) -> tuple[list[dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    _require_root_authorized(expected_root)
    if product_fields is not None and len(product_fields) != len(expected_views):
        raise ValueError("product-field/evaluation-view counts differ")
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ProtocolInvalid("evaluation archive has duplicate ZIP members")
        if "metadata_utf8" not in archive.files:
            raise ProtocolInvalid("evaluation archive lacks metadata")
        metadata = _strict_metadata_json(
            np.asarray(archive["metadata_utf8"], dtype=np.uint8).tobytes(),
            label="evaluation archive",
        )
        digest = dict(metadata)
        stored = digest.pop("semantic_sha256", None)
        if stored != canonical_hash(digest):
            raise ProtocolInvalid("evaluation-bank metadata digest changed")
        if (
            metadata.get("schema") != "rtgs.compact_responsibility_birth_iter3.banks.v1"
            or metadata.get("seed_domain") != EVALUATION_SEED_DOMAIN
            or metadata.get("evaluation_root") != expected_root
            or metadata.get("attempts_per_bank") != attempts
            or metadata.get("tensor_names") != list(BANK_TENSOR_NAMES)
        ):
            raise ProtocolInvalid("evaluation-bank metadata differs from contract")
        expected_keys = {"metadata_utf8"}
        loaded: list[dict[str, dict[str, np.ndarray]]] = []
        view_records = metadata.get("views")
        if not isinstance(view_records, list) or len(view_records) != len(expected_views):
            raise ProtocolInvalid("evaluation-bank view count changed")
        for view_index, (view_name, record) in enumerate(
            zip(expected_views, view_records, strict=True)
        ):
            if record.get("view_index") != view_index or record.get("view_name") != view_name:
                raise ProtocolInvalid("evaluation-bank view order changed")
            measures: dict[str, dict[str, np.ndarray]] = {}
            if set(record.get("banks", {})) != {"uniform", "proposal"}:
                raise ProtocolInvalid("evaluation-bank measure schema changed")
            for measure in ("uniform", "proposal"):
                measure_record = record["banks"][measure]
                derived_seed = evaluation_bank_seed(expected_root, view_name, measure)
                initial_generator = torch.Generator(device="cpu")
                initial_generator.manual_seed(derived_seed)
                if (
                    measure_record.get("seed_domain") != EVALUATION_SEED_DOMAIN
                    or measure_record.get("root") != expected_root
                    or measure_record.get("derived_seed") != derived_seed
                    or measure_record.get("view_name") != view_name
                    or measure_record.get("measure") != measure
                    or measure_record.get("attempts") != attempts
                    or measure_record.get("tensor_names") != list(BANK_TENSOR_NAMES)
                    or set(measure_record.get("tensors", {})) != set(BANK_TENSOR_NAMES)
                    or measure_record.get("generator_state_before_sha256")
                    != generator_state_hash(initial_generator)
                    or not _is_lower_sha256(measure_record.get("generator_state_after_sha256"))
                ):
                    raise ProtocolInvalid("evaluation-bank seed/tensor binding changed")
                values: dict[str, np.ndarray] = {}
                for name in BANK_TENSOR_NAMES:
                    descriptor = measure_record["tensors"][name]
                    key = f"v{view_index}_{measure}_{name}"
                    expected_keys.add(key)
                    if key not in archive.files:
                        raise ProtocolInvalid(f"bank tensor is missing: {key}")
                    value = np.asarray(archive[key]).copy()
                    actual = {
                        "dtype": value.dtype.str,
                        "shape": list(value.shape),
                        "sha256": array_hash(value),
                    }
                    if actual != descriptor:
                        raise ProtocolInvalid(f"bank tensor changed: {key}")
                    values[name] = value
                draw_sha = canonical_hash(
                    {name: measure_record["tensors"][name]["sha256"] for name in BANK_TENSOR_NAMES}
                )
                if draw_sha != measure_record.get("draw_sha256"):
                    raise ProtocolInvalid("evaluation-bank draw hash changed")
                active = values["active"]
                if active.dtype != np.dtype(np.bool_) or tuple(active.shape) != (attempts,):
                    raise ProtocolInvalid("evaluation-bank active tensor schema changed")
                active_count = int(active.sum())
                null_count = attempts - active_count
                active_fraction = active_count / attempts
                if (
                    measure_record.get("active_count") != active_count
                    or measure_record.get("null_count") != null_count
                    or measure_record.get("active_fraction") != active_fraction
                    or (
                        measure == "uniform"
                        and (active_count != attempts or null_count != 0 or active_fraction != 1.0)
                    )
                ):
                    raise ProtocolInvalid("evaluation-bank active/null count or fraction changed")
                if product_fields is not None:
                    samples = ObservationSamples(
                        xy=torch.from_numpy(values["xy"]),
                        proposal_component_ids=torch.from_numpy(values["proposal_component_ids"]),
                        proposal_density=torch.from_numpy(values["proposal_density"]),
                        joint_density=torch.from_numpy(values["joint_density"]),
                        target_density=torch.from_numpy(values["target_density"]),
                        importance=torch.from_numpy(values["importance"]),
                        inside_fit_window=torch.from_numpy(values["inside_fit_window"]),
                        active=torch.from_numpy(values["active"]),
                        risk_measure="continuous_area",
                    )
                    _validate_bank(
                        samples=samples,
                        color=torch.from_numpy(values["color"]),
                        product=product_fields[view_index],
                        measure=measure,
                        attempts=attempts,
                    )
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
    """Evaluate frozen J_U/J_Q banks using detached float64 accumulation."""
    target_device = torch.device(device)
    model = snapshot.to(target_device)
    renderer = TorchPointRasterizer(point_chunk=point_chunk, gaussian_chunk=gaussian_chunk)
    background = torch.zeros(3, device=target_device, dtype=model.means.dtype)
    per_view = []
    with torch.no_grad():
        for view_index, (name, camera, view_banks) in enumerate(
            zip(inputs.view_names, inputs.cameras, banks, strict=True)
        ):
            risks: dict[str, float] = {}
            diagnostics: dict[str, Any] = {}
            for measure, risk_name in (("uniform", "J_U"), ("proposal", "J_Q")):
                bank = view_banks[measure]
                xy = torch.from_numpy(bank["xy"]).to(target_device)
                target = torch.from_numpy(bank["color"]).to(target_device)
                active = torch.from_numpy(bank["active"]).to(device=target_device, dtype=torch.bool)
                rendered = renderer.render_points(
                    model,
                    camera.to(target_device),
                    xy,
                    background=background,
                    sh_degree=0,
                )
                prediction = rendered.color
                rendered_alpha = rendered.alpha.detach().to(torch.float64)
                if not bool(
                    torch.isfinite(prediction).all() and torch.isfinite(rendered_alpha).all()
                ):
                    raise ProtocolInvalid("evaluation render is non-finite")
                point_loss = (
                    (prediction.detach().to(torch.float64) - target.detach().to(torch.float64))
                    .square()
                    .mean(dim=-1)
                )
                weighted = (
                    point_loss if measure == "uniform" else point_loss * active.to(torch.float64)
                )
                attempts = int(xy.shape[0])
                loss_sum = weighted.sum(dtype=torch.float64)
                risks[risk_name] = float(loss_sum / attempts)
                diagnostics[measure] = {
                    "attempts": attempts,
                    "active_count": int(active.sum()),
                    "loss_sum_float64": float(loss_sum),
                    "alpha_sum_float64": float(rendered_alpha.sum(dtype=torch.float64)),
                    "alpha_mean_float64": float(rendered_alpha.mean()),
                    "alpha_max_float64": float(rendered_alpha.max()),
                    "active_alpha_sum_float64": float(
                        (rendered_alpha * active.to(dtype=torch.float64)).sum(dtype=torch.float64)
                    ),
                    "authorizing": False,
                }
            per_view.append(
                {
                    "view_index": view_index,
                    "view_name": name,
                    **risks,
                    "banks": diagnostics,
                }
            )
    return {
        "J_U": math.fsum(item["J_U"] for item in per_view) / len(per_view),
        "J_Q": math.fsum(item["J_Q"] for item in per_view) / len(per_view),
        "worst_view_J_U": max(item["J_U"] for item in per_view),
        "worst_view_J_Q": max(item["J_Q"] for item in per_view),
        "per_view": per_view,
    }


def _validate_checkpoint_metric_recomputation(
    loaded_snapshots: Mapping[str, Gaussians3D],
    stored_metrics: Mapping[str, Any],
    inputs: ReconstructionInputs,
    banks: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
    *,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Recompute every decision-bearing checkpoint metric, caching common states."""
    if set(loaded_snapshots) != set(stored_metrics):
        raise ProtocolInvalid("checkpoint snapshot/metric labels differ")
    metric_cache = {} if cache is None else cache
    receipts: dict[str, str] = {}
    for label, snapshot in loaded_snapshots.items():
        semantic = factorial.gaussians_hash(snapshot)
        if semantic not in metric_cache:
            metric_cache[semantic] = evaluate_snapshot(snapshot, inputs, banks)
        recomputed = metric_cache[semantic]
        if recomputed != stored_metrics[label]:
            raise ProtocolInvalid("Phase-B checkpoint metric recomputation changed")
        receipts[label] = canonical_hash(recomputed)
    return receipts


def recovery_log_auc(risks: Sequence[float]) -> float:
    if len(risks) != 4:
        raise ValueError("recovery curve needs 35_post,70,105,140")
    if any(not math.isfinite(value) or value <= 0 for value in risks):
        raise ValueError("recovery risks must be finite and positive")
    x = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    logs = [math.log(max(value, 1e-12)) for value in risks]
    return math.fsum(
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
    """Apply the frozen ordered and exhaustive terminal decision map."""
    comparisons: dict[str, Any] = {}
    for comparator in ("G", "U"):
        final_q_ratios: list[float] = []
        auc_q_ratios: list[float] = []
        final_u_ratios: list[float] = []
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


def _validate_final_ply_roundtrip(
    source: Gaussians3D,
    replay: Gaussians3D,
    *,
    expected_count: int = 867,
) -> dict[str, Any]:
    if replay.n != expected_count:
        raise ProtocolInvalid(f"final PLY count differs from {expected_count}")
    fields: dict[str, Any] = {}
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        loaded = getattr(replay, name).detach().cpu().to(dtype=torch.float32)
        expected = getattr(source, name).detach().cpu().to(dtype=torch.float32)
        if loaded.shape != expected.shape or not bool(torch.isfinite(loaded).all()):
            raise ProtocolInvalid(f"final PLY field {name} is invalid")
        absolute = (loaded - expected).abs()
        tolerance = 1e-6 + 1e-6 * expected.abs()
        normalized_excess = ((absolute - tolerance) / tolerance).clamp_min(0)
        if bool((absolute > tolerance).any()):
            raise ProtocolInvalid(f"final PLY field {name} exceeds tolerance")
        fields[name] = {
            "max_abs_error": float(absolute.max()) if absolute.numel() else 0.0,
            "max_normalized_excess": (
                float(normalized_excess.max()) if normalized_excess.numel() else 0.0
            ),
        }
    return fields


def _save_final_ply(
    path: Path,
    snapshot: Gaussians3D,
    *,
    expected_count: int = 867,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_ply(path)
    replay = Gaussians3D.load_ply(path)
    fields = _validate_final_ply_roundtrip(snapshot, replay, expected_count=expected_count)
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "n_gaussians": replay.n,
        "source_semantic_sha256": factorial.gaussians_hash(snapshot),
        "roundtrip": fields,
    }


def _review_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    verdicts = [line for line in lines if line.startswith("Verdict:")]
    unresolved = [line for line in lines if line.startswith("Unresolved findings:")]
    return verdicts == ["Verdict: PASS"] and unresolved == ["Unresolved findings: none"]


def _verify_frozen_lifecycle_documents() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_LIFECYCLE_SHA256.items():
        if not path.is_file() or path.is_symlink():
            raise ProtocolInvalid(f"frozen lifecycle document is missing or linked: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ProtocolInvalid(f"frozen lifecycle document digest changed: {path}")
        observed[path.relative_to(ROOT).as_posix()] = actual
    if not _review_passed(PREREGISTRATION_REVIEW):
        raise ProtocolInvalid("iter3 preregistration addendum lacks its independent PASS")
    initial_lines = PREREGISTRATION_INITIAL_FAIL_REVIEW.read_text(encoding="utf-8").splitlines()
    if [line for line in initial_lines if line.startswith("Verdict:")] != ["Verdict: FAIL"] or [
        line for line in initial_lines if line.startswith("Unresolved findings:")
    ] != ["Unresolved findings: 2"]:
        raise ProtocolInvalid("iter3 initial FAIL review changed disposition")
    return observed


def _source_paths(*, include_implementation_review: bool = True) -> tuple[Path, ...]:
    """Return the explicit stable transitive closure used by official execution."""
    _verify_frozen_lifecycle_documents()
    paths: set[Path] = {
        Path(__file__).resolve(),
        ROOT / "benchmarks/compact_occupancy_refinement_factorial.py",
        ROOT / "src/rtgs/__init__.py",
        ROOT / "src/rtgs/core/__init__.py",
        ROOT / "src/rtgs/core/camera.py",
        ROOT / "src/rtgs/core/gaussians2d.py",
        ROOT / "src/rtgs/core/gaussians3d.py",
        ROOT / "src/rtgs/core/metrics.py",
        ROOT / "src/rtgs/core/observation2d.py",
        ROOT / "src/rtgs/core/sh.py",
        ROOT / "src/rtgs/data/__init__.py",
        ROOT / "src/rtgs/data/reconstruction_inputs.py",
        ROOT / "src/rtgs/data/scene.py",
        ROOT / "src/rtgs/data/synthetic.py",
        ROOT / "src/rtgs/image2gs/__init__.py",
        ROOT / "src/rtgs/image2gs/adapters.py",
        ROOT / "src/rtgs/image2gs/fit.py",
        ROOT / "src/rtgs/image2gs/renderer2d.py",
        ROOT / "src/rtgs/optim/__init__.py",
        ROOT / "src/rtgs/optim/compact_trainer.py",
        ROOT / "src/rtgs/optim/density.py",
        ROOT / "src/rtgs/optim/strategies.py",
        ROOT / "src/rtgs/optim/trainer.py",
        ROOT / "src/rtgs/render/__init__.py",
        ROOT / "src/rtgs/render/base.py",
        ROOT / "src/rtgs/render/point_base.py",
        ROOT / "src/rtgs/render/torch_points.py",
        ROOT / "src/rtgs/render/torch_ref.py",
        ROOT / "tests/conftest.py",
        ROOT / "tests/test_compact_responsibility_birth_allocation_iter3.py",
        ROOT / "tests/test_point_render.py",
        ROOT / "tests/test_compact_trainer.py",
        ROOT / "tests/test_optim.py",
        ROOT / "scripts/verify.sh",
        ROOT / "scripts/docs_sync.py",
        ROOT / "pyproject.toml",
        ROOT / "CLAUDE.md",
        PREREGISTRATION,
        PREREGISTRATION_ADDENDUM,
        PREREGISTRATION_INITIAL_FAIL_REVIEW,
        PREREGISTRATION_REVIEW,
        IMPORTED_PREREGISTRATION,
        IMPORTED_PREMATURE_REVIEW,
        IMPORTED_INITIAL_FAIL_REVIEW,
        IMPORTED_ADDENDUM_REVIEW,
        FAILURE_AUDIT,
        ITER2_FAILURE_AUDIT,
        ITER2_FAILURE_AUDIT_ADDENDUM,
        ITER2_PREREGISTRATION,
        ITER2_PREREGISTRATION_REVIEW,
        ITER2_IMPLEMENTATION_REVIEW,
    }
    if include_implementation_review:
        paths.add(IMPLEMENTATION_REVIEW)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise ProtocolInvalid(f"source closure has missing files: {missing}")
    return tuple(sorted(paths))


def _source_hashes() -> tuple[dict[str, str], str]:
    origin_violations = module_origin_violations()
    if origin_violations:
        raise ProtocolInvalid(f"loaded rtgs module origin mismatch: {origin_violations}")
    unbound = unbound_loaded_local_sources()
    if unbound:
        raise ProtocolInvalid(f"loaded local source closure is not sealed: {unbound}")
    records = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in _source_paths()}
    return records, canonical_hash(records)


def reviewed_source_hashes() -> tuple[dict[str, str], str]:
    """Hash the outcome-free implementation snapshot without self-reference."""
    records = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in _source_paths(include_implementation_review=False)
    }
    return records, canonical_hash(records)


def implementation_review_passed(path: Path = IMPLEMENTATION_REVIEW) -> bool:
    if not _review_passed(path):
        return False
    prefix = "Reviewed source aggregate SHA-256: "
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)
    ]
    if len(lines) != 1:
        return False
    claimed = lines[0][len(prefix) :]
    if len(claimed) != 64 or any(character not in "0123456789abcdef" for character in claimed):
        return False
    return claimed == reviewed_source_hashes()[1]


def loaded_local_sources() -> tuple[str, ...]:
    """Return every live repository-owned Python source in this process."""
    repository = ROOT.resolve()
    paths: set[str] = set()
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if not value:
            continue
        path = Path(value).resolve()
        if path.suffix in {".pyc", ".pyo"}:
            source = path.with_suffix(".py")
            if source.is_file():
                path = source
        if not path.is_file() or path.is_symlink() or path.suffix != ".py":
            continue
        if repository == path or repository in path.parents:
            paths.add(path.relative_to(repository).as_posix())
    return tuple(sorted(paths))


def unbound_loaded_local_sources() -> tuple[str, ...]:
    bound = {
        path.relative_to(ROOT).as_posix()
        for path in _source_paths(include_implementation_review=False)
    }
    return tuple(path for path in loaded_local_sources() if path not in bound)


def expected_rtgs_module_origins() -> dict[str, str]:
    origins: dict[str, str] = {}
    for path in _source_paths(include_implementation_review=False):
        relative = path.relative_to(ROOT)
        parts = relative.parts
        if len(parts) < 3 or parts[:2] != ("src", "rtgs") or relative.suffix != ".py":
            continue
        module_parts = list(parts[1:])
        if module_parts[-1] == "__init__.py":
            module_parts.pop()
        else:
            module_parts[-1] = Path(module_parts[-1]).stem
        origins[".".join(module_parts)] = str(path.resolve())
    return origins


def loaded_rtgs_module_origins() -> dict[str, str | None]:
    origins: dict[str, str | None] = {}
    for name, module in tuple(sys.modules.items()):
        if name != "rtgs" and not name.startswith("rtgs."):
            continue
        value = getattr(module, "__file__", None)
        origins[name] = None if value is None else str(Path(value).resolve())
    return dict(sorted(origins.items()))


def module_origin_violations() -> tuple[str, ...]:
    expected = expected_rtgs_module_origins()
    actual = loaded_rtgs_module_origins()
    violations = []
    for name, origin in actual.items():
        if name not in expected:
            violations.append(f"unbound:{name}={origin}")
        elif origin != expected[name]:
            violations.append(f"shadowed:{name}={origin};expected={expected[name]}")
    expected_harness = str(
        (ROOT / "benchmarks/compact_responsibility_birth_allocation_iter3.py").resolve()
    )
    if str(Path(__file__).resolve()) != expected_harness:
        violations.append(
            f"shadowed:harness={Path(__file__).resolve()};expected={expected_harness}"
        )
    return tuple(violations)


def runtime_binding() -> dict[str, Any]:
    """Bind the iter3 ABI receipt while validating the iter3 source closure."""
    expected_preload_sha = factorial.EXPECTED_PRELOAD_SHA256
    if not PRELOAD.is_file() or sha256_file(PRELOAD) != expected_preload_sha:
        raise ProtocolInvalid("system libstdc++ binding changed")
    effective_preload = os.environ.get("LD_PRELOAD")
    if effective_preload != str(PRELOAD):
        raise ProtocolInvalid("effective LD_PRELOAD differs from the frozen runtime")
    if not torch.cuda.is_available():
        raise ProtocolInvalid("official iter3 experiment requires CUDA")
    import_path = factorial._torch_runtime_import_path_binding()
    origin_violations = module_origin_violations()
    if origin_violations:
        raise ProtocolInvalid(f"loaded rtgs module origin mismatch: {origin_violations}")
    unbound = unbound_loaded_local_sources()
    if unbound:
        raise ProtocolInvalid(f"loaded local source closure is not sealed: {unbound}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 3050" or capability != (8, 6):
        raise ProtocolInvalid("official CUDA device binding changed")
    properties = torch.cuda.get_device_properties(0)
    try:
        gsplat_version = importlib.metadata.version("gsplat")
    except importlib.metadata.PackageNotFoundError:
        gsplat_version = None
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,pci.bus_id,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if driver.returncode != 0 or len(driver.stdout.strip().splitlines()) != 1:
        raise ProtocolInvalid("cannot bind NVIDIA driver/device identity")
    record = {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "gsplat": gsplat_version,
        "cuda_device": name,
        "cuda_capability": list(capability),
        "cuda_total_memory": int(properties.total_memory),
        "cuda_multiprocessor_count": int(properties.multi_processor_count),
        "cuda_uuid": str(properties.uuid),
        "cuda_pci_bus_id": int(properties.pci_bus_id),
        "nvidia_smi_driver_device": driver.stdout.strip(),
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "module_origins": expected_rtgs_module_origins(),
        "loaded_local_sources": list(loaded_local_sources()),
        "unbound_loaded_local_sources": list(unbound),
        "sys_path": import_path["normalized_sys_path"],
        "torch_generated_import_path": {
            key: value for key, value in import_path.items() if key != "normalized_sys_path"
        },
        "pythonpath": os.environ.get("PYTHONPATH"),
        "preload": str(PRELOAD),
        "preload_sha256": expected_preload_sha,
        "effective_ld_preload": effective_preload,
    }
    frozen_result = _load_hash_frozen_json(
        RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json",
        expected_sha256=EXPECTED_PREREQUISITE_SHA256[
            "benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json"
        ],
    )
    frozen_runtime = frozen_result.get("runtime")
    if (
        not isinstance(frozen_runtime, dict)
        or canonical_hash(frozen_runtime) != EXPECTED_FROZEN_RUNTIME_SHA256
    ):
        raise ProtocolInvalid("frozen iter3 runtime receipt changed")
    current_projection = {key: record[key] for key in FROZEN_RUNTIME_PROJECTION_FIELDS}
    frozen_projection = {key: frozen_runtime[key] for key in FROZEN_RUNTIME_PROJECTION_FIELDS}
    if current_projection != frozen_projection:
        raise ProtocolInvalid("current ABI/path runtime differs from frozen iter3 runtime")
    return record


def _static_root_use_proof() -> dict[str, Any]:
    """Allow fresh official integer literals only in four root declarations."""
    dangerous = {
        "manual_seed",
        "Generator",
        "build_view_schedule",
        "step_sample_seed",
        "domain_seed",
        "evaluation_bank_seed",
        "split_seed",
        "shuffle_seed",
    }
    findings: list[dict[str, Any]] = []
    occurrence_findings: list[dict[str, Any]] = []
    checked = []
    for path in _source_paths(include_implementation_review=False):
        if path.suffix != ".py":
            continue
        checked.append(path.relative_to(ROOT).as_posix())
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        allowed_constant_ids: set[int] = set()
        if path.resolve() == Path(__file__).resolve():
            for statement in tree.body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id
                    in {
                        "TRAIN_ROOTS",
                        "EVALUATION_ROOTS",
                        "SPLIT_ROOTS",
                        "SHUFFLE_ROOTS",
                    }
                ):
                    allowed_constant_ids.update(
                        id(node)
                        for node in ast.walk(statement.value)
                        if isinstance(node, ast.Constant)
                        and type(node.value) is int
                        and node.value in OFFICIAL_ROOTS
                    )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and type(node.value) is int
                and node.value in OFFICIAL_ROOTS
                and id(node) not in allowed_constant_ids
            ):
                occurrence_findings.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "line": node.lineno,
                        "root": node.value,
                    }
                )
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            if function_name not in dangerous:
                continue
            for argument in (*node.args, *(item.value for item in node.keywords)):
                if (
                    isinstance(argument, ast.Constant)
                    and type(argument.value) is int
                    and argument.value in OFFICIAL_ROOTS
                ):
                    findings.append(
                        {
                            "path": path.relative_to(ROOT).as_posix(),
                            "line": node.lineno,
                            "function": function_name,
                            "root": argument.value,
                        }
                    )
    if findings or occurrence_findings:
        raise ProtocolInvalid(
            "official root escaped its declaration or entered a pre-marker call: "
            f"occurrences={occurrence_findings}, calls={findings}"
        )
    return {
        "checked_python_files": checked,
        "dangerous_call_names": sorted(dangerous),
        "direct_official_literal_calls": findings,
        "official_literal_occurrences_outside_root_declarations": occurrence_findings,
        "allowed_declarations": [
            "TRAIN_ROOTS",
            "EVALUATION_ROOTS",
            "SPLIT_ROOTS",
            "SHUFFLE_ROOTS",
        ],
        "passed": True,
    }


def _dynamic_pre_marker_root_proof() -> dict[str, Any]:
    if _AUTHORIZED_PHASES:
        raise ProtocolInvalid("seal process already authorized an official phase")
    from rtgs.optim import compact_trainer as compact_trainer_module

    calls = {
        "generator": 0,
        "schedule": 0,
        "sampler": 0,
        "trainer": 0,
        "surgery": 0,
    }
    original_generator = torch.Generator
    original_schedule = compact_trainer_module.build_view_schedule
    original_sample = GaussianPointProposal.sample
    original_train = CompactTrainer.train
    original_surgery = globals()["apply_selected_birth_surgery"]

    def generator_spy(*args: Any, **kwargs: Any) -> Any:
        calls["generator"] += 1
        return original_generator(*args, **kwargs)

    def schedule_spy(*args: Any, **kwargs: Any) -> Any:
        calls["schedule"] += 1
        return original_schedule(*args, **kwargs)

    def sample_spy(*args: Any, **kwargs: Any) -> Any:
        calls["sampler"] += 1
        return original_sample(*args, **kwargs)

    def train_spy(*args: Any, **kwargs: Any) -> Any:
        calls["trainer"] += 1
        return original_train(*args, **kwargs)

    def surgery_spy(*args: Any, **kwargs: Any) -> Any:
        calls["surgery"] += 1
        return original_surgery(*args, **kwargs)

    gateways: list[tuple[str, Callable[[], Any]]] = [
        ("training_config", lambda: frozen_config(TRAIN_ROOTS[0])),
        (
            "training_domain",
            lambda: domain_seed("pre_marker_probe", TRAIN_ROOTS[0]),
        ),
        (
            "evaluation_seed",
            lambda: evaluation_bank_seed(EVALUATION_ROOTS[0], "probe", "uniform"),
        ),
        ("split_seed", lambda: split_seed(SPLIT_ROOTS[0])),
        (
            "shuffle_seed",
            lambda: shuffle_seed(SHUFFLE_ROOTS[0], STRATA[0]),
        ),
        (
            "selection",
            lambda: build_matched_selections(
                gradient_score=torch.empty(0),
                residual_score=torch.empty(0),
                support_score=torch.empty(0),
                support_by_view=torch.empty(0, 0),
                visible_step_count=torch.empty(0),
                scale_max=torch.empty(0),
                persistent_ids=torch.empty(0, dtype=torch.long),
                extent=1.0,
                shuffle_root=SHUFFLE_ROOTS[0],
            ),
        ),
        (
            "evaluation_bank",
            lambda: generate_evaluation_bank(
                evaluation_root=EVALUATION_ROOTS[0],
                teachers=None,  # type: ignore[arg-type]
                product_fields=(),
                path=ROOT / "never-created.npz",
            ),
        ),
    ]
    rejected: list[str] = []
    torch.Generator = generator_spy  # type: ignore[assignment,misc]
    compact_trainer_module.build_view_schedule = schedule_spy
    GaussianPointProposal.sample = sample_spy  # type: ignore[method-assign]
    CompactTrainer.train = train_spy  # type: ignore[method-assign]
    globals()["apply_selected_birth_surgery"] = surgery_spy
    try:
        for name, gateway in gateways:
            try:
                gateway()
            except ProtocolInvalid:
                rejected.append(name)
            else:
                raise ProtocolInvalid(f"pre-marker gateway {name} did not reject")
    finally:
        torch.Generator = original_generator  # type: ignore[assignment,misc]
        compact_trainer_module.build_view_schedule = original_schedule
        GaussianPointProposal.sample = original_sample  # type: ignore[method-assign]
        CompactTrainer.train = original_train  # type: ignore[method-assign]
        globals()["apply_selected_birth_surgery"] = original_surgery
    passed = len(rejected) == len(gateways) and not any(calls.values())
    if not passed:
        raise ProtocolInvalid(f"official root reached a pre-marker mechanism: calls={calls}")
    return {
        "gateways": [name for name, _ in gateways],
        "rejected_before_mechanism": rejected,
        "mechanism_spy_calls": calls,
        "no_generator_or_schedule_constructed": (
            calls["generator"] == 0 and calls["schedule"] == 0
        ),
        "passed": passed,
    }


def _config_record() -> dict[str, Any]:
    return {
        "schema": "rtgs.compact_responsibility_birth_iter3.config.v1",
        "training_roots": list(TRAIN_ROOTS),
        "evaluation_roots": list(EVALUATION_ROOTS),
        "split_roots": list(SPLIT_ROOTS),
        "shuffle_roots": list(SHUFFLE_ROOTS),
        "focused_training_roots": list(FOCUSED_TRAIN_ROOTS),
        "focused_evaluation_roots": list(FOCUSED_EVALUATION_ROOTS),
        "focused_split_roots": list(FOCUSED_SPLIT_ROOTS),
        "focused_shuffle_roots": list(FOCUSED_SHUFFLE_ROOTS),
        "first_failed_roots": sorted(FIRST_FAILED_ROOTS),
        "iter2_retired_roots": sorted(ITER2_RETIRED_ROOTS),
        "contaminated_candidate_roots": sorted(CONTAMINATED_CANDIDATE_ROOTS),
        "root_sets_pairwise_disjoint": len(OFFICIAL_ROOTS | FOCUSED_ROOTS | FAILED_ROOTS)
        == len(OFFICIAL_ROOTS) + len(FOCUSED_ROOTS) + len(FAILED_ROOTS),
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
            "order": list(GROUP_ORDER),
            "algorithm": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-15,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "fused": False,
        },
    }


def _git_binding() -> dict[str, Any]:
    relative = [path.relative_to(ROOT).as_posix() for path in _source_paths()]

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ProtocolInvalid(
                f"git {' '.join(arguments)} failed: "
                + completed.stderr.decode(errors="replace")[-2000:]
            )
        return completed.stdout

    head = git("rev-parse", "HEAD").decode().strip()
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *relative,
    )
    unstaged = git("diff", "--binary", "--", *relative)
    staged = git("diff", "--cached", "--binary", "--", *relative)
    tracked_output = git("ls-files", "--", *relative).decode().splitlines()
    tracked = set(tracked_output)
    return {
        "head": head,
        "scoped_paths": relative,
        "tracked_paths": sorted(tracked),
        "untracked_paths": sorted(set(relative) - tracked),
        "status_porcelain_v1": status.decode(errors="strict").splitlines(),
        "unstaged_diff_bytes": len(unstaged),
        "unstaged_diff_sha256": hashlib.sha256(unstaged).hexdigest(),
        "staged_diff_bytes": len(staged),
        "staged_diff_sha256": hashlib.sha256(staged).hexdigest(),
    }


def _binding_state() -> dict[str, Any]:
    sources, aggregate = _source_hashes()
    lifecycle_documents = _verify_frozen_lifecycle_documents()
    prerequisites: dict[str, str] = {}
    for relative, expected in EXPECTED_PREREQUISITE_SHA256.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ProtocolInvalid(f"frozen prerequisite digest changed: {relative}")
        prerequisites[relative] = actual
    frozen_result = _load_hash_frozen_json(
        RESULTS / "20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json",
        expected_sha256=EXPECTED_PREREQUISITE_SHA256[
            "benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json"
        ],
    )
    frozen_inputs = frozen_result.get("inputs")
    if (
        not isinstance(frozen_inputs, dict)
        or canonical_hash(frozen_inputs) != EXPECTED_FROZEN_INPUT_BINDINGS_SHA256
    ):
        raise ProtocolInvalid("frozen iter3 input-binding record changed")
    current_inputs = factorial.input_bindings()
    if current_inputs != frozen_inputs:
        raise ProtocolInvalid("current inputs differ from frozen iter3 input bindings")
    return {
        "source_hashes": sources,
        "source_aggregate_sha256": aggregate,
        "inputs": current_inputs,
        "runtime": runtime_binding(),
        "git": _git_binding(),
        "config": _config_record(),
        "post_result_visualizer": {
            "path": VISUALIZER.relative_to(ROOT).as_posix(),
            "execution_role": "post-result-only; excluded from decision execution closure",
            "existence_does_not_change_seal": True,
        },
        "lifecycle_documents": lifecycle_documents,
        "prerequisite_artifacts": prerequisites,
    }


def _run_command(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
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
    stdout = completed.stdout.encode("utf-8")
    stderr = completed.stderr.encode("utf-8")
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_tail": completed.stdout[-8000:],
        "stderr_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_tail": completed.stderr[-8000:],
        "elapsed_seconds": time.perf_counter() - started,
    }


def _bounded_utf8_tail(data: bytes) -> tuple[int, str]:
    tail = data[-SEAL_TRANSCRIPT_TAIL_BYTES:]
    text = tail.decode("utf-8", errors="replace")
    while len(text.encode("utf-8")) > SEAL_TRANSCRIPT_TAIL_BYTES:
        tail = tail[1:]
        text = tail.decode("utf-8", errors="replace")
    return len(tail), text


def _stream_receipt(data: bytes) -> dict[str, Any]:
    tail = data[-SEAL_TRANSCRIPT_TAIL_BYTES:]
    return {
        "byte_count": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "tail_source_byte_count": len(tail),
        "tail_utf8": tail.decode("utf-8", errors="replace"),
    }


def _exception_receipt(error: BaseException) -> dict[str, str]:
    _, bounded_message = _bounded_utf8_tail(str(error).encode("utf-8"))
    return {"type": type(error).__name__, "message": bounded_message}


def _run_verification_command(
    ordinal: int,
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int | None = WORKER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Return a bounded but complete-hash transcript even for spawn/timeout failure."""
    started_at = factorial.timestamp_utc()
    stdout = b""
    stderr = b""
    returncode: int | None = None
    timed_out = False
    exception: dict[str, str] | None = None
    try:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env=None if env is None else dict(env),
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout if isinstance(error.stdout, bytes) else (error.stdout or "").encode()
        stderr = error.stderr if isinstance(error.stderr, bytes) else (error.stderr or "").encode()
        exception = _exception_receipt(error)
    except Exception as error:
        exception = _exception_receipt(error)
    return {
        "ordinal": ordinal,
        "command": list(command),
        "cwd": str(ROOT),
        "started_at_utc": started_at,
        "finished_at_utc": factorial.timestamp_utc(),
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "returncode": returncode,
        "exception": exception,
        "stdout": _stream_receipt(stdout),
        "stderr": _stream_receipt(stderr),
    }


def _is_rfc3339_utc(value: object) -> bool:
    if not isinstance(value, str) or not (value.endswith("+00:00") or value.endswith("Z")):
        return False
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0)


def _validate_stream_receipt(record: object) -> None:
    if not isinstance(record, dict) or set(record) != {
        "byte_count",
        "sha256",
        "tail_source_byte_count",
        "tail_utf8",
    }:
        raise ProtocolInvalid("verification stream receipt schema changed")
    byte_count = record["byte_count"]
    tail_count = record["tail_source_byte_count"]
    digest = record["sha256"]
    tail_utf8 = record["tail_utf8"]
    if (
        type(byte_count) is not int
        or byte_count < 0
        or type(tail_count) is not int
        or tail_count != min(byte_count, SEAL_TRANSCRIPT_TAIL_BYTES)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(tail_utf8, str)
    ):
        raise ProtocolInvalid("verification stream receipt value changed")


def _validate_exception_receipt(record: object) -> None:
    if (
        not isinstance(record, dict)
        or set(record) != {"type", "message"}
        or not isinstance(record["type"], str)
        or not record["type"]
        or not isinstance(record["message"], str)
        or len(record["message"].encode("utf-8")) > SEAL_TRANSCRIPT_TAIL_BYTES
    ):
        raise ProtocolInvalid("verification exception receipt changed")


def _seal_verification_commands() -> tuple[tuple[str, ...], ...]:
    return (
        (
            str(ROOT / ".venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            "tests/test_compact_responsibility_birth_allocation_iter3.py",
            "tests/test_point_render.py",
            "tests/test_compact_trainer.py",
            "tests/test_optim.py",
        ),
        (str(ROOT / "scripts/verify.sh"),),
        ("git", "diff", "--check"),
    )


def _validate_verification_record(
    record: object,
    *,
    ordinal: int,
    command: Sequence[str],
) -> None:
    if not isinstance(record, dict) or set(record) != {
        "ordinal",
        "command",
        "cwd",
        "started_at_utc",
        "finished_at_utc",
        "timeout_seconds",
        "timed_out",
        "returncode",
        "exception",
        "stdout",
        "stderr",
    }:
        raise ProtocolInvalid("seal verification transcript schema changed")
    timeout_seconds = record["timeout_seconds"]
    returncode = record["returncode"]
    exception = record["exception"]
    if (
        record["ordinal"] != ordinal
        or record["command"] != list(command)
        or record["cwd"] != str(ROOT)
        or not _is_rfc3339_utc(record["started_at_utc"])
        or not _is_rfc3339_utc(record["finished_at_utc"])
        or timeout_seconds != WORKER_TIMEOUT_SECONDS
        or type(record["timed_out"]) is not bool
        or (
            returncode is not None
            and (type(returncode) is not int or not -(2**31) <= returncode < 2**31)
        )
    ):
        raise ProtocolInvalid("seal verification transcript value changed")
    if exception is not None:
        _validate_exception_receipt(exception)
    _validate_stream_receipt(record["stdout"])
    _validate_stream_receipt(record["stderr"])
    started = dt.datetime.fromisoformat(str(record["started_at_utc"]).replace("Z", "+00:00"))
    finished = dt.datetime.fromisoformat(str(record["finished_at_utc"]).replace("Z", "+00:00"))
    if finished < started:
        raise ProtocolInvalid("seal verification transcript time moved backwards")
    empty_stream = {
        "byte_count": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "tail_source_byte_count": 0,
        "tail_utf8": "",
    }
    if record["timed_out"]:
        if returncode is not None or exception is None:
            raise ProtocolInvalid("seal timeout transcript combination changed")
    elif exception is not None:
        if (
            returncode is not None
            or record["stdout"] != empty_stream
            or record["stderr"] != empty_stream
        ):
            raise ProtocolInvalid("seal spawn-exception transcript combination changed")
    elif returncode is None:
        raise ProtocolInvalid("seal completed transcript lacks a return code")


def _validate_verification_transcript(
    records: object,
    *,
    require_complete_success: bool,
) -> None:
    if not isinstance(records, list):
        raise ProtocolInvalid("seal verification transcript is not a list")
    commands = _seal_verification_commands()
    if len(records) > len(commands) or (require_complete_success and len(records) != len(commands)):
        raise ProtocolInvalid("seal verification transcript item count changed")
    for ordinal, (record, command) in enumerate(zip(records, commands, strict=False), start=1):
        _validate_verification_record(record, ordinal=ordinal, command=command)
        if require_complete_success and not _verification_record_passed(record):
            raise ProtocolInvalid("seal verification transcript contains a failed item")


def _verification_record_passed(record: Mapping[str, Any]) -> bool:
    return (
        record.get("timed_out") is False
        and record.get("returncode") == 0
        and record.get("exception") is None
    )


def _relative_or_absolute(path: Path) -> str:
    absolute = path.absolute()
    repository = ROOT.absolute()
    return (
        absolute.relative_to(repository).as_posix()
        if absolute.is_relative_to(repository)
        else str(absolute)
    )


def _seal_output_paths() -> dict[str, Path]:
    return {
        "seal_attempt": SEAL_ATTEMPT,
        "seal_failure": SEAL_FAILURE,
        "seal": SEAL,
        "executed_sources": EXECUTED_SOURCES,
        "phase_a_attempt": PHASE_A_ATTEMPT,
        "phase_a_result": PHASE_A_RESULT,
        "phase_a_audit": PHASE_A_AUDIT,
        "phase_b_attempt": PHASE_B_ATTEMPT,
        "result": RESULT,
        "run_directory": RUN_DIR,
        "visualizer": VISUALIZER,
        "failure_audit": ITER3_FAILURE_AUDIT,
    }


def _seal_namespace_paths() -> dict[str, Path]:
    return {
        "harness": Path(__file__).absolute(),
        "focused_test": ROOT / "tests/test_compact_responsibility_birth_allocation_iter3.py",
        "preregistration": PREREGISTRATION,
        "preregistration_addendum": PREREGISTRATION_ADDENDUM,
        "initial_preregistration_review": PREREGISTRATION_INITIAL_FAIL_REVIEW,
        "passing_preregistration_review": PREREGISTRATION_REVIEW,
        "implementation_review": IMPLEMENTATION_REVIEW,
        **_seal_output_paths(),
    }


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ProtocolInvalid(f"cannot inspect frozen lifecycle path: {path}") from error
    return True


def _require_pristine_seal_entry_namespace() -> None:
    present = [name for name, path in _seal_output_paths().items() if _path_present(path)]
    if present:
        raise ProtocolInvalid(f"iter3 seal-entry namespace is not pristine: {present}")


def _artifact_inventory() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for path in _seal_output_paths().values():
        label = _relative_or_absolute(path)
        record: dict[str, Any] = {"state": "absent", "sha256": None, "error": None}
        try:
            path.lstat()
        except FileNotFoundError:
            entries[label] = record
            continue
        except Exception as error:
            record["state"] = "inspection_error"
            record["error"] = _exception_receipt(error)
            entries[label] = record
            continue
        try:
            if path.is_symlink():
                record["state"] = "other"
            elif path.is_file() and not path.is_dir():
                record["state"] = "regular_file"
                try:
                    record["sha256"] = sha256_file(path)
                except Exception as error:
                    record["error"] = _exception_receipt(error)
            elif path.is_dir():
                record["state"] = "directory"
            else:
                record["state"] = "other"
        except Exception as error:
            record["state"] = "inspection_error"
            record["error"] = _exception_receipt(error)
        entries[label] = record
    return {
        "captured_at_utc": factorial.timestamp_utc(),
        "capture_boundary": "immediately before exclusive SEAL_FAILURE publication",
        "entries": entries,
    }


def _attempt_protocol_record() -> dict[str, Any]:
    return {
        "preregistration_path": _relative_or_absolute(PREREGISTRATION),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "preregistration_addendum_path": _relative_or_absolute(PREREGISTRATION_ADDENDUM),
        "preregistration_addendum_sha256": sha256_file(PREREGISTRATION_ADDENDUM),
        "passing_preregistration_review_path": _relative_or_absolute(PREREGISTRATION_REVIEW),
        "passing_preregistration_review_sha256": sha256_file(PREREGISTRATION_REVIEW),
        "implementation_review_path": _relative_or_absolute(IMPLEMENTATION_REVIEW),
        "implementation_review_sha256": sha256_file(IMPLEMENTATION_REVIEW),
    }


def _validate_current_attempt_protocol(attempt: Mapping[str, Any]) -> None:
    if attempt.get("protocol") != _attempt_protocol_record():
        raise ProtocolInvalid("current protocol documents differ from the seal attempt")


def _seal_attempt_payload(argv: Sequence[str] | None = None) -> dict[str, Any]:
    preload_resolved = PRELOAD.resolve()
    roots = {
        "official_training": list(TRAIN_ROOTS),
        "official_evaluation": list(EVALUATION_ROOTS),
        "official_split": list(SPLIT_ROOTS),
        "official_shuffle": list(SHUFFLE_ROOTS),
        "focused_training": list(FOCUSED_TRAIN_ROOTS),
        "focused_evaluation": list(FOCUSED_EVALUATION_ROOTS),
        "focused_split": list(FOCUSED_SPLIT_ROOTS),
        "focused_shuffle": list(FOCUSED_SHUFFLE_ROOTS),
    }
    return {
        "artifact_type": "compact_responsibility_birth_iter3_seal_attempt_v1",
        "status": "CLAIMED",
        "scientific_decision": "UNAVAILABLE",
        "timestamp_utc": factorial.timestamp_utc(),
        "command": {
            "argv": list(sys.argv if argv is None else argv),
            "literal": SEAL_LITERAL_COMMAND,
            "cwd": str(Path.cwd().absolute()),
        },
        "process": {
            "executable": str(Path(sys.executable).absolute()),
            "pid": os.getpid(),
        },
        "environment": {
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
            "preload_path": str(preload_resolved),
            "preload_sha256": sha256_file(preload_resolved),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "focused_test_environment_key_at_entry": os.environ.get(FOCUSED_TEST_ENV),
        },
        "protocol": _attempt_protocol_record(),
        "namespace": {
            "paths": {
                name: _relative_or_absolute(path) for name, path in _seal_namespace_paths().items()
            },
            "roots": roots,
        },
    }


def _validate_seal_attempt(payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "artifact_type",
        "status",
        "scientific_decision",
        "timestamp_utc",
        "command",
        "process",
        "environment",
        "protocol",
        "namespace",
    }:
        raise ProtocolInvalid("seal-attempt schema changed")
    if (
        payload.get("artifact_type") != "compact_responsibility_birth_iter3_seal_attempt_v1"
        or payload.get("status") != "CLAIMED"
        or payload.get("scientific_decision") != "UNAVAILABLE"
        or not _is_rfc3339_utc(payload.get("timestamp_utc"))
    ):
        raise ProtocolInvalid("seal-attempt type/status/timestamp changed")
    command = payload.get("command")
    process = payload.get("process")
    environment = payload.get("environment")
    protocol = payload.get("protocol")
    if (
        not isinstance(command, dict)
        or set(command) != {"argv", "literal", "cwd"}
        or not isinstance(command["argv"], list)
        or not all(isinstance(item, str) for item in command["argv"])
        or command["literal"] != SEAL_LITERAL_COMMAND
        or not isinstance(command["cwd"], str)
        or not Path(command["cwd"]).is_absolute()
        or not isinstance(process, dict)
        or set(process) != {"executable", "pid"}
        or not isinstance(process["executable"], str)
        or not Path(process["executable"]).is_absolute()
        or type(process["pid"]) is not int
        or process["pid"] <= 0
        or not isinstance(environment, dict)
        or set(environment)
        != {
            "LD_PRELOAD",
            "preload_path",
            "preload_sha256",
            "PYTHONPATH",
            "focused_test_environment_key_at_entry",
        }
        or environment["preload_path"] != str(PRELOAD.resolve())
        or not isinstance(environment["preload_sha256"], str)
        or len(environment["preload_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in environment["preload_sha256"])
        or not (environment["LD_PRELOAD"] is None or isinstance(environment["LD_PRELOAD"], str))
        or not (environment["PYTHONPATH"] is None or isinstance(environment["PYTHONPATH"], str))
        or not (
            environment["focused_test_environment_key_at_entry"] is None
            or isinstance(environment["focused_test_environment_key_at_entry"], str)
        )
        or not isinstance(protocol, dict)
        or set(protocol)
        != {
            "preregistration_path",
            "preregistration_sha256",
            "preregistration_addendum_path",
            "preregistration_addendum_sha256",
            "passing_preregistration_review_path",
            "passing_preregistration_review_sha256",
            "implementation_review_path",
            "implementation_review_sha256",
        }
    ):
        raise ProtocolInvalid("seal-attempt command/process/environment changed")
    expected_protocol_paths = {
        "preregistration_path": _relative_or_absolute(PREREGISTRATION),
        "preregistration_addendum_path": _relative_or_absolute(PREREGISTRATION_ADDENDUM),
        "passing_preregistration_review_path": _relative_or_absolute(PREREGISTRATION_REVIEW),
        "implementation_review_path": _relative_or_absolute(IMPLEMENTATION_REVIEW),
    }
    if any(protocol[key] != value for key, value in expected_protocol_paths.items()):
        raise ProtocolInvalid("seal-attempt protocol binding changed")
    expected_protocol_hashes = {
        "preregistration_sha256": EXPECTED_LIFECYCLE_SHA256[PREREGISTRATION],
        "preregistration_addendum_sha256": EXPECTED_LIFECYCLE_SHA256[PREREGISTRATION_ADDENDUM],
        "passing_preregistration_review_sha256": EXPECTED_LIFECYCLE_SHA256[PREREGISTRATION_REVIEW],
    }
    if any(protocol[key] != value for key, value in expected_protocol_hashes.items()):
        raise ProtocolInvalid("seal-attempt frozen protocol digest changed")
    implementation_digest = protocol["implementation_review_sha256"]
    if (
        not isinstance(implementation_digest, str)
        or len(implementation_digest) != 64
        or any(character not in "0123456789abcdef" for character in implementation_digest)
    ):
        raise ProtocolInvalid("seal-attempt implementation-review digest changed")
    expected_paths = {
        name: _relative_or_absolute(path) for name, path in _seal_namespace_paths().items()
    }
    expected_roots = {
        "official_training": list(TRAIN_ROOTS),
        "official_evaluation": list(EVALUATION_ROOTS),
        "official_split": list(SPLIT_ROOTS),
        "official_shuffle": list(SHUFFLE_ROOTS),
        "focused_training": list(FOCUSED_TRAIN_ROOTS),
        "focused_evaluation": list(FOCUSED_EVALUATION_ROOTS),
        "focused_split": list(FOCUSED_SPLIT_ROOTS),
        "focused_shuffle": list(FOCUSED_SHUFFLE_ROOTS),
    }
    if payload.get("namespace") != {"paths": expected_paths, "roots": expected_roots}:
        raise ProtocolInvalid("seal-attempt namespace/root declaration changed")


def _read_seal_attempt(
    *,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    if SEAL_ATTEMPT.is_symlink() or not SEAL_ATTEMPT.is_file():
        raise ProtocolInvalid("seal-attempt marker is missing, linked, or not regular")
    payload, digest = _strict_json_with_sha256(SEAL_ATTEMPT)
    _validate_seal_attempt(payload)
    if expected is not None and payload != dict(expected):
        raise ProtocolInvalid("seal-attempt marker changed on immediate reread")
    return payload, digest


def _reserve_seal_attempt() -> int:
    """Durably consume the attempt path before any payload materialization."""
    descriptor = os.open(
        SEAL_ATTEMPT,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        os.fsync(descriptor)
        _fsync_parent(SEAL_ATTEMPT)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _complete_reserved_seal_attempt(
    descriptor: int,
    payload: Mapping[str, Any],
) -> str:
    """Fully write and sync a previously reserved attempt without cleanup."""
    data = canonical_bytes(payload) + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if type(count) is not int or count <= 0:
            raise OSError("seal-attempt write made no forward progress")
        written += count
    if written != len(data):
        raise OSError("seal-attempt write was incomplete")
    os.fsync(descriptor)
    _fsync_parent(SEAL_ATTEMPT)
    return digest


def _claim_seal_attempt(argv: Sequence[str] | None = None) -> tuple[dict[str, Any], str]:
    _require_pristine_seal_entry_namespace()
    descriptor = _reserve_seal_attempt()
    try:
        payload = _seal_attempt_payload(argv)
        claimed_sha = _complete_reserved_seal_attempt(descriptor, payload)
    finally:
        os.close(descriptor)
    reread, reread_sha = _read_seal_attempt(expected=payload)
    if reread_sha != claimed_sha:
        raise ProtocolInvalid("seal-attempt digest changed on immediate reread")
    return reread, reread_sha


def _require_post_claim_namespace(attempt_sha256: str) -> None:
    _, actual_sha = _read_seal_attempt()
    if actual_sha != attempt_sha256:
        raise ProtocolInvalid("claimed seal-attempt digest changed")
    present = [
        name
        for name, path in _seal_output_paths().items()
        if name != "seal_attempt" and _path_present(path)
    ]
    if present:
        raise ProtocolInvalid(f"iter3 post-claim namespace is not pristine: {present}")


def _validate_inventory_entry(record: object) -> None:
    if not isinstance(record, dict) or set(record) != {"state", "sha256", "error"}:
        raise ProtocolInvalid("seal-failure inventory-entry schema changed")
    state = record["state"]
    digest = record["sha256"]
    error = record["error"]
    if state not in {
        "absent",
        "regular_file",
        "directory",
        "other",
        "inspection_error",
    }:
        raise ProtocolInvalid("seal-failure inventory state changed")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ProtocolInvalid("seal-failure inventory digest changed")
    if digest is not None and state != "regular_file":
        raise ProtocolInvalid("seal-failure inventory hashes a non-regular file")
    if error is not None:
        _validate_exception_receipt(error)
    if state == "absent" and (digest is not None or error is not None):
        raise ProtocolInvalid("seal-failure absent inventory entry has evidence")


def _validate_seal_failure(payload: Mapping[str, Any], *, attempt_sha256: str) -> None:
    if set(payload) != {
        "artifact_type",
        "status",
        "scientific_decision",
        "timestamp_utc",
        "command",
        "seal_attempt",
        "protocol",
        "exception",
        "failure_stage",
        "verification",
        "binding",
        "artifact_inventory",
    }:
        raise ProtocolInvalid("seal-failure schema changed")
    if (
        payload.get("artifact_type") != "compact_responsibility_birth_iter3_seal_failure_v1"
        or payload.get("status") != "FAIL"
        or payload.get("scientific_decision") != "UNAVAILABLE"
        or not _is_rfc3339_utc(payload.get("timestamp_utc"))
        or payload.get("seal_attempt")
        != {"path": _relative_or_absolute(SEAL_ATTEMPT), "sha256": attempt_sha256}
        or payload.get("failure_stage") not in SEAL_FAILURE_STAGES
    ):
        raise ProtocolInvalid("seal-failure status/attempt/transcript changed")
    attempt, actual_attempt_sha = _read_seal_attempt()
    if (
        actual_attempt_sha != attempt_sha256
        or payload.get("command") != attempt["command"]
        or payload.get("protocol") != attempt["protocol"]
    ):
        raise ProtocolInvalid("seal-failure attempt protocol binding changed")
    exception = payload.get("exception")
    if (
        not isinstance(exception, dict)
        or set(exception) != {"type", "message", "bounded_traceback_tail"}
        or not isinstance(exception["bounded_traceback_tail"], str)
        or len(exception["bounded_traceback_tail"].encode("utf-8")) > SEAL_TRANSCRIPT_TAIL_BYTES
    ):
        raise ProtocolInvalid("seal-failure exception schema changed")
    _validate_exception_receipt(
        {"type": exception.get("type"), "message": exception.get("message")}
    )
    _validate_verification_transcript(
        payload.get("verification"),
        require_complete_success=False,
    )
    binding = payload.get("binding")
    if not isinstance(binding, dict) or set(binding) != {
        "available",
        "stage",
        "canonical_sha256",
        "value",
        "unavailable_reason",
    }:
        raise ProtocolInvalid("seal-failure binding schema changed")
    if type(binding["available"]) is not bool:
        raise ProtocolInvalid("seal-failure binding availability changed")
    if binding["available"]:
        if (
            binding["stage"]
            not in {
                "before_verification",
                "after_verification",
                "after_archive",
            }
            or not isinstance(binding["value"], dict)
            or binding["canonical_sha256"] != canonical_hash(binding["value"])
            or binding["unavailable_reason"] is not None
        ):
            raise ProtocolInvalid("seal-failure available binding changed")
    elif (
        binding["stage"] is not None
        or binding["canonical_sha256"] is not None
        or binding["value"] is not None
        or not isinstance(binding["unavailable_reason"], str)
        or not binding["unavailable_reason"]
    ):
        raise ProtocolInvalid("seal-failure unavailable binding changed")
    inventory = payload.get("artifact_inventory")
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"captured_at_utc", "capture_boundary", "entries"}
        or not _is_rfc3339_utc(inventory.get("captured_at_utc"))
        or inventory.get("capture_boundary")
        != "immediately before exclusive SEAL_FAILURE publication"
        or not isinstance(inventory.get("entries"), dict)
    ):
        raise ProtocolInvalid("seal-failure prepublication inventory changed")
    entries = inventory["entries"]
    expected_labels = {_relative_or_absolute(path) for path in _seal_output_paths().values()}
    if set(entries) != expected_labels:
        raise ProtocolInvalid("seal-failure inventory path set changed")
    for record in entries.values():
        _validate_inventory_entry(record)
    failure_label = _relative_or_absolute(SEAL_FAILURE)
    if entries[failure_label] != {"state": "absent", "sha256": None, "error": None}:
        raise ProtocolInvalid("seal-failure was not absent at inventory boundary")


def _publish_seal_failure(
    *,
    attempt: Mapping[str, Any],
    attempt_sha256: str,
    error: BaseException,
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    traceback_bytes = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    ).encode("utf-8")
    _, bounded_traceback = _bounded_utf8_tail(traceback_bytes)
    binding_value = state.get("binding")
    binding_available = isinstance(binding_value, dict)
    stage = state.get("stage")
    if stage not in SEAL_FAILURE_STAGES:
        raise ProtocolInvalid("seal failure reached an unenumerated lifecycle stage")
    binding_error = state.get("binding_error")
    if not binding_available and not isinstance(binding_error, str):
        binding_error = f"no complete binding captured before {stage}"
    payload = {
        "artifact_type": "compact_responsibility_birth_iter3_seal_failure_v1",
        "status": "FAIL",
        "scientific_decision": "UNAVAILABLE",
        "timestamp_utc": factorial.timestamp_utc(),
        "command": attempt["command"],
        "seal_attempt": {
            "path": _relative_or_absolute(SEAL_ATTEMPT),
            "sha256": attempt_sha256,
        },
        "protocol": attempt["protocol"],
        "exception": {
            **_exception_receipt(error),
            "bounded_traceback_tail": bounded_traceback,
        },
        "failure_stage": stage,
        "verification": list(state.get("verification", [])),
        "binding": {
            "available": binding_available,
            "stage": state.get("binding_stage") if binding_available else None,
            "canonical_sha256": canonical_hash(binding_value) if binding_available else None,
            "value": binding_value if binding_available else None,
            "unavailable_reason": None if binding_available else binding_error,
        },
        "artifact_inventory": _artifact_inventory(),
    }
    failure_sha = _durable_exclusive_json(SEAL_FAILURE, payload)
    reread, reread_sha = _strict_json_with_sha256(SEAL_FAILURE)
    _validate_seal_failure(reread, attempt_sha256=attempt_sha256)
    if reread != payload or reread_sha != failure_sha:
        raise ProtocolInvalid("seal-failure receipt changed on immediate reread")
    _, reread_attempt_sha = _read_seal_attempt(expected=attempt)
    if reread_attempt_sha != attempt_sha256:
        raise ProtocolInvalid("seal-attempt changed during failure publication")
    return reread, failure_sha


def _create_source_tar(path: Path, sources: Sequence[Path]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for source in sources:
                relative = source.relative_to(ROOT)
                info = archive.gettarinfo(source, arcname=relative.as_posix())
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                with source.open("rb") as source_stream:
                    archive.addfile(info, source_stream)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_parent(path)
    return sha256_file(path)


def _verify_source_tar(path: Path, expected_hashes: Mapping[str, str]) -> dict[str, Any]:
    observed: dict[str, str] = {}
    with tarfile.open(path, "r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected_hashes):
            raise ProtocolInvalid("executed-source tar member set changed")
        for member in members:
            if not member.isfile():
                raise ProtocolInvalid("executed-source tar has a non-file member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ProtocolInvalid("executed-source tar member cannot be read")
            observed[member.name] = hashlib.sha256(stream.read()).hexdigest()
    if observed != dict(expected_hashes):
        raise ProtocolInvalid("executed-source tar bytes differ from sealed sources")
    return {
        "member_count": len(observed),
        "member_hashes_sha256": canonical_hash(observed),
        "matches_source_hashes": True,
    }


def _official_namespace_absent_for_seal() -> bool:
    return not any(_path_present(path) for path in _seal_output_paths().values())


def _validate_official_seal_entry(attempt: Mapping[str, Any]) -> None:
    if (
        attempt["command"]["cwd"] != str(ROOT)
        or attempt["process"]["executable"] != str((ROOT / ".venv/bin/python").absolute())
        or attempt["environment"]["LD_PRELOAD"] != str(PRELOAD)
        or attempt["environment"]["focused_test_environment_key_at_entry"] is not None
    ):
        raise ProtocolInvalid(
            "official seal entry requires the frozen cwd, executable, preload, "
            "and an absent focused-test key"
        )


def _run_seal_verification_transcript(state: dict[str, Any]) -> None:
    environment = dict(os.environ)
    environment[FOCUSED_TEST_ENV] = "1"
    for ordinal, command in enumerate(_seal_verification_commands(), start=1):
        state["stage"] = f"verification_{ordinal}"
        record = _run_verification_command(
            ordinal,
            command,
            env=environment,
        )
        state["verification"].append(record)
        _validate_verification_record(record, ordinal=ordinal, command=command)


def _create_seal_after_claim(
    attempt: Mapping[str, Any],
    attempt_sha256: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    state["stage"] = "post_claim_entry_validation"
    _validate_official_seal_entry(attempt)
    state["stage"] = "post_claim_namespace_validation"
    _require_post_claim_namespace(attempt_sha256)
    state["stage"] = "lifecycle_document_validation"
    _verify_frozen_lifecycle_documents()
    state["stage"] = "review_validation"
    _validate_current_attempt_protocol(attempt)
    if not _review_passed(PREREGISTRATION_REVIEW):
        raise ProtocolInvalid("iter3 preregistration lacks its independent PASS")
    if not implementation_review_passed():
        raise ProtocolInvalid("iter3 implementation review PASS/source aggregate is invalid")
    state["stage"] = "static_root_proof"
    static_proof = _static_root_use_proof()
    state["stage"] = "dynamic_root_proof"
    dynamic_proof = _dynamic_pre_marker_root_proof()
    state["stage"] = "binding_before_verification"
    before = _binding_state()
    state["binding"] = before
    state["binding_stage"] = "before_verification"
    _run_seal_verification_transcript(state)
    state["stage"] = "verification_gate"
    if any(not _verification_record_passed(item) for item in state["verification"]):
        raise ProtocolInvalid("seal verification failed")
    _validate_verification_transcript(
        state["verification"],
        require_complete_success=True,
    )
    state["stage"] = "binding_after_verification"
    after = _binding_state()
    state["binding"] = after
    state["binding_stage"] = "after_verification"
    if before != after:
        raise ProtocolInvalid("source/input/runtime drifted during seal verification")
    state["stage"] = "archive_creation"
    executed_sha = _create_source_tar(EXECUTED_SOURCES, _source_paths())
    state["stage"] = "archive_validation"
    tar_validation = _verify_source_tar(EXECUTED_SOURCES, after["source_hashes"])
    state["stage"] = "binding_after_archive"
    final_binding = _binding_state()
    state["binding"] = final_binding
    state["binding_stage"] = "after_archive"
    if final_binding != after:
        raise ProtocolInvalid("binding drifted while creating executed-source tar")
    _, current_attempt_sha = _read_seal_attempt(expected=attempt)
    if current_attempt_sha != attempt_sha256:
        raise ProtocolInvalid("seal-attempt changed before seal publication")
    _validate_current_attempt_protocol(attempt)
    if _path_present(SEAL_FAILURE):
        raise ProtocolInvalid("seal-failure path appeared before seal publication")
    payload = {
        "artifact_type": "compact_responsibility_birth_iter3_seal_v1",
        "timestamp_utc": factorial.timestamp_utc(),
        "status": "PASS",
        "bindings": final_binding,
        "bindings_sha256": canonical_hash(final_binding),
        "reviews": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (PREREGISTRATION_REVIEW, IMPLEMENTATION_REVIEW)
        },
        "pre_marker_root_use_proof": {
            "static": static_proof,
            "dynamic": dynamic_proof,
        },
        "verification": list(state["verification"]),
        "executed_sources": {
            "path": EXECUTED_SOURCES.relative_to(ROOT).as_posix(),
            "sha256": executed_sha,
            "validation": tar_validation,
        },
        "seal_attempt": {
            "path": SEAL_ATTEMPT.relative_to(ROOT).as_posix(),
            "sha256": attempt_sha256,
        },
    }
    state["stage"] = "seal_publication"
    seal_sha = _durable_exclusive_json(SEAL, payload)
    reread, reread_sha = _strict_json_with_sha256(SEAL)
    if reread != payload or reread_sha != seal_sha:
        raise ProtocolInvalid("seal changed on immediate post-publication reread")
    state["stage"] = "post_publication_verification"
    return verify_seal()


def _seal_failure_locator(
    *,
    failure_sha256: str,
    attempt_sha256: str,
) -> dict[str, Any]:
    return {
        "status": "FAIL",
        "path": _relative_or_absolute(SEAL_FAILURE),
        "sha256": failure_sha256,
        "attempt_sha256": attempt_sha256,
    }


def create_seal(argv: Sequence[str] | None = None) -> dict[str, Any]:
    attempt, attempt_sha256 = _claim_seal_attempt(argv)
    state: dict[str, Any] = {
        "stage": "post_claim_entry_validation",
        "verification": [],
        "binding": None,
        "binding_stage": None,
        "binding_error": "no complete binding has been captured",
    }
    try:
        return _create_seal_after_claim(attempt, attempt_sha256, state)
    except BaseException as error:
        try:
            _, failure_sha = _publish_seal_failure(
                attempt=attempt,
                attempt_sha256=attempt_sha256,
                error=error,
                state=state,
            )
        except BaseException as publication_error:
            raise publication_error from error
        return _seal_failure_locator(
            failure_sha256=failure_sha,
            attempt_sha256=attempt_sha256,
        )


def verify_seal() -> dict[str, Any]:
    if _path_present(SEAL_FAILURE):
        raise ProtocolInvalid("a seal-failure receipt forbids seal authorization")
    attempt, attempt_sha256 = _read_seal_attempt()
    _validate_official_seal_entry(attempt)
    _validate_current_attempt_protocol(attempt)
    if SEAL.is_symlink() or not SEAL.is_file():
        raise ProtocolInvalid("seal is missing, linked, or not a regular file")
    sealed = strict_json(SEAL)
    if set(sealed) != {
        "artifact_type",
        "timestamp_utc",
        "status",
        "bindings",
        "bindings_sha256",
        "reviews",
        "pre_marker_root_use_proof",
        "verification",
        "executed_sources",
        "seal_attempt",
    } or (
        sealed.get("artifact_type") != "compact_responsibility_birth_iter3_seal_v1"
        or sealed.get("status") != "PASS"
        or not _is_rfc3339_utc(sealed.get("timestamp_utc"))
        or sealed.get("bindings_sha256") != canonical_hash(sealed.get("bindings"))
        or sealed.get("seal_attempt")
        != {
            "path": SEAL_ATTEMPT.relative_to(ROOT).as_posix(),
            "sha256": attempt_sha256,
        }
    ):
        raise ProtocolInvalid("seal schema/status/digest is invalid")
    _, reread_attempt_sha = _read_seal_attempt(expected=attempt)
    if reread_attempt_sha != attempt_sha256:
        raise ProtocolInvalid("seal-attempt changed during seal verification")
    expected_reviews = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in (PREREGISTRATION_REVIEW, IMPLEMENTATION_REVIEW)
    }
    if sealed.get("reviews") != expected_reviews:
        raise ProtocolInvalid("seal review binding changed")
    proofs = sealed.get("pre_marker_root_use_proof")
    if (
        not isinstance(proofs, dict)
        or set(proofs) != {"static", "dynamic"}
        or proofs["static"].get("passed") is not True
        or proofs["dynamic"].get("passed") is not True
    ):
        raise ProtocolInvalid("seal pre-marker root proof changed")
    _validate_verification_transcript(
        sealed.get("verification"),
        require_complete_success=True,
    )
    if sealed["bindings"] != _binding_state():
        raise ProtocolInvalid("current source/input/runtime differs from seal")
    executed = sealed.get("executed_sources")
    if not isinstance(executed, dict) or set(executed) != {
        "path",
        "sha256",
        "validation",
    }:
        raise ProtocolInvalid("executed-source archive schema changed")
    archive_path = ROOT / executed["path"]
    if (
        executed["path"] != EXECUTED_SOURCES.relative_to(ROOT).as_posix()
        or archive_path.is_symlink()
        or not archive_path.is_file()
        or sha256_file(archive_path) != executed["sha256"]
    ):
        raise ProtocolInvalid("executed-source archive changed")
    validation = _verify_source_tar(archive_path, sealed["bindings"]["source_hashes"])
    if executed["validation"] != validation:
        raise ProtocolInvalid("executed-source archive validation changed")
    return sealed


def _verified_binding_receipt() -> dict[str, Any]:
    sealed = verify_seal()
    current = _binding_state()
    if current != sealed["bindings"]:
        raise ProtocolInvalid("current binding differs after seal verification")
    return {
        "seal_sha256": sha256_file(SEAL),
        "seal_attempt_sha256": sealed["seal_attempt"]["sha256"],
        "binding_sha256": canonical_hash(current),
        "source_aggregate_sha256": current["source_aggregate_sha256"],
        "input_binding_sha256": canonical_hash(current["inputs"]),
        "runtime_binding_sha256": canonical_hash(current["runtime"]),
        "git_binding_sha256": canonical_hash(current["git"]),
    }


def _load_frozen_inputs() -> tuple[
    ReconstructionInputs,
    list[GaussianObservationField],
    Gaussians3D,
    list[dict[str, Any]],
]:
    inputs = ReconstructionInputs.load(TEACHER_BUNDLE, strict=True)
    proxies = ReconstructionInputs.load(PROXY_BUNDLE, strict=True)
    products, alignment = factorial.build_product_fields(inputs, proxies)
    initial = Gaussians3D.load_ply(INIT_PLY)
    if (
        tuple(inputs.view_names) != EXPECTED_VIEWS
        or initial.n != 835
        or sha256_file(INIT_PLY) != EXPECTED_INIT_PLY_SHA256
    ):
        raise ProtocolInvalid("frozen compact inputs changed")
    return inputs, products, initial, alignment


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
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "semantic_sha256": semantic,
        "n_gaussians": snapshot.n,
    }


_REPLAY_STEP_FIELDS = (
    "step",
    "view_index",
    "view_name",
    "sample_seed",
    "xy_sha256",
    "active_sha256",
    "inside_fit_window_sha256",
    "proposal_density_sha256",
    "joint_density_sha256",
    "target_density_sha256",
    "importance_sha256",
    "proposal_component_ids_sha256",
    "attempts",
    "active_count",
    "null_count",
    "invalid_count",
    "uniform_attempt_count",
    "gaussian_attempt_count",
    "gaussian_accepted_count",
    "gaussian_rejected_count",
    "visible_count",
    "sampled_loss",
    "importance_max",
    "importance_ess",
    "importance_ess_per_attempt",
    "rendered_point_gaussian_pairs",
    "teacher_query_attempts",
    "student_query_attempts",
    "teacher_query_calls",
    "student_query_calls",
    "group_lrs_used",
    "gradient_max",
    "cardinality",
)


def prefix_replay_record(
    history: Mapping[str, Any],
    controller_record: Mapping[str, Any],
) -> dict[str, Any]:
    steps = history["steps"][:SCORE_STEPS]
    normalized = [{name: step[name] for name in _REPLAY_STEP_FIELDS} for step in steps]
    selection = controller_record["selection"]
    record = {
        "view_schedule_prefix": history["view_schedule"][:SCORE_STEPS],
        "steps": normalized,
        "controller_step_evidence": controller_record["step_evidence"],
        "scores": selection["scores"],
        "assigned_residual": selection["assigned_residual"],
        "selection_semantic_sha256": selection["semantic_sha256"],
        "pre_surgery_state_sha256": selection["pre_surgery_state"]["semantic_sha256"],
        "teacher_digest": history["teacher_digest_before"],
        "proposal_digest": history["proposal_digest_before"],
    }
    record["semantic_sha256"] = canonical_hash(record)
    return record


def _complexity_accounting(
    history: Mapping[str, Any],
    controller: Mapping[str, Any],
    *,
    inputs: ReconstructionInputs | None = None,
    products: Sequence[GaussianObservationField] | None = None,
) -> dict[str, Any]:
    steps = history["steps"]
    visible = [int(step["visible_count"]) for step in steps]
    point_pairs = [int(step["rendered_point_gaussian_pairs"]) for step in steps]
    vjp_bytes = 0
    for evidence in controller["step_evidence"]:
        visible_rows = len(evidence["visible_to_global"])
        vjp_bytes += visible_rows * 2 * torch.tensor([], dtype=torch.float32).element_size()
    cuda = {
        "available": torch.cuda.is_available(),
        "max_memory_allocated": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "max_memory_reserved": (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        ),
    }

    def tensor_object_bytes(value: object) -> int:
        return int(value.numel() * value.element_size()) if isinstance(value, torch.Tensor) else 0

    teacher_bytes = 0
    camera_bytes = 0
    if inputs is not None:
        for field in inputs.observations:
            teacher_bytes += sum(
                tensor_object_bytes(getattr(field, name, None))
                for name in (
                    "means",
                    "log_scales",
                    "rotations",
                    "colors",
                    "amplitudes",
                    "color_grads",
                    "filter_variance",
                )
            )
        for camera in inputs.cameras:
            camera_bytes += sum(tensor_object_bytes(value) for value in vars(camera).values())
    proposal_bytes = 0
    if products is not None:
        for field in products:
            proposal_bytes += sum(
                tensor_object_bytes(getattr(field, name, None))
                for name in (
                    "means",
                    "log_scales",
                    "rotations",
                    "colors",
                    "amplitudes",
                    "color_grads",
                    "filter_variance",
                )
            )

    dtype_bytes = {
        "torch.float32": 4,
        "torch.float64": 8,
        "torch.int64": 8,
        "torch.bool": 1,
    }

    def receipt_bytes(receipt: Mapping[str, Any]) -> int:
        elements = math.prod(int(value) for value in receipt["shape"])
        return elements * dtype_bytes[receipt["dtype"]]

    state_checkpoints = controller["state_checkpoints"]
    final_label = "140" if "140" in state_checkpoints else "35_pre"
    final_state = state_checkpoints[final_label]
    parameter_bytes = sum(
        receipt_bytes(item["parameter"]) for item in final_state["optimizers"]["groups"].values()
    )
    moment_bytes = sum(
        receipt_bytes(moment)
        for item in final_state["optimizers"]["groups"].values()
        for moment in item["moments"].values()
    )
    checkpoint_bytes = 0
    for state in state_checkpoints.values():
        checkpoint_bytes += receipt_bytes(state["persistent_ids"])
        for item in state["optimizers"]["groups"].values():
            checkpoint_bytes += receipt_bytes(item["parameter"])
            checkpoint_bytes += sum(receipt_bytes(moment) for moment in item["moments"].values())
    score_bytes = sum(
        (
            len(item["visible_to_global"]) * 8
            + len(item["native_residual_visible_float32"]) * 4
            + len(item["native_support_visible_float32"]) * 4
            + len(item["active_float32"]) * 4
            + len(item["error_float32"]) * 4
            + len(item["alpha_float32"]) * 4
            + len(item["residual_global_divided_float64"]) * 8
            + len(item["support_global_divided_float64"]) * 8
            + len(item["gradient_visible_float32"]) * 4
            + (
                0
                if item["means2d_gradient"] is None
                else math.prod(item["means2d_gradient"]["shape"]) * 4
            )
        )
        for item in controller["step_evidence"]
    )
    byte_categories = {
        "teacher_tensor_bytes": teacher_bytes,
        "proposal_tensor_bytes": proposal_bytes,
        "camera_tensor_bytes": camera_bytes,
        "current_parameter_bytes": parameter_bytes,
        "current_moment_bytes": moment_bytes,
        "score_evidence_bytes": score_bytes,
        "raw_state_checkpoint_tensor_bytes": checkpoint_bytes,
    }
    return {
        "teacher_components_per_view": [
            int(item["components"]) for item in history["preflight"]["views"]
        ],
        "proposal_components_per_view": [
            int(item["components"]) for item in history["proposal_preflight"]["views"]
        ],
        "teacher_index_diagnostics": history["index_diagnostics"],
        "proposal_index_diagnostics": history["proposal_index_diagnostics"],
        "teacher_preflight_views": history["preflight"]["views"],
        "proposal_preflight_views": history["proposal_preflight"]["views"],
        "configured_chunks": {
            "point_chunk": 256,
            "gaussian_chunk": 256,
            "outer_microbatch": 128,
            "query_component_chunk": 640,
            "maximum_point_by_gaussian_chunk_pairs": 256 * 256,
        },
        "visible_count_min": min(visible) if visible else 0,
        "visible_count_max": max(visible) if visible else 0,
        "rendered_point_gaussian_pairs_sum": sum(point_pairs),
        "rendered_point_gaussian_pairs_max": max(point_pairs) if point_pairs else 0,
        "teacher_query_attempts_sum": sum(int(step["teacher_query_attempts"]) for step in steps),
        "student_query_attempts_sum": sum(int(step["student_query_attempts"]) for step in steps),
        "vjp_native_output_bytes": vjp_bytes,
        "bytes_by_category": byte_categories,
        "sum_reported_category_bytes": sum(byte_categories.values()),
        "byte_category_note": (
            "Raw tensor/evidence accounting; checkpoints intentionally overlap "
            "current parameter/moment categories and support no performance claim."
        ),
        "n_before": history["n_init_3d"],
        "n_after": history["n_opt_3d"],
        "peak_rss_bytes": max((int(step["peak_rss_bytes"]) for step in steps), default=0),
        "process_peak_rss_bytes": (int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        "cuda": cuda,
    }


def _live_guard_receipt(guard: RGBAccessGuard) -> dict[str, Any]:
    loaded = guard._loaded_forbidden_modules()
    guard.forbidden_modules_at_exit = loaded
    receipt = guard.record()
    receipt["boundary_active_at_receipt"] = True
    if not receipt["passed"]:
        raise ProtocolInvalid("RGB denial boundary receipt failed")
    return receipt


def _phase_a_worker_inside_guard(
    *,
    guard: RGBAccessGuard,
    training_root: int,
    shuffle_root_value: int,
    marker_sha256: str,
    seal_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    _, actual_marker_sha = _authorize_marker(
        "phase-a",
        PHASE_A_ATTEMPT,
        artifact_type="compact_responsibility_birth_iter3_phase_a_attempt_v1",
    )
    if actual_marker_sha != marker_sha256 or sha256_file(SEAL) != seal_sha256:
        raise ProtocolInvalid("Phase-A worker marker/seal binding changed")
    expected_output = RUN_DIR / f"seed_{training_root}" / "phase_a_worker_result.json"
    if output_path.resolve() != expected_output.resolve():
        raise ProtocolInvalid("Phase-A worker output path is not the frozen path")
    entry_binding = _verified_binding_receipt()
    replicate = TRAIN_ROOTS.index(training_root)
    if SHUFFLE_ROOTS[replicate] != shuffle_root_value:
        raise ProtocolInvalid("Phase-A training/shuffle root pairing changed")
    worker_dir = RUN_DIR / f"seed_{training_root}" / "phase_a"
    worker_dir.mkdir(parents=True, exist_ok=False)
    inputs, products, initial, alignment = _load_frozen_inputs()
    pre_snapshots: list[Gaussians3D] = []
    controller = ResponsibilityBirthController(
        arm=None,
        shuffle_root=shuffle_root_value,
        split_root=None,
        on_pre_surgery=lambda snapshot, _: pre_snapshots.append(snapshot.to("cpu")),
    )
    final, history = CompactTrainer(frozen_config(training_root)).train(
        inputs,
        initial,
        proposal_fields=products,
        bundle_path=TEACHER_BUNDLE,
        topology_controller=controller,
        stop_after_step=SCORE_STEPS,
    )
    if (
        final.n != 835
        or len(pre_snapshots) != 1
        or factorial.gaussians_hash(final.to("cpu")) != factorial.gaussians_hash(pre_snapshots[0])
    ):
        raise ProtocolInvalid("Phase-A prefix/snapshot contract changed")
    controller_record = controller.history_record()
    if controller_record["split_root"] is not None:
        raise ProtocolInvalid("Phase A derived or retained a split root")
    replay = prefix_replay_record(history, controller_record)
    snapshot_record = _save_npz_snapshot(worker_dir / "gaussians_35_pre.npz", pre_snapshots[0])
    state_archive = save_state_archive(worker_dir / "states.npz", controller)
    load_state_archive(
        ROOT / state_archive["path"],
        expected_labels=("0", "35_pre"),
        expected_clocks={"0": 0, "35_pre": 35},
    )
    history_path = worker_dir / "history.json"
    history_sha = exclusive_json(history_path, history)
    complexity = _complexity_accounting(
        history, controller_record, inputs=inputs, products=products
    )
    denial = _live_guard_receipt(guard)
    exit_binding = _verified_binding_receipt()
    if exit_binding != entry_binding:
        raise ProtocolInvalid("Phase-A worker binding drifted during execution")
    payload = {
        "artifact_type": "compact_responsibility_birth_iter3_phase_a_worker_v1",
        "status": "PASS",
        "training_root": training_root,
        "shuffle_root": shuffle_root_value,
        "split_root": None,
        "n_init_3d": 835,
        "n_opt_3d": 835,
        "m_init_i_2d": inputs.n_init_2d,
        "m_opt_i_2d": inputs.n_opt_2d,
        "sum_m_opt_i_2d": sum(inputs.n_opt_2d),
        "alignment": alignment,
        "rgb_denial": denial,
        "binding_receipts": {
            "entry": entry_binding,
            "exit": exit_binding,
            "exact_match": True,
        },
        "snapshot_35_pre": snapshot_record,
        "raw_state_archive": state_archive,
        "selection": controller_record["selection"],
        "replay": replay,
        "history": {
            "path": history_path.relative_to(ROOT).as_posix(),
            "sha256": history_sha,
            "view_schedule_sha256": history["view_schedule_sha256"],
            "steps": history["steps"],
            "controller": controller_record,
        },
        "complexity_accounting": complexity,
    }
    exclusive_json(output_path, payload)
    return strict_json(output_path)


def _phase_a_worker(
    *,
    training_root: int,
    shuffle_root_value: int,
    marker_sha256: str,
    seal_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    guard = RGBAccessGuard()
    with guard:
        return _phase_a_worker_inside_guard(
            guard=guard,
            training_root=training_root,
            shuffle_root_value=shuffle_root_value,
            marker_sha256=marker_sha256,
            seal_sha256=seal_sha256,
            output_path=output_path,
        )


def _worker_command(*arguments: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        *arguments,
    ]


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = str(PRELOAD)
    environment.pop(FOCUSED_TEST_ENV, None)
    return environment


def _run_worker(command: Sequence[str]) -> dict[str, Any]:
    return _run_command(command, env=_worker_environment())


def _require_worker_success(record: Mapping[str, Any]) -> None:
    if record["returncode"] != 0:
        raise ProtocolInvalid(
            "official worker failed:\n"
            + str(record["stderr_tail"])[-6000:]
            + "\n"
            + str(record["stdout_tail"])[-6000:]
        )


def _validate_phase_a_worker_artifacts(
    worker: Mapping[str, Any],
) -> dict[str, Any]:
    history = worker["history"]
    if sha256_file(ROOT / history["path"]) != history["sha256"]:
        raise ProtocolInvalid("Phase-A history artifact changed")
    strict_json(ROOT / history["path"])
    archive = worker["raw_state_archive"]
    if sha256_file(ROOT / archive["path"]) != archive["sha256"]:
        raise ProtocolInvalid("Phase-A state archive changed")
    load_state_archive(
        ROOT / archive["path"],
        expected_labels=("0", "35_pre"),
        expected_clocks={"0": 0, "35_pre": 35},
    )
    snapshot = worker["snapshot_35_pre"]
    path = ROOT / snapshot["path"]
    if sha256_file(path) != snapshot["sha256"]:
        raise ProtocolInvalid("Phase-A snapshot artifact changed")
    loaded = Gaussians3D.load_npz(path)
    if loaded.n != 835 or factorial.gaussians_hash(loaded) != snapshot["semantic_sha256"]:
        raise ProtocolInvalid("Phase-A snapshot semantic changed")
    receipt = {
        "training_root": worker["training_root"],
        "history_sha256": history["sha256"],
        "state_archive_sha256": archive["sha256"],
        "snapshot_sha256": snapshot["sha256"],
    }
    receipt["semantic_sha256"] = canonical_hash(receipt)
    return receipt


def _validate_phase_b_worker_artifacts(
    worker: Mapping[str, Any],
) -> dict[str, Any]:
    training_root = int(worker["training_root"])
    replicate = TRAIN_ROOTS.index(training_root)
    expected_roots = (
        EVALUATION_ROOTS[replicate],
        SPLIT_ROOTS[replicate],
        SHUFFLE_ROOTS[replicate],
    )
    if (
        worker.get("artifact_type") != "compact_responsibility_birth_iter3_phase_b_worker_v1"
        or worker.get("status") != "PASS"
        or (
            worker.get("evaluation_root"),
            worker.get("split_root"),
            worker.get("shuffle_root"),
        )
        != expected_roots
        or worker.get("arm_order") != list(ARM_ORDER[training_root])
        or worker.get("n_init_3d") != 835
        or worker.get("n_opt_3d") != 867
        or set(worker.get("arms", {})) != set(ARMS)
    ):
        raise ProtocolInvalid("Phase-B worker root/cardinality schema changed")
    denial = worker.get("rgb_denial", {})
    if (
        denial.get("passed") is not True
        or denial.get("source_rgb_open_attempts") != 0
        or denial.get("forbidden_import_attempts") != 0
        or denial.get("negative_control_denials") != 3
        or denial.get("forbidden_modules_at_entry") != []
        or denial.get("forbidden_modules_at_exit") != []
        or denial.get("boundary_active_at_receipt") is not True
    ):
        raise ProtocolInvalid("Phase-B worker RGB denial receipt changed")
    current_binding = _verified_binding_receipt()
    bindings = worker.get("binding_receipts", {})
    if (
        bindings.get("exact_match") is not True
        or bindings.get("entry") != bindings.get("exit")
        or bindings.get("entry") != current_binding
    ):
        raise ProtocolInvalid("Phase-B worker binding receipts changed")
    inputs, products, frozen_initial, alignment = _load_frozen_inputs()
    if (
        worker.get("m_init_i_2d") != inputs.n_init_2d
        or worker.get("m_opt_i_2d") != inputs.n_opt_2d
        or worker.get("sum_m_opt_i_2d") != sum(inputs.n_opt_2d)
        or worker.get("alignment") != alignment
    ):
        raise ProtocolInvalid("Phase-B worker 2D component accounting changed")
    worker_dir = RUN_DIR / f"seed_{training_root}" / "phase_b"
    bank = worker["bank"]
    bank_path = ROOT / bank["path"]
    if (
        bank_path.resolve() != (worker_dir / "evaluation_banks.npz").resolve()
        or sha256_file(bank_path) != bank["sha256"]
        or bank_path.stat().st_size != bank["bytes"]
    ):
        raise ProtocolInvalid("Phase-B bank archive changed")
    banks, metadata = load_evaluation_bank(
        bank_path,
        expected_root=worker["evaluation_root"],
        product_fields=products,
    )
    if metadata != bank["metadata"]:
        raise ProtocolInvalid("Phase-B bank metadata changed")
    if [record.get("m_opt_2d") for record in metadata["views"]] != inputs.n_opt_2d:
        raise ProtocolInvalid("Phase-B bank/product cardinality binding changed")
    labels = ("0", "35_pre", "35_post", "70", "105", "140")
    clocks = {
        "0": 0,
        "35_pre": 35,
        "35_post": 35,
        "70": 70,
        "105": 105,
        "140": 140,
    }
    counts = {
        "0": 835,
        "35_pre": 835,
        "35_post": 867,
        "70": 867,
        "105": 867,
        "140": 867,
    }
    phase_a_result = strict_json(PHASE_A_RESULT)
    phase_a_worker = phase_a_result["workers"][replicate]
    phase_a_controller = phase_a_worker["history"]["controller"]
    initialized = frozen_initial.with_sh_degree(0).to("cuda:0")
    expected_initial = {
        "means": initialized.means.detach().cpu(),
        "quats": initialized.quats.detach().cpu(),
        "scales": initialized.log_scales.detach().cpu(),
        "opacities": torch.logit(initialized.opacity.clamp(1e-4, 1.0 - 1e-4)).detach().cpu(),
        "sh0": initialized.sh[:, :1].detach().cpu(),
        "shN": initialized.sh[:, 1:].detach().cpu(),
    }
    metric_cache: dict[str, dict[str, Any]] = {}
    arms: dict[str, Any] = {}
    for arm in ARMS:
        record = worker["arms"][arm]
        arm_dir = worker_dir / f"arm_{arm}"
        if (
            record.get("arm") != arm
            or record.get("arm_order_index") != ARM_ORDER[training_root].index(arm)
            or record.get("is_first_arm") is not (arm == ARM_ORDER[training_root][0])
            or record.get("n_init_3d") != 835
            or record.get("n_opt_3d") != 867
            or record.get("checkpoint_counts") != counts
            or set(record.get("checkpoint_metrics", {})) != set(labels)
            or set(record.get("snapshots", {})) != set(labels)
        ):
            raise ProtocolInvalid("Phase-B arm/cardinality schema changed")
        controller = record["controller"]
        _verify_semantic_record(controller)
        if (
            controller.get("arm") != arm
            or controller.get("score_steps") != SCORE_STEPS
            or controller.get("shuffle_root") != worker["shuffle_root"]
            or controller.get("split_root") != worker["split_root"]
            or set(controller.get("state_checkpoints", {})) != set(labels)
        ):
            raise ProtocolInvalid("Phase-B controller schema changed")
        history = record["history"]
        history_path = ROOT / history["path"]
        if (
            history_path.resolve() != (arm_dir / "history.json").resolve()
            or sha256_file(history_path) != history["sha256"]
        ):
            raise ProtocolInvalid("Phase-B history artifact changed")
        loaded_history = strict_json(history_path)
        if (
            loaded_history.get("steps") != history.get("steps")
            or loaded_history.get("view_schedule_sha256") != history.get("view_schedule_sha256")
            or loaded_history.get("teacher_digest_before") != history.get("teacher_digest_before")
            or loaded_history.get("teacher_digest_after") != history.get("teacher_digest_after")
            or loaded_history.get("proposal_digest_before") != history.get("proposal_digest_before")
            or loaded_history.get("proposal_digest_after") != history.get("proposal_digest_after")
            or loaded_history.get("topology_control") != controller
        ):
            raise ProtocolInvalid("Phase-B history/controller duplicate records differ")
        replay = record["replay"]
        _verify_semantic_record(replay)
        if replay != prefix_replay_record(loaded_history, controller) or record.get(
            "paired_sample_signature"
        ) != _paired_sample_signature(loaded_history):
            raise ProtocolInvalid("Phase-B history replay/pairing record changed")
        state = record["raw_state_archive"]
        state_path = ROOT / state["path"]
        if (
            state_path.resolve() != (arm_dir / "states.npz").resolve()
            or sha256_file(state_path) != state["sha256"]
            or state_path.stat().st_size != state["bytes"]
        ):
            raise ProtocolInvalid("Phase-B state archive changed")
        arrays, state_metadata = load_state_archive(
            state_path,
            expected_labels=labels,
            expected_clocks=clocks,
        )
        if state_metadata != state["metadata"]:
            raise ProtocolInvalid("Phase-B state metadata receipt changed")
        reconstructed_parameters: dict[str, dict[str, torch.Tensor]] = {}
        for label in labels:
            reconstructed, parameters, _ = _reconstruct_phase_a_state(
                arrays,
                state_metadata,
                label=label,
                n=counts[label],
                expected_step=clocks[label],
                require_initial_ids=label in {"0", "35_pre"},
            )
            if reconstructed != controller["state_checkpoints"][label]:
                raise ProtocolInvalid("Phase-B raw state/controller receipt changed")
            if (
                label in {"0", "35_pre"}
                and reconstructed != phase_a_controller["state_checkpoints"][label]
            ):
                raise ProtocolInvalid("Phase-B common raw state differs from Phase A")
            if label == "0" and any(
                not torch.equal(parameters[name], expected_initial[name]) for name in GROUP_ORDER
            ):
                raise ProtocolInvalid("Phase-B label-0 state differs from frozen INIT_PLY")
            reconstructed_parameters[label] = parameters
        loaded_snapshots: dict[str, Gaussians3D] = {}
        for label in labels:
            snapshot = record["snapshots"][label]
            snapshot_path = ROOT / snapshot["path"]
            if (
                snapshot_path.resolve() != (arm_dir / f"gaussians_{label}.npz").resolve()
                or sha256_file(snapshot_path) != snapshot["sha256"]
                or snapshot.get("n_gaussians") != counts[label]
            ):
                raise ProtocolInvalid("Phase-B snapshot artifact changed")
            loaded = Gaussians3D.load_npz(snapshot_path)
            raw_snapshot = _params_to_gaussians(reconstructed_parameters[label]).detach()
            if (
                loaded.n != counts[label]
                or factorial.gaussians_hash(loaded) != snapshot["semantic_sha256"]
                or factorial.gaussians_hash(raw_snapshot) != snapshot["semantic_sha256"]
                or any(
                    not torch.equal(getattr(raw_snapshot, name), getattr(loaded, name))
                    for name in ("means", "quats", "log_scales", "opacity", "sh")
                )
            ):
                raise ProtocolInvalid("Phase-B raw-state/snapshot semantic changed")
            loaded_snapshots[label] = loaded
        metric_receipts = _validate_checkpoint_metric_recomputation(
            loaded_snapshots,
            record["checkpoint_metrics"],
            inputs,
            banks,
            cache=metric_cache,
        )
        final_snapshot = loaded_snapshots["140"]
        ply = record["final_ply"]
        ply_path = ROOT / ply["path"]
        if (
            ply_path.resolve() != (arm_dir / "gaussians_final.ply").resolve()
            or sha256_file(ply_path) != ply["sha256"]
            or ply_path.stat().st_size != ply["bytes"]
            or ply.get("n_gaussians") != 867
            or ply.get("source_semantic_sha256") != factorial.gaussians_hash(final_snapshot)
        ):
            raise ProtocolInvalid("Phase-B PLY artifact changed")
        loaded_ply = Gaussians3D.load_ply(ply_path)
        roundtrip = _validate_final_ply_roundtrip(final_snapshot, loaded_ply)
        if roundtrip != ply["roundtrip"]:
            raise ProtocolInvalid("Phase-B PLY round-trip receipt changed")
        arms[arm] = {
            "history_sha256": history["sha256"],
            "state_archive_sha256": state["sha256"],
            "final_ply_sha256": ply["sha256"],
            "snapshot_manifest_sha256": canonical_hash(record["snapshots"]),
            "checkpoint_metric_sha256": metric_receipts,
        }
    sample_signatures = {
        worker["arms"][arm]["paired_sample_signature"]["semantic_sha256"] for arm in ARMS
    }
    split_draws = {
        (
            worker["arms"][arm]["controller"]["surgery"]["receipt"]["raw_split_child0_sha256"],
            worker["arms"][arm]["controller"]["surgery"]["receipt"]["raw_split_child1_sha256"],
            worker["arms"][arm]["controller"]["surgery"]["receipt"][
                "generator_state_before_sha256"
            ],
            worker["arms"][arm]["controller"]["surgery"]["receipt"]["generator_state_after_sha256"],
        )
        for arm in ARMS
    }
    first_arm = ARM_ORDER[training_root][0]
    expected_common_states = {
        label: worker["arms"][first_arm]["controller"]["state_checkpoints"][label][
            "semantic_sha256"
        ]
        for label in ("0", "35_pre")
    }
    pairing = worker.get("pairing", {})
    if (
        len(sample_signatures) != 1
        or len(split_draws) != 1
        or pairing.get("sample_signature_sha256") != next(iter(sample_signatures))
        or pairing.get("raw_split_draw_hash_tuple") != list(next(iter(split_draws)))
        or pairing.get("all_140_steps_identical") is not True
        or pairing.get("split_draws_identical") is not True
        or pairing.get("common_state_hashes") != expected_common_states
    ):
        raise ProtocolInvalid("Phase-B cross-arm pairing receipt changed")
    receipt: dict[str, Any] = {
        "training_root": worker["training_root"],
        "bank_sha256": bank["sha256"],
        "arms": arms,
    }
    receipt["semantic_sha256"] = canonical_hash(receipt)
    return receipt


def _phase_b_secondary_diagnostics(
    workers: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for worker in workers:
        arms = worker["arms"]
        comparisons: dict[str, Any] = {}
        for comparator in ("G", "U"):
            r_final = arms["R"]["checkpoint_metrics"]["140"]
            c_final = arms[comparator]["checkpoint_metrics"]["140"]
            per_view = []
            for r_view, c_view in zip(r_final["per_view"], c_final["per_view"], strict=True):
                per_view.append(
                    {
                        "view_name": r_view["view_name"],
                        "J_Q_ratio_R_over_C": r_view["J_Q"] / c_view["J_Q"],
                        "J_U_ratio_R_over_C": r_view["J_U"] / c_view["J_U"],
                    }
                )
            comparisons[comparator] = {
                "per_view_final_ratios": per_view,
                "worst_view_J_Q_ratio_R_over_C": (
                    r_final["worst_view_J_Q"] / c_final["worst_view_J_Q"]
                ),
                "worst_view_J_U_ratio_R_over_C": (
                    r_final["worst_view_J_U"] / c_final["worst_view_J_U"]
                ),
            }
        arm_diagnostics: dict[str, Any] = {}
        for arm, record in arms.items():
            metrics = record["checkpoint_metrics"]
            losses = [float(step["sampled_loss"]) for step in record["history"]["steps"]]
            selection = record["controller"]["selection"]
            arm_diagnostics[arm] = {
                "immediate_35_pre_to_35_post": {
                    "J_Q_delta": (metrics["35_post"]["J_Q"] - metrics["35_pre"]["J_Q"]),
                    "J_U_delta": (metrics["35_post"]["J_U"] - metrics["35_pre"]["J_U"]),
                },
                "sampled_loss": {
                    "steps": len(losses),
                    "mean": math.fsum(losses) / len(losses),
                    "minimum": min(losses),
                    "maximum": max(losses),
                    "final": losses[-1],
                },
                "selected_parent_diagnostics": {
                    "operator_counts": selection["operator_counts"][arm],
                    "scale_opacity_support": selection["non_gating_diagnostics"][
                        "selected_distributions"
                    ][arm],
                    "lineage": selection["non_gating_diagnostics"]["selection_lineage"][arm],
                },
                "final_ply": {
                    "bytes": record["final_ply"]["bytes"],
                    "sha256": record["final_ply"]["sha256"],
                },
                "complexity": record["complexity_accounting"],
            }
        roots[str(worker["training_root"])] = {
            "comparisons": comparisons,
            "arms": arm_diagnostics,
        }
    return {
        "authorizing": False,
        "changes_no_gate": True,
        "roots": roots,
        "worker_process_descriptives": [
            {
                "elapsed_seconds": command["elapsed_seconds"],
                "stdout_bytes": command["stdout_bytes"],
                "stderr_bytes": command["stderr_bytes"],
                "timing_and_memory_support_no_performance_claim": True,
            }
            for command in commands
        ],
    }


def _run_phase_a_inside_guard(guard: RGBAccessGuard) -> dict[str, Any]:
    sealed = verify_seal()
    parent_entry_binding = _verified_binding_receipt()
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
        "artifact_type": "compact_responsibility_birth_iter3_phase_a_attempt_v1",
        "timestamp_utc": factorial.timestamp_utc(),
        "seal_sha256": sha256_file(SEAL),
        "seal_attempt_sha256": sealed["seal_attempt"]["sha256"],
        "bindings_sha256": sealed["bindings_sha256"],
    }
    marker_sha = exclusive_json(PHASE_A_ATTEMPT, marker)
    # This strict reread is deliberately the very next operation.
    _, reread_sha = _authorize_marker(
        "phase-a",
        PHASE_A_ATTEMPT,
        artifact_type=marker["artifact_type"],
        expected=marker,
    )
    if reread_sha != marker_sha:
        raise ProtocolInvalid("Phase-A marker changed on immediate reread")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    workers: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    artifact_receipts: list[dict[str, Any]] = []
    try:
        for training_root, shuffle_root_value in zip(TRAIN_ROOTS, SHUFFLE_ROOTS, strict=True):
            output = RUN_DIR / f"seed_{training_root}" / "phase_a_worker_result.json"
            command = _worker_command(
                "_phase-a-worker",
                "--training-root",
                str(training_root),
                "--shuffle-root",
                str(shuffle_root_value),
                "--attempt-sha256",
                marker_sha,
                "--seal-sha256",
                sha256_file(SEAL),
                "--worker-output",
                str(output),
            )
            command_record = _run_worker(command)
            commands.append(command_record)
            _require_worker_success(command_record)
            worker = strict_json(output)
            workers.append(worker)
            artifact_receipts.append(_validate_phase_a_worker_artifacts(worker))
        gates = {
            str(worker["training_root"]): worker["selection"]["gates_without_assigned_fraction"]
            for worker in workers
        }
        all_gates = all(
            all(bool(value) for value in root_gates.values()) for root_gates in gates.values()
        )
        parent_exit_binding = _verified_binding_receipt()
        if parent_exit_binding != parent_entry_binding:
            raise ProtocolInvalid("Phase-A parent binding drifted during workers")
        publication_receipts = [_validate_phase_a_worker_artifacts(worker) for worker in workers]
        if publication_receipts != artifact_receipts:
            raise ProtocolInvalid("Phase-A artifacts drifted before publication")
        payload = {
            "artifact_type": "compact_responsibility_birth_iter3_phase_a_result_v1",
            "timestamp_utc": factorial.timestamp_utc(),
            "status": "PASS",
            "phase_a_decision": ("AUTHORIZE_AUDIT" if all_gates else "STOP_PHASE_B"),
            "seal_sha256": sha256_file(SEAL),
            "seal_attempt_sha256": sealed["seal_attempt"]["sha256"],
            "phase_a_attempt_sha256": marker_sha,
            "workers": workers,
            "gates": gates,
            "all_phase_a_gates_pass": all_gates,
            "commands": commands,
            "artifact_validation": {
                "after_worker": artifact_receipts,
                "before_publication": publication_receipts,
                "exact_match": True,
            },
            "parent_binding_receipts": {
                "entry": parent_entry_binding,
                "exit": parent_exit_binding,
                "exact_match": True,
            },
        }
    except BaseException as error:
        payload = {
            "artifact_type": "compact_responsibility_birth_iter3_phase_a_result_v1",
            "timestamp_utc": factorial.timestamp_utc(),
            "status": "FAIL",
            "phase_a_decision": "STOP_PHASE_B",
            "scientific_decision": "UNAVAILABLE",
            "seal_sha256": sha256_file(SEAL),
            "seal_attempt_sha256": sealed["seal_attempt"]["sha256"],
            "phase_a_attempt_sha256": marker_sha,
            "error": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "workers": workers,
            "commands": commands,
        }
    payload["parent_rgb_denial"] = _live_guard_receipt(guard)
    exclusive_json(PHASE_A_RESULT, payload)
    return strict_json(PHASE_A_RESULT)


def run_phase_a() -> dict[str, Any]:
    guard = RGBAccessGuard()
    with guard:
        return _run_phase_a_inside_guard(guard)


def _verify_semantic_record(record: Mapping[str, Any]) -> None:
    digest = dict(record)
    stored = digest.pop("semantic_sha256", None)
    try:
        actual = canonical_hash(digest)
    except (TypeError, ValueError) as error:
        raise ProtocolInvalid("semantic record is noncanonical or non-finite") from error
    if stored != actual:
        raise ProtocolInvalid("semantic record digest changed")


def _tensor_from_evidence(
    values: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=dtype)
    if shape is not None and tuple(tensor.shape) != shape:
        raise ProtocolInvalid("raw Phase-A evidence shape changed")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ProtocolInvalid("raw Phase-A evidence is non-finite")
    return tensor


def _phase_a_history_identity(
    training_root: int,
    history: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate all non-generative schedule/step identities before GPU replay."""
    expected_header = {
        "schema": "rtgs.compact_train_history.v2",
        "proposal_mode": "area_gaussian",
        "schedule_mode": "balanced_cycle",
        "target_mode": "proposal_attempt",
        "seed": training_root,
        "iterations": TRAIN_ITERATIONS,
        "attempts_per_step": TRAIN_ATTEMPTS,
        "uniform_fraction": 0.25,
        "extent": EXPLICIT_EXTENT,
        "extent_source": "explicit",
        "n_init_3d": 835,
        "n_opt_3d": 835,
        "completed_iterations": SCORE_STEPS,
        "stop_after_step": SCORE_STEPS,
        "topology_control_enabled": True,
        "proposal_field_source": "explicit",
    }
    if any(history.get(key) != value for key, value in expected_header.items()):
        raise ProtocolInvalid("Phase-A history frozen configuration changed")
    schedule = history.get("view_schedule")
    if not isinstance(schedule, list) or len(schedule) != TRAIN_ITERATIONS:
        raise ProtocolInvalid("Phase-A full view schedule length changed")
    if any(type(view) is not int or not 0 <= view < len(EXPECTED_VIEWS) for view in schedule):
        raise ProtocolInvalid("Phase-A view schedule contains an invalid view")
    schedule_sha = hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
    if schedule_sha != history.get("view_schedule_sha256"):
        raise ProtocolInvalid("Phase-A stored view schedule hash changed")
    visit_counts = [schedule.count(view) for view in range(len(EXPECTED_VIEWS))]
    prefix_visit_counts = [
        schedule[:SCORE_STEPS].count(view) for view in range(len(EXPECTED_VIEWS))
    ]
    if (
        visit_counts != [TRAIN_ITERATIONS // len(EXPECTED_VIEWS)] * len(EXPECTED_VIEWS)
        or history.get("planned_view_visit_counts") != visit_counts
        or history.get("view_visit_counts") != prefix_visit_counts
        or prefix_visit_counts != [SCORE_STEPS // len(EXPECTED_VIEWS)] * len(EXPECTED_VIEWS)
    ):
        raise ProtocolInvalid("Phase-A full schedule balance changed")
    steps = history.get("steps")
    if not isinstance(steps, list) or len(steps) != SCORE_STEPS or len(evidence) != SCORE_STEPS:
        raise ProtocolInvalid("Phase-A score-window history length changed")
    identities: list[dict[str, Any]] = []
    for step_index, (step, item) in enumerate(zip(steps, evidence, strict=True)):
        view_index = schedule[step_index]
        expected = {
            "step": step_index + 1,
            "view_index": view_index,
            "view_name": EXPECTED_VIEWS[view_index],
            "sample_seed": step_sample_seed(training_root, step_index),
            "attempts": TRAIN_ATTEMPTS,
        }
        if any(step.get(key) != value for key, value in expected.items()):
            raise ProtocolInvalid("Phase-A step sample/view/seed identity changed")
        if item.get("step") != step_index + 1 or item.get("view_index") != view_index:
            raise ProtocolInvalid("Phase-A controller/history view-step identity differs")
        if item.get("attempts") != TRAIN_ATTEMPTS:
            raise ProtocolInvalid("Phase-A controller attempt count changed")
        for name in (
            "active_count",
            "null_count",
            "invalid_count",
            "uniform_attempt_count",
            "gaussian_attempt_count",
            "gaussian_accepted_count",
            "gaussian_rejected_count",
            "visible_count",
        ):
            if type(step.get(name)) is not int or step[name] < 0:
                raise ProtocolInvalid(f"Phase-A step {name} is invalid")
        if (
            step["active_count"] + step["null_count"] != TRAIN_ATTEMPTS
            or step["uniform_attempt_count"] + step["gaussian_attempt_count"] != TRAIN_ATTEMPTS
            or step["gaussian_accepted_count"] + step["gaussian_rejected_count"]
            != step["gaussian_attempt_count"]
            or step["active_count"]
            != step["uniform_attempt_count"] + step["gaussian_accepted_count"]
            or step["visible_count"] != len(item.get("visible_to_global", ()))
            or step.get("active_sha256") != item.get("active_sha256")
        ):
            raise ProtocolInvalid("Phase-A active/sample count/hash identities changed")
        identities.append(expected)
    branch_counts = {
        "uniform": sum(step["uniform_attempt_count"] for step in steps),
        "gaussian": sum(step["gaussian_attempt_count"] for step in steps),
        "gaussian_accepted": sum(step["gaussian_accepted_count"] for step in steps),
        "gaussian_rejected": sum(step["gaussian_rejected_count"] for step in steps),
    }
    if history.get("proposal_branch_counts") != branch_counts:
        raise ProtocolInvalid("Phase-A aggregate proposal branch counts changed")
    view_diagnostics: list[dict[str, Any]] = []
    for view_index, view_name in enumerate(EXPECTED_VIEWS):
        view_steps = [step for step in steps if step["view_index"] == view_index]
        attempts = sum(step["attempts"] for step in view_steps)
        active = sum(step["active_count"] for step in view_steps)
        null = sum(step["null_count"] for step in view_steps)
        gaussian_attempts = sum(step["gaussian_attempt_count"] for step in view_steps)
        gaussian_accepted = sum(step["gaussian_accepted_count"] for step in view_steps)
        view_diagnostics.append(
            {
                "view_index": view_index,
                "view_name": view_name,
                "steps": len(view_steps),
                "attempts": attempts,
                "active_count": active,
                "null_count": null,
                "active_fraction": None if attempts == 0 else active / attempts,
                "null_fraction": None if attempts == 0 else null / attempts,
                "gaussian_attempt_count": gaussian_attempts,
                "gaussian_accepted_count": gaussian_accepted,
                "gaussian_acceptance_fraction": (
                    None if gaussian_attempts == 0 else gaussian_accepted / gaussian_attempts
                ),
            }
        )
    if history.get("proposal_view_diagnostics") != view_diagnostics:
        raise ProtocolInvalid("Phase-A aggregate per-view proposal diagnostics changed")
    record = {
        "view_schedule_sha256": schedule_sha,
        "view_visit_counts": visit_counts,
        "prefix_view_visit_counts": prefix_visit_counts,
        "step_identities_sha256": canonical_hash(identities),
        "proposal_branch_counts": branch_counts,
        "proposal_view_diagnostics_sha256": canonical_hash(view_diagnostics),
    }
    record["semantic_sha256"] = canonical_hash(record)
    return record


def _replay_phase_a_samples(
    training_root: int,
    history: Mapping[str, Any],
) -> dict[str, Any]:
    inputs, products, _, alignment = _load_frozen_inputs()
    device = torch.device("cuda:0")
    config = frozen_config(training_root)
    working_inputs = compact_trainer_module._compact_working_inputs(inputs, device)
    working_products = [field if field.device == device else field.to(device) for field in products]
    proposal_inputs = ReconstructionInputs(
        observations=working_products,
        cameras=working_inputs.cameras,
        view_names=list(working_inputs.view_names),
        points=None,
        point_visibility=None,
        bounds_hint=None,
        name=f"{working_inputs.name}-proposal",
        archive_stats=None,
    )
    _, proposal_backends, _ = compact_trainer_module._prepare_backends(
        proposal_inputs,
        config,
        None,
        role="proposal",
    )
    teacher_digest = compact_trainer_module.observation_digest(working_inputs)
    proposal_digest = compact_trainer_module.observation_digest(proposal_inputs)
    if (
        history.get("teacher_digest_before") != teacher_digest
        or history.get("teacher_digest_after") != teacher_digest
        or history.get("proposal_digest_before") != proposal_digest
        or history.get("proposal_digest_after") != proposal_digest
    ):
        raise ProtocolInvalid("Phase-A teacher/proposal semantic digest changed")
    schedule = build_view_schedule(
        working_inputs.n_views,
        TRAIN_ITERATIONS,
        training_root,
        mode="balanced_cycle",
    )
    if list(schedule) != history["view_schedule"]:
        raise ProtocolInvalid("Phase-A view schedule replay changed")
    schedule_sha = hashlib.sha256(
        json.dumps(list(schedule), separators=(",", ":")).encode()
    ).hexdigest()
    if schedule_sha != history["view_schedule_sha256"]:
        raise ProtocolInvalid("Phase-A view schedule hash changed")
    proposals = [
        GaussianPointProposal(field, backend)
        for field, backend in zip(working_products, proposal_backends, strict=True)
    ]
    step_hashes: list[str] = []
    for step_index, view_index in enumerate(schedule[:SCORE_STEPS]):
        step = history["steps"][step_index]
        seed = step_sample_seed(training_root, step_index)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        samples = proposals[view_index].sample(
            TRAIN_ATTEMPTS,
            uniform_fraction=0.25,
            generator=generator,
        )
        samples = compact_trainer_module._retarget_samples(samples, "proposal_attempt")
        uniform = samples.proposal_component_ids == -1
        gaussian = ~uniform
        replay = {
            "step": step_index + 1,
            "view_index": view_index,
            "view_name": working_inputs.view_names[view_index],
            "sample_seed": seed,
            "xy_sha256": tensor_hash(samples.xy),
            "active_sha256": tensor_hash(samples.active),
            "inside_fit_window_sha256": tensor_hash(samples.inside_fit_window),
            "proposal_component_ids_sha256": tensor_hash(samples.proposal_component_ids),
            "proposal_density_sha256": tensor_hash(samples.proposal_density),
            "joint_density_sha256": tensor_hash(samples.joint_density),
            "target_density_sha256": tensor_hash(samples.target_density),
            "importance_sha256": tensor_hash(samples.importance),
            "attempts": TRAIN_ATTEMPTS,
            "active_count": int(samples.active.sum()),
            "null_count": int((~samples.active).sum()),
            "invalid_count": int((~samples.inside_fit_window).sum()),
            "uniform_attempt_count": int(uniform.sum()),
            "gaussian_attempt_count": int(gaussian.sum()),
            "gaussian_accepted_count": int((gaussian & samples.active).sum()),
            "gaussian_rejected_count": int((gaussian & ~samples.active).sum()),
        }
        if any(step[key] != value for key, value in replay.items()):
            raise ProtocolInvalid("Phase-A proposal/sample replay changed")
        step_hashes.append(canonical_hash(replay))
    camera_dimensions = [
        {"width": camera.width, "height": camera.height} for camera in working_inputs.cameras
    ]
    record = {
        "view_schedule_sha256": schedule_sha,
        "step_replay_sha256": canonical_hash(step_hashes),
        "steps_replayed": SCORE_STEPS,
        "teacher_digest": teacher_digest,
        "proposal_digest": proposal_digest,
        "camera_dimensions": camera_dimensions,
        "m_init_i_2d": inputs.n_init_2d,
        "m_opt_i_2d": inputs.n_opt_2d,
        "sum_m_opt_i_2d": sum(inputs.n_opt_2d),
        "alignment": alignment,
    }
    record["semantic_sha256"] = canonical_hash(record)
    return record


def _expected_phase_a_optimizer_group(name: str, step: int) -> dict[str, Any]:
    initial_lrs = {
        "means": 1.6e-4 * EXPLICIT_EXTENT,
        "quats": 1e-3,
        "scales": 5e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
    }
    lr = initial_lrs[name]
    if name == "means":
        gamma = 0.01 ** (1.0 / TRAIN_ITERATIONS)
        for _ in range(step):
            lr *= gamma
    return {
        "name": name,
        "lr": lr,
        "betas": [0.9, 0.999],
        "eps": 1e-15,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "differentiable": False,
    }


def _array_tensor_receipt(
    array: np.ndarray,
    *,
    device: str = "cuda:0",
    include_values: bool = False,
) -> dict[str, Any]:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    record = _tensor_receipt(tensor, include_values=include_values)
    record["device"] = device
    return record


def _reconstruct_phase_a_state(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    label: str,
    n: int,
    expected_step: int | None = None,
    require_initial_ids: bool = True,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor]:
    if expected_step is None:
        if label not in {"0", "35_pre"}:
            raise ValueError("expected_step is required outside Phase-A labels")
        expected_step = 0 if label == "0" else SCORE_STEPS
    state_metadata = metadata["states"][label]
    expected_shapes = {
        "means": (n, 3),
        "quats": (n, 4),
        "scales": (n, 3),
        "opacities": (n,),
        "sh0": (n, 1, 3),
        "shN": (n, 0, 3),
    }
    parameters: dict[str, torch.Tensor] = {}
    parameter_receipts: dict[str, Any] = {}
    optimizer_groups: dict[str, Any] = {}
    for name in GROUP_ORDER:
        group_metadata = state_metadata["groups"][name]
        if group_metadata["step"] != expected_step or group_metadata[
            "group"
        ] != _expected_phase_a_optimizer_group(name, expected_step):
            raise ProtocolInvalid("Phase-A optimizer group/clock changed")
        tensor_records: dict[str, Any] = {}
        for tensor_name, descriptor in group_metadata["tensors"].items():
            array = arrays[descriptor["key"]]
            if (
                tuple(array.shape) != expected_shapes[name]
                or array.dtype != np.dtype(np.float32)
                or not bool(np.isfinite(array).all())
            ):
                raise ProtocolInvalid("Phase-A raw parameter/moment shape or dtype changed")
            tensor_records[tensor_name] = _array_tensor_receipt(array)
            if tensor_name == "parameter":
                parameters[name] = torch.from_numpy(np.ascontiguousarray(array).copy())
        parameter_receipts[name] = tensor_records["parameter"]
        moments = {key: value for key, value in tensor_records.items() if key != "parameter"}
        expected_moments = set() if expected_step == 0 else {"exp_avg", "exp_avg_sq"}
        if set(moments) != expected_moments:
            raise ProtocolInvalid("Phase-A raw optimizer moment set changed")
        optimizer_groups[name] = {
            "parameter": tensor_records["parameter"],
            "step": expected_step,
            "group": group_metadata["group"],
            "moments": moments,
        }
    ids_descriptor = state_metadata["persistent_ids"]
    ids_array = arrays[ids_descriptor["key"]]
    if ids_array.dtype != np.dtype(np.int64) or tuple(ids_array.shape) != (n,):
        raise ProtocolInvalid("Phase-A raw persistent-ID dtype/shape changed")
    ids = torch.from_numpy(np.ascontiguousarray(ids_array).copy())
    if ids.unique().numel() != n or int(ids.min()) < 0:
        raise ProtocolInvalid("Phase-A raw persistent IDs changed")
    if require_initial_ids and not torch.equal(ids, torch.arange(n, dtype=torch.long)):
        raise ProtocolInvalid("Phase-A raw persistent IDs changed")
    parameters_record: dict[str, Any] = {
        "order": list(GROUP_ORDER),
        "parameters": parameter_receipts,
    }
    parameters_record["semantic_sha256"] = canonical_hash(parameters_record)
    optimizer_record: dict[str, Any] = {
        "order": list(GROUP_ORDER),
        "groups": optimizer_groups,
    }
    optimizer_record["semantic_sha256"] = canonical_hash(optimizer_record)
    record: dict[str, Any] = {
        "parameters": parameters_record,
        "optimizers": optimizer_record,
        "persistent_ids": _array_tensor_receipt(ids_array, include_values=True),
    }
    record["semantic_sha256"] = canonical_hash(record)
    return record, parameters, ids


def recompute_phase_a_worker(worker: Mapping[str, Any]) -> dict[str, Any]:
    """Independently reconstruct all Phase-A scores, selections and gates."""
    if (
        worker.get("artifact_type") != "compact_responsibility_birth_iter3_phase_a_worker_v1"
        or worker.get("status") != "PASS"
        or worker.get("split_root") is not None
        or worker.get("n_init_3d") != 835
        or worker.get("n_opt_3d") != 835
    ):
        raise ProtocolInvalid("Phase-A worker status/root contract changed")
    training_root = int(worker["training_root"])
    focused_recompute = training_root in FOCUSED_TRAIN_ROOTS
    replicate = TRAIN_ROOTS.index(training_root)
    shuffle_root_value = SHUFFLE_ROOTS[replicate]
    if worker.get("shuffle_root") != shuffle_root_value:
        raise ProtocolInvalid("Phase-A shuffle-root pairing changed")
    expected_worker_dir = RUN_DIR / f"seed_{training_root}" / "phase_a"
    denial = worker.get("rgb_denial", {})
    if (
        denial.get("passed") is not True
        or denial.get("source_rgb_open_attempts") != 0
        or denial.get("forbidden_import_attempts") != 0
        or denial.get("negative_control_denials") != 3
        or denial.get("forbidden_modules_at_entry") != []
        or denial.get("forbidden_modules_at_exit") != []
        or denial.get("boundary_active_at_receipt") is not True
    ):
        raise ProtocolInvalid("Phase-A RGB denial evidence changed")
    current_binding = _verified_binding_receipt()
    bindings = worker.get("binding_receipts", {})
    if (
        bindings.get("exact_match") is not True
        or bindings.get("entry") != bindings.get("exit")
        or bindings.get("entry") != current_binding
    ):
        raise ProtocolInvalid("Phase-A worker binding receipts differ")

    history_record = worker["history"]
    history_path = ROOT / history_record["path"]
    if (
        not focused_recompute
        and history_path.resolve() != (expected_worker_dir / "history.json").resolve()
    ) or sha256_file(history_path) != history_record["sha256"]:
        raise ProtocolInvalid("Phase-A history file changed")
    history = strict_json(history_path)
    if (
        history["steps"] != history_record["steps"]
        or history_record.get("view_schedule_sha256") != history["view_schedule_sha256"]
    ):
        raise ProtocolInvalid("Phase-A embedded/file step or schedule-hash evidence differs")
    controller = history_record["controller"]
    if (
        controller.get("schema") != "rtgs.compact_responsibility_birth_iter3.controller.v1"
        or controller.get("arm") is not None
        or controller.get("score_steps") != SCORE_STEPS
        or controller.get("shuffle_root") != shuffle_root_value
        or controller.get("split_root") is not None
        or controller.get("selection") != worker.get("selection")
        or controller.get("state_checkpoints", {}).get("35_pre")
        != worker["selection"].get("pre_surgery_state")
    ):
        raise ProtocolInvalid("Phase-A controller/worker duplicate records differ")
    evidence = controller["step_evidence"]
    n = int(worker["n_init_3d"])
    history_identity = _phase_a_history_identity(training_root, history, evidence)
    replay_record = worker["replay"]

    archive = worker["raw_state_archive"]
    archive_path = ROOT / archive["path"]
    if (
        (
            not focused_recompute
            and archive_path.resolve() != (expected_worker_dir / "states.npz").resolve()
        )
        or sha256_file(archive_path) != archive["sha256"]
        or archive_path.stat().st_size != archive["bytes"]
    ):
        raise ProtocolInvalid("Phase-A raw state archive changed")
    raw_arrays, state_metadata = load_state_archive(
        archive_path,
        expected_labels=("0", "35_pre"),
        expected_clocks={"0": 0, "35_pre": SCORE_STEPS},
    )
    if state_metadata != archive["metadata"]:
        raise ProtocolInvalid("Phase-A raw state metadata changed")
    reconstructed_zero, zero_parameters, zero_ids = _reconstruct_phase_a_state(
        raw_arrays,
        state_metadata,
        label="0",
        n=n,
    )
    reconstructed_pre, raw_parameters, persistent_ids = _reconstruct_phase_a_state(
        raw_arrays,
        state_metadata,
        label="35_pre",
        n=n,
    )
    if (
        reconstructed_zero != controller["state_checkpoints"].get("0")
        or reconstructed_pre != controller["state_checkpoints"].get("35_pre")
        or reconstructed_pre != worker["selection"]["pre_surgery_state"]
        or replay_record.get("pre_surgery_state_sha256") != reconstructed_pre["semantic_sha256"]
        or not torch.equal(zero_ids, persistent_ids)
    ):
        raise ProtocolInvalid("Phase-A raw archive/state receipt binding changed")
    evidence_device = reconstructed_pre["parameters"]["parameters"]["means"]["device"]
    initial_parameter_hashes: dict[str, str] | None = None
    if not focused_recompute:
        _, _, frozen_initial, _ = _load_frozen_inputs()
        initialized = frozen_initial.with_sh_degree(0).to(evidence_device)
        expected_initial = {
            "means": initialized.means,
            "quats": initialized.quats,
            "scales": initialized.log_scales,
            "opacities": torch.logit(initialized.opacity.clamp(1e-4, 1.0 - 1e-4)),
            "sh0": initialized.sh[:, :1],
            "shN": initialized.sh[:, 1:],
        }
        if any(
            not torch.equal(zero_parameters[name], expected_initial[name].detach().cpu())
            for name in GROUP_ORDER
        ):
            raise ProtocolInvalid("Phase-A raw label-0 state differs from frozen INIT_PLY")
        initial_parameter_hashes = {
            name: tensor_hash(expected_initial[name]) for name in GROUP_ORDER
        }
    snapshot_record = worker["snapshot_35_pre"]
    snapshot_path = ROOT / snapshot_record["path"]
    if (
        (
            not focused_recompute
            and snapshot_path.resolve() != (expected_worker_dir / "gaussians_35_pre.npz").resolve()
        )
        or sha256_file(snapshot_path) != snapshot_record["sha256"]
        or snapshot_record.get("n_gaussians") != n
    ):
        raise ProtocolInvalid("Phase-A step-35 snapshot artifact changed")
    raw_snapshot = _params_to_gaussians(raw_parameters).detach()
    loaded_snapshot = Gaussians3D.load_npz(snapshot_path)
    if (
        factorial.gaussians_hash(raw_snapshot) != snapshot_record["semantic_sha256"]
        or factorial.gaussians_hash(loaded_snapshot) != snapshot_record["semantic_sha256"]
        or any(
            not torch.equal(getattr(raw_snapshot, name), getattr(loaded_snapshot, name))
            for name in ("means", "quats", "log_scales", "opacity", "sh")
        )
    ):
        raise ProtocolInvalid("Phase-A raw archive/snapshot binding changed")
    sample_replay = _replay_phase_a_samples(training_root, history)
    camera_dimensions = sample_replay["camera_dimensions"]
    if any(
        worker.get(key) != sample_replay[key]
        for key in ("m_init_i_2d", "m_opt_i_2d", "sum_m_opt_i_2d", "alignment")
    ):
        raise ProtocolInvalid("Phase-A 2D product cardinality/alignment changed")

    n_views = len(EXPECTED_VIEWS)
    residual_by_view = torch.zeros(n_views, n, dtype=torch.float64)
    support_by_view = torch.zeros_like(residual_by_view)
    view_counts = torch.zeros(n_views, dtype=torch.int64)
    gradient_sum = torch.zeros(n, dtype=torch.float32)
    gradient_count = torch.zeros(n, dtype=torch.int64)
    assigned_numerator = 0.0
    assigned_denominator = 0.0
    step_identity_hashes: list[str] = []
    for expected_step, item in enumerate(evidence, start=1):
        history_step = history["steps"][expected_step - 1]
        if (
            item["step"] != expected_step
            or item.get("attempts") != TRAIN_ATTEMPTS
            or item.get("view_index") != history_step["view_index"]
        ):
            raise ProtocolInvalid("Phase-A evidence step order changed")
        view = int(item["view_index"])
        visible = _tensor_from_evidence(item["visible_to_global"], dtype=torch.long)
        if (
            visible.numel() != history_step["visible_count"]
            or (visible.numel() and (int(visible.min()) < 0 or int(visible.max()) >= n))
            or visible.unique().numel() != visible.numel()
        ):
            raise ProtocolInvalid("Phase-A visible-to-global mapping changed")
        native_r = _tensor_from_evidence(
            item["native_residual_visible_float32"],
            dtype=torch.float32,
            shape=(visible.numel(),),
        )
        native_s = _tensor_from_evidence(
            item["native_support_visible_float32"],
            dtype=torch.float32,
            shape=(visible.numel(),),
        )
        active = _tensor_from_evidence(
            item["active_float32"],
            dtype=torch.float32,
            shape=(TRAIN_ATTEMPTS,),
        )
        error = _tensor_from_evidence(
            item["error_float32"],
            dtype=torch.float32,
            shape=(TRAIN_ATTEMPTS,),
        )
        alpha = _tensor_from_evidence(
            item["alpha_float32"],
            dtype=torch.float32,
            shape=(TRAIN_ATTEMPTS,),
        )
        if (
            not bool(((active == 0.0) | (active == 1.0)).all())
            or bool((error < 0.0).any())
            or bool((alpha < 0.0).any())
            or bool((alpha > 1.0 + ALPHA_EVIDENCE_UPPER_TOLERANCE).any())
            or bool((native_r < 0.0).any())
            or bool((native_s < 0.0).any())
            or tensor_hash(active.bool()) != item["active_sha256"]
            or tensor_hash(error) != item["point_loss_float32_sha256"]
        ):
            raise ProtocolInvalid("Phase-A active/error native evidence changed")
        gradient_receipts = item.get("six_parameter_gradients")
        if (
            not isinstance(gradient_receipts, dict)
            or set(gradient_receipts) != set(GROUP_ORDER)
            or canonical_hash(gradient_receipts) != item.get("six_parameter_gradients_sha256")
        ):
            raise ProtocolInvalid("Phase-A six-gradient receipt digest changed")
        for name, receipt in gradient_receipts.items():
            if (
                set(receipt) != {"dtype", "shape", "device", "finite", "sha256"}
                or receipt["dtype"] != "torch.float32"
                or receipt["shape"] != list(raw_parameters[name].shape)
                or receipt["device"] != evidence_device
                or receipt["finite"] is not True
                or not isinstance(receipt["sha256"], str)
                or len(receipt["sha256"]) != 64
            ):
                raise ProtocolInvalid("Phase-A six-gradient tensor receipt changed")
        hashes = {
            "visible": tensor_hash(visible),
            "native_r": tensor_hash(native_r),
            "native_s": tensor_hash(native_s),
            "active": tensor_hash(active),
            "error": tensor_hash(error),
            "alpha": tensor_hash(alpha),
        }
        expected_hashes = {
            "visible": item["visible_to_global_sha256"],
            "native_r": item["native_residual_visible_sha256"],
            "native_s": item["native_support_visible_sha256"],
            "active": item["active_float32_sha256"],
            "error": item["error_float32_sha256"],
            "alpha": item["alpha_float32_sha256"],
        }
        if hashes != expected_hashes:
            raise ProtocolInvalid("Phase-A raw evidence hash changed")
        native_r_sum = native_r.sum(dtype=torch.float32)
        native_s_sum = native_s.sum(dtype=torch.float32)
        alpha_r = (active * error * alpha).sum(dtype=torch.float32)
        alpha_s = (active * alpha).sum(dtype=torch.float32)
        denominator = (active * error).sum(dtype=torch.float32)
        if (
            float(native_r_sum) != item["native_residual_parent_sum_float32"]
            or float(native_s_sum) != item["native_support_parent_sum_float32"]
            or float(alpha_r) != item["alpha_residual_sum_float32"]
            or float(alpha_s) != item["alpha_support_sum_float32"]
            or float(denominator) != item["active_error_sum_float32"]
            or not torch.allclose(native_r_sum, alpha_r, atol=2e-6, rtol=2e-5)
            or not torch.allclose(native_s_sum, alpha_s, atol=2e-6, rtol=2e-5)
        ):
            raise ProtocolInvalid("Phase-A VJP/alpha identity recomputation failed")
        global_r = torch.zeros(n, dtype=torch.float64)
        global_s = torch.zeros(n, dtype=torch.float64)
        global_r.index_add_(0, visible, native_r.to(torch.float64) / TRAIN_ATTEMPTS)
        global_s.index_add_(0, visible, native_s.to(torch.float64) / TRAIN_ATTEMPTS)
        stored_global_r = _tensor_from_evidence(
            item["residual_global_divided_float64"],
            dtype=torch.float64,
            shape=(n,),
        )
        stored_global_s = _tensor_from_evidence(
            item["support_global_divided_float64"],
            dtype=torch.float64,
            shape=(n,),
        )
        if (
            not torch.equal(global_r, stored_global_r)
            or not torch.equal(global_s, stored_global_s)
            or tensor_hash(global_r) != item["residual_global_divided_sha256"]
            or tensor_hash(global_s) != item["support_global_divided_sha256"]
        ):
            raise ProtocolInvalid("Phase-A global VJP mapping changed")
        residual_by_view[view].add_(global_r)
        support_by_view[view].add_(global_s)
        view_counts[view] += 1

        scale = item["screen_gradient_scale"]
        camera = camera_dimensions[view]
        expected_scale = {
            "width": camera["width"],
            "height": camera["height"],
            "factor": max(camera["width"], camera["height"]) * 0.5,
        }
        if scale != expected_scale:
            raise ProtocolInvalid("Phase-A screen-gradient scale changed")
        means_receipt = item["means2d_gradient"]
        if visible.numel() == 0:
            if means_receipt is not None:
                raise ProtocolInvalid("empty-visible Phase-A step has a means2d gradient")
            means_grad = torch.empty(0, 2, dtype=torch.float32)
            native_g = torch.empty(0, dtype=torch.float32)
        else:
            if (
                not isinstance(means_receipt, dict)
                or set(means_receipt) != {"dtype", "shape", "device", "finite", "sha256", "values"}
                or means_receipt["dtype"] != "torch.float32"
                or means_receipt["shape"] != [visible.numel(), 2]
                or means_receipt["device"] != evidence_device
                or means_receipt["finite"] is not True
            ):
                raise ProtocolInvalid("Phase-A means2d gradient receipt changed")
            means_grad = _tensor_from_evidence(
                means_receipt["values"],
                dtype=torch.float32,
                shape=(visible.numel(), 2),
            )
            if tensor_hash(means_grad) != means_receipt["sha256"]:
                raise ProtocolInvalid("Phase-A means2d gradient hash changed")
            native_g = (torch.linalg.vector_norm(means_grad, dim=-1) * scale["factor"]).to(
                torch.float32
            )
        stored_g = _tensor_from_evidence(
            item["gradient_visible_float32"],
            dtype=torch.float32,
            shape=(visible.numel(),),
        )
        if (
            not torch.equal(native_g, stored_g)
            or bool((native_g < 0.0).any())
            or tensor_hash(native_g) != item["gradient_visible_sha256"]
        ):
            raise ProtocolInvalid("Phase-A native G recomputation failed")
        gradient_sum.index_add_(0, visible, native_g)
        gradient_count.index_add_(0, visible, torch.ones_like(visible, dtype=torch.int64))
        assigned_numerator += float(native_r_sum)
        assigned_denominator += float(denominator)
        step_identity_hashes.append(canonical_hash(hashes))

    if not torch.equal(view_counts, torch.full((n_views,), 5, dtype=torch.int64)):
        raise ProtocolInvalid("Phase-A does not have exact five visits per view")
    _verify_semantic_record(controller)
    _verify_semantic_record(replay_record)
    expected_replay_record = prefix_replay_record(history, controller)
    if history.get("topology_control") != controller or replay_record != expected_replay_record:
        raise ProtocolInvalid("Phase-A controller/replay duplicate record changed")
    r_view = residual_by_view / view_counts[:, None]
    s_view = support_by_view / view_counts[:, None]
    residual = r_view.mean(dim=0)
    support = s_view.mean(dim=0)
    gradient = gradient_sum / gradient_count.clamp_min(1).to(torch.float32)
    selection = worker["selection"]
    scores = selection["scores"]
    raw_scale_max = raw_parameters["scales"].exp().amax(dim=-1).to(torch.float32)
    stored_scale_max = _tensor_from_evidence(
        scores["scale_max"],
        dtype=torch.float32,
        shape=(n,),
    )
    if (
        not torch.equal(raw_scale_max, stored_scale_max)
        or scores.get("visible_step_count") != gradient_count.tolist()
        or scores.get("view_step_count") != view_counts.tolist()
    ):
        raise ProtocolInvalid("Phase-A raw-state scale/count score evidence changed")
    comparisons = (
        (gradient, scores["G_float32"], torch.float32),
        (residual, scores["R_float64"], torch.float64),
        (support, scores["S_float64"], torch.float64),
        (r_view, scores["R_by_view_float64"], torch.float64),
        (s_view, scores["S_by_view_float64"], torch.float64),
    )
    for actual, stored, dtype in comparisons:
        expected = _tensor_from_evidence(stored, dtype=dtype, shape=tuple(actual.shape))
        if not torch.equal(actual, expected):
            raise ProtocolInvalid("Phase-A score reduction changed")
    if (
        not math.isfinite(assigned_numerator)
        or not math.isfinite(assigned_denominator)
        or assigned_denominator <= 0
    ):
        raise ProtocolInvalid("Phase-A assigned fraction inputs changed")
    fraction = assigned_numerator / assigned_denominator
    assigned = selection["assigned_residual"]
    if (
        not math.isfinite(fraction)
        or assigned_numerator != assigned["native_float32_numerator_reduced_before_division"]
        or assigned_denominator != assigned["native_float32_denominator"]
        or fraction != assigned["fraction"]
    ):
        raise ProtocolInvalid("Phase-A assigned fraction changed")
    recomputed_selection = build_matched_selections(
        gradient_score=gradient,
        residual_score=residual,
        support_score=support,
        support_by_view=s_view,
        visible_step_count=gradient_count,
        scale_max=raw_scale_max,
        persistent_ids=persistent_ids,
        extent=EXPLICIT_EXTENT,
        shuffle_root=shuffle_root_value,
    )
    selection_keys = (
        "eligible_rows",
        "eligible_ids",
        "positive_support_view_count",
        "strata",
        "shuffle",
        "selected_rows",
        "selected_ids",
        "overlaps",
        "shuffle_moved_fraction",
        "residual_score_sums",
        "operator_counts",
    )
    for key in selection_keys:
        if recomputed_selection[key] != selection[key]:
            raise ProtocolInvalid(f"Phase-A selection field changed: {key}")
    expected_gates = dict(recomputed_selection["gates_without_assigned_fraction"])
    expected_gates["assigned_fraction_ge_0_10"] = (
        math.isfinite(assigned_numerator)
        and math.isfinite(assigned_denominator)
        and assigned_denominator > 0
        and math.isfinite(fraction)
        and fraction >= 0.10
    )
    if (
        expected_gates != selection["gates_without_assigned_fraction"]
        or bool(all(expected_gates.values())) != selection["all_phase_a_gates_pass"]
    ):
        raise ProtocolInvalid("Phase-A gate recomputation changed")
    if (
        selection.get("step_evidence_sha256") != canonical_hash(evidence)
        or controller.get("persistent_ids") != persistent_ids.tolist()
    ):
        raise ProtocolInvalid("Phase-A controller selection evidence binding changed")
    selection_semantic = canonical_hash(
        {
            "selection": {
                key: selection[key]
                for key in (
                    "eligible_rows",
                    "strata",
                    "shuffle",
                    "selected_rows",
                    "overlaps",
                )
            },
            "scores": selection["scores"],
            "assigned_residual": selection["assigned_residual"],
            "gates": selection["gates_without_assigned_fraction"],
            "step_evidence_sha256": selection["step_evidence_sha256"],
            "pre_surgery_state_sha256": reconstructed_pre["semantic_sha256"],
        }
    )
    if (
        selection.get("semantic_sha256") != selection_semantic
        or replay_record.get("selection_semantic_sha256") != selection_semantic
        or replay_record.get("scores") != selection["scores"]
        or replay_record.get("assigned_residual") != selection["assigned_residual"]
        or replay_record.get("controller_step_evidence") != evidence
    ):
        raise ProtocolInvalid("Phase-A selection/replay semantic binding changed")
    summary: dict[str, Any] = {
        "training_root": training_root,
        "step_identity_sha256": canonical_hash(step_identity_hashes),
        "history_identity": history_identity,
        "sample_replay": sample_replay,
        "view_counts": view_counts.tolist(),
        "G_sha256": tensor_hash(gradient),
        "R_sha256": tensor_hash(residual),
        "S_sha256": tensor_hash(support),
        "assigned_residual": {
            "numerator": assigned_numerator,
            "denominator": assigned_denominator,
            "fraction": fraction,
        },
        "selection_sha256": canonical_hash(
            {key: recomputed_selection[key] for key in selection_keys}
        ),
        "selection_semantic_sha256": selection_semantic,
        "raw_pre_state_semantic_sha256": reconstructed_pre["semantic_sha256"],
        "initial_parameter_hashes": initial_parameter_hashes,
        "snapshot_semantic_sha256": snapshot_record["semantic_sha256"],
        "state_archive_sha256": archive["sha256"],
        "gates": expected_gates,
        "all_phase_a_gates_pass": all(expected_gates.values()),
        "rgb_denial_passed": True,
        "binding_receipts_match": True,
    }
    summary["semantic_sha256"] = canonical_hash(summary)
    return summary


def recompute_phase_a_result(result: Mapping[str, Any]) -> dict[str, Any]:
    workers = result.get("workers")
    if not isinstance(workers, list) or len(workers) != len(TRAIN_ROOTS):
        raise ProtocolInvalid("Phase-A result worker count changed")
    if [worker.get("training_root") for worker in workers] != list(TRAIN_ROOTS):
        raise ProtocolInvalid("Phase-A result worker order changed")
    summaries = [recompute_phase_a_worker(worker) for worker in workers]
    gates = {str(item["training_root"]): item["gates"] for item in summaries}
    all_gates = all(item["all_phase_a_gates_pass"] for item in summaries)
    if (
        result.get("gates") != gates
        or result.get("all_phase_a_gates_pass") is not all_gates
        or result.get("phase_a_decision") != ("AUTHORIZE_AUDIT" if all_gates else "STOP_PHASE_B")
    ):
        raise ProtocolInvalid("Phase-A parent gate/decision reduction changed")
    record: dict[str, Any] = {
        "schema": "rtgs.compact_responsibility_birth_iter3.phase_a_recompute.v1",
        "workers": summaries,
        "gates": gates,
        "all_phase_a_gates_pass": all_gates,
        "phase_a_decision": result["phase_a_decision"],
    }
    record["semantic_sha256"] = canonical_hash(record)
    return record


def _validate_phase_a_parent_receipts(
    result: Mapping[str, Any],
    *,
    phase_a_marker_sha256: str,
    seal_sha256: str,
) -> None:
    current_binding = _verified_binding_receipt()
    parent_bindings = result.get("parent_binding_receipts", {})
    if (
        parent_bindings.get("exact_match") is not True
        or parent_bindings.get("entry") != parent_bindings.get("exit")
        or parent_bindings.get("entry") != current_binding
    ):
        raise ProtocolInvalid("Phase-A parent binding receipts changed")
    denial = result.get("parent_rgb_denial", {})
    if (
        denial.get("passed") is not True
        or denial.get("source_rgb_open_attempts") != 0
        or denial.get("forbidden_import_attempts") != 0
        or denial.get("negative_control_denials") != 3
        or denial.get("forbidden_modules_at_entry") != []
        or denial.get("forbidden_modules_at_exit") != []
        or denial.get("boundary_active_at_receipt") is not True
    ):
        raise ProtocolInvalid("Phase-A parent RGB denial receipt changed")
    commands = result.get("commands")
    if not isinstance(commands, list) or len(commands) != len(TRAIN_ROOTS):
        raise ProtocolInvalid("Phase-A worker command count changed")
    for record, training_root, shuffle_root_value in zip(
        commands,
        TRAIN_ROOTS,
        SHUFFLE_ROOTS,
        strict=True,
    ):
        output = RUN_DIR / f"seed_{training_root}" / "phase_a_worker_result.json"
        expected = _worker_command(
            "_phase-a-worker",
            "--training-root",
            str(training_root),
            "--shuffle-root",
            str(shuffle_root_value),
            "--attempt-sha256",
            phase_a_marker_sha256,
            "--seal-sha256",
            seal_sha256,
            "--worker-output",
            str(output),
        )
        if (
            record.get("command") != expected
            or record.get("returncode") != 0
            or type(record.get("stdout_bytes")) is not int
            or type(record.get("stderr_bytes")) is not int
            or not 0 <= record["stdout_bytes"] <= 8000
            or not 0 <= record["stderr_bytes"] <= 8000
            or len(str(record.get("stdout_tail", "")).encode()) != record["stdout_bytes"]
            or len(str(record.get("stderr_tail", "")).encode()) != record["stderr_bytes"]
            or hashlib.sha256(str(record["stdout_tail"]).encode()).hexdigest()
            != record.get("stdout_sha256")
            or hashlib.sha256(str(record["stderr_tail"]).encode()).hexdigest()
            != record.get("stderr_sha256")
            or not isinstance(record.get("elapsed_seconds"), (int, float))
            or isinstance(record.get("elapsed_seconds"), bool)
            or not math.isfinite(record["elapsed_seconds"])
            or record["elapsed_seconds"] < 0
        ):
            raise ProtocolInvalid("Phase-A worker command receipt changed")
    independently_validated = [
        _validate_phase_a_worker_artifacts(worker) for worker in result["workers"]
    ]
    artifact_validation = result.get("artifact_validation", {})
    if (
        artifact_validation.get("exact_match") is not True
        or artifact_validation.get("after_worker") != independently_validated
        or artifact_validation.get("before_publication") != independently_validated
    ):
        raise ProtocolInvalid("Phase-A parent artifact-validation receipt changed")


def verify_phase_a_authorization(
    phase_a_result_path: Path,
    phase_a_audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if phase_a_result_path.resolve() != PHASE_A_RESULT.resolve():
        raise ProtocolInvalid("Phase B requires the exact frozen Phase-A result path")
    if phase_a_audit_path.resolve() != PHASE_A_AUDIT.resolve():
        raise ProtocolInvalid("Phase B requires the exact frozen Phase-A audit path")
    marker, phase_a_marker_sha = _authorize_marker(
        "phase-a",
        PHASE_A_ATTEMPT,
        artifact_type="compact_responsibility_birth_iter3_phase_a_attempt_v1",
    )
    sealed = verify_seal()
    if (
        set(marker)
        != {
            "artifact_type",
            "timestamp_utc",
            "seal_sha256",
            "seal_attempt_sha256",
            "bindings_sha256",
        }
        or not isinstance(marker["timestamp_utc"], str)
        or not marker["timestamp_utc"].strip()
        or marker["seal_sha256"] != sha256_file(SEAL)
        or marker["seal_attempt_sha256"] != sealed["seal_attempt"]["sha256"]
        or marker["bindings_sha256"] != sealed["bindings_sha256"]
    ):
        raise ProtocolInvalid("Phase-A attempt marker binding changed")
    result = strict_json(phase_a_result_path)
    audit = strict_json(phase_a_audit_path)
    if (
        set(result)
        != {
            "artifact_type",
            "timestamp_utc",
            "status",
            "phase_a_decision",
            "seal_sha256",
            "seal_attempt_sha256",
            "phase_a_attempt_sha256",
            "workers",
            "gates",
            "all_phase_a_gates_pass",
            "commands",
            "artifact_validation",
            "parent_binding_receipts",
            "parent_rgb_denial",
        }
        or result.get("artifact_type") != "compact_responsibility_birth_iter3_phase_a_result_v1"
        or result.get("status") != "PASS"
        or result.get("phase_a_decision") != "AUTHORIZE_AUDIT"
        or result.get("all_phase_a_gates_pass") is not True
        or not isinstance(result.get("timestamp_utc"), str)
        or not result["timestamp_utc"].strip()
        or result.get("seal_sha256") != sha256_file(SEAL)
        or result.get("seal_attempt_sha256") != sealed["seal_attempt"]["sha256"]
        or result.get("phase_a_attempt_sha256") != phase_a_marker_sha
        or len(result.get("workers", [])) != 3
    ):
        raise ProtocolInvalid("Phase-A result does not authorize an audit/Phase B")
    _validate_phase_a_parent_receipts(
        result,
        phase_a_marker_sha256=phase_a_marker_sha,
        seal_sha256=sha256_file(SEAL),
    )
    for worker, training_root in zip(result["workers"], TRAIN_ROOTS, strict=True):
        if (
            worker.get("status") != "PASS"
            or worker.get("training_root") != training_root
            or worker.get("split_root") is not None
            or worker["selection"].get("all_phase_a_gates_pass") is not True
        ):
            raise ProtocolInvalid("Phase-A worker evidence is incomplete")
        _verify_semantic_record(worker["replay"])
        archive = worker["raw_state_archive"]
        archive_path = ROOT / archive["path"]
        if sha256_file(archive_path) != archive["sha256"]:
            raise ProtocolInvalid("Phase-A raw state archive changed")
        _, metadata = load_state_archive(
            archive_path,
            expected_labels=("0", "35_pre"),
            expected_clocks={"0": 0, "35_pre": 35},
        )
        if metadata != archive["metadata"]:
            raise ProtocolInvalid("Phase-A raw state metadata changed")
        labels = metadata["labels"]
        if labels != ["0", "35_pre"]:
            raise ProtocolInvalid("Phase-A state labels changed")
    recomputed = recompute_phase_a_result(result)
    if recomputed["all_phase_a_gates_pass"] is not True:
        raise ProtocolInvalid("raw Phase-A recomputation does not pass all gates")
    bindings = audit.get("bindings", {})
    auditor = audit.get("auditor")
    if (
        audit.get("artifact_type") != "compact_responsibility_birth_phase_a_audit_v1"
        or audit.get("verdict") != "PASS"
        or audit.get("unresolved_findings") != []
        or bindings.get("preregistration_sha256") != sha256_file(PREREGISTRATION)
        or bindings.get("seal_sha256") != sha256_file(SEAL)
        or bindings.get("phase_a_attempt_sha256") != phase_a_marker_sha
        or bindings.get("phase_a_result_sha256") != sha256_file(PHASE_A_RESULT)
        or audit.get("recomputed") != recomputed
        or not isinstance(auditor, dict)
        or not isinstance(auditor.get("identity"), str)
        or not auditor["identity"].strip()
        or not isinstance(auditor.get("provenance"), str)
        or not auditor["provenance"].strip()
    ):
        raise ProtocolInvalid("independent Phase-A audit is invalid or unbound")
    return result, audit


_PAIRING_FIELDS = (
    "view_index",
    "sample_seed",
    "xy_sha256",
    "active_sha256",
    "inside_fit_window_sha256",
    "proposal_component_ids_sha256",
    "proposal_density_sha256",
    "joint_density_sha256",
    "target_density_sha256",
    "importance_sha256",
    "attempts",
    "active_count",
    "null_count",
    "invalid_count",
)


def _paired_sample_signature(history: Mapping[str, Any]) -> dict[str, Any]:
    steps = [{field: record[field] for field in _PAIRING_FIELDS} for record in history["steps"]]
    record = {
        "view_schedule_sha256": history["view_schedule_sha256"],
        "steps": steps,
    }
    record["semantic_sha256"] = canonical_hash(record)
    return record


def _phase_b_worker_inside_guard(
    *,
    guard: RGBAccessGuard,
    training_root: int,
    evaluation_root: int,
    split_root_value: int,
    shuffle_root_value: int,
    marker_sha256: str,
    seal_sha256: str,
    phase_a_result_sha256: str,
    phase_a_audit_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    _, actual_marker_sha = _authorize_marker(
        "phase-b",
        PHASE_B_ATTEMPT,
        artifact_type="compact_responsibility_birth_iter3_phase_b_attempt_v1",
    )
    if (
        actual_marker_sha != marker_sha256
        or sha256_file(SEAL) != seal_sha256
        or sha256_file(PHASE_A_RESULT) != phase_a_result_sha256
        or sha256_file(PHASE_A_AUDIT) != phase_a_audit_sha256
    ):
        raise ProtocolInvalid("Phase-B worker lifecycle binding changed")
    expected_output = RUN_DIR / f"seed_{training_root}" / "phase_b_worker_result.json"
    if output_path.resolve() != expected_output.resolve():
        raise ProtocolInvalid("Phase-B worker output path is not the frozen path")
    entry_binding = _verified_binding_receipt()
    phase_a_result, _ = verify_phase_a_authorization(PHASE_A_RESULT, PHASE_A_AUDIT)
    replicate = TRAIN_ROOTS.index(training_root)
    expected_roots = (
        EVALUATION_ROOTS[replicate],
        SPLIT_ROOTS[replicate],
        SHUFFLE_ROOTS[replicate],
    )
    if (
        evaluation_root,
        split_root_value,
        shuffle_root_value,
    ) != expected_roots:
        raise ProtocolInvalid("Phase-B replicate root pairing changed")
    phase_a_worker = phase_a_result["workers"][replicate]
    worker_dir = RUN_DIR / f"seed_{training_root}" / "phase_b"
    worker_dir.mkdir(parents=True, exist_ok=False)
    with torch.cuda.device(0):
        inputs, products, initial, alignment = _load_frozen_inputs()
        bank_record = generate_evaluation_bank(
            evaluation_root=evaluation_root,
            teachers=inputs,
            product_fields=products,
            path=worker_dir / "evaluation_banks.npz",
        )
        banks, bank_metadata = load_evaluation_bank(
            ROOT / bank_record["path"],
            expected_root=evaluation_root,
            product_fields=products,
        )
        if bank_metadata != bank_record["metadata"]:
            raise ProtocolInvalid("fresh evaluation bank changed on strict reload")

        common_metrics: dict[str, dict[str, Any]] = {}
        common_hashes: dict[str, str] = {}
        common_state_hashes: dict[str, str] = {}
        arm_records: dict[str, Any] = {}
        first_arm = ARM_ORDER[training_root][0]
        for arm_index, arm in enumerate(ARM_ORDER[training_root]):
            arm_dir = worker_dir / f"arm_{arm}"
            arm_dir.mkdir(parents=True, exist_ok=False)
            snapshots: dict[str, Gaussians3D] = {}
            checkpoint_metrics: dict[str, dict[str, Any]] = {}
            controller_holder: dict[str, ResponsibilityBirthController] = {}

            def checkpoint_callback(
                snapshot: Gaussians3D,
                step: int,
                *,
                current_arm_index: int = arm_index,
                current_snapshots: dict[str, Gaussians3D] = snapshots,
                current_metrics: dict[str, dict[str, Any]] = checkpoint_metrics,
                holder: dict[str, ResponsibilityBirthController] = controller_holder,
            ) -> None:
                label = "0" if step == 0 else "35_post" if step == 35 else str(step)
                cpu_snapshot = snapshot.to("cpu")
                current_snapshots[label] = cpu_snapshot
                semantic = factorial.gaussians_hash(cpu_snapshot)
                if label == "0":
                    state_hash = holder["controller"]._state_checkpoints["0"]["semantic_sha256"]
                else:
                    state_hash = ""
                if label == "0" and current_arm_index > 0:
                    if semantic != common_hashes["0"] or state_hash != common_state_hashes["0"]:
                        raise ProtocolInvalid("later arm step-0 state differs")
                    current_metrics[label] = common_metrics[label]
                else:
                    metric = evaluate_snapshot(snapshot, inputs, banks)
                    current_metrics[label] = metric
                    if label == "0":
                        common_hashes[label] = semantic
                        common_state_hashes[label] = state_hash
                        common_metrics[label] = metric

            def pre_surgery_callback(
                snapshot: Gaussians3D,
                selection: dict[str, Any],
                *,
                current_arm_index: int = arm_index,
                current_snapshots: dict[str, Gaussians3D] = snapshots,
                current_metrics: dict[str, dict[str, Any]] = checkpoint_metrics,
            ) -> None:
                cpu_snapshot = snapshot.to("cpu")
                current_snapshots["35_pre"] = cpu_snapshot
                semantic = factorial.gaussians_hash(cpu_snapshot)
                if current_arm_index == 0:
                    metric = evaluate_snapshot(snapshot, inputs, banks)
                    common_hashes["35_pre"] = semantic
                    common_state_hashes["35_pre"] = selection["pre_surgery_state"][
                        "semantic_sha256"
                    ]
                    common_metrics["35_pre"] = metric
                    current_metrics["35_pre"] = metric
                else:
                    if (
                        semantic != common_hashes["35_pre"]
                        or selection["pre_surgery_state"]["semantic_sha256"]
                        != common_state_hashes["35_pre"]
                        or selection["pre_surgery_state"]["semantic_sha256"]
                        != phase_a_worker["selection"]["pre_surgery_state"]["semantic_sha256"]
                    ):
                        raise ProtocolInvalid("later arm step-35 pre-state differs")
                    current_metrics["35_pre"] = common_metrics["35_pre"]

            controller = ResponsibilityBirthController(
                arm=arm,
                shuffle_root=shuffle_root_value,
                split_root=split_root_value,
                expected_selection_sha256=phase_a_worker["selection"]["semantic_sha256"],
                expected_pre_state_sha256=phase_a_worker["selection"]["pre_surgery_state"][
                    "semantic_sha256"
                ],
                on_pre_surgery=pre_surgery_callback,
            )
            controller_holder["controller"] = controller
            final, history = CompactTrainer(frozen_config(training_root)).train(
                inputs,
                initial,
                proposal_fields=products,
                bundle_path=TEACHER_BUNDLE,
                checkpoint_callback=checkpoint_callback,
                topology_controller=controller,
            )
            controller_record = controller.history_record()
            replay = prefix_replay_record(history, controller_record)
            if replay != phase_a_worker["replay"]:
                raise ProtocolInvalid("Phase-B common prefix differs exactly from Phase A")
            if set(checkpoint_metrics) != {
                "0",
                "35_pre",
                "35_post",
                "70",
                "105",
                "140",
            }:
                raise ProtocolInvalid("Phase-B evaluation labels changed")
            if final.n != 867 or history["n_opt_3d"] != 867:
                raise ProtocolInvalid("Phase-B final cardinality changed")
            final_state = controller_record["state_checkpoints"].get("140")
            if final_state is None or any(
                group["step"] != 140 for group in final_state["optimizers"]["groups"].values()
            ):
                raise ProtocolInvalid("final Adam clocks differ from 140")
            if len(controller_record["lineage"]) != 48:
                raise ProtocolInvalid("final lineage count differs from 48")
            state_archive = save_state_archive(arm_dir / "states.npz", controller)
            _, loaded_state_metadata = load_state_archive(
                ROOT / state_archive["path"],
                expected_labels=(
                    "0",
                    "35_pre",
                    "35_post",
                    "70",
                    "105",
                    "140",
                ),
                expected_clocks={
                    "0": 0,
                    "35_pre": 35,
                    "35_post": 35,
                    "70": 70,
                    "105": 105,
                    "140": 140,
                },
            )
            if loaded_state_metadata != state_archive["metadata"]:
                raise ProtocolInvalid("arm state archive changed on reload")
            snapshot_records = {
                label: _save_npz_snapshot(arm_dir / f"gaussians_{label}.npz", snapshot)
                for label, snapshot in snapshots.items()
            }
            final_ply = _save_final_ply(arm_dir / "gaussians_final.ply", final.to("cpu"))
            history_path = arm_dir / "history.json"
            history_sha = exclusive_json(history_path, history)
            complexity = _complexity_accounting(
                history, controller_record, inputs=inputs, products=products
            )
            arm_records[arm] = {
                "arm": arm,
                "arm_order_index": arm_index,
                "is_first_arm": arm == first_arm,
                "n_init_3d": 835,
                "checkpoint_counts": {
                    "0": 835,
                    "35_pre": 835,
                    "35_post": 867,
                    "70": 867,
                    "105": 867,
                    "140": 867,
                },
                "n_opt_3d": final.n,
                "checkpoint_metrics": checkpoint_metrics,
                "snapshots": snapshot_records,
                "final_ply": final_ply,
                "raw_state_archive": state_archive,
                "controller": controller_record,
                "replay": replay,
                "paired_sample_signature": _paired_sample_signature(history),
                "history": {
                    "path": history_path.relative_to(ROOT).as_posix(),
                    "sha256": history_sha,
                    "steps": history["steps"],
                    "view_schedule_sha256": history["view_schedule_sha256"],
                    "teacher_digest_before": history["teacher_digest_before"],
                    "teacher_digest_after": history["teacher_digest_after"],
                    "proposal_digest_before": history["proposal_digest_before"],
                    "proposal_digest_after": history["proposal_digest_after"],
                },
                "complexity_accounting": complexity,
            }

        signatures = {
            arm_records[arm]["paired_sample_signature"]["semantic_sha256"] for arm in ARMS
        }
        split_draws = {
            (
                arm_records[arm]["controller"]["surgery"]["receipt"]["raw_split_child0_sha256"],
                arm_records[arm]["controller"]["surgery"]["receipt"]["raw_split_child1_sha256"],
                arm_records[arm]["controller"]["surgery"]["receipt"][
                    "generator_state_before_sha256"
                ],
                arm_records[arm]["controller"]["surgery"]["receipt"][
                    "generator_state_after_sha256"
                ],
            )
            for arm in ARMS
        }
        if len(signatures) != 1 or len(split_draws) != 1:
            raise ProtocolInvalid("paired samples or raw split draws differ across arms")
    denial = _live_guard_receipt(guard)
    exit_binding = _verified_binding_receipt()
    if exit_binding != entry_binding:
        raise ProtocolInvalid("Phase-B worker binding drifted during execution")
    payload = {
        "artifact_type": "compact_responsibility_birth_iter3_phase_b_worker_v1",
        "status": "PASS",
        "training_root": training_root,
        "evaluation_root": evaluation_root,
        "split_root": split_root_value,
        "shuffle_root": shuffle_root_value,
        "arm_order": list(ARM_ORDER[training_root]),
        "m_init_i_2d": inputs.n_init_2d,
        "m_opt_i_2d": inputs.n_opt_2d,
        "sum_m_opt_i_2d": sum(inputs.n_opt_2d),
        "n_init_3d": 835,
        "n_opt_3d": 867,
        "alignment": alignment,
        "bank": bank_record,
        "arms": arm_records,
        "pairing": {
            "sample_signature_sha256": next(iter(signatures)),
            "raw_split_draw_hash_tuple": list(next(iter(split_draws))),
            "all_140_steps_identical": True,
            "split_draws_identical": True,
            "common_state_hashes": common_state_hashes,
        },
        "rgb_denial": denial,
        "binding_receipts": {
            "entry": entry_binding,
            "exit": exit_binding,
            "exact_match": True,
        },
    }
    exclusive_json(output_path, payload)
    return strict_json(output_path)


def _phase_b_worker(
    *,
    training_root: int,
    evaluation_root: int,
    split_root_value: int,
    shuffle_root_value: int,
    marker_sha256: str,
    seal_sha256: str,
    phase_a_result_sha256: str,
    phase_a_audit_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    guard = RGBAccessGuard()
    with guard:
        return _phase_b_worker_inside_guard(
            guard=guard,
            training_root=training_root,
            evaluation_root=evaluation_root,
            split_root_value=split_root_value,
            shuffle_root_value=shuffle_root_value,
            marker_sha256=marker_sha256,
            seal_sha256=seal_sha256,
            phase_a_result_sha256=phase_a_result_sha256,
            phase_a_audit_sha256=phase_a_audit_sha256,
            output_path=output_path,
        )


def _run_phase_b_inside_guard(
    guard: RGBAccessGuard,
    phase_a_result_path: Path,
    phase_a_audit_path: Path,
) -> dict[str, Any]:
    sealed = verify_seal()
    parent_entry_binding = _verified_binding_receipt()
    phase_a_result, phase_a_audit = verify_phase_a_authorization(
        phase_a_result_path, phase_a_audit_path
    )
    if PHASE_B_ATTEMPT.exists() or RESULT.exists():
        raise ProtocolInvalid("Phase-B namespace is not pristine")
    marker = {
        "artifact_type": "compact_responsibility_birth_iter3_phase_b_attempt_v1",
        "timestamp_utc": factorial.timestamp_utc(),
        "seal_sha256": sha256_file(SEAL),
        "seal_attempt_sha256": sealed["seal_attempt"]["sha256"],
        "bindings_sha256": sealed["bindings_sha256"],
        "phase_a_attempt_sha256": sha256_file(PHASE_A_ATTEMPT),
        "phase_a_result_sha256": sha256_file(PHASE_A_RESULT),
        "phase_a_audit_sha256": sha256_file(PHASE_A_AUDIT),
    }
    marker_sha = exclusive_json(PHASE_B_ATTEMPT, marker)
    # This strict reread is deliberately the very next operation.
    _, reread_sha = _authorize_marker(
        "phase-b",
        PHASE_B_ATTEMPT,
        artifact_type=marker["artifact_type"],
        expected=marker,
    )
    if reread_sha != marker_sha:
        raise ProtocolInvalid("Phase-B marker changed on immediate reread")
    workers: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    artifact_receipts: list[dict[str, Any]] = []
    try:
        for training_root, evaluation_root, split_root_value, shuffle_root_value in zip(
            TRAIN_ROOTS,
            EVALUATION_ROOTS,
            SPLIT_ROOTS,
            SHUFFLE_ROOTS,
            strict=True,
        ):
            output = RUN_DIR / f"seed_{training_root}" / "phase_b_worker_result.json"
            command = _worker_command(
                "_phase-b-worker",
                "--training-root",
                str(training_root),
                "--evaluation-root",
                str(evaluation_root),
                "--split-root",
                str(split_root_value),
                "--shuffle-root",
                str(shuffle_root_value),
                "--attempt-sha256",
                marker_sha,
                "--seal-sha256",
                sha256_file(SEAL),
                "--phase-a-result-sha256",
                sha256_file(PHASE_A_RESULT),
                "--phase-a-audit-sha256",
                sha256_file(PHASE_A_AUDIT),
                "--worker-output",
                str(output),
            )
            command_record = _run_worker(command)
            commands.append(command_record)
            _require_worker_success(command_record)
            worker = strict_json(output)
            workers.append(worker)
            artifact_receipts.append(_validate_phase_b_worker_artifacts(worker))
        records = {worker["training_root"]: worker["arms"] for worker in workers}
        active_fractions = [
            float(view["banks"]["proposal"]["active_fraction"])
            for worker in workers
            for view in worker["bank"]["metadata"]["views"]
        ]
        structural = (
            len(workers) == 3
            and all(worker["status"] == "PASS" for worker in workers)
            and all(worker["pairing"]["all_140_steps_identical"] for worker in workers)
            and all(worker["pairing"]["split_draws_identical"] for worker in workers)
            and all(worker["rgb_denial"]["passed"] for worker in workers)
            and all(
                arm_record["n_opt_3d"] == 867
                for worker in workers
                for arm_record in worker["arms"].values()
            )
        )
        decision = compute_terminal_decision(
            records,
            active_fractions,
            structural_passed=structural,
        )
        status = (
            "PASS" if structural and decision["scientific_decision"] != "UNAVAILABLE" else "FAIL"
        )
        secondary_diagnostics = _phase_b_secondary_diagnostics(workers, commands)
        parent_exit_binding = _verified_binding_receipt()
        if parent_exit_binding != parent_entry_binding:
            raise ProtocolInvalid("Phase-B parent binding drifted during workers")
        publication_receipts = [_validate_phase_b_worker_artifacts(worker) for worker in workers]
        if publication_receipts != artifact_receipts:
            raise ProtocolInvalid("Phase-B artifacts drifted before publication")
        payload = {
            "artifact_type": "compact_responsibility_birth_iter3_result_v1",
            "timestamp_utc": factorial.timestamp_utc(),
            "status": status,
            "scientific_decision": decision["scientific_decision"],
            "decision": decision,
            "seal_sha256": sha256_file(SEAL),
            "seal_attempt_sha256": sealed["seal_attempt"]["sha256"],
            "phase_a_attempt_sha256": sha256_file(PHASE_A_ATTEMPT),
            "phase_a_result_sha256": sha256_file(PHASE_A_RESULT),
            "phase_a_audit_sha256": sha256_file(PHASE_A_AUDIT),
            "phase_b_attempt_sha256": marker_sha,
            "phase_a_result_binding": {
                "phase_a_decision": phase_a_result["phase_a_decision"],
                "all_phase_a_gates_pass": phase_a_result["all_phase_a_gates_pass"],
            },
            "phase_a_audit_binding": {
                "verdict": phase_a_audit["verdict"],
                "unresolved_findings": phase_a_audit["unresolved_findings"],
            },
            "structural_invariants_passed": structural,
            "component_accounting": {
                "m_init_i_2d": workers[0]["m_init_i_2d"],
                "m_opt_i_2d": workers[0]["m_opt_i_2d"],
                "sum_m_opt_i_2d": workers[0]["sum_m_opt_i_2d"],
                "N_init_3d": 835,
                "N_opt_3d": 867,
                "count_trajectory": {
                    "0": 835,
                    "35_pre": 835,
                    "35_post": 867,
                    "70": 867,
                    "105": 867,
                    "140": 867,
                },
                "clone_parents": 16,
                "split_parents": 16,
                "newborn_rows": 48,
                "net_growth": 32,
                "pruned": 0,
            },
            "secondary_diagnostics": secondary_diagnostics,
            "workers": workers,
            "commands": commands,
            "artifact_validation": {
                "after_worker": artifact_receipts,
                "before_publication": publication_receipts,
                "exact_match": True,
            },
            "parent_binding_receipts": {
                "entry": parent_entry_binding,
                "exit": parent_exit_binding,
                "exact_match": True,
            },
        }
    except BaseException as error:
        payload = {
            "artifact_type": "compact_responsibility_birth_iter3_result_v1",
            "timestamp_utc": factorial.timestamp_utc(),
            "status": "FAIL",
            "scientific_decision": "UNAVAILABLE",
            "seal_sha256": sha256_file(SEAL),
            "seal_attempt_sha256": sealed["seal_attempt"]["sha256"],
            "phase_b_attempt_sha256": marker_sha,
            "error": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "workers": workers,
            "commands": commands,
        }
    payload["parent_rgb_denial"] = _live_guard_receipt(guard)
    exclusive_json(RESULT, payload)
    return strict_json(RESULT)


def run_phase_b(
    phase_a_result_path: Path,
    phase_a_audit_path: Path,
) -> dict[str, Any]:
    guard = RGBAccessGuard()
    with guard:
        return _run_phase_b_inside_guard(guard, phase_a_result_path, phase_a_audit_path)


def parameter_state_record(params: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    values = {name: _tensor_receipt(params[name]) for name in GROUP_ORDER}
    record = {"order": list(GROUP_ORDER), "parameters": values}
    record["semantic_sha256"] = canonical_hash(record)
    return record


def _params_to_gaussians(params: Mapping[str, torch.Tensor]) -> Gaussians3D:
    return Gaussians3D(
        means=params["means"],
        quats=params["quats"],
        log_scales=params["scales"],
        opacity=torch.sigmoid(params["opacities"]),
        sh=torch.cat([params["sh0"], params["shN"]], dim=1),
    )


def _score_order(
    members: Sequence[int],
    score: torch.Tensor,
    persistent_ids: torch.Tensor,
) -> list[int]:
    return sorted(
        members,
        key=lambda row: (-float(score[row]), int(persistent_ids[row])),
    )


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
    """Construct exact G/R/U allocations and auditable shuffle receipts."""
    _require_root_authorized(shuffle_root)
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
        raise ValueError("selection vectors must be aligned 1-D tensors")
    if support_by_view.ndim != 2 or support_by_view.shape[1] != n:
        raise ValueError("support_by_view must have shape (V,N)")
    if persistent_ids.dtype != torch.long or persistent_ids.unique().numel() != n:
        raise ValueError("persistent_ids must be unique int64 values")
    if not bool(torch.isfinite(scale_max).all()):
        raise ProtocolInvalid("selection scales are non-finite")

    finite = (
        torch.isfinite(gradient_score)
        & torch.isfinite(residual_score)
        & torch.isfinite(support_score)
    )
    positive_views = (support_by_view > 0).sum(dim=0)
    eligible_mask = finite & (support_score > 0) & (positive_views >= 2) & (visible_step_count > 0)
    eligible = eligible_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    boundary = 0.01 * float(extent)
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
        raise ProtocolInvalid("strata are not exhaustive")
    if sum(map(len, stratum_rows.values())) != len(eligible):
        raise ProtocolInvalid("strata overlap")

    selected: dict[str, list[int]] = {arm: [] for arm in ARMS}
    shuffle_records: dict[str, Any] = {}
    stratum_records: dict[str, Any] = {}
    for stratum in STRATA:
        members = stratum_rows[stratum]
        if len(members) < quota_per_stratum:
            raise ProtocolInvalid(f"stratum {stratum} lacks its frozen quota")
        g_order = _score_order(members, gradient_score, persistent_ids)
        r_order = _score_order(members, residual_score, persistent_ids)
        selected["G"].extend(g_order[:quota_per_stratum])
        selected["R"].extend(r_order[:quota_per_stratum])

        recipients = sorted(members, key=lambda row: int(persistent_ids[row]))
        derived = shuffle_seed(shuffle_root, stratum)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derived)
        state_before = generator_state_hash(generator)
        permutation_tensor = torch.randperm(len(recipients), generator=generator)
        state_after = generator_state_hash(generator)
        permutation = [int(value) for value in permutation_tensor.tolist()]
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
        winners = sorted(assignments, key=lambda item: item["source_rank"])[:quota_per_stratum]
        selected["U"].extend(item["recipient_row"] for item in winners)
        moved = sum(item["recipient_row"] != item["source_row"] for item in assignments)
        shuffle_records[stratum] = {
            "root": shuffle_root,
            "derived_seed": derived,
            "generator_state_before_sha256": state_before,
            "generator_state_after_sha256": state_after,
            "permutation": permutation,
            "permutation_sha256": tensor_hash(permutation_tensor),
            "draw_sha256": canonical_hash(permutation),
            "assignments": assignments,
            "assignment_sha256": canonical_hash(assignments),
            "fixed_points": len(assignments) - moved,
            "moved": moved,
            "selected_rows": [item["recipient_row"] for item in winners],
        }
        member_set = set(members)
        stratum_records[stratum] = {
            "members": members,
            "member_ids": [int(persistent_ids[row]) for row in members],
            "gradient_order": g_order,
            "residual_order": r_order,
            "selected": {arm: [row for row in selected[arm] if row in member_set] for arm in ARMS},
        }

    expected = len(STRATA) * quota_per_stratum
    for arm, rows in selected.items():
        if len(rows) != expected or len(set(rows)) != expected:
            raise ProtocolInvalid(f"arm {arm} count or uniqueness changed")
        if not set(rows) <= set(eligible):
            raise ProtocolInvalid(f"arm {arm} selected an ineligible row")
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
    moved = sum(item["moved"] for item in shuffle_records.values())
    assigned = sum(len(item["assignments"]) for item in shuffle_records.values())
    score_sums = {
        arm: float(
            residual_score[torch.tensor(rows, device=residual_score.device, dtype=torch.long)].sum(
                dtype=torch.float64
            )
        )
        for arm, rows in selected.items()
    }
    operator_counts = {
        arm: {
            "clone": sum(float(scale_max[row]) <= boundary for row in rows),
            "split": sum(float(scale_max[row]) > boundary for row in rows),
        }
        for arm, rows in selected.items()
    }
    expected_operator_count = 2 * quota_per_stratum
    if any(
        counts
        != {
            "clone": expected_operator_count,
            "split": expected_operator_count,
        }
        for counts in operator_counts.values()
    ):
        raise ProtocolInvalid("matched selection clone/split operator counts changed")
    gates = {
        "all_strata_have_quota": all(
            len(stratum_rows[name]) >= quota_per_stratum for name in STRATA
        ),
        "selection_count_and_match": all(
            len(rows) == expected
            and len(set(rows)) == expected
            and operator_counts[arm]
            == {
                "clone": expected_operator_count,
                "split": expected_operator_count,
            }
            for arm, rows in selected.items()
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
    record: dict[str, Any] = {
        "schema": "rtgs.compact_responsibility_birth_iter3.selection.v1",
        "scale_boundary": boundary,
        "eligible_rows": eligible,
        "eligible_ids": [int(persistent_ids[row]) for row in eligible],
        "positive_support_view_count": positive_views.cpu().tolist(),
        "strata": stratum_records,
        "shuffle": shuffle_records,
        "selected_rows": selected,
        "selected_ids": {
            arm: [int(persistent_ids[row]) for row in rows] for arm, rows in selected.items()
        },
        "overlaps": overlaps,
        "shuffle_moved_fraction": moved / assigned,
        "residual_score_sums": score_sums,
        "operator_counts": operator_counts,
        "gates_without_assigned_fraction": gates,
    }
    record["semantic_sha256"] = canonical_hash(record)
    return record


def run_focused_smoke(path: Path) -> dict[str, Any]:
    """Exercise fresh domains and selection without touching an official root."""
    if path.exists():
        raise FileExistsError(path)
    n = 40
    persistent_ids = torch.arange(n, dtype=torch.long)
    support = torch.arange(1, n + 1, dtype=torch.float64)
    residual = torch.arange(n, 0, -1, dtype=torch.float64).square()
    gradient = torch.roll(residual.to(torch.float32), 7)
    scale_max = torch.cat((torch.full((20,), 0.005), torch.full((20,), 0.02)))
    support_by_view = support[None, :].repeat(3, 1) / 3.0
    selection = build_matched_selections(
        gradient_score=gradient,
        residual_score=residual,
        support_score=support,
        support_by_view=support_by_view,
        visible_step_count=torch.ones(n, dtype=torch.int64),
        scale_max=scale_max,
        persistent_ids=persistent_ids,
        extent=1.0,
        shuffle_root=FOCUSED_SHUFFLE_ROOTS[0],
    )
    seed_receipts = {
        "evaluation": evaluation_bank_seed(FOCUSED_EVALUATION_ROOTS[0], "focused_view", "uniform"),
        "split": split_seed(FOCUSED_SPLIT_ROOTS[0]),
        "shuffle": {stratum: shuffle_seed(FOCUSED_SHUFFLE_ROOTS[0], stratum) for stratum in STRATA},
    }
    payload = {
        "artifact_type": "compact_responsibility_birth_iter3_focused_smoke_v1",
        "status": "PASS",
        "decision_bearing": False,
        "official_roots_consumed": [],
        "official_artifacts_created": [],
        "roots_used": [
            FOCUSED_EVALUATION_ROOTS[0],
            FOCUSED_SPLIT_ROOTS[0],
            FOCUSED_SHUFFLE_ROOTS[0],
        ],
        "focused_roots": {
            "evaluation": FOCUSED_EVALUATION_ROOTS[0],
            "split": FOCUSED_SPLIT_ROOTS[0],
            "shuffle": FOCUSED_SHUFFLE_ROOTS[0],
        },
        "seed_receipts": seed_receipts,
        "selection_sha256": selection["semantic_sha256"],
        "selection_counts": {arm: len(selection["selected_rows"][arm]) for arm in ARMS},
        "root_guard": _dynamic_pre_marker_root_proof(),
    }
    exclusive_json(path, payload)
    return strict_json(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact responsibility-birth allocation iter3 lifecycle"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal")
    commands.add_parser("phase-a")
    phase_b = commands.add_parser("phase-b")
    phase_b.add_argument("phase_a_result", type=Path)
    phase_b.add_argument("phase_a_audit", type=Path)

    phase_a_worker = commands.add_parser("_phase-a-worker")
    phase_a_worker.add_argument("--training-root", type=int, required=True)
    phase_a_worker.add_argument("--shuffle-root", type=int, required=True)
    phase_a_worker.add_argument("--attempt-sha256", required=True)
    phase_a_worker.add_argument("--seal-sha256", required=True)
    phase_a_worker.add_argument("--worker-output", type=Path, required=True)

    phase_b_worker = commands.add_parser("_phase-b-worker")
    phase_b_worker.add_argument("--training-root", type=int, required=True)
    phase_b_worker.add_argument("--evaluation-root", type=int, required=True)
    phase_b_worker.add_argument("--split-root", type=int, required=True)
    phase_b_worker.add_argument("--shuffle-root", type=int, required=True)
    phase_b_worker.add_argument("--attempt-sha256", required=True)
    phase_b_worker.add_argument("--seal-sha256", required=True)
    phase_b_worker.add_argument("--phase-a-result-sha256", required=True)
    phase_b_worker.add_argument("--phase-a-audit-sha256", required=True)
    phase_b_worker.add_argument("--worker-output", type=Path, required=True)

    focused = commands.add_parser("_focused-smoke-worker")
    focused.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "seal":
        effective_argv = list(sys.argv) if argv is None else [sys.argv[0], *argv]
        result = create_seal(effective_argv)
    elif arguments.command == "phase-a":
        result = run_phase_a()
    elif arguments.command == "phase-b":
        result = run_phase_b(arguments.phase_a_result, arguments.phase_a_audit)
    elif arguments.command == "_phase-a-worker":
        result = _phase_a_worker(
            training_root=arguments.training_root,
            shuffle_root_value=arguments.shuffle_root,
            marker_sha256=arguments.attempt_sha256,
            seal_sha256=arguments.seal_sha256,
            output_path=arguments.worker_output,
        )
    elif arguments.command == "_phase-b-worker":
        result = _phase_b_worker(
            training_root=arguments.training_root,
            evaluation_root=arguments.evaluation_root,
            split_root_value=arguments.split_root,
            shuffle_root_value=arguments.shuffle_root,
            marker_sha256=arguments.attempt_sha256,
            seal_sha256=arguments.seal_sha256,
            phase_a_result_sha256=arguments.phase_a_result_sha256,
            phase_a_audit_sha256=arguments.phase_a_audit_sha256,
            output_path=arguments.worker_output,
        )
    elif arguments.command == "_focused-smoke-worker":
        result = run_focused_smoke(arguments.output)
    else:  # pragma: no cover - argparse guarantees exhaustiveness.
        raise AssertionError(arguments.command)
    if arguments.command in {
        "_phase-a-worker",
        "_phase-b-worker",
        "_focused-smoke-worker",
    }:
        output_path = (
            arguments.worker_output if hasattr(arguments, "worker_output") else arguments.output
        )
        display: Mapping[str, Any] = {
            "status": result.get("status"),
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        }
    else:
        display = result
    print(canonical_bytes(display).decode("utf-8"))
    return 0 if result.get("status") != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
