#!/usr/bin/env python3
"""Preregistered Stage-1 fit-time appearance-parameterization experiment.

This module deliberately separates the outcome-free seal from the once-only scientific run.
The official protocol is frozen in ``20260716_stage1_fit_parameterization_PREREG.md``.  It is
not a general benchmark API: names, seeds, ordering, checkpoints, environment, and gates are
hard-coded so that a later scientist can recompute the result from the raw NPZ archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.metrics import psnr, ssim
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import (
    FitConfig,
    NativeFitDiagnostic,
    NonFiniteRawParameterError,
    fit_image_from_initialization,
    init_gaussians_2d,
)
from rtgs.image2gs.renderer2d import render_gaussians_2d

ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = Path("benchmarks/results/20260716_stage1_fit_parameterization_PREREG.md")
PREREGISTRATION_SHA256 = "d1440fde596667fd59e996113dd4ffa4414e23e8c783a401343d7476f00afb22"
IMPLEMENTATION_REVIEW = Path(
    "benchmarks/results/20260716_stage1_fit_parameterization_IMPLEMENTATION_REVIEW.md"
)
HARNESS = Path("benchmarks/stage1_fit_parameterization.py")
FOCUSED_TESTS = Path("tests/test_stage1_fit_parameterization.py")
FIT_SEAM = Path("src/rtgs/image2gs/fit.py")
SEAM_TESTS = Path("tests/test_stage1_fit_seam.py")
DEFAULT_SEAL = ROOT / "benchmarks/results/20260716_stage1_fit_parameterization_SEAL.json"
ATTEMPT = ROOT / "benchmarks/results/20260716_stage1_fit_parameterization_ATTEMPT.json"

ARTIFACT_TYPE = "stage1_fit_parameterization_result"
INVALID_ARTIFACT_TYPE = "stage1_fit_parameterization_invalid_attempt"
SEAL_ARTIFACT_TYPE = "stage1_fit_parameterization_implementation_seal"
ATTEMPT_ARTIFACT_TYPE = "stage1_fit_parameterization_once_only_attempt"
RAW_SCHEMA = "stage1_fit_parameterization_raw_v1"

ARMS = ("weight_color_9p", "unit_weight_bounded_8p")
BLOCKS = ("appearance_only", "joint")
BLOCK_SEEDS = {
    "appearance_only": (7727, 8837, 9941),
    "joint": (10007, 11003, 12007),
}
SELECTED_VIEWS = (0, 1, 2, 4, 5, 6, 8, 9, 10)
CHECKPOINTS = (0, 1, 5, 10, 20, 40, 80, 120)
HISTORY_ITERATIONS = (0, 50, 100, 119)
POSITIVE_CHECKPOINTS = CHECKPOINTS[1:]
COMPONENTS = 150
UPDATES = 120
IMAGE_SIZE = 48
HISTOGRAM_EDGES = (0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)
CHAIN_RULE_ABSOLUTE_LIMIT = 2e-6
CHAIN_RULE_RELATIVE_LIMIT = 1e-4
CHAIN_RULE_RELATIVE_MAGNITUDE_THRESHOLD = 1e-5
PREREGISTRATION_LAST_AMENDMENT_UTC = datetime(2026, 7, 16, 8, 45, 22, tzinfo=timezone.utc)
EXPECTED_TORCH_VERSION = "2.9.0+cu128"
EXPECTED_PYTHON_EXECUTABLE = str((ROOT / ".venv/bin/python").absolute())
EXPECTED_PYTHON_EXECUTABLE_RESOLVED = str((ROOT / ".venv/bin/python").resolve())
EXPECTED_PYTHON_PREFIX = str((ROOT / ".venv").absolute())
PROTOCOL_CARDINALITIES = {
    "blocks": 2,
    "scenes": 6,
    "initializers": 54,
    "paired_views": 54,
    "fits": 108,
    "updates": 12_960,
    "checkpoints": 864,
    "callback_events": 26_784,
    "checkpoint_component_rows": 129_600,
    "view_metric_cells": 864,
    "seed_arm_checkpoint_aggregates": 96,
    "seed_arm_aucs": 12,
    "mechanism_fits": 54,
    "mechanism_updates": 6_480,
    "mechanism_component_update_rows": 972_000,
    "current_null_row_universe": 486_000,
    "weak_rows_per_mechanism_arm": 32_400,
    "weak_rows_per_seed_arm": 10_800,
    "joint_fits": 54,
    "joint_positive_checkpoint_updates": 378,
}

# The harness is complete only when this flag is true and the explicit gap tuple is empty. Seal
# creation still requires an independently authored implementation-review artifact; scientific
# execution additionally requires the exact process CLI and the verified seal.
IMPLEMENTATION_COMPLETE = True
IMPLEMENTATION_READY = IMPLEMENTATION_COMPLETE
IMPLEMENTATION_GAPS: tuple[str, ...] = ()
_CLI_AUTHORIZATION: tuple[str, Path | None, Path | None] | None = None
TARGET_GENERATOR_SOURCE_PATHS = (
    Path("src/rtgs/core/camera.py"),
    Path("src/rtgs/core/gaussians3d.py"),
    Path("src/rtgs/core/sh.py"),
    Path("src/rtgs/data/scene.py"),
    Path("src/rtgs/data/synthetic.py"),
    Path("src/rtgs/render/base.py"),
    Path("src/rtgs/render/torch_ref.py"),
)
FROZEN_SOURCE_HASHES = {
    "src/rtgs/image2gs/fit.py": (
        "2a9b76d41e83cc444fa98b3a0f3aa45eb8b6032806fa3d899377acfd98257e18"
    ),
    "src/rtgs/image2gs/renderer2d.py": (
        "d0bd6b90b8a690a2ebb36cbc55c8cceb56c3fc33c04fd3895a123e0abb660144"
    ),
    "src/rtgs/core/gaussians2d.py": (
        "390c6940bea8f4f1c80df19396a38ee29585dfd3127c8a3823654ffe09098351"
    ),
    "src/rtgs/data/synthetic.py": (
        "b2b16f02a92c89003439062085e39d1f5ced2cc9ebaf5b8874cf80c0fd4d70b2"
    ),
    "src/rtgs/core/metrics.py": (
        "d489c07c65ac4c74f0f927d41c62b887724cf3216f2ef28a116ff169d08272d4"
    ),
    "src/rtgs/render/torch_ref.py": (
        "61716787329e85a186982f81c2a89cb270255473ca26688c409191a1b53bd86e"
    ),
    "src/rtgs/core/camera.py": ("1e6a42c7cd9fa14b2ffff19808e6e88c106df4562d30fc18b0ca107c00072ac2"),
    "src/rtgs/core/gaussians3d.py": (
        "d417a4a103ae7ea1e3f4a7799c2b709597014b8966acb0e72b2bd447a0ad0ba5"
    ),
    "src/rtgs/core/sh.py": ("554f3a25e25c7312248a98c15685e9bf805c85a81a96f56e13e1481619eb4687"),
    "src/rtgs/data/scene.py": ("3fa557f03bab5eb7666476968e0a70ff3e5639d6e24251807905691df36004c3"),
    "src/rtgs/render/base.py": ("1175cf359e2800ff3a518849b43c4d9a6fd6dccc3dfb7c24459f13e9f81ca0b9"),
}


class ProtocolInvalid(RuntimeError):
    """Fail-closed protocol error with a precise phase and reached evidence."""

    def __init__(
        self,
        phase: str,
        message: str,
        evidence: Mapping[str, Any] | None = None,
        *,
        raw_arrays: Mapping[str, torch.Tensor | np.ndarray] | None = None,
    ):
        super().__init__(message)
        self.phase = phase
        self.reason = message
        self.evidence = finite_json_evidence(dict(evidence or {}))
        self.raw_arrays = {
            str(name): (
                value.detach().contiguous().cpu().numpy().copy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value).copy()
            )
            for name, value in (raw_arrays or {}).items()
        }


def finite_json_evidence(value: Any) -> Any:
    """Convert failure evidence to strict, finite, deterministic JSON values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().contiguous().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): finite_json_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json_evidence(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        classification = 1 if math.isnan(value) else (2 if value > 0 else 3)
        return {"value": 0.0, "nonfinite_classification": classification}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_artifact_utc(value: Any, *, label: str) -> datetime:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", value) is None
    ):
        raise RuntimeError(f"{label} timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError(f"{label} timestamp is malformed") from error
    if parsed.tzinfo != timezone.utc:
        raise RuntimeError(f"{label} timestamp is not UTC")
    return parsed


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def little_endian_array(value: np.ndarray) -> np.ndarray:
    """Return the C-contiguous value used by the frozen semantic array hash."""
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise TypeError(f"raw arrays must be numeric or boolean, got {array.dtype}")
    target = array.dtype.newbyteorder("<") if array.dtype.itemsize > 1 else array.dtype
    converted = array.astype(target, copy=False)
    # np.ascontiguousarray promotes a 0-D scalar to shape (1,), which would silently violate
    # every scalar field in the frozen table schema and semantic hash contract.
    return converted.copy() if converted.ndim == 0 else np.ascontiguousarray(converted)


def array_content_sha256(value: np.ndarray) -> str:
    array = little_endian_array(value)
    token = np.dtype(array.dtype).newbyteorder("<").str.encode("ascii")
    shape = np.asarray(array.shape, dtype="<i8").tobytes(order="C")
    return sha256_bytes(token + b"\0" + shape + b"\0" + array.tobytes(order="C"))


def array_manifest(arrays: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    digest_pairs: list[list[str]] = []
    for name in sorted(arrays):
        value = little_endian_array(arrays[name])
        digest = array_content_sha256(value)
        entries.append(
            {
                "name": name,
                "dtype": np.dtype(value.dtype).str,
                "shape": list(value.shape),
                "byte_length": int(value.nbytes),
                "content_sha256": digest,
            }
        )
        digest_pairs.append([name, digest])
    collection = sha256_bytes(
        json.dumps(digest_pairs, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    return entries, collection


def nonfinite_classification(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    mask = np.zeros(array.shape, dtype=np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        mask[np.isnan(array)] = 1
        mask[np.isposinf(array)] = 2
        mask[np.isneginf(array)] = 3
    return mask


def _validate_raw_logical_name(name: Any, *, allow_classification: bool) -> str:
    segments = name.split("/") if isinstance(name, str) else []
    if (
        not isinstance(name, str)
        or not name
        or name == "file"
        or name.startswith("/")
        or name.endswith("/")
        or "//" in name
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or any(
            segment in {"", ".", ".."}
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", segment) is None
            for segment in segments
        )
    ):
        raise ValueError(f"invalid raw-array logical name: {name!r}")
    if not allow_classification and name.endswith("/nonfinite_classification"):
        raise ValueError("non-finite classification names are reserved")
    return name


class RawArchive:
    """Append-only, name-unique in-memory archive for one official attempt."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.phase = "preflight"
        self.completed_blocks = 0
        self.completed_seeds = 0
        self.completed_views = 0
        self.completed_initializers = 0
        self.completed_paired_views = 0
        self.completed_fits = 0
        self.completed_updates = 0
        self.completed_checkpoints = 0
        self.completed_callback_events = 0
        self.completed_mechanism_updates = 0
        self.completed_joint_positive_checkpoint_updates = 0
        self.completed_checkpoint_component_rows = 0
        self.completed_null_rows = 0

    def add(
        self, name: str, value: torch.Tensor | np.ndarray | Iterable[Any] | int | float | bool
    ) -> np.ndarray:
        try:
            _validate_raw_logical_name(name, allow_classification=False)
        except ValueError as error:
            raise ProtocolInvalid(self.phase, str(error)) from error
        classification_name = f"{name}/nonfinite_classification"
        if name in self.arrays or classification_name in self.arrays:
            raise ProtocolInvalid(self.phase, f"raw-array name reused: {name}")
        if isinstance(value, torch.Tensor):
            value = value.detach().contiguous().cpu().numpy()
        try:
            array = little_endian_array(np.asarray(value))
        except (TypeError, ValueError) as error:
            raise ProtocolInvalid(self.phase, f"unsupported raw array {name}: {error}") from error
        self.arrays[name] = array.copy()
        classification = nonfinite_classification(array)
        if bool(classification.any()):
            self.arrays[classification_name] = classification
            raise ProtocolInvalid(self.phase, f"non-finite raw array: {name}")
        return self.arrays[name]

    def add_nullable(self, name: str, value: Any, defined: bool) -> None:
        try:
            _validate_raw_logical_name(name, allow_classification=False)
        except ValueError as error:
            raise ProtocolInvalid(self.phase, str(error)) from error
        if not isinstance(defined, bool):
            raise ProtocolInvalid(self.phase, "nullable defined flag must be boolean")
        numeric = 0 if not defined else value
        self.add(f"{name}/value", numeric)
        self.add(f"{name}/defined", np.asarray(defined, dtype=np.bool_))

    def completion_array_values(self, *, phase: str | None = None) -> dict[str, np.ndarray]:
        values = {
            "completion/phase_code": np.asarray(
                _phase_code(self.phase if phase is None else phase), dtype=np.int64
            )
        }
        for name in (
            "completed_blocks",
            "completed_seeds",
            "completed_views",
            "completed_initializers",
            "completed_paired_views",
            "completed_fits",
            "completed_updates",
            "completed_checkpoints",
            "completed_callback_events",
            "completed_mechanism_updates",
            "completed_joint_positive_checkpoint_updates",
            "completed_checkpoint_component_rows",
            "completed_null_rows",
        ):
            values[f"completion/{name}"] = np.asarray(getattr(self, name), dtype=np.int64)
        return values

    def completion_arrays(self, *, phase: str | None = None) -> None:
        for name, value in self.completion_array_values(phase=phase).items():
            self.add(name, value)


def _phase_code(phase: str) -> int:
    names = (
        "preflight",
        "scene",
        "initialization",
        "equivalence",
        "appearance_only",
        "joint",
        "metrics",
        "reduction",
        "serialization",
        "complete",
        "invalid",
    )
    return names.index(phase) if phase in names else -1


def frozen_fit_config(arm: str) -> FitConfig:
    if arm not in ARMS:
        raise ValueError(f"unknown appearance arm: {arm}")
    return FitConfig(
        n_gaussians=COMPONENTS,
        max_gaussians=5_000,
        iterations=UPDATES,
        backend="native",
        adaptive_density=True,
        growth_waves=5,
        relocate_fraction=0.0,
        structsplat_renderer="auto",
        lr=0.01,
        grad_init_mix=0.7,
        row_chunk=64,
        log_every=50,
        convergence_patience=0,
        convergence_tol=0.05,
        convergence_check_every=25,
        appearance_parameterization=arm,
    )


def fit_config_contract() -> dict[str, dict[str, dict[str, Any]]]:
    """Serialize the literal block-by-arm configurations used by the official run."""
    contract: dict[str, dict[str, dict[str, Any]]] = {}
    for block in BLOCKS:
        contract[block] = {}
        for arm in ARMS:
            config = frozen_fit_config(arm)
            config.freeze_geometry = block == "appearance_only"
            contract[block][arm] = asdict(config)
    return contract


def optimizer_scheduler_contract() -> dict[str, Any]:
    """Return the exact effective Adam/Cosine configuration frozen by the protocol."""
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.Adam(
        [parameter],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
        decoupled_weight_decay=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UPDATES, eta_min=0.001, last_epoch=-1
    )
    defaults = {key: value for key, value in optimizer.defaults.items() if key != "params"}
    group = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    return finite_json_evidence(
        {
            "optimizer_class": "torch.optim.Adam",
            "optimizer_defaults": defaults,
            "optimizer_param_group_without_params": group,
            "scheduler_class": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "scheduler_state_at_construction": scheduler.state_dict(),
        }
    )


def arm_order(seed_position: int) -> tuple[str, str]:
    return ARMS if seed_position in (0, 2) else tuple(reversed(ARMS))


def softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def common_raw_initialization(g0: Gaussians2D, height: int, width: int) -> dict[str, torch.Tensor]:
    wh = g0.xy.new_tensor([width, height])
    xy_raw = torch.logit((g0.xy / wh).clamp(1e-4, 1 - 1e-4))
    diag_raw = softplus_inverse(g0.chol[:, [0, 2]] - 0.3)
    off_raw = g0.chol[:, 1].clone()
    u = torch.logit(g0.color.clamp(1e-3, 1 - 1e-3))
    s = torch.logit(g0.weight.clamp(1e-3, 1 - 1e-3))
    w = torch.sigmoid(s)
    c = torch.sigmoid(u)
    amplitude = w[:, None] * c
    r = torch.logit(amplitude)
    return {
        "xy_raw": xy_raw,
        "diag_raw": diag_raw,
        "off_raw": off_raw,
        "u": u,
        "s": s,
        "r": r,
        "amplitude": amplitude,
    }


def build_from_raw(
    raw: Mapping[str, torch.Tensor], arm: str, height: int, width: int
) -> Gaussians2D:
    wh = raw["xy_raw"].new_tensor([width, height])
    diag = torch.nn.functional.softplus(raw["diag_raw"]) + 0.3
    chol = torch.stack([diag[:, 0], raw["off_raw"], diag[:, 1]], dim=-1)
    if arm == "weight_color_9p":
        color = torch.sigmoid(raw["u"])
        weight = torch.sigmoid(raw["s"])
    elif arm == "unit_weight_bounded_8p":
        color = torch.sigmoid(raw["r"])
        weight = torch.ones(raw["r"].shape[0], dtype=raw["r"].dtype, device=raw["r"].device)
    else:
        raise ValueError(arm)
    return Gaussians2D(
        xy=torch.sigmoid(raw["xy_raw"]) * wh,
        chol=chol,
        color=color,
        weight=weight,
    )


def current_jacobian(weight: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 1 or color.ndim != 2 or color.shape != (weight.shape[0], 3):
        raise ValueError("current Jacobian requires weight (N,) and color (N,3)")
    if (
        weight.numel() == 0
        or not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(color).all())
    ):
        raise ValueError("current Jacobian inputs must be nonempty and finite")
    w = weight.to(torch.float64).reshape(-1, 1)
    c = color.to(torch.float64)
    diagonal = torch.diag_embed(w * c * (1.0 - c))
    scalar = (w * (1.0 - w) * c).unsqueeze(-1)
    return torch.cat([diagonal, scalar], dim=-1)


def candidate_jacobian(amplitude: torch.Tensor) -> torch.Tensor:
    if amplitude.ndim != 2 or amplitude.shape[1] != 3 or amplitude.shape[0] == 0:
        raise ValueError("candidate Jacobian requires amplitude (N,3)")
    if not bool(torch.isfinite(amplitude).all()):
        raise ValueError("candidate Jacobian input must be finite")
    a = amplitude.to(torch.float64)
    return torch.diag_embed(a * (1.0 - a))


def jacobian_diagnostics(jacobian: torch.Tensor, *, current: bool) -> dict[str, torch.Tensor]:
    value = jacobian.to(torch.float64)
    expected_tail = (3, 4) if current else (3, 3)
    if value.ndim != 3 or tuple(value.shape[-2:]) != expected_tail or value.shape[0] == 0:
        raise ValueError(f"Jacobian diagnostics require (N,{expected_tail[0]},{expected_tail[1]})")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Jacobian diagnostics require finite values")
    _u, singular, vh = torch.linalg.svd(value, full_matrices=True)
    # Singular-vector signs are mathematically arbitrary. Canonicalize every Vh row by making
    # its largest-magnitude coordinate nonnegative so recorded vectors and their signed dot
    # products have a deterministic representation while sign-invariant conclusions are kept.
    pivots = vh.abs().argmax(dim=-1, keepdim=True)
    pivot_values = torch.gather(vh, dim=-1, index=pivots)
    signs = torch.where(
        pivot_values < 0, -torch.ones_like(pivot_values), torch.ones_like(pivot_values)
    )
    vh = vh * signs
    largest = singular[:, 0]
    tau = max(value.shape[-2:]) * torch.finfo(torch.float64).eps * largest
    rank = (singular > tau[:, None]).sum(dim=-1)
    positive = torch.where(singular > tau[:, None], singular, torch.inf)
    smallest = positive.min(dim=-1).values
    has_positive = rank > 0
    smallest = torch.where(has_positive, smallest, torch.zeros_like(smallest))
    condition_defined = rank == 3
    condition = torch.where(
        condition_defined, largest / smallest.clamp_min(torch.finfo(torch.float64).tiny), 0.0
    )
    weak = (rank < 3) | (smallest < 1e-4)
    result = {
        "jacobian": value,
        "singular_values": singular,
        "vh": vh,
        "largest": largest,
        "tau": tau,
        "rank": rank,
        "smallest_positive": smallest,
        "smallest_positive_defined": has_positive,
        "condition": condition,
        "condition_defined": condition_defined,
        "weakly_responsive": weak,
    }
    if current:
        result["null_vector"] = vh[:, -1, :]
        result["null_residual"] = torch.linalg.vector_norm(value @ vh[:, -1, :, None], dim=(-2, -1))
    return result


def analytic_current_null(
    weight: torch.Tensor, color: torch.Tensor, svd_null: torch.Tensor, rank: torch.Tensor
) -> dict[str, torch.Tensor]:
    w = weight.to(torch.float64).reshape(-1, 1)
    c = color.to(torch.float64)
    defined = ((1.0 - c) > 0).all(dim=-1) & (rank == 3)
    vector = torch.cat([-(1.0 - w) / (1.0 - c), torch.ones_like(w)], dim=-1)
    vector = torch.nn.functional.normalize(vector, dim=-1)
    alignment = (vector * svd_null).sum(dim=-1).abs()
    vector = torch.where(defined[:, None], vector, torch.zeros_like(vector))
    alignment = torch.where(defined, alignment, torch.zeros_like(alignment))
    return {"vector": vector, "alignment": alignment, "defined": defined}


def histogram_counts(value: torch.Tensor) -> torch.Tensor:
    flat = value.to(torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError("histogram values must be nonempty and finite")
    edges = torch.tensor(HISTOGRAM_EDGES, dtype=torch.float64, device=flat.device)
    if bool(((flat < edges[0]) | (flat > edges[-1])).any()):
        raise ValueError("histogram value outside [0,1]")
    bins = torch.bucketize(flat, edges[1:-1], right=True)
    counts = torch.bincount(bins, minlength=len(HISTOGRAM_EDGES) - 1)
    if int(counts.sum()) != flat.numel():
        raise AssertionError("histogram did not assign every value exactly once")
    return counts


def saturation_diagnostics(raw: torch.Tensor, output: torch.Tensor) -> dict[str, torch.Tensor]:
    raw64 = raw.to(torch.float64)
    out64 = output.to(torch.float64)
    if raw64.numel() == 0 or out64.numel() == 0:
        raise ValueError("saturation diagnostics require nonempty fields")
    if not bool(torch.isfinite(raw64).all()) or not bool(torch.isfinite(out64).all()):
        raise ValueError("saturation diagnostics require finite fields")
    if not bool(((out64 >= 0) & (out64 <= 1)).all()):
        raise ValueError("saturation output must lie in [0,1]")
    derivative = out64 * (1.0 - out64)
    denominator_raw = max(raw64.numel(), 1)
    denominator_out = max(out64.numel(), 1)
    return {
        "raw_abs_ge_8_count": (raw64.abs() >= 8).sum(),
        "raw_count": torch.tensor(denominator_raw, dtype=torch.int64),
        "raw_abs_ge_8_fraction": (raw64.abs() >= 8).sum(dtype=torch.float64) / denominator_raw,
        "output_low_count": (out64 <= 1e-3).sum(),
        "output_high_count": (out64 >= 1.0 - 1e-3).sum(),
        "output_count": torch.tensor(denominator_out, dtype=torch.int64),
        "output_low_fraction": (out64 <= 1e-3).sum(dtype=torch.float64) / denominator_out,
        "output_high_fraction": (out64 >= 1.0 - 1e-3).sum(dtype=torch.float64) / denominator_out,
        "derivative_le_1e_4_count": (derivative <= 1e-4).sum(),
        "derivative_le_1e_4_fraction": (derivative <= 1e-4).sum(dtype=torch.float64)
        / denominator_out,
    }


def render_metrics(
    prediction: torch.Tensor, target: torch.Tensor, *, phase: str = "metrics"
) -> dict[str, Any]:
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise ProtocolInvalid(phase, "render and target must be float32")
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ProtocolInvalid(
            phase,
            "render and target must have one identical nonempty (H,W,3) shape",
            {"prediction_shape": tuple(prediction.shape), "target_shape": tuple(target.shape)},
        )
    if prediction.numel() == 0:
        raise ProtocolInvalid(phase, "render and target must be nonempty")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raw_arrays = {}
        if not bool(torch.isfinite(prediction).all()):
            raw_arrays["prediction"] = prediction
        if not bool(torch.isfinite(target).all()):
            raw_arrays["target"] = target
        raise ProtocolInvalid(phase, "render and target must be finite", raw_arrays=raw_arrays)
    if not bool(((target >= 0) & (target <= 1)).all()):
        raise ProtocolInvalid(phase, "target must lie in [0,1]")
    difference64 = prediction.to(torch.float64) - target.to(torch.float64)
    sse = difference64.square().sum()
    count = prediction.numel()
    mse64 = sse / count
    clamped = prediction.clamp(0, 1)
    below = prediction < 0
    above = prediction > 1
    return {
        "sse": float(sse),
        "channel_count": int(count),
        "raw_mse_float64": float(mse64),
        "objective_loss_float32": float(torch.nn.functional.mse_loss(prediction, target)),
        "psnr": psnr(clamped, target),
        "ssim": float(ssim(clamped, target)),
        "below_count": int(below.sum()),
        "above_count": int(above.sum()),
        "below_fraction": float(below.sum(dtype=torch.float64) / count),
        "above_fraction": float(above.sum(dtype=torch.float64) / count),
        "clamped": clamped,
        "below_mask": below,
        "above_mask": above,
    }


def normalized_trapezoid_auc(values: Iterable[float]) -> float:
    points = [float(value) for value in values]
    if len(points) != len(CHECKPOINTS):
        raise ValueError("AUC requires the exact frozen checkpoints")
    if not all(math.isfinite(value) for value in points):
        raise ValueError("AUC values must be finite")
    total = 0.0
    for left, right, value_left, value_right in zip(
        CHECKPOINTS[:-1], CHECKPOINTS[1:], points[:-1], points[1:], strict=True
    ):
        total += (right - left) * (value_left + value_right) / 2.0
    return total / UPDATES


def chain_rule_expected(
    arm: str, grad_amplitude: torch.Tensor, weight: torch.Tensor, color: torch.Tensor
) -> dict[str, torch.Tensor]:
    if arm == "weight_color_9p":
        grad_u = grad_amplitude * weight[:, None] * color * (1.0 - color)
        grad_s = (grad_amplitude * weight[:, None] * (1.0 - weight[:, None]) * color).sum(dim=-1)
        return {"u": grad_u, "s": grad_s}
    if arm == "unit_weight_bounded_8p":
        return {"r": grad_amplitude * color * (1.0 - color)}
    raise ValueError(arm)


def error_pair(
    actual: torch.Tensor, expected: torch.Tensor, threshold: float = 1e-8
) -> tuple[float, float]:
    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError("error pair requires identical nonempty shapes")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise ValueError("error pair requires finite tensors")
    difference = (actual - expected).abs()
    maximum_absolute = float(difference.max()) if difference.numel() else 0.0
    eligible = expected.abs() > threshold
    maximum_relative = (
        float((difference[eligible] / expected[eligible].abs()).max())
        if bool(eligible.any())
        else 0.0
    )
    return maximum_absolute, maximum_relative


def adam_reconstruct(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_before: int,
    lr: float,
) -> dict[str, torch.Tensor | int]:
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step_after = step_before + 1
    # Match torch.optim.adam._single_tensor_adam's CPU operation order, including its in-place
    # lerp/mul/addcmul rounding.  Algebraically equivalent expressions are not bit-equivalent.
    next_avg = exp_avg.clone()
    next_avg.lerp_(gradient, 1.0 - beta1)
    next_sq = exp_avg_sq.clone()
    next_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
    correction1 = 1.0 - beta1**step_after
    correction2 = 1.0 - beta2**step_after
    step_size = lr / correction1
    denominator = (next_sq.sqrt() / math.sqrt(correction2)).add_(eps)
    parameter_after = parameter.clone()
    parameter_after.addcdiv_(next_avg, denominator, value=-step_size)
    displacement = parameter_after - parameter
    return {
        "exp_avg_after": next_avg,
        "exp_avg_sq_after": next_sq,
        "step_after": step_after,
        "displacement": displacement,
        "parameter_after": parameter_after,
    }


def null_update_rows(
    jacobian: torch.Tensor, gradient: torch.Tensor, displacement: torch.Tensor
) -> dict[str, torch.Tensor]:
    diagnostics = jacobian_diagnostics(jacobian, current=True)
    null = diagnostics["null_vector"]
    displacement64 = displacement.to(torch.float64)
    gradient64 = gradient.to(torch.float64)
    dot = (null * displacement64).sum(dim=-1)
    update_norm = torch.linalg.vector_norm(displacement64, dim=-1)
    grad_dot = (null * gradient64).sum(dim=-1)
    grad_norm = torch.linalg.vector_norm(gradient64, dim=-1)
    eligible = (diagnostics["rank"] == 3) & (update_norm > 1e-12)
    fraction = torch.where(
        eligible, dot.abs() / update_norm.clamp_min(torch.finfo(torch.float64).tiny), 0.0
    )
    cosine_defined = grad_norm > 1e-12
    cosine = torch.where(
        cosine_defined, grad_dot.abs() / grad_norm.clamp_min(torch.finfo(torch.float64).tiny), 0.0
    )
    return {
        "null_vector": null,
        "rank": diagnostics["rank"],
        "dot": dot,
        "update_norm": update_norm,
        "eligible": eligible,
        "null_fraction": fraction,
        "squared_projection": dot.square(),
        "squared_update_norm": update_norm.square(),
        "gradient_dot": grad_dot,
        "gradient_norm": grad_norm,
        "gradient_cosine": cosine,
        "gradient_cosine_defined": cosine_defined,
    }


def pooled_null(rows: Iterable[Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    eligible_count = 0
    large_count = 0
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        eligible = row["eligible"].to(torch.bool)
        eligible_count += int(eligible.sum())
        large_count += int((row["null_fraction"][eligible] >= 0.10).sum())
        numerator += float(row["squared_projection"][eligible].sum(dtype=torch.float64))
        denominator += float(row["squared_update_norm"][eligible].sum(dtype=torch.float64))
    if not all(math.isfinite(value) for value in (numerator, denominator)):
        raise ProtocolInvalid("reduction", "non-finite null-pool reduction")
    defined = eligible_count > 0 and denominator > 0
    return {
        "eligible_count": eligible_count,
        "large_count": large_count,
        "projection_energy_numerator": numerator,
        "update_energy_denominator": denominator,
        "null_energy_ratio": numerator / denominator if defined else 0.0,
        "null_large_fraction": large_count / eligible_count if eligible_count else 0.0,
        "defined": defined,
    }


def _finite_triplet(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three seed values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} contains a non-numeric seed value")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} contains a non-finite seed value")
        result.append(number)
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a JSON numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _bounded_number(value: Any, name: str, lower: float, upper: float) -> float:
    number = _finite_number(value, name)
    if number < lower or number > upper:
        raise ValueError(f"{name} must lie in [{lower},{upper}]")
    return number


def frozen_decisions(
    summary: Mapping[str, Any], *, global_validity_passed: bool
) -> dict[str, bool]:
    """Apply every frozen decision conjunct to one strictly shaped three-seed summary."""
    if not isinstance(global_validity_passed, bool):
        raise ValueError("global_validity_passed must be boolean")
    if not isinstance(summary, Mapping) or not {"appearance_only", "joint"}.issubset(summary):
        raise ValueError("summary must contain appearance_only and joint mappings")
    mechanism = summary["appearance_only"]
    joint = summary["joint"]
    if not isinstance(mechanism, Mapping) or not isinstance(joint, Mapping):
        raise ValueError("appearance_only and joint summaries must be mappings")
    d_auc = _finite_triplet(mechanism.get("delta_auc_by_seed"), "appearance delta AUC")
    d_final = _finite_triplet(
        mechanism.get("delta_final_psnr_by_seed"), "appearance final PSNR delta"
    )
    d_ssim = _finite_triplet(
        mechanism.get("delta_final_ssim_by_seed"), "appearance final SSIM delta"
    )
    curve = global_validity_passed and (
        sum(d_auc) / 3 >= 0.10
        and sum(value >= 0.05 for value in d_auc) >= 2
        and min(d_auc) >= -0.10
        and sum(d_final) / 3 >= -0.05
        and min(d_final) >= -0.15
        and sum(d_ssim) / 3 >= -0.002
        and min(d_ssim) >= -0.010
    )
    null_global = mechanism.get("null_global")
    null_seed = mechanism.get("null_by_seed")
    if not isinstance(null_global, Mapping) or not isinstance(null_seed, (list, tuple)):
        raise ValueError("null summaries must contain one global mapping and three seed mappings")
    if len(null_seed) != 3 or any(not isinstance(item, Mapping) for item in null_seed):
        raise ValueError("null_by_seed must contain exactly three mappings")
    global_defined = null_global.get("defined")
    if not isinstance(global_defined, bool):
        raise ValueError("null_global.defined must be boolean")
    null_global_ratio = _bounded_number(
        null_global.get("null_energy_ratio"),
        "null_global.null_energy_ratio",
        0.0,
        1.0 + 1e-12,
    )
    null_global_large = _bounded_number(
        null_global.get("null_large_fraction"),
        "null_global.null_large_fraction",
        0.0,
        1.0,
    )
    if not global_defined and (null_global_ratio != 0.0 or null_global_large != 0.0):
        raise ValueError("undefined global null statistics must use exact zero placeholders")
    seed_null_material = []
    for index, item in enumerate(null_seed):
        defined = item.get("defined")
        if not isinstance(defined, bool):
            raise ValueError(f"null_by_seed[{index}].defined must be boolean")
        ratio = _bounded_number(
            item.get("null_energy_ratio"),
            f"null_by_seed[{index}].null_energy_ratio",
            0.0,
            1.0 + 1e-12,
        )
        if not defined and ratio != 0.0:
            raise ValueError(f"undefined null_by_seed[{index}] must use an exact zero placeholder")
        seed_null_material.append(defined and ratio >= 0.005)
    null_material = global_validity_passed and (
        global_defined
        and null_global_ratio >= 0.01
        and null_global_large >= 0.10
        and sum(seed_null_material) >= 2
    )
    weak_global = _bounded_number(
        mechanism.get("weak_fraction_delta_global"), "weak_fraction_delta_global", -1.0, 1.0
    )
    weak_seed = _finite_triplet(
        mechanism.get("weak_fraction_delta_by_seed"), "weak_fraction_delta_by_seed"
    )
    if any(value < -1.0 or value > 1.0 for value in weak_seed):
        raise ValueError("weak_fraction_delta_by_seed values must lie in [-1,1]")
    saturation = global_validity_passed and weak_global <= 0.05 and max(weak_seed) <= 0.10
    j_psnr = _finite_triplet(joint.get("delta_final_psnr_by_seed"), "joint final PSNR delta")
    j_ssim = _finite_triplet(joint.get("delta_final_ssim_by_seed"), "joint final SSIM delta")
    j_auc = _finite_triplet(joint.get("delta_auc_by_seed"), "joint delta AUC")
    noninferior = global_validity_passed and (
        sum(j_psnr) / 3 >= -0.10
        and sum(value >= -0.10 for value in j_psnr) >= 2
        and min(j_psnr) >= -0.30
        and sum(j_ssim) / 3 >= -0.002
        and min(j_ssim) >= -0.010
        and sum(j_auc) / 3 >= -0.10
        and min(j_auc) >= -0.30
    )
    material = global_validity_passed and (
        noninferior
        and sum(j_psnr) / 3 >= 0.10
        and sum(value >= 0.05 for value in j_psnr) >= 2
        and min(j_psnr) >= -0.10
        and sum(j_auc) / 3 >= 0.10
        and sum(j_ssim) / 3 >= 0.0
    )
    return {
        "appearance_curve_improved": curve,
        "null_update_material": null_material,
        "candidate_saturation_guard_passed": saturation,
        "fit_time_redundant_coordinate_interference_consistent": curve
        and null_material
        and saturation,
        "joint_stage1_noninferior": noninferior,
        "joint_stage1_material_improvement": material,
    }


def source_snapshot(paths: Iterable[Path]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for path in sorted(set(paths), key=str):
        absolute = ROOT / path
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        hashes[str(path)] = sha256_file(absolute)
        sizes[str(path)] = absolute.stat().st_size
    return {
        "paths": sorted(hashes),
        "sha256": hashes,
        "byte_sizes": sizes,
        "collection_sha256": canonical_json_hash(
            [[name, hashes[name], sizes[name]] for name in sorted(hashes)]
        ),
    }


def sealed_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                PREREGISTRATION,
                IMPLEMENTATION_REVIEW,
                HARNESS,
                FOCUSED_TESTS,
                Path("pyproject.toml"),
                *(path.relative_to(ROOT) for path in (ROOT / "src/rtgs").rglob("*.py")),
                *(path.relative_to(ROOT) for path in (ROOT / "tests").rglob("*.py")),
            },
            key=str,
        )
    )


def verify_preregistration() -> dict[str, str]:
    digest = sha256_file(ROOT / PREREGISTRATION)
    if digest != PREREGISTRATION_SHA256:
        raise RuntimeError(f"preregistration drift: {digest}")
    return {"path": str(PREREGISTRATION), "sha256": digest}


def verify_preimplementation_sources() -> dict[str, Any]:
    observed: dict[str, str] = {}
    for name, expected in FROZEN_SOURCE_HASHES.items():
        digest = sha256_file(ROOT / name)
        observed[name] = digest
        if name == "src/rtgs/image2gs/fit.py":
            continue
        if digest != expected:
            raise RuntimeError(
                f"non-authorized frozen source drift: {name}: {digest} != {expected}"
            )
    return {
        "expected": FROZEN_SOURCE_HASHES,
        "observed_at_seal": observed,
        "fit_only_authorized_to_differ": True,
    }


def verify_implementation_review(*, require_artifacts_absent: bool = False) -> dict[str, str]:
    path = ROOT / IMPLEMENTATION_REVIEW
    if not path.is_file():
        raise FileNotFoundError(f"missing implementation review: {path}")
    review_text = path.read_text(encoding="utf-8")

    def exact_line(label: str) -> str:
        matches = re.findall(rf"(?m)^{re.escape(label)}: ([^\r\n]+)$", review_text)
        if len(matches) != 1:
            raise RuntimeError(f"implementation review lacks unique binding: {label}")
        return matches[0]

    reviewed_paths = tuple(item for item in sealed_paths() if item != IMPLEMENTATION_REVIEW)
    reviewed_snapshot = source_snapshot(reviewed_paths)
    expected = {
        "Preregistration-SHA256": PREREGISTRATION_SHA256,
        "Frozen-Expected-Map-SHA256": canonical_json_hash(FROZEN_SOURCE_HASHES),
        "Reviewed-Source-Collection-SHA256": reviewed_snapshot["collection_sha256"],
        "Harness-SHA256": sha256_file(ROOT / HARNESS),
        "Focused-Tests-SHA256": sha256_file(ROOT / FOCUSED_TESTS),
        "Fit-Seam-SHA256": sha256_file(ROOT / FIT_SEAM),
        "Seam-Tests-SHA256": sha256_file(ROOT / SEAM_TESTS),
        "Official-Seeds-Touched": "none",
        "Official-Artifact-State": "seal=absent; attempt=absent; result=absent",
        "Verdict": "PASS",
    }
    for label, expected_value in expected.items():
        if exact_line(label) != expected_value:
            raise RuntimeError(f"implementation review binding differs: {label}")
    reviewer = exact_line("Reviewer")
    reviewed_at = exact_line("Reviewed-At-UTC")
    if not reviewer.strip():
        raise RuntimeError("implementation review reviewer identity is empty")
    try:
        parsed_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("implementation review timestamp is malformed") from error
    if parsed_time.tzinfo is None or parsed_time.utcoffset() != timezone.utc.utcoffset(parsed_time):
        raise RuntimeError("implementation review timestamp is not UTC")
    now_utc = _utc_now()
    if parsed_time < PREREGISTRATION_LAST_AMENDMENT_UTC or parsed_time > now_utc:
        raise RuntimeError("implementation review timestamp is outside review chronology")
    if require_artifacts_absent:
        generated_artifacts = [
            candidate
            for candidate in (ROOT / "benchmarks/results").glob("*")
            if candidate.is_file() and _is_generated_protocol_path(str(candidate.relative_to(ROOT)))
        ]
        if ATTEMPT.exists() or DEFAULT_SEAL.exists() or generated_artifacts:
            raise RuntimeError("implementation review absence binding is no longer true")
    return {
        "path": str(IMPLEMENTATION_REVIEW),
        "sha256": sha256_file(path),
        "reviewer": reviewer,
        "reviewed_at_utc": reviewed_at,
        "reviewed_source_collection_sha256": reviewed_snapshot["collection_sha256"],
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        # Keep both spellings: ``python_executable`` binds the repository venv entry point,
        # while the resolved spelling binds the interpreter binary behind that entry point.
        "python_executable": str(Path(sys.executable).absolute()),
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).absolute()),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "optional_backend_modules_loaded": sorted(
            name
            for name in sys.modules
            if name == "gsplat"
            or name.startswith("gsplat.")
            or name == "structsplat"
            or name.startswith("structsplat.")
            or name == "rtgs.image2gs.structsplat_backend"
        ),
    }


def official_environment_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "python",
            "python_executable",
            "python_executable_resolved",
            "python_prefix",
            "platform",
            "processor",
            "torch",
            "numpy",
            "cuda_visible_devices",
            "torch_cuda_available",
            "torch_cuda_device_count",
            "omp_num_threads",
            "mkl_num_threads",
            "torch_num_threads",
            "deterministic_algorithms",
            "deterministic_warn_only",
            "optional_backend_modules_loaded",
        )
    }


def assert_official_environment(value: Mapping[str, Any]) -> None:
    expected = {
        "python_executable": EXPECTED_PYTHON_EXECUTABLE,
        "python_executable_resolved": EXPECTED_PYTHON_EXECUTABLE_RESOLVED,
        "python_prefix": EXPECTED_PYTHON_PREFIX,
        "torch": EXPECTED_TORCH_VERSION,
        "cuda_visible_devices": "",
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "omp_num_threads": "4",
        "mkl_num_threads": "4",
        "torch_num_threads": 4,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "optional_backend_modules_loaded": [],
    }
    actual = {key: value[key] for key in expected}
    if canonical_json(actual) != canonical_json(expected):
        raise RuntimeError(f"official environment mismatch: {actual!r} != {expected!r}")


def _is_generated_protocol_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized in {
        str(DEFAULT_SEAL.relative_to(ROOT)),
        str(ATTEMPT.relative_to(ROOT)),
    }:
        return True
    relative = Path(normalized)
    if relative.parent != Path("benchmarks/results"):
        return False
    name = relative.name
    return bool(
        re.fullmatch(
            r"\d{8}T\d{6}Z_cpu_stage1_fit_parameterization"
            r"(?:_invalid)?(?:\.json|_RAW\.npz|_RESULT\.md|_AUDIT\.md|_SCIENTIST_REVIEW\.json)",
            name,
        )
    )


def git_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    filtered_status = []
    for line in status.splitlines():
        path = line[3:] if len(line) >= 4 else ""
        if not _is_generated_protocol_path(path):
            filtered_status.append(line)
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    untracked: dict[str, dict[str, Any]] = {}
    for encoded in untracked_raw.split(b"\0"):
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        if _is_generated_protocol_path(str(relative)):
            continue
        absolute = ROOT / relative
        if absolute.is_file():
            untracked[str(relative)] = {
                "sha256": sha256_file(absolute),
                "byte_size": absolute.stat().st_size,
            }
    return {
        "revision": revision,
        "dirty": bool(filtered_status),
        "status": filtered_status,
        "tracked_binary_diff": tracked_diff,
        "tracked_binary_diff_sha256": sha256_bytes(tracked_diff.encode("utf-8")),
        "untracked_files": untracked,
        "untracked_collection_sha256": canonical_json_hash(untracked),
    }


def verification_commands() -> tuple[tuple[str, ...], ...]:
    return (
        (".venv/bin/python", "-m", "ruff", "check", "."),
        (".venv/bin/python", "-m", "ruff", "format", "--check", "."),
        (".venv/bin/python", "-m", "pytest", "-q", "-m", "not slow"),
        (".venv/bin/python", "scripts/docs_sync.py"),
        ("git", "diff", "--check"),
    )


VERIFICATION_LITERAL_COMMANDS = (
    ".venv/bin/python -m ruff check .",
    ".venv/bin/python -m ruff format --check .",
    '.venv/bin/python -m pytest -q -m "not slow"',
    ".venv/bin/python scripts/docs_sync.py",
    "git diff --check",
)


def run_verification() -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    records = []
    for command, literal in zip(
        verification_commands(), VERIFICATION_LITERAL_COMMANDS, strict=True
    ):
        started = time.perf_counter()
        result = subprocess.run(
            command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
        )
        record = {
            "command": list(command),
            "literal_command": literal,
            "returncode": result.returncode,
            "seconds": time.perf_counter() - started,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "stdout_sha256": sha256_bytes(result.stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(result.stderr.encode("utf-8")),
        }
        records.append(record)
        if result.returncode != 0:
            raise RuntimeError(
                f"verification failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}"
            )
    return {"passed": True, "commands": records}


def seal_snapshot(*, require_artifacts_absent: bool = False) -> dict[str, Any]:
    paths = sealed_paths()
    return {
        "preregistration": verify_preregistration(),
        "implementation_review": verify_implementation_review(
            require_artifacts_absent=require_artifacts_absent
        ),
        "preimplementation_sources": verify_preimplementation_sources(),
        "sealed_sources": source_snapshot(paths),
        "loaded_repository_sources": loaded_repository_source_snapshot(),
        "git": git_metadata(),
        "fit_configs": fit_config_contract(),
        "optimizer_scheduler_contract": optimizer_scheduler_contract(),
        "raw_schema": raw_schema_record(),
        "command_templates": command_templates(),
    }


def create_seal() -> dict[str, Any]:
    require_implementation_ready()
    _require_cli_authorization("seal")
    environment = environment_metadata()
    assert_official_environment(environment)
    before = seal_snapshot(require_artifacts_absent=True)
    verification = run_verification()
    after = seal_snapshot(require_artifacts_absent=True)
    if before != after:
        raise RuntimeError("seal snapshot drifted during verification")
    return {
        "artifact_type": SEAL_ARTIFACT_TYPE,
        "created_at_utc": _utc_now().isoformat(timespec="seconds"),
        "environment": environment,
        **before,
        "verification": verification,
        "verification_snapshot_sha256": canonical_json_hash(before),
        "command": seal_command_record(),
    }


def raw_schema_record() -> dict[str, Any]:
    return {
        "name": RAW_SCHEMA,
        "format": "uncompressed numpy.savez",
        "allow_pickle": False,
        "name_separator": "/",
        "content_hash": (
            "sha256(dtype_token + NUL + little_endian_int64_shape + NUL + little_endian_C_bytes)"
        ),
        "nullable": "numeric value plus boolean defined mask",
        "invalid_nonfinite_codes": {"0": "finite", "1": "NaN", "2": "+Inf", "3": "-Inf"},
        "scientific": scientific_schema_record(),
    }


def command_templates() -> dict[str, list[str]]:
    return {
        "seal": [
            "CUDA_VISIBLE_DEVICES=''",
            "OMP_NUM_THREADS=4",
            "MKL_NUM_THREADS=4",
            ".venv/bin/python",
            str(HARNESS),
            "seal",
            "--output",
            str(DEFAULT_SEAL.relative_to(ROOT)),
        ],
        "run": [
            "CUDA_VISIBLE_DEVICES=''",
            "OMP_NUM_THREADS=4",
            "MKL_NUM_THREADS=4",
            ".venv/bin/python",
            str(HARNESS),
            "run",
            "--seal",
            str(DEFAULT_SEAL.relative_to(ROOT)),
            "--output",
            "benchmarks/results/<fresh-UTC>_cpu_stage1_fit_parameterization.json",
        ],
    }


def seal_command_record() -> list[str]:
    """Canonical, absolute record of the sole authorized seal invocation."""
    return [
        EXPECTED_PYTHON_EXECUTABLE,
        str((ROOT / HARNESS).absolute()),
        "seal",
        "--output",
        str(DEFAULT_SEAL.absolute()),
    ]


def run_command_record(output: Path) -> list[str]:
    """Canonical, absolute record of the sole authorized scientific invocation."""
    return [
        EXPECTED_PYTHON_EXECUTABLE,
        str((ROOT / HARNESS).absolute()),
        "run",
        "--seal",
        str(DEFAULT_SEAL.absolute()),
        "--output",
        str(output.absolute()),
    ]


def implementation_status() -> dict[str, Any]:
    ready = IMPLEMENTATION_COMPLETE and not IMPLEMENTATION_GAPS
    return {
        "ready": ready,
        "missing_clause_count": len(IMPLEMENTATION_GAPS),
        "missing_clauses": list(IMPLEMENTATION_GAPS),
        "seal_authorized": ready,
        "scientific_run_authorized": ready,
    }


def require_implementation_ready() -> None:
    if not IMPLEMENTATION_COMPLETE or IMPLEMENTATION_GAPS:
        detail = "\n- ".join(IMPLEMENTATION_GAPS)
        raise RuntimeError(
            "Stage-1 fit-parameterization harness is outcome-free but incomplete; "
            "seal/run are disabled before verification, marker creation, or scene construction.\n"
            f"- {detail}"
        )


def _require_cli_authorization(
    operation: str, *, seal_path: Path | None = None, output: Path | None = None
) -> None:
    expected = (
        operation,
        None if seal_path is None else seal_path.absolute(),
        None if output is None else output.absolute(),
    )
    if expected != _CLI_AUTHORIZATION:
        raise RuntimeError(
            f"{operation} is authorized only through the exact preregistered process CLI"
        )


def official_output_paths(output: Path) -> dict[str, Path]:
    """Derive every mutually exclusive valid/invalid path before an official attempt."""
    absolute = output.absolute()
    results = (ROOT / "benchmarks/results").absolute()
    if (
        absolute.parent != results
        or re.fullmatch(r"\d{8}T\d{6}Z_cpu_stage1_fit_parameterization\.json", absolute.name)
        is None
    ):
        raise ValueError(
            "official output must be benchmarks/results/"
            "<fresh-UTC>_cpu_stage1_fit_parameterization.json"
        )
    timestamp_text = absolute.name[:16]
    try:
        output_time = datetime.strptime(timestamp_text, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError("official output timestamp is not a real UTC instant") from error
    now = _utc_now()
    if output_time < now - timedelta(minutes=10) or output_time > now + timedelta(seconds=30):
        raise ValueError("official output timestamp is not fresh")
    stem = absolute.stem
    return {
        "valid_json": absolute,
        "valid_raw": absolute.with_name(f"{stem}_RAW.npz"),
        "valid_note": absolute.with_name(f"{stem}_RESULT.md"),
        "invalid_json": absolute.with_name(f"{stem}_invalid.json"),
        "invalid_raw": absolute.with_name(f"{stem}_invalid_RAW.npz"),
        "invalid_note": absolute.with_name(f"{stem}_invalid_RESULT.md"),
    }


def preflight_absent(paths: Iterable[Path]) -> None:
    present = [str(path) for path in paths if os.path.lexists(path)]
    if present:
        raise FileExistsError(
            f"refusing overwrite/resume; prospective paths already exist: {present}"
        )


@contextmanager
def _exclusive_binary_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def exclusive_write_bytes(path: Path, payload: bytes) -> str:
    with _exclusive_binary_writer(path) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest = sha256_file(path)
    if digest != sha256_bytes(payload):
        raise RuntimeError(f"exclusive write verification failed for {path}")
    return digest


def exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    rendered = (json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = exclusive_write_bytes(path, rendered)
    if canonical_json_hash(strict_json_load(path)) != canonical_json_hash(payload):
        raise RuntimeError(f"strict JSON round trip differs for {path}")
    return digest


def _validate_nonfinite_contract(
    arrays: Mapping[str, np.ndarray], *, invalid_evidence: bool
) -> None:
    suffix = "/nonfinite_classification"
    for name, value in arrays.items():
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if not base or base not in arrays:
                raise ValueError(f"orphan non-finite classification array: {name}")
            expected = nonfinite_classification(arrays[base])
            if (
                value.dtype != np.uint8
                or value.shape != expected.shape
                or not np.array_equal(value, expected)
            ):
                raise ValueError(f"non-finite classification differs from offending array: {name}")
            if not bool(expected.any()):
                raise ValueError(f"classification array has no offending value: {name}")
            continue
        classification = nonfinite_classification(value)
        mask_name = f"{name}{suffix}"
        if bool(classification.any()):
            if not invalid_evidence:
                raise ValueError(f"valid raw sidecar contains non-finite values: {name}")
            if mask_name not in arrays:
                raise ValueError(f"invalid raw sidecar lacks classification array: {name}")
        elif mask_name in arrays:
            raise ValueError(f"finite raw array has a classification sibling: {name}")


def _arrays_byte_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def write_raw_sidecar(
    path: Path, arrays: Mapping[str, np.ndarray], *, invalid_evidence: bool = False
) -> dict[str, Any]:
    """Write one uncompressed, pickle-free NPZ with a semantic manifest."""
    if not isinstance(invalid_evidence, bool):
        raise ValueError("invalid_evidence must be boolean")
    for name in arrays:
        _validate_raw_logical_name(name, allow_classification=True)
    normalized = {name: little_endian_array(value) for name, value in arrays.items()}
    if len(normalized) != len(arrays):  # pragma: no cover - Mapping names are unique by contract.
        raise ValueError("raw sidecar contains duplicate logical names")
    _validate_nonfinite_contract(normalized, invalid_evidence=invalid_evidence)
    manifest, collection = array_manifest(normalized)
    with _exclusive_binary_writer(path) as handle:
        np.savez(handle, **normalized)
        handle.flush()
        os.fsync(handle.fileno())
    with ZipFile(path, "r") as archive:
        members = archive.infolist()
        if len({item.filename for item in members}) != len(members):
            raise RuntimeError("raw sidecar contains duplicate ZIP members")
        if any(item.compress_type != ZIP_STORED for item in members):
            raise RuntimeError("raw sidecar is compressed")
    loaded_arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as loaded:
        if sorted(loaded.files) != sorted(normalized):
            raise RuntimeError("raw sidecar keys differ after serialization")
        for name, expected in normalized.items():
            actual = little_endian_array(loaded[name])
            loaded_arrays[name] = actual
            if not _arrays_byte_exact(actual, expected):
                raise RuntimeError(f"raw sidecar round trip differs: {name}")
    _validate_nonfinite_contract(loaded_arrays, invalid_evidence=invalid_evidence)
    recomputed_manifest, recomputed_collection = array_manifest(loaded_arrays)
    if manifest != recomputed_manifest or collection != recomputed_collection:
        raise RuntimeError("raw sidecar manifest changed during write")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "collection_sha256": collection,
        "array_count": len(normalized),
        "manifest": manifest,
        "format": "numpy.savez/uncompressed",
        "allow_pickle": False,
        "invalid_evidence": invalid_evidence,
    }


def validate_raw_sidecar(
    path: Path, binding: Mapping[str, Any], *, invalid_evidence: bool | None = None
) -> dict[str, np.ndarray]:
    if not isinstance(binding, Mapping):
        raise ValueError("raw sidecar binding must be a mapping")
    try:
        bound_path = Path(str(binding["path"])).resolve()
    except KeyError as error:
        raise ValueError("raw sidecar binding lacks path") from error
    if bound_path != path.resolve():
        raise ValueError("raw sidecar path differs from binding")
    if (
        binding.get("format") != "numpy.savez/uncompressed"
        or binding.get("allow_pickle") is not False
    ):
        raise ValueError("raw sidecar format or allow_pickle metadata differs")
    if sha256_file(path) != binding.get("sha256"):
        raise ValueError("raw sidecar file SHA-256 differs")
    bound_invalid = binding.get("invalid_evidence")
    if not isinstance(bound_invalid, bool):
        raise ValueError("raw sidecar binding lacks boolean invalid_evidence")
    if invalid_evidence is not None and invalid_evidence is not bound_invalid:
        raise ValueError("raw sidecar invalid-evidence mode differs from binding")
    bound_count = binding.get("array_count")
    if isinstance(bound_count, bool) or not isinstance(bound_count, int) or bound_count < 0:
        raise ValueError("raw sidecar array_count must be a nonnegative integer")
    arrays: dict[str, np.ndarray] = {}
    with ZipFile(path, "r") as archive:
        members = archive.infolist()
        if len({item.filename for item in members}) != len(members):
            raise ValueError("raw sidecar contains duplicate ZIP members")
        if any(item.compress_type != ZIP_STORED for item in members):
            raise ValueError("raw sidecar is not uncompressed")
    with np.load(path, allow_pickle=False) as loaded:
        for name in loaded.files:
            _validate_raw_logical_name(name, allow_classification=True)
            arrays[name] = little_endian_array(loaded[name])
    _validate_nonfinite_contract(arrays, invalid_evidence=bound_invalid)
    manifest, collection = array_manifest(arrays)
    bound_manifest = binding.get("manifest")
    if (
        not isinstance(bound_manifest, list)
        or canonical_json(manifest) != canonical_json(bound_manifest)
        or collection != binding.get("collection_sha256")
    ):
        raise ValueError("raw sidecar semantic manifest or collection digest differs")
    if len(arrays) != bound_count:
        raise ValueError("raw sidecar array count differs")
    return arrays


def loaded_repository_source_snapshot() -> dict[str, Any]:
    paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(ROOT) or not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] != ".venv":
            paths.add(relative)
    paths.update(
        (PREREGISTRATION, IMPLEMENTATION_REVIEW, HARNESS, FOCUSED_TESTS, Path("pyproject.toml"))
    )
    sealed = set(sealed_paths())
    unexpected = paths - sealed
    if unexpected:
        raise RuntimeError(
            f"loaded repository sources are outside seal: {sorted(map(str, unexpected))}"
        )
    return source_snapshot(paths)


def load_and_verify_seal(path: Path) -> dict[str, Any]:
    if path.absolute() != DEFAULT_SEAL.absolute() or path.is_symlink():
        raise ValueError("official run requires the sole frozen seal path")
    payload = strict_json_load(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError("seal payload is not an object")
    if payload.get("artifact_type") != SEAL_ARTIFACT_TYPE:
        raise ValueError("seal artifact type differs")
    expected_top_level = {
        "artifact_type",
        "created_at_utc",
        "environment",
        "preregistration",
        "implementation_review",
        "preimplementation_sources",
        "sealed_sources",
        "loaded_repository_sources",
        "git",
        "fit_configs",
        "optimizer_scheduler_contract",
        "raw_schema",
        "command_templates",
        "verification",
        "verification_snapshot_sha256",
        "command",
    }
    if set(payload) != expected_top_level:
        raise RuntimeError("seal top-level schema differs")
    if payload.get("command") != seal_command_record():
        raise RuntimeError("seal command binding differs")
    current = seal_snapshot()
    for key, value in current.items():
        if key not in payload or canonical_json(payload[key]) != canonical_json(value):
            raise RuntimeError(f"seal binding drift: {key}")
    if payload.get("verification_snapshot_sha256") != canonical_json_hash(current):
        raise RuntimeError("seal snapshot digest differs")
    created_at = _parse_artifact_utc(payload.get("created_at_utc"), label="seal creation")
    reviewed_text = payload["implementation_review"].get("reviewed_at_utc")
    try:
        reviewed_at = datetime.fromisoformat(str(reviewed_text).replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("seal implementation-review timestamp is malformed") from error
    if reviewed_at.tzinfo != timezone.utc or created_at < reviewed_at or created_at > _utc_now():
        raise RuntimeError("seal creation chronology differs")
    verification = payload.get("verification", {})
    if not isinstance(verification, Mapping) or set(verification) != {"passed", "commands"}:
        raise RuntimeError("seal verification schema differs")
    expected_commands = [list(command) for command in verification_commands()]
    records = verification.get("commands", [])
    if (
        verification.get("passed") is not True
        or not isinstance(records, list)
        or len(records) != len(expected_commands)
        or any(not isinstance(row, Mapping) for row in records)
        or [row.get("command") for row in records] != expected_commands
    ):
        raise RuntimeError("seal lacks the exact passing verification sequence")
    if [row.get("literal_command") for row in records] != list(VERIFICATION_LITERAL_COMMANDS):
        raise RuntimeError("seal verification literal commands differ")
    expected_record_keys = {
        "command",
        "literal_command",
        "returncode",
        "seconds",
        "stdout",
        "stderr",
        "stdout_sha256",
        "stderr_sha256",
    }
    for row in records:
        if set(row) != expected_record_keys:
            raise RuntimeError("seal verification record schema differs")
        returncode = row.get("returncode")
        seconds = row.get("seconds")
        if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode != 0:
            raise RuntimeError("seal records a failed or malformed verification return code")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) < 0.0
        ):
            raise RuntimeError("seal records a malformed verification duration")
        stdout = row.get("stdout")
        stderr = row.get("stderr")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise RuntimeError("seal records malformed verification output")
        if row.get("stdout_sha256") != sha256_bytes(stdout.encode("utf-8")):
            raise RuntimeError("seal stdout hash differs")
        if row.get("stderr_sha256") != sha256_bytes(stderr.encode("utf-8")):
            raise RuntimeError("seal stderr hash differs")
    current_environment = environment_metadata()
    if not isinstance(payload["environment"], Mapping) or set(payload["environment"]) != set(
        current_environment
    ):
        raise RuntimeError("seal environment schema differs")
    assert_official_environment(current_environment)
    sealed_fingerprint = official_environment_fingerprint(payload["environment"])
    current_fingerprint = official_environment_fingerprint(current_environment)
    if canonical_json(sealed_fingerprint) != canonical_json(current_fingerprint):
        raise RuntimeError("runtime environment differs from seal")
    return {
        "path": str(path.absolute()),
        "sha256": sha256_file(path),
        "source_collection_sha256": payload["sealed_sources"]["collection_sha256"],
        "implementation_review": payload["implementation_review"],
        "payload": payload,
    }


def claim_attempt(path: Path, paths: Mapping[str, Path], seal: Mapping[str, Any]) -> dict[str, Any]:
    require_implementation_ready()
    _require_cli_authorization("run", seal_path=DEFAULT_SEAL, output=Path(paths["valid_json"]))
    if path.absolute() != ATTEMPT.absolute() or path.is_symlink():
        raise ValueError("official attempt must use the sole preregistered marker path")
    preflight_absent((path, *paths.values()))
    claimed_at = _utc_now().replace(microsecond=0)
    seal_created_text = seal.get("payload", {}).get("created_at_utc")
    seal_created_at = _parse_artifact_utc(seal_created_text, label="attempt seal creation")
    if claimed_at < seal_created_at:
        raise RuntimeError("attempt predates its implementation seal")
    output_name = Path(paths["valid_json"]).name
    try:
        output_at = datetime.strptime(output_name[:16], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise RuntimeError("attempt output timestamp is malformed") from error
    payload = {
        "artifact_type": ATTEMPT_ARTIFACT_TYPE,
        "claimed_at_utc": claimed_at.isoformat(timespec="seconds"),
        "seal_created_at_utc": seal_created_at.isoformat(timespec="seconds"),
        "output_timestamp_utc": output_at.isoformat(timespec="seconds"),
        "prospective_paths": {key: str(value.absolute()) for key, value in sorted(paths.items())},
        "seal_sha256": seal["sha256"],
        "sealed_source_collection_sha256": seal["source_collection_sha256"],
        "command": run_command_record(Path(paths["valid_json"])),
        "environment": environment_metadata(),
        "resume_permitted": False,
    }
    digest = exclusive_write_json(path, payload)
    return {
        "path": str(path.absolute()),
        "sha256": digest,
        "payload_sha256": canonical_json_hash(payload),
        "payload": payload,
    }


def validate_attempt_binding(expected: Mapping[str, Any]) -> None:
    if Path(str(expected.get("path", ""))).absolute() != ATTEMPT.absolute() or ATTEMPT.is_symlink():
        raise RuntimeError("attempt path binding differs")
    actual = strict_json_load(ATTEMPT)
    expected_keys = {
        "artifact_type",
        "claimed_at_utc",
        "seal_created_at_utc",
        "output_timestamp_utc",
        "prospective_paths",
        "seal_sha256",
        "sealed_source_collection_sha256",
        "command",
        "environment",
        "resume_permitted",
    }
    if not isinstance(actual, Mapping) or set(actual) != expected_keys:
        raise RuntimeError("attempt marker schema differs")
    if actual.get("artifact_type") != ATTEMPT_ARTIFACT_TYPE:
        raise RuntimeError("attempt artifact type differs")
    if sha256_file(ATTEMPT) != expected.get("sha256") or canonical_json_hash(
        actual
    ) != expected.get("payload_sha256"):
        raise RuntimeError("attempt marker changed")
    claimed_at = _parse_artifact_utc(actual.get("claimed_at_utc"), label="attempt claim")
    seal_created_at = _parse_artifact_utc(
        actual.get("seal_created_at_utc"), label="attempt seal creation"
    )
    output_at = _parse_artifact_utc(actual.get("output_timestamp_utc"), label="attempt output")
    if (
        claimed_at < seal_created_at
        or claimed_at > _utc_now()
        or output_at < claimed_at - timedelta(minutes=10)
        or output_at > claimed_at + timedelta(seconds=30)
    ):
        raise RuntimeError("attempt chronology differs")
    expected_payload = expected.get("payload")
    if (
        not isinstance(expected_payload, Mapping)
        or not isinstance(actual["environment"], Mapping)
        or set(actual["environment"]) != set(expected_payload.get("environment", {}))
        or canonical_json(actual["environment"])
        != canonical_json(expected_payload.get("environment"))
    ):
        raise RuntimeError("attempt claim-time environment binding differs")


_SCIENTIFIC_AUDIT_HOOK_INSTALLED = False
_SCIENTIFIC_GUARD_ACTIVE = False
_SCIENTIFIC_GUARD_ARCHIVE: RawArchive | None = None


def _scientific_runtime_audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _SCIENTIFIC_GUARD_ACTIVE:
        return
    optional_import = (
        event == "import"
        and bool(args)
        and str(args[0]).split(".", maxsplit=1)[0] in {"gsplat", "structsplat"}
    )
    forbidden = (
        optional_import
        or event.startswith("socket.")
        or event
        in {
            "subprocess.Popen",
            "os.exec",
            "os.fork",
            "os.forkpty",
            "os.system",
            "os.posix_spawn",
            "os.posix_spawnp",
            "pty.spawn",
        }
    )
    if forbidden:
        phase = (
            "preflight" if _SCIENTIFIC_GUARD_ARCHIVE is None else _SCIENTIFIC_GUARD_ARCHIVE.phase
        )
        raise ProtocolInvalid(
            phase,
            f"network, child process, or optional-backend import attempted: {event}",
        )


def _install_scientific_runtime_audit_hook() -> None:
    global _SCIENTIFIC_AUDIT_HOOK_INSTALLED
    if not _SCIENTIFIC_AUDIT_HOOK_INSTALLED:
        sys.addaudithook(_scientific_runtime_audit_hook)
        _SCIENTIFIC_AUDIT_HOOK_INSTALLED = True


@contextmanager
def offline_scientific_guard(archive: RawArchive):
    """Block network, child-process escape, and optional backend imports during science."""
    global _SCIENTIFIC_GUARD_ACTIVE, _SCIENTIFIC_GUARD_ARCHIVE
    if _SCIENTIFIC_GUARD_ACTIVE:
        raise RuntimeError("scientific isolation guard cannot be nested")
    _install_scientific_runtime_audit_hook()
    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise ProtocolInvalid(archive.phase, "network access attempted during scientific phase")

    class OfflineSocket(original_socket):
        def connect(self, *_args: Any, **_kwargs: Any) -> Any:
            return blocked()

        def connect_ex(self, *_args: Any, **_kwargs: Any) -> Any:
            return blocked()

    socket.socket = OfflineSocket
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    _SCIENTIFIC_GUARD_ARCHIVE = archive
    _SCIENTIFIC_GUARD_ACTIVE = True
    try:
        yield
    finally:
        _SCIENTIFIC_GUARD_ACTIVE = False
        _SCIENTIFIC_GUARD_ARCHIVE = None
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo


# ---------------------------------------------------------------------------
# Frozen scientific evidence engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScientificPlan:
    """Exact cardinalities of the sole preregistered scientific execution."""

    scenes: int = 6
    initializers: int = 54
    step_zero_arm_cells: int = 108
    fits: int = 108
    optimizer_updates: int = 12_960
    checkpoints: int = 864
    checkpoint_component_rows: int = 129_600
    callback_events: int = 26_784
    mechanism_updates_per_arm: int = 3_240
    mechanism_updates: int = 6_480
    mechanism_state_rows: int = 6_534
    current_null_rows: int = 486_000
    weak_rows_per_arm: int = 32_400
    weak_rows_per_seed_arm: int = 10_800
    joint_positive_checkpoints_per_arm: int = 189
    joint_positive_checkpoints: int = 378
    view_metric_cells: int = 864
    seed_arm_checkpoint_cells: int = 96
    seed_arm_auc_cells: int = 12


SCIENTIFIC_PLAN = ScientificPlan()
RNG_STATE_BYTES = 5_056
POSITIVE_CHECKPOINTS = CHECKPOINTS[1:]
EVENT_CODES = {"initial": 0, "pre_update": 1, "post_update": 2, "checkpoint": 3}


@dataclass(frozen=True)
class ScientificFieldSpec:
    dtype: str
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScientificTableSpec:
    rows: int
    fields: Mapping[str, ScientificFieldSpec]


def _sf(dtype: str, *shape: int) -> ScientificFieldSpec:
    return ScientificFieldSpec(np.dtype(dtype).str, tuple(shape))


def _joint_fields(current: bool) -> dict[str, ScientificFieldSpec]:
    shapes: dict[str, tuple[int, ...]] = {
        "xy_raw": (COMPONENTS, 2),
        "diag_raw": (COMPONENTS, 2),
        "off_raw": (COMPONENTS,),
    }
    if current:
        shapes.update(
            {
                "color_raw": (COMPONENTS, 3),
                "weight_raw": (COMPONENTS,),
            }
        )
    else:
        shapes["amplitude_raw"] = (COMPONENTS, 3)
    fields: dict[str, ScientificFieldSpec] = {
        "fit_index": _sf("<i8"),
        "step": _sf("<i8"),
        "defined": _sf("|b1"),
        "lr_used": _sf("<f8"),
    }
    for name, shape in shapes.items():
        for prefix in (
            "pre",
            "gradient",
            "exp_avg_before",
            "exp_avg_sq_before",
            "post",
            "exp_avg_after",
            "exp_avg_sq_after",
            "displacement",
        ):
            fields[f"{prefix}_{name}"] = _sf("<f4", *shape)
    parameter_count = 5 if current else 4
    fields["step_before"] = _sf("<i8", parameter_count)
    fields["step_after"] = _sf("<i8", parameter_count)
    fields["state_present_before"] = _sf("|b1", parameter_count)
    fields["state_present_after"] = _sf("|b1", parameter_count)
    return fields


SCIENTIFIC_TABLE_SPECS: dict[str, ScientificTableSpec] = {
    "prerequisite": ScientificTableSpec(
        SCIENTIFIC_PLAN.initializers,
        {
            "block_index": _sf("<i8"),
            "seed_position": _sf("<i8"),
            "seed": _sf("<i8"),
            "local_view": _sf("<i8"),
            "original_view": _sf("<i8"),
            "initializer_seed": _sf("<i8"),
            "target": _sf("<f4", IMAGE_SIZE, IMAGE_SIZE, 3),
            "rng_state_before": _sf("|u1", RNG_STATE_BYTES),
            "rng_state_after": _sf("|u1", RNG_STATE_BYTES),
            "g0_xy": _sf("<f4", COMPONENTS, 2),
            "g0_chol": _sf("<f4", COMPONENTS, 3),
            "g0_color": _sf("<f4", COMPONENTS, 3),
            "g0_weight": _sf("<f4", COMPONENTS),
            "current_xy_raw": _sf("<f4", COMPONENTS, 2),
            "current_diag_raw": _sf("<f4", COMPONENTS, 2),
            "current_off_raw": _sf("<f4", COMPONENTS),
            "current_color_raw": _sf("<f4", COMPONENTS, 3),
            "current_weight_raw": _sf("<f4", COMPONENTS),
            "candidate_xy_raw": _sf("<f4", COMPONENTS, 2),
            "candidate_diag_raw": _sf("<f4", COMPONENTS, 2),
            "candidate_off_raw": _sf("<f4", COMPONENTS),
            "candidate_amplitude_raw": _sf("<f4", COMPONENTS, 3),
            "current_built_xy": _sf("<f4", COMPONENTS, 2),
            "current_built_chol": _sf("<f4", COMPONENTS, 3),
            "current_built_color": _sf("<f4", COMPONENTS, 3),
            "current_built_weight": _sf("<f4", COMPONENTS),
            "current_built_amplitude": _sf("<f4", COMPONENTS, 3),
            "candidate_built_xy": _sf("<f4", COMPONENTS, 2),
            "candidate_built_chol": _sf("<f4", COMPONENTS, 3),
            "candidate_built_color": _sf("<f4", COMPONENTS, 3),
            "candidate_built_weight": _sf("<f4", COMPONENTS),
            "candidate_built_amplitude": _sf("<f4", COMPONENTS, 3),
            "current_render": _sf("<f4", IMAGE_SIZE, IMAGE_SIZE, 3),
            "candidate_render": _sf("<f4", IMAGE_SIZE, IMAGE_SIZE, 3),
            "current_loss": _sf("<f4"),
            "candidate_loss": _sf("<f4"),
            # candidate-color abs/rel, render abs/relative numerator/denominator/value,
            # render MSE/PSNR, and loss abs/relative value.
            "gate_values": _sf("<f8", 10),
            "target_sha256": _sf("|u1", 64),
            "g0_field_sha256": _sf("|u1", 4, 64),
            "rng_state_sha256": _sf("|u1", 2, 64),
        },
    ),
    "fit_event": ScientificTableSpec(
        SCIENTIFIC_PLAN.callback_events,
        {
            "fit_index": _sf("<i8"),
            "event_code": _sf("<i8"),
            "step": _sf("<i8"),
            "lr_used": _sf("<f8"),
            "lr_used_defined": _sf("|b1"),
            "next_lr": _sf("<f8"),
            "scheduler_last_epoch": _sf("<i8"),
            "scheduler_step_count": _sf("<i8"),
        },
    ),
    "fit": ScientificTableSpec(
        SCIENTIFIC_PLAN.fits,
        {
            "fit_index": _sf("<i8"),
            "block_index": _sf("<i8"),
            "seed_position": _sf("<i8"),
            "seed": _sf("<i8"),
            "local_view": _sf("<i8"),
            "original_view": _sf("<i8"),
            "arm_code": _sf("<i8"),
            "arm_order_position": _sf("<i8"),
            "optimizer_param_count": _sf("<i8"),
            "optimizer_defaults_bytes": _sf("|u1", 4_096),
            "optimizer_defaults_length": _sf("<i8"),
            "optimizer_groups_bytes": _sf("|u1", 4_096),
            "optimizer_groups_length": _sf("<i8"),
            "scheduler_initial_bytes": _sf("|u1", 4_096),
            "scheduler_initial_length": _sf("<i8"),
            "lr_trace": _sf("<f8", UPDATES),
            "next_lr_trace": _sf("<f8", UPDATES + 1),
            "stopped_iter": _sf("<i8"),
            "final_psnr_full": _sf("<f4"),
            "final_psnr": _sf("<f4"),
            "result_xy": _sf("<f4", COMPONENTS, 2),
            "result_chol": _sf("<f4", COMPONENTS, 3),
            "result_color": _sf("<f4", COMPONENTS, 3),
            "result_weight": _sf("<f4", COMPONENTS),
            "global_rng_before": _sf("|u1", RNG_STATE_BYTES),
            "global_rng_after": _sf("|u1", RNG_STATE_BYTES),
        },
    ),
    "checkpoint": ScientificTableSpec(
        SCIENTIFIC_PLAN.checkpoints,
        {
            "checkpoint_index": _sf("<i8"),
            "fit_index": _sf("<i8"),
            "step": _sf("<i8"),
            "xy_raw": _sf("<f4", COMPONENTS, 2),
            "diag_raw": _sf("<f4", COMPONENTS, 2),
            "off_raw": _sf("<f4", COMPONENTS),
            "built_xy": _sf("<f4", COMPONENTS, 2),
            "built_chol": _sf("<f4", COMPONENTS, 3),
            "built_weight": _sf("<f4", COMPONENTS),
            "built_color": _sf("<f4", COMPONENTS, 3),
            "built_amplitude": _sf("<f4", COMPONENTS, 3),
            "render": _sf("<f4", IMAGE_SIZE, IMAGE_SIZE, 3),
            "target": _sf("<f4", IMAGE_SIZE, IMAGE_SIZE, 3),
            "clamped_render": _sf("<f4", IMAGE_SIZE, IMAGE_SIZE, 3),
            "below_mask": _sf("|b1", IMAGE_SIZE, IMAGE_SIZE, 3),
            "above_mask": _sf("|b1", IMAGE_SIZE, IMAGE_SIZE, 3),
            "objective_loss": _sf("<f4"),
            "sse": _sf("<f8"),
            "channel_count": _sf("<i8"),
            "raw_mse": _sf("<f8"),
            "psnr": _sf("<f4"),
            "ssim": _sf("<f4"),
            "below_count": _sf("<i8"),
            "above_count": _sf("<i8"),
            "below_fraction": _sf("<f8"),
            "above_fraction": _sf("<f8"),
        },
    ),
    "checkpoint_current": ScientificTableSpec(
        SCIENTIFIC_PLAN.checkpoints // 2,
        {
            "checkpoint_index": _sf("<i8"),
            "fit_index": _sf("<i8"),
            "step": _sf("<i8"),
            "color_raw": _sf("<f4", COMPONENTS, 3),
            "weight_raw": _sf("<f4", COMPONENTS),
            "jacobian": _sf("<f8", COMPONENTS, 3, 4),
            "singular_values": _sf("<f8", COMPONENTS, 3),
            "vh": _sf("<f8", COMPONENTS, 4, 4),
            "largest": _sf("<f8", COMPONENTS),
            "tau": _sf("<f8", COMPONENTS),
            "rank": _sf("<i8", COMPONENTS),
            "smallest_positive": _sf("<f8", COMPONENTS),
            "smallest_positive_defined": _sf("|b1", COMPONENTS),
            "condition": _sf("<f8", COMPONENTS),
            "condition_defined": _sf("|b1", COMPONENTS),
            "weakly_responsive": _sf("|b1", COMPONENTS),
            "null_vector": _sf("<f8", COMPONENTS, 4),
            "null_residual": _sf("<f8", COMPONENTS),
            "analytic_null_vector": _sf("<f8", COMPONENTS, 4),
            "analytic_null_alignment": _sf("<f8", COMPONENTS),
            "analytic_null_defined": _sf("|b1", COMPONENTS),
            # fields=(color, weight, amplitude), columns=(raw>=8, raw_n, low, high,
            # output_n, derivative<=1e-4); nullable entries use the separate mask.
            "saturation_counts": _sf("<i8", 3, 6),
            "saturation_count_defined": _sf("|b1", 3, 6),
            # columns=(raw fraction, low fraction, high fraction, derivative fraction).
            "saturation_fractions": _sf("<f8", 3, 4),
            "saturation_fraction_defined": _sf("|b1", 3, 4),
            "histograms": _sf("<i8", 3, len(HISTOGRAM_EDGES) - 1),
        },
    ),
    "checkpoint_candidate": ScientificTableSpec(
        SCIENTIFIC_PLAN.checkpoints // 2,
        {
            "checkpoint_index": _sf("<i8"),
            "fit_index": _sf("<i8"),
            "step": _sf("<i8"),
            "amplitude_raw": _sf("<f4", COMPONENTS, 3),
            "jacobian": _sf("<f8", COMPONENTS, 3, 3),
            "singular_values": _sf("<f8", COMPONENTS, 3),
            "vh": _sf("<f8", COMPONENTS, 3, 3),
            "largest": _sf("<f8", COMPONENTS),
            "tau": _sf("<f8", COMPONENTS),
            "rank": _sf("<i8", COMPONENTS),
            "smallest_positive": _sf("<f8", COMPONENTS),
            "smallest_positive_defined": _sf("|b1", COMPONENTS),
            "condition": _sf("<f8", COMPONENTS),
            "condition_defined": _sf("|b1", COMPONENTS),
            "weakly_responsive": _sf("|b1", COMPONENTS),
            "saturation_counts": _sf("<i8", 1, 6),
            "saturation_count_defined": _sf("|b1", 1, 6),
            "saturation_fractions": _sf("<f8", 1, 4),
            "saturation_fraction_defined": _sf("|b1", 1, 4),
            "histograms": _sf("<i8", 1, len(HISTOGRAM_EDGES) - 1),
        },
    ),
    "mechanism_geometry": ScientificTableSpec(
        SCIENTIFIC_PLAN.mechanism_state_rows,
        {
            "fit_index": _sf("<i8"),
            "step": _sf("<i8"),
            "xy_raw": _sf("<f4", COMPONENTS, 2),
            "diag_raw": _sf("<f4", COMPONENTS, 2),
            "off_raw": _sf("<f4", COMPONENTS),
            "built_xy": _sf("<f4", COMPONENTS, 2),
            "built_chol": _sf("<f4", COMPONENTS, 3),
            "built_weight": _sf("<f4", COMPONENTS),
        },
    ),
    "mechanism_current": ScientificTableSpec(
        SCIENTIFIC_PLAN.mechanism_updates_per_arm,
        {
            "fit_index": _sf("<i8"),
            "completed_step": _sf("<i8"),
            "update_number": _sf("<i8"),
            "color_raw_pre": _sf("<f4", COMPONENTS, 3),
            "weight_raw_pre": _sf("<f4", COMPONENTS),
            "built_color": _sf("<f4", COMPONENTS, 3),
            "built_weight": _sf("<f4", COMPONENTS),
            "built_amplitude": _sf("<f4", COMPONENTS, 3),
            "gradient_color": _sf("<f4", COMPONENTS, 3),
            "gradient_weight": _sf("<f4", COMPONENTS),
            "gradient_amplitude": _sf("<f4", COMPONENTS, 3),
            "objective_loss": _sf("<f4"),
            # render maxabs, rel numerator, denominator, ratio, probe loss, loss abs,
            # optimization-loss denominator, loss ratio, and element count.
            "probe_values": _sf("<f8", 9),
            "exp_avg_color_before": _sf("<f4", COMPONENTS, 3),
            "exp_avg_sq_color_before": _sf("<f4", COMPONENTS, 3),
            "exp_avg_weight_before": _sf("<f4", COMPONENTS),
            "exp_avg_sq_weight_before": _sf("<f4", COMPONENTS),
            "step_before": _sf("<i8", 2),
            "state_present_before": _sf("|b1", 2),
            "lr_used": _sf("<f8"),
            "color_raw_after": _sf("<f4", COMPONENTS, 3),
            "weight_raw_after": _sf("<f4", COMPONENTS),
            "exp_avg_color_after": _sf("<f4", COMPONENTS, 3),
            "exp_avg_sq_color_after": _sf("<f4", COMPONENTS, 3),
            "exp_avg_weight_after": _sf("<f4", COMPONENTS),
            "exp_avg_sq_weight_after": _sf("<f4", COMPONENTS),
            "step_after": _sf("<i8", 2),
            "state_present_after": _sf("|b1", 2),
            "displacement_color": _sf("<f4", COMPONENTS, 3),
            "displacement_weight": _sf("<f4", COMPONENTS),
            "expected_gradient_color": _sf("<f4", COMPONENTS, 3),
            "expected_gradient_weight": _sf("<f4", COMPONENTS),
            "reconstructed_displacement_color": _sf("<f4", COMPONENTS, 3),
            "reconstructed_displacement_weight": _sf("<f4", COMPONENTS),
            # grad abs/rel for color, weight; displacement abs/rel for color, weight.
            "equation_errors": _sf("<f8", 8),
            "jacobian": _sf("<f8", COMPONENTS, 3, 4),
            "singular_values": _sf("<f8", COMPONENTS, 3),
            "vh": _sf("<f8", COMPONENTS, 4, 4),
            "rank": _sf("<i8", COMPONENTS),
            "null_vector": _sf("<f8", COMPONENTS, 4),
            "null_residual": _sf("<f8", COMPONENTS),
            "analytic_null_vector": _sf("<f8", COMPONENTS, 4),
            "analytic_null_alignment": _sf("<f8", COMPONENTS),
            "analytic_null_defined": _sf("|b1", COMPONENTS),
            "null_dot": _sf("<f8", COMPONENTS),
            "update_norm": _sf("<f8", COMPONENTS),
            "null_eligible": _sf("|b1", COMPONENTS),
            "null_fraction": _sf("<f8", COMPONENTS),
            "squared_projection": _sf("<f8", COMPONENTS),
            "squared_update_norm": _sf("<f8", COMPONENTS),
            "gradient_dot": _sf("<f8", COMPONENTS),
            "gradient_norm": _sf("<f8", COMPONENTS),
            "gradient_cosine": _sf("<f8", COMPONENTS),
            "gradient_cosine_defined": _sf("|b1", COMPONENTS),
        },
    ),
    "mechanism_candidate": ScientificTableSpec(
        SCIENTIFIC_PLAN.mechanism_updates_per_arm,
        {
            "fit_index": _sf("<i8"),
            "completed_step": _sf("<i8"),
            "update_number": _sf("<i8"),
            "amplitude_raw_pre": _sf("<f4", COMPONENTS, 3),
            "built_color": _sf("<f4", COMPONENTS, 3),
            "built_weight": _sf("<f4", COMPONENTS),
            "built_amplitude": _sf("<f4", COMPONENTS, 3),
            "gradient_amplitude_raw": _sf("<f4", COMPONENTS, 3),
            "gradient_amplitude": _sf("<f4", COMPONENTS, 3),
            "objective_loss": _sf("<f4"),
            "probe_values": _sf("<f8", 9),
            "exp_avg_before": _sf("<f4", COMPONENTS, 3),
            "exp_avg_sq_before": _sf("<f4", COMPONENTS, 3),
            "step_before": _sf("<i8"),
            "state_present_before": _sf("|b1"),
            "lr_used": _sf("<f8"),
            "amplitude_raw_after": _sf("<f4", COMPONENTS, 3),
            "exp_avg_after": _sf("<f4", COMPONENTS, 3),
            "exp_avg_sq_after": _sf("<f4", COMPONENTS, 3),
            "step_after": _sf("<i8"),
            "state_present_after": _sf("|b1"),
            "displacement": _sf("<f4", COMPONENTS, 3),
            "expected_gradient": _sf("<f4", COMPONENTS, 3),
            "reconstructed_displacement": _sf("<f4", COMPONENTS, 3),
            # grad abs/rel and displacement abs/rel.
            "equation_errors": _sf("<f8", 4),
        },
    ),
    "joint_current": ScientificTableSpec(
        3 * len(SELECTED_VIEWS) * len(CHECKPOINTS), _joint_fields(True)
    ),
    "joint_candidate": ScientificTableSpec(
        3 * len(SELECTED_VIEWS) * len(CHECKPOINTS), _joint_fields(False)
    ),
}


def scientific_schema_record() -> dict[str, Any]:
    """Outcome-free raw schema bound by a future implementation seal."""
    tables: dict[str, Any] = {}
    array_count = 0
    for name, spec in SCIENTIFIC_TABLE_SPECS.items():
        tables[name] = {
            "rows": spec.rows,
            "fields": {
                field_name: {"dtype": field_spec.dtype, "row_shape": list(field_spec.shape)}
                for field_name, field_spec in sorted(spec.fields.items())
            },
        }
        array_count += 1 + len(spec.fields)
    constants = {
        "identity/block_name_bytes",
        "identity/block_name_lengths",
        "identity/arm_name_bytes",
        "identity/arm_name_lengths",
        "identity/block_seeds",
        "identity/selected_views",
        "identity/checkpoints",
        "identity/component_indices",
        "identity/histogram_edges",
        "identity/arm_order",
    }
    completion = {
        "scientific_completion/phase_code",
        "scientific_completion/count_name_bytes",
        "scientific_completion/count_name_lengths",
        "scientific_completion/count_values",
    }
    archive_completion = {"completion/phase_code"} | {
        f"completion/{name}"
        for name in (
            "completed_blocks",
            "completed_seeds",
            "completed_views",
            "completed_initializers",
            "completed_paired_views",
            "completed_fits",
            "completed_updates",
            "completed_checkpoints",
            "completed_callback_events",
            "completed_mechanism_updates",
            "completed_joint_positive_checkpoint_updates",
            "completed_checkpoint_component_rows",
            "completed_null_rows",
        )
    }
    constant_specs = {
        "identity/block_name_bytes": {"dtype": "|u1", "shape": [len(BLOCKS), 32]},
        "identity/block_name_lengths": {"dtype": "<i8", "shape": [len(BLOCKS)]},
        "identity/arm_name_bytes": {"dtype": "|u1", "shape": [len(ARMS), 32]},
        "identity/arm_name_lengths": {"dtype": "<i8", "shape": [len(ARMS)]},
        "identity/block_seeds": {"dtype": "<i8", "shape": [len(BLOCKS), 3]},
        "identity/selected_views": {"dtype": "<i8", "shape": [len(SELECTED_VIEWS)]},
        "identity/checkpoints": {"dtype": "<i8", "shape": [len(CHECKPOINTS)]},
        "identity/component_indices": {"dtype": "<i8", "shape": [COMPONENTS]},
        "identity/histogram_edges": {"dtype": "<f8", "shape": [len(HISTOGRAM_EDGES)]},
        "identity/arm_order": {"dtype": "<i8", "shape": [len(BLOCKS), 3, 2]},
    }
    scientific_completion_specs = {
        "scientific_completion/phase_code": {"dtype": "<i8", "shape": []},
        "scientific_completion/count_name_bytes": {"dtype": "|u1", "shape": [11, 64]},
        "scientific_completion/count_name_lengths": {"dtype": "<i8", "shape": [11]},
        "scientific_completion/count_values": {"dtype": "<i8", "shape": [11]},
    }
    archive_completion_specs = {
        name: {"dtype": "<i8", "shape": []} for name in sorted(archive_completion)
    }
    return {
        "plan": asdict(SCIENTIFIC_PLAN),
        "tables": tables,
        "constant_arrays": sorted(constants),
        "constant_array_specs": constant_specs,
        "completion_arrays": sorted(completion),
        "completion_array_specs": scientific_completion_specs,
        "archive_completion_arrays": sorted(archive_completion),
        "archive_completion_array_specs": archive_completion_specs,
        "expected_valid_array_count": (
            array_count + len(constants) + len(completion) + len(archive_completion)
        ),
        "invalid_tables_are_leading_prefixes": True,
    }


@dataclass
class ScientificTableBuffer:
    name: str
    archive: RawArchive
    rows: list[dict[str, np.ndarray]] = field(default_factory=list)
    flushed: bool = False

    @property
    def spec(self) -> ScientificTableSpec:
        return SCIENTIFIC_TABLE_SPECS[self.name]

    def append(self, **values: Any) -> None:
        if self.flushed:
            raise ProtocolInvalid(self.archive.phase, f"table already flushed: {self.name}")
        expected = set(self.spec.fields)
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ProtocolInvalid(
                self.archive.phase,
                f"table field mismatch: {self.name}",
                {"missing": missing, "extra": extra},
            )
        if len(self.rows) >= self.spec.rows:
            raise ProtocolInvalid(self.archive.phase, f"too many rows for {self.name}")
        normalized: dict[str, np.ndarray] = {}
        for field_name, field_spec in self.spec.fields.items():
            value = values[field_name]
            if isinstance(value, torch.Tensor):
                value = value.detach().contiguous().cpu().numpy()
            array = np.asarray(value, dtype=np.dtype(field_spec.dtype))
            array = little_endian_array(array)
            if array.shape != field_spec.shape:
                raise ProtocolInvalid(
                    self.archive.phase,
                    f"row shape mismatch: {self.name}/{field_name}",
                    {"actual": list(array.shape), "expected": list(field_spec.shape)},
                )
            classification = nonfinite_classification(array)
            if bool(classification.any()):
                failure_name = f"failure/{self.name}/{field_name}/{len(self.rows):06d}"
                self.archive.add(failure_name, array)
                raise AssertionError("RawArchive.add must invalidate a non-finite row")
            normalized[field_name] = array.copy()
        self.rows.append(normalized)

    def flush(self, *, require_complete: bool) -> None:
        if self.flushed:
            return
        if require_complete and len(self.rows) != self.spec.rows:
            raise ProtocolInvalid(
                "serialization",
                f"table completeness mismatch: {self.name}",
                {"actual": len(self.rows), "expected": self.spec.rows},
            )
        if not self.rows:
            self.flushed = True
            return
        prefix = f"scientific/{self.name}"
        self.archive.add(f"{prefix}/row_count", np.asarray(len(self.rows), dtype=np.int64))
        for field_name in self.spec.fields:
            stacked = np.stack([row[field_name] for row in self.rows], axis=0)
            self.archive.add(f"{prefix}/{field_name}", stacked)
        self.flushed = True


@dataclass(frozen=True)
class PreparedFitCell:
    prerequisite_index: int
    block_index: int
    seed_position: int
    seed: int
    local_view: int
    original_view: int
    target: torch.Tensor
    g0: Gaussians2D
    current_raw: Mapping[str, torch.Tensor]
    candidate_raw: Mapping[str, torch.Tensor]
    current_built: Gaussians2D
    candidate_built: Gaussians2D
    current_render: torch.Tensor
    candidate_render: torch.Tensor
    current_loss: torch.Tensor
    candidate_loss: torch.Tensor


@dataclass
class ScientificEvidence:
    archive: RawArchive
    tables: dict[str, ScientificTableBuffer] = field(init=False)
    prepared: dict[tuple[int, int, int], PreparedFitCell] = field(default_factory=dict)
    optimization_unlocked: bool = False
    completed_scenes: int = 0
    completed_initializers: int = 0
    completed_fits: int = 0
    completed_updates: int = 0
    completed_checkpoints: int = 0
    completed_callback_events: int = 0

    def __post_init__(self) -> None:
        self.tables = {
            name: ScientificTableBuffer(name=name, archive=self.archive)
            for name in SCIENTIFIC_TABLE_SPECS
        }

    def completion_array_values(self, *, phase: str | None = None) -> dict[str, np.ndarray]:
        counts = {
            "scenes": self.completed_scenes,
            "initializers": self.completed_initializers,
            "fits": self.completed_fits,
            "optimizer_updates": self.completed_updates,
            "checkpoints": self.completed_checkpoints,
            "callback_events": self.completed_callback_events,
            "mechanism_current_updates": len(self.tables["mechanism_current"].rows),
            "mechanism_candidate_updates": len(self.tables["mechanism_candidate"].rows),
            "mechanism_geometry_states": len(self.tables["mechanism_geometry"].rows),
            "joint_current_positive_checkpoints": sum(
                bool(row["defined"]) for row in self.tables["joint_current"].rows
            ),
            "joint_candidate_positive_checkpoints": sum(
                bool(row["defined"]) for row in self.tables["joint_candidate"].rows
            ),
        }
        names = sorted(counts)
        name_bytes, name_lengths = _utf8_rows(names, 64)
        return {
            "scientific_completion/phase_code": np.asarray(
                _phase_code(self.archive.phase if phase is None else phase), dtype=np.int64
            ),
            "scientific_completion/count_name_bytes": name_bytes,
            "scientific_completion/count_name_lengths": name_lengths,
            "scientific_completion/count_values": np.asarray(
                [counts[name] for name in names], dtype=np.int64
            ),
        }

    def completion_arrays(self, *, phase: str | None = None) -> None:
        for name, value in self.completion_array_values(phase=phase).items():
            self.archive.add(name, value)

    def flush(self, *, require_complete: bool, include_completion: bool = True) -> None:
        _add_scientific_constants(self.archive)
        for table in self.tables.values():
            table.flush(require_complete=require_complete)
        if include_completion:
            self.completion_arrays()


def _add_scientific_constants(archive: RawArchive) -> None:
    if "identity/block_name_bytes" in archive.arrays:
        return
    block_bytes, block_lengths = _utf8_rows(BLOCKS, 32)
    arm_bytes, arm_lengths = _utf8_rows(ARMS, 32)
    archive.add("identity/block_name_bytes", block_bytes)
    archive.add("identity/block_name_lengths", block_lengths)
    archive.add("identity/arm_name_bytes", arm_bytes)
    archive.add("identity/arm_name_lengths", arm_lengths)
    archive.add(
        "identity/block_seeds",
        np.asarray([BLOCK_SEEDS[block] for block in BLOCKS], dtype=np.int64),
    )
    archive.add("identity/selected_views", np.asarray(SELECTED_VIEWS, dtype=np.int64))
    archive.add("identity/checkpoints", np.asarray(CHECKPOINTS, dtype=np.int64))
    archive.add("identity/component_indices", np.arange(COMPONENTS, dtype=np.int64))
    archive.add("identity/histogram_edges", np.asarray(HISTOGRAM_EDGES, dtype=np.float64))
    archive.add(
        "identity/arm_order",
        np.asarray(
            [
                [tuple(ARMS.index(arm) for arm in arm_order(position)) for position in range(3)]
                for _block in BLOCKS
            ],
            dtype=np.int64,
        ),
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    return array_content_sha256(value.detach().contiguous().cpu().numpy())


def _utf8_buffer(value: str, width: int) -> tuple[np.ndarray, int]:
    encoded = value.encode("utf-8")
    if len(encoded) > width:
        raise ProtocolInvalid("serialization", "UTF-8 evidence exceeds fixed raw width")
    result = np.zeros(width, dtype=np.uint8)
    result[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return result, len(encoded)


def _utf8_rows(values: Iterable[str], width: int) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    lengths: list[int] = []
    for value in values:
        row, length = _utf8_buffer(value, width)
        rows.append(row)
        lengths.append(length)
    return np.stack(rows, axis=0), np.asarray(lengths, dtype=np.int64)


def _sha256_bytes(value: str) -> np.ndarray:
    result, length = _utf8_buffer(value, 64)
    if length != 64:
        raise ProtocolInvalid("serialization", "SHA-256 text is not 64 ASCII bytes")
    return result


def _clone_gaussians(value: Gaussians2D) -> Gaussians2D:
    return value.detach()


def _assert_tensor_equal(
    name: str, actual: torch.Tensor, expected: torch.Tensor, phase: str
) -> None:
    if (
        actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or not torch.equal(actual, expected)
    ):
        raise ProtocolInvalid(
            phase,
            f"bit-exact tensor mismatch: {name}",
            raw_arrays={"actual": actual, "expected": expected},
        )


def _assert_error(
    phase: str,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    absolute_limit: float,
    relative_limit: float,
    relative_threshold: float = 1e-8,
) -> tuple[float, float]:
    maximum_absolute, maximum_relative = error_pair(actual, expected, relative_threshold)
    if maximum_absolute > absolute_limit or maximum_relative > relative_limit:
        raise ProtocolInvalid(
            phase,
            f"{name} tolerance failed",
            {
                "maximum_absolute": maximum_absolute,
                "maximum_relative": maximum_relative,
                "absolute_limit": absolute_limit,
                "relative_limit": relative_limit,
            },
            raw_arrays={"actual": actual, "expected": expected},
        )
    return maximum_absolute, maximum_relative


def _validate_target_and_initializer(target: torch.Tensor, g0: Gaussians2D) -> None:
    reached = {
        "target": target,
        "g0_xy": g0.xy,
        "g0_chol": g0.chol,
        "g0_color": g0.color,
        "g0_weight": g0.weight,
    }
    if target.dtype != torch.float32 or target.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ProtocolInvalid("initialization", "target shape/dtype differs", raw_arrays=reached)
    if not bool(torch.isfinite(target).all()):
        raise ProtocolInvalid(
            "initialization",
            "target is non-finite",
            raw_arrays=reached,
        )
    if not bool(((target >= 0) & (target <= 1)).all()):
        raise ProtocolInvalid("initialization", "target is outside [0,1]", raw_arrays=reached)
    if g0.n != COMPONENTS:
        raise ProtocolInvalid(
            "initialization", "initializer component count differs", raw_arrays=reached
        )
    expected_shapes = {
        "xy": (COMPONENTS, 2),
        "chol": (COMPONENTS, 3),
        "color": (COMPONENTS, 3),
        "weight": (COMPONENTS,),
    }
    for name, shape in expected_shapes.items():
        value = getattr(g0, name)
        if value.dtype != torch.float32 or value.shape != shape:
            raise ProtocolInvalid(
                "initialization",
                f"initializer field invalid: {name}",
                raw_arrays=reached,
            )
        if not bool(torch.isfinite(value).all()):
            raise ProtocolInvalid(
                "initialization",
                f"initializer field is non-finite: {name}",
                raw_arrays=reached,
            )
    if not bool(
        (
            (g0.xy[:, 0] >= 0)
            & (g0.xy[:, 0] < IMAGE_SIZE)
            & (g0.xy[:, 1] >= 0)
            & (g0.xy[:, 1] < IMAGE_SIZE)
        ).all()
    ):
        raise ProtocolInvalid(
            "initialization", "initializer center outside image", raw_arrays=reached
        )
    if not bool((g0.chol[:, (0, 2)] > 0.3).all()):
        raise ProtocolInvalid(
            "initialization",
            "initializer Cholesky diagonal does not exceed production minimum",
            raw_arrays=reached,
        )
    if not bool(((g0.color >= 0) & (g0.color <= 1)).all()):
        raise ProtocolInvalid(
            "initialization", "initializer color outside [0,1]", raw_arrays=reached
        )
    if not torch.equal(g0.weight, torch.full_like(g0.weight, 0.5)):
        raise ProtocolInvalid(
            "initialization", "initializer weight is not exact 0.5", raw_arrays=reached
        )


def _prepare_step_zero(
    target: torch.Tensor,
    g0: Gaussians2D,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    Gaussians2D,
    Gaussians2D,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    common = common_raw_initialization(g0, IMAGE_SIZE, IMAGE_SIZE)
    current_raw = {
        "xy_raw": common["xy_raw"].clone(),
        "diag_raw": common["diag_raw"].clone(),
        "off_raw": common["off_raw"].clone(),
        "u": common["u"].clone(),
        "s": common["s"].clone(),
    }
    candidate_raw = {
        "xy_raw": common["xy_raw"].clone(),
        "diag_raw": common["diag_raw"].clone(),
        "off_raw": common["off_raw"].clone(),
        "r": common["r"].clone(),
    }
    current = build_from_raw(current_raw, ARMS[0], IMAGE_SIZE, IMAGE_SIZE)
    candidate = build_from_raw(candidate_raw, ARMS[1], IMAGE_SIZE, IMAGE_SIZE)
    step_zero_arrays = {
        "target": target,
        **{f"current_raw_{name}": value for name, value in current_raw.items()},
        **{f"candidate_raw_{name}": value for name, value in candidate_raw.items()},
        "current_built_xy": current.xy,
        "current_built_chol": current.chol,
        "current_built_color": current.color,
        "current_built_weight": current.weight,
        "candidate_built_xy": candidate.xy,
        "candidate_built_chol": candidate.chol,
        "candidate_built_color": candidate.color,
        "candidate_built_weight": candidate.weight,
    }
    for field_name in ("xy_raw", "diag_raw", "off_raw"):
        _assert_tensor_equal(
            f"step_zero/{field_name}",
            current_raw[field_name],
            candidate_raw[field_name],
            "equivalence",
        )
    _assert_tensor_equal("step_zero/built_xy", current.xy, candidate.xy, "equivalence")
    _assert_tensor_equal("step_zero/built_chol", current.chol, candidate.chol, "equivalence")
    if not torch.equal(candidate.weight, torch.ones_like(candidate.weight)):
        raise ProtocolInvalid(
            "equivalence",
            "candidate step-zero weight is not exact one",
            raw_arrays=step_zero_arrays,
        )
    amplitude0 = common["amplitude"]
    color_abs, color_rel = error_pair(candidate.color, amplitude0)
    if color_abs > 1e-7 or color_rel > 1e-6:
        raise ProtocolInvalid(
            "equivalence",
            "candidate common-amplitude construction failed",
            {"maximum_absolute": color_abs, "maximum_relative": color_rel},
            raw_arrays={**step_zero_arrays, "common_amplitude": amplitude0},
        )
    current_render = render_gaussians_2d(current, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64)
    candidate_render = render_gaussians_2d(candidate, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64)
    current_loss = torch.nn.functional.mse_loss(current_render, target)
    candidate_loss = torch.nn.functional.mse_loss(candidate_render, target)
    forward_arrays = {
        **step_zero_arrays,
        "current_render": current_render,
        "candidate_render": candidate_render,
        "current_loss": current_loss,
        "candidate_loss": candidate_loss,
    }
    render_difference = (candidate_render - current_render).abs()
    render_abs = float(render_difference.max())
    render_numerator = float(render_difference.to(torch.float64).sum())
    render_denominator = float(current_render.abs().to(torch.float64).sum())
    if not math.isfinite(render_denominator) or render_denominator <= 0:
        raise ProtocolInvalid(
            "equivalence",
            "step-zero render denominator is invalid",
            raw_arrays=forward_arrays,
        )
    render_relative = render_numerator / render_denominator
    render_mse = float(
        (candidate_render.to(torch.float64) - current_render.to(torch.float64)).square().mean()
    )
    render_psnr = -10.0 * math.log10(max(render_mse, 1e-12))
    loss_abs = abs(float(candidate_loss) - float(current_loss))
    current_loss_value = float(current_loss)
    if not math.isfinite(current_loss_value) or current_loss_value <= 0:
        raise ProtocolInvalid(
            "equivalence",
            "step-zero current loss denominator is invalid",
            raw_arrays=forward_arrays,
        )
    loss_relative = loss_abs / current_loss_value
    gate_values = np.asarray(
        [
            color_abs,
            color_rel,
            render_abs,
            render_numerator,
            render_denominator,
            render_relative,
            render_mse,
            render_psnr,
            loss_abs,
            loss_relative,
        ],
        dtype=np.float64,
    )
    if (
        render_abs > 5e-6
        or render_relative > 1e-6
        or render_psnr < 100.0
        or loss_abs > 1e-7
        or loss_relative > 1e-6
    ):
        raise ProtocolInvalid(
            "equivalence",
            "global step-zero equivalence gate failed",
            raw_arrays={**forward_arrays, "gate_values": gate_values},
        )
    return (
        current_raw,
        candidate_raw,
        current,
        candidate,
        current_render,
        candidate_render,
        current_loss,
        candidate_loss,
        gate_values,
    )


def build_global_prerequisite(evidence: ScientificEvidence) -> None:
    """Materialize and validate every common forward before any optimizer may exist."""
    if evidence.prepared or evidence.optimization_unlocked:
        raise ProtocolInvalid("initialization", "global prerequisite was entered more than once")
    table = evidence.tables["prerequisite"]
    prerequisite_index = 0
    for block_index, block in enumerate(BLOCKS):
        for seed_position, seed in enumerate(BLOCK_SEEDS[block]):
            evidence.archive.phase = "scene"
            scene = make_synthetic_scene(
                n_gaussians=40,
                n_cameras=12,
                image_size=IMAGE_SIZE,
                seed=seed,
            )
            if len(scene.images) != 12:
                raise ProtocolInvalid("scene", "synthetic scene view count differs")
            selected_targets = [scene.images[index].detach().clone() for index in SELECTED_VIEWS]
            # The disallowed cameras, depths, points, and GT object become unreachable here.
            del scene
            evidence.completed_scenes += 1
            for local_view, (original_view, target) in enumerate(
                zip(SELECTED_VIEWS, selected_targets, strict=True)
            ):
                evidence.archive.phase = "initialization"
                generator = torch.Generator(device="cpu").manual_seed(seed + local_view)
                rng_before = generator.get_state().clone()
                g0 = init_gaussians_2d(
                    target,
                    n=COMPONENTS,
                    grad_mix=0.7,
                    generator=generator,
                )
                rng_after = generator.get_state().clone()
                _validate_target_and_initializer(target, g0)
                evidence.archive.phase = "equivalence"
                (
                    current_raw,
                    candidate_raw,
                    current,
                    candidate,
                    current_render,
                    candidate_render,
                    current_loss,
                    candidate_loss,
                    gate_values,
                ) = _prepare_step_zero(target, g0)
                table.append(
                    block_index=block_index,
                    seed_position=seed_position,
                    seed=seed,
                    local_view=local_view,
                    original_view=original_view,
                    initializer_seed=seed + local_view,
                    target=target,
                    rng_state_before=rng_before,
                    rng_state_after=rng_after,
                    g0_xy=g0.xy,
                    g0_chol=g0.chol,
                    g0_color=g0.color,
                    g0_weight=g0.weight,
                    current_xy_raw=current_raw["xy_raw"],
                    current_diag_raw=current_raw["diag_raw"],
                    current_off_raw=current_raw["off_raw"],
                    current_color_raw=current_raw["u"],
                    current_weight_raw=current_raw["s"],
                    candidate_xy_raw=candidate_raw["xy_raw"],
                    candidate_diag_raw=candidate_raw["diag_raw"],
                    candidate_off_raw=candidate_raw["off_raw"],
                    candidate_amplitude_raw=candidate_raw["r"],
                    current_built_xy=current.xy,
                    current_built_chol=current.chol,
                    current_built_color=current.color,
                    current_built_weight=current.weight,
                    current_built_amplitude=current.weight[:, None] * current.color,
                    candidate_built_xy=candidate.xy,
                    candidate_built_chol=candidate.chol,
                    candidate_built_color=candidate.color,
                    candidate_built_weight=candidate.weight,
                    candidate_built_amplitude=candidate.weight[:, None] * candidate.color,
                    current_render=current_render,
                    candidate_render=candidate_render,
                    current_loss=current_loss,
                    candidate_loss=candidate_loss,
                    gate_values=gate_values,
                    target_sha256=_sha256_bytes(_tensor_sha256(target)),
                    g0_field_sha256=np.stack(
                        [
                            _sha256_bytes(_tensor_sha256(getattr(g0, name)))
                            for name in ("xy", "chol", "color", "weight")
                        ],
                        axis=0,
                    ),
                    rng_state_sha256=np.stack(
                        [
                            _sha256_bytes(_tensor_sha256(rng_before)),
                            _sha256_bytes(_tensor_sha256(rng_after)),
                        ],
                        axis=0,
                    ),
                )
                key = (block_index, seed_position, local_view)
                evidence.prepared[key] = PreparedFitCell(
                    prerequisite_index=prerequisite_index,
                    block_index=block_index,
                    seed_position=seed_position,
                    seed=seed,
                    local_view=local_view,
                    original_view=original_view,
                    target=target.detach().clone(),
                    g0=_clone_gaussians(g0),
                    current_raw={
                        name: value.detach().clone() for name, value in current_raw.items()
                    },
                    candidate_raw={
                        name: value.detach().clone() for name, value in candidate_raw.items()
                    },
                    current_built=_clone_gaussians(current),
                    candidate_built=_clone_gaussians(candidate),
                    current_render=current_render.detach().clone(),
                    candidate_render=candidate_render.detach().clone(),
                    current_loss=current_loss.detach().clone(),
                    candidate_loss=candidate_loss.detach().clone(),
                )
                prerequisite_index += 1
                evidence.completed_initializers += 1
                evidence.archive.completed_initializers += 1
                evidence.archive.completed_paired_views += 1
    if (
        evidence.completed_scenes != SCIENTIFIC_PLAN.scenes
        or evidence.completed_initializers != SCIENTIFIC_PLAN.initializers
        or len(evidence.prepared) != SCIENTIFIC_PLAN.initializers
        or len(table.rows) != SCIENTIFIC_PLAN.initializers
    ):
        raise ProtocolInvalid("equivalence", "global prerequisite completeness failed")
    evidence.optimization_unlocked = True


def _diagnostic_json_value(value: Any, *, phase: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_json_value(item, phase=phase) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json_value(item, phase=phase) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ProtocolInvalid(phase, f"unsupported diagnostic metadata: {type(value)!r}")


def _diagnostic_json(value: Any, *, phase: str) -> str:
    rendered = canonical_json(_diagnostic_json_value(value, phase=phase))
    if len(rendered) > 4_096:
        raise ProtocolInvalid(phase, "diagnostic metadata exceeds raw schema width")
    return rendered


def _diagnostic_failure_arrays(
    snapshot: NativeFitDiagnostic, *, prefix: str
) -> dict[str, torch.Tensor | np.ndarray]:
    """Flatten every materialized numeric callback value needed to audit a failed gate."""
    result: dict[str, torch.Tensor | np.ndarray] = {
        f"{prefix}_step": np.asarray(snapshot.step, dtype=np.int64),
        f"{prefix}_target": snapshot.target,
        f"{prefix}_built_xy": snapshot.gaussians.xy,
        f"{prefix}_built_chol": snapshot.gaussians.chol,
        f"{prefix}_built_color": snapshot.gaussians.color,
        f"{prefix}_built_weight": snapshot.gaussians.weight,
        f"{prefix}_next_lr": np.asarray(snapshot.next_lr, dtype=np.float64),
    }
    if snapshot.lr_used is not None:
        result[f"{prefix}_lr_used"] = np.asarray(snapshot.lr_used, dtype=np.float64)
    if snapshot.rendered is not None:
        result[f"{prefix}_render"] = snapshot.rendered
    if snapshot.loss is not None:
        result[f"{prefix}_loss"] = snapshot.loss
    if snapshot.initial_gaussians is not None:
        for name in ("xy", "chol", "color", "weight"):
            result[f"{prefix}_initial_{name}"] = getattr(snapshot.initial_gaussians, name)
    for name, value in snapshot.raw_parameters.items():
        result[f"{prefix}_raw_{name}"] = value
    for name, value in snapshot.gradients.items():
        if value is not None:
            result[f"{prefix}_gradient_{name}"] = value
    for name, state in snapshot.optimizer_state.items():
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            if state_name not in state:
                continue
            value = state[state_name]
            result[f"{prefix}_optimizer_{name}_{state_name}"] = (
                value if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float64)
            )
    return result


def _optimizer_state_at(
    snapshot: NativeFitDiagnostic,
    name: str,
    expected_step: int,
    *,
    phase: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    parameter = snapshot.raw_parameters[name]
    state = snapshot.optimizer_state.get(name)
    if state is None:
        raise ProtocolInvalid(phase, f"optimizer state is absent from snapshot: {name}")
    if not state:
        if expected_step != 0:
            raise ProtocolInvalid(phase, f"optimizer state missing after step zero: {name}")
        return torch.zeros_like(parameter), torch.zeros_like(parameter), 0
    allowed = {"step", "exp_avg", "exp_avg_sq"}
    if set(state) != allowed:
        raise ProtocolInvalid(
            phase,
            f"optimizer state keys differ: {name}",
            {"keys": sorted(state)},
        )
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    step_value = state["step"]
    if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
        raise ProtocolInvalid(phase, f"optimizer moments are not tensors: {name}")
    if exp_avg.shape != parameter.shape or exp_avg_sq.shape != parameter.shape:
        raise ProtocolInvalid(phase, f"optimizer moment shape differs: {name}")
    if not bool(torch.isfinite(exp_avg).all()) or not bool(torch.isfinite(exp_avg_sq).all()):
        raw_arrays = {}
        if not bool(torch.isfinite(exp_avg).all()):
            raw_arrays[f"optimizer_{name}_exp_avg"] = exp_avg
        if not bool(torch.isfinite(exp_avg_sq).all()):
            raw_arrays[f"optimizer_{name}_exp_avg_sq"] = exp_avg_sq
        raise ProtocolInvalid(
            phase,
            f"optimizer moment is non-finite: {name}",
            raw_arrays=raw_arrays,
        )
    if isinstance(step_value, torch.Tensor):
        if step_value.numel() != 1:
            raise ProtocolInvalid(phase, f"optimizer step is not scalar: {name}")
        numeric_step = float(step_value)
    else:
        numeric_step = float(step_value)
    if not math.isfinite(numeric_step) or numeric_step != int(numeric_step):
        raise ProtocolInvalid(phase, f"optimizer step is not finite integral: {name}")
    step = int(numeric_step)
    if step != expected_step:
        raise ProtocolInvalid(
            phase,
            f"optimizer step differs: {name}",
            {"actual": step, "expected": expected_step},
        )
    return exp_avg, exp_avg_sq, step


def _validate_optimizer_metadata(
    snapshot: NativeFitDiagnostic, expected_names: tuple[str, ...], *, phase: str
) -> None:
    if snapshot.optimizer_param_names != expected_names:
        raise ProtocolInvalid(
            phase,
            "optimizer parameter identity/order differs",
            {"actual": list(snapshot.optimizer_param_names), "expected": list(expected_names)},
        )
    expected_defaults = {
        "lr": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }
    if snapshot.optimizer_defaults != expected_defaults:
        raise ProtocolInvalid(
            phase,
            "effective Adam defaults differ",
            {"actual": _diagnostic_json_value(snapshot.optimizer_defaults, phase=phase)},
        )
    if len(snapshot.optimizer_param_groups) != 1:
        raise ProtocolInvalid(phase, "Adam group count differs")
    group = snapshot.optimizer_param_groups[0]
    if tuple(group.get("params", ())) != expected_names:
        raise ProtocolInvalid(phase, "Adam group parameter order differs")
    for name, expected in expected_defaults.items():
        if name == "lr":
            continue
        if group.get(name) != expected:
            raise ProtocolInvalid(phase, f"Adam group flag differs: {name}")
    if group.get("lr") != snapshot.next_lr:
        raise ProtocolInvalid(phase, "Adam group LR differs from callback state")
    if group.get("initial_lr") != 0.01:
        raise ProtocolInvalid(phase, "Adam initial_lr differs")
    scheduler = snapshot.scheduler_state
    expected_scheduler = {
        "T_max": UPDATES,
        "eta_min": 0.001,
        "base_lrs": [0.01],
    }
    for name, expected in expected_scheduler.items():
        if scheduler.get(name) != expected:
            raise ProtocolInvalid(phase, f"scheduler field differs: {name}")


def _probe_amplitude(snapshot: NativeFitDiagnostic) -> dict[str, Any]:
    if snapshot.rendered is None or snapshot.loss is None:
        raise ProtocolInvalid("appearance_only", "pre-update snapshot lacks render/loss")
    rng_before = torch.random.get_rng_state().clone()
    amplitude = (
        (snapshot.gaussians.weight[:, None] * snapshot.gaussians.color)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    probe_gaussians = Gaussians2D(
        xy=snapshot.gaussians.xy.detach().clone(),
        chol=snapshot.gaussians.chol.detach().clone(),
        color=amplitude,
        weight=torch.ones_like(snapshot.gaussians.weight),
    )
    with torch.enable_grad():
        probe_render = render_gaussians_2d(probe_gaussians, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64)
        probe_loss = torch.nn.functional.mse_loss(probe_render, snapshot.target)
        (gradient_amplitude,) = torch.autograd.grad(probe_loss, amplitude)
    probe_fields = {
        "optimization_render": snapshot.rendered,
        "optimization_loss": snapshot.loss,
        "probe_render": probe_render,
        "probe_loss": probe_loss,
        "probe_gradient": gradient_amplitude,
    }
    if not torch.equal(torch.random.get_rng_state(), rng_before):
        raise ProtocolInvalid(
            "appearance_only",
            "amplitude probe changed CPU RNG",
            raw_arrays=probe_fields,
        )
    nonfinite = {
        name: value for name, value in probe_fields.items() if not bool(torch.isfinite(value).all())
    }
    if nonfinite:
        raise ProtocolInvalid(
            "appearance_only", "amplitude probe is non-finite", raw_arrays=nonfinite
        )
    difference = (probe_render.detach() - snapshot.rendered).abs()
    maximum_absolute = float(difference.max())
    numerator = float(difference.to(torch.float64).sum())
    denominator = float(snapshot.rendered.abs().to(torch.float64).sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ProtocolInvalid(
            "appearance_only",
            "probe render denominator is invalid",
            raw_arrays=probe_fields,
        )
    relative = numerator / denominator
    optimization_loss = float(snapshot.loss)
    probe_loss_value = float(probe_loss)
    if not math.isfinite(optimization_loss) or optimization_loss <= 0:
        raise ProtocolInvalid(
            "appearance_only",
            "probe loss denominator is invalid",
            raw_arrays=probe_fields,
        )
    loss_absolute = abs(probe_loss_value - optimization_loss)
    loss_relative = loss_absolute / optimization_loss
    if maximum_absolute > 2e-5 or relative > 2e-6 or loss_absolute > 2e-6 or loss_relative > 2e-5:
        raise ProtocolInvalid(
            "appearance_only",
            "direct-amplitude probe equivalence failed",
            {
                "render_maximum_absolute": maximum_absolute,
                "render_relative_l1": relative,
                "loss_absolute": loss_absolute,
                "loss_relative": loss_relative,
            },
            raw_arrays=probe_fields,
        )
    return {
        "gradient_amplitude": gradient_amplitude.detach(),
        "values": np.asarray(
            [
                maximum_absolute,
                numerator,
                denominator,
                relative,
                probe_loss_value,
                loss_absolute,
                optimization_loss,
                loss_relative,
                snapshot.rendered.numel(),
            ],
            dtype=np.float64,
        ),
    }


def _learned_saturation_row(
    raw: torch.Tensor, output: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    diagnostics = saturation_diagnostics(raw, output)
    counts = np.asarray(
        [
            int(diagnostics["raw_abs_ge_8_count"]),
            int(diagnostics["raw_count"]),
            int(diagnostics["output_low_count"]),
            int(diagnostics["output_high_count"]),
            int(diagnostics["output_count"]),
            int(diagnostics["derivative_le_1e_4_count"]),
        ],
        dtype=np.int64,
    )
    fractions = np.asarray(
        [
            float(diagnostics["raw_abs_ge_8_fraction"]),
            float(diagnostics["output_low_fraction"]),
            float(diagnostics["output_high_fraction"]),
            float(diagnostics["derivative_le_1e_4_fraction"]),
        ],
        dtype=np.float64,
    )
    return counts, fractions


def _amplitude_saturation_row(
    output: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = output.to(torch.float64)
    count = value.numel()
    low = int((value <= 1e-3).sum())
    high = int((value >= 1.0 - 1e-3).sum())
    counts = np.asarray([0, 0, low, high, count, 0], dtype=np.int64)
    count_defined = np.asarray([False, False, True, True, True, False], dtype=np.bool_)
    fractions = np.asarray([0.0, low / count, high / count, 0.0], dtype=np.float64)
    fraction_defined = np.asarray([False, True, True, False], dtype=np.bool_)
    return counts, count_defined, fractions, fraction_defined


def _checkpoint_saturation(
    arm: str,
    snapshot: NativeFitDiagnostic,
) -> dict[str, np.ndarray]:
    amplitude = snapshot.gaussians.weight[:, None] * snapshot.gaussians.color
    if arm == ARMS[0]:
        color_counts, color_fractions = _learned_saturation_row(
            snapshot.raw_parameters["color_raw"], snapshot.gaussians.color
        )
        weight_counts, weight_fractions = _learned_saturation_row(
            snapshot.raw_parameters["weight_raw"], snapshot.gaussians.weight
        )
        (
            amplitude_counts,
            amplitude_count_defined,
            amplitude_fractions,
            amplitude_fraction_defined,
        ) = _amplitude_saturation_row(amplitude)
        return {
            "saturation_counts": np.stack([color_counts, weight_counts, amplitude_counts], axis=0),
            "saturation_count_defined": np.stack(
                [
                    np.ones(6, dtype=np.bool_),
                    np.ones(6, dtype=np.bool_),
                    amplitude_count_defined,
                ],
                axis=0,
            ),
            "saturation_fractions": np.stack(
                [color_fractions, weight_fractions, amplitude_fractions], axis=0
            ),
            "saturation_fraction_defined": np.stack(
                [
                    np.ones(4, dtype=np.bool_),
                    np.ones(4, dtype=np.bool_),
                    amplitude_fraction_defined,
                ],
                axis=0,
            ),
            "histograms": np.stack(
                [
                    histogram_counts(snapshot.gaussians.color).cpu().numpy(),
                    histogram_counts(snapshot.gaussians.weight).cpu().numpy(),
                    histogram_counts(amplitude).cpu().numpy(),
                ],
                axis=0,
            ),
        }
    counts, fractions = _learned_saturation_row(
        snapshot.raw_parameters["amplitude_raw"], snapshot.gaussians.color
    )
    return {
        "saturation_counts": counts[None],
        "saturation_count_defined": np.ones((1, 6), dtype=np.bool_),
        "saturation_fractions": fractions[None],
        "saturation_fraction_defined": np.ones((1, 4), dtype=np.bool_),
        "histograms": histogram_counts(amplitude).cpu().numpy()[None],
    }


def _snapshot_expected_raw(cell: PreparedFitCell, arm: str) -> dict[str, torch.Tensor]:
    raw = cell.current_raw if arm == ARMS[0] else cell.candidate_raw
    result = {
        "xy_raw": raw["xy_raw"],
        "diag_raw": raw["diag_raw"],
        "off_raw": raw["off_raw"],
    }
    if arm == ARMS[0]:
        result.update({"color_raw": raw["u"], "weight_raw": raw["s"]})
    else:
        result["amplitude_raw"] = raw["r"]
    return result


class NativeEvidenceRecorder:
    """Fail-closed adapter from the production native callback to frozen raw tables."""

    def __init__(
        self,
        evidence: ScientificEvidence,
        cell: PreparedFitCell,
        *,
        fit_index: int,
        arm: str,
        geometry_frozen: bool,
    ) -> None:
        self.evidence = evidence
        self.cell = cell
        self.fit_index = fit_index
        self.arm = arm
        self.geometry_frozen = geometry_frozen
        self.expected_names = (
            (("color_raw", "weight_raw") if arm == ARMS[0] else ("amplitude_raw",))
            if geometry_frozen
            else (
                ("xy_raw", "diag_raw", "off_raw", "color_raw", "weight_raw")
                if arm == ARMS[0]
                else ("xy_raw", "diag_raw", "off_raw", "amplitude_raw")
            )
        )
        self.expected_events: list[tuple[str, int]] = [("initial", 0)]
        for completed_step in range(UPDATES):
            self.expected_events.append(("pre_update", completed_step))
            self.expected_events.append(("post_update", completed_step + 1))
            if completed_step + 1 in POSITIVE_CHECKPOINTS:
                self.expected_events.append(("checkpoint", completed_step + 1))
        self.event_position = 0
        self.pending_pre: NativeFitDiagnostic | None = None
        self.pending_probe: dict[str, Any] | None = None
        self.last_post: NativeFitDiagnostic | None = None
        self.lr_trace: list[float] = []
        self.next_lr_trace: list[float] = []
        self.checkpoint_count = 0
        self.initial_defaults_json = ""
        self.initial_groups_json = ""
        self.initial_scheduler_json = ""
        self.final_checkpoint: NativeFitDiagnostic | None = None

    def __call__(self, snapshot: NativeFitDiagnostic) -> None:
        try:
            self._consume_snapshot(snapshot)
        except ProtocolInvalid as error:
            reached = _diagnostic_failure_arrays(snapshot, prefix="snapshot")
            if self.pending_pre is not None and self.pending_pre is not snapshot:
                reached.update(_diagnostic_failure_arrays(self.pending_pre, prefix="pending_pre"))
            reached.update({f"gate_{name}": value for name, value in error.raw_arrays.items()})
            raise ProtocolInvalid(
                error.phase,
                error.reason,
                error.evidence,
                raw_arrays=reached,
            ) from error

    def _consume_snapshot(self, snapshot: NativeFitDiagnostic) -> None:
        if self.event_position >= len(self.expected_events):
            raise ProtocolInvalid(self.evidence.archive.phase, "extra native diagnostic event")
        expected_event, expected_step = self.expected_events[self.event_position]
        if (snapshot.event, snapshot.step) != (expected_event, expected_step):
            raise ProtocolInvalid(
                self.evidence.archive.phase,
                "native diagnostic event order differs",
                {
                    "actual": [snapshot.event, snapshot.step],
                    "expected": [expected_event, expected_step],
                    "fit_index": self.fit_index,
                },
            )
        if snapshot.appearance_parameterization != self.arm:
            raise ProtocolInvalid(self.evidence.archive.phase, "callback arm identity differs")
        if snapshot.geometry_frozen is not self.geometry_frozen:
            raise ProtocolInvalid(
                self.evidence.archive.phase, "callback geometry-freeze identity differs"
            )
        _validate_optimizer_metadata(
            snapshot, self.expected_names, phase=self.evidence.archive.phase
        )
        if not torch.equal(snapshot.target, self.cell.target):
            raise ProtocolInvalid(self.evidence.archive.phase, "callback target differs")
        scheduler_last_epoch = int(snapshot.scheduler_state.get("last_epoch", -1))
        scheduler_step_count = int(snapshot.scheduler_state.get("_step_count", -1))
        self.evidence.tables["fit_event"].append(
            fit_index=self.fit_index,
            event_code=EVENT_CODES[snapshot.event],
            step=snapshot.step,
            lr_used=0.0 if snapshot.lr_used is None else snapshot.lr_used,
            lr_used_defined=snapshot.lr_used is not None,
            next_lr=snapshot.next_lr,
            scheduler_last_epoch=scheduler_last_epoch,
            scheduler_step_count=scheduler_step_count,
        )
        self.evidence.completed_callback_events += 1
        self.evidence.archive.completed_callback_events += 1
        if snapshot.event == "initial":
            self._record_initial(snapshot)
        elif snapshot.event == "pre_update":
            self._record_pre(snapshot)
        elif snapshot.event == "post_update":
            self._record_post(snapshot)
        else:
            self._record_checkpoint(snapshot)
        self.event_position += 1

    def _record_initial(self, snapshot: NativeFitDiagnostic) -> None:
        if snapshot.lr_used is not None or snapshot.step != 0:
            raise ProtocolInvalid(self.evidence.archive.phase, "initial LR/step contract differs")
        if snapshot.initial_gaussians is None:
            raise ProtocolInvalid(self.evidence.archive.phase, "initial callback lacks g0")
        for field_name in ("xy", "chol", "color", "weight"):
            _assert_tensor_equal(
                f"callback/g0/{field_name}",
                getattr(snapshot.initial_gaussians, field_name),
                getattr(self.cell.g0, field_name),
                self.evidence.archive.phase,
            )
        if (
            snapshot.initializer_rng_state_before is not None
            or snapshot.initializer_rng_state_after is not None
        ):
            raise ProtocolInvalid(
                self.evidence.archive.phase,
                "paired from-initialization callback unexpectedly carries initializer RNG state",
            )
        expected_raw = _snapshot_expected_raw(self.cell, self.arm)
        if set(snapshot.raw_parameters) != set(expected_raw):
            raise ProtocolInvalid(self.evidence.archive.phase, "initial raw-parameter names differ")
        for name, expected in expected_raw.items():
            _assert_tensor_equal(
                f"callback/initial/{name}",
                snapshot.raw_parameters[name],
                expected,
                self.evidence.archive.phase,
            )
        expected_built = (
            self.cell.current_built if self.arm == ARMS[0] else self.cell.candidate_built
        )
        for field_name in ("xy", "chol", "color", "weight"):
            _assert_tensor_equal(
                f"callback/initial/built/{field_name}",
                getattr(snapshot.gaussians, field_name),
                getattr(expected_built, field_name),
                self.evidence.archive.phase,
            )
        expected_render = (
            self.cell.current_render if self.arm == ARMS[0] else self.cell.candidate_render
        )
        expected_loss = self.cell.current_loss if self.arm == ARMS[0] else self.cell.candidate_loss
        if snapshot.rendered is None or snapshot.loss is None:
            raise ProtocolInvalid(self.evidence.archive.phase, "initial callback lacks render/loss")
        _assert_tensor_equal(
            "callback/initial/render",
            snapshot.rendered,
            expected_render,
            self.evidence.archive.phase,
        )
        _assert_tensor_equal(
            "callback/initial/loss", snapshot.loss, expected_loss, self.evidence.archive.phase
        )
        for name in self.expected_names:
            _optimizer_state_at(snapshot, name, 0, phase=self.evidence.archive.phase)
        self.initial_defaults_json = _diagnostic_json(
            snapshot.optimizer_defaults, phase=self.evidence.archive.phase
        )
        self.initial_groups_json = _diagnostic_json(
            snapshot.optimizer_param_groups, phase=self.evidence.archive.phase
        )
        self.initial_scheduler_json = _diagnostic_json(
            snapshot.scheduler_state, phase=self.evidence.archive.phase
        )
        self.next_lr_trace.append(snapshot.next_lr)
        if self.geometry_frozen:
            self._record_mechanism_geometry(snapshot)
        else:
            self._record_joint_undefined()
        self._append_checkpoint(snapshot)

    def _record_pre(self, snapshot: NativeFitDiagnostic) -> None:
        if self.pending_pre is not None or snapshot.lr_used is None:
            raise ProtocolInvalid(self.evidence.archive.phase, "pre-update pairing/LR differs")
        pre_fields = {
            "pre_render": snapshot.rendered,
            "pre_loss": snapshot.loss,
            "pre_built_xy": snapshot.gaussians.xy,
            "pre_built_chol": snapshot.gaussians.chol,
            "pre_built_color": snapshot.gaussians.color,
            "pre_built_weight": snapshot.gaussians.weight,
        }
        nonfinite_pre = {
            name: value
            for name, value in pre_fields.items()
            if value is not None and not bool(torch.isfinite(value).all())
        }
        if nonfinite_pre:
            raise ProtocolInvalid(
                self.evidence.archive.phase,
                "pre-update state is non-finite",
                raw_arrays=nonfinite_pre,
            )
        for name in self.expected_names:
            gradient = snapshot.gradients.get(name)
            if gradient is None:
                raise ProtocolInvalid(
                    self.evidence.archive.phase, f"pre-update gradient invalid: {name}"
                )
            if not bool(torch.isfinite(gradient).all()):
                raise ProtocolInvalid(
                    self.evidence.archive.phase,
                    f"pre-update gradient is non-finite: {name}",
                    raw_arrays={f"gradient_{name}": gradient},
                )
            _optimizer_state_at(snapshot, name, snapshot.step, phase=self.evidence.archive.phase)
        if self.geometry_frozen:
            for name in ("xy_raw", "diag_raw", "off_raw"):
                if snapshot.gradients.get(name) is not None:
                    raise ProtocolInvalid(
                        self.evidence.archive.phase, "frozen geometry received gradient"
                    )
        self.pending_pre = snapshot
        self.pending_probe = _probe_amplitude(snapshot) if self.geometry_frozen else None
        self.lr_trace.append(snapshot.lr_used)

    def _record_post(self, snapshot: NativeFitDiagnostic) -> None:
        if self.pending_pre is None:
            raise ProtocolInvalid(
                self.evidence.archive.phase, "post-update lacks preceding pre-update"
            )
        if snapshot.step != self.pending_pre.step + 1:
            raise ProtocolInvalid(self.evidence.archive.phase, "post-update step pairing differs")
        for name in self.expected_names:
            _optimizer_state_at(snapshot, name, snapshot.step, phase=self.evidence.archive.phase)
        if self.geometry_frozen:
            self._record_mechanism_update(self.pending_pre, snapshot, self.pending_probe)
            self._record_mechanism_geometry(snapshot)
        elif snapshot.step in POSITIVE_CHECKPOINTS:
            self._record_joint_transition(self.pending_pre, snapshot)
        self.next_lr_trace.append(snapshot.next_lr)
        self.last_post = snapshot
        self.pending_pre = None
        self.pending_probe = None
        self.evidence.completed_updates += 1
        self.evidence.archive.completed_updates += 1

    def _record_checkpoint(self, snapshot: NativeFitDiagnostic) -> None:
        if self.last_post is None or snapshot.step != self.last_post.step:
            raise ProtocolInvalid(
                self.evidence.archive.phase, "checkpoint lacks matching post-update"
            )
        for name in snapshot.raw_parameters:
            _assert_tensor_equal(
                f"checkpoint/post/{name}",
                snapshot.raw_parameters[name],
                self.last_post.raw_parameters[name],
                self.evidence.archive.phase,
            )
        self._append_checkpoint(snapshot)

    def _record_mechanism_geometry(self, snapshot: NativeFitDiagnostic) -> None:
        expected_raw = _snapshot_expected_raw(self.cell, self.arm)
        for name in ("xy_raw", "diag_raw", "off_raw"):
            _assert_tensor_equal(
                f"frozen/{name}",
                snapshot.raw_parameters[name],
                expected_raw[name],
                self.evidence.archive.phase,
            )
        expected_built = (
            self.cell.current_built if self.arm == ARMS[0] else self.cell.candidate_built
        )
        _assert_tensor_equal(
            "frozen/built_xy", snapshot.gaussians.xy, expected_built.xy, self.evidence.archive.phase
        )
        _assert_tensor_equal(
            "frozen/built_chol",
            snapshot.gaussians.chol,
            expected_built.chol,
            self.evidence.archive.phase,
        )
        if self.arm == ARMS[1] and not torch.equal(
            snapshot.gaussians.weight, torch.ones_like(snapshot.gaussians.weight)
        ):
            raise ProtocolInvalid("appearance_only", "candidate mechanism weight differs from one")
        self.evidence.tables["mechanism_geometry"].append(
            fit_index=self.fit_index,
            step=snapshot.step,
            xy_raw=snapshot.raw_parameters["xy_raw"],
            diag_raw=snapshot.raw_parameters["diag_raw"],
            off_raw=snapshot.raw_parameters["off_raw"],
            built_xy=snapshot.gaussians.xy,
            built_chol=snapshot.gaussians.chol,
            built_weight=snapshot.gaussians.weight,
        )

    def _record_mechanism_update(
        self,
        pre: NativeFitDiagnostic,
        post: NativeFitDiagnostic,
        probe: Mapping[str, Any] | None,
    ) -> None:
        if probe is None or pre.lr_used is None:
            raise ProtocolInvalid("appearance_only", "mechanism update lacks probe/LR")
        grad_a = probe["gradient_amplitude"]
        if self.arm == ARMS[0]:
            self._record_current_mechanism_update(pre, post, grad_a, probe["values"])
        else:
            self._record_candidate_mechanism_update(pre, post, grad_a, probe["values"])

    def _record_current_mechanism_update(
        self,
        pre: NativeFitDiagnostic,
        post: NativeFitDiagnostic,
        grad_a: torch.Tensor,
        probe_values: np.ndarray,
    ) -> None:
        color = pre.gaussians.color
        weight = pre.gaussians.weight
        actual_color_grad = pre.gradients["color_raw"]
        actual_weight_grad = pre.gradients["weight_raw"]
        if actual_color_grad is None or actual_weight_grad is None or pre.lr_used is None:
            raise ProtocolInvalid("appearance_only", "current mechanism gradient/LR is absent")
        expected = chain_rule_expected(self.arm, grad_a, weight, color)
        color_grad_error = _assert_error(
            "appearance_only",
            "current color chain rule",
            actual_color_grad,
            expected["u"],
            CHAIN_RULE_ABSOLUTE_LIMIT,
            CHAIN_RULE_RELATIVE_LIMIT,
            CHAIN_RULE_RELATIVE_MAGNITUDE_THRESHOLD,
        )
        weight_grad_error = _assert_error(
            "appearance_only",
            "current weight chain rule",
            actual_weight_grad,
            expected["s"],
            CHAIN_RULE_ABSOLUTE_LIMIT,
            CHAIN_RULE_RELATIVE_LIMIT,
            CHAIN_RULE_RELATIVE_MAGNITUDE_THRESHOLD,
        )
        color_avg_before, color_sq_before, color_step_before = _optimizer_state_at(
            pre, "color_raw", pre.step, phase=self.evidence.archive.phase
        )
        weight_avg_before, weight_sq_before, weight_step_before = _optimizer_state_at(
            pre, "weight_raw", pre.step, phase=self.evidence.archive.phase
        )
        color_avg_after, color_sq_after, color_step_after = _optimizer_state_at(
            post, "color_raw", post.step, phase=self.evidence.archive.phase
        )
        weight_avg_after, weight_sq_after, weight_step_after = _optimizer_state_at(
            post, "weight_raw", post.step, phase=self.evidence.archive.phase
        )
        reconstructed_color = adam_reconstruct(
            pre.raw_parameters["color_raw"],
            actual_color_grad,
            color_avg_before,
            color_sq_before,
            color_step_before,
            pre.lr_used,
        )
        reconstructed_weight = adam_reconstruct(
            pre.raw_parameters["weight_raw"],
            actual_weight_grad,
            weight_avg_before,
            weight_sq_before,
            weight_step_before,
            pre.lr_used,
        )
        if not torch.equal(color_avg_after, reconstructed_color["exp_avg_after"]):
            raise ProtocolInvalid("appearance_only", "current color first moment differs")
        if not torch.equal(color_sq_after, reconstructed_color["exp_avg_sq_after"]):
            raise ProtocolInvalid("appearance_only", "current color second moment differs")
        if not torch.equal(weight_avg_after, reconstructed_weight["exp_avg_after"]):
            raise ProtocolInvalid("appearance_only", "current weight first moment differs")
        if not torch.equal(weight_sq_after, reconstructed_weight["exp_avg_sq_after"]):
            raise ProtocolInvalid("appearance_only", "current weight second moment differs")
        displacement_color = post.raw_parameters["color_raw"] - pre.raw_parameters["color_raw"]
        displacement_weight = post.raw_parameters["weight_raw"] - pre.raw_parameters["weight_raw"]
        color_displacement_error = _assert_error(
            "appearance_only",
            "current color Adam displacement",
            displacement_color,
            reconstructed_color["displacement"],
            2e-7,
            2e-5,
        )
        weight_displacement_error = _assert_error(
            "appearance_only",
            "current weight Adam displacement",
            displacement_weight,
            reconstructed_weight["displacement"],
            2e-7,
            2e-5,
        )
        jacobian = current_jacobian(weight, color)
        diagnostics = jacobian_diagnostics(jacobian, current=True)
        analytic = analytic_current_null(
            weight, color, diagnostics["null_vector"], diagnostics["rank"]
        )
        theta_gradient = torch.cat([actual_color_grad, actual_weight_grad[:, None]], dim=-1)
        theta_displacement = torch.cat([displacement_color, displacement_weight[:, None]], dim=-1)
        null = null_update_rows(jacobian, theta_gradient, theta_displacement)
        gradient_absolute = null["gradient_dot"].abs()
        if bool((gradient_absolute > 2e-6).any()):
            raise ProtocolInvalid(
                "appearance_only", "current gradient/null absolute identity failed"
            )
        cosine_check = null["gradient_cosine_defined"] & (null["gradient_cosine"] > 2e-5)
        if bool(cosine_check.any()):
            raise ProtocolInvalid("appearance_only", "current gradient/null cosine identity failed")
        self.evidence.tables["mechanism_current"].append(
            fit_index=self.fit_index,
            completed_step=pre.step,
            update_number=post.step,
            color_raw_pre=pre.raw_parameters["color_raw"],
            weight_raw_pre=pre.raw_parameters["weight_raw"],
            built_color=color,
            built_weight=weight,
            built_amplitude=weight[:, None] * color,
            gradient_color=actual_color_grad,
            gradient_weight=actual_weight_grad,
            gradient_amplitude=grad_a,
            objective_loss=pre.loss,
            probe_values=probe_values,
            exp_avg_color_before=color_avg_before,
            exp_avg_sq_color_before=color_sq_before,
            exp_avg_weight_before=weight_avg_before,
            exp_avg_sq_weight_before=weight_sq_before,
            step_before=np.asarray([color_step_before, weight_step_before], dtype=np.int64),
            state_present_before=np.asarray(
                [
                    bool(pre.optimizer_state["color_raw"]),
                    bool(pre.optimizer_state["weight_raw"]),
                ],
                dtype=np.bool_,
            ),
            lr_used=pre.lr_used,
            color_raw_after=post.raw_parameters["color_raw"],
            weight_raw_after=post.raw_parameters["weight_raw"],
            exp_avg_color_after=color_avg_after,
            exp_avg_sq_color_after=color_sq_after,
            exp_avg_weight_after=weight_avg_after,
            exp_avg_sq_weight_after=weight_sq_after,
            step_after=np.asarray([color_step_after, weight_step_after], dtype=np.int64),
            state_present_after=np.asarray(
                [
                    bool(post.optimizer_state["color_raw"]),
                    bool(post.optimizer_state["weight_raw"]),
                ],
                dtype=np.bool_,
            ),
            displacement_color=displacement_color,
            displacement_weight=displacement_weight,
            expected_gradient_color=expected["u"],
            expected_gradient_weight=expected["s"],
            reconstructed_displacement_color=reconstructed_color["displacement"],
            reconstructed_displacement_weight=reconstructed_weight["displacement"],
            equation_errors=np.asarray(
                [
                    *color_grad_error,
                    *weight_grad_error,
                    *color_displacement_error,
                    *weight_displacement_error,
                ],
                dtype=np.float64,
            ),
            jacobian=jacobian,
            singular_values=diagnostics["singular_values"],
            vh=diagnostics["vh"],
            rank=diagnostics["rank"],
            null_vector=diagnostics["null_vector"],
            null_residual=diagnostics["null_residual"],
            analytic_null_vector=analytic["vector"],
            analytic_null_alignment=analytic["alignment"],
            analytic_null_defined=analytic["defined"],
            null_dot=null["dot"],
            update_norm=null["update_norm"],
            null_eligible=null["eligible"],
            null_fraction=null["null_fraction"],
            squared_projection=null["squared_projection"],
            squared_update_norm=null["squared_update_norm"],
            gradient_dot=null["gradient_dot"],
            gradient_norm=null["gradient_norm"],
            gradient_cosine=null["gradient_cosine"],
            gradient_cosine_defined=null["gradient_cosine_defined"],
        )
        self.evidence.archive.completed_mechanism_updates += 1
        self.evidence.archive.completed_null_rows += COMPONENTS

    def _record_candidate_mechanism_update(
        self,
        pre: NativeFitDiagnostic,
        post: NativeFitDiagnostic,
        grad_a: torch.Tensor,
        probe_values: np.ndarray,
    ) -> None:
        actual_gradient = pre.gradients["amplitude_raw"]
        if actual_gradient is None or pre.lr_used is None:
            raise ProtocolInvalid("appearance_only", "candidate mechanism gradient/LR is absent")
        expected_gradient = chain_rule_expected(
            self.arm, grad_a, pre.gaussians.weight, pre.gaussians.color
        )["r"]
        gradient_error = _assert_error(
            "appearance_only",
            "candidate amplitude chain rule",
            actual_gradient,
            expected_gradient,
            CHAIN_RULE_ABSOLUTE_LIMIT,
            CHAIN_RULE_RELATIVE_LIMIT,
            CHAIN_RULE_RELATIVE_MAGNITUDE_THRESHOLD,
        )
        avg_before, sq_before, step_before = _optimizer_state_at(
            pre, "amplitude_raw", pre.step, phase=self.evidence.archive.phase
        )
        avg_after, sq_after, step_after = _optimizer_state_at(
            post, "amplitude_raw", post.step, phase=self.evidence.archive.phase
        )
        reconstructed = adam_reconstruct(
            pre.raw_parameters["amplitude_raw"],
            actual_gradient,
            avg_before,
            sq_before,
            step_before,
            pre.lr_used,
        )
        if not torch.equal(avg_after, reconstructed["exp_avg_after"]):
            raise ProtocolInvalid("appearance_only", "candidate first moment differs")
        if not torch.equal(sq_after, reconstructed["exp_avg_sq_after"]):
            raise ProtocolInvalid("appearance_only", "candidate second moment differs")
        displacement = post.raw_parameters["amplitude_raw"] - pre.raw_parameters["amplitude_raw"]
        displacement_error = _assert_error(
            "appearance_only",
            "candidate Adam displacement",
            displacement,
            reconstructed["displacement"],
            2e-7,
            2e-5,
        )
        self.evidence.tables["mechanism_candidate"].append(
            fit_index=self.fit_index,
            completed_step=pre.step,
            update_number=post.step,
            amplitude_raw_pre=pre.raw_parameters["amplitude_raw"],
            built_color=pre.gaussians.color,
            built_weight=pre.gaussians.weight,
            built_amplitude=pre.gaussians.weight[:, None] * pre.gaussians.color,
            gradient_amplitude_raw=actual_gradient,
            gradient_amplitude=grad_a,
            objective_loss=pre.loss,
            probe_values=probe_values,
            exp_avg_before=avg_before,
            exp_avg_sq_before=sq_before,
            step_before=step_before,
            state_present_before=bool(pre.optimizer_state["amplitude_raw"]),
            lr_used=pre.lr_used,
            amplitude_raw_after=post.raw_parameters["amplitude_raw"],
            exp_avg_after=avg_after,
            exp_avg_sq_after=sq_after,
            step_after=step_after,
            state_present_after=bool(post.optimizer_state["amplitude_raw"]),
            displacement=displacement,
            expected_gradient=expected_gradient,
            reconstructed_displacement=reconstructed["displacement"],
            equation_errors=np.asarray([*gradient_error, *displacement_error], dtype=np.float64),
        )
        self.evidence.archive.completed_mechanism_updates += 1

    def _record_joint_transition(self, pre: NativeFitDiagnostic, post: NativeFitDiagnostic) -> None:
        if pre.lr_used is None:
            raise ProtocolInvalid("joint", "joint transition lacks LR")
        row: dict[str, Any] = {
            "fit_index": self.fit_index,
            "step": post.step,
            "defined": True,
            "lr_used": pre.lr_used,
        }
        step_before: list[int] = []
        step_after: list[int] = []
        present_before: list[bool] = []
        present_after: list[bool] = []
        for name in self.expected_names:
            gradient = pre.gradients[name]
            if gradient is None:
                raise ProtocolInvalid("joint", f"joint gradient absent: {name}")
            avg_before, sq_before, before = _optimizer_state_at(
                pre, name, pre.step, phase=self.evidence.archive.phase
            )
            avg_after, sq_after, after = _optimizer_state_at(
                post, name, post.step, phase=self.evidence.archive.phase
            )
            reconstructed = adam_reconstruct(
                pre.raw_parameters[name],
                gradient,
                avg_before,
                sq_before,
                before,
                pre.lr_used,
            )
            if not torch.equal(avg_after, reconstructed["exp_avg_after"]):
                raise ProtocolInvalid("joint", f"joint first moment differs: {name}")
            if not torch.equal(sq_after, reconstructed["exp_avg_sq_after"]):
                raise ProtocolInvalid("joint", f"joint second moment differs: {name}")
            displacement = post.raw_parameters[name] - pre.raw_parameters[name]
            _assert_error(
                "joint",
                f"joint Adam displacement: {name}",
                displacement,
                reconstructed["displacement"],
                2e-7,
                2e-5,
            )
            row.update(
                {
                    f"pre_{name}": pre.raw_parameters[name],
                    f"gradient_{name}": gradient,
                    f"exp_avg_before_{name}": avg_before,
                    f"exp_avg_sq_before_{name}": sq_before,
                    f"post_{name}": post.raw_parameters[name],
                    f"exp_avg_after_{name}": avg_after,
                    f"exp_avg_sq_after_{name}": sq_after,
                    f"displacement_{name}": displacement,
                }
            )
            step_before.append(before)
            step_after.append(after)
            present_before.append(bool(pre.optimizer_state[name]))
            present_after.append(bool(post.optimizer_state[name]))
        row["step_before"] = np.asarray(step_before, dtype=np.int64)
        row["step_after"] = np.asarray(step_after, dtype=np.int64)
        row["state_present_before"] = np.asarray(present_before, dtype=np.bool_)
        row["state_present_after"] = np.asarray(present_after, dtype=np.bool_)
        table_name = "joint_current" if self.arm == ARMS[0] else "joint_candidate"
        self.evidence.tables[table_name].append(**row)
        self.evidence.archive.completed_joint_positive_checkpoint_updates += 1

    def _record_joint_undefined(self) -> None:
        table_name = "joint_current" if self.arm == ARMS[0] else "joint_candidate"
        spec = SCIENTIFIC_TABLE_SPECS[table_name]
        row: dict[str, Any] = {}
        for name, field_spec in spec.fields.items():
            if name == "fit_index":
                row[name] = self.fit_index
            elif name == "step":
                row[name] = 0
            elif name == "defined":
                row[name] = False
            else:
                row[name] = np.zeros(field_spec.shape, dtype=np.dtype(field_spec.dtype))
        self.evidence.tables[table_name].append(**row)

    def _append_checkpoint(self, snapshot: NativeFitDiagnostic) -> None:
        if snapshot.step not in CHECKPOINTS:
            raise ProtocolInvalid(self.evidence.archive.phase, "unexpected checkpoint step")
        if snapshot.rendered is None or snapshot.loss is None:
            raise ProtocolInvalid(self.evidence.archive.phase, "checkpoint lacks render/loss")
        checkpoint_fields = {
            "checkpoint_render": snapshot.rendered,
            "checkpoint_loss": snapshot.loss,
            "checkpoint_built_xy": snapshot.gaussians.xy,
            "checkpoint_built_chol": snapshot.gaussians.chol,
            "checkpoint_built_color": snapshot.gaussians.color,
            "checkpoint_built_weight": snapshot.gaussians.weight,
        }
        nonfinite_checkpoint = {
            name: value
            for name, value in checkpoint_fields.items()
            if not bool(torch.isfinite(value).all())
        }
        if nonfinite_checkpoint:
            raise ProtocolInvalid(
                self.evidence.archive.phase,
                "checkpoint state is non-finite",
                raw_arrays=nonfinite_checkpoint,
            )
        if self.arm == ARMS[1] and not torch.equal(
            snapshot.gaussians.weight, torch.ones_like(snapshot.gaussians.weight)
        ):
            raise ProtocolInvalid(
                self.evidence.archive.phase, "candidate checkpoint weight differs"
            )
        rerendered = render_gaussians_2d(snapshot.gaussians, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64)
        _assert_tensor_equal(
            "checkpoint/fresh_render", snapshot.rendered, rerendered, self.evidence.archive.phase
        )
        metrics = render_metrics(
            snapshot.rendered, snapshot.target, phase=self.evidence.archive.phase
        )
        checkpoint_index = len(self.evidence.tables["checkpoint"].rows)
        self.evidence.tables["checkpoint"].append(
            checkpoint_index=checkpoint_index,
            fit_index=self.fit_index,
            step=snapshot.step,
            xy_raw=snapshot.raw_parameters["xy_raw"],
            diag_raw=snapshot.raw_parameters["diag_raw"],
            off_raw=snapshot.raw_parameters["off_raw"],
            built_xy=snapshot.gaussians.xy,
            built_chol=snapshot.gaussians.chol,
            built_color=snapshot.gaussians.color,
            built_weight=snapshot.gaussians.weight,
            built_amplitude=snapshot.gaussians.weight[:, None] * snapshot.gaussians.color,
            render=snapshot.rendered,
            target=snapshot.target,
            clamped_render=metrics["clamped"],
            below_mask=metrics["below_mask"],
            above_mask=metrics["above_mask"],
            objective_loss=snapshot.loss,
            sse=metrics["sse"],
            channel_count=metrics["channel_count"],
            raw_mse=metrics["raw_mse_float64"],
            psnr=np.float32(metrics["psnr"]),
            ssim=np.float32(metrics["ssim"]),
            below_count=metrics["below_count"],
            above_count=metrics["above_count"],
            below_fraction=metrics["below_fraction"],
            above_fraction=metrics["above_fraction"],
        )
        saturation = _checkpoint_saturation(self.arm, snapshot)
        if self.arm == ARMS[0]:
            jacobian = current_jacobian(snapshot.gaussians.weight, snapshot.gaussians.color)
            diagnostics = jacobian_diagnostics(jacobian, current=True)
            analytic = analytic_current_null(
                snapshot.gaussians.weight,
                snapshot.gaussians.color,
                diagnostics["null_vector"],
                diagnostics["rank"],
            )
            self.evidence.tables["checkpoint_current"].append(
                checkpoint_index=checkpoint_index,
                fit_index=self.fit_index,
                step=snapshot.step,
                color_raw=snapshot.raw_parameters["color_raw"],
                weight_raw=snapshot.raw_parameters["weight_raw"],
                jacobian=jacobian,
                singular_values=diagnostics["singular_values"],
                vh=diagnostics["vh"],
                largest=diagnostics["largest"],
                tau=diagnostics["tau"],
                rank=diagnostics["rank"],
                smallest_positive=diagnostics["smallest_positive"],
                smallest_positive_defined=diagnostics["smallest_positive_defined"],
                condition=diagnostics["condition"],
                condition_defined=diagnostics["condition_defined"],
                weakly_responsive=diagnostics["weakly_responsive"],
                null_vector=diagnostics["null_vector"],
                null_residual=diagnostics["null_residual"],
                analytic_null_vector=analytic["vector"],
                analytic_null_alignment=analytic["alignment"],
                analytic_null_defined=analytic["defined"],
                **saturation,
            )
        else:
            jacobian = candidate_jacobian(snapshot.gaussians.color)
            diagnostics = jacobian_diagnostics(jacobian, current=False)
            self.evidence.tables["checkpoint_candidate"].append(
                checkpoint_index=checkpoint_index,
                fit_index=self.fit_index,
                step=snapshot.step,
                amplitude_raw=snapshot.raw_parameters["amplitude_raw"],
                jacobian=jacobian,
                singular_values=diagnostics["singular_values"],
                vh=diagnostics["vh"],
                largest=diagnostics["largest"],
                tau=diagnostics["tau"],
                rank=diagnostics["rank"],
                smallest_positive=diagnostics["smallest_positive"],
                smallest_positive_defined=diagnostics["smallest_positive_defined"],
                condition=diagnostics["condition"],
                condition_defined=diagnostics["condition_defined"],
                weakly_responsive=diagnostics["weakly_responsive"],
                **saturation,
            )
        self.checkpoint_count += 1
        self.evidence.completed_checkpoints += 1
        self.evidence.archive.completed_checkpoints += 1
        self.evidence.archive.completed_checkpoint_component_rows += COMPONENTS
        if snapshot.step == UPDATES:
            self.final_checkpoint = snapshot

    def finish(self) -> None:
        if self.event_position != len(self.expected_events) or self.pending_pre is not None:
            raise ProtocolInvalid(
                self.evidence.archive.phase, "callback event stream is incomplete"
            )
        if self.checkpoint_count != len(CHECKPOINTS):
            raise ProtocolInvalid(self.evidence.archive.phase, "callback checkpoint count differs")
        if len(self.lr_trace) != UPDATES or len(self.next_lr_trace) != UPDATES + 1:
            raise ProtocolInvalid(self.evidence.archive.phase, "callback LR trace count differs")
        if self.final_checkpoint is None:
            raise ProtocolInvalid(self.evidence.archive.phase, "terminal checkpoint is absent")


def _append_fit_result(
    evidence: ScientificEvidence,
    recorder: NativeEvidenceRecorder,
    result: Gaussians2D,
    history: Mapping[str, Any],
    *,
    fit_index: int,
    block_index: int,
    seed_position: int,
    cell: PreparedFitCell,
    arm_code: int,
    arm_order_position: int,
    rng_before: torch.Tensor,
    rng_after: torch.Tensor,
) -> None:
    recorder.finish()
    terminal = recorder.final_checkpoint
    if terminal is None:
        raise ProtocolInvalid(evidence.archive.phase, "fit result lacks terminal checkpoint")
    for field_name in ("xy", "chol", "color", "weight"):
        _assert_tensor_equal(
            f"result/checkpoint120/{field_name}",
            getattr(result, field_name),
            getattr(terminal.gaussians, field_name),
            evidence.archive.phase,
        )
    if history.get("stopped_iter") != UPDATES - 1:
        raise ProtocolInvalid(evidence.archive.phase, "fit stopped iteration differs")
    history_rows = history.get("psnr")
    if not isinstance(history_rows, list) or len(history_rows) != 4:
        raise ProtocolInvalid(evidence.archive.phase, "fit history checkpoint count differs")
    history_steps = np.asarray([row[0] for row in history_rows], dtype=np.int64)
    if not np.array_equal(history_steps, np.asarray(HISTORY_ITERATIONS, dtype=np.int64)):
        raise ProtocolInvalid(evidence.archive.phase, "fit history iteration schedule differs")
    history_values = np.asarray([row[1] for row in history_rows], dtype=np.float32)
    if not bool(np.isfinite(history_values).all()):
        raise ProtocolInvalid(evidence.archive.phase, "fit history value is non-finite")
    if not torch.equal(rng_before, rng_after):
        raise ProtocolInvalid(evidence.archive.phase, "paired native fit changed global CPU RNG")
    defaults_bytes, defaults_length = _utf8_buffer(recorder.initial_defaults_json, 4_096)
    groups_bytes, groups_length = _utf8_buffer(recorder.initial_groups_json, 4_096)
    scheduler_bytes, scheduler_length = _utf8_buffer(recorder.initial_scheduler_json, 4_096)
    evidence.tables["fit"].append(
        fit_index=fit_index,
        block_index=block_index,
        seed_position=seed_position,
        seed=cell.seed,
        local_view=cell.local_view,
        original_view=cell.original_view,
        arm_code=arm_code,
        arm_order_position=arm_order_position,
        optimizer_param_count=len(recorder.expected_names),
        optimizer_defaults_bytes=defaults_bytes,
        optimizer_defaults_length=defaults_length,
        optimizer_groups_bytes=groups_bytes,
        optimizer_groups_length=groups_length,
        scheduler_initial_bytes=scheduler_bytes,
        scheduler_initial_length=scheduler_length,
        lr_trace=np.asarray(recorder.lr_trace, dtype=np.float64),
        next_lr_trace=np.asarray(recorder.next_lr_trace, dtype=np.float64),
        stopped_iter=history["stopped_iter"],
        final_psnr_full=np.float32(history["final_psnr_full"]),
        final_psnr=np.float32(history["final_psnr"]),
        result_xy=result.xy,
        result_chol=result.chol,
        result_color=result.color,
        result_weight=result.weight,
        global_rng_before=rng_before,
        global_rng_after=rng_after,
    )


def execute_scientific_fits(evidence: ScientificEvidence) -> None:
    if not evidence.optimization_unlocked:
        raise ProtocolInvalid("equivalence", "optimization opened before global prerequisite")
    fit_index = 0
    for block_index, block in enumerate(BLOCKS):
        evidence.archive.phase = block
        geometry_frozen = block == "appearance_only"
        for seed_position, _seed in enumerate(BLOCK_SEEDS[block]):
            order = arm_order(seed_position)
            for local_view in range(len(SELECTED_VIEWS)):
                cell = evidence.prepared[(block_index, seed_position, local_view)]
                for arm_order_position, arm in enumerate(order):
                    config = frozen_fit_config(arm)
                    config.freeze_geometry = geometry_frozen
                    recorder = NativeEvidenceRecorder(
                        evidence,
                        cell,
                        fit_index=fit_index,
                        arm=arm,
                        geometry_frozen=geometry_frozen,
                    )
                    rng_before = torch.random.get_rng_state().clone()
                    result, history = fit_image_from_initialization(
                        cell.target,
                        _clone_gaussians(cell.g0),
                        config,
                        mask=None,
                        diagnostic_callback=recorder,
                        diagnostic_steps=POSITIVE_CHECKPOINTS,
                    )
                    rng_after = torch.random.get_rng_state().clone()
                    arm_code = ARMS.index(arm)
                    _append_fit_result(
                        evidence,
                        recorder,
                        result,
                        history,
                        fit_index=fit_index,
                        block_index=block_index,
                        seed_position=seed_position,
                        cell=cell,
                        arm_code=arm_code,
                        arm_order_position=arm_order_position,
                        rng_before=rng_before,
                        rng_after=rng_after,
                    )
                    evidence.completed_fits += 1
                    evidence.archive.completed_fits += 1
                    fit_index += 1
                evidence.archive.completed_views += 1
            evidence.archive.completed_seeds += 1
        evidence.archive.completed_blocks += 1
    if (
        evidence.completed_fits != SCIENTIFIC_PLAN.fits
        or evidence.completed_updates != SCIENTIFIC_PLAN.optimizer_updates
        or evidence.completed_checkpoints != SCIENTIFIC_PLAN.checkpoints
        or evidence.completed_callback_events != SCIENTIFIC_PLAN.callback_events
    ):
        raise ProtocolInvalid(evidence.archive.phase, "scientific fit completeness differs")


def _scientific_required_names() -> set[str]:
    record = scientific_schema_record()
    names = (
        set(record["constant_arrays"])
        | set(record["completion_arrays"])
        | set(record["archive_completion_arrays"])
    )
    for table_name, spec in SCIENTIFIC_TABLE_SPECS.items():
        prefix = f"scientific/{table_name}"
        names.add(f"{prefix}/row_count")
        names.update(f"{prefix}/{field_name}" for field_name in spec.fields)
    return names


def _raw_table(
    arrays: Mapping[str, np.ndarray], name: str, *, require_complete: bool
) -> dict[str, np.ndarray]:
    spec = SCIENTIFIC_TABLE_SPECS[name]
    prefix = f"scientific/{name}"
    count_name = f"{prefix}/row_count"
    if count_name not in arrays:
        raise ProtocolInvalid("reduction", f"raw table is absent: {name}")
    count_array = arrays[count_name]
    if count_array.dtype != np.dtype("<i8") or count_array.shape != ():
        raise ProtocolInvalid("reduction", f"raw row count schema differs: {name}")
    rows = int(count_array)
    if rows < 0 or rows > spec.rows or (require_complete and rows != spec.rows):
        raise ProtocolInvalid(
            "reduction",
            f"raw row count differs: {name}",
            {"actual": rows, "expected": spec.rows},
        )
    result: dict[str, np.ndarray] = {}
    for field_name, field_spec in spec.fields.items():
        logical_name = f"{prefix}/{field_name}"
        if logical_name not in arrays:
            raise ProtocolInvalid("reduction", f"raw table field is absent: {logical_name}")
        value = arrays[logical_name]
        expected_shape = (rows, *field_spec.shape)
        if value.dtype != np.dtype(field_spec.dtype) or value.shape != expected_shape:
            raise ProtocolInvalid(
                "reduction",
                f"raw table field schema differs: {logical_name}",
                {
                    "actual_dtype": value.dtype.str,
                    "expected_dtype": field_spec.dtype,
                    "actual_shape": list(value.shape),
                    "expected_shape": list(expected_shape),
                },
            )
        if np.issubdtype(value.dtype, np.floating) and not bool(np.isfinite(value).all()):
            raise ProtocolInvalid(
                "reduction", f"valid raw table field is non-finite: {logical_name}"
            )
        result[field_name] = value
    return result


def _decode_utf8_rows(values: np.ndarray, lengths: np.ndarray) -> list[str]:
    if values.dtype != np.uint8 or lengths.dtype != np.dtype("<i8"):
        raise ProtocolInvalid("reduction", "UTF-8 raw evidence dtype differs")
    if values.ndim != 2 or lengths.shape != (values.shape[0],):
        raise ProtocolInvalid("reduction", "UTF-8 raw evidence shape differs")
    result: list[str] = []
    for row, length_value in zip(values, lengths, strict=True):
        length = int(length_value)
        if length < 0 or length > row.shape[0] or bool(row[length:].any()):
            raise ProtocolInvalid("reduction", "UTF-8 raw evidence length/padding differs")
        try:
            result.append(row[:length].tobytes().decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ProtocolInvalid("reduction", "UTF-8 raw evidence is malformed") from error
    return result


def _validate_scientific_constants(arrays: Mapping[str, np.ndarray]) -> None:
    expected_block_bytes, expected_block_lengths = _utf8_rows(BLOCKS, 32)
    expected_arm_bytes, expected_arm_lengths = _utf8_rows(ARMS, 32)
    for name, expected in {
        "identity/block_name_bytes": expected_block_bytes,
        "identity/block_name_lengths": expected_block_lengths,
        "identity/arm_name_bytes": expected_arm_bytes,
        "identity/arm_name_lengths": expected_arm_lengths,
    }.items():
        actual = arrays.get(name)
        if actual is None or not _arrays_byte_exact(actual, expected):
            raise ProtocolInvalid("reduction", f"raw identity array differs: {name}")
    block_names = _decode_utf8_rows(
        arrays["identity/block_name_bytes"], arrays["identity/block_name_lengths"]
    )
    arm_names = _decode_utf8_rows(
        arrays["identity/arm_name_bytes"], arrays["identity/arm_name_lengths"]
    )
    if block_names != list(BLOCKS) or arm_names != list(ARMS):
        raise ProtocolInvalid("reduction", "raw block/arm names differ")
    exact = {
        "identity/block_seeds": np.asarray(
            [BLOCK_SEEDS[block] for block in BLOCKS], dtype=np.int64
        ),
        "identity/selected_views": np.asarray(SELECTED_VIEWS, dtype=np.int64),
        "identity/checkpoints": np.asarray(CHECKPOINTS, dtype=np.int64),
        "identity/component_indices": np.arange(COMPONENTS, dtype=np.int64),
        "identity/histogram_edges": np.asarray(HISTOGRAM_EDGES, dtype=np.float64),
        "identity/arm_order": np.asarray(
            [
                [tuple(ARMS.index(arm) for arm in arm_order(position)) for position in range(3)]
                for _block in BLOCKS
            ],
            dtype=np.int64,
        ),
    }
    for name, expected in exact.items():
        actual = arrays.get(name)
        if actual is None or not _arrays_byte_exact(actual, expected):
            raise ProtocolInvalid("reduction", f"raw identity array differs: {name}")


def _validate_scientific_completion(arrays: Mapping[str, np.ndarray]) -> None:
    phase = arrays["scientific_completion/phase_code"]
    if phase.dtype != np.dtype("<i8") or phase.shape != () or int(phase) != _phase_code("complete"):
        raise ProtocolInvalid("reduction", "scientific completion phase differs")
    names = _decode_utf8_rows(
        arrays["scientific_completion/count_name_bytes"],
        arrays["scientific_completion/count_name_lengths"],
    )
    expected = {
        "callback_events": SCIENTIFIC_PLAN.callback_events,
        "checkpoints": SCIENTIFIC_PLAN.checkpoints,
        "fits": SCIENTIFIC_PLAN.fits,
        "initializers": SCIENTIFIC_PLAN.initializers,
        "joint_candidate_positive_checkpoints": SCIENTIFIC_PLAN.joint_positive_checkpoints_per_arm,
        "joint_current_positive_checkpoints": SCIENTIFIC_PLAN.joint_positive_checkpoints_per_arm,
        "mechanism_candidate_updates": SCIENTIFIC_PLAN.mechanism_updates_per_arm,
        "mechanism_current_updates": SCIENTIFIC_PLAN.mechanism_updates_per_arm,
        "mechanism_geometry_states": SCIENTIFIC_PLAN.mechanism_state_rows,
        "optimizer_updates": SCIENTIFIC_PLAN.optimizer_updates,
        "scenes": SCIENTIFIC_PLAN.scenes,
    }
    values = arrays["scientific_completion/count_values"]
    expected_names = sorted(expected)
    expected_values = np.asarray([expected[name] for name in expected_names], dtype=np.int64)
    expected_name_bytes, expected_name_lengths = _utf8_rows(expected_names, 64)
    if (
        names != expected_names
        or not _arrays_byte_exact(
            arrays["scientific_completion/count_name_bytes"], expected_name_bytes
        )
        or not _arrays_byte_exact(
            arrays["scientific_completion/count_name_lengths"], expected_name_lengths
        )
        or not _arrays_byte_exact(values, expected_values)
    ):
        raise ProtocolInvalid("reduction", "scientific completion counts differ")


def _validate_archive_completion(arrays: Mapping[str, np.ndarray]) -> None:
    expected = {
        "phase_code": _phase_code("complete"),
        "completed_blocks": len(BLOCKS),
        "completed_seeds": SCIENTIFIC_PLAN.scenes,
        "completed_views": SCIENTIFIC_PLAN.initializers,
        "completed_initializers": SCIENTIFIC_PLAN.initializers,
        "completed_paired_views": SCIENTIFIC_PLAN.initializers,
        "completed_fits": SCIENTIFIC_PLAN.fits,
        "completed_updates": SCIENTIFIC_PLAN.optimizer_updates,
        "completed_checkpoints": SCIENTIFIC_PLAN.checkpoints,
        "completed_callback_events": SCIENTIFIC_PLAN.callback_events,
        "completed_mechanism_updates": SCIENTIFIC_PLAN.mechanism_updates,
        "completed_joint_positive_checkpoint_updates": SCIENTIFIC_PLAN.joint_positive_checkpoints,
        "completed_checkpoint_component_rows": SCIENTIFIC_PLAN.checkpoint_component_rows,
        "completed_null_rows": SCIENTIFIC_PLAN.current_null_rows,
    }
    for suffix, expected_value in expected.items():
        name = f"completion/{suffix}"
        actual = arrays[name]
        expected_array = np.asarray(expected_value, dtype=np.int64)
        if not _arrays_byte_exact(actual, expected_array):
            raise ProtocolInvalid("reduction", f"archive completion count differs: {name}")


def _expected_fit_rows() -> list[tuple[int, int, int, int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int, int, int, int]] = []
    fit_index = 0
    for block_index, block in enumerate(BLOCKS):
        for seed_position, seed in enumerate(BLOCK_SEEDS[block]):
            for local_view, original_view in enumerate(SELECTED_VIEWS):
                for arm_position, arm in enumerate(arm_order(seed_position)):
                    rows.append(
                        (
                            fit_index,
                            block_index,
                            seed_position,
                            seed,
                            local_view,
                            original_view,
                            ARMS.index(arm),
                            arm_position,
                        )
                    )
                    fit_index += 1
    return rows


def _validate_scientific_order(tables: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    prerequisite = tables["prerequisite"]
    expected_prerequisite = []
    for block_index, block in enumerate(BLOCKS):
        for seed_position, seed in enumerate(BLOCK_SEEDS[block]):
            for local_view, original_view in enumerate(SELECTED_VIEWS):
                expected_prerequisite.append(
                    (block_index, seed_position, seed, local_view, original_view, seed + local_view)
                )
    actual_prerequisite = list(
        zip(
            prerequisite["block_index"].tolist(),
            prerequisite["seed_position"].tolist(),
            prerequisite["seed"].tolist(),
            prerequisite["local_view"].tolist(),
            prerequisite["original_view"].tolist(),
            prerequisite["initializer_seed"].tolist(),
            strict=True,
        )
    )
    if actual_prerequisite != expected_prerequisite:
        raise ProtocolInvalid("reduction", "prerequisite row order differs")
    fit = tables["fit"]
    expected_fits = _expected_fit_rows()
    actual_fits = list(
        zip(
            fit["fit_index"].tolist(),
            fit["block_index"].tolist(),
            fit["seed_position"].tolist(),
            fit["seed"].tolist(),
            fit["local_view"].tolist(),
            fit["original_view"].tolist(),
            fit["arm_code"].tolist(),
            fit["arm_order_position"].tolist(),
            strict=True,
        )
    )
    if actual_fits != expected_fits:
        raise ProtocolInvalid("reduction", "fit execution order differs")
    expected_events: list[tuple[int, int, int]] = []
    for fit_index in range(SCIENTIFIC_PLAN.fits):
        expected_events.append((fit_index, EVENT_CODES["initial"], 0))
        for completed_step in range(UPDATES):
            expected_events.append((fit_index, EVENT_CODES["pre_update"], completed_step))
            expected_events.append((fit_index, EVENT_CODES["post_update"], completed_step + 1))
            if completed_step + 1 in POSITIVE_CHECKPOINTS:
                expected_events.append((fit_index, EVENT_CODES["checkpoint"], completed_step + 1))
    event = tables["fit_event"]
    actual_events = list(
        zip(
            event["fit_index"].tolist(),
            event["event_code"].tolist(),
            event["step"].tolist(),
            strict=True,
        )
    )
    if actual_events != expected_events:
        raise ProtocolInvalid("reduction", "callback event transcript differs")
    expected_checkpoints = [
        (checkpoint_index, fit_index, step)
        for checkpoint_index, (fit_index, step) in enumerate(
            pair
            for fit_index in range(SCIENTIFIC_PLAN.fits)
            for pair in [(fit_index, s) for s in CHECKPOINTS]
        )
    ]
    checkpoint = tables["checkpoint"]
    actual_checkpoints = list(
        zip(
            checkpoint["checkpoint_index"].tolist(),
            checkpoint["fit_index"].tolist(),
            checkpoint["step"].tolist(),
            strict=True,
        )
    )
    if actual_checkpoints != expected_checkpoints:
        raise ProtocolInvalid("reduction", "checkpoint transcript differs")
    for arm_code, table_name in enumerate(("checkpoint_current", "checkpoint_candidate")):
        expected_arm_rows = [
            (checkpoint_index, fit_index, step)
            for checkpoint_index, fit_index, step in expected_checkpoints
            if int(fit["arm_code"][fit_index]) == arm_code
        ]
        table = tables[table_name]
        actual_arm_rows = list(
            zip(
                table["checkpoint_index"].tolist(),
                table["fit_index"].tolist(),
                table["step"].tolist(),
                strict=True,
            )
        )
        if actual_arm_rows != expected_arm_rows:
            raise ProtocolInvalid("reduction", f"checkpoint arm transcript differs: {table_name}")
    fit_block = fit["block_index"]
    fit_arm = fit["arm_code"]
    expected_geometry = [
        (fit_index, step)
        for fit_index in range(SCIENTIFIC_PLAN.fits)
        if fit_block[fit_index] == 0
        for step in range(UPDATES + 1)
    ]
    geometry = tables["mechanism_geometry"]
    if (
        list(zip(geometry["fit_index"].tolist(), geometry["step"].tolist(), strict=True))
        != expected_geometry
    ):
        raise ProtocolInvalid("reduction", "mechanism geometry transcript differs")
    for arm_code, table_name in enumerate(("mechanism_current", "mechanism_candidate")):
        expected_updates = [
            (fit_index, completed_step, completed_step + 1)
            for fit_index in range(SCIENTIFIC_PLAN.fits)
            if fit_block[fit_index] == 0 and fit_arm[fit_index] == arm_code
            for completed_step in range(UPDATES)
        ]
        table = tables[table_name]
        actual = list(
            zip(
                table["fit_index"].tolist(),
                table["completed_step"].tolist(),
                table["update_number"].tolist(),
                strict=True,
            )
        )
        if actual != expected_updates:
            raise ProtocolInvalid("reduction", f"mechanism update transcript differs: {table_name}")
    for arm_code, table_name in enumerate(("joint_current", "joint_candidate")):
        expected_rows = [
            (fit_index, step, step > 0)
            for fit_index in range(SCIENTIFIC_PLAN.fits)
            if fit_block[fit_index] == 1 and fit_arm[fit_index] == arm_code
            for step in CHECKPOINTS
        ]
        table = tables[table_name]
        actual = list(
            zip(
                table["fit_index"].tolist(),
                table["step"].tolist(),
                table["defined"].tolist(),
                strict=True,
            )
        )
        if actual != expected_rows:
            raise ProtocolInvalid("reduction", f"joint transition transcript differs: {table_name}")


def _decode_utf8_scalar(value: np.ndarray, length: np.ndarray) -> str:
    return _decode_utf8_rows(
        value.reshape(1, value.shape[0]),
        np.asarray([int(length)], dtype=np.int64),
    )[0]


def _expected_optimizer_evidence(
    names: tuple[str, ...],
) -> tuple[str, str, str, np.ndarray, np.ndarray]:
    parameters = [torch.nn.Parameter(torch.zeros(1, dtype=torch.float32)) for _ in names]
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES, eta_min=0.001)
    defaults_json = _diagnostic_json(optimizer.defaults, phase="reduction")
    name_by_id = {id(parameter): name for parameter, name in zip(parameters, names, strict=True)}
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        groups.append(
            {
                key: (
                    tuple(name_by_id[id(parameter)] for parameter in value)
                    if key == "params"
                    else value
                )
                for key, value in group.items()
            }
        )
    groups_json = _diagnostic_json(tuple(groups), phase="reduction")
    scheduler_json = _diagnostic_json(scheduler.state_dict(), phase="reduction")
    lr_used: list[float] = []
    next_lr: list[float] = [float(optimizer.param_groups[0]["lr"])]
    for _ in range(UPDATES):
        lr_used.append(float(optimizer.param_groups[0]["lr"]))
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        next_lr.append(float(optimizer.param_groups[0]["lr"]))
    return (
        defaults_json,
        groups_json,
        scheduler_json,
        np.asarray(lr_used, dtype=np.float64),
        np.asarray(next_lr, dtype=np.float64),
    )


def _validate_fit_and_event_evidence(tables: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    fit = tables["fit"]
    event = tables["fit_event"]
    checkpoint = tables["checkpoint"]
    checkpoint_by_key = {
        (int(fit_index), int(step)): row
        for row, (fit_index, step) in enumerate(
            zip(checkpoint["fit_index"], checkpoint["step"], strict=True)
        )
    }
    contracts: dict[tuple[str, ...], tuple[str, str, str, np.ndarray, np.ndarray]] = {}
    event_row = 0
    global_rng_reference = fit["global_rng_before"][0]
    for fit_index in range(SCIENTIFIC_PLAN.fits):
        current = int(fit["arm_code"][fit_index]) == 0
        frozen = int(fit["block_index"][fit_index]) == 0
        names = (
            (("color_raw", "weight_raw") if current else ("amplitude_raw",))
            if frozen
            else (
                ("xy_raw", "diag_raw", "off_raw", "color_raw", "weight_raw")
                if current
                else ("xy_raw", "diag_raw", "off_raw", "amplitude_raw")
            )
        )
        if names not in contracts:
            contracts[names] = _expected_optimizer_evidence(names)
        contract = contracts[names]
        defaults_json, groups_json, scheduler_json, expected_lr, expected_next_lr = contract
        if int(fit["optimizer_param_count"][fit_index]) != len(names):
            raise ProtocolInvalid("reduction", "fit optimizer parameter count differs")
        serialized = (
            _decode_utf8_scalar(
                fit["optimizer_defaults_bytes"][fit_index],
                fit["optimizer_defaults_length"][fit_index],
            ),
            _decode_utf8_scalar(
                fit["optimizer_groups_bytes"][fit_index],
                fit["optimizer_groups_length"][fit_index],
            ),
            _decode_utf8_scalar(
                fit["scheduler_initial_bytes"][fit_index],
                fit["scheduler_initial_length"][fit_index],
            ),
        )
        if serialized != (defaults_json, groups_json, scheduler_json):
            raise ProtocolInvalid("reduction", "fit optimizer/scheduler metadata differs")
        _assert_np_exact("fit/lr_trace", fit["lr_trace"][fit_index], expected_lr)
        _assert_np_exact("fit/next_lr_trace", fit["next_lr_trace"][fit_index], expected_next_lr)

        def validate_event(
            code: int,
            step: int,
            lr: float,
            lr_defined: bool,
            next_value: float,
            scheduler_epoch: int,
            scheduler_count: int,
            fit_identity: int = fit_index,
        ) -> None:
            nonlocal event_row
            expected_scalars = {
                "fit_index": fit_identity,
                "event_code": code,
                "step": step,
                "lr_used": lr,
                "lr_used_defined": lr_defined,
                "next_lr": next_value,
                "scheduler_last_epoch": scheduler_epoch,
                "scheduler_step_count": scheduler_count,
            }
            for field_name, expected_value in expected_scalars.items():
                actual_value = event[field_name][event_row]
                expected_array = np.asarray(expected_value, dtype=actual_value.dtype)
                if not _arrays_byte_exact(np.asarray(actual_value), expected_array):
                    raise ProtocolInvalid("reduction", f"fit event field differs: {field_name}")
            event_row += 1

        validate_event(EVENT_CODES["initial"], 0, 0.0, False, float(expected_next_lr[0]), 0, 1)
        for completed_step in range(UPDATES):
            lr = float(expected_lr[completed_step])
            validate_event(
                EVENT_CODES["pre_update"],
                completed_step,
                lr,
                True,
                lr,
                completed_step,
                completed_step + 1,
            )
            update_number = completed_step + 1
            validate_event(
                EVENT_CODES["post_update"],
                update_number,
                lr,
                True,
                float(expected_next_lr[update_number]),
                update_number,
                update_number + 1,
            )
            if update_number in POSITIVE_CHECKPOINTS:
                validate_event(
                    EVENT_CODES["checkpoint"],
                    update_number,
                    lr,
                    True,
                    float(expected_next_lr[update_number]),
                    update_number,
                    update_number + 1,
                )
        if int(fit["stopped_iter"][fit_index]) != UPDATES - 1:
            raise ProtocolInvalid("reduction", "fit stopped iteration differs")
        if not _arrays_byte_exact(
            fit["global_rng_before"][fit_index], fit["global_rng_after"][fit_index]
        ):
            raise ProtocolInvalid("reduction", "fit changed global CPU RNG")
        if not _arrays_byte_exact(fit["global_rng_before"][fit_index], global_rng_reference):
            raise ProtocolInvalid("reduction", "fit global CPU RNG transcript differs")
        final_checkpoint = checkpoint_by_key[(fit_index, UPDATES)]
        for result_name, checkpoint_name in (
            ("result_xy", "built_xy"),
            ("result_chol", "built_chol"),
            ("result_color", "built_color"),
            ("result_weight", "built_weight"),
        ):
            _assert_np_exact(
                f"fit/{result_name}/checkpoint120",
                fit[result_name][fit_index],
                checkpoint[checkpoint_name][final_checkpoint],
            )
        _assert_np_exact(
            "fit/final_psnr_full",
            fit["final_psnr_full"][fit_index],
            checkpoint["psnr"][final_checkpoint],
        )
        _assert_np_exact(
            "fit/final_psnr",
            fit["final_psnr"][fit_index],
            checkpoint["psnr"][final_checkpoint],
        )
        if not current and not np.array_equal(
            fit["result_weight"][fit_index], np.ones(COMPONENTS, dtype=np.float32)
        ):
            raise ProtocolInvalid("reduction", "candidate fit result weight differs")
    if event_row != SCIENTIFIC_PLAN.callback_events:
        raise ProtocolInvalid("reduction", "fit event validation count differs")


def _torch_row(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value))


def _assert_np_exact(name: str, actual: np.ndarray, expected: torch.Tensor | np.ndarray) -> None:
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().contiguous().cpu().numpy()
    expected_array = little_endian_array(np.asarray(expected))
    actual_array = little_endian_array(actual)
    if not _arrays_byte_exact(actual_array, expected_array):
        raise ProtocolInvalid("reduction", f"raw recomputation differs bit-exactly: {name}")


def _assert_np_close(
    name: str,
    actual: np.ndarray,
    expected: torch.Tensor | np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().contiguous().cpu().numpy()
    expected_array = np.asarray(expected)
    if actual.shape != expected_array.shape or not np.allclose(
        actual, expected_array, atol=atol, rtol=rtol
    ):
        raise ProtocolInvalid("reduction", f"raw recomputation differs: {name}")


def _assert_frozen_error_pair(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    absolute_limit: float,
    relative_limit: float,
    relative_threshold: float = 1e-8,
    recorded: np.ndarray | None = None,
) -> None:
    pair = np.asarray(error_pair(actual, expected, relative_threshold), dtype=np.float64)
    if recorded is not None:
        _assert_np_exact(f"{name}/equation_errors", recorded, pair)
    if float(pair[0]) > absolute_limit:
        raise ProtocolInvalid("reduction", f"{name} maximum-absolute gate failed")
    if float(pair[1]) > relative_limit:
        raise ProtocolInvalid("reduction", f"{name} maximum-relative gate failed")


def _recompute_prerequisite(table: Mapping[str, np.ndarray], *, deep: bool) -> None:
    for index in range(SCIENTIFIC_PLAN.initializers):
        target = _torch_row(table["target"][index])
        g0 = Gaussians2D(
            xy=_torch_row(table["g0_xy"][index]),
            chol=_torch_row(table["g0_chol"][index]),
            color=_torch_row(table["g0_color"][index]),
            weight=_torch_row(table["g0_weight"][index]),
        )
        _validate_target_and_initializer(target, g0)
        expected_target_hash = _sha256_bytes(_tensor_sha256(target))
        _assert_np_exact(
            "prerequisite/target_sha256", table["target_sha256"][index], expected_target_hash
        )
        for field_index, field_name in enumerate(("xy", "chol", "color", "weight")):
            expected_hash = _sha256_bytes(_tensor_sha256(getattr(g0, field_name)))
            _assert_np_exact(
                f"prerequisite/g0_field_sha256/{field_name}",
                table["g0_field_sha256"][index, field_index],
                expected_hash,
            )
        for state_index, state_name in enumerate(("rng_state_before", "rng_state_after")):
            state = _torch_row(table[state_name][index])
            expected_hash = _sha256_bytes(_tensor_sha256(state))
            _assert_np_exact(
                f"prerequisite/rng_state_sha256/{state_name}",
                table["rng_state_sha256"][index, state_index],
                expected_hash,
            )
        if not deep:
            continue
        generator = torch.Generator(device="cpu").manual_seed(int(table["initializer_seed"][index]))
        _assert_np_exact(
            "prerequisite/rng_state_before/rederived",
            table["rng_state_before"][index],
            generator.get_state(),
        )
        rederived_g0 = init_gaussians_2d(
            target,
            n=COMPONENTS,
            grad_mix=0.7,
            generator=generator,
        )
        _assert_np_exact(
            "prerequisite/rng_state_after/rederived",
            table["rng_state_after"][index],
            generator.get_state(),
        )
        for field_name in ("xy", "chol", "color", "weight"):
            _assert_np_exact(
                f"prerequisite/g0/{field_name}/rederived",
                table[f"g0_{field_name}"][index],
                getattr(rederived_g0, field_name),
            )
        (
            current_raw,
            candidate_raw,
            current,
            candidate,
            current_render,
            candidate_render,
            current_loss,
            candidate_loss,
            gate_values,
        ) = _prepare_step_zero(target, g0)
        recomputed: dict[str, torch.Tensor | np.ndarray] = {
            "current_xy_raw": current_raw["xy_raw"],
            "current_diag_raw": current_raw["diag_raw"],
            "current_off_raw": current_raw["off_raw"],
            "current_color_raw": current_raw["u"],
            "current_weight_raw": current_raw["s"],
            "candidate_xy_raw": candidate_raw["xy_raw"],
            "candidate_diag_raw": candidate_raw["diag_raw"],
            "candidate_off_raw": candidate_raw["off_raw"],
            "candidate_amplitude_raw": candidate_raw["r"],
            "current_built_xy": current.xy,
            "current_built_chol": current.chol,
            "current_built_color": current.color,
            "current_built_weight": current.weight,
            "current_built_amplitude": current.weight[:, None] * current.color,
            "candidate_built_xy": candidate.xy,
            "candidate_built_chol": candidate.chol,
            "candidate_built_color": candidate.color,
            "candidate_built_weight": candidate.weight,
            "candidate_built_amplitude": candidate.weight[:, None] * candidate.color,
            "current_render": current_render,
            "candidate_render": candidate_render,
            "current_loss": current_loss,
            "candidate_loss": candidate_loss,
            "gate_values": gate_values,
        }
        for name, expected in recomputed.items():
            _assert_np_exact(f"prerequisite/{name}", table[name][index], expected)


def _dummy_checkpoint_snapshot(
    arm: str,
    step: int,
    raw_parameters: Mapping[str, torch.Tensor],
    gaussians: Gaussians2D,
    target: torch.Tensor,
    render: torch.Tensor,
    loss: torch.Tensor,
) -> NativeFitDiagnostic:
    return NativeFitDiagnostic(
        event="checkpoint",
        step=step,
        appearance_parameterization=arm,
        geometry_frozen=False,
        initial_gaussians=None,
        initializer_rng_state_before=None,
        initializer_rng_state_after=None,
        raw_parameters=dict(raw_parameters),
        gradients={name: None for name in raw_parameters},
        gaussians=gaussians,
        target=target,
        rendered=render,
        loss=loss,
        optimizer_param_names=(),
        optimizer_state={},
        optimizer_param_groups=(),
        optimizer_defaults={},
        scheduler_state={},
        lr_used=None,
        next_lr=0.0,
    )


def _recompute_checkpoints(tables: Mapping[str, Mapping[str, np.ndarray]], *, deep: bool) -> None:
    checkpoint = tables["checkpoint"]
    fit = tables["fit"]
    prerequisite = tables["prerequisite"]
    prerequisite_index = {
        (
            int(prerequisite["block_index"][row]),
            int(prerequisite["seed_position"][row]),
            int(prerequisite["local_view"][row]),
        ): row
        for row in range(SCIENTIFIC_PLAN.initializers)
    }
    prerequisite = tables["prerequisite"]
    current = tables["checkpoint_current"]
    candidate = tables["checkpoint_candidate"]
    current_by_checkpoint = {
        int(checkpoint_index): row
        for row, checkpoint_index in enumerate(current["checkpoint_index"])
    }
    candidate_by_checkpoint = {
        int(checkpoint_index): row
        for row, checkpoint_index in enumerate(candidate["checkpoint_index"])
    }
    prerequisite_by_key = {
        (
            int(prerequisite["block_index"][row]),
            int(prerequisite["seed_position"][row]),
            int(prerequisite["local_view"][row]),
        ): row
        for row in range(SCIENTIFIC_PLAN.initializers)
    }
    for row in range(SCIENTIFIC_PLAN.checkpoints):
        fit_index = int(checkpoint["fit_index"][row])
        arm_code = int(fit["arm_code"][fit_index])
        arm = ARMS[arm_code]
        raw: dict[str, torch.Tensor] = {
            "xy_raw": _torch_row(checkpoint["xy_raw"][row]),
            "diag_raw": _torch_row(checkpoint["diag_raw"][row]),
            "off_raw": _torch_row(checkpoint["off_raw"][row]),
        }
        arm_table = current if arm_code == 0 else candidate
        arm_row = current_by_checkpoint[row] if arm_code == 0 else candidate_by_checkpoint[row]
        if arm_code == 0:
            raw.update(
                {
                    "u": _torch_row(arm_table["color_raw"][arm_row]),
                    "s": _torch_row(arm_table["weight_raw"][arm_row]),
                }
            )
        else:
            raw["r"] = _torch_row(arm_table["amplitude_raw"][arm_row])
        prerequisite_row = prerequisite_by_key[
            (
                int(fit["block_index"][fit_index]),
                int(fit["seed_position"][fit_index]),
                int(fit["local_view"][fit_index]),
            )
        ]
        _assert_np_exact(
            "checkpoint/target/prerequisite",
            checkpoint["target"][row],
            prerequisite["target"][prerequisite_row],
        )
        if int(checkpoint["step"][row]) == 0:
            prefix = "current" if arm_code == 0 else "candidate"
            for checkpoint_name, prerequisite_name in (
                ("xy_raw", f"{prefix}_xy_raw"),
                ("diag_raw", f"{prefix}_diag_raw"),
                ("off_raw", f"{prefix}_off_raw"),
            ):
                _assert_np_exact(
                    f"checkpoint/step_zero/{checkpoint_name}",
                    checkpoint[checkpoint_name][row],
                    prerequisite[prerequisite_name][prerequisite_row],
                )
            appearance_pairs = (
                (("color_raw", "current_color_raw"), ("weight_raw", "current_weight_raw"))
                if arm_code == 0
                else (("amplitude_raw", "candidate_amplitude_raw"),)
            )
            for checkpoint_name, prerequisite_name in appearance_pairs:
                _assert_np_exact(
                    f"checkpoint/step_zero/{checkpoint_name}",
                    arm_table[checkpoint_name][arm_row],
                    prerequisite[prerequisite_name][prerequisite_row],
                )
        built = build_from_raw(raw, arm, IMAGE_SIZE, IMAGE_SIZE)
        for name, expected in {
            "built_xy": built.xy,
            "built_chol": built.chol,
            "built_color": built.color,
            "built_weight": built.weight,
            "built_amplitude": built.weight[:, None] * built.color,
        }.items():
            _assert_np_exact(f"checkpoint/{name}", checkpoint[name][row], expected)
        if arm_code == 1 and not np.array_equal(
            checkpoint["built_weight"][row], np.ones(COMPONENTS, dtype=np.float32)
        ):
            raise ProtocolInvalid("reduction", "candidate checkpoint weight is not exact one")
        render = _torch_row(checkpoint["render"][row])
        target = _torch_row(checkpoint["target"][row])
        fit_key = (
            int(fit["block_index"][fit_index]),
            int(fit["seed_position"][fit_index]),
            int(fit["local_view"][fit_index]),
        )
        _assert_np_exact(
            "checkpoint/target_prerequisite",
            checkpoint["target"][row],
            prerequisite["target"][prerequisite_index[fit_key]],
        )
        if deep:
            expected_render = render_gaussians_2d(built, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64)
            _assert_np_exact("checkpoint/render", checkpoint["render"][row], expected_render)
        metrics = render_metrics(render, target, phase="reduction")
        objective = torch.nn.functional.mse_loss(render, target)
        _assert_np_exact("checkpoint/objective_loss", checkpoint["objective_loss"][row], objective)
        exact_metrics: dict[str, Any] = {
            "clamped_render": metrics["clamped"],
            "below_mask": metrics["below_mask"],
            "above_mask": metrics["above_mask"],
            "sse": np.asarray(metrics["sse"], dtype=np.float64),
            "channel_count": np.asarray(metrics["channel_count"], dtype=np.int64),
            "raw_mse": np.asarray(metrics["raw_mse_float64"], dtype=np.float64),
            "psnr": np.asarray(metrics["psnr"], dtype=np.float32),
            "ssim": np.asarray(metrics["ssim"], dtype=np.float32),
            "below_count": np.asarray(metrics["below_count"], dtype=np.int64),
            "above_count": np.asarray(metrics["above_count"], dtype=np.int64),
            "below_fraction": np.asarray(metrics["below_fraction"], dtype=np.float64),
            "above_fraction": np.asarray(metrics["above_fraction"], dtype=np.float64),
        }
        for name, expected in exact_metrics.items():
            _assert_np_exact(f"checkpoint/{name}", checkpoint[name][row], expected)
        raw_parameters = {
            "xy_raw": raw["xy_raw"],
            "diag_raw": raw["diag_raw"],
            "off_raw": raw["off_raw"],
        }
        if arm_code == 0:
            raw_parameters.update({"color_raw": raw["u"], "weight_raw": raw["s"]})
        else:
            raw_parameters["amplitude_raw"] = raw["r"]
        snapshot = _dummy_checkpoint_snapshot(
            arm,
            int(checkpoint["step"][row]),
            raw_parameters,
            built,
            target,
            render,
            objective,
        )
        saturation = _checkpoint_saturation(arm, snapshot)
        for name, expected in saturation.items():
            _assert_np_exact(f"checkpoint/{arm}/{name}", arm_table[name][arm_row], expected)
        if arm_code == 0:
            jacobian = current_jacobian(built.weight, built.color)
            diagnostics = jacobian_diagnostics(jacobian, current=True)
            analytic = analytic_current_null(
                built.weight, built.color, diagnostics["null_vector"], diagnostics["rank"]
            )
            expected_diag = {
                "jacobian": jacobian,
                "singular_values": diagnostics["singular_values"],
                "vh": diagnostics["vh"],
                "largest": diagnostics["largest"],
                "tau": diagnostics["tau"],
                "rank": diagnostics["rank"],
                "smallest_positive": diagnostics["smallest_positive"],
                "smallest_positive_defined": diagnostics["smallest_positive_defined"],
                "condition": diagnostics["condition"],
                "condition_defined": diagnostics["condition_defined"],
                "weakly_responsive": diagnostics["weakly_responsive"],
                "null_vector": diagnostics["null_vector"],
                "null_residual": diagnostics["null_residual"],
                "analytic_null_vector": analytic["vector"],
                "analytic_null_alignment": analytic["alignment"],
                "analytic_null_defined": analytic["defined"],
            }
        else:
            jacobian = candidate_jacobian(built.color)
            diagnostics = jacobian_diagnostics(jacobian, current=False)
            expected_diag = {
                "jacobian": jacobian,
                "singular_values": diagnostics["singular_values"],
                "vh": diagnostics["vh"],
                "largest": diagnostics["largest"],
                "tau": diagnostics["tau"],
                "rank": diagnostics["rank"],
                "smallest_positive": diagnostics["smallest_positive"],
                "smallest_positive_defined": diagnostics["smallest_positive_defined"],
                "condition": diagnostics["condition"],
                "condition_defined": diagnostics["condition_defined"],
                "weakly_responsive": diagnostics["weakly_responsive"],
            }
        for name, expected in expected_diag.items():
            _assert_np_exact(f"checkpoint/{arm}/{name}", arm_table[name][arm_row], expected)


def _recompute_probe(
    target: torch.Tensor,
    xy: torch.Tensor,
    chol: torch.Tensor,
    color: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray]:
    optimization = Gaussians2D(xy=xy, chol=chol, color=color, weight=weight)
    optimization_render = render_gaussians_2d(optimization, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64)
    optimization_loss = torch.nn.functional.mse_loss(optimization_render, target)
    amplitude = (weight[:, None] * color).detach().clone().requires_grad_(True)
    probe = Gaussians2D(
        xy=xy.detach().clone(),
        chol=chol.detach().clone(),
        color=amplitude,
        weight=torch.ones_like(weight),
    )
    with torch.enable_grad():
        probe_render = render_gaussians_2d(probe, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64)
        probe_loss = torch.nn.functional.mse_loss(probe_render, target)
        (gradient,) = torch.autograd.grad(probe_loss, amplitude)
    difference = (probe_render.detach() - optimization_render).abs()
    numerator = float(difference.to(torch.float64).sum())
    denominator = float(optimization_render.abs().to(torch.float64).sum())
    values = np.asarray(
        [
            float(difference.max()),
            numerator,
            denominator,
            numerator / denominator,
            float(probe_loss),
            abs(float(probe_loss) - float(optimization_loss)),
            float(optimization_loss),
            abs(float(probe_loss) - float(optimization_loss)) / float(optimization_loss),
            optimization_render.numel(),
        ],
        dtype=np.float64,
    )
    return gradient.detach(), values


def _recompute_mechanism(tables: Mapping[str, Mapping[str, np.ndarray]], *, deep: bool) -> None:
    fit = tables["fit"]
    prerequisite = tables["prerequisite"]
    geometry = tables["mechanism_geometry"]
    geometry_index = {
        (int(fit_index), int(step)): row
        for row, (fit_index, step) in enumerate(
            zip(geometry["fit_index"], geometry["step"], strict=True)
        )
    }
    prerequisite_index = {
        (
            int(prerequisite["block_index"][row]),
            int(prerequisite["seed_position"][row]),
            int(prerequisite["local_view"][row]),
        ): row
        for row in range(SCIENTIFIC_PLAN.initializers)
    }
    for row in range(geometry["fit_index"].shape[0]):
        fit_index = int(geometry["fit_index"][row])
        first = geometry_index[(fit_index, 0)]
        fit_key = (
            int(fit["block_index"][fit_index]),
            int(fit["seed_position"][fit_index]),
            int(fit["local_view"][fit_index]),
        )
        target_row = prerequisite_index[fit_key]
        for name in ("xy_raw", "diag_raw", "off_raw", "built_xy", "built_chol"):
            _assert_np_exact(
                f"mechanism_geometry/{name}", geometry[name][row], geometry[name][first]
            )
        for name in ("xy_raw", "diag_raw", "off_raw"):
            _assert_np_exact(
                f"mechanism_geometry/initial/{name}",
                geometry[name][first],
                prerequisite[f"current_{name}"][target_row],
            )
        xy_raw = _torch_row(geometry["xy_raw"][row])
        diag_raw = _torch_row(geometry["diag_raw"][row])
        expected_xy = torch.sigmoid(xy_raw) * xy_raw.new_tensor([IMAGE_SIZE, IMAGE_SIZE])
        expected_diag = torch.nn.functional.softplus(diag_raw) + 0.3
        expected_chol = torch.stack(
            [expected_diag[:, 0], _torch_row(geometry["off_raw"][row]), expected_diag[:, 1]],
            dim=-1,
        )
        _assert_np_exact(
            "mechanism_geometry/built_xy_from_raw", geometry["built_xy"][row], expected_xy
        )
        _assert_np_exact(
            "mechanism_geometry/built_chol_from_raw", geometry["built_chol"][row], expected_chol
        )
        if int(fit["arm_code"][fit_index]) == 1 and not np.array_equal(
            geometry["built_weight"][row], np.ones(COMPONENTS, dtype=np.float32)
        ):
            raise ProtocolInvalid("reduction", "candidate mechanism state weight differs")
    checkpoint = tables["checkpoint"]
    for row in range(checkpoint["fit_index"].shape[0]):
        fit_index = int(checkpoint["fit_index"][row])
        if int(fit["block_index"][fit_index]) != 0:
            continue
        step = int(checkpoint["step"][row])
        geometry_row = geometry_index[(fit_index, step)]
        for name in (
            "xy_raw",
            "diag_raw",
            "off_raw",
            "built_xy",
            "built_chol",
            "built_weight",
        ):
            _assert_np_exact(
                f"mechanism_geometry/checkpoint/{name}",
                checkpoint[name][row],
                geometry[name][geometry_row],
            )
    for arm_code, table_name in enumerate(("mechanism_current", "mechanism_candidate")):
        table = tables[table_name]
        for row in range(table["fit_index"].shape[0]):
            fit_index = int(table["fit_index"][row])
            completed_step = int(table["completed_step"][row])
            lr = float(table["lr_used"][row])
            expected_lr = float(fit["lr_trace"][fit_index, completed_step])
            if lr != expected_lr:
                raise ProtocolInvalid("reduction", f"{table_name} LR differs from fit transcript")
            geometry_row = geometry_index[(fit_index, completed_step)]
            fit_key = (
                int(fit["block_index"][fit_index]),
                int(fit["seed_position"][fit_index]),
                int(fit["local_view"][fit_index]),
            )
            target_row = prerequisite_index[fit_key]
            target = _torch_row(prerequisite["target"][target_row])
            xy = _torch_row(geometry["built_xy"][geometry_row])
            chol = _torch_row(geometry["built_chol"][geometry_row])
            if arm_code == 0:
                color_raw = _torch_row(table["color_raw_pre"][row])
                weight_raw = _torch_row(table["weight_raw_pre"][row])
                color = torch.sigmoid(color_raw)
                weight = torch.sigmoid(weight_raw)
                _assert_np_exact("mechanism_current/built_color", table["built_color"][row], color)
                _assert_np_exact(
                    "mechanism_current/built_weight", table["built_weight"][row], weight
                )
                _assert_np_exact(
                    "mechanism_current/built_amplitude",
                    table["built_amplitude"][row],
                    weight[:, None] * color,
                )
                grad_a = _torch_row(table["gradient_amplitude"][row])
                actual_color_grad = _torch_row(table["gradient_color"][row])
                actual_weight_grad = _torch_row(table["gradient_weight"][row])
                expected = chain_rule_expected(ARMS[0], grad_a, weight, color)
                _assert_np_exact(
                    "mechanism_current/expected_gradient_color",
                    table["expected_gradient_color"][row],
                    expected["u"],
                )
                _assert_np_exact(
                    "mechanism_current/expected_gradient_weight",
                    table["expected_gradient_weight"][row],
                    expected["s"],
                )
                _assert_frozen_error_pair(
                    "mechanism_current/gradient_color",
                    actual_color_grad,
                    expected["u"],
                    absolute_limit=CHAIN_RULE_ABSOLUTE_LIMIT,
                    relative_limit=CHAIN_RULE_RELATIVE_LIMIT,
                    relative_threshold=CHAIN_RULE_RELATIVE_MAGNITUDE_THRESHOLD,
                    recorded=table["equation_errors"][row, 0:2],
                )
                _assert_frozen_error_pair(
                    "mechanism_current/gradient_weight",
                    actual_weight_grad,
                    expected["s"],
                    absolute_limit=CHAIN_RULE_ABSOLUTE_LIMIT,
                    relative_limit=CHAIN_RULE_RELATIVE_LIMIT,
                    relative_threshold=CHAIN_RULE_RELATIVE_MAGNITUDE_THRESHOLD,
                    recorded=table["equation_errors"][row, 2:4],
                )
                names = ("color", "weight")
                raw_names = ("color_raw", "weight_raw")
                gradients = (actual_color_grad, actual_weight_grad)
                for position, (short, raw_name, gradient) in enumerate(
                    zip(names, raw_names, gradients, strict=True)
                ):
                    before = _torch_row(table[f"{raw_name}_pre"][row])
                    avg_before = _torch_row(table[f"exp_avg_{short}_before"][row])
                    sq_before = _torch_row(table[f"exp_avg_sq_{short}_before"][row])
                    if not bool(table["state_present_before"][row, position]) and (
                        bool(avg_before.any()) or bool(sq_before.any())
                    ):
                        raise ProtocolInvalid(
                            "reduction", "absent mechanism Adam state is not exact zero"
                        )
                    reconstructed = adam_reconstruct(
                        before,
                        gradient,
                        avg_before,
                        sq_before,
                        int(table["step_before"][row, position]),
                        lr,
                    )
                    if int(table["step_before"][row, position]) != completed_step:
                        raise ProtocolInvalid("reduction", "mechanism current pre-step differs")
                    if int(table["step_after"][row, position]) != completed_step + 1:
                        raise ProtocolInvalid("reduction", "mechanism current post-step differs")
                    _assert_np_exact(
                        f"mechanism_current/exp_avg_{short}_after",
                        table[f"exp_avg_{short}_after"][row],
                        reconstructed["exp_avg_after"],
                    )
                    _assert_np_exact(
                        f"mechanism_current/exp_avg_sq_{short}_after",
                        table[f"exp_avg_sq_{short}_after"][row],
                        reconstructed["exp_avg_sq_after"],
                    )
                    _assert_np_exact(
                        f"mechanism_current/reconstructed_displacement_{short}",
                        table[f"reconstructed_displacement_{short}"][row],
                        reconstructed["displacement"],
                    )
                    actual_displacement = _torch_row(table[f"{raw_name}_after"][row]) - before
                    _assert_np_exact(
                        f"mechanism_current/displacement_raw_difference_{short}",
                        table[f"displacement_{short}"][row],
                        actual_displacement,
                    )
                    error_offset = 4 + 2 * position
                    _assert_frozen_error_pair(
                        f"mechanism_current/displacement_{short}",
                        _torch_row(table[f"displacement_{short}"][row]),
                        reconstructed["displacement"],
                        absolute_limit=2e-7,
                        relative_limit=2e-5,
                        recorded=table["equation_errors"][row, error_offset : error_offset + 2],
                    )
                    expected_present = completed_step > 0
                    if bool(table["state_present_before"][row, position]) is not expected_present:
                        raise ProtocolInvalid(
                            "reduction", "mechanism current state presence differs"
                        )
                    if not bool(table["state_present_after"][row, position]):
                        raise ProtocolInvalid("reduction", "mechanism current post state is absent")
                jacobian = current_jacobian(weight, color)
                diagnostics = jacobian_diagnostics(jacobian, current=True)
                analytic = analytic_current_null(
                    weight, color, diagnostics["null_vector"], diagnostics["rank"]
                )
                theta_gradient = torch.cat([actual_color_grad, actual_weight_grad[:, None]], dim=-1)
                theta_displacement = torch.cat(
                    [
                        _torch_row(table["displacement_color"][row]),
                        _torch_row(table["displacement_weight"][row])[:, None],
                    ],
                    dim=-1,
                )
                null = null_update_rows(jacobian, theta_gradient, theta_displacement)
                for name, expected_value in {
                    "jacobian": jacobian,
                    "singular_values": diagnostics["singular_values"],
                    "vh": diagnostics["vh"],
                    "rank": diagnostics["rank"],
                    "null_vector": diagnostics["null_vector"],
                    "null_residual": diagnostics["null_residual"],
                    "analytic_null_vector": analytic["vector"],
                    "analytic_null_alignment": analytic["alignment"],
                    "analytic_null_defined": analytic["defined"],
                    "null_dot": null["dot"],
                    "update_norm": null["update_norm"],
                    "null_eligible": null["eligible"],
                    "null_fraction": null["null_fraction"],
                    "squared_projection": null["squared_projection"],
                    "squared_update_norm": null["squared_update_norm"],
                    "gradient_dot": null["gradient_dot"],
                    "gradient_norm": null["gradient_norm"],
                    "gradient_cosine": null["gradient_cosine"],
                    "gradient_cosine_defined": null["gradient_cosine_defined"],
                }.items():
                    _assert_np_exact(f"mechanism_current/{name}", table[name][row], expected_value)
                if bool((null["gradient_dot"].abs() > 2e-6).any()):
                    raise ProtocolInvalid("reduction", "current gradient/null absolute gate failed")
                if bool((null["gradient_cosine_defined"] & (null["gradient_cosine"] > 2e-5)).any()):
                    raise ProtocolInvalid("reduction", "current gradient/null relative gate failed")
            else:
                amplitude_raw = _torch_row(table["amplitude_raw_pre"][row])
                color = torch.sigmoid(amplitude_raw)
                weight = torch.ones(COMPONENTS, dtype=color.dtype)
                _assert_np_exact(
                    "mechanism_candidate/built_color", table["built_color"][row], color
                )
                _assert_np_exact(
                    "mechanism_candidate/built_weight", table["built_weight"][row], weight
                )
                _assert_np_exact(
                    "mechanism_candidate/built_amplitude", table["built_amplitude"][row], color
                )
                if not torch.equal(weight, torch.ones_like(weight)):
                    raise ProtocolInvalid("reduction", "candidate mechanism update weight differs")
                grad_a = _torch_row(table["gradient_amplitude"][row])
                actual_gradient = _torch_row(table["gradient_amplitude_raw"][row])
                expected_gradient = chain_rule_expected(ARMS[1], grad_a, weight, color)["r"]
                _assert_np_exact(
                    "mechanism_candidate/expected_gradient",
                    table["expected_gradient"][row],
                    expected_gradient,
                )
                _assert_frozen_error_pair(
                    "mechanism_candidate/gradient",
                    actual_gradient,
                    expected_gradient,
                    absolute_limit=CHAIN_RULE_ABSOLUTE_LIMIT,
                    relative_limit=CHAIN_RULE_RELATIVE_LIMIT,
                    relative_threshold=CHAIN_RULE_RELATIVE_MAGNITUDE_THRESHOLD,
                    recorded=table["equation_errors"][row, 0:2],
                )
                reconstructed = adam_reconstruct(
                    _torch_row(table["amplitude_raw_pre"][row]),
                    actual_gradient,
                    _torch_row(table["exp_avg_before"][row]),
                    _torch_row(table["exp_avg_sq_before"][row]),
                    int(table["step_before"][row]),
                    lr,
                )
                if not bool(table["state_present_before"][row]) and (
                    bool(table["exp_avg_before"][row].any())
                    or bool(table["exp_avg_sq_before"][row].any())
                ):
                    raise ProtocolInvalid(
                        "reduction", "absent candidate Adam state is not exact zero"
                    )
                if int(table["step_before"][row]) != completed_step:
                    raise ProtocolInvalid("reduction", "candidate mechanism pre-step differs")
                if int(table["step_after"][row]) != completed_step + 1:
                    raise ProtocolInvalid("reduction", "candidate mechanism post-step differs")
                _assert_np_exact(
                    "mechanism_candidate/exp_avg_after",
                    table["exp_avg_after"][row],
                    reconstructed["exp_avg_after"],
                )
                _assert_np_exact(
                    "mechanism_candidate/exp_avg_sq_after",
                    table["exp_avg_sq_after"][row],
                    reconstructed["exp_avg_sq_after"],
                )
                _assert_np_exact(
                    "mechanism_candidate/reconstructed_displacement",
                    table["reconstructed_displacement"][row],
                    reconstructed["displacement"],
                )
                _assert_np_exact(
                    "mechanism_candidate/displacement_raw_difference",
                    table["displacement"][row],
                    _torch_row(table["amplitude_raw_after"][row])
                    - _torch_row(table["amplitude_raw_pre"][row]),
                )
                _assert_frozen_error_pair(
                    "mechanism_candidate/displacement",
                    _torch_row(table["displacement"][row]),
                    reconstructed["displacement"],
                    absolute_limit=2e-7,
                    relative_limit=2e-5,
                    recorded=table["equation_errors"][row, 2:4],
                )
                if bool(table["state_present_before"][row]) is not (completed_step > 0):
                    raise ProtocolInvalid("reduction", "candidate state presence differs")
                if not bool(table["state_present_after"][row]):
                    raise ProtocolInvalid("reduction", "candidate post state is absent")
            if deep:
                expected_grad_a, probe_values = _recompute_probe(target, xy, chol, color, weight)
                _assert_np_exact(
                    f"{table_name}/gradient_amplitude",
                    table["gradient_amplitude"][row],
                    expected_grad_a,
                )
                _assert_np_exact(
                    f"{table_name}/probe_values",
                    table["probe_values"][row],
                    probe_values,
                )
                _assert_np_exact(
                    f"{table_name}/objective_loss",
                    table["objective_loss"][row],
                    np.asarray(probe_values[6], dtype=np.float32),
                )
                if (
                    probe_values[2] <= 0.0
                    or probe_values[6] <= 0.0
                    or probe_values[0] > 2e-5
                    or probe_values[3] > 2e-6
                    or probe_values[5] > 2e-6
                    or probe_values[7] > 2e-5
                ):
                    raise ProtocolInvalid("reduction", f"{table_name} amplitude-probe gate failed")

    checkpoint_tables = {
        0: tables["checkpoint_current"],
        1: tables["checkpoint_candidate"],
    }
    for arm_code, table_name in enumerate(("mechanism_current", "mechanism_candidate")):
        table = tables[table_name]
        update_by_key = {
            (int(fit_index), int(step)): row
            for row, (fit_index, step) in enumerate(
                zip(table["fit_index"], table["completed_step"], strict=True)
            )
        }
        arm_checkpoint = checkpoint_tables[arm_code]
        checkpoint_by_key = {
            (int(fit_index), int(step)): row
            for row, (fit_index, step) in enumerate(
                zip(arm_checkpoint["fit_index"], arm_checkpoint["step"], strict=True)
            )
            if int(fit["block_index"][int(fit_index)]) == 0
        }
        raw_pairs = (
            (
                ("color_raw_pre", "color_raw_after", "color_raw"),
                ("weight_raw_pre", "weight_raw_after", "weight_raw"),
            )
            if arm_code == 0
            else (("amplitude_raw_pre", "amplitude_raw_after", "amplitude_raw"),)
        )
        for row in range(table["fit_index"].shape[0]):
            fit_index = int(table["fit_index"][row])
            completed_step = int(table["completed_step"][row])
            geometry_pre = geometry_index[(fit_index, completed_step)]
            _assert_np_exact(
                f"{table_name}/geometry_weight_pre",
                geometry["built_weight"][geometry_pre],
                table["built_weight"][row],
            )
            for pre_name, after_name, checkpoint_name in raw_pairs:
                if completed_step + 1 < UPDATES:
                    next_row = update_by_key[(fit_index, completed_step + 1)]
                    _assert_np_exact(
                        f"{table_name}/state_chain/{checkpoint_name}",
                        table[pre_name][next_row],
                        table[after_name][row],
                    )
                if completed_step in CHECKPOINTS:
                    checkpoint_row = checkpoint_by_key[(fit_index, completed_step)]
                    _assert_np_exact(
                        f"{table_name}/checkpoint_pre/{checkpoint_name}",
                        arm_checkpoint[checkpoint_name][checkpoint_row],
                        table[pre_name][row],
                    )
                if completed_step == UPDATES - 1:
                    checkpoint_row = checkpoint_by_key[(fit_index, UPDATES)]
                    _assert_np_exact(
                        f"{table_name}/checkpoint_terminal/{checkpoint_name}",
                        arm_checkpoint[checkpoint_name][checkpoint_row],
                        table[after_name][row],
                    )
            geometry_post = geometry_index[(fit_index, completed_step + 1)]
            if arm_code == 0:
                expected_post_weight = torch.sigmoid(_torch_row(table["weight_raw_after"][row]))
            else:
                expected_post_weight = torch.ones(COMPONENTS, dtype=torch.float32)
            _assert_np_exact(
                f"{table_name}/geometry_weight_post",
                geometry["built_weight"][geometry_post],
                expected_post_weight,
            )
            if completed_step + 1 < UPDATES:
                next_row = update_by_key[(fit_index, completed_step + 1)]
                if arm_code == 0:
                    moment_pairs = (
                        ("exp_avg_color_after", "exp_avg_color_before"),
                        ("exp_avg_sq_color_after", "exp_avg_sq_color_before"),
                        ("exp_avg_weight_after", "exp_avg_weight_before"),
                        ("exp_avg_sq_weight_after", "exp_avg_sq_weight_before"),
                        ("step_after", "step_before"),
                        ("state_present_after", "state_present_before"),
                    )
                else:
                    moment_pairs = (
                        ("exp_avg_after", "exp_avg_before"),
                        ("exp_avg_sq_after", "exp_avg_sq_before"),
                        ("step_after", "step_before"),
                        ("state_present_after", "state_present_before"),
                    )
                for after_name, before_name in moment_pairs:
                    _assert_np_exact(
                        f"{table_name}/optimizer_state_chain/{after_name}",
                        table[after_name][row],
                        table[before_name][next_row],
                    )
            if arm_code == 0 and completed_step in CHECKPOINTS:
                checkpoint_row = checkpoint_by_key[(fit_index, completed_step)]
                for name in (
                    "jacobian",
                    "singular_values",
                    "vh",
                    "rank",
                    "null_vector",
                    "null_residual",
                    "analytic_null_vector",
                    "analytic_null_alignment",
                    "analytic_null_defined",
                ):
                    _assert_np_exact(
                        f"{table_name}/checkpoint_subset/{name}",
                        table[name][row],
                        arm_checkpoint[name][checkpoint_row],
                    )


def _recompute_joint(tables: Mapping[str, Mapping[str, np.ndarray]], *, deep: bool) -> None:
    fit = tables["fit"]
    checkpoint = tables["checkpoint"]
    prerequisite = tables["prerequisite"]
    current_checkpoint = tables["checkpoint_current"]
    candidate_checkpoint = tables["checkpoint_candidate"]
    current_raw_by_key = {
        (int(fit_index), int(step)): row
        for row, (fit_index, step) in enumerate(
            zip(current_checkpoint["fit_index"], current_checkpoint["step"], strict=True)
        )
    }
    candidate_raw_by_key = {
        (int(fit_index), int(step)): row
        for row, (fit_index, step) in enumerate(
            zip(candidate_checkpoint["fit_index"], candidate_checkpoint["step"], strict=True)
        )
    }
    checkpoint_by_key = {
        (int(fit_index), int(step)): row
        for row, (fit_index, step) in enumerate(
            zip(checkpoint["fit_index"], checkpoint["step"], strict=True)
        )
    }
    prerequisite_by_key = {
        (
            int(prerequisite["block_index"][row]),
            int(prerequisite["seed_position"][row]),
            int(prerequisite["local_view"][row]),
        ): row
        for row in range(SCIENTIFIC_PLAN.initializers)
    }
    for current, table_name in ((True, "joint_current"), (False, "joint_candidate")):
        table = tables[table_name]
        names = (
            ("xy_raw", "diag_raw", "off_raw", "color_raw", "weight_raw")
            if current
            else ("xy_raw", "diag_raw", "off_raw", "amplitude_raw")
        )
        for row in range(table["fit_index"].shape[0]):
            fit_index = int(table["fit_index"][row])
            step = int(table["step"][row])
            defined = bool(table["defined"][row])
            if step == 0:
                if defined:
                    raise ProtocolInvalid("reduction", "joint step-zero transition is defined")
                for name, value in table.items():
                    if name in {"fit_index", "step", "defined"}:
                        continue
                    if bool(value[row].any()):
                        raise ProtocolInvalid(
                            "reduction", f"joint step-zero null value differs: {name}"
                        )
                continue
            if not defined:
                raise ProtocolInvalid("reduction", "positive joint transition is undefined")
            _assert_np_exact(
                f"{table_name}/lr_used",
                table["lr_used"][row],
                fit["lr_trace"][fit_index, step - 1],
            )
            for position, name in enumerate(names):
                expected_present = step > 1
                if bool(table["state_present_before"][row, position]) is not expected_present:
                    raise ProtocolInvalid("reduction", "joint pre-state presence differs")
                if not bool(table["state_present_after"][row, position]):
                    raise ProtocolInvalid("reduction", "joint post-state is absent")
                if int(table["step_before"][row, position]) != step - 1:
                    raise ProtocolInvalid("reduction", "joint optimizer pre-step differs")
                if int(table["step_after"][row, position]) != step:
                    raise ProtocolInvalid("reduction", "joint optimizer post-step differs")
                if not bool(table["state_present_before"][row, position]) and (
                    bool(table[f"exp_avg_before_{name}"][row].any())
                    or bool(table[f"exp_avg_sq_before_{name}"][row].any())
                ):
                    raise ProtocolInvalid("reduction", "absent joint Adam state is not exact zero")
                reconstructed = adam_reconstruct(
                    _torch_row(table[f"pre_{name}"][row]),
                    _torch_row(table[f"gradient_{name}"][row]),
                    _torch_row(table[f"exp_avg_before_{name}"][row]),
                    _torch_row(table[f"exp_avg_sq_before_{name}"][row]),
                    int(table["step_before"][row, position]),
                    float(table["lr_used"][row]),
                )
                _assert_np_exact(
                    f"{table_name}/exp_avg_after/{name}",
                    table[f"exp_avg_after_{name}"][row],
                    reconstructed["exp_avg_after"],
                )
                _assert_np_exact(
                    f"{table_name}/exp_avg_sq_after/{name}",
                    table[f"exp_avg_sq_after_{name}"][row],
                    reconstructed["exp_avg_sq_after"],
                )
                _assert_np_exact(
                    f"{table_name}/displacement_raw_difference/{name}",
                    table[f"displacement_{name}"][row],
                    _torch_row(table[f"post_{name}"][row]) - _torch_row(table[f"pre_{name}"][row]),
                )
                _assert_frozen_error_pair(
                    f"{table_name}/displacement/{name}",
                    _torch_row(table[f"displacement_{name}"][row]),
                    reconstructed["displacement"],
                    absolute_limit=2e-7,
                    relative_limit=2e-5,
                )
            if step == 1:
                checkpoint_zero = checkpoint_by_key[(fit_index, 0)]
                for name in ("xy_raw", "diag_raw", "off_raw"):
                    _assert_np_exact(
                        f"{table_name}/step1_pre_checkpoint0/{name}",
                        table[f"pre_{name}"][row],
                        checkpoint[name][checkpoint_zero],
                    )
                raw_table = current_checkpoint if current else candidate_checkpoint
                raw_map = current_raw_by_key if current else candidate_raw_by_key
                raw_zero = raw_map[(fit_index, 0)]
                appearance_names = ("color_raw", "weight_raw") if current else ("amplitude_raw",)
                for name in appearance_names:
                    _assert_np_exact(
                        f"{table_name}/step1_pre_checkpoint0/{name}",
                        table[f"pre_{name}"][row],
                        raw_table[name][raw_zero],
                    )
            if deep:
                raw_variables = {
                    name: _torch_row(table[f"pre_{name}"][row])
                    .detach()
                    .clone()
                    .requires_grad_(True)
                    for name in names
                }
                build_raw: dict[str, torch.Tensor] = {
                    "xy_raw": raw_variables["xy_raw"],
                    "diag_raw": raw_variables["diag_raw"],
                    "off_raw": raw_variables["off_raw"],
                }
                arm = ARMS[0] if current else ARMS[1]
                if current:
                    build_raw.update(
                        {"u": raw_variables["color_raw"], "s": raw_variables["weight_raw"]}
                    )
                else:
                    build_raw["r"] = raw_variables["amplitude_raw"]
                built = build_from_raw(build_raw, arm, IMAGE_SIZE, IMAGE_SIZE)
                rendered = render_gaussians_2d(built, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64)
                fit_key = (
                    int(fit["block_index"][fit_index]),
                    int(fit["seed_position"][fit_index]),
                    int(fit["local_view"][fit_index]),
                )
                target = _torch_row(prerequisite["target"][prerequisite_by_key[fit_key]])
                loss = torch.nn.functional.mse_loss(rendered, target)
                gradients = torch.autograd.grad(loss, tuple(raw_variables[name] for name in names))
                for name, gradient in zip(names, gradients, strict=True):
                    _assert_np_exact(
                        f"{table_name}/recomputed_gradient/{name}",
                        table[f"gradient_{name}"][row],
                        gradient,
                    )
            checkpoint_row = checkpoint_by_key[(fit_index, step)]
            for name in ("xy_raw", "diag_raw", "off_raw"):
                _assert_np_exact(
                    f"{table_name}/post_checkpoint/{name}",
                    table[f"post_{name}"][row],
                    checkpoint[name][checkpoint_row],
                )
            raw_table = current_checkpoint if current else candidate_checkpoint
            raw_map = current_raw_by_key if current else candidate_raw_by_key
            raw_row = raw_map[(fit_index, step)]
            appearance_names = ("color_raw", "weight_raw") if current else ("amplitude_raw",)
            for name in appearance_names:
                _assert_np_exact(
                    f"{table_name}/post_checkpoint/{name}",
                    table[f"post_{name}"][row],
                    raw_table[name][raw_row],
                )
    if deep:
        _replay_joint_trajectories(tables)


def _replay_joint_checkpoint(
    snapshot: NativeFitDiagnostic,
    *,
    checkpoint: Mapping[str, np.ndarray],
    checkpoint_row: int,
    detail: Mapping[str, np.ndarray],
    detail_row: int,
    current: bool,
) -> None:
    """Bind a retained joint checkpoint to a deterministic production-path replay."""
    if snapshot.rendered is None or snapshot.loss is None:
        raise ProtocolInvalid("reduction", "joint replay checkpoint lacks render/loss")
    for name in ("xy_raw", "diag_raw", "off_raw"):
        _assert_np_exact(
            f"joint_replay/checkpoint/{name}",
            checkpoint[name][checkpoint_row],
            snapshot.raw_parameters[name],
        )
    appearance_names = ("color_raw", "weight_raw") if current else ("amplitude_raw",)
    for name in appearance_names:
        _assert_np_exact(
            f"joint_replay/checkpoint/{name}",
            detail[name][detail_row],
            snapshot.raw_parameters[name],
        )
    for stored_name, built_name in (
        ("built_xy", "xy"),
        ("built_chol", "chol"),
        ("built_color", "color"),
        ("built_weight", "weight"),
    ):
        _assert_np_exact(
            f"joint_replay/checkpoint/{stored_name}",
            checkpoint[stored_name][checkpoint_row],
            getattr(snapshot.gaussians, built_name),
        )
    _assert_np_exact(
        "joint_replay/checkpoint/built_amplitude",
        checkpoint["built_amplitude"][checkpoint_row],
        snapshot.gaussians.weight[:, None] * snapshot.gaussians.color,
    )
    _assert_np_exact(
        "joint_replay/checkpoint/render",
        checkpoint["render"][checkpoint_row],
        snapshot.rendered,
    )
    _assert_np_exact(
        "joint_replay/checkpoint/target",
        checkpoint["target"][checkpoint_row],
        snapshot.target,
    )
    _assert_np_exact(
        "joint_replay/checkpoint/objective_loss",
        checkpoint["objective_loss"][checkpoint_row],
        snapshot.loss,
    )


def _replay_joint_transition(
    snapshot: NativeFitDiagnostic,
    *,
    transition: Mapping[str, np.ndarray],
    transition_row: int,
    names: tuple[str, ...],
    before: bool,
) -> None:
    """Compare one retained joint transition endpoint to the replayed Adam state."""
    prefix = "before" if before else "after"
    raw_prefix = "pre" if before else "post"
    state_prefix = "before" if before else "after"
    expected_step = snapshot.step
    if before:
        if snapshot.lr_used is None:
            raise ProtocolInvalid("reduction", "joint replay pre-update LR is absent")
        _assert_np_exact(
            "joint_replay/transition/lr_used",
            transition["lr_used"][transition_row],
            np.asarray(snapshot.lr_used, dtype=np.float64),
        )
    for position, name in enumerate(names):
        _assert_np_exact(
            f"joint_replay/transition/{raw_prefix}_{name}",
            transition[f"{raw_prefix}_{name}"][transition_row],
            snapshot.raw_parameters[name],
        )
        exp_avg, exp_avg_sq, step = _optimizer_state_at(
            snapshot, name, expected_step, phase="reduction"
        )
        _assert_np_exact(
            f"joint_replay/transition/exp_avg_{state_prefix}_{name}",
            transition[f"exp_avg_{state_prefix}_{name}"][transition_row],
            exp_avg,
        )
        _assert_np_exact(
            f"joint_replay/transition/exp_avg_sq_{state_prefix}_{name}",
            transition[f"exp_avg_sq_{state_prefix}_{name}"][transition_row],
            exp_avg_sq,
        )
        _assert_np_exact(
            f"joint_replay/transition/step_{prefix}",
            transition[f"step_{prefix}"][transition_row, position],
            np.asarray(step, dtype=np.int64),
        )
        _assert_np_exact(
            f"joint_replay/transition/state_present_{prefix}",
            transition[f"state_present_{prefix}"][transition_row, position],
            np.asarray(bool(snapshot.optimizer_state[name]), dtype=np.bool_),
        )
        if before:
            gradient = snapshot.gradients[name]
            if gradient is None:
                raise ProtocolInvalid("reduction", f"joint replay gradient is absent: {name}")
            _assert_np_exact(
                f"joint_replay/transition/gradient_{name}",
                transition[f"gradient_{name}"][transition_row],
                gradient,
            )


def _replay_joint_trajectories(
    tables: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    """Replay all 54 joint fits from stored common initializers through 120 exact updates."""
    fit = tables["fit"]
    checkpoint = tables["checkpoint"]
    prerequisite = tables["prerequisite"]
    checkpoint_by_key = {
        (int(fit_index), int(step)): row
        for row, (fit_index, step) in enumerate(
            zip(checkpoint["fit_index"], checkpoint["step"], strict=True)
        )
    }
    prerequisite_by_key = {
        (int(block), int(seed_position), int(local_view)): row
        for row, (block, seed_position, local_view) in enumerate(
            zip(
                prerequisite["block_index"],
                prerequisite["seed_position"],
                prerequisite["local_view"],
                strict=True,
            )
        )
    }
    detail_tables = {
        0: tables["checkpoint_current"],
        1: tables["checkpoint_candidate"],
    }
    detail_by_key = {
        arm_code: {
            (int(fit_index), int(step)): row
            for row, (fit_index, step) in enumerate(
                zip(table["fit_index"], table["step"], strict=True)
            )
        }
        for arm_code, table in detail_tables.items()
    }
    transition_tables = {
        0: tables["joint_current"],
        1: tables["joint_candidate"],
    }
    transition_by_key = {
        arm_code: {
            (int(fit_index), int(step)): row
            for row, (fit_index, step) in enumerate(
                zip(table["fit_index"], table["step"], strict=True)
            )
        }
        for arm_code, table in transition_tables.items()
    }

    replayed_fits = 0
    for fit_index in range(SCIENTIFIC_PLAN.fits):
        if int(fit["block_index"][fit_index]) != 1:
            continue
        arm_code = int(fit["arm_code"][fit_index])
        current = arm_code == 0
        arm = ARMS[arm_code]
        names = (
            ("xy_raw", "diag_raw", "off_raw", "color_raw", "weight_raw")
            if current
            else ("xy_raw", "diag_raw", "off_raw", "amplitude_raw")
        )
        key = (
            int(fit["block_index"][fit_index]),
            int(fit["seed_position"][fit_index]),
            int(fit["local_view"][fit_index]),
        )
        prerequisite_row = prerequisite_by_key[key]
        target = _torch_row(prerequisite["target"][prerequisite_row])
        g0 = Gaussians2D(
            xy=_torch_row(prerequisite["g0_xy"][prerequisite_row]),
            chol=_torch_row(prerequisite["g0_chol"][prerequisite_row]),
            color=_torch_row(prerequisite["g0_color"][prerequisite_row]),
            weight=_torch_row(prerequisite["g0_weight"][prerequisite_row]),
        )
        detail = detail_tables[arm_code]
        transition = transition_tables[arm_code]
        event_counter = [0]

        def replay_callback(
            snapshot: NativeFitDiagnostic,
            fit_index: int = fit_index,
            arm_code: int = arm_code,
            current: bool = current,
            names: tuple[str, ...] = names,
            detail: Mapping[str, np.ndarray] = detail,
            transition: Mapping[str, np.ndarray] = transition,
            event_counter: list[int] = event_counter,
        ) -> None:
            event_counter[0] += 1
            if snapshot.event == "initial":
                checkpoint_row = checkpoint_by_key[(fit_index, 0)]
                detail_row = detail_by_key[arm_code][(fit_index, 0)]
                _replay_joint_checkpoint(
                    snapshot,
                    checkpoint=checkpoint,
                    checkpoint_row=checkpoint_row,
                    detail=detail,
                    detail_row=detail_row,
                    current=current,
                )
                return
            if snapshot.event == "pre_update" and snapshot.step + 1 in POSITIVE_CHECKPOINTS:
                transition_row = transition_by_key[arm_code][(fit_index, snapshot.step + 1)]
                _replay_joint_transition(
                    snapshot,
                    transition=transition,
                    transition_row=transition_row,
                    names=names,
                    before=True,
                )
                return
            if snapshot.event == "post_update" and snapshot.step in POSITIVE_CHECKPOINTS:
                transition_row = transition_by_key[arm_code][(fit_index, snapshot.step)]
                _replay_joint_transition(
                    snapshot,
                    transition=transition,
                    transition_row=transition_row,
                    names=names,
                    before=False,
                )
                return
            if snapshot.event == "checkpoint":
                checkpoint_row = checkpoint_by_key[(fit_index, snapshot.step)]
                detail_row = detail_by_key[arm_code][(fit_index, snapshot.step)]
                _replay_joint_checkpoint(
                    snapshot,
                    checkpoint=checkpoint,
                    checkpoint_row=checkpoint_row,
                    detail=detail,
                    detail_row=detail_row,
                    current=current,
                )

        config = frozen_fit_config(arm)
        config.freeze_geometry = False
        rng_before = torch.random.get_rng_state().clone()
        result, history = fit_image_from_initialization(
            target,
            g0,
            config,
            mask=None,
            diagnostic_callback=replay_callback,
            diagnostic_steps=POSITIVE_CHECKPOINTS,
        )
        rng_after = torch.random.get_rng_state().clone()
        if not torch.equal(rng_before, rng_after):
            raise ProtocolInvalid("reduction", "joint trajectory replay changed CPU RNG")
        if event_counter[0] != 248:
            raise ProtocolInvalid("reduction", "joint trajectory replay event count differs")
        for result_name in ("xy", "chol", "color", "weight"):
            _assert_np_exact(
                f"joint_replay/result/{result_name}",
                fit[f"result_{result_name}"][fit_index],
                getattr(result, result_name),
            )
        if history.get("stopped_iter") != UPDATES - 1:
            raise ProtocolInvalid("reduction", "joint trajectory replay stopped early")
        _assert_np_exact(
            "joint_replay/final_psnr_full",
            fit["final_psnr_full"][fit_index],
            np.asarray(history["final_psnr_full"], dtype=np.float32),
        )
        _assert_np_exact(
            "joint_replay/final_psnr",
            fit["final_psnr"][fit_index],
            np.asarray(history["final_psnr"], dtype=np.float32),
        )
        replayed_fits += 1
    if replayed_fits != SCIENTIFIC_PLAN.fits // 2:
        raise ProtocolInvalid("reduction", "joint trajectory replay fit count differs")


def _pool_null_arrays(table: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    eligible = table["null_eligible"] & mask[:, None]
    universe_count = int(mask.sum()) * COMPONENTS
    eligible_count = int(eligible.sum())
    large_count = int(((table["null_fraction"] >= 0.10) & eligible).sum())
    numerator = float(table["squared_projection"][eligible].sum(dtype=np.float64))
    denominator = float(table["squared_update_norm"][eligible].sum(dtype=np.float64))
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ProtocolInvalid("reduction", "null reduction is non-finite")
    defined = eligible_count > 0 and denominator > 0
    return {
        "universe_count": universe_count,
        "eligible_count": eligible_count,
        "ineligible_count": universe_count - eligible_count,
        "large_count": large_count,
        "projection_energy_numerator": numerator,
        "update_energy_denominator": denominator,
        "null_energy_ratio": numerator / denominator if defined else 0.0,
        "null_large_fraction": large_count / eligible_count if eligible_count else 0.0,
        "defined": defined,
    }


def _fractions_from_saturation_counts(
    counts: np.ndarray, fraction_defined: np.ndarray
) -> np.ndarray:
    fractions = np.zeros(fraction_defined.shape, dtype=np.float64)
    denominators = (counts[:, 1], counts[:, 4], counts[:, 4], counts[:, 1])
    numerators = (counts[:, 0], counts[:, 2], counts[:, 3], counts[:, 5])
    for column, (numerator, denominator) in enumerate(zip(numerators, denominators, strict=True)):
        defined = fraction_defined[:, column]
        if bool((defined & (denominator <= 0)).any()):
            raise ProtocolInvalid("reduction", "defined saturation denominator is nonpositive")
        fractions[defined, column] = numerator[defined] / denominator[defined]
    return fractions


def _aggregate_checkpoint_diagnostics(
    cells: list[Mapping[str, Any]], identity: Mapping[str, Any]
) -> dict[str, Any]:
    if not cells:
        raise ProtocolInvalid("reduction", "empty checkpoint diagnostic pool")
    saturation_count_defined = np.asarray(cells[0]["saturation_count_defined"], dtype=np.bool_)
    saturation_fraction_defined = np.asarray(
        cells[0]["saturation_fraction_defined"], dtype=np.bool_
    )
    for cell in cells[1:]:
        if not np.array_equal(
            np.asarray(cell["saturation_count_defined"], dtype=np.bool_),
            saturation_count_defined,
        ) or not np.array_equal(
            np.asarray(cell["saturation_fraction_defined"], dtype=np.bool_),
            saturation_fraction_defined,
        ):
            raise ProtocolInvalid("reduction", "saturation defined masks differ within pool")
    saturation_counts = np.sum(
        np.asarray([cell["saturation_counts"] for cell in cells], dtype=np.int64),
        axis=0,
        dtype=np.int64,
    )
    histograms = np.sum(
        np.asarray([cell["histograms"] for cell in cells], dtype=np.int64),
        axis=0,
        dtype=np.int64,
    )
    saturation_fractions = _fractions_from_saturation_counts(
        saturation_counts, saturation_fraction_defined
    )
    rank_counts = np.sum(
        np.asarray([cell["rank_counts"] for cell in cells], dtype=np.int64),
        axis=0,
        dtype=np.int64,
    )
    component_count = len(cells) * COMPONENTS
    weak_count = sum(int(cell["weak_count"]) for cell in cells)
    below_count = sum(int(cell["clamp"]["below_count"]) for cell in cells)
    above_count = sum(int(cell["clamp"]["above_count"]) for cell in cells)
    channel_count = sum(int(cell["clamp"]["channel_count"]) for cell in cells)
    if int(rank_counts.sum()) != component_count or not 0 <= weak_count <= component_count:
        raise ProtocolInvalid("reduction", "checkpoint diagnostic pool population differs")
    if not np.array_equal(histograms.sum(axis=1, dtype=np.int64), saturation_counts[:, 4]):
        raise ProtocolInvalid("reduction", "checkpoint diagnostic histogram pool differs")
    if channel_count <= 0 or below_count + above_count > channel_count:
        raise ProtocolInvalid("reduction", "checkpoint clamp pool population differs")
    return {
        **identity,
        "checkpoint_cell_count": len(cells),
        "component_count": component_count,
        "rank_counts": rank_counts.tolist(),
        "smallest_positive_defined_count": sum(
            int(cell["smallest_positive_defined_count"]) for cell in cells
        ),
        "condition_defined_count": sum(int(cell["condition_defined_count"]) for cell in cells),
        "weak_count": weak_count,
        "weak_fraction": weak_count / component_count,
        "saturation_counts": saturation_counts.tolist(),
        "saturation_count_defined": saturation_count_defined.tolist(),
        "saturation_fractions": saturation_fractions.tolist(),
        "saturation_fraction_defined": saturation_fraction_defined.tolist(),
        "histograms": histograms.tolist(),
        "clamp": {
            "below_count": below_count,
            "above_count": above_count,
            "channel_count": channel_count,
            "below_fraction": below_count / channel_count,
            "above_fraction": above_count / channel_count,
        },
    }


def _checkpoint_diagnostic_reductions(
    tables: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    fit = tables["fit"]
    checkpoint = tables["checkpoint"]
    detail_by_checkpoint: dict[int, tuple[int, int]] = {}
    for arm_code, table_name in enumerate(("checkpoint_current", "checkpoint_candidate")):
        table = tables[table_name]
        for row, checkpoint_index in enumerate(table["checkpoint_index"]):
            key = int(checkpoint_index)
            if key in detail_by_checkpoint:
                raise ProtocolInvalid("reduction", "checkpoint diagnostic detail overlaps")
            detail_by_checkpoint[key] = (arm_code, row)
    expected_indices = set(range(SCIENTIFIC_PLAN.checkpoints))
    if set(detail_by_checkpoint) != expected_indices:
        raise ProtocolInvalid("reduction", "checkpoint diagnostic detail partition differs")

    cells: list[dict[str, Any]] = []
    for checkpoint_row in range(SCIENTIFIC_PLAN.checkpoints):
        checkpoint_index = int(checkpoint["checkpoint_index"][checkpoint_row])
        arm_code, detail_row = detail_by_checkpoint[checkpoint_index]
        detail_name = "checkpoint_current" if arm_code == 0 else "checkpoint_candidate"
        detail = tables[detail_name]
        fit_index = int(checkpoint["fit_index"][checkpoint_row])
        if (
            int(detail["fit_index"][detail_row]) != fit_index
            or int(detail["step"][detail_row]) != int(checkpoint["step"][checkpoint_row])
            or int(fit["arm_code"][fit_index]) != arm_code
        ):
            raise ProtocolInvalid("reduction", "checkpoint diagnostic identity differs")
        rank = detail["rank"][detail_row]
        if bool(((rank < 0) | (rank > 3)).any()):
            raise ProtocolInvalid("reduction", "checkpoint diagnostic rank is out of range")
        rank_counts = np.bincount(rank, minlength=4).astype(np.int64, copy=False)
        weak_count = int(detail["weakly_responsive"][detail_row].sum())
        saturation_counts = detail["saturation_counts"][detail_row]
        saturation_fraction_defined = detail["saturation_fraction_defined"][detail_row]
        saturation_fractions = detail["saturation_fractions"][detail_row]
        recomputed_fractions = _fractions_from_saturation_counts(
            saturation_counts, saturation_fraction_defined
        )
        if not np.array_equal(saturation_fractions, recomputed_fractions) or not np.array_equal(
            detail["histograms"][detail_row].sum(axis=1, dtype=np.int64),
            saturation_counts[:, 4],
        ):
            raise ProtocolInvalid("reduction", "checkpoint saturation reduction differs")
        channel_count = int(checkpoint["channel_count"][checkpoint_row])
        below_count = int(checkpoint["below_count"][checkpoint_row])
        above_count = int(checkpoint["above_count"][checkpoint_row])
        cell = {
            "checkpoint_index": checkpoint_index,
            "block": BLOCKS[int(fit["block_index"][fit_index])],
            "seed_position": int(fit["seed_position"][fit_index]),
            "seed": int(fit["seed"][fit_index]),
            "local_view": int(fit["local_view"][fit_index]),
            "original_view": int(fit["original_view"][fit_index]),
            "arm": ARMS[arm_code],
            "step": int(checkpoint["step"][checkpoint_row]),
            "component_count": COMPONENTS,
            "rank_counts": rank_counts.tolist(),
            "smallest_positive_defined_count": int(
                detail["smallest_positive_defined"][detail_row].sum()
            ),
            "condition_defined_count": int(detail["condition_defined"][detail_row].sum()),
            "weak_count": weak_count,
            "weak_fraction": weak_count / COMPONENTS,
            "saturation_counts": saturation_counts.tolist(),
            "saturation_count_defined": detail["saturation_count_defined"][detail_row].tolist(),
            "saturation_fractions": saturation_fractions.tolist(),
            "saturation_fraction_defined": detail["saturation_fraction_defined"][
                detail_row
            ].tolist(),
            "histograms": detail["histograms"][detail_row].tolist(),
            "clamp": {
                "below_count": below_count,
                "above_count": above_count,
                "channel_count": channel_count,
                "below_fraction": below_count / channel_count,
                "above_fraction": above_count / channel_count,
            },
        }
        if int(rank_counts.sum()) != COMPONENTS:
            raise ProtocolInvalid("reduction", "checkpoint rank population differs")
        cells.append(cell)

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for cell in cells:
        keys = (
            (
                "seed_step",
                cell["block"],
                cell["seed_position"],
                cell["seed"],
                cell["arm"],
                cell["step"],
            ),
            ("seed_arm", cell["block"], cell["seed_position"], cell["seed"], cell["arm"]),
            ("block_arm", cell["block"], cell["arm"]),
        )
        for key in keys:
            grouped.setdefault(key, []).append(cell)

    seed_step_pools = [
        _aggregate_checkpoint_diagnostics(
            grouped[("seed_step", block, seed_position, seed, arm, step)],
            {
                "block": block,
                "seed_position": seed_position,
                "seed": seed,
                "arm": arm,
                "step": step,
            },
        )
        for block in BLOCKS
        for seed_position, seed in enumerate(BLOCK_SEEDS[block])
        for arm in ARMS
        for step in CHECKPOINTS
    ]
    seed_arm_pools = [
        _aggregate_checkpoint_diagnostics(
            grouped[("seed_arm", block, seed_position, seed, arm)],
            {
                "block": block,
                "seed_position": seed_position,
                "seed": seed,
                "arm": arm,
            },
        )
        for block in BLOCKS
        for seed_position, seed in enumerate(BLOCK_SEEDS[block])
        for arm in ARMS
    ]
    block_arm_pools = [
        _aggregate_checkpoint_diagnostics(
            grouped[("block_arm", block, arm)], {"block": block, "arm": arm}
        )
        for block in BLOCKS
        for arm in ARMS
    ]
    if not (
        len(cells) == SCIENTIFIC_PLAN.view_metric_cells
        and len(seed_step_pools) == SCIENTIFIC_PLAN.seed_arm_checkpoint_cells
        and len(seed_arm_pools) == 12
        and len(block_arm_pools) == 4
    ):
        raise ProtocolInvalid("reduction", "checkpoint diagnostic reduction cardinality differs")
    return {
        "metadata": {
            "histogram_edges": list(HISTOGRAM_EDGES),
            "rank_count_order": [0, 1, 2, 3],
            "saturation_count_columns": [
                "raw_abs_ge_8",
                "raw_count",
                "output_low",
                "output_high",
                "output_count",
                "derivative_le_1e_4",
            ],
            "saturation_fraction_columns": [
                "raw_abs_ge_8",
                "output_low",
                "output_high",
                "derivative_le_1e_4",
            ],
            "arm_fields": {ARMS[0]: ["color", "weight", "amplitude"], ARMS[1]: ["amplitude"]},
        },
        "checkpoint_cells": cells,
        "seed_step_pools": seed_step_pools,
        "seed_arm_pools": seed_arm_pools,
        "block_arm_pools": block_arm_pools,
    }


def reduce_scientific_raw(tables: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    fit = tables["fit"]
    checkpoint = tables["checkpoint"]
    psnr_values = checkpoint["psnr"].astype(np.float64)
    ssim_values = checkpoint["ssim"].astype(np.float64)
    checkpoint_lookup = {
        (int(fit_index), int(step)): row
        for row, (fit_index, step) in enumerate(
            zip(checkpoint["fit_index"], checkpoint["step"], strict=True)
        )
    }
    block_metrics: dict[str, Any] = {}
    auc_by_key: dict[tuple[int, int, int], float] = {}
    curve_by_key: dict[tuple[int, int, int], list[float]] = {}
    ssim_curve_by_key: dict[tuple[int, int, int], list[float]] = {}
    for block_index, block in enumerate(BLOCKS):
        seed_rows: list[dict[str, Any]] = []
        for seed_position, seed in enumerate(BLOCK_SEEDS[block]):
            arm_rows: dict[str, Any] = {}
            for arm_code, arm in enumerate(ARMS):
                matching_fits = [
                    index
                    for index in range(SCIENTIFIC_PLAN.fits)
                    if int(fit["block_index"][index]) == block_index
                    and int(fit["seed_position"][index]) == seed_position
                    and int(fit["arm_code"][index]) == arm_code
                ]
                if len(matching_fits) != len(SELECTED_VIEWS):
                    raise ProtocolInvalid("reduction", "seed/arm fit population differs")
                psnr_curve: list[float] = []
                ssim_curve: list[float] = []
                pooled_sse: list[float] = []
                pooled_count: list[int] = []
                pooled_below_count: list[int] = []
                pooled_above_count: list[int] = []
                view_rows: list[dict[str, Any]] = []
                for fit_index in matching_fits:
                    rows = [checkpoint_lookup[(fit_index, step)] for step in CHECKPOINTS]
                    view_rows.append(
                        {
                            "local_view": int(fit["local_view"][fit_index]),
                            "original_view": int(fit["original_view"][fit_index]),
                            "psnr_by_checkpoint": psnr_values[rows].tolist(),
                            "ssim_by_checkpoint": ssim_values[rows].tolist(),
                            "sse_by_checkpoint": checkpoint["sse"][rows].tolist(),
                            "channel_count_by_checkpoint": checkpoint["channel_count"][
                                rows
                            ].tolist(),
                            "raw_mse_by_checkpoint": checkpoint["raw_mse"][rows].tolist(),
                            "below_count_by_checkpoint": checkpoint["below_count"][rows].tolist(),
                            "above_count_by_checkpoint": checkpoint["above_count"][rows].tolist(),
                            "below_fraction_by_checkpoint": checkpoint["below_fraction"][
                                rows
                            ].tolist(),
                            "above_fraction_by_checkpoint": checkpoint["above_fraction"][
                                rows
                            ].tolist(),
                        }
                    )
                if [row["local_view"] for row in view_rows] != list(range(len(SELECTED_VIEWS))):
                    raise ProtocolInvalid("reduction", "per-view metric order differs")
                for step in CHECKPOINTS:
                    rows = [checkpoint_lookup[(fit_index, step)] for fit_index in matching_fits]
                    psnr_curve.append(float(psnr_values[rows].mean(dtype=np.float64)))
                    ssim_curve.append(float(ssim_values[rows].mean(dtype=np.float64)))
                    pooled_sse.append(float(checkpoint["sse"][rows].sum(dtype=np.float64)))
                    pooled_count.append(int(checkpoint["channel_count"][rows].sum(dtype=np.int64)))
                    pooled_below_count.append(
                        int(checkpoint["below_count"][rows].sum(dtype=np.int64))
                    )
                    pooled_above_count.append(
                        int(checkpoint["above_count"][rows].sum(dtype=np.int64))
                    )
                auc = normalized_trapezoid_auc(psnr_curve)
                auc_by_key[(block_index, seed_position, arm_code)] = auc
                curve_by_key[(block_index, seed_position, arm_code)] = psnr_curve
                ssim_curve_by_key[(block_index, seed_position, arm_code)] = ssim_curve
                arm_rows[arm] = {
                    "psnr_by_checkpoint": psnr_curve,
                    "ssim_by_checkpoint": ssim_curve,
                    "psnr_auc_db": auc,
                    "pooled_sse_by_checkpoint": pooled_sse,
                    "pooled_channel_count_by_checkpoint": pooled_count,
                    "pooled_below_count_by_checkpoint": pooled_below_count,
                    "pooled_above_count_by_checkpoint": pooled_above_count,
                    "pooled_below_fraction_by_checkpoint": [
                        count / denominator
                        for count, denominator in zip(pooled_below_count, pooled_count, strict=True)
                    ],
                    "pooled_above_fraction_by_checkpoint": [
                        count / denominator
                        for count, denominator in zip(pooled_above_count, pooled_count, strict=True)
                    ],
                    "views": view_rows,
                }
            seed_rows.append({"seed": seed, "arms": arm_rows})
        block_metrics[block] = {"seeds": seed_rows}
    mechanism_d_auc = [
        auc_by_key[(0, seed_position, 1)] - auc_by_key[(0, seed_position, 0)]
        for seed_position in range(3)
    ]
    mechanism_d_final = [
        curve_by_key[(0, seed_position, 1)][-1] - curve_by_key[(0, seed_position, 0)][-1]
        for seed_position in range(3)
    ]
    mechanism_d_ssim = [
        ssim_curve_by_key[(0, seed_position, 1)][-1] - ssim_curve_by_key[(0, seed_position, 0)][-1]
        for seed_position in range(3)
    ]
    joint_d_auc = [
        auc_by_key[(1, seed_position, 1)] - auc_by_key[(1, seed_position, 0)]
        for seed_position in range(3)
    ]
    joint_d_final = [
        curve_by_key[(1, seed_position, 1)][-1] - curve_by_key[(1, seed_position, 0)][-1]
        for seed_position in range(3)
    ]
    joint_d_ssim = [
        ssim_curve_by_key[(1, seed_position, 1)][-1] - ssim_curve_by_key[(1, seed_position, 0)][-1]
        for seed_position in range(3)
    ]
    mechanism_current = tables["mechanism_current"]
    current_fit_seed = fit["seed_position"][mechanism_current["fit_index"]]
    null_by_seed = [
        _pool_null_arrays(mechanism_current, current_fit_seed == seed_position)
        for seed_position in range(3)
    ]
    null_global = _pool_null_arrays(
        mechanism_current, np.ones(mechanism_current["fit_index"].shape[0], dtype=np.bool_)
    )
    current_checkpoint = tables["checkpoint_current"]
    candidate_checkpoint = tables["checkpoint_candidate"]
    current_fit_indices = current_checkpoint["fit_index"]
    candidate_fit_indices = candidate_checkpoint["fit_index"]
    current_mechanism = fit["block_index"][current_fit_indices] == 0
    candidate_mechanism = fit["block_index"][candidate_fit_indices] == 0
    if int(current_mechanism.sum()) != 3 * 9 * 8 or int(candidate_mechanism.sum()) != 3 * 9 * 8:
        raise ProtocolInvalid("reduction", "weak checkpoint cell population differs")
    current_weak = current_checkpoint["weakly_responsive"][current_mechanism]
    candidate_weak = candidate_checkpoint["weakly_responsive"][candidate_mechanism]
    if (
        current_weak.size != SCIENTIFIC_PLAN.weak_rows_per_arm
        or candidate_weak.size != SCIENTIFIC_PLAN.weak_rows_per_arm
    ):
        raise ProtocolInvalid("reduction", "weak component population differs")
    current_weak_count = int(current_weak.sum())
    candidate_weak_count = int(candidate_weak.sum())
    weak_delta_global = float(
        candidate_weak_count / candidate_weak.size - current_weak_count / current_weak.size
    )
    weak_delta_by_seed: list[float] = []
    weak_counts_by_seed: list[dict[str, int]] = []
    for seed_position in range(3):
        current_seed_mask = current_mechanism & (
            fit["seed_position"][current_fit_indices] == seed_position
        )
        candidate_seed_mask = candidate_mechanism & (
            fit["seed_position"][candidate_fit_indices] == seed_position
        )
        current_seed = current_checkpoint["weakly_responsive"][current_seed_mask]
        candidate_seed = candidate_checkpoint["weakly_responsive"][candidate_seed_mask]
        if (
            current_seed.size != SCIENTIFIC_PLAN.weak_rows_per_seed_arm
            or candidate_seed.size != SCIENTIFIC_PLAN.weak_rows_per_seed_arm
        ):
            raise ProtocolInvalid("reduction", "per-seed weak population differs")
        current_seed_count = int(current_seed.sum())
        candidate_seed_count = int(candidate_seed.sum())
        denominator = int(current_seed.size)
        weak_counts_by_seed.append(
            {
                "seed": BLOCK_SEEDS["appearance_only"][seed_position],
                "current_count": current_seed_count,
                "candidate_count": candidate_seed_count,
                "denominator_per_arm": denominator,
            }
        )
        weak_delta_by_seed.append(
            float(candidate_seed_count / denominator - current_seed_count / denominator)
        )
    decision_summary = {
        "appearance_only": {
            "delta_auc_by_seed": mechanism_d_auc,
            "delta_final_psnr_by_seed": mechanism_d_final,
            "delta_final_ssim_by_seed": mechanism_d_ssim,
            "null_global": null_global,
            "null_by_seed": null_by_seed,
            "weak_counts_global": {
                "current_count": current_weak_count,
                "candidate_count": candidate_weak_count,
                "denominator_per_arm": int(current_weak.size),
            },
            "weak_counts_by_seed": weak_counts_by_seed,
            "weak_fraction_delta_global": weak_delta_global,
            "weak_fraction_delta_by_seed": weak_delta_by_seed,
        },
        "joint": {
            "delta_final_psnr_by_seed": joint_d_final,
            "delta_final_ssim_by_seed": joint_d_ssim,
            "delta_auc_by_seed": joint_d_auc,
        },
    }
    decisions = frozen_decisions(decision_summary, global_validity_passed=True)
    diagnostics = _checkpoint_diagnostic_reductions(tables)
    return {
        "metrics": block_metrics,
        "diagnostics": diagnostics,
        "decision_inputs": decision_summary,
        "decisions": decisions,
        "learned_parameter_reduction_per_component": 1,
        "learned_parameters_per_component": {ARMS[0]: 9, ARMS[1]: 8},
    }


def validate_and_recompute_scientific_raw(
    arrays: Mapping[str, np.ndarray], *, deep: bool = True
) -> dict[str, Any]:
    """Recompute the complete valid result without trusting stored JSON flags."""
    required = _scientific_required_names()
    actual = set(arrays)
    if actual != required:
        raise ProtocolInvalid(
            "reduction",
            "scientific raw name set differs",
            {"missing": sorted(required - actual), "extra": sorted(actual - required)},
        )
    expected_array_count = scientific_schema_record()["expected_valid_array_count"]
    if len(arrays) != expected_array_count:
        raise ProtocolInvalid("reduction", "scientific raw array count differs from schema")
    _validate_scientific_constants(arrays)
    _validate_scientific_completion(arrays)
    _validate_archive_completion(arrays)
    tables = {
        name: _raw_table(arrays, name, require_complete=True) for name in SCIENTIFIC_TABLE_SPECS
    }
    _validate_scientific_order(tables)
    _validate_fit_and_event_evidence(tables)
    _recompute_prerequisite(tables["prerequisite"], deep=deep)
    _recompute_checkpoints(tables, deep=deep)
    _recompute_mechanism(tables, deep=deep)
    _recompute_joint(tables, deep=deep)
    result = reduce_scientific_raw(tables)
    expected_counts = asdict(SCIENTIFIC_PLAN)
    result["plan"] = expected_counts
    result["raw_evidence_recomputed"] = True
    return result


def _invalid_prefix_table(
    arrays: Mapping[str, np.ndarray], name: str, consumed: set[str]
) -> dict[str, np.ndarray]:
    """Load the exact nonempty representation emitted by a flushed prefix table."""
    spec = SCIENTIFIC_TABLE_SPECS[name]
    prefix = f"scientific/{name}"
    count_name = f"{prefix}/row_count"
    table_names = {logical for logical in arrays if logical.startswith(f"{prefix}/")}
    if count_name not in arrays:
        if table_names:
            raise ProtocolInvalid("serialization", f"invalid table lacks row count: {name}")
        return {
            field_name: np.empty((0, *field_spec.shape), dtype=np.dtype(field_spec.dtype))
            for field_name, field_spec in spec.fields.items()
        }
    count = arrays[count_name]
    if count.dtype != np.dtype("<i8") or count.shape != ():
        raise ProtocolInvalid("serialization", f"invalid table row-count schema differs: {name}")
    rows = int(count)
    if rows <= 0 or rows > spec.rows:
        raise ProtocolInvalid("serialization", f"invalid table prefix row count differs: {name}")
    expected_names = {count_name} | {f"{prefix}/{field_name}" for field_name in spec.fields}
    if table_names != expected_names:
        raise ProtocolInvalid(
            "serialization",
            f"invalid table field set differs: {name}",
            {
                "missing": sorted(expected_names - table_names),
                "extra": sorted(table_names - expected_names),
            },
        )
    result: dict[str, np.ndarray] = {}
    for field_name, field_spec in spec.fields.items():
        logical_name = f"{prefix}/{field_name}"
        value = arrays[logical_name]
        if value.dtype != np.dtype(field_spec.dtype) or value.shape != (
            rows,
            *field_spec.shape,
        ):
            raise ProtocolInvalid(
                "serialization", f"invalid table prefix schema differs: {name}/{field_name}"
            )
        if np.issubdtype(value.dtype, np.floating) and not bool(np.isfinite(value).all()):
            raise ProtocolInvalid(
                "serialization",
                f"invalid scientific table field is non-finite: {name}/{field_name}",
            )
        result[field_name] = value
    consumed.update(expected_names)
    return result


def _expected_prerequisite_rows() -> list[tuple[int, int, int, int, int, int]]:
    return [
        (block_index, seed_position, seed, local_view, original_view, seed + local_view)
        for block_index, block in enumerate(BLOCKS)
        for seed_position, seed in enumerate(BLOCK_SEEDS[block])
        for local_view, original_view in enumerate(SELECTED_VIEWS)
    ]


def _expected_fit_event_rows() -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for fit_index in range(SCIENTIFIC_PLAN.fits):
        rows.append((fit_index, EVENT_CODES["initial"], 0))
        for completed_step in range(UPDATES):
            rows.append((fit_index, EVENT_CODES["pre_update"], completed_step))
            rows.append((fit_index, EVENT_CODES["post_update"], completed_step + 1))
            if completed_step + 1 in POSITIVE_CHECKPOINTS:
                rows.append((fit_index, EVENT_CODES["checkpoint"], completed_step + 1))
    return rows


def _expected_checkpoint_rows() -> list[tuple[int, int, int]]:
    return [
        (checkpoint_index, fit_index, step)
        for checkpoint_index, (fit_index, step) in enumerate(
            (fit_index, step) for fit_index in range(SCIENTIFIC_PLAN.fits) for step in CHECKPOINTS
        )
    ]


def _expected_checkpoint_arm_rows(arm_code: int) -> list[tuple[int, int, int]]:
    fit_rows = _expected_fit_rows()
    return [row for row in _expected_checkpoint_rows() if fit_rows[row[1]][6] == arm_code]


def _expected_mechanism_geometry_rows() -> list[tuple[int, int]]:
    return [
        (fit_index, step)
        for fit_index, fit_row in enumerate(_expected_fit_rows())
        if fit_row[1] == 0
        for step in range(UPDATES + 1)
    ]


def _expected_mechanism_update_rows(arm_code: int) -> list[tuple[int, int, int]]:
    return [
        (fit_index, completed_step, completed_step + 1)
        for fit_index, fit_row in enumerate(_expected_fit_rows())
        if fit_row[1] == 0 and fit_row[6] == arm_code
        for completed_step in range(UPDATES)
    ]


def _expected_joint_rows(arm_code: int) -> list[tuple[int, int, bool]]:
    return [
        (fit_index, step, step > 0)
        for fit_index, fit_row in enumerate(_expected_fit_rows())
        if fit_row[1] == 1 and fit_row[6] == arm_code
        for step in CHECKPOINTS
    ]


def _assert_identity_leading_prefix(
    table: Mapping[str, np.ndarray],
    fields: tuple[str, ...],
    expected_rows: list[tuple[Any, ...]],
    label: str,
) -> None:
    actual_rows = list(zip(*(table[field].tolist() for field in fields), strict=True))
    if actual_rows != expected_rows[: len(actual_rows)]:
        raise ProtocolInvalid("serialization", f"invalid table is not a canonical prefix: {label}")


def _invalid_scalar_int(arrays: Mapping[str, np.ndarray], name: str) -> int:
    value = arrays.get(name)
    if value is None or value.dtype != np.dtype("<i8") or value.shape != ():
        raise ProtocolInvalid("serialization", f"invalid completion scalar differs: {name}")
    return int(value)


def _validate_invalid_failure_names(
    arrays: Mapping[str, np.ndarray],
    consumed: set[str],
    tables: Mapping[str, Mapping[str, np.ndarray]],
    *,
    expected_failure: Mapping[str, Any] | None,
) -> None:
    parameter_shapes = {
        "xy_raw": (COMPONENTS, 2),
        "diag_raw": (COMPONENTS, 2),
        "off_raw": (COMPONENTS,),
        "color_raw": (COMPONENTS, 3),
        "weight_raw": (COMPONENTS,),
        "amplitude_raw": (COMPONENTS, 3),
    }
    classification_suffix = "/nonfinite_classification"
    parameter_names = {
        "xy_raw",
        "diag_raw",
        "off_raw",
        "color_raw",
        "weight_raw",
        "amplitude_raw",
    }
    snapshot_suffixes = {
        "step",
        "target",
        "built_xy",
        "built_chol",
        "built_color",
        "built_weight",
        "next_lr",
        "lr_used",
        "render",
        "loss",
        "initial_xy",
        "initial_chol",
        "initial_color",
        "initial_weight",
    }
    snapshot_suffixes.update(f"raw_{name}" for name in parameter_names)
    snapshot_suffixes.update(f"gradient_{name}" for name in parameter_names)
    snapshot_suffixes.update(
        f"optimizer_{name}_{state_name}"
        for name in parameter_names
        for state_name in ("step", "exp_avg", "exp_avg_sq")
    )
    step_zero_fields = {
        "target",
        "current_raw_xy_raw",
        "current_raw_diag_raw",
        "current_raw_off_raw",
        "current_raw_u",
        "current_raw_s",
        "candidate_raw_xy_raw",
        "candidate_raw_diag_raw",
        "candidate_raw_off_raw",
        "candidate_raw_r",
        "current_built_xy",
        "current_built_chol",
        "current_built_color",
        "current_built_weight",
        "candidate_built_xy",
        "candidate_built_chol",
        "candidate_built_color",
        "candidate_built_weight",
        "common_amplitude",
        "current_render",
        "candidate_render",
        "current_loss",
        "candidate_loss",
        "gate_values",
    }
    gate_fields = {
        "actual",
        "expected",
        "render",
        "prediction",
        "target",
        "optimization_render",
        "optimization_loss",
        "probe_render",
        "probe_loss",
        "probe_gradient",
        "pre_render",
        "pre_loss",
        "pre_built_xy",
        "pre_built_chol",
        "pre_built_color",
        "pre_built_weight",
        "checkpoint_render",
        "checkpoint_loss",
        "checkpoint_built_xy",
        "checkpoint_built_chol",
        "checkpoint_built_color",
        "checkpoint_built_weight",
    }
    gate_fields.update(f"gradient_{name}" for name in parameter_names)
    gate_fields.update(
        f"optimizer_{name}_{state_name}"
        for name in parameter_names
        for state_name in ("exp_avg", "exp_avg_sq")
    )

    def has_preserved_nonfinite(base: str) -> bool:
        value = arrays[base]
        return (
            np.issubdtype(value.dtype, np.floating)
            and bool((~np.isfinite(value)).any())
            and f"{base}{classification_suffix}" in arrays
        )

    def valid_protocol_failure(base: str, field_name: str) -> bool:
        value = arrays[base]
        finite = not bool(nonfinite_classification(value).any())
        classification_present = f"{base}{classification_suffix}" in arrays
        field_allowed = (
            field_name in {"actual", "expected", "target"}
            or field_name in step_zero_fields
            or field_name in {"g0_xy", "g0_chol", "g0_color", "g0_weight"}
            or (
                field_name.startswith("snapshot_")
                and field_name.removeprefix("snapshot_") in snapshot_suffixes
            )
            or (
                field_name.startswith("pending_pre_")
                and field_name.removeprefix("pending_pre_") in snapshot_suffixes
            )
            or (field_name.startswith("gate_") and field_name.removeprefix("gate_") in gate_fields)
        )
        return (
            field_allowed
            and value.dtype.kind in "biuf"
            and value.ndim <= 4
            and value.size <= 1_000_000
            and value.nbytes <= 8_000_000
            and (finite or classification_present)
        )

    for name in sorted(set(arrays) - consumed):
        if name.endswith(classification_suffix):
            base = name.removesuffix(classification_suffix)
            expected_classification = (
                nonfinite_classification(arrays[base]) if base in arrays else None
            )
            if (
                base not in arrays
                or base in consumed
                or not base.startswith("failure/")
                or expected_classification is None
                or not _arrays_byte_exact(arrays[name], expected_classification)
                or not bool(expected_classification.any())
            ):
                raise ProtocolInvalid("serialization", "orphan invalid failure classification")
            continue
        parts = name.split("/")
        valid = False
        if len(parts) == 3 and parts[:2] == ["failure", "protocol"]:
            valid = re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", parts[2]
            ) is not None and valid_protocol_failure(name, parts[2])
        elif len(parts) == 3 and parts[:2] == ["failure", "raw_parameter"]:
            shape = parameter_shapes.get(parts[2])
            valid = (
                shape is not None
                and arrays[name].dtype == np.dtype("<f4")
                and arrays[name].shape == shape
                and has_preserved_nonfinite(name)
            )
        elif len(parts) == 4 and parts[0] == "failure" and parts[1] in SCIENTIFIC_TABLE_SPECS:
            table_name, field_name, row_text = parts[1:]
            field_spec = SCIENTIFIC_TABLE_SPECS[table_name].fields.get(field_name)
            valid = (
                field_spec is not None
                and re.fullmatch(r"\d{6}", row_text) is not None
                and int(row_text) == len(tables[table_name][next(iter(tables[table_name]))])
                and arrays[name].dtype == np.dtype(field_spec.dtype)
                and arrays[name].shape == field_spec.shape
                and has_preserved_nonfinite(name)
            )
        if not valid:
            raise ProtocolInvalid("serialization", f"unexpected invalid raw name: {name}")

    failure_names = sorted(name for name in set(arrays) - consumed if name.startswith("failure/"))
    protocol_fields = {
        name.split("/", maxsplit=2)[2]
        for name in failure_names
        if name.startswith("failure/protocol/") and not name.endswith(classification_suffix)
    }
    for prefix in ("snapshot", "pending_pre"):
        present = {
            name.removeprefix(f"{prefix}_")
            for name in protocol_fields
            if name.startswith(f"{prefix}_")
        }
        if present and not {
            "step",
            "target",
            "built_xy",
            "built_chol",
            "built_color",
            "built_weight",
            "next_lr",
        }.issubset(present):
            raise ProtocolInvalid(
                "serialization", f"invalid {prefix} failure evidence is incomplete"
            )
    if protocol_fields & {"g0_xy", "g0_chol", "g0_color", "g0_weight"} and not {
        "target",
        "g0_xy",
        "g0_chol",
        "g0_color",
        "g0_weight",
    }.issubset(protocol_fields):
        raise ProtocolInvalid("serialization", "initializer failure evidence is incomplete")
    if bool(protocol_fields & {"actual", "expected"}) and not {
        "actual",
        "expected",
    }.issubset(protocol_fields):
        raise ProtocolInvalid("serialization", "paired failure evidence is incomplete")
    gate_evidence = {name for name in protocol_fields if name.startswith("gate_")}
    if gate_evidence and not any(name.startswith("snapshot_") for name in protocol_fields):
        raise ProtocolInvalid("serialization", "gate failure evidence lacks its snapshot core")
    if bool(gate_evidence & {"gate_actual", "gate_expected"}) and not {
        "gate_actual",
        "gate_expected",
    }.issubset(gate_evidence):
        raise ProtocolInvalid("serialization", "paired gate evidence is incomplete")
    if failure_names:
        if not isinstance(expected_failure, Mapping):
            raise ProtocolInvalid("serialization", "invalid failure evidence lacks metadata")
        receipt = expected_failure.get("raw_evidence")
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "names",
            "manifest",
            "collection_sha256",
        }:
            raise ProtocolInvalid("serialization", "invalid failure receipt schema differs")
        evidence_arrays = {name: arrays[name] for name in failure_names}
        manifest, collection_sha256 = array_manifest(evidence_arrays)
        if (
            receipt.get("names") != failure_names
            or receipt.get("manifest") != manifest
            or receipt.get("collection_sha256") != collection_sha256
        ):
            raise ProtocolInvalid("serialization", "invalid failure receipt binding differs")
    elif isinstance(expected_failure, Mapping):
        receipt = expected_failure.get("raw_evidence")
        if not isinstance(receipt, Mapping) or receipt.get("names") != []:
            raise ProtocolInvalid("serialization", "empty failure receipt differs")


def validate_invalid_scientific_prefix(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_phase: str,
    expected_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate that a failed attempt is one exact, reachable scientific prefix."""
    phase_code = _phase_code(expected_phase)
    if phase_code < 0 or expected_phase in {"complete", "invalid", "metrics"}:
        raise ProtocolInvalid("serialization", "invalid failure phase is not reachable")
    schema = scientific_schema_record()
    consumed = set(schema["constant_arrays"])
    consumed.update(schema["completion_arrays"])
    consumed.update(schema["archive_completion_arrays"])
    missing_fixed = consumed - set(arrays)
    if missing_fixed:
        raise ProtocolInvalid(
            "serialization",
            "invalid archive lacks fixed evidence",
            {"missing": sorted(missing_fixed)},
        )
    _validate_scientific_constants(arrays)
    tables = {
        name: _invalid_prefix_table(arrays, name, consumed) for name in SCIENTIFIC_TABLE_SPECS
    }

    prefix_specs = {
        "prerequisite": (
            (
                "block_index",
                "seed_position",
                "seed",
                "local_view",
                "original_view",
                "initializer_seed",
            ),
            _expected_prerequisite_rows(),
        ),
        "fit": (
            (
                "fit_index",
                "block_index",
                "seed_position",
                "seed",
                "local_view",
                "original_view",
                "arm_code",
                "arm_order_position",
            ),
            _expected_fit_rows(),
        ),
        "fit_event": (("fit_index", "event_code", "step"), _expected_fit_event_rows()),
        "checkpoint": (("checkpoint_index", "fit_index", "step"), _expected_checkpoint_rows()),
        "checkpoint_current": (
            ("checkpoint_index", "fit_index", "step"),
            _expected_checkpoint_arm_rows(0),
        ),
        "checkpoint_candidate": (
            ("checkpoint_index", "fit_index", "step"),
            _expected_checkpoint_arm_rows(1),
        ),
        "mechanism_geometry": (
            ("fit_index", "step"),
            _expected_mechanism_geometry_rows(),
        ),
        "mechanism_current": (
            ("fit_index", "completed_step", "update_number"),
            _expected_mechanism_update_rows(0),
        ),
        "mechanism_candidate": (
            ("fit_index", "completed_step", "update_number"),
            _expected_mechanism_update_rows(1),
        ),
        "joint_current": (("fit_index", "step", "defined"), _expected_joint_rows(0)),
        "joint_candidate": (("fit_index", "step", "defined"), _expected_joint_rows(1)),
    }
    for name, (fields, expected) in prefix_specs.items():
        _assert_identity_leading_prefix(tables[name], fields, expected, name)

    counts = {name: len(next(iter(table.values()))) for name, table in tables.items()}
    q, f, e, c = (counts[name] for name in ("prerequisite", "fit", "fit_event", "checkpoint"))
    detail_count = counts["checkpoint_current"] + counts["checkpoint_candidate"]
    post_events = int((tables["fit_event"]["event_code"] == EVENT_CODES["post_update"]).sum())
    event_rows = list(
        zip(
            tables["fit_event"]["fit_index"].tolist(),
            tables["fit_event"]["event_code"].tolist(),
            tables["fit_event"]["step"].tolist(),
            strict=True,
        )
    )
    fit_plan = _expected_fit_rows()

    def trigger_count(predicate: Callable[[int, int, int], bool]) -> tuple[int, bool]:
        count = sum(bool(predicate(*row)) for row in event_rows)
        return count, bool(event_rows and predicate(*event_rows[-1]))

    def require_trigger_prefix(
        label: str, actual: int, predicate: Callable[[int, int, int], bool]
    ) -> None:
        triggered, final_is_trigger = trigger_count(predicate)
        allowed = {triggered}
        if final_is_trigger:
            allowed.add(triggered - 1)
        if actual not in allowed:
            raise ProtocolInvalid("serialization", f"invalid {label} trigger prefix is unreachable")

    def checkpoint_trigger(_fit: int, event_code: int, _step: int) -> bool:
        return event_code in {
            EVENT_CODES["initial"],
            EVENT_CODES["checkpoint"],
        }

    if not (248 * f <= e <= min(248 * (f + 1), SCIENTIFIC_PLAN.callback_events)):
        raise ProtocolInvalid("serialization", "invalid fit/event prefix is unreachable")
    require_trigger_prefix("checkpoint/event", c, checkpoint_trigger)
    require_trigger_prefix("checkpoint-detail/event", detail_count, checkpoint_trigger)
    detail_rows = sorted(
        [
            (
                int(table["checkpoint_index"][row]),
                int(table["fit_index"][row]),
                int(table["step"][row]),
            )
            for table_name in ("checkpoint_current", "checkpoint_candidate")
            for table in (tables[table_name],)
            for row in range(counts[table_name])
        ],
        key=lambda row: row[0],
    )
    if detail_rows != _expected_checkpoint_rows()[:detail_count]:
        raise ProtocolInvalid(
            "serialization", "invalid checkpoint-detail union is not a canonical prefix"
        )

    def mechanism_geometry_trigger(fit_index: int, event_code: int, _step: int) -> bool:
        return fit_plan[fit_index][1] == 0 and event_code in {
            EVENT_CODES["initial"],
            EVENT_CODES["post_update"],
        }

    require_trigger_prefix(
        "mechanism-geometry/event",
        counts["mechanism_geometry"],
        mechanism_geometry_trigger,
    )
    for arm_code, table_name in enumerate(("mechanism_current", "mechanism_candidate")):
        require_trigger_prefix(
            f"{table_name}/event",
            counts[table_name],
            lambda fit_index, event_code, _step, arm_code=arm_code: (
                fit_plan[fit_index][1] == 0
                and fit_plan[fit_index][6] == arm_code
                and event_code == EVENT_CODES["post_update"]
            ),
        )
    for arm_code, table_name in enumerate(("joint_current", "joint_candidate")):
        require_trigger_prefix(
            f"{table_name}/event",
            counts[table_name],
            lambda fit_index, event_code, step, arm_code=arm_code: (
                fit_plan[fit_index][1] == 1
                and fit_plan[fit_index][6] == arm_code
                and (
                    event_code == EVENT_CODES["initial"]
                    or (event_code == EVENT_CODES["post_update"] and step in POSITIVE_CHECKPOINTS)
                )
            ),
        )
    for arm_code, table_name in enumerate(("checkpoint_current", "checkpoint_candidate")):
        require_trigger_prefix(
            f"{table_name}/event",
            counts[table_name],
            lambda fit_index, event_code, _step, arm_code=arm_code: (
                fit_plan[fit_index][6] == arm_code
                and event_code in {EVENT_CODES["initial"], EVENT_CODES["checkpoint"]}
            ),
        )

    dependent_counts = {
        "checkpoint": c,
        "checkpoint_current": counts["checkpoint_current"],
        "checkpoint_candidate": counts["checkpoint_candidate"],
        "mechanism_geometry": counts["mechanism_geometry"],
        "mechanism_current": counts["mechanism_current"],
        "mechanism_candidate": counts["mechanism_candidate"],
        "joint_current": counts["joint_current"],
        "joint_candidate": counts["joint_candidate"],
    }

    def emitted_operations(event: tuple[int, int, int]) -> list[str]:
        fit_index, event_code, step = event
        block_index = fit_plan[fit_index][1]
        arm_code = fit_plan[fit_index][6]
        arm_suffix = "current" if arm_code == 0 else "candidate"
        if event_code == EVENT_CODES["initial"]:
            first = "mechanism_geometry" if block_index == 0 else f"joint_{arm_suffix}"
            return [first, "checkpoint", f"checkpoint_{arm_suffix}"]
        if event_code == EVENT_CODES["post_update"]:
            if block_index == 0:
                return [f"mechanism_{arm_suffix}", "mechanism_geometry"]
            if step in POSITIVE_CHECKPOINTS:
                return [f"joint_{arm_suffix}"]
            return []
        if event_code == EVENT_CODES["checkpoint"]:
            return ["checkpoint", f"checkpoint_{arm_suffix}"]
        return []

    prior_operation_counts = {name: 0 for name in dependent_counts}
    for event in event_rows[:-1]:
        for operation in emitted_operations(event):
            prior_operation_counts[operation] += 1
    final_operations = emitted_operations(event_rows[-1]) if event_rows else []
    final_operation_set = set(final_operations)
    for name, actual in dependent_counts.items():
        prior = prior_operation_counts[name]
        allowed = {prior, prior + 1} if name in final_operation_set else {prior}
        if actual not in allowed:
            raise ProtocolInvalid(
                "serialization", f"invalid dependent-operation prefix differs: {name}"
            )
    included = [
        dependent_counts[name] == prior_operation_counts[name] + 1 for name in final_operations
    ]
    if included != sorted(included, reverse=True):
        raise ProtocolInvalid("serialization", "invalid final callback operation order differs")
    partial_event_allowed = e > 248 * f
    if not partial_event_allowed and not all(included):
        raise ProtocolInvalid(
            "serialization", "completed fit has an incomplete final callback operation"
        )

    downstream = sum(counts[name] for name in counts if name != "prerequisite")
    if q < SCIENTIFIC_PLAN.initializers and downstream:
        raise ProtocolInvalid("serialization", "optimization evidence precedes global prerequisite")
    if expected_phase == "preflight" and q:
        raise ProtocolInvalid("serialization", "preflight phase contains initializer evidence")
    if expected_phase in {"preflight", "scene", "initialization", "equivalence"} and downstream:
        raise ProtocolInvalid("serialization", "invalid phase contains downstream evidence")
    if expected_phase == "scene" and q % len(SELECTED_VIEWS) != 0:
        raise ProtocolInvalid("serialization", "scene-phase initializer prefix is unreachable")
    if expected_phase in {"scene", "initialization"} and q >= SCIENTIFIC_PLAN.initializers:
        raise ProtocolInvalid("serialization", "initializer phase prefix is already complete")
    if expected_phase == "appearance_only" and (
        f > SCIENTIFIC_PLAN.fits // 2
        or e > SCIENTIFIC_PLAN.callback_events // 2
        or counts["joint_current"]
        or counts["joint_candidate"]
    ):
        raise ProtocolInvalid("serialization", "appearance-only phase contains joint evidence")
    if expected_phase == "joint":
        if q != SCIENTIFIC_PLAN.initializers or f < SCIENTIFIC_PLAN.fits // 2:
            raise ProtocolInvalid("serialization", "joint phase precedes mechanism completion")
        for name in ("mechanism_current", "mechanism_candidate"):
            if counts[name] != SCIENTIFIC_PLAN.mechanism_updates_per_arm:
                raise ProtocolInvalid(
                    "serialization", "joint phase lacks complete mechanism evidence"
                )
    if expected_phase in {"reduction", "serialization"} and any(
        counts[name] != spec.rows for name, spec in SCIENTIFIC_TABLE_SPECS.items()
    ):
        raise ProtocolInvalid("serialization", "post-fit invalid phase lacks complete tables")

    scientific_names = [
        "callback_events",
        "checkpoints",
        "fits",
        "initializers",
        "joint_candidate_positive_checkpoints",
        "joint_current_positive_checkpoints",
        "mechanism_candidate_updates",
        "mechanism_current_updates",
        "mechanism_geometry_states",
        "optimizer_updates",
        "scenes",
    ]
    name_bytes, name_lengths = _utf8_rows(scientific_names, 64)
    if not _arrays_byte_exact(
        arrays["scientific_completion/count_name_bytes"], name_bytes
    ) or not _arrays_byte_exact(arrays["scientific_completion/count_name_lengths"], name_lengths):
        raise ProtocolInvalid("serialization", "invalid scientific completion names differ")
    values = arrays["scientific_completion/count_values"]
    if values.dtype != np.dtype("<i8") or values.shape != (len(scientific_names),):
        raise ProtocolInvalid("serialization", "invalid scientific completion values differ")
    scientific = dict(zip(scientific_names, values.tolist(), strict=True))
    if not _arrays_byte_exact(
        arrays["scientific_completion/phase_code"], np.asarray(phase_code, dtype=np.int64)
    ) or not _arrays_byte_exact(
        arrays["completion/phase_code"], np.asarray(phase_code, dtype=np.int64)
    ):
        raise ProtocolInvalid("serialization", "invalid completion phase differs")
    expected_exact = {
        "initializers": q,
        "fits": f,
        "callback_events": e,
        "checkpoints": detail_count,
        "mechanism_current_updates": counts["mechanism_current"],
        "mechanism_candidate_updates": counts["mechanism_candidate"],
        "mechanism_geometry_states": counts["mechanism_geometry"],
        "joint_current_positive_checkpoints": int(tables["joint_current"]["defined"].sum()),
        "joint_candidate_positive_checkpoints": int(tables["joint_candidate"]["defined"].sum()),
    }
    if any(scientific[name] != value for name, value in expected_exact.items()):
        raise ProtocolInvalid("serialization", "invalid scientific completion count differs")
    optimizer_updates = scientific["optimizer_updates"]
    final_event_is_post = (
        partial_event_allowed
        and e > 0
        and int(tables["fit_event"]["event_code"][-1]) == EVENT_CODES["post_update"]
    )
    optimizer_update_counts = {post_events}
    if final_event_is_post:
        optimizer_update_counts = {post_events - 1}
        if all(included):
            optimizer_update_counts.add(post_events)
    if optimizer_updates not in optimizer_update_counts:
        raise ProtocolInvalid("serialization", "invalid optimizer-update completion count differs")
    scenes = scientific["scenes"]
    minimum_scenes = math.ceil(q / len(SELECTED_VIEWS))
    maximum_scenes = min(SCIENTIFIC_PLAN.scenes, q // len(SELECTED_VIEWS) + 1)
    if not minimum_scenes <= scenes <= maximum_scenes or (
        q == SCIENTIFIC_PLAN.initializers and scenes != SCIENTIFIC_PLAN.scenes
    ):
        raise ProtocolInvalid("serialization", "invalid scene completion count differs")
    if expected_phase == "preflight":
        phase_scene_count = 0
    elif expected_phase == "scene":
        phase_scene_count = q // len(SELECTED_VIEWS)
    elif expected_phase in {"initialization", "equivalence"}:
        phase_scene_count = min(SCIENTIFIC_PLAN.scenes, q // len(SELECTED_VIEWS) + 1)
    else:
        phase_scene_count = SCIENTIFIC_PLAN.scenes
    if scenes != phase_scene_count:
        raise ProtocolInvalid("serialization", "invalid scene count differs for failure phase")

    archive_expected = {
        "completed_blocks": f // 54,
        "completed_seeds": f // 18,
        "completed_views": f // 2,
        "completed_initializers": q,
        "completed_paired_views": q,
        "completed_fits": f,
        "completed_updates": optimizer_updates,
        "completed_checkpoints": detail_count,
        "completed_callback_events": e,
        "completed_mechanism_updates": counts["mechanism_current"] + counts["mechanism_candidate"],
        "completed_joint_positive_checkpoint_updates": expected_exact[
            "joint_current_positive_checkpoints"
        ]
        + expected_exact["joint_candidate_positive_checkpoints"],
        "completed_checkpoint_component_rows": detail_count * COMPONENTS,
        "completed_null_rows": counts["mechanism_current"] * COMPONENTS,
    }
    for suffix, expected_value in archive_expected.items():
        if _invalid_scalar_int(arrays, f"completion/{suffix}") != expected_value:
            raise ProtocolInvalid(
                "serialization", f"invalid archive completion count differs: {suffix}"
            )
    _validate_invalid_failure_names(arrays, consumed, tables, expected_failure=expected_failure)
    return {
        "phase": expected_phase,
        "table_row_counts": counts,
        "scientific_completion": scientific,
        "scientific_decisions_present": False,
    }


def _replay_source_target_rows(
    prerequisite: Mapping[str, np.ndarray],
    expected_rows: list[tuple[int, int, int, int, int, int]],
    *,
    scene_factory: Callable[..., Any],
) -> dict[str, Any]:
    """Reviewer-only source replay core; never part of the once-only scientific process."""
    actual_rows = list(
        zip(
            prerequisite["block_index"].tolist(),
            prerequisite["seed_position"].tolist(),
            prerequisite["seed"].tolist(),
            prerequisite["local_view"].tolist(),
            prerequisite["original_view"].tolist(),
            prerequisite["initializer_seed"].tolist(),
            strict=True,
        )
    )
    if actual_rows != expected_rows:
        raise ProtocolInvalid("source_replay", "source target identity rows differ")
    targets = prerequisite["target"]
    target_hashes = prerequisite["target_sha256"]
    if targets.dtype != np.dtype("<f4") or targets.shape != (
        len(expected_rows),
        IMAGE_SIZE,
        IMAGE_SIZE,
        3,
    ):
        raise ProtocolInvalid("source_replay", "source target schema differs")
    if target_hashes.dtype != np.dtype("|u1") or target_hashes.shape != (
        len(expected_rows),
        64,
    ):
        raise ProtocolInvalid("source_replay", "source target hash schema differs")
    for row in range(len(expected_rows)):
        expected_hash = _sha256_bytes(array_content_sha256(targets[row]))
        if not _arrays_byte_exact(target_hashes[row], expected_hash):
            raise ProtocolInvalid("source_replay", "stored source target hash differs")

    grouped: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for row, (block_index, seed_position, seed, local_view, original_view, _init) in enumerate(
        expected_rows
    ):
        grouped.setdefault((block_index, seed_position, seed), []).append(
            (row, local_view, original_view)
        )
    rng_before = torch.random.get_rng_state().clone()
    digest_rows: list[list[Any]] = []
    scene_call_count = 0
    for (block_index, seed_position, seed), rows in grouped.items():
        scene = scene_factory(
            n_gaussians=40,
            n_cameras=12,
            image_size=IMAGE_SIZE,
            seed=seed,
        )
        scene_call_count += 1
        if not hasattr(scene, "images") or len(scene.images) != 12:
            raise ProtocolInvalid("source_replay", "replayed source scene view count differs")
        for row, local_view, original_view in rows:
            regenerated = scene.images[original_view].detach().contiguous().cpu()
            stored = torch.from_numpy(np.ascontiguousarray(targets[row]))
            if (
                regenerated.dtype != torch.float32
                or regenerated.shape != (IMAGE_SIZE, IMAGE_SIZE, 3)
                or not torch.equal(regenerated, stored)
            ):
                raise ProtocolInvalid(
                    "source_replay",
                    "replayed source target differs",
                    {
                        "block_index": block_index,
                        "seed_position": seed_position,
                        "seed": seed,
                        "local_view": local_view,
                        "original_view": original_view,
                    },
                )
            digest = _tensor_sha256(regenerated)
            if not _arrays_byte_exact(target_hashes[row], _sha256_bytes(digest)):
                raise ProtocolInvalid("source_replay", "replayed source target hash differs")
            digest_rows.append(
                [block_index, seed_position, seed, local_view, original_view, digest]
            )
        del scene
    rng_unchanged = torch.equal(torch.random.get_rng_state(), rng_before)
    if not rng_unchanged:
        raise ProtocolInvalid("source_replay", "source target replay changed global CPU RNG")
    return {
        "passed": True,
        "reviewer_only": True,
        "scene_call_count": scene_call_count,
        "target_count": len(expected_rows),
        "selected_views": list(SELECTED_VIEWS),
        "target_collection_sha256": canonical_json_hash(digest_rows),
        "torch_rng_unchanged": True,
    }


def reviewer_replay_source_targets(
    arrays: Mapping[str, np.ndarray], *, expected_raw_collection_sha256: str
) -> dict[str, Any]:
    """Replay all six target scenes once in a separate, post-result reviewer process."""
    if set(arrays) != _scientific_required_names():
        raise ProtocolInvalid(
            "source_replay", "reviewer source replay requires exact valid raw names"
        )
    _manifest, raw_collection = array_manifest(arrays)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_raw_collection_sha256)
        or raw_collection != expected_raw_collection_sha256
    ):
        raise ProtocolInvalid("source_replay", "reviewer source replay raw binding differs")
    _validate_scientific_constants(arrays)
    _validate_scientific_completion(arrays)
    _validate_archive_completion(arrays)
    tables = {
        name: _raw_table(arrays, name, require_complete=True) for name in SCIENTIFIC_TABLE_SPECS
    }
    _validate_scientific_order(tables)
    environment = environment_metadata()
    assert_official_environment(environment)
    source_before = source_snapshot(TARGET_GENERATOR_SOURCE_PATHS)
    receipt = _replay_source_target_rows(
        tables["prerequisite"],
        _expected_prerequisite_rows(),
        scene_factory=make_synthetic_scene,
    )
    source_after = source_snapshot(TARGET_GENERATOR_SOURCE_PATHS)
    environment_after = environment_metadata()
    if canonical_json(source_before) != canonical_json(source_after):
        raise ProtocolInvalid("source_replay", "target generator sources changed during replay")
    if canonical_json(official_environment_fingerprint(environment)) != canonical_json(
        official_environment_fingerprint(environment_after)
    ):
        raise ProtocolInvalid("source_replay", "reviewer environment changed during replay")
    receipt["raw_collection_sha256"] = raw_collection
    receipt["target_generator_source_collection_sha256"] = source_before["collection_sha256"]
    receipt["environment_fingerprint_sha256"] = canonical_json_hash(
        official_environment_fingerprint(environment)
    )
    return receipt


def _scientific_result_note(
    *,
    valid: bool,
    result_path: Path,
    result_sha256: str,
    raw_binding: Mapping[str, Any],
    recomputed: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None,
) -> bytes:
    lines = [
        "# Stage-1 fit-time appearance-parameterization result",
        "",
        f"- Status: {'valid' if valid else 'invalid'}",
        f"- Scientific JSON: `{result_path}`",
        f"- Scientific JSON SHA-256: `{result_sha256}`",
        f"- Raw sidecar: `{raw_binding['path']}`",
        f"- Raw sidecar SHA-256: `{raw_binding['sha256']}`",
        f"- Raw collection SHA-256: `{raw_binding['collection_sha256']}`",
        f"- Raw arrays: `{raw_binding['array_count']}`",
    ]
    if valid and recomputed is not None:
        decisions = recomputed["decisions"]
        lines.extend(
            [
                "",
                "## Frozen decisions",
                "",
                *[f"- `{name}`: `{str(bool(value)).lower()}`" for name, value in decisions.items()],
                "",
                "All values above were recomputed from the reloaded raw archive. This CPU",
                "synthetic Stage-1 result authorizes no default or downstream change.",
            ]
        )
    elif failure is not None:
        lines.extend(
            [
                "",
                "## Failure boundary",
                "",
                f"- Phase: `{failure['phase']}`",
                f"- Reason: {failure['reason']}",
                "",
                "No scientific decision or quantitative claim is authorized.",
            ]
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_scientific_outcome(
    *,
    valid: bool,
    paths: Mapping[str, Path],
    archive: RawArchive,
    seal: Mapping[str, Any],
    attempt: Mapping[str, Any],
    source_before: Mapping[str, Any],
    recomputed: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    label = "valid" if valid else "invalid"
    raw_path = paths[f"{label}_raw"]
    json_path = paths[f"{label}_json"]
    note_path = paths[f"{label}_note"]
    alternate = "invalid" if valid else "valid"
    if any(paths[f"{alternate}_{suffix}"].exists() for suffix in ("json", "raw", "note")):
        raise RuntimeError("mutually exclusive scientific artifact route is occupied")
    validate_attempt_binding(attempt)
    source_after = loaded_repository_source_snapshot()
    source_unchanged = canonical_json(source_after) == canonical_json(source_before)
    current_environment = environment_metadata()
    sealed_environment = official_environment_fingerprint(seal["payload"]["environment"])
    runtime_environment = official_environment_fingerprint(current_environment)
    environment_unchanged = canonical_json(sealed_environment) == canonical_json(
        runtime_environment
    )
    try:
        assert_official_environment(current_environment)
    except RuntimeError:
        environment_unchanged = False
    if valid and not source_unchanged:
        raise RuntimeError("runtime loaded-source snapshot changed during scientific execution")
    if valid and not environment_unchanged:
        raise RuntimeError("runtime environment changed after scientific execution")
    in_memory_prefix_validation: dict[str, Any] | None = None
    if not valid:
        if failure is None:
            raise RuntimeError("invalid outcome lacks failure metadata")
        _validate_nonfinite_contract(archive.arrays, invalid_evidence=True)
        in_memory_prefix_validation = validate_invalid_scientific_prefix(
            archive.arrays,
            expected_phase=str(failure["phase"]),
            expected_failure=failure,
        )
    raw_binding = write_raw_sidecar(raw_path, archive.arrays, invalid_evidence=not valid)
    loaded = validate_raw_sidecar(raw_path, raw_binding, invalid_evidence=not valid)
    if valid:
        loaded_recomputed = validate_and_recompute_scientific_raw(loaded, deep=True)
        if recomputed is None or canonical_json_hash(loaded_recomputed) != canonical_json_hash(
            recomputed
        ):
            raise RuntimeError("reloaded raw scientific reduction differs from in-memory reduction")
        prefix_validation = None
    else:
        loaded_recomputed = None
        prefix_validation = validate_invalid_scientific_prefix(
            loaded,
            expected_phase=str(failure["phase"]),
            expected_failure=failure,
        )
        if canonical_json(prefix_validation) != canonical_json(in_memory_prefix_validation):
            raise RuntimeError("reloaded invalid-prefix validation differs from memory")
    validate_attempt_binding(attempt)
    created_at = _utc_now().replace(microsecond=0)
    attempt_claimed_at = _parse_artifact_utc(
        attempt["payload"]["claimed_at_utc"], label="result attempt claim"
    )
    seal_created_at = _parse_artifact_utc(
        seal["payload"]["created_at_utc"], label="result seal creation"
    )
    if created_at < attempt_claimed_at or attempt_claimed_at < seal_created_at:
        raise RuntimeError("result chronology differs")
    payload: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE if valid else INVALID_ARTIFACT_TYPE,
        "valid": valid,
        "created_at_utc": created_at.isoformat(timespec="seconds"),
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
        },
        "seal": {key: seal[key] for key in ("path", "sha256", "source_collection_sha256")},
        "attempt": {key: attempt[key] for key in ("path", "sha256", "payload_sha256")},
        "runtime_sources_before": source_before,
        "runtime_sources_after": source_after,
        "runtime_sources_unchanged": source_unchanged,
        "environment": current_environment,
        "environment_unchanged": environment_unchanged,
        "fit_configs": fit_config_contract(),
        "scientific_schema": scientific_schema_record(),
        "raw": raw_binding,
        "command": run_command_record(paths["valid_json"]),
        "default_change_authorized": False,
    }
    if valid:
        payload["recomputed"] = loaded_recomputed
    else:
        payload["failure"] = failure
        payload["invalid_prefix_validation"] = prefix_validation
        payload["scientific_decisions"] = None
    result_sha256 = exclusive_write_json(json_path, payload)
    note = _scientific_result_note(
        valid=valid,
        result_path=json_path,
        result_sha256=result_sha256,
        raw_binding=raw_binding,
        recomputed=loaded_recomputed,
        failure=failure,
    )
    note_sha256 = exclusive_write_bytes(note_path, note)
    return {
        "valid": valid,
        "json_path": str(json_path),
        "json_sha256": result_sha256,
        "raw_path": str(raw_path),
        "raw_sha256": raw_binding["sha256"],
        "note_path": str(note_path),
        "note_sha256": note_sha256,
        "recomputed": loaded_recomputed,
    }


def _failure_raw_receipt(archive: RawArchive) -> dict[str, Any]:
    names = sorted(name for name in archive.arrays if name.startswith("failure/"))
    evidence_arrays = {name: archive.arrays[name] for name in names}
    manifest, collection_sha256 = array_manifest(evidence_arrays)
    return {
        "names": names,
        "manifest": manifest,
        "collection_sha256": collection_sha256,
    }


def _scientific_failure_record(error: Exception, archive: RawArchive) -> dict[str, Any]:
    if isinstance(error, ProtocolInvalid):
        for raw_name, value in error.raw_arrays.items():
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._-") or "value"
            logical_name = f"failure/protocol/{safe_name}"
            suffix = 1
            while logical_name in archive.arrays:
                logical_name = f"failure/protocol/{safe_name}_{suffix}"
                suffix += 1
            with suppress(ProtocolInvalid):
                archive.add(logical_name, value)
        result = {
            "phase": error.phase,
            "reason": error.reason,
            "exception_class": type(error).__name__,
            "evidence": error.evidence,
        }
        result["raw_evidence"] = _failure_raw_receipt(archive)
        return result
    evidence: dict[str, Any] = {"exception_class": type(error).__name__}
    if isinstance(error, NonFiniteRawParameterError):
        evidence.update(
            {
                "parameter_name": error.parameter_name,
                "row_indices": list(error.row_indices),
            }
        )
        failure_name = f"failure/raw_parameter/{error.parameter_name}"
        # RawArchive deliberately raises after preserving both the exact offending bytes
        # and their uint8 non-finite classification mask.
        with suppress(ProtocolInvalid):
            archive.add(failure_name, error.raw_tensor)
    result = {
        "phase": archive.phase,
        "reason": str(error) or type(error).__name__,
        "exception_class": type(error).__name__,
        "evidence": finite_json_evidence(evidence),
    }
    result["raw_evidence"] = _failure_raw_receipt(archive)
    return result


def run_bound_scientific_experiment(seal_path: Path, output: Path) -> dict[str, Any]:
    """Consume the sole N78 attempt through the sealed, fail-closed scientific path."""
    require_implementation_ready()
    _require_cli_authorization("run", seal_path=seal_path, output=output)
    paths = official_output_paths(output)
    preflight_absent((*paths.values(), ATTEMPT))
    seal = load_and_verify_seal(seal_path)
    source_before = loaded_repository_source_snapshot()
    # All output possibilities, source/environment bindings, and the sole seal are validated
    # before this exclusive marker is created. No official scene exists above this line.
    attempt = claim_attempt(ATTEMPT, paths, seal)
    archive = RawArchive()
    evidence = ScientificEvidence(archive)
    failure: dict[str, Any] | None = None
    recomputed: dict[str, Any] | None = None
    with offline_scientific_guard(archive):
        try:
            build_global_prerequisite(evidence)
            execute_scientific_fits(evidence)
            for table in evidence.tables.values():
                if len(table.rows) != table.spec.rows:
                    raise ProtocolInvalid(
                        archive.phase,
                        f"scientific table is incomplete: {table.name}",
                        {"actual": len(table.rows), "expected": table.spec.rows},
                    )
            archive.phase = "reduction"
            # Validate the exact would-be complete archive before committing completion arrays.
            # A failed independent recomputation can therefore still receive truthful invalid
            # phase/count arrays without mutating or deleting prior evidence.
            evidence.flush(require_complete=True, include_completion=False)
            candidate_arrays = dict(archive.arrays)
            candidate_arrays.update(evidence.completion_array_values(phase="complete"))
            candidate_arrays.update(archive.completion_array_values(phase="complete"))
            recomputed = validate_and_recompute_scientific_raw(candidate_arrays, deep=True)
        except Exception as error:  # noqa: BLE001 - every post-marker failure must route invalid.
            failure = _scientific_failure_record(error, archive)

    if failure is None:
        try:
            load_and_verify_seal(seal_path)
            validate_attempt_binding(attempt)
            if canonical_json(loaded_repository_source_snapshot()) != canonical_json(source_before):
                raise ProtocolInvalid(
                    "serialization", "runtime loaded-source snapshot changed during execution"
                )
        except Exception as error:  # noqa: BLE001 - consumed attempts must terminate explicitly.
            failure = _scientific_failure_record(error, archive)

    if failure is not None:
        archive.phase = str(failure["phase"])
        evidence.flush(require_complete=False, include_completion=False)
        evidence.completion_arrays(phase=archive.phase)
        archive.completion_arrays(phase=archive.phase)
        return _write_scientific_outcome(
            valid=False,
            paths=paths,
            archive=archive,
            seal=seal,
            attempt=attempt,
            source_before=source_before,
            recomputed=None,
            failure=failure,
        )
    archive.phase = "complete"
    evidence.completion_arrays(phase="complete")
    archive.completion_arrays(phase="complete")
    return _write_scientific_outcome(
        valid=True,
        paths=paths,
        archive=archive,
        seal=seal,
        attempt=attempt,
        source_before=source_before,
        recomputed=recomputed,
        failure=None,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    seal = subparsers.add_parser(
        "seal", help="verify and seal a complete outcome-free implementation"
    )
    seal.add_argument("--output", type=Path, default=DEFAULT_SEAL)
    run = subparsers.add_parser("run", help="consume the single preregistered scientific attempt")
    run.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    run.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _authorize_actual_process_cli(
    args: argparse.Namespace, argv: list[str] | None
) -> tuple[str, Path | None, Path | None]:
    if argv is not None:
        raise RuntimeError("seal/run require the actual preregistered process CLI")
    if Path.cwd().absolute() != ROOT.absolute():
        raise RuntimeError("seal/run require the repository root as working directory")
    harness_argument = str(HARNESS)
    if args.command_name == "seal":
        output = args.output.absolute()
        expected_argv = [
            harness_argument,
            "seal",
            "--output",
            str(DEFAULT_SEAL.relative_to(ROOT)),
        ]
        authorization = ("seal", None, None)
    else:
        seal_path = args.seal.absolute()
        output = args.output.absolute()
        try:
            output_argument = str(output.relative_to(ROOT))
        except ValueError as error:
            raise RuntimeError("scientific output is outside the repository") from error
        expected_argv = [
            harness_argument,
            "run",
            "--seal",
            str(DEFAULT_SEAL.relative_to(ROOT)),
            "--output",
            output_argument,
        ]
        authorization = ("run", seal_path, output)
    if sys.argv != expected_argv:
        raise RuntimeError(f"actual CLI differs from the sole preregistered command: {sys.argv!r}")
    return authorization


def main(argv: list[str] | None = None) -> int:
    global _CLI_AUTHORIZATION
    args = parse_args(argv)
    # This must remain the first operational check.  Incomplete code cannot run verification,
    # inspect an official output, create a marker, or construct a scene.
    require_implementation_ready()
    _CLI_AUTHORIZATION = _authorize_actual_process_cli(args, argv)
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        if args.command_name == "seal":
            output = args.output.absolute()
            if output != DEFAULT_SEAL.absolute():
                raise ValueError("seal output must be the sole preregistered path")
            preflight_absent((output,))
            payload = create_seal()
            digest = exclusive_write_json(output, payload)
            print(f"saved {output} (sha256={digest})", flush=True)
            return 0
        if args.command_name == "run":
            outcome = run_bound_scientific_experiment(args.seal.absolute(), args.output.absolute())
            print(
                f"saved {outcome['json_path']} (sha256={outcome['json_sha256']})",
                flush=True,
            )
            return 0
        raise AssertionError(args.command_name)
    finally:
        _CLI_AUTHORIZATION = None


if __name__ == "__main__":
    raise SystemExit(main())
