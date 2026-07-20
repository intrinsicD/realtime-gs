#!/usr/bin/env python3
"""Frozen Stage-1 scalar/color semantic-factorial experiment.

The module is intentionally a closed scientific harness, not a configurable benchmark API.
Its mechanism and utility commands consume once-only namespaces whose exact protocol is frozen
in ``20260716_stage1_semantic_factorial_PREREG.md``.  Pure helpers are kept public so focused
tests and an independent reviewer can recompute every consequential transformation and decision.
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
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import image_metrics
from rtgs.data.scene import SceneData
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.depth.mock import GroundTruthDepth
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.image2gs.renderer2d import render_gaussian_coverage_2d, render_gaussians_2d
from rtgs.lift.base import bilinear_sample, lift_view_at_depth, lift_view_from_depth_map
from rtgs.lift.carve import CarveLifter, _ray_box
from rtgs.lift.depth import DepthLifter
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.torch_ref import TorchRasterizer

ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = Path("benchmarks/results/20260716_stage1_semantic_factorial_PREREG.md")
PREREGISTRATION_SHA256 = "f53146f12894d5e804baf699b0ba0df51d5768ef708884f5a0343c523d96e1ce"
PREREGISTRATION_REVIEW = Path(
    "benchmarks/results/20260716_stage1_semantic_factorial_PREREG_REVIEW.md"
)
IMPLEMENTATION_REVIEW = Path(
    "benchmarks/results/20260716_stage1_semantic_factorial_IMPLEMENTATION_REVIEW.md"
)
HARNESS = Path("benchmarks/stage1_semantic_factorial.py")
FOCUSED_TESTS = Path("tests/test_stage1_semantic_factorial.py")
DEFAULT_SEAL = ROOT / "benchmarks/results/20260716_stage1_semantic_factorial_SEAL.json"
PHASE_A_ATTEMPT = (
    ROOT / "benchmarks/results/20260716_stage1_semantic_factorial_PHASE_A_ATTEMPT.json"
)
PHASE_B_ATTEMPT = (
    ROOT / "benchmarks/results/20260716_stage1_semantic_factorial_PHASE_B_ATTEMPT.json"
)

SEAL_ARTIFACT_TYPE = "stage1_semantic_factorial_implementation_seal"
PHASE_A_ARTIFACT_TYPE = "stage1_semantic_factorial_mechanism"
PHASE_A_INVALID_TYPE = "stage1_semantic_factorial_mechanism_invalid"
PHASE_B_ARTIFACT_TYPE = "stage1_semantic_factorial_utility"
PHASE_B_INVALID_TYPE = "stage1_semantic_factorial_utility_invalid"
ATTEMPT_ARTIFACT_TYPE = "stage1_semantic_factorial_once_only_attempt"
RAW_SCHEMA = "stage1_semantic_factorial_raw_v1"

PHASE_A_REVIEW_GATES = (
    "raw_archive_manifest_and_finiteness",
    "preparation_and_fit_contract",
    "source_equivalence",
    "candidate_and_factorial_integrity",
    "coverage_and_retention_invariance",
    "production_independent_lift_parity",
    "mechanism_lift_field_and_source_key_invariance",
    "mechanism_lift_render_invariance",
    "completion_accounting_and_global_decision",
)

PHASE_A_SEEDS = (1103, 2203, 3301)
PHASE_B_SEEDS = (4409, 5519, 6637)
TRAIN_INDICES = (0, 1, 2, 4, 5, 6, 8, 9, 10)
HELD_OUT_INDICES = (3, 7, 11)
GAUGES = ("identity", "unit_weight", "peak_color")
TRANSFORMED_GAUGES = ("unit_weight", "peak_color")
MECHANISM_REPRESENTATIONS = (
    "m_amp__rgb_obs",
    "m_amp__h_norm",
    "unit_weight__a_amp",
)
UTILITY_ARMS = ("w_fit__c_fit", "m_amp__c_fit", "w_fit__rgb_obs", "m_amp__rgb_obs")
FACTORIAL_CODES = {
    "w_fit__c_fit": "00",
    "m_amp__c_fit": "10",
    "w_fit__rgb_obs": "01",
    "m_amp__rgb_obs": "11",
}
BACKENDS = ("Depth", "Carve")
BACKEND_CODES = {"Depth": 0, "Carve": 1}
IMAGE_SIZE = 48
COMPONENTS_PER_VIEW = 150
FIT_ITERATIONS = 120
TRAIN_ITERATIONS = 120
TRAIN_CHECKPOINTS = (0, 30, 60, 90, 120)
EXPECTED_TORCH_VERSION = "2.9.0+cu128"
RANK_DOMAIN = "stage1-semantic-factorial-v1"

SOURCE_RENDER_MAX_ABS = 5e-6
SOURCE_RENDER_REL_L1 = 1e-6
SOURCE_RENDER_MIN_PSNR = 100.0
AMPLITUDE_MAX_ABS = 1e-7
AMPLITUDE_MAX_REL = 1e-6
CANDIDATE_M_MAX_ABS = 1e-7
CANDIDATE_M_MAX_REL = 1e-6
CANDIDATE_H_MAX_ABS = 2e-6
CANDIDATE_H_MAX_REL = 2e-5
PRODUCT_MAX_ABS = 2e-7
PRODUCT_MAX_REL = 2e-6
COVERAGE_MAX_ABS = 2e-6
COVERAGE_REL_L1 = 1e-6
LIFT_ATOL = 2e-6
LIFT_RTOL = 2e-5
CHANGE_EPSILON = 1e-7

# This is true only because both scientific phases, all fail-closed routing, and the raw archive
# are implemented below.  An independent implementation review is still mandatory before seal.
IMPLEMENTATION_COMPLETE = True
IMPLEMENTATION_GAPS: tuple[str, ...] = ()


class ProtocolInvalid(ValueError):
    """Expected fail-closed invalidation with the phase that reached it."""

    def __init__(self, phase: str, reason: str, evidence: Mapping[str, Any] | None = None):
        super().__init__(reason)
        self.phase = phase
        self.reason = reason
        self.evidence = finite_json_evidence(dict(evidence or {}))


def finite_json_evidence(value: Any) -> Any:
    """Convert scientific failure evidence to strict, finite, deterministic JSON values."""
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


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def little_endian_array(value: np.ndarray) -> np.ndarray:
    """Return the canonical C-contiguous numeric/boolean array."""
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise TypeError(f"raw arrays must be numeric or boolean, got {array.dtype}")
    target = array.dtype.newbyteorder("<") if array.dtype.itemsize > 1 else array.dtype
    return np.ascontiguousarray(array.astype(target, copy=False))


def nonfinite_classification(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    mask = np.zeros(array.shape, dtype=np.uint8)
    if np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating):
        mask[np.isnan(array)] = 1
        mask[np.isposinf(array)] = 2
        mask[np.isneginf(array)] = 3
    return mask


def array_content_sha256(value: np.ndarray) -> str:
    array = little_endian_array(value)
    dtype_token = np.dtype(array.dtype).newbyteorder("<").str.encode("ascii")
    shape_bytes = np.asarray(array.shape, dtype="<i8").tobytes(order="C")
    return sha256_bytes(dtype_token + b"\0" + shape_bytes + b"\0" + array.tobytes(order="C"))


# Explicit protocol spelling retained as the public test/reviewer entry point.
raw_content_sha256 = array_content_sha256


def nullable_raw(value: Any, defined: bool) -> dict[str, np.ndarray]:
    """Return the two-array numeric nullable encoding used by :class:`RawArchive`."""
    return {
        "value": little_endian_array(np.asarray(value if defined else 0)),
        "defined": np.asarray(bool(defined), dtype=np.bool_),
    }


def array_manifest(arrays: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    pairs: list[list[str]] = []
    for name in sorted(arrays):
        value = little_endian_array(arrays[name])
        digest = array_content_sha256(value)
        entries.append(
            {
                "name": name,
                "dtype": np.dtype(value.dtype).str,
                "shape": list(value.shape),
                "byte_length": int(value.nbytes),
                "raw_content_sha256": digest,
            }
        )
        pairs.append([name, digest])
    collection = sha256_bytes(
        json.dumps(pairs, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    return entries, collection


def raw_collection_sha256(manifest: Sequence[Mapping[str, Any]]) -> str:
    """Recompute the frozen name/content collection digest from a manifest."""
    pairs = sorted(
        [[str(row["name"]), str(row["raw_content_sha256"])] for row in manifest],
        key=lambda row: row[0],
    )
    return sha256_bytes(json.dumps(pairs, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))


class RawArchive:
    """Name-unique raw tensor collection with conservative non-finite preservation."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.phase = "preflight"
        self.completed_seeds = 0
        self.completed_lifts = 0
        self.completed_models = 0

    def add(
        self,
        name: str,
        value: torch.Tensor | np.ndarray | Iterable[Any] | int | float | bool,
    ) -> np.ndarray:
        if not name or name.startswith("/") or name.endswith("/") or "//" in name:
            raise ProtocolInvalid(self.phase, f"invalid raw-array name: {name!r}")
        if name in self.arrays or f"{name}/nonfinite_classification" in self.arrays:
            raise ProtocolInvalid(self.phase, f"raw-array name reused: {name}")
        if isinstance(value, torch.Tensor):
            value = value.detach().contiguous().cpu().numpy()
        array = little_endian_array(np.asarray(value))
        classification = nonfinite_classification(array)
        if bool(classification.any()):
            # A valid archive cannot contain NaN/Inf.  Preserve their exact classes and locations
            # while replacing only the invalid values by zero in the numeric value array.
            sanitized = array.copy()
            sanitized[classification != 0] = 0
            self.arrays[name] = little_endian_array(sanitized)
            self.arrays[f"{name}/nonfinite_classification"] = classification
            raise ProtocolInvalid(self.phase, f"non-finite raw array: {name}")
        self.arrays[name] = array.copy()
        return self.arrays[name]

    def add_nullable(self, name: str, value: Any, *, defined: bool) -> None:
        self.add(f"{name}/value", value if defined else 0)
        self.add(f"{name}/defined", np.asarray(defined, dtype=np.bool_))

    def add_gaussians2d(self, prefix: str, value: Gaussians2D) -> None:
        for field in ("xy", "chol", "color", "weight"):
            self.add(f"{prefix}/{field}", getattr(value, field))

    def add_gaussians3d(self, prefix: str, value: Gaussians3D) -> None:
        for field in ("means", "quats", "log_scales", "opacity", "sh"):
            self.add(f"{prefix}/{field}", getattr(value, field))
        self.add(f"{prefix}/covariance", value.covariance())

    def add_fit_history(self, prefix: str, history: Mapping[str, Any]) -> None:
        """Archive the complete decision-relevant native-fit history without pickle."""
        psnr_rows = np.asarray(history.get("psnr", []), dtype=np.float64).reshape(-1, 2)
        self.add(f"{prefix}/psnr_iterations", psnr_rows[:, 0].astype(np.int64))
        self.add(f"{prefix}/psnr_values", psnr_rows[:, 1])
        self.add(f"{prefix}/stopped_iter", np.asarray(history["stopped_iter"], dtype=np.int64))
        self.add(f"{prefix}/final_psnr", np.asarray(history["final_psnr"], dtype=np.float64))
        self.add(
            f"{prefix}/final_psnr_full",
            np.asarray(history["final_psnr_full"], dtype=np.float64),
        )

    def completion_arrays(self) -> None:
        self.add("completion/phase_code", np.asarray(_phase_code(self.phase), dtype=np.int64))
        self.add("completion/completed_seeds", np.asarray(self.completed_seeds, dtype=np.int64))
        self.add("completion/completed_lifts", np.asarray(self.completed_lifts, dtype=np.int64))
        self.add("completion/completed_models", np.asarray(self.completed_models, dtype=np.int64))

    def manifest(self) -> list[dict[str, Any]]:
        return array_manifest(self.arrays)[0]

    def collection_sha256(self) -> str:
        return array_manifest(self.arrays)[1]

    def write(self, path: Path) -> dict[str, Any]:
        """Exclusively write and verify this archive in uncompressed NPZ form."""
        return write_raw_sidecar(path, self.arrays)


def _add_raw_value(archive: RawArchive, prefix: str, value: Any) -> None:
    """Encode structured configuration/history values as numeric, pickle-free arrays."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if isinstance(value, Mapping):
        for key in sorted(value):
            _add_raw_value(archive, f"{prefix}/{key}", value[key])
        return
    if isinstance(value, str):
        archive.add(f"{prefix}/utf8", np.frombuffer(value.encode("utf-8"), dtype=np.uint8))
        return
    if value is None:
        archive.add_nullable(prefix, 0, defined=False)
        return
    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        if array.dtype.kind in "biufc":
            archive.add(prefix, array)
            return
        for index, item in enumerate(value):
            _add_raw_value(archive, f"{prefix}/item={index}", item)
        return
    archive.add(prefix, value)


def add_config_raw(archive: RawArchive, prefix: str, config: Any) -> None:
    _add_raw_value(archive, prefix, config)


def add_fit_histories_raw(
    archive: RawArchive,
    prefix: str,
    histories: Sequence[Mapping[str, Any]],
) -> None:
    for view, history in enumerate(histories):
        _add_raw_value(archive, f"{prefix}/view={view}", history)


def _phase_code(name: str) -> int:
    phases = (
        "preflight",
        "preparation",
        "source_equivalence",
        "candidate_fields",
        "coverage_retention",
        "mechanism_lifts",
        "utility_lifts",
        "capacity",
        "refinement",
        "heldout",
        "reduction",
        "complete",
        "invalid",
    )
    return phases.index(name) if name in phases else -1


class HeldOutGuard:
    """A payload whose held-out fields are inaccessible until one explicit unlock."""

    def __init__(
        self,
        images: Sequence[torch.Tensor],
        cameras: Sequence[Any],
        depths: Sequence[torch.Tensor],
        original_indices: Sequence[int] = HELD_OUT_INDICES,
    ) -> None:
        if len(images) != 3 or len(cameras) != 3 or len(depths) != 3:
            raise ValueError("held-out payload must contain exactly three aligned views")
        self._images = tuple(images)
        self._cameras = tuple(cameras)
        self._depths = tuple(depths)
        self._original_indices = tuple(int(index) for index in original_indices)
        self._unlocked = False

    def _require_unlocked(self) -> None:
        if not self._unlocked:
            raise RuntimeError("held-out payload is locked")

    @property
    def images(self) -> tuple[torch.Tensor, ...]:
        self._require_unlocked()
        return self._images

    @property
    def cameras(self) -> tuple[Any, ...]:
        self._require_unlocked()
        return self._cameras

    @property
    def depths(self) -> tuple[torch.Tensor, ...]:
        self._require_unlocked()
        return self._depths

    @property
    def original_indices(self) -> tuple[int, ...]:
        self._require_unlocked()
        return self._original_indices

    @property
    def unlocked(self) -> bool:
        return self._unlocked

    def unlock(self) -> None:
        if self._unlocked:
            raise RuntimeError("held-out payload may be unlocked only once")
        self._unlocked = True


def frozen_fit_config() -> FitConfig:
    return FitConfig(
        n_gaussians=COMPONENTS_PER_VIEW,
        max_gaussians=5_000,
        iterations=FIT_ITERATIONS,
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
    )


fit_config = frozen_fit_config


def depth_config() -> dict[str, Any]:
    return {
        "sh_degree": 0,
        "min_weight": 0.05,
        "init_opacity": 0.1,
        "normal_thickness": 0.15,
        "covariance_mode": "surface",
        "isotropic_sigma": None,
        "robust_depth_gradients": True,
        "merge": False,
        "merge_voxel_frac": 0.01,
    }


def carve_config() -> dict[str, Any]:
    return {
        "grid_res": 48,
        "bounds_scale": 0.5,
        "min_views": 2,
        "hull_fraction": 0.85,
        "color_std_sigma": 0.20,
        "color_match_sigma": 0.35,
        "coverage_thresh": 0.40,
        "samples_per_ray": 64,
        "min_score": 0.05,
        "min_weight": 0.05,
        "merge": False,
        "merge_voxel_scale": 1.0,
        "init_opacity": 0.1,
        "sh_degree": 0,
    }


def trainer_seed(seed: int, backend: str) -> int:
    if backend not in BACKEND_CODES:
        raise ValueError(f"unknown backend token: {backend!r}")
    return 2_000_000 + 10 * int(seed) + BACKEND_CODES[backend]


def frozen_train_config(seed: int, backend: str) -> TrainConfig:
    return TrainConfig(
        iterations=120,
        lr_means=1.6e-4,
        lr_quats=1e-3,
        lr_scales=5e-3,
        lr_opacity=5e-2,
        lr_sh=2.5e-3,
        lr_sh_rest=1.25e-4,
        ssim_lambda=0.2,
        rasterizer="torch",
        device="cpu",
        densify=False,
        density_strategy="classic",
        eval_every=30,
        target_sh_degree=0,
        sh_degree_interval=None,
        use_masks=False,
        outside_alpha_lambda=0.01,
        mask_alpha_lambda=0.05,
        random_background=False,
        opacity_reg=0.0,
        scale_reg=0.0,
        packed=False,
        antialiased=False,
        sh_color_activation="hard",
        collect_sh_color_diagnostics=False,
        kernel_support_mode="hard",
        collect_kernel_support_diagnostics=False,
        visibility_margin_sigma=3.0,
        validate_render_finite=True,
        quaternion_update_policy="current",
        seed=trainer_seed(seed, backend),
    )


def frozen_renderer() -> TorchRasterizer:
    return TorchRasterizer(
        sh_color_activation="hard",
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
    )


def maximum_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    relative_floor: float,
) -> tuple[float, float]:
    absolute = (actual - expected).abs()
    maximum_absolute = float(absolute.max()) if absolute.numel() else 0.0
    relative_mask = expected.abs() > relative_floor
    maximum_relative = (
        float((absolute[relative_mask] / expected[relative_mask].abs()).max())
        if bool(relative_mask.any())
        else 0.0
    )
    return maximum_absolute, maximum_relative


def _construct_view_gauges(source: Gaussians2D) -> dict[str, Gaussians2D]:
    """Construct the three frozen product-preserving gauges for one source view."""
    amplitude = (source.weight[:, None] * source.color).detach()
    peak = amplitude.max(dim=-1).values
    positive = peak > 0
    peak_chroma = torch.zeros_like(amplitude)
    peak_chroma[positive] = amplitude[positive] / peak[positive, None]
    gauges = {
        "identity": source.detach(),
        "unit_weight": Gaussians2D(
            source.xy.detach().clone(),
            source.chol.detach().clone(),
            amplitude.detach().clone(),
            torch.ones_like(source.weight),
        ),
        "peak_color": Gaussians2D(
            source.xy.detach().clone(),
            source.chol.detach().clone(),
            peak_chroma,
            torch.where(positive, peak, torch.zeros_like(peak)),
        ),
    }
    for name, gauge in gauges.items():
        if not torch.equal(gauge.xy, source.xy) or not torch.equal(gauge.chol, source.chol):
            raise ProtocolInvalid("preparation", f"{name} changed xy/chol")
        for field in ("xy", "chol", "color", "weight"):
            if not bool(torch.isfinite(getattr(gauge, field)).all()):
                raise ProtocolInvalid("preparation", f"{name}/{field} is non-finite")
        if not bool(((gauge.color >= 0) & (gauge.color <= 1)).all()):
            raise ProtocolInvalid("preparation", f"{name}/color is out of range")
        if not bool(((gauge.weight >= 0) & (gauge.weight <= 1)).all()):
            raise ProtocolInvalid("preparation", f"{name}/weight is out of range")
        max_abs, max_rel = maximum_errors(
            gauge.weight[:, None] * gauge.color, amplitude, relative_floor=1e-8
        )
        if max_abs > AMPLITUDE_MAX_ABS or max_rel > AMPLITUDE_MAX_REL:
            raise ProtocolInvalid("preparation", f"{name} does not preserve amplitude")
    return gauges


def construct_gauges(
    source: Gaussians2D | Sequence[Gaussians2D], seed: int | None = None
) -> dict[str, Gaussians2D] | dict[str, list[Gaussians2D]]:
    """Construct gauges for one view or an ordered view list.

    ``seed`` is accepted solely to make source-key provenance explicit at call sites; gauge
    arithmetic does not depend on it.
    """
    if seed is not None:
        int(seed)
    if isinstance(source, Gaussians2D):
        return _construct_view_gauges(source)
    per_view = [_construct_view_gauges(value) for value in source]
    return {name: [record[name] for record in per_view] for name in GAUGES}


def candidate_fields(gauge: Gaussians2D, image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute detached ``a,m,h,o`` under the exact frozen pixel-center convention."""
    with torch.no_grad():
        amplitude = gauge.weight[:, None] * gauge.color
        peak = amplitude.max(dim=-1).values
        normalized = torch.zeros_like(amplitude)
        positive = peak > 0
        normalized[positive] = amplitude[positive] / peak[positive, None]
        observation = bilinear_sample(image.to(gauge.xy), gauge.xy)
    fields = {
        "a": amplitude.detach().clone(),
        "m": peak.detach().clone(),
        "h": normalized.detach().clone(),
        "o": observation.detach().clone(),
    }
    for name, value in fields.items():
        if not bool(torch.isfinite(value).all()):
            raise ProtocolInvalid("candidate_fields", f"candidate {name} is non-finite")
        if not bool(((value >= 0) & (value <= 1)).all()):
            raise ProtocolInvalid("candidate_fields", f"candidate {name} is out of range")
    if not torch.equal(fields["h"][~positive], torch.zeros_like(fields["h"][~positive])):
        raise ProtocolInvalid("candidate_fields", "candidate h violated exact zero handling")
    return fields


def mechanism_representations(
    gauge: Gaussians2D, fields: Mapping[str, torch.Tensor]
) -> dict[str, Gaussians2D]:
    return {
        "m_amp__rgb_obs": Gaussians2D(
            gauge.xy.detach().clone(),
            gauge.chol.detach().clone(),
            fields["o"].detach().clone(),
            fields["m"].detach().clone(),
        ),
        "m_amp__h_norm": Gaussians2D(
            gauge.xy.detach().clone(),
            gauge.chol.detach().clone(),
            fields["h"].detach().clone(),
            fields["m"].detach().clone(),
        ),
        "unit_weight__a_amp": Gaussians2D(
            gauge.xy.detach().clone(),
            gauge.chol.detach().clone(),
            fields["a"].detach().clone(),
            torch.ones_like(gauge.weight),
        ),
    }


def utility_representations(
    fitted: Gaussians2D, fields: Mapping[str, torch.Tensor]
) -> dict[str, Gaussians2D]:
    return {
        "w_fit__c_fit": fitted.detach(),
        "m_amp__c_fit": Gaussians2D(
            fitted.xy.detach().clone(),
            fitted.chol.detach().clone(),
            fitted.color.detach().clone(),
            fields["m"].detach().clone(),
        ),
        "w_fit__rgb_obs": Gaussians2D(
            fitted.xy.detach().clone(),
            fitted.chol.detach().clone(),
            fields["o"].detach().clone(),
            fitted.weight.detach().clone(),
        ),
        "m_amp__rgb_obs": Gaussians2D(
            fitted.xy.detach().clone(),
            fitted.chol.detach().clone(),
            fields["o"].detach().clone(),
            fields["m"].detach().clone(),
        ),
    }


construct_utility_arms = utility_representations


def validate_factorial_integrity(
    fitted: Gaussians2D,
    arms: Mapping[str, Gaussians2D],
    *,
    require_material_separation: bool = True,
) -> dict[str, Any]:
    """Mechanically verify the frozen 2x2 intervention for one fitted view."""
    if set(arms) != set(UTILITY_ARMS):
        raise ProtocolInvalid("candidate_fields", "utility arm names differ from frozen arms")
    a00 = arms["w_fit__c_fit"]
    a10 = arms["m_amp__c_fit"]
    a01 = arms["w_fit__rgb_obs"]
    a11 = arms["m_amp__rgb_obs"]
    for name, arm in arms.items():
        if not torch.equal(arm.xy, fitted.xy) or not torch.equal(arm.chol, fitted.chol):
            raise ProtocolInvalid("candidate_fields", f"{name} changed xy/chol/order")
    conditions = {
        "00_is_fitted": torch.equal(a00.weight, fitted.weight)
        and torch.equal(a00.color, fitted.color),
        "10_only_scalar": torch.equal(a10.color, a00.color),
        "01_only_color": torch.equal(a01.weight, a00.weight),
        "11_scalar_from_10": torch.equal(a11.weight, a10.weight),
        "11_color_from_01": torch.equal(a11.color, a01.color),
    }
    if not all(conditions.values()):
        raise ProtocolInvalid("candidate_fields", "factorial field identity check failed")
    scalar_changed = int(((a10.weight - a00.weight).abs() > CHANGE_EPSILON).sum())
    color_changed = int(((a01.color - a00.color).abs() > CHANGE_EPSILON).any(dim=-1).sum())
    if require_material_separation and (
        scalar_changed < math.ceil(0.10 * fitted.n) or color_changed < math.ceil(0.10 * fitted.n)
    ):
        raise ProtocolInvalid("candidate_fields", "factorial treatment separation is below 10%")
    return {
        **conditions,
        "component_count": fitted.n,
        "scalar_changed_count": scalar_changed,
        "color_changed_count": color_changed,
        "scalar_changed_fraction": scalar_changed / fitted.n,
        "color_changed_fraction": color_changed / fitted.n,
    }


def source_key(seed: int, local_view: int, component: int) -> tuple[int, int, int]:
    return int(seed), int(local_view), int(component)


def keys_array(keys: Iterable[tuple[int, int, int]]) -> np.ndarray:
    values = list(keys)
    return np.asarray(values, dtype=np.int64).reshape(-1, 3)


def rho_tie_break(seed: int, backend: str, local_view: int, component: int) -> bytes:
    """Return the exact frozen UTF-8 tie-break digest.

    Integers use Python's unpadded decimal spelling and backend tokens are literal ``Depth`` or
    ``Carve``.  Returning digest bytes makes lexicographic order explicit and testable.
    """
    if backend not in BACKENDS:
        raise ValueError("backend token must be literal 'Depth' or 'Carve'")
    text = f"{RANK_DOMAIN}|{int(seed)}|{backend}|{int(local_view)}|{int(component)}"
    return hashlib.sha256(text.encode("utf-8")).digest()


def rank_available_keys(
    *,
    seed: int,
    backend: str,
    local_view: int,
    available_components: Iterable[int],
    rho: torch.Tensor | np.ndarray,
) -> list[int]:
    """Rank available component indices by frozen density mass and exact tie break."""
    mass = np.asarray(rho, dtype=np.float64)
    if mass.shape != (COMPONENTS_PER_VIEW,):
        raise ValueError(f"rho must have shape ({COMPONENTS_PER_VIEW},)")
    components = [int(value) for value in available_components]
    if len(components) != len(set(components)) or any(
        value < 0 or value >= COMPONENTS_PER_VIEW for value in components
    ):
        raise ValueError("available component indices are invalid")
    return sorted(
        components,
        key=lambda component: (
            -float(mass[component]),
            rho_tie_break(seed, backend, local_view, component),
            component,
        ),
    )


def integrated_mass_rho(gaussian: Gaussians2D, peak_amplitude: torch.Tensor) -> torch.Tensor:
    """Frozen float64 ``m * L11 * L22`` mass used for capacity ranking."""
    if peak_amplitude.shape != (gaussian.n,):
        raise ValueError("peak amplitude shape differs from the 2D component count")
    rho = (
        peak_amplitude.detach().to(torch.float64)
        * gaussian.chol[:, 0].detach().to(torch.float64)
        * gaussian.chol[:, 2].detach().to(torch.float64)
    )
    if not bool(torch.isfinite(rho).all()) or not bool((rho >= 0).all()):
        raise ProtocolInvalid("capacity", "integrated mass is invalid")
    return rho


def expected_schedule(
    seed: int,
    backend: str,
    n_steps: int = TRAIN_ITERATIONS,
    n_views: int = len(TRAIN_INDICES),
) -> np.ndarray:
    if n_steps <= 0 or n_views <= 0:
        raise ValueError("schedule dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(trainer_seed(seed, backend))
    return np.asarray(
        [int(torch.randint(0, n_views, (1,), generator=generator)) for _ in range(n_steps)],
        dtype=np.int64,
    )


def factorial_estimands(values: Mapping[str, float]) -> dict[str, float]:
    y00 = float(values["w_fit__c_fit"])
    y10 = float(values["m_amp__c_fit"])
    y01 = float(values["w_fit__rgb_obs"])
    y11 = float(values["m_amp__rgb_obs"])
    return {
        "scalar_main_effect": 0.5 * ((y10 - y00) + (y11 - y01)),
        "color_main_effect": 0.5 * ((y01 - y00) + (y11 - y10)),
        "interaction": y11 - y10 - y01 + y00,
        "full_candidate_difference": y11 - y00,
    }


def material_driver(effects: Sequence[float]) -> bool:
    mean = sum(float(value) for value in effects) / len(effects)
    if abs(mean) < 0.25 or mean == 0:
        return False
    sign_matches = sum((value > 0) == (mean > 0) and value != 0 for value in effects)
    return sign_matches >= 2


def backend_decision(
    psnr_differences: Sequence[float],
    ssim_differences: Sequence[float],
    *,
    validity_gates_pass: bool,
) -> dict[str, Any]:
    if len(psnr_differences) != 3 or len(ssim_differences) != 3:
        raise ValueError("backend decisions require exactly three seed differences")
    psnr_values = [float(value) for value in psnr_differences]
    ssim_values = [float(value) for value in ssim_differences]
    mean_psnr = sum(psnr_values) / 3
    mean_ssim = sum(ssim_values) / 3
    noninferior = (
        validity_gates_pass
        and mean_psnr >= -0.25
        and sum(value >= -0.25 for value in psnr_values) >= 2
        and min(psnr_values) >= -0.75
        and mean_ssim >= -0.005
        and min(ssim_values) >= -0.020
    )
    improved = (
        noninferior
        and mean_psnr >= 0.25
        and sum(value >= 0.10 for value in psnr_values) >= 2
        and min(psnr_values) >= -0.25
        and mean_ssim >= 0.0
    )
    return {
        "mean_psnr_difference_db": mean_psnr,
        "worst_psnr_difference_db": min(psnr_values),
        "mean_ssim_difference": mean_ssim,
        "worst_ssim_difference": min(ssim_values),
        "noninferior": noninferior,
        "material_improvement": improved,
    }


def frozen_decisions(
    psnr_differences: Mapping[str, Sequence[float]],
    ssim_differences: Mapping[str, Sequence[float]],
    *,
    validity_gates_pass: bool,
) -> dict[str, Any]:
    by_backend = {
        backend: backend_decision(
            psnr_differences[backend],
            ssim_differences[backend],
            validity_gates_pass=validity_gates_pass,
        )
        for backend in BACKENDS
    }
    return {
        "by_backend": by_backend,
        "repair_utility_survives": all(row["noninferior"] for row in by_backend.values()),
        "cross_backend_material_improvement": all(
            row["material_improvement"] for row in by_backend.values()
        ),
    }


def official_output_paths(output: Path, phase: str) -> dict[str, Path]:
    """Derive every mutually exclusive result path before an attempt is claimed."""
    if phase not in {"mechanism", "utility"}:
        raise ValueError("phase must be 'mechanism' or 'utility'")
    absolute = output.resolve()
    results = (ROOT / "benchmarks/results").resolve()
    suffix = f"_cpu_stage1_semantic_factorial_{phase}"
    pattern = rf"\d{{8}}T\d{{6}}Z{re.escape(suffix)}\.json"
    if absolute.parent != results or re.fullmatch(pattern, absolute.name) is None:
        raise ValueError(f"official output must be benchmarks/results/<fresh-UTC>{suffix}.json")
    stem = absolute.stem
    return {
        "valid_json": absolute,
        "valid_raw": absolute.with_name(f"{stem}_RAW.npz"),
        "valid_note": absolute.with_name(f"{stem}_RESULT.md"),
        "invalid_json": absolute.with_name(f"{stem}_invalid.json"),
        "invalid_raw": absolute.with_name(f"{stem}_invalid_RAW.npz"),
        "invalid_note": absolute.with_name(f"{stem}_invalid_RESULT.md"),
    }


def coverage_array_name(seed: int, gauge: str, arm: str, view: int) -> str:
    if gauge not in GAUGES or arm not in (*MECHANISM_REPRESENTATIONS, *UTILITY_ARMS):
        raise ValueError("coverage name uses an unknown gauge or representation")
    if view not in range(len(TRAIN_INDICES)):
        raise ValueError("coverage view is out of range")
    return f"coverage/seed={int(seed)}/gauge={gauge}/arm={arm}/view={int(view)}"


def match_exact_capacity(
    *,
    seed: int,
    backend: str,
    availability_by_arm_view: Mapping[str, Sequence[Sequence[int]]],
    rho_by_view: Sequence[torch.Tensor | np.ndarray],
) -> dict[str, Any]:
    """Apply the frozen per-view minimum quota and common-rho rank."""
    if set(availability_by_arm_view) != set(UTILITY_ARMS):
        raise ValueError("capacity availability must contain the four frozen arms")
    if len(rho_by_view) != len(TRAIN_INDICES):
        raise ValueError("rho must contain exactly nine source views")
    for arm in UTILITY_ARMS:
        if len(availability_by_arm_view[arm]) != len(TRAIN_INDICES):
            raise ValueError("every arm must contain nine availability lists")
    quotas = [
        min(len(availability_by_arm_view[arm][view]) for arm in UTILITY_ARMS)
        for view in range(len(TRAIN_INDICES))
    ]
    if any(quota < 8 for quota in quotas) or sum(quotas) < 270:
        raise ProtocolInvalid(
            "capacity",
            "exact-capacity floor failed",
            {"per_view_quotas": quotas, "total_quota": sum(quotas)},
        )
    selected: dict[str, list[list[int]]] = {}
    ranks: dict[str, list[list[int]]] = {}
    for arm in UTILITY_ARMS:
        selected[arm] = []
        ranks[arm] = []
        for view, quota in enumerate(quotas):
            ranked = rank_available_keys(
                seed=seed,
                backend=backend,
                local_view=view,
                available_components=availability_by_arm_view[arm][view],
                rho=rho_by_view[view],
            )
            ranks[arm].append(ranked)
            selected[arm].append(sorted(ranked[:quota]))
    return {
        "per_view_quotas": quotas,
        "total_quota": sum(quotas),
        "ranked_components": ranks,
        "selected_components": selected,
    }


@dataclass
class PreparedSeed:
    seed: int
    scene: SceneData
    fitted: list[Gaussians2D]
    fit_histories: list[dict[str, Any]]
    gauges: dict[str, list[Gaussians2D]]
    heldout: HeldOutGuard | None
    preparation: dict[str, Any]


@dataclass
class LiftResult:
    output: Gaussians3D
    keys: list[tuple[int, int, int]]
    evidence: dict[str, Any]
    sidecar_arrays: dict[str, torch.Tensor]


@dataclass
class UtilityModel:
    natural: Gaussians3D
    matched: Gaussians3D
    final: Gaussians3D
    keys: list[tuple[int, int, int]]
    checkpoints: dict[int, Gaussians3D]
    history: dict[str, Any]
    evidence: dict[str, Any]


def _tensor_digest(value: torch.Tensor) -> str:
    return raw_content_sha256(value.detach().contiguous().cpu().numpy())


def _gaussians3d_digest(value: Gaussians3D) -> str:
    return canonical_json_hash(
        {
            field: _tensor_digest(getattr(value, field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        }
    )


def _archive_camera(archive: RawArchive, prefix: str, camera: Any) -> None:
    archive.add(
        f"{prefix}/intrinsics_and_size",
        np.asarray(
            [camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height],
            dtype=np.float64,
        ),
    )
    archive.add(f"{prefix}/R", camera.R)
    archive.add(f"{prefix}/t", camera.t)


def _minimal_train_scene(train_scene: SceneData) -> SceneData:
    center, extent = train_scene.center_and_extent()
    scene = SceneData(
        images=list(train_scene.images),
        cameras=list(train_scene.cameras),
        view_names=None if train_scene.view_names is None else list(train_scene.view_names),
        points=train_scene.points,
        point_visibility=train_scene.point_visibility,
        masks=train_scene.masks,
        gt_depths=train_scene.gt_depths,
        gt_gaussians=None,
        train_indices=list(range(len(TRAIN_INDICES))),
        test_indices=[],
        bounds_hint=train_scene.bounds_hint,
        name=f"{train_scene.name}-field-minimal",
        _extent_cache=(center.detach().clone(), float(extent)),
    )
    scene.validate()
    rebound_center, rebound_extent = scene.center_and_extent()
    if not torch.equal(center, rebound_center) or float(extent) != float(rebound_extent):
        raise ProtocolInvalid("preparation", "field-minimal scene changed center/extent")
    if scene.training_views != list(range(9)) or scene.testing_views:
        raise ProtocolInvalid("preparation", "physical training subset has an invalid split")
    if scene.gt_gaussians is not None or scene.gt_depths is None:
        raise ProtocolInvalid("preparation", "field-minimal lift scene has invalid GT fields")
    return scene


def _archive_training_inputs(archive: RawArchive, seed: int, scene: SceneData) -> None:
    prefix = f"input/seed={seed}"
    archive.add(f"{prefix}/local_to_original", np.asarray(TRAIN_INDICES, dtype=np.int64))
    center, extent = scene.center_and_extent()
    archive.add(f"{prefix}/center", center)
    archive.add(f"{prefix}/extent", np.asarray(extent, dtype=np.float64))
    for view, (image, camera, depth) in enumerate(
        zip(scene.images, scene.cameras, scene.gt_depths or (), strict=True)
    ):
        archive.add(f"{prefix}/view={view}/image", image)
        archive.add(f"{prefix}/view={view}/depth", depth)
        _archive_camera(archive, f"{prefix}/view={view}/camera", camera)
        if scene.masks is None:
            archive.add_nullable(f"{prefix}/view={view}/mask", 0, defined=False)
        else:
            archive.add_nullable(f"{prefix}/view={view}/mask", scene.masks[view], defined=True)
    archive.add_nullable(
        f"{prefix}/points",
        0 if scene.points is None else scene.points,
        defined=scene.points is not None,
    )
    if scene.point_visibility is None:
        archive.add_nullable(f"{prefix}/point_visibility", 0, defined=False)
    else:
        visibility_pairs = np.asarray(
            [
                (view, int(index))
                for view, indices in enumerate(scene.point_visibility)
                for index in indices.tolist()
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        archive.add_nullable(f"{prefix}/point_visibility", visibility_pairs, defined=True)
        for view, indices in enumerate(scene.point_visibility):
            archive.add(f"{prefix}/point_visibility/view={view}", indices)
    if scene.bounds_hint is None:
        archive.add_nullable(f"{prefix}/bounds_hint", 0, defined=False)
    else:
        bounds_value = torch.cat(
            [
                scene.bounds_hint[0].detach().reshape(-1).to(dtype=torch.float64),
                torch.as_tensor([scene.bounds_hint[1]], dtype=torch.float64),
            ]
        )
        archive.add_nullable(f"{prefix}/bounds_hint", bounds_value, defined=True)
        archive.add(f"{prefix}/bounds_hint/center", scene.bounds_hint[0])
        archive.add(
            f"{prefix}/bounds_hint/extent",
            np.asarray(scene.bounds_hint[1], dtype=np.float64),
        )


def validate_fitted_views(fitted: Sequence[Gaussians2D]) -> None:
    if len(fitted) != len(TRAIN_INDICES):
        raise ProtocolInvalid("preparation", "fit did not return exactly nine views")
    for view, gaussian in enumerate(fitted):
        expected = {
            "xy": (COMPONENTS_PER_VIEW, 2),
            "chol": (COMPONENTS_PER_VIEW, 3),
            "color": (COMPONENTS_PER_VIEW, 3),
            "weight": (COMPONENTS_PER_VIEW,),
        }
        if gaussian.n != COMPONENTS_PER_VIEW:
            raise ProtocolInvalid("preparation", f"fit view {view} count is not 150")
        for field, shape in expected.items():
            value = getattr(gaussian, field)
            if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ProtocolInvalid("preparation", f"fit view {view}/{field} is invalid")
        if not bool((gaussian.chol[:, (0, 2)] > 0).all()):
            raise ProtocolInvalid("preparation", f"fit view {view} has nonpositive chol")
        if not bool(((gaussian.color >= 0) & (gaussian.color <= 1)).all()):
            raise ProtocolInvalid("preparation", f"fit view {view} color is out of range")
        if not bool(((gaussian.weight >= 0) & (gaussian.weight <= 1)).all()):
            raise ProtocolInvalid("preparation", f"fit view {view} weight is out of range")
        valid_xy = (
            (gaussian.xy[:, 0] >= 0)
            & (gaussian.xy[:, 0] < IMAGE_SIZE)
            & (gaussian.xy[:, 1] >= 0)
            & (gaussian.xy[:, 1] < IMAGE_SIZE)
        )
        if not bool(valid_xy.all()):
            raise ProtocolInvalid("preparation", f"fit view {view} xy is out of bounds")


def prepare_seed(seed: int, phase: str, archive: RawArchive) -> PreparedSeed:
    if phase not in {"mechanism", "utility"}:
        raise ValueError("unknown preparation phase")
    archive.phase = "preparation"
    full_scene = make_synthetic_scene(
        n_gaussians=40,
        n_cameras=12,
        image_size=IMAGE_SIZE,
        seed=seed,
    )
    if full_scene.gt_depths is None:
        raise ProtocolInvalid("preparation", "synthetic scene has no depth fields")
    heldout = None
    if phase == "utility":
        heldout = HeldOutGuard(
            [full_scene.images[index] for index in HELD_OUT_INDICES],
            [full_scene.cameras[index] for index in HELD_OUT_INDICES],
            [full_scene.gt_depths[index] for index in HELD_OUT_INDICES],
        )
    train_scene = full_scene.subset(list(TRAIN_INDICES), name_suffix=f"n77-{phase}-train")
    del full_scene
    train_scene.validate()
    scene = _minimal_train_scene(train_scene)
    del train_scene
    _archive_training_inputs(archive, seed, scene)
    add_config_raw(archive, f"fit/seed={seed}/config", frozen_fit_config())
    fitted, histories = fit_views(
        scene.images,
        frozen_fit_config(),
        seed=seed,
        masks=scene.masks,
    )
    validate_fitted_views(fitted)
    add_fit_histories_raw(archive, f"fit/seed={seed}/histories", histories)
    for view, gaussian in enumerate(fitted):
        archive.add_gaussians2d(f"fit/seed={seed}/view={view}", gaussian)
        archive.add(
            f"fit/seed={seed}/view={view}/source_keys",
            keys_array(source_key(seed, view, component) for component in range(gaussian.n)),
        )
    if phase == "mechanism":
        built = construct_gauges(fitted, seed=seed)
        assert isinstance(built["identity"], list)
        gauges = built
    else:
        # Phase B forms arms directly from the identity fit and must not recreate Phase-A gauges.
        gauges = {"identity": [gaussian.detach() for gaussian in fitted]}
    for gauge in gauges:
        for view, gaussian in enumerate(gauges[gauge]):
            archive.add_gaussians2d(f"gauge/seed={seed}/gauge={gauge}/view={view}", gaussian)
    preparation = {
        "seed": seed,
        "synthetic_config": {"n_gaussians": 40, "n_cameras": 12, "image_size": IMAGE_SIZE},
        "split": {"train": list(TRAIN_INDICES), "held_out": list(HELD_OUT_INDICES)},
        "fit_config": asdict(frozen_fit_config()),
        "fit_histories": histories,
        "fit_field_digest": canonical_json_hash(
            [
                {
                    field: _tensor_digest(getattr(gaussian, field))
                    for field in ("xy", "chol", "color", "weight")
                }
                for gaussian in fitted
            ]
        ),
        "heldout_guarded": heldout is not None and not heldout.unlocked,
    }
    return PreparedSeed(
        seed=seed,
        scene=scene,
        fitted=fitted,
        fit_histories=histories,
        gauges=gauges,
        heldout=heldout,
        preparation=preparation,
    )


def source_equivalence(prepared: Sequence[PreparedSeed], archive: RawArchive) -> dict[str, Any]:
    """Complete the global source-render prerequisite before any candidate construction."""
    archive.phase = "source_equivalence"
    rows = []
    for item in prepared:
        for view in range(len(TRAIN_INDICES)):
            with torch.no_grad():
                identity = render_gaussians_2d(
                    item.gauges["identity"][view],
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    background=torch.zeros(3),
                    row_chunk=64,
                )
            archive.add(
                f"source_render/seed={item.seed}/gauge=identity/view={view}/color", identity
            )
            denominator = float(identity.double().abs().sum())
            if denominator <= 0 or not math.isfinite(denominator):
                raise ProtocolInvalid("source_equivalence", "identity render denominator invalid")
            for gauge in TRANSFORMED_GAUGES:
                with torch.no_grad():
                    changed = render_gaussians_2d(
                        item.gauges[gauge][view],
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                        background=torch.zeros(3),
                        row_chunk=64,
                    )
                delta = changed - identity
                archive.add(
                    f"source_render/seed={item.seed}/gauge={gauge}/view={view}/color", changed
                )
                archive.add(
                    f"source_render/seed={item.seed}/gauge={gauge}/view={view}/delta", delta
                )
                numerator = float(delta.double().abs().sum())
                maximum = float(delta.abs().max())
                ratio = numerator / denominator
                mse = float(delta.double().square().mean())
                diagnostic_psnr = -10.0 * math.log10(max(mse, 1e-12))
                row = {
                    "seed": item.seed,
                    "local_train_view_index": view,
                    "original_view_index": TRAIN_INDICES[view],
                    "transform": gauge,
                    "maximum_absolute_rgb_error": maximum,
                    "delta_l1_float64": numerator,
                    "identity_l1_float64": denominator,
                    "delta_over_identity": ratio,
                    "diagnostic_psnr_db": diagnostic_psnr,
                }
                rows.append(row)
                if (
                    maximum > SOURCE_RENDER_MAX_ABS
                    or ratio > SOURCE_RENDER_REL_L1
                    or diagnostic_psnr < SOURCE_RENDER_MIN_PSNR
                ):
                    raise ProtocolInvalid(
                        "source_equivalence", "global source equivalence failed", row
                    )
    expected = len(PHASE_A_SEEDS) * len(TRAIN_INDICES) * len(TRANSFORMED_GAUGES)
    if len(rows) != expected:
        raise ProtocolInvalid("source_equivalence", "source equivalence check count differs")
    return {
        "passed": True,
        "completed_checks": len(rows),
        "expected_checks": expected,
        "thresholds": {
            "maximum_absolute_rgb_error": SOURCE_RENDER_MAX_ABS,
            "delta_over_identity": SOURCE_RENDER_REL_L1,
            "minimum_diagnostic_psnr_db": SOURCE_RENDER_MIN_PSNR,
            "mse_floor": 1e-12,
        },
        "rows": rows,
    }


def construct_and_validate_candidates(
    prepared: Sequence[PreparedSeed], archive: RawArchive
) -> tuple[
    dict[int, dict[str, list[dict[str, torch.Tensor]]]],
    dict[int, dict[str, list[Gaussians2D]]],
    dict[int, dict[str, list[Gaussians2D]]],
    dict[str, Any],
]:
    archive.phase = "candidate_fields"
    all_fields: dict[int, dict[str, list[dict[str, torch.Tensor]]]] = {}
    mechanism: dict[int, dict[str, list[Gaussians2D]]] = {}
    utility: dict[int, dict[str, list[Gaussians2D]]] = {}
    seed_records = []
    pooled_scalar = pooled_color = pooled_count = 0
    for item in prepared:
        all_fields[item.seed] = {gauge: [] for gauge in GAUGES}
        mechanism[item.seed] = {
            f"{gauge}/{representation}": []
            for gauge in GAUGES
            for representation in MECHANISM_REPRESENTATIONS
        }
        utility[item.seed] = {arm: [] for arm in UTILITY_ARMS}
        scalar_changed = color_changed = component_count = 0
        candidate_rows = []
        for view in range(len(TRAIN_INDICES)):
            identity_fields: dict[str, torch.Tensor] | None = None
            for gauge in GAUGES:
                source = item.gauges[gauge][view]
                fields = candidate_fields(source, item.scene.images[view])
                all_fields[item.seed][gauge].append(fields)
                for field, value in fields.items():
                    archive.add(
                        f"candidate/seed={item.seed}/gauge={gauge}/view={view}/{field}", value
                    )
                reps = mechanism_representations(source, fields)
                for representation, gaussian in reps.items():
                    if not torch.equal(gaussian.xy, source.xy) or not torch.equal(
                        gaussian.chol, source.chol
                    ):
                        raise ProtocolInvalid("candidate_fields", "representation changed geometry")
                    mechanism[item.seed][f"{gauge}/{representation}"].append(gaussian)
                    archive.add_gaussians2d(
                        f"representation/seed={item.seed}/gauge={gauge}/arm={representation}/view={view}",
                        gaussian,
                    )
                if identity_fields is None:
                    identity_fields = fields
                else:
                    m_abs, m_rel = maximum_errors(
                        fields["m"], identity_fields["m"], relative_floor=1e-8
                    )
                    h_abs, h_rel = maximum_errors(
                        fields["h"], identity_fields["h"], relative_floor=1e-6
                    )
                    if not torch.equal(
                        source.xy, item.gauges["identity"][view].xy
                    ) or not torch.equal(source.chol, item.gauges["identity"][view].chol):
                        raise ProtocolInvalid("candidate_fields", "gauge geometry differs")
                    if not torch.equal(fields["o"], identity_fields["o"]):
                        raise ProtocolInvalid(
                            "candidate_fields", "sampled observation is not bit-exact"
                        )
                    if (
                        m_abs > CANDIDATE_M_MAX_ABS
                        or m_rel > CANDIDATE_M_MAX_REL
                        or h_abs > CANDIDATE_H_MAX_ABS
                        or h_rel > CANDIDATE_H_MAX_REL
                    ):
                        raise ProtocolInvalid(
                            "candidate_fields", "candidate gauge invariance failed"
                        )
                product_abs, product_rel = maximum_errors(
                    fields["m"][:, None] * fields["h"],
                    item.gauges["identity"][view].weight[:, None]
                    * item.gauges["identity"][view].color,
                    relative_floor=1e-8,
                )
                unit_abs, unit_rel = maximum_errors(
                    reps["unit_weight__a_amp"].color,
                    item.gauges["identity"][view].weight[:, None]
                    * item.gauges["identity"][view].color,
                    relative_floor=1e-8,
                )
                if (
                    product_abs > PRODUCT_MAX_ABS
                    or product_rel > PRODUCT_MAX_REL
                    or unit_abs > AMPLITUDE_MAX_ABS
                    or unit_rel > AMPLITUDE_MAX_REL
                    or not torch.equal(
                        reps["unit_weight__a_amp"].weight,
                        torch.ones_like(reps["unit_weight__a_amp"].weight),
                    )
                ):
                    raise ProtocolInvalid("candidate_fields", "exact-product control failed")
                candidate_rows.append(
                    {
                        "view": view,
                        "gauge": gauge,
                        "m_max_abs": 0.0 if identity_fields is fields else m_abs,
                        "m_max_rel": 0.0 if identity_fields is fields else m_rel,
                        "h_max_abs": 0.0 if identity_fields is fields else h_abs,
                        "h_max_rel": 0.0 if identity_fields is fields else h_rel,
                        "product_max_abs": product_abs,
                        "product_max_rel": product_rel,
                    }
                )
            assert identity_fields is not None
            arms = utility_representations(item.fitted[view], identity_fields)
            integrity = validate_factorial_integrity(
                item.fitted[view], arms, require_material_separation=False
            )
            scalar_changed += integrity["scalar_changed_count"]
            color_changed += integrity["color_changed_count"]
            component_count += integrity["component_count"]
            for arm, gaussian in arms.items():
                utility[item.seed][arm].append(gaussian)
                archive.add_gaussians2d(
                    f"utility_arm/seed={item.seed}/arm={arm}/view={view}", gaussian
                )
        scalar_fraction = scalar_changed / component_count
        color_fraction = color_changed / component_count
        if scalar_fraction < 0.10 or color_fraction < 0.10:
            raise ProtocolInvalid("candidate_fields", "per-seed factorial identifiability failed")
        seed_records.append(
            {
                "seed": item.seed,
                "component_count": component_count,
                "scalar_changed_count": scalar_changed,
                "scalar_changed_fraction": scalar_fraction,
                "color_changed_count": color_changed,
                "color_changed_fraction": color_fraction,
                "candidate_rows": candidate_rows,
            }
        )
        pooled_scalar += scalar_changed
        pooled_color += color_changed
        pooled_count += component_count
    if pooled_scalar / pooled_count < 0.10 or pooled_color / pooled_count < 0.10:
        raise ProtocolInvalid("candidate_fields", "pooled factorial identifiability failed")
    return (
        all_fields,
        mechanism,
        utility,
        {
            "passed": True,
            "seeds": seed_records,
            "pooled": {
                "component_count": pooled_count,
                "scalar_changed_count": pooled_scalar,
                "scalar_changed_fraction": pooled_scalar / pooled_count,
                "color_changed_count": pooled_color,
                "color_changed_fraction": pooled_color / pooled_count,
            },
        },
    )


def _assert_gaussian_parity(
    production: Gaussians3D,
    independent: Gaussians3D,
    *,
    phase: str,
) -> dict[str, float]:
    if production.n != independent.n:
        raise ProtocolInvalid(phase, "production/independent lift counts differ")
    pairs = {
        "means": (production.means, independent.means),
        "covariance": (production.covariance(), independent.covariance()),
        "opacity": (production.opacity, independent.opacity),
        "sh": (production.sh, independent.sh),
    }
    errors = {}
    for name, (actual, expected) in pairs.items():
        errors[name] = float((actual - expected).abs().max()) if actual.numel() else 0.0
        if not torch.allclose(actual, expected, atol=LIFT_ATOL, rtol=LIFT_RTOL):
            raise ProtocolInvalid(phase, f"production/independent {name} parity failed", errors)
    return errors


def _depth_independent(
    seed: int,
    views: Sequence[Gaussians2D],
    scene: SceneData,
) -> tuple[Gaussians3D | None, list[tuple[int, int, int]], dict[str, torch.Tensor]]:
    if scene.gt_depths is None:
        raise ProtocolInvalid("lift", "Depth sidecar requires training depth maps")
    parts: list[Gaussians3D] = []
    keys: list[tuple[int, int, int]] = []
    arrays: dict[str, torch.Tensor] = {}
    for view, (gaussian, camera, depth_map) in enumerate(
        zip(views, scene.cameras, scene.gt_depths, strict=True)
    ):
        depth_map = depth_map.to(gaussian.xy)
        sampled_depth = bilinear_sample(depth_map, gaussian.xy)
        finite = torch.isfinite(sampled_depth)
        depth_valid = sampled_depth > 0.05
        retained = gaussian.weight > 0.05
        mask_valid = torch.ones_like(retained)
        if scene.masks is not None:
            mask_valid = bilinear_sample(scene.masks[view].to(gaussian.xy), gaussian.xy) > 0.5
        confidence = torch.ones_like(sampled_depth)
        confidence_valid = confidence > 0.1
        valid = finite & depth_valid & retained & mask_valid & confidence_valid
        component_indices = torch.where(valid)[0]
        keys.extend(source_key(seed, view, int(index)) for index in component_indices.tolist())
        arrays[f"view={view}/sampled_depth"] = sampled_depth
        arrays[f"view={view}/finite"] = finite
        arrays[f"view={view}/depth_valid"] = depth_valid
        arrays[f"view={view}/retained"] = retained
        arrays[f"view={view}/mask_valid"] = mask_valid
        arrays[f"view={view}/confidence"] = confidence
        arrays[f"view={view}/confidence_valid"] = confidence_valid
        arrays[f"view={view}/valid"] = valid
        arrays[f"view={view}/component_indices"] = component_indices
        if component_indices.numel() == 0:
            continue
        selected = Gaussians2D(
            xy=gaussian.xy[valid],
            chol=gaussian.chol[valid],
            color=gaussian.color[valid],
            weight=gaussian.weight[valid],
        )
        z = sampled_depth[valid]
        parts.append(
            lift_view_from_depth_map(
                camera,
                selected,
                depth_map,
                z,
                0,
                opacity=torch.full_like(z, 0.1),
                normal_thickness=0.15,
                robust_depth_gradients=True,
            )
        )
    return (Gaussians3D.cat(parts) if parts else None), keys, arrays


def _carve_independent(
    seed: int,
    views: Sequence[Gaussians2D],
    scene: SceneData,
    coverage_maps: Sequence[torch.Tensor],
) -> tuple[Gaussians3D | None, list[tuple[int, int, int]], dict[str, torch.Tensor]]:
    """Reconstruct the current unmerged Carve path without a second production call."""
    cfg = carve_config()
    center, extent = scene.center_and_extent()
    half = extent * cfg["bounds_scale"]
    lower = center - half
    voxel = 2.0 * half / cfg["grid_res"]
    grid = cfg["grid_res"]
    device = center.device
    axis = torch.arange(grid, dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    idx3 = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    centers = lower[None, :] + (idx3 + 0.5) * voxel
    voxel_count = centers.shape[0]
    n_seen = torch.zeros(voxel_count, device=device)
    n_covered = torch.zeros(voxel_count, device=device)
    color_sum = torch.zeros(voxel_count, 3, device=device)
    color_square_sum = torch.zeros(voxel_count, 3, device=device)
    for image, coverage, camera in zip(scene.images, coverage_maps, scene.cameras, strict=True):
        uv, depth = camera.project(centers)
        inside = (depth > 0.05) & camera.in_image(uv, margin=-0.5)
        indices = inside.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        covered = bilinear_sample(coverage, uv[indices]) > cfg["coverage_thresh"]
        colors = bilinear_sample(image, uv[indices])
        n_seen[indices] += 1
        n_covered[indices] += covered.float()
        color_sum[indices] += colors
        color_square_sum[indices] += colors.square()
    seen_ok = n_seen >= cfg["min_views"]
    hull = seen_ok & (n_covered >= cfg["hull_fraction"] * n_seen) & (n_covered >= cfg["min_views"])
    count = n_seen.clamp_min(1.0)
    mean_color = color_sum / count[:, None]
    variance = (color_square_sum / count[:, None] - mean_color.square()).clamp_min(0.0)
    standard_deviation = variance.mean(dim=-1).sqrt()
    consistency = hull.float() * torch.exp(
        -(standard_deviation.square()) / (2 * cfg["color_std_sigma"] ** 2)
    )
    arrays: dict[str, torch.Tensor] = {
        "volume/idx3": idx3,
        "volume/centers": centers,
        "volume/n_seen": n_seen,
        "volume/n_covered": n_covered,
        "volume/color_sum": color_sum,
        "volume/color_square_sum": color_square_sum,
        "volume/seen_ok": seen_ok,
        "volume/hull": hull,
        "volume/count": count,
        "volume/mean_color": mean_color,
        "volume/variance": variance,
        "volume/standard_deviation": standard_deviation,
        "volume/consistency": consistency,
    }
    parts: list[Gaussians3D] = []
    keys: list[tuple[int, int, int]] = []
    samples = cfg["samples_per_ray"]
    for view, (gaussian, camera) in enumerate(zip(views, scene.cameras, strict=True)):
        retained = gaussian.weight > cfg["min_weight"]
        if scene.masks is not None:
            retained &= bilinear_sample(scene.masks[view].to(gaussian.xy), gaussian.xy) > 0.5
        retained_indices = torch.where(retained)[0]
        arrays[f"view={view}/retained"] = retained
        arrays[f"view={view}/retained_indices"] = retained_indices
        if retained_indices.numel() == 0:
            arrays[f"view={view}/placed_indices"] = retained_indices
            continue
        selected = Gaussians2D(
            xy=gaussian.xy[retained],
            chol=gaussian.chol[retained],
            color=gaussian.color[retained],
            weight=gaussian.weight[retained],
        )
        origin, directions = camera.pixel_rays(selected.xy)
        entry, exit = _ray_box(origin, directions, lower, lower + 2 * half)
        valid_ray = exit > entry.clamp_min(0.05)
        entry = entry.clamp_min(0.05)
        steps = torch.linspace(0.0, 1.0, samples, device=device)
        sample_depths = entry[:, None] + (exit - entry).clamp_min(0.0)[:, None] * steps[None]
        points = origin.reshape(1, 1, 3) + sample_depths[:, :, None] * directions[:, None]
        voxel_indices = torch.floor((points - lower) / voxel).long()
        in_grid = ((voxel_indices >= 0) & (voxel_indices < grid)).all(dim=-1) & valid_ray[:, None]
        voxel_indices = voxel_indices.clamp(0, grid - 1)
        flat = (voxel_indices[..., 2] * grid + voxel_indices[..., 1]) * grid + voxel_indices[..., 0]
        score = consistency[flat] * in_grid.float()
        voxel_color = mean_color[flat]
        color_difference = ((voxel_color - selected.color[:, None]).square()).sum(dim=-1)
        score = score * torch.exp(-color_difference / (2 * cfg["color_match_sigma"] ** 2))
        best_score, best_index = score.max(dim=1)
        placed = best_score > cfg["min_score"]
        selected_depth = torch.gather(sample_depths, 1, best_index[:, None])[:, 0]
        near_peak = (sample_depths - selected_depth[:, None]).abs() <= 3.0 * voxel
        local_weight = score * near_peak.float()
        weight_sum = local_weight.sum(dim=1).clamp_min(1e-8)
        mean_depth = (local_weight * sample_depths).sum(dim=1) / weight_sum
        depth_variance = (local_weight * (sample_depths - mean_depth[:, None]).square()).sum(
            dim=1
        ) / weight_sum
        sigma_ray = depth_variance.sqrt().clamp(0.25 * voxel, 2.0 * voxel)
        placed_indices = retained_indices[placed]
        arrays[f"view={view}/valid_ray"] = valid_ray
        arrays[f"view={view}/sample_depths"] = sample_depths
        arrays[f"view={view}/in_grid"] = in_grid
        arrays[f"view={view}/voxel_indices"] = voxel_indices
        arrays[f"view={view}/score"] = score
        arrays[f"view={view}/best_score"] = best_score
        arrays[f"view={view}/best_index"] = best_index
        arrays[f"view={view}/placed"] = placed
        arrays[f"view={view}/selected_depth"] = selected_depth
        arrays[f"view={view}/depth_variance"] = depth_variance
        arrays[f"view={view}/sigma_ray"] = sigma_ray
        arrays[f"view={view}/placed_indices"] = placed_indices
        if placed_indices.numel() == 0:
            continue
        keys.extend(source_key(seed, view, int(index)) for index in placed_indices.tolist())
        sub = Gaussians2D(
            xy=selected.xy[placed],
            chol=selected.chol[placed],
            color=selected.color[placed],
            weight=selected.weight[placed],
        )
        parts.append(
            lift_view_at_depth(
                camera,
                sub,
                selected_depth[placed],
                sigma_ray[placed],
                0,
                opacity=torch.full_like(selected_depth[placed], 0.1),
            )
        )
    return (Gaussians3D.cat(parts) if parts else None), keys, arrays


def _archive_lift(
    archive: RawArchive,
    prefix: str,
    production: Gaussians3D,
    independent: Gaussians3D,
    keys: Sequence[tuple[int, int, int]],
    sidecar_arrays: Mapping[str, torch.Tensor],
) -> None:
    archive.add_gaussians3d(f"{prefix}/production", production)
    archive.add_gaussians3d(f"{prefix}/independent", independent)
    archive.add(f"{prefix}/source_keys", keys_array(keys))
    for name, value in sidecar_arrays.items():
        archive.add(f"{prefix}/sidecar/{name}", value)


def run_lift(
    *,
    seed: int,
    backend: str,
    views: Sequence[Gaussians2D],
    scene: SceneData,
    archive: RawArchive,
    prefix: str,
    coverage_maps: Sequence[torch.Tensor] | None = None,
) -> LiftResult:
    """Call one ordinary production lift and independently reconstruct its ordered output."""
    if backend == "Depth":
        independent, keys, sidecar = _depth_independent(seed, views, scene)
        if scene.gt_depths is None:
            raise ProtocolInvalid(archive.phase, "Depth lift requires depths")
        lifter = DepthLifter(backend=GroundTruthDepth(scene.gt_depths), **depth_config())
    elif backend == "Carve":
        if coverage_maps is None:
            with torch.no_grad():
                coverage_maps = [
                    (
                        scene.masks[view].to(gaussian.xy).float()
                        if scene.masks is not None
                        else render_gaussian_coverage_2d(
                            gaussian, camera.height, camera.width, row_chunk=64
                        )
                    )
                    for view, (gaussian, camera) in enumerate(
                        zip(views, scene.cameras, strict=True)
                    )
                ]
        independent, keys, sidecar = _carve_independent(seed, views, scene, coverage_maps)
        lifter = CarveLifter(**carve_config())
    else:
        raise ValueError(f"unknown backend: {backend}")
    try:
        production = lifter.lift(list(views), scene)
    except ValueError as error:
        raise ProtocolInvalid(archive.phase, f"{backend} production lift is empty") from error
    if independent is None or independent.n == 0 or production.n == 0:
        raise ProtocolInvalid(archive.phase, f"{backend} independent lift is empty")
    _archive_lift(archive, prefix, production, independent, keys, sidecar)
    parity = _assert_gaussian_parity(production, independent, phase=archive.phase)
    if production.n != len(keys):
        raise ProtocolInvalid(archive.phase, f"{backend} source-key count differs")
    archive.completed_lifts += 1
    return LiftResult(
        output=production,
        keys=keys,
        evidence={
            "backend": backend,
            "ordinary_production_call_count": 1,
            "output_count": production.n,
            "ordered_source_keys": [list(key) for key in keys],
            "production_digest": _gaussians3d_digest(production),
            "independent_digest": _gaussians3d_digest(independent),
            "parity_errors": parity,
            "parity_tolerances": {"atol": LIFT_ATOL, "rtol": LIFT_RTOL},
        },
        sidecar_arrays=sidecar,
    )


def mechanism_coverage_retention(
    prepared: Sequence[PreparedSeed],
    mechanism: Mapping[int, Mapping[str, Sequence[Gaussians2D]]],
    archive: RawArchive,
) -> tuple[dict[tuple[int, str, str], list[torch.Tensor]], dict[str, Any]]:
    archive.phase = "coverage_retention"
    coverage: dict[tuple[int, str, str], list[torch.Tensor]] = {}
    rows = []
    for item in prepared:
        for gauge in GAUGES:
            for representation in MECHANISM_REPRESENTATIONS:
                key = (item.seed, gauge, representation)
                views = mechanism[item.seed][f"{gauge}/{representation}"]
                coverage[key] = []
                for view, gaussian in enumerate(views):
                    with torch.no_grad():
                        value = render_gaussian_coverage_2d(
                            gaussian, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64
                        )
                    name = coverage_array_name(item.seed, gauge, representation, view)
                    archive.add(name, value)
                    coverage[key].append(value)
        for representation in MECHANISM_REPRESENTATIONS:
            identity_views = mechanism[item.seed][f"identity/{representation}"]
            identity_coverage = coverage[(item.seed, "identity", representation)]
            for gauge in TRANSFORMED_GAUGES:
                changed_views = mechanism[item.seed][f"{gauge}/{representation}"]
                changed_coverage = coverage[(item.seed, gauge, representation)]
                for view, (identity_map, changed_map, identity_g, changed_g) in enumerate(
                    zip(
                        identity_coverage,
                        changed_coverage,
                        identity_views,
                        changed_views,
                        strict=True,
                    )
                ):
                    delta = changed_map - identity_map
                    denominator = float(identity_map.double().abs().sum())
                    numerator = float(delta.double().abs().sum())
                    maximum = float(delta.abs().max())
                    crossings = (changed_map > 0.40) ^ (identity_map > 0.40)
                    identity_indices = torch.where(identity_g.weight > 0.05)[0]
                    changed_indices = torch.where(changed_g.weight > 0.05)[0]
                    archive.add(
                        f"coverage_delta/seed={item.seed}/gauge={gauge}/arm={representation}/view={view}",
                        delta,
                    )
                    archive.add(
                        f"coverage_crossing/seed={item.seed}/gauge={gauge}/arm={representation}/view={view}",
                        crossings,
                    )
                    archive.add(
                        f"retention/seed={item.seed}/gauge={gauge}/arm={representation}/view={view}/identity_components",
                        identity_indices,
                    )
                    archive.add(
                        f"retention/seed={item.seed}/gauge={gauge}/arm={representation}/view={view}/transformed_components",
                        changed_indices,
                    )
                    if denominator <= 0 or not math.isfinite(denominator):
                        raise ProtocolInvalid(
                            "coverage_retention", "coverage denominator is invalid"
                        )
                    row = {
                        "seed": item.seed,
                        "gauge": gauge,
                        "representation": representation,
                        "local_train_view_index": view,
                        "maximum_absolute_coverage_error": maximum,
                        "coverage_delta_l1_float64": numerator,
                        "coverage_identity_l1_float64": denominator,
                        "coverage_delta_over_identity": numerator / denominator,
                        "coverage_threshold_crossing_count": int(crossings.sum()),
                        "retained_count": int(identity_indices.numel()),
                        "retention_keys_exact": torch.equal(identity_indices, changed_indices),
                    }
                    rows.append(row)
                    if (
                        maximum > COVERAGE_MAX_ABS
                        or numerator / denominator > COVERAGE_REL_L1
                        or row["coverage_threshold_crossing_count"] != 0
                        or not row["retention_keys_exact"]
                    ):
                        raise ProtocolInvalid(
                            "coverage_retention", "coverage or retention invariance failed", row
                        )
    return coverage, {
        "passed": True,
        "completed_comparisons": len(rows),
        "thresholds": {
            "maximum_absolute_coverage_error": COVERAGE_MAX_ABS,
            "coverage_delta_over_identity": COVERAGE_REL_L1,
            "coverage_threshold": 0.40,
            "strict_retention_threshold": 0.05,
        },
        "rows": rows,
    }


def _render_views_raw(
    gaussians: Gaussians3D,
    cameras: Sequence[Any],
    archive: RawArchive,
    prefix: str,
) -> list[dict[str, torch.Tensor]]:
    renderer = frozen_renderer()
    outputs = []
    with torch.no_grad():
        for view, camera in enumerate(cameras):
            rendered = renderer.render(
                gaussians,
                camera,
                background=torch.zeros(3, dtype=gaussians.means.dtype),
                sh_degree=0,
            )
            row = {
                "color": rendered.color.detach(),
                "alpha": rendered.alpha.detach(),
                "depth": rendered.depth.detach(),
            }
            for field, value in row.items():
                archive.add(f"{prefix}/view={view}/{field}", value)
            outputs.append(row)
    return outputs


def _compare_mechanism_lifts(
    *,
    identity: LiftResult,
    changed: LiftResult,
    identity_renders: Sequence[Mapping[str, torch.Tensor]],
    changed_renders: Sequence[Mapping[str, torch.Tensor]],
    archive: RawArchive,
    prefix: str,
) -> dict[str, Any]:
    if identity.keys != changed.keys or identity.output.n != changed.output.n:
        raise ProtocolInvalid("mechanism_lifts", "transformed lift source keys/count differ")
    field_pairs = {
        "means": (changed.output.means, identity.output.means),
        "covariance": (changed.output.covariance(), identity.output.covariance()),
        "opacity": (changed.output.opacity, identity.output.opacity),
        "sh": (changed.output.sh, identity.output.sh),
    }
    field_errors = {}
    for name, (actual, expected) in field_pairs.items():
        field_errors[name] = float((actual - expected).abs().max())
        archive.add(f"{prefix}/field_delta/{name}", actual - expected)
        if not torch.allclose(actual, expected, atol=LIFT_ATOL, rtol=LIFT_RTOL):
            raise ProtocolInvalid("mechanism_lifts", f"lift {name} invariance failed")
    color_maximum = 0.0
    numerator = denominator = 0.0
    for view, (reference, candidate) in enumerate(
        zip(identity_renders, changed_renders, strict=True)
    ):
        color_delta = candidate["color"] - reference["color"]
        alpha_delta = candidate["alpha"] - reference["alpha"]
        depth_delta = candidate["depth"] - reference["depth"]
        archive.add(f"{prefix}/render_delta/view={view}/color", color_delta)
        archive.add(f"{prefix}/render_delta/view={view}/alpha", alpha_delta)
        archive.add(f"{prefix}/render_delta/view={view}/depth", depth_delta)
        color_maximum = max(color_maximum, float(color_delta.abs().max()))
        numerator += float(color_delta.double().abs().sum())
        denominator += float(reference["color"].double().abs().sum())
    if denominator <= 0 or not math.isfinite(denominator):
        raise ProtocolInvalid("mechanism_lifts", "lift-render denominator is invalid")
    ratio = numerator / denominator
    if color_maximum > SOURCE_RENDER_MAX_ABS or ratio > SOURCE_RENDER_REL_L1:
        raise ProtocolInvalid("mechanism_lifts", "lift-render color invariance failed")
    return {
        "ordered_source_keys_exact": True,
        "count": identity.output.n,
        "field_maximum_absolute_errors": field_errors,
        "render_color_maximum_absolute_error": color_maximum,
        "render_color_delta_l1_float64": numerator,
        "render_color_identity_l1_float64": denominator,
        "render_color_delta_over_identity": ratio,
    }


def run_mechanism_lifts(
    prepared: Sequence[PreparedSeed],
    mechanism: Mapping[int, Mapping[str, Sequence[Gaussians2D]]],
    coverage: Mapping[tuple[int, str, str], Sequence[torch.Tensor]],
    archive: RawArchive,
) -> dict[str, Any]:
    archive.phase = "mechanism_lifts"
    seed_records = []
    for item in prepared:
        lifts: dict[tuple[str, str, str], LiftResult] = {}
        renders: dict[tuple[str, str, str], list[dict[str, torch.Tensor]]] = {}
        for backend in BACKENDS:
            for representation in MECHANISM_REPRESENTATIONS:
                for gauge in GAUGES:
                    views = mechanism[item.seed][f"{gauge}/{representation}"]
                    prefix = (
                        f"mechanism_lift/seed={item.seed}/backend={backend}/"
                        f"gauge={gauge}/arm={representation}"
                    )
                    lift = run_lift(
                        seed=item.seed,
                        backend=backend,
                        views=views,
                        scene=item.scene,
                        archive=archive,
                        prefix=prefix,
                        coverage_maps=coverage[(item.seed, gauge, representation)],
                    )
                    if backend == "Carve":
                        lift.evidence["common_coverage_arrays"] = [
                            {
                                "name": coverage_array_name(item.seed, gauge, representation, view),
                                "raw_content_sha256": _tensor_digest(
                                    coverage[(item.seed, gauge, representation)][view]
                                ),
                            }
                            for view in range(len(TRAIN_INDICES))
                        ]
                    lifts[(backend, representation, gauge)] = lift
                    renders[(backend, representation, gauge)] = _render_views_raw(
                        lift.output,
                        item.scene.cameras,
                        archive,
                        f"{prefix}/train_render",
                    )
        comparisons = []
        for backend in BACKENDS:
            for representation in MECHANISM_REPRESENTATIONS:
                identity = lifts[(backend, representation, "identity")]
                for gauge in TRANSFORMED_GAUGES:
                    prefix = (
                        f"mechanism_comparison/seed={item.seed}/backend={backend}/"
                        f"gauge={gauge}/arm={representation}"
                    )
                    comparison = _compare_mechanism_lifts(
                        identity=identity,
                        changed=lifts[(backend, representation, gauge)],
                        identity_renders=renders[(backend, representation, "identity")],
                        changed_renders=renders[(backend, representation, gauge)],
                        archive=archive,
                        prefix=prefix,
                    )
                    comparisons.append(
                        {
                            "backend": backend,
                            "representation": representation,
                            "gauge": gauge,
                            **comparison,
                        }
                    )
        seed_records.append(
            {
                "seed": item.seed,
                "lifts": [
                    {
                        "backend": backend,
                        "representation": representation,
                        "gauge": gauge,
                        **lifts[(backend, representation, gauge)].evidence,
                    }
                    for backend in BACKENDS
                    for representation in MECHANISM_REPRESENTATIONS
                    for gauge in GAUGES
                ],
                "comparisons": comparisons,
            }
        )
        archive.completed_seeds += 1
    return {
        "passed": True,
        "ordinary_production_lift_count": archive.completed_lifts,
        "expected_production_lift_count": (
            len(PHASE_A_SEEDS) * len(BACKENDS) * len(MECHANISM_REPRESENTATIONS) * len(GAUGES)
        ),
        "seeds": seed_records,
    }


def execute_mechanism(archive: RawArchive) -> dict[str, Any]:
    """Execute Phase A in its globally fail-closed order."""
    add_config_raw(archive, "protocol/fit_config", frozen_fit_config())
    add_config_raw(archive, "protocol/depth_config", depth_config())
    add_config_raw(archive, "protocol/carve_config", carve_config())
    prepared = [prepare_seed(seed, "mechanism", archive) for seed in PHASE_A_SEEDS]
    equivalence = source_equivalence(prepared, archive)
    _, mechanism, _, candidates = construct_and_validate_candidates(prepared, archive)
    coverage, coverage_record = mechanism_coverage_retention(prepared, mechanism, archive)
    lifts = run_mechanism_lifts(prepared, mechanism, coverage, archive)
    if lifts["ordinary_production_lift_count"] != lifts["expected_production_lift_count"]:
        raise ProtocolInvalid("mechanism_lifts", "production lift call count differs")
    archive.phase = "complete"
    archive.completion_arrays()
    return {
        "artifact_type": PHASE_A_ARTIFACT_TYPE,
        "phase_a_pass": True,
        "preparations": [item.preparation for item in prepared],
        "source_equivalence": equivalence,
        "candidate_and_factorial_integrity": candidates,
        "coverage_and_retention": coverage_record,
        "production_lifts": lifts,
        "decision": {
            "phase_a_pass": True,
            "phase_b_authorized_by_result_alone": False,
            "independent_scientist_review_required": True,
            "default_change_authorized": False,
        },
    }


def construct_utility_candidates(
    prepared: Sequence[PreparedSeed], archive: RawArchive
) -> tuple[
    dict[int, list[dict[str, torch.Tensor]]],
    dict[int, dict[str, list[Gaussians2D]]],
    dict[str, Any],
]:
    """Form Phase-B arms from identity fits without constructing a transformed gauge."""
    archive.phase = "candidate_fields"
    fields_by_seed: dict[int, list[dict[str, torch.Tensor]]] = {}
    arms_by_seed: dict[int, dict[str, list[Gaussians2D]]] = {}
    records = []
    pooled_scalar = pooled_color = pooled_count = 0
    for item in prepared:
        if set(item.gauges) != {"identity"}:
            raise ProtocolInvalid("candidate_fields", "Phase B recreated a transformed gauge")
        fields_by_seed[item.seed] = []
        arms_by_seed[item.seed] = {arm: [] for arm in UTILITY_ARMS}
        scalar_changed = color_changed = count = 0
        for view, fitted in enumerate(item.fitted):
            fields = candidate_fields(fitted, item.scene.images[view])
            fields_by_seed[item.seed].append(fields)
            for name, value in fields.items():
                archive.add(f"candidate/seed={item.seed}/gauge=identity/view={view}/{name}", value)
            arms = utility_representations(fitted, fields)
            integrity = validate_factorial_integrity(
                fitted, arms, require_material_separation=False
            )
            scalar_changed += integrity["scalar_changed_count"]
            color_changed += integrity["color_changed_count"]
            count += integrity["component_count"]
            for arm, gaussian in arms.items():
                arms_by_seed[item.seed][arm].append(gaussian)
                archive.add_gaussians2d(
                    f"utility_arm/seed={item.seed}/arm={arm}/view={view}", gaussian
                )
        if scalar_changed / count < 0.10 or color_changed / count < 0.10:
            raise ProtocolInvalid("candidate_fields", "Phase-B arm identifiability failed")
        records.append(
            {
                "seed": item.seed,
                "component_count": count,
                "scalar_changed_count": scalar_changed,
                "scalar_changed_fraction": scalar_changed / count,
                "color_changed_count": color_changed,
                "color_changed_fraction": color_changed / count,
            }
        )
        pooled_scalar += scalar_changed
        pooled_color += color_changed
        pooled_count += count
    if pooled_scalar / pooled_count < 0.10 or pooled_color / pooled_count < 0.10:
        raise ProtocolInvalid("candidate_fields", "Phase-B pooled identifiability failed")
    return (
        fields_by_seed,
        arms_by_seed,
        {
            "passed": True,
            "seeds": records,
            "pooled": {
                "component_count": pooled_count,
                "scalar_changed_fraction": pooled_scalar / pooled_count,
                "color_changed_fraction": pooled_color / pooled_count,
            },
        },
    )


def _availability_by_view(keys: Sequence[tuple[int, int, int]], seed: int) -> list[list[int]]:
    result = [[] for _ in TRAIN_INDICES]
    for key_seed, view, component in keys:
        if key_seed != seed or view not in range(len(TRAIN_INDICES)):
            raise ProtocolInvalid("capacity", "lift emitted an invalid source key")
        result[view].append(component)
    for values in result:
        if values != sorted(values) or len(values) != len(set(values)):
            raise ProtocolInvalid("capacity", "lift source-key order is not canonical")
    return result


def _matched_initialization(
    lift: LiftResult,
    seed: int,
    selected_components: Sequence[Sequence[int]],
) -> tuple[Gaussians3D, list[tuple[int, int, int]]]:
    lookup = {key: index for index, key in enumerate(lift.keys)}
    selected_keys = [
        source_key(seed, view, component)
        for view, components in enumerate(selected_components)
        for component in components
    ]
    if len(selected_keys) != len(set(selected_keys)) or any(
        key not in lookup for key in selected_keys
    ):
        raise ProtocolInvalid("capacity", "selected keys are unavailable or duplicated")
    indices = torch.as_tensor(
        [lookup[key] for key in selected_keys],
        dtype=torch.long,
        device=lift.output.means.device,
    )
    matched = lift.output.subset(indices).detach()
    if matched.n != len(selected_keys):
        raise ProtocolInvalid("capacity", "matched model count differs")
    return matched, selected_keys


def _archive_train_history(archive: RawArchive, prefix: str, history: Mapping[str, Any]) -> None:
    archive.add(f"{prefix}/loss", np.asarray(history["loss"], dtype=np.float64))
    archive.add(
        f"{prefix}/sampled_train_views",
        np.asarray(history["sampled_train_views"], dtype=np.int64),
    )
    for name in ("psnr", "elapsed", "n_gaussians", "active_sh_degree"):
        rows = np.asarray(history[name], dtype=np.float64).reshape(-1, 2)
        archive.add(f"{prefix}/{name}/steps", rows[:, 0].astype(np.int64))
        dtype = np.int64 if name in {"n_gaussians", "active_sh_degree"} else np.float64
        archive.add(f"{prefix}/{name}/values", rows[:, 1].astype(dtype))
    terms = history["loss_terms"]
    for name in ("l1", "alpha", "opacity_reg", "scale_reg"):
        archive.add(
            f"{prefix}/loss_terms/{name}",
            np.asarray([row[name] for row in terms], dtype=np.float64),
        )


def _checkpoint_train_diagnostics(
    *,
    seed: int,
    backend: str,
    arm: str,
    scene: SceneData,
    checkpoints: Mapping[int, Gaussians3D],
    archive: RawArchive,
) -> list[dict[str, Any]]:
    records = []
    for step in TRAIN_CHECKPOINTS:
        model = checkpoints[step]
        prefix = f"utility_checkpoint/seed={seed}/backend={backend}/arm={arm}/step={step}"
        archive.add_gaussians3d(f"{prefix}/model", model)
        renders = _render_views_raw(model, scene.cameras, archive, f"{prefix}/train_render")
        metrics = []
        for view, (rendered, target) in enumerate(zip(renders, scene.images, strict=True)):
            values = image_metrics(rendered["color"].clamp(0, 1), target.clamp(0, 1), None)
            metrics.append(
                {
                    "view": view,
                    "psnr": values["psnr"],
                    "ssim": values["ssim"],
                }
            )
        records.append(
            {
                "step": step,
                "count": model.n,
                "mean_train_psnr": sum(row["psnr"] for row in metrics) / len(metrics),
                "mean_train_ssim": sum(row["ssim"] for row in metrics) / len(metrics),
                "per_view": metrics,
            }
        )
    return records


def refine_utility_model(
    *,
    seed: int,
    backend: str,
    arm: str,
    scene: SceneData,
    natural: Gaussians3D,
    matched: Gaussians3D,
    selected_keys: list[tuple[int, int, int]],
    archive: RawArchive,
) -> UtilityModel:
    archive.phase = "refinement"
    config = frozen_train_config(seed, backend)
    prefix = f"utility_model/seed={seed}/backend={backend}/arm={arm}"
    add_config_raw(archive, f"{prefix}/train_config", config)
    checkpoints: dict[int, Gaussians3D] = {0: matched.detach()}

    def checkpoint_callback(snapshot: Gaussians3D, step: int) -> None:
        if step not in TRAIN_CHECKPOINTS[1:] or step in checkpoints:
            raise ProtocolInvalid("refinement", f"unexpected or repeated checkpoint {step}")
        checkpoints[step] = snapshot.detach()

    try:
        final, history = Trainer(config).train(
            scene,
            matched,
            checkpoint_callback=checkpoint_callback,
        )
    except RuntimeError as error:
        if str(error).startswith("training render contains non-finite "):
            raise ProtocolInvalid(
                "refinement",
                str(error),
                {"seed": seed, "backend": backend, "arm": arm},
            ) from error
        raise
    if set(checkpoints) != set(TRAIN_CHECKPOINTS):
        raise ProtocolInvalid("refinement", "checkpoint set differs from frozen schedule")
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        if not torch.equal(getattr(final, field), getattr(checkpoints[120], field)):
            raise ProtocolInvalid("refinement", "step-120 checkpoint differs from final model")
    schedule = np.asarray(history["sampled_train_views"], dtype=np.int64)
    expected = expected_schedule(seed, backend)
    archive.add(f"{prefix}/expected_schedule", expected)
    archive.add(f"{prefix}/recorded_schedule", schedule)
    if not np.array_equal(schedule, expected) or set(schedule.tolist()) != set(range(9)):
        raise ProtocolInvalid("refinement", "trainer schedule failed exact contract")
    optimizer_rows = np.asarray(history["n_gaussians"], dtype=np.int64).reshape(-1, 2)
    checkpoint_counts = np.asarray(
        [[step, checkpoints[step].n] for step in TRAIN_CHECKPOINTS], dtype=np.int64
    )
    archive.add(f"{prefix}/optimizer_count_vector", optimizer_rows)
    archive.add(f"{prefix}/checkpoint_count_vector", checkpoint_counts)
    if (
        final.n != matched.n
        or not bool((optimizer_rows[:, 1] == matched.n).all())
        or not bool((checkpoint_counts[:, 1] == matched.n).all())
    ):
        raise ProtocolInvalid("refinement", "fixed-topology count trajectory changed")
    _archive_train_history(archive, f"{prefix}/history", history)
    checkpoint_records = _checkpoint_train_diagnostics(
        seed=seed,
        backend=backend,
        arm=arm,
        scene=scene,
        checkpoints=checkpoints,
        archive=archive,
    )
    archive.add_gaussians3d(f"{prefix}/natural", natural)
    archive.add_gaussians3d(f"{prefix}/matched", matched)
    archive.add_gaussians3d(f"{prefix}/final", final)
    archive.add(f"{prefix}/selected_source_keys", keys_array(selected_keys))
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        archive.add(
            f"{prefix}/final_parameter_delta/{field}",
            getattr(final, field) - getattr(matched, field),
        )
    archive.add(
        f"{prefix}/final_parameter_delta/covariance",
        final.covariance() - matched.covariance(),
    )
    archive.completed_models += 1
    return UtilityModel(
        natural=natural,
        matched=matched,
        final=final,
        keys=selected_keys,
        checkpoints=checkpoints,
        history=history,
        evidence={
            "natural_count": natural.n,
            "matched_count": matched.n,
            "final_count": final.n,
            "selected_source_keys": [list(key) for key in selected_keys],
            "expected_schedule_sha256": raw_content_sha256(expected),
            "recorded_schedule_sha256": raw_content_sha256(schedule),
            "all_training_views_sampled": True,
            "optimizer_count_vector": optimizer_rows.tolist(),
            "checkpoint_count_vector": checkpoint_counts.tolist(),
            "checkpoint_train_diagnostics": checkpoint_records,
            "initial_digest": _gaussians3d_digest(matched),
            "final_digest": _gaussians3d_digest(final),
        },
    )


def run_utility_preunlock(
    prepared: Sequence[PreparedSeed],
    fields_by_seed: Mapping[int, Sequence[Mapping[str, torch.Tensor]]],
    arms_by_seed: Mapping[int, Mapping[str, Sequence[Gaussians2D]]],
    archive: RawArchive,
) -> tuple[dict[tuple[int, str, str], UtilityModel], dict[str, Any]]:
    # Pass 1 is global and outcome-free: every lift/parity/capacity/matched-initialization gate
    # must pass for all six seed/backend cells before the first optimization step is allowed.
    natural_lifts: dict[tuple[int, str, str], LiftResult] = {}
    matched_initials: dict[tuple[int, str, str], Gaussians3D] = {}
    selected_keys_by_model: dict[tuple[int, str, str], list[tuple[int, int, int]]] = {}
    capacity_by_cell: dict[tuple[int, str], dict[str, Any]] = {}
    pre_backend_records: dict[tuple[int, str], dict[str, Any]] = {}
    for item in prepared:
        archive.phase = "utility_lifts"
        rho_by_view = [
            integrated_mass_rho(item.fitted[view], fields_by_seed[item.seed][view]["m"])
            for view in range(len(TRAIN_INDICES))
        ]
        for view, rho in enumerate(rho_by_view):
            archive.add(f"capacity/seed={item.seed}/view={view}/rho", rho)
        arm_coverage: dict[str, list[torch.Tensor]] = {}
        for arm in UTILITY_ARMS:
            arm_coverage[arm] = []
            for view, gaussian in enumerate(arms_by_seed[item.seed][arm]):
                with torch.no_grad():
                    coverage = render_gaussian_coverage_2d(
                        gaussian, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64
                    )
                archive.add(coverage_array_name(item.seed, "identity", arm, view), coverage)
                arm_coverage[arm].append(coverage)
        for backend in BACKENDS:
            natural: dict[str, LiftResult] = {}
            availability: dict[str, list[list[int]]] = {}
            for arm in UTILITY_ARMS:
                prefix = f"utility_lift/seed={item.seed}/backend={backend}/arm={arm}"
                natural[arm] = run_lift(
                    seed=item.seed,
                    backend=backend,
                    views=arms_by_seed[item.seed][arm],
                    scene=item.scene,
                    archive=archive,
                    prefix=prefix,
                    coverage_maps=arm_coverage[arm],
                )
                if backend == "Carve":
                    natural[arm].evidence["common_coverage_arrays"] = [
                        {
                            "name": coverage_array_name(item.seed, "identity", arm, view),
                            "raw_content_sha256": _tensor_digest(arm_coverage[arm][view]),
                        }
                        for view in range(len(TRAIN_INDICES))
                    ]
                availability[arm] = _availability_by_view(natural[arm].keys, item.seed)
                natural_lifts[(item.seed, backend, arm)] = natural[arm]
            archive.phase = "capacity"
            capacity = match_exact_capacity(
                seed=item.seed,
                backend=backend,
                availability_by_arm_view=availability,
                rho_by_view=rho_by_view,
            )
            capacity_by_cell[(item.seed, backend)] = capacity
            archive.add(
                f"capacity/seed={item.seed}/backend={backend}/per_view_quotas",
                np.asarray(capacity["per_view_quotas"], dtype=np.int64),
            )
            archive.add(
                f"capacity/seed={item.seed}/backend={backend}/total_quota",
                np.asarray(capacity["total_quota"], dtype=np.int64),
            )
            for view in range(len(TRAIN_INDICES)):
                archive.add(
                    f"capacity/seed={item.seed}/backend={backend}/view={view}/tie_break_digest_bytes",
                    np.stack(
                        [
                            np.frombuffer(
                                rho_tie_break(item.seed, backend, view, component),
                                dtype=np.uint8,
                            )
                            for component in range(COMPONENTS_PER_VIEW)
                        ]
                    ),
                )
            for arm in UTILITY_ARMS:
                for view in range(len(TRAIN_INDICES)):
                    available_mask = np.zeros(COMPONENTS_PER_VIEW, dtype=np.bool_)
                    available_mask[availability[arm][view]] = True
                    selected_mask = np.zeros(COMPONENTS_PER_VIEW, dtype=np.bool_)
                    selected_mask[capacity["selected_components"][arm][view]] = True
                    prefix = f"capacity/seed={item.seed}/backend={backend}/arm={arm}/view={view}"
                    archive.add(f"{prefix}/availability", available_mask)
                    archive.add(
                        f"{prefix}/ranked_components",
                        np.asarray(capacity["ranked_components"][arm][view], dtype=np.int64),
                    )
                    archive.add(f"{prefix}/selected", selected_mask)
                    archive.add(
                        f"{prefix}/selected_components",
                        np.asarray(capacity["selected_components"][arm][view], dtype=np.int64),
                    )
                matched, chosen_keys = _matched_initialization(
                    natural[arm],
                    item.seed,
                    capacity["selected_components"][arm],
                )
                key = (item.seed, backend, arm)
                matched_initials[key] = matched
                selected_keys_by_model[key] = chosen_keys
                archive.add_gaussians3d(
                    f"capacity/seed={item.seed}/backend={backend}/arm={arm}/matched_initial",
                    matched,
                )
                archive.add(
                    f"capacity/seed={item.seed}/backend={backend}/arm={arm}/matched_source_keys",
                    keys_array(chosen_keys),
                )
            expected_count = capacity["total_quota"]
            if any(
                matched_initials[(item.seed, backend, arm)].n != expected_count
                for arm in UTILITY_ARMS
            ):
                raise ProtocolInvalid("capacity", "cross-arm matched counts differ")
            overlaps = {}
            for left_index, left in enumerate(UTILITY_ARMS):
                for right in UTILITY_ARMS[left_index + 1 :]:
                    left_set = set(natural[left].keys)
                    right_set = set(natural[right].keys)
                    intersection = left_set & right_set
                    union = left_set | right_set
                    overlaps[f"{left}__vs__{right}"] = {
                        "intersection_count": len(intersection),
                        "union_count": len(union),
                        "jaccard": len(intersection) / len(union) if union else 1.0,
                    }
            pre_backend_records[(item.seed, backend)] = {
                "backend": backend,
                "natural_lifts": {arm: natural[arm].evidence for arm in UTILITY_ARMS},
                "natural_counts_by_view": {
                    arm: [len(values) for values in availability[arm]] for arm in UTILITY_ARMS
                },
                "per_view_quotas": capacity["per_view_quotas"],
                "matched_total_count": expected_count,
                "natural_source_set_overlaps": overlaps,
            }
    expected_lifts = len(PHASE_B_SEEDS) * len(BACKENDS) * len(UTILITY_ARMS)
    if archive.completed_lifts != expected_lifts or len(matched_initials) != expected_lifts:
        raise ProtocolInvalid(
            "capacity",
            "global lift/matched-initialization completeness count differs before refinement",
            {
                "completed_lifts": archive.completed_lifts,
                "expected_lifts": expected_lifts,
                "matched_initializations": len(matched_initials),
                "expected_matched_initializations": expected_lifts,
            },
        )
    if len(capacity_by_cell) != len(PHASE_B_SEEDS) * len(BACKENDS):
        raise ProtocolInvalid("capacity", "global capacity-cell count differs before refinement")
    if any(item.heldout is None or item.heldout.unlocked for item in prepared):
        raise ProtocolInvalid("capacity", "held-out guard opened before refinement")

    # Pass 2 starts only after every outcome-free gate above has passed globally.
    models: dict[tuple[int, str, str], UtilityModel] = {}
    seed_records = []
    for item in prepared:
        backend_records = []
        for backend in BACKENDS:
            backend_models: dict[str, UtilityModel] = {}
            for arm in UTILITY_ARMS:
                key = (item.seed, backend, arm)
                model = refine_utility_model(
                    seed=item.seed,
                    backend=backend,
                    arm=arm,
                    scene=item.scene,
                    natural=natural_lifts[key].output,
                    matched=matched_initials[key],
                    selected_keys=selected_keys_by_model[key],
                    archive=archive,
                )
                models[key] = model
                backend_models[arm] = model
            expected_count = capacity_by_cell[(item.seed, backend)]["total_quota"]
            if any(
                model.matched.n != expected_count or model.final.n != expected_count
                for model in backend_models.values()
            ):
                raise ProtocolInvalid("refinement", "cross-arm matched counts differ")
            expected = expected_schedule(item.seed, backend)
            if any(
                not np.array_equal(model.history["sampled_train_views"], expected)
                for model in backend_models.values()
            ):
                raise ProtocolInvalid("refinement", "cross-arm schedules differ")
            backend_record = dict(pre_backend_records[(item.seed, backend)])
            backend_record["models"] = {arm: backend_models[arm].evidence for arm in UTILITY_ARMS}
            backend_records.append(backend_record)
        archive.completed_seeds += 1
        seed_records.append({"seed": item.seed, "backends": backend_records})

    expected_models = expected_lifts
    if archive.completed_models != expected_models or archive.completed_seeds != len(PHASE_B_SEEDS):
        raise ProtocolInvalid(
            "refinement",
            "pre-held-out model/seed completeness count differs",
            {
                "completed_models": archive.completed_models,
                "expected_models": expected_models,
                "completed_seeds": archive.completed_seeds,
                "expected_seeds": len(PHASE_B_SEEDS),
            },
        )
    # Freeze model/source/schedule digests before the first held-out accessor can succeed.
    preunlock_digests = {
        f"seed={seed}/backend={backend}/arm={arm}": {
            "natural": _gaussians3d_digest(models[(seed, backend, arm)].natural),
            "matched": _gaussians3d_digest(models[(seed, backend, arm)].matched),
            "final": _gaussians3d_digest(models[(seed, backend, arm)].final),
            "source_keys": raw_content_sha256(keys_array(models[(seed, backend, arm)].keys)),
            "schedule": raw_content_sha256(
                np.asarray(
                    models[(seed, backend, arm)].history["sampled_train_views"],
                    dtype=np.int64,
                )
            ),
        }
        for seed in PHASE_B_SEEDS
        for backend in BACKENDS
        for arm in UTILITY_ARMS
    }
    if any(item.heldout is None or item.heldout.unlocked for item in prepared):
        raise ProtocolInvalid("refinement", "held-out guard opened before global unlock")
    return models, {
        "preunlock_validity_pass": True,
        "ordinary_production_lift_count": archive.completed_lifts,
        "final_model_count": archive.completed_models,
        "seed_records": seed_records,
        "preunlock_digests": preunlock_digests,
    }


def _heldout_render_metrics(
    *,
    seed: int,
    backend: str,
    arm: str,
    state: str,
    model: Gaussians3D,
    guard: HeldOutGuard,
    extent: float,
    archive: RawArchive,
) -> dict[str, Any]:
    renderer = frozen_renderer()
    cameras = guard.cameras
    images = guard.images
    depths = guard.depths
    original_indices = guard.original_indices
    per_camera = []
    with torch.no_grad():
        for local, (camera, target, target_depth, original) in enumerate(
            zip(cameras, images, depths, original_indices, strict=True)
        ):
            rendered = renderer.render(
                model,
                camera,
                background=torch.zeros(3, dtype=model.means.dtype),
                sh_degree=0,
            )
            prefix = (
                f"heldout/seed={seed}/backend={backend}/arm={arm}/state={state}/view={original}"
            )
            archive.add(f"{prefix}/color", rendered.color)
            archive.add(f"{prefix}/alpha", rendered.alpha)
            archive.add(f"{prefix}/accumulated_depth", rendered.depth)
            prediction = rendered.color.clamp(0, 1)
            target_clamped = target.clamp(0, 1)
            archive.add(f"{prefix}/clamped_color", prediction)
            mse = float((prediction.double() - target_clamped.double()).square().mean())
            values = image_metrics(prediction, target_clamped, mask=None)
            target_valid = target_depth > 0.05
            predicted_depth = rendered.depth / rendered.alpha.clamp_min(1e-6)
            normalized_depth_rmse = float(
                (
                    (predicted_depth[target_valid] - target_depth[target_valid])
                    .double()
                    .square()
                    .mean()
                ).sqrt()
                / extent
            )
            depth_coverage = float((rendered.alpha[target_valid] > 0.05).float().mean())
            archive.add(f"{prefix}/target_valid_depth_mask", target_valid)
            archive.add(f"{prefix}/predicted_depth", predicted_depth)
            archive.add(f"{prefix}/mse_float64", np.asarray(mse, dtype=np.float64))
            archive.add(f"{prefix}/psnr", np.asarray(values["psnr"], dtype=np.float64))
            archive.add(f"{prefix}/ssim", np.asarray(values["ssim"], dtype=np.float64))
            archive.add(
                f"{prefix}/normalized_depth_rmse",
                np.asarray(normalized_depth_rmse, dtype=np.float64),
            )
            archive.add(
                f"{prefix}/target_depth_coverage",
                np.asarray(depth_coverage, dtype=np.float64),
            )
            per_camera.append(
                {
                    "local_heldout_index": local,
                    "original_view_index": original,
                    "mse_float64": mse,
                    "psnr": values["psnr"],
                    "ssim": values["ssim"],
                    "alpha_sum_float64": float(rendered.alpha.double().sum()),
                    "normalized_depth_rmse": normalized_depth_rmse,
                    "target_depth_coverage_alpha_gt_0_05": depth_coverage,
                }
            )
    return {
        "per_camera": per_camera,
        "mean_psnr": sum(row["psnr"] for row in per_camera) / len(per_camera),
        "mean_ssim": sum(row["ssim"] for row in per_camera) / len(per_camera),
    }


def report_heldout(
    prepared: Sequence[PreparedSeed],
    models: Mapping[tuple[int, str, str], UtilityModel],
    archive: RawArchive,
) -> tuple[dict[str, Any], bool]:
    archive.phase = "heldout"
    # One global transition: all 24 models and their gates exist before any payload opens.
    for item in prepared:
        if item.heldout is None:
            raise ProtocolInvalid("heldout", "held-out guard is absent")
        item.heldout.unlock()
    records: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    all_final_psnr_at_least_10 = True
    for item in prepared:
        assert item.heldout is not None
        guard = item.heldout
        center, extent = item.scene.center_and_extent()
        archive.add(f"heldout_target/seed={item.seed}/center", center)
        archive.add(f"heldout_target/seed={item.seed}/extent", np.asarray(extent, dtype=np.float64))
        for local, (image, camera, depth, original) in enumerate(
            zip(
                guard.images,
                guard.cameras,
                guard.depths,
                guard.original_indices,
                strict=True,
            )
        ):
            prefix = f"heldout_target/seed={item.seed}/view={original}"
            archive.add(f"{prefix}/image", image)
            archive.add(f"{prefix}/depth", depth)
            _archive_camera(archive, f"{prefix}/camera", camera)
            archive.add(f"{prefix}/local_index", np.asarray(local, dtype=np.int64))
        records[item.seed] = {}
        for backend in BACKENDS:
            records[item.seed][backend] = {}
            for arm in UTILITY_ARMS:
                model = models[(item.seed, backend, arm)]
                states = {
                    "natural_unpruned": model.natural,
                    "matched_initial": model.matched,
                    "final": model.final,
                }
                records[item.seed][backend][arm] = {}
                for state, gaussian in states.items():
                    metrics = _heldout_render_metrics(
                        seed=item.seed,
                        backend=backend,
                        arm=arm,
                        state=state,
                        model=gaussian,
                        guard=guard,
                        extent=extent,
                        archive=archive,
                    )
                    records[item.seed][backend][arm][state] = metrics
                    if state == "final" and any(
                        row["psnr"] < 10.0 for row in metrics["per_camera"]
                    ):
                        all_final_psnr_at_least_10 = False
    return {"seeds": records}, all_final_psnr_at_least_10


def reduce_utility_results(
    reporting: Mapping[str, Any],
    *,
    validity_gates_pass: bool,
) -> dict[str, Any]:
    archive_records = reporting["seeds"]
    estimands: dict[str, Any] = {backend: {} for backend in BACKENDS}
    differences = {metric: {backend: [] for backend in BACKENDS} for metric in ("psnr", "ssim")}
    estimand_names = (
        "scalar_main_effect",
        "color_main_effect",
        "interaction",
        "full_candidate_difference",
    )
    effect_vectors: dict[str, dict[str, dict[str, list[float]]]] = {
        metric: {backend: {name: [] for name in estimand_names} for backend in BACKENDS}
        for metric in ("psnr", "ssim")
    }
    for backend in BACKENDS:
        for seed in PHASE_B_SEEDS:
            psnr_values = {
                arm: archive_records[seed][backend][arm]["final"]["mean_psnr"]
                for arm in UTILITY_ARMS
            }
            ssim_values = {
                arm: archive_records[seed][backend][arm]["final"]["mean_ssim"]
                for arm in UTILITY_ARMS
            }
            psnr_effects = factorial_estimands(psnr_values)
            ssim_effects = factorial_estimands(ssim_values)
            estimands[backend][str(seed)] = {
                "final_mean_by_arm": {"psnr": psnr_values, "ssim": ssim_values},
                "psnr": psnr_effects,
                "ssim": ssim_effects,
            }
            differences["psnr"][backend].append(psnr_effects["full_candidate_difference"])
            differences["ssim"][backend].append(ssim_effects["full_candidate_difference"])
            for name in estimand_names:
                effect_vectors["psnr"][backend][name].append(psnr_effects[name])
                effect_vectors["ssim"][backend][name].append(ssim_effects[name])
    decisions = frozen_decisions(
        differences["psnr"],
        differences["ssim"],
        validity_gates_pass=validity_gates_pass,
    )
    summaries = {}
    for backend in BACKENDS:
        summaries[backend] = {}
        for metric in ("psnr", "ssim"):
            summaries[backend][metric] = {
                name: {
                    "seed_effects": values,
                    "mean": sum(values) / len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for name, values in effect_vectors[metric][backend].items()
            }
        summaries[backend]["psnr_material_drivers"] = {
            name: material_driver(effect_vectors["psnr"][backend][name])
            for name in ("scalar_main_effect", "color_main_effect", "interaction")
        }
    return {
        "estimands": estimands,
        "paired_summaries": summaries,
        "decisions": decisions,
    }


def execute_utility(archive: RawArchive) -> dict[str, Any]:
    add_config_raw(archive, "protocol/fit_config", frozen_fit_config())
    add_config_raw(archive, "protocol/depth_config", depth_config())
    add_config_raw(archive, "protocol/carve_config", carve_config())
    prepared = [prepare_seed(seed, "utility", archive) for seed in PHASE_B_SEEDS]
    fields, arms, integrity = construct_utility_candidates(prepared, archive)
    models, preunlock = run_utility_preunlock(prepared, fields, arms, archive)
    reporting, final_psnr_gate = report_heldout(prepared, models, archive)
    decision_validity = bool(preunlock["preunlock_validity_pass"] and final_psnr_gate)
    reduction = reduce_utility_results(reporting, validity_gates_pass=decision_validity)
    archive.phase = "complete"
    archive.completion_arrays()
    return {
        "artifact_type": PHASE_B_ARTIFACT_TYPE,
        "phase_b_valid": True,
        "execution_valid": True,
        "preparations": [item.preparation for item in prepared],
        "factorial_integrity": integrity,
        "pre_heldout": preunlock,
        "heldout_reporting": reporting,
        "reduction": reduction,
        "decision": {
            **reduction["decisions"],
            "decision_validity_gates_pass": decision_validity,
            "all_final_heldout_psnr_at_least_10_db": final_psnr_gate,
            "default_change_authorized": False,
            "independent_results_audit_required": True,
        },
    }


# ---------------------------------------------------------------------------
# Pickle-free raw serialization and exclusive append-only artifact writes.


def _preflight_absent(paths: Iterable[Path]) -> None:
    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise FileExistsError(f"prospective artifact paths already exist: {occupied}")


def _exclusive_write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest = sha256_file(path)
    if digest != sha256_bytes(payload):
        raise RuntimeError(f"exclusive write verification failed: {path}")
    return digest


def _exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = _exclusive_write_bytes(path, encoded)
    if canonical_json(strict_json_load(path)) != canonical_json(payload):
        raise RuntimeError(f"strict JSON round trip differs: {path}")
    return digest


def write_raw_sidecar(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Exclusively write and validate one uncompressed, pickle-free NPZ sidecar."""
    normalized = {name: little_endian_array(value) for name, value in arrays.items()}
    if len(normalized) != len(arrays):  # pragma: no cover - mapping invariant
        raise ValueError("raw sidecar contains duplicate logical names")
    for name, value in normalized.items():
        classification = nonfinite_classification(value)
        if bool(classification.any()):
            raise ValueError(f"raw sidecar contains non-finite values: {name}")
    manifest, collection = array_manifest(normalized)
    if raw_collection_sha256(manifest) != collection:
        raise RuntimeError("raw collection digest implementation disagrees")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez(handle, **normalized)
        handle.flush()
        os.fsync(handle.fileno())
    with ZipFile(path, "r") as zipped:
        if any(info.compress_type != ZIP_STORED for info in zipped.infolist()):
            raise RuntimeError("raw sidecar is compressed")
    with np.load(path, allow_pickle=False) as loaded:
        if sorted(loaded.files) != sorted(normalized):
            raise RuntimeError("raw sidecar logical names changed during serialization")
        for name, expected in normalized.items():
            actual = loaded[name]
            if (
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not np.array_equal(actual, expected)
            ):
                raise RuntimeError(f"raw sidecar round trip differs: {name}")
    return {
        "path": str(path.resolve()),
        "npz_sha256": sha256_file(path),
        "collection_sha256": collection,
        "array_count": len(normalized),
        "manifest": manifest,
        "format": "numpy.savez/uncompressed",
        "allow_pickle": False,
    }


def validate_raw_sidecar(path: Path, binding: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if sha256_file(path) != binding.get("npz_sha256"):
        raise ValueError("raw NPZ SHA-256 differs from its binding")
    with ZipFile(path, "r") as zipped:
        if any(info.compress_type != ZIP_STORED for info in zipped.infolist()):
            raise ValueError("raw sidecar is compressed")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as loaded:
        for name in loaded.files:
            arrays[name] = little_endian_array(loaded[name])
            if bool(nonfinite_classification(arrays[name]).any()):
                raise ValueError(f"raw sidecar contains non-finite values: {name}")
    manifest, collection = array_manifest(arrays)
    if (
        manifest != binding.get("manifest")
        or collection != binding.get("collection_sha256")
        or len(arrays) != binding.get("array_count")
    ):
        raise ValueError("raw semantic manifest or collection digest differs")
    return arrays


def claim_attempt_marker(
    marker: Path,
    payload: Mapping[str, Any],
    prospective_paths: Iterable[Path],
) -> dict[str, Any]:
    """Preflight every sibling, then claim the marker by exclusive creation."""
    paths = tuple(prospective_paths)
    _preflight_absent((marker, *paths))
    complete = dict(payload)
    complete.setdefault(
        "prospective_paths", [str(path.resolve()) for path in sorted(paths, key=str)]
    )
    complete.setdefault("resume_permitted", False)
    digest = _exclusive_write_json(marker, complete)
    return {
        "path": str(marker.resolve()),
        "sha256": digest,
        "payload_sha256": canonical_json_hash(complete),
        "payload": complete,
    }


def _validate_marker(binding: Mapping[str, Any]) -> None:
    path = Path(str(binding["path"]))
    if sha256_file(path) != binding["sha256"]:
        raise RuntimeError("once-only marker changed during execution")
    if canonical_json_hash(strict_json_load(path)) != binding["payload_sha256"]:
        raise RuntimeError("once-only marker payload changed during execution")


# ---------------------------------------------------------------------------
# Outcome-free implementation seal.


def seal_source_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                PREREGISTRATION,
                PREREGISTRATION_REVIEW,
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


SEALED_PATHS = seal_source_paths()
VERIFICATION_COMMANDS = (
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


def _implementation_review_record() -> dict[str, str]:
    path = ROOT / IMPLEMENTATION_REVIEW
    if not path.is_file():
        raise RuntimeError(f"implementation review is missing: {path}")
    text = path.read_text(encoding="utf-8")
    if re.search(r"(?m)^Verdict: PASS\s*$", text) is None:
        raise RuntimeError("implementation review lacks exact 'Verdict: PASS'")
    return {"path": str(IMPLEMENTATION_REVIEW), "sha256": sha256_file(path)}


def assert_implementation_ready() -> None:
    if not IMPLEMENTATION_COMPLETE or IMPLEMENTATION_GAPS:
        gaps = "; ".join(IMPLEMENTATION_GAPS) or "implementation completeness flag is false"
        raise RuntimeError(f"semantic-factorial implementation is incomplete: {gaps}")
    _implementation_review_record()


def _verify_preregistration() -> dict[str, str]:
    path = ROOT / PREREGISTRATION
    digest = sha256_file(path)
    if digest != PREREGISTRATION_SHA256:
        raise RuntimeError(f"preregistration drift: {digest}")
    review = ROOT / PREREGISTRATION_REVIEW
    if (
        not review.is_file()
        or re.search(r"(?m)^Verdict: PASS\s*$", review.read_text(encoding="utf-8")) is None
    ):
        raise RuntimeError("independent preregistration review is absent or not PASS")
    return {
        "path": str(PREREGISTRATION),
        "sha256": digest,
        "review_path": str(PREREGISTRATION_REVIEW),
        "review_sha256": sha256_file(review),
    }


def _source_snapshot(paths: Iterable[Path]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for relative in sorted(set(paths), key=str):
        absolute = ROOT / relative
        if not absolute.is_file():
            raise FileNotFoundError(f"sealed source is absent: {absolute}")
        hashes[str(relative)] = sha256_file(absolute)
        sizes[str(relative)] = absolute.stat().st_size
    return {
        "paths": sorted(hashes),
        "sha256": hashes,
        "byte_sizes": sizes,
        "collection_sha256": canonical_json_hash(
            [[name, hashes[name], sizes[name]] for name in sorted(hashes)]
        ),
    }


def _loaded_repository_sources() -> dict[str, Any]:
    paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(ROOT) and path.is_file():
            relative = path.relative_to(ROOT)
            if not relative.parts or relative.parts[0] != ".venv":
                paths.add(relative)
    missing = paths - set(SEALED_PATHS)
    if missing:
        raise RuntimeError(
            f"loaded repository sources are outside the seal: {sorted(map(str, missing))}"
        )
    return _source_snapshot(paths)


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _environment_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "python",
        "platform",
        "processor",
        "torch",
        "numpy",
        "cuda_visible_devices",
        "omp_num_threads",
        "mkl_num_threads",
        "torch_num_threads",
        "deterministic_algorithms",
    )
    return {key: value[key] for key in keys}


def _assert_official_environment(value: Mapping[str, Any]) -> None:
    expected = {
        "torch": EXPECTED_TORCH_VERSION,
        "cuda_visible_devices": "",
        "omp_num_threads": "4",
        "mkl_num_threads": "4",
        "torch_num_threads": 4,
        "deterministic_algorithms": True,
    }
    actual = {key: value[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"official environment mismatch: {actual!r} != {expected!r}")


def git_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
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
        absolute = ROOT / relative
        if absolute.is_file():
            untracked[str(relative)] = {
                "sha256": sha256_file(absolute),
                "byte_size": absolute.stat().st_size,
            }
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_binary_diff": tracked_diff,
        "tracked_binary_diff_sha256": sha256_bytes(tracked_diff.encode("utf-8")),
        "untracked_files": untracked,
        "untracked_collection_sha256": canonical_json_hash(untracked),
    }


def raw_schema_record() -> dict[str, Any]:
    return {
        "name": RAW_SCHEMA,
        "format": "uncompressed numpy.savez",
        "allow_pickle": False,
        "array_types": "numeric or boolean only",
        "logical_name_separator": "/",
        "manifest_digest_field": "raw_content_sha256",
        "content_digest": (
            "SHA256(little-endian dtype token || NUL || little-endian int64 shape || NUL || "
            "C-contiguous little-endian data)"
        ),
        "nullable": "numeric value array plus boolean defined mask; undefined value is zero",
        "nonfinite_classification": {"0": "finite", "1": "NaN", "2": "+Inf", "3": "-Inf"},
    }


def command_templates() -> dict[str, list[str]]:
    prefix = ["CUDA_VISIBLE_DEVICES=''", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4"]
    return {
        "seal": [
            *prefix,
            ".venv/bin/python",
            str(HARNESS),
            "seal",
            "--output",
            str(DEFAULT_SEAL.relative_to(ROOT)),
        ],
        "mechanism": [
            *prefix,
            ".venv/bin/python",
            str(HARNESS),
            "mechanism",
            "--seal",
            str(DEFAULT_SEAL.relative_to(ROOT)),
            "--output",
            "benchmarks/results/<fresh-UTC>_cpu_stage1_semantic_factorial_mechanism.json",
        ],
        "utility": [
            *prefix,
            ".venv/bin/python",
            str(HARNESS),
            "utility",
            "--seal",
            str(DEFAULT_SEAL.relative_to(ROOT)),
            "--phase-a",
            "benchmarks/results/<phase-a-UTC>_cpu_stage1_semantic_factorial_mechanism.json",
            "--phase-a-raw",
            "benchmarks/results/<phase-a-UTC>_cpu_stage1_semantic_factorial_mechanism_RAW.npz",
            "--phase-a-review",
            "benchmarks/results/<phase-a-UTC>_cpu_stage1_semantic_factorial_mechanism_SCIENTIST_REVIEW.json",
            "--output",
            "benchmarks/results/<fresh-UTC>_cpu_stage1_semantic_factorial_utility.json",
        ],
    }


def protocol_record() -> dict[str, Any]:
    return {
        "phase_a_seeds": list(PHASE_A_SEEDS),
        "phase_b_seeds": list(PHASE_B_SEEDS),
        "train_indices": list(TRAIN_INDICES),
        "held_out_indices": list(HELD_OUT_INDICES),
        "gauges": list(GAUGES),
        "mechanism_representations": list(MECHANISM_REPRESENTATIONS),
        "utility_arms": list(UTILITY_ARMS),
        "backends": list(BACKENDS),
        "fit_config": asdict(frozen_fit_config()),
        "depth_config": depth_config(),
        "carve_config": carve_config(),
        "train_configs": {
            backend: asdict(frozen_train_config(PHASE_B_SEEDS[0], backend)) for backend in BACKENDS
        },
        "raw_schema": raw_schema_record(),
        "phase_a_review_gates": list(PHASE_A_REVIEW_GATES),
        "command_templates": command_templates(),
        "artifact_rules": {
            "exclusive_creation": True,
            "resume": False,
            "retry": False,
            "valid_invalid_siblings_preflighted": True,
            "phase_b_requires_independent_machine_review": True,
            "python_socket_and_child_process_audit_guard_during_scientific_compute": True,
        },
    }


def _run_verification() -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    rows = []
    for command, literal in zip(VERIFICATION_COMMANDS, VERIFICATION_LITERAL_COMMANDS, strict=True):
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        row = {
            "command": list(command),
            "literal_command": literal,
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
        }
        rows.append(row)
        if completed.returncode != 0:
            raise RuntimeError(
                f"seal verification failed: {literal}\n{completed.stdout}\n{completed.stderr}"
            )
    return {"passed": True, "commands": rows}


def _seal_snapshot() -> dict[str, Any]:
    assert_implementation_ready()
    return {
        "preregistration": _verify_preregistration(),
        "implementation_review": _implementation_review_record(),
        "sealed_sources": _source_snapshot(SEALED_PATHS),
        "loaded_repository_sources": _loaded_repository_sources(),
        "git": git_metadata(),
        "protocol": protocol_record(),
    }


def create_seal() -> dict[str, Any]:
    environment = environment_metadata()
    _assert_official_environment(environment)
    before = _seal_snapshot()
    verification = _run_verification()
    after = _seal_snapshot()
    if before != after:
        raise RuntimeError("seal-bound source/git/protocol snapshot drifted during verification")
    return {
        "artifact_type": SEAL_ARTIFACT_TYPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment,
        **before,
        "verification": verification,
        "snapshot_sha256": canonical_json_hash(before),
        "command": [sys.executable, *sys.argv],
    }


def _validate_verification_record(value: Mapping[str, Any]) -> None:
    rows = value.get("commands", [])
    if not value.get("passed") or [row.get("command") for row in rows] != [
        list(command) for command in VERIFICATION_COMMANDS
    ]:
        raise RuntimeError("seal lacks the exact passing verification sequence")
    if [row.get("literal_command") for row in rows] != list(VERIFICATION_LITERAL_COMMANDS) or any(
        row.get("returncode") != 0 for row in rows
    ):
        raise RuntimeError("seal verification literal commands or return codes differ")
    for row in rows:
        if row.get("stdout_sha256") != sha256_bytes(row.get("stdout", "").encode("utf-8")):
            raise RuntimeError("seal stdout digest differs")
        if row.get("stderr_sha256") != sha256_bytes(row.get("stderr", "").encode("utf-8")):
            raise RuntimeError("seal stderr digest differs")


def load_and_verify_seal(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_SEAL.resolve():
        raise ValueError("scientific commands require the sole preregistered seal path")
    payload = strict_json_load(path)
    if payload.get("artifact_type") != SEAL_ARTIFACT_TYPE:
        raise ValueError("seal artifact type differs")
    assert_implementation_ready()
    if payload.get("preregistration") != _verify_preregistration():
        raise RuntimeError("seal preregistration binding differs")
    if payload.get("implementation_review") != _implementation_review_record():
        raise RuntimeError("seal implementation-review binding differs")
    current_sources = _source_snapshot(SEALED_PATHS)
    if payload.get("sealed_sources") != current_sources:
        raise RuntimeError("sealed repository source changed")
    if payload.get("protocol") != protocol_record():
        raise RuntimeError("sealed protocol/config/artifact rules changed")
    _validate_verification_record(payload.get("verification", {}))
    environment = environment_metadata()
    _assert_official_environment(environment)
    if _environment_fingerprint(payload["environment"]) != _environment_fingerprint(environment):
        raise RuntimeError("runtime environment differs from seal")
    snapshot = {
        "preregistration": payload["preregistration"],
        "implementation_review": payload["implementation_review"],
        "sealed_sources": payload["sealed_sources"],
        "loaded_repository_sources": payload["loaded_repository_sources"],
        "git": payload["git"],
        "protocol": payload["protocol"],
    }
    if payload.get("snapshot_sha256") != canonical_json_hash(snapshot):
        raise RuntimeError("seal snapshot digest differs")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_collection_sha256": current_sources["collection_sha256"],
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Phase-A independent-review authorization for Phase B.


def _phase_a_sibling_paths(phase_a: Path) -> dict[str, Path]:
    name = phase_a.name
    match = re.fullmatch(r"(\d{8}T\d{6}Z_cpu_stage1_semantic_factorial_mechanism)\.json", name)
    if phase_a.resolve().parent != (ROOT / "benchmarks/results").resolve() or match is None:
        raise ValueError("Phase-A JSON path is not a valid mechanism result")
    stem = match.group(1)
    return {
        "json": phase_a.resolve(),
        "raw": phase_a.with_name(f"{stem}_RAW.npz").resolve(),
        "audit": phase_a.with_name(f"{stem}_AUDIT.md").resolve(),
        "review": phase_a.with_name(f"{stem}_SCIENTIST_REVIEW.json").resolve(),
    }


def _require_utc_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} is not a UTC timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} is not explicitly UTC")
    return value


def _validate_phase_a_review_recomputation(
    review: Mapping[str, Any], raw_binding: Mapping[str, Any]
) -> None:
    raw = review.get("raw_archive_recomputation", {})
    required_true = (
        "loaded_with_allow_pickle_false",
        "npz_sha256_recomputed",
        "semantic_manifest_recomputed",
        "collection_sha256_recomputed",
        "all_raw_floating_arrays_finite",
    )
    if any(raw.get(name) is not True for name in required_true):
        raise ValueError("Phase-A review did not independently validate the raw archive")
    if raw.get("array_count") != raw_binding.get("array_count") or raw.get(
        "collection_sha256"
    ) != raw_binding.get("collection_sha256"):
        raise ValueError("Phase-A review raw recomputation summary differs from the result")
    gates = review.get("decisive_gate_recomputation", {})
    if set(gates) != set(PHASE_A_REVIEW_GATES):
        raise ValueError("Phase-A review decisive-gate schema differs")
    for name in PHASE_A_REVIEW_GATES:
        row = gates[name]
        if not isinstance(row, Mapping) or row.get("recomputed") is not True:
            raise ValueError(f"Phase-A review did not recompute gate: {name}")
        if row.get("passed") is not True:
            raise ValueError(f"Phase-A review gate did not pass: {name}")
        evidence = row.get("evidence")
        if not isinstance(evidence, Mapping) or not evidence:
            raise ValueError(f"Phase-A review gate lacks recomputed evidence: {name}")
        canonical_json(evidence)


def validate_phase_a_authorization(
    *,
    phase_a: Path,
    phase_a_raw: Path,
    phase_a_review: Path,
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    siblings = _phase_a_sibling_paths(phase_a)
    if phase_a_raw.resolve() != siblings["raw"] or phase_a_review.resolve() != siblings["review"]:
        raise ValueError("Phase-A raw/review paths do not match the Phase-A JSON namespace")
    if not siblings["audit"].is_file():
        raise FileNotFoundError("Phase-A independent audit Markdown is absent")
    result = strict_json_load(siblings["json"])
    if (
        result.get("artifact_type") != PHASE_A_ARTIFACT_TYPE
        or result.get("phase_a_pass") is not True
    ):
        raise ValueError("Phase-A artifact is not a valid passing mechanism result")
    marker_path = PHASE_A_ATTEMPT.resolve()
    if not marker_path.is_file():
        raise FileNotFoundError("Phase-A once-only attempt marker is absent")
    expected_artifact_bindings = {
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_path": seal["path"],
        "seal_sha256": seal["sha256"],
        "sealed_source_collection_sha256": seal["source_collection_sha256"],
        "harness_sha256": sha256_file(ROOT / HARNESS),
        "focused_tests_sha256": sha256_file(ROOT / FOCUSED_TESTS),
        "attempt_marker_path": str(marker_path),
        "attempt_marker_sha256": sha256_file(marker_path),
    }
    if result.get("artifact_bindings") != expected_artifact_bindings:
        raise ValueError("Phase-A result artifact bindings differ")
    marker = strict_json_load(marker_path)
    prospective = official_output_paths(siblings["json"], "mechanism")
    expected_prospective = [str(path.resolve()) for path in sorted(prospective.values(), key=str)]
    if (
        marker.get("artifact_type") != ATTEMPT_ARTIFACT_TYPE
        or marker.get("phase") != "mechanism"
        or marker.get("seal_path") != seal["path"]
        or marker.get("seal_sha256") != seal["sha256"]
        or marker.get("sealed_source_collection_sha256") != seal["source_collection_sha256"]
        or marker.get("prospective_paths") != expected_prospective
        or marker.get("resume_permitted") is not False
    ):
        raise ValueError("Phase-A attempt marker binding differs")
    if not prospective["valid_note"].is_file() or any(
        prospective[f"invalid_{suffix}"].exists() for suffix in ("json", "raw", "note")
    ):
        raise ValueError("Phase-A valid/invalid sibling routing differs")
    raw_binding = result.get("raw_archive", {})
    if Path(str(raw_binding.get("path", ""))).resolve() != siblings["raw"]:
        raise ValueError("Phase-A result raw path binding differs")
    validate_raw_sidecar(siblings["raw"], raw_binding)
    review = strict_json_load(siblings["review"])
    if review.get("verdict") != "PASS" or review.get("phase_b_authorized") is not True:
        raise ValueError("Phase-A scientist review does not authorize Phase B")
    if review.get("recomputed_phase_a_pass") is not True:
        raise ValueError("Phase-A scientist review did not recompute a passing global decision")
    if (
        not isinstance(review.get("reviewer_identity"), str)
        or not review["reviewer_identity"].strip()
    ):
        raise ValueError("Phase-A scientist review lacks reviewer identity")
    _require_utc_timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc")
    _validate_phase_a_review_recomputation(review, raw_binding)
    bindings = review.get("bindings", {})
    expected_bindings = {
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_sha256": seal["sha256"],
        "phase_a_json_sha256": sha256_file(siblings["json"]),
        "phase_a_raw_sha256": sha256_file(siblings["raw"]),
        "harness_sha256": sha256_file(ROOT / HARNESS),
        "focused_tests_sha256": sha256_file(ROOT / FOCUSED_TESTS),
        "audit_markdown_sha256": sha256_file(siblings["audit"]),
    }
    if bindings != expected_bindings:
        raise ValueError("Phase-A scientist review mutual artifact bindings differ")
    return {
        "paths": {name: str(path) for name, path in siblings.items()},
        "sha256": {
            "phase_a_json": expected_bindings["phase_a_json_sha256"],
            "phase_a_raw": expected_bindings["phase_a_raw_sha256"],
            "phase_a_review": sha256_file(siblings["review"]),
            "audit_markdown": expected_bindings["audit_markdown_sha256"],
        },
        "reviewer_identity": review["reviewer_identity"],
        "reviewed_at_utc": review["reviewed_at_utc"],
        "phase_b_authorized": True,
    }


def _verify_authorization_unchanged(authorization: Mapping[str, Any]) -> None:
    paths = authorization["paths"]
    expected = authorization["sha256"]
    actual = {
        "phase_a_json": sha256_file(Path(paths["json"])),
        "phase_a_raw": sha256_file(Path(paths["raw"])),
        "phase_a_review": sha256_file(Path(paths["review"])),
        "audit_markdown": sha256_file(Path(paths["audit"])),
    }
    if actual != expected:
        raise RuntimeError("Phase-A authorization input changed during Phase B")


# ---------------------------------------------------------------------------
# Bound phase execution and CLI.  Help/argument parsing never crosses a readiness boundary.


_NETWORK_AUDIT_HOOK_INSTALLED = False
_NETWORK_GUARD_ACTIVE = False
_NETWORK_GUARD_ARCHIVE: RawArchive | None = None


def _scientific_runtime_audit_hook(event: str, _args: tuple[Any, ...]) -> None:
    if not _NETWORK_GUARD_ACTIVE:
        return
    forbidden = event.startswith("socket.") or event in {
        "subprocess.Popen",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.system",
        "os.posix_spawn",
        "os.posix_spawnp",
        "pty.spawn",
    }
    if forbidden:
        phase = "preflight" if _NETWORK_GUARD_ARCHIVE is None else _NETWORK_GUARD_ARCHIVE.phase
        raise ProtocolInvalid(
            phase,
            f"network access or child process attempted during scientific phase: {event}",
        )


def _install_scientific_runtime_audit_hook() -> None:
    global _NETWORK_AUDIT_HOOK_INSTALLED
    if not _NETWORK_AUDIT_HOOK_INSTALLED:
        sys.addaudithook(_scientific_runtime_audit_hook)
        _NETWORK_AUDIT_HOOK_INSTALLED = True


@contextmanager
def offline_network_guard(archive: RawArchive):
    """Block Python socket/DNS use and child-process escape during scientific computation."""
    global _NETWORK_GUARD_ACTIVE, _NETWORK_GUARD_ARCHIVE
    if _NETWORK_GUARD_ACTIVE:
        raise RuntimeError("scientific network guard cannot be nested")
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
    _NETWORK_GUARD_ARCHIVE = archive
    _NETWORK_GUARD_ACTIVE = True
    try:
        yield
    finally:
        _NETWORK_GUARD_ACTIVE = False
        _NETWORK_GUARD_ARCHIVE = None
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo


def _artifact_bindings(
    seal: Mapping[str, Any],
    marker: Mapping[str, Any],
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bindings = {
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_path": seal["path"],
        "seal_sha256": seal["sha256"],
        "sealed_source_collection_sha256": seal["source_collection_sha256"],
        "harness_sha256": sha256_file(ROOT / HARNESS),
        "focused_tests_sha256": sha256_file(ROOT / FOCUSED_TESTS),
        "attempt_marker_path": marker["path"],
        "attempt_marker_sha256": marker["sha256"],
    }
    if authorization is not None:
        bindings["phase_a_authorization"] = {
            "paths": authorization["paths"],
            "sha256": authorization["sha256"],
            "reviewer_identity": authorization["reviewer_identity"],
            "reviewed_at_utc": authorization["reviewed_at_utc"],
        }
    return bindings


def _invalid_completion(archive: RawArchive) -> None:
    if any(name.startswith("completion/") for name in archive.arrays):
        raise RuntimeError("invalid attempt already contains completion arrays")
    archive.phase = "invalid"
    archive.completion_arrays()


def _result_note(
    *,
    phase: str,
    valid: bool,
    payload: Mapping[str, Any],
    json_path: Path,
    json_sha256: str,
    raw_binding: Mapping[str, Any],
) -> str:
    status = "VALID" if valid else "INVALID"
    decision = payload.get("decision")
    return (
        f"# Stage-1 semantic factorial {phase} result ({status})\n\n"
        f"- JSON: `{json_path.name}`\n"
        f"- JSON SHA-256: `{json_sha256}`\n"
        f"- Raw NPZ: `{Path(str(raw_binding['path'])).name}`\n"
        f"- Raw NPZ SHA-256: `{raw_binding['npz_sha256']}`\n"
        f"- Raw collection SHA-256: `{raw_binding['collection_sha256']}`\n"
        f"- Raw array count: `{raw_binding['array_count']}`\n\n"
        "The JSON and this note bind the completed uncompressed raw archive. Quantitative "
        "claims require the independent review specified by the preregistration.\n\n"
        "## Frozen decision\n\n"
        f"```json\n{json.dumps(decision, indent=2, ensure_ascii=True, allow_nan=False)}\n```\n"
    )


def _write_phase_artifact(
    *,
    phase: str,
    valid: bool,
    paths: Mapping[str, Path],
    archive: RawArchive,
    scientific_payload: Mapping[str, Any],
    seal: Mapping[str, Any],
    marker: Mapping[str, Any],
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    label = "valid" if valid else "invalid"
    raw_path = paths[f"{label}_raw"]
    json_path = paths[f"{label}_json"]
    note_path = paths[f"{label}_note"]
    base_payload = {
        **dict(scientific_payload),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact_bindings": _artifact_bindings(seal, marker, authorization),
        "environment": environment_metadata(),
        "command": [sys.executable, *sys.argv],
    }
    # Strictly validate every result-controlled JSON value before creating the first sibling.
    # The raw binding appended below is generated solely from finite numeric arrays and strings.
    canonical_json(base_payload)
    _preflight_absent(paths.values())
    raw_binding = archive.write(raw_path)
    validate_raw_sidecar(raw_path, raw_binding)
    payload = {
        **base_payload,
        "raw_archive": raw_binding,
    }
    canonical_json(payload)
    _preflight_absent(path for path in paths.values() if path != raw_path)
    json_digest = _exclusive_write_json(json_path, payload)
    note = _result_note(
        phase=phase,
        valid=valid,
        payload=payload,
        json_path=json_path,
        json_sha256=json_digest,
        raw_binding=raw_binding,
    )
    _preflight_absent(path for path in paths.values() if path not in {raw_path, json_path})
    note_digest = _exclusive_write_bytes(note_path, note.encode("utf-8"))
    alternate = "invalid" if valid else "valid"
    _preflight_absent(paths[f"{alternate}_{suffix}"] for suffix in ("raw", "json", "note"))
    return {
        "valid": valid,
        "json_path": str(json_path),
        "json_sha256": json_digest,
        "raw_path": str(raw_path),
        "raw_sha256": raw_binding["npz_sha256"],
        "note_path": str(note_path),
        "note_sha256": note_digest,
    }


def run_bound_mechanism(seal_path: Path, output: Path) -> dict[str, Any]:
    paths = official_output_paths(output, "mechanism")
    seal = load_and_verify_seal(seal_path)
    marker_payload = {
        "artifact_type": ATTEMPT_ARTIFACT_TYPE,
        "phase": "mechanism",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seal_path": seal["path"],
        "seal_sha256": seal["sha256"],
        "sealed_source_collection_sha256": seal["source_collection_sha256"],
        "command": [sys.executable, *sys.argv],
        "environment": environment_metadata(),
    }
    marker = claim_attempt_marker(PHASE_A_ATTEMPT, marker_payload, paths.values())
    archive = RawArchive()
    valid = True
    with offline_network_guard(archive):
        try:
            scientific = execute_mechanism(archive)
        except ProtocolInvalid as error:
            valid = False
            _invalid_completion(archive)
            scientific = {
                "artifact_type": PHASE_A_INVALID_TYPE,
                "phase_a_pass": False,
                "invalid_stage": error.phase,
                "reason": error.reason,
                "evidence_reached_before_invalidation": error.evidence,
                "decision": {
                    "phase_a_pass": False,
                    "phase_b_authorized": False,
                    "default_change_authorized": False,
                },
            }
    # Source and marker drift after observation consume the attempt without emitting a result.
    load_and_verify_seal(seal_path)
    _validate_marker(marker)
    return _write_phase_artifact(
        phase="mechanism",
        valid=valid,
        paths=paths,
        archive=archive,
        scientific_payload=scientific,
        seal=seal,
        marker=marker,
    )


def run_bound_utility(
    *,
    seal_path: Path,
    phase_a: Path,
    phase_a_raw: Path,
    phase_a_review: Path,
    output: Path,
) -> dict[str, Any]:
    paths = official_output_paths(output, "utility")
    seal = load_and_verify_seal(seal_path)
    authorization = validate_phase_a_authorization(
        phase_a=phase_a,
        phase_a_raw=phase_a_raw,
        phase_a_review=phase_a_review,
        seal=seal,
    )
    marker_payload = {
        "artifact_type": ATTEMPT_ARTIFACT_TYPE,
        "phase": "utility",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seal_path": seal["path"],
        "seal_sha256": seal["sha256"],
        "sealed_source_collection_sha256": seal["source_collection_sha256"],
        "phase_a_authorization_paths": authorization["paths"],
        "phase_a_authorization_sha256": authorization["sha256"],
        "command": [sys.executable, *sys.argv],
        "environment": environment_metadata(),
    }
    marker = claim_attempt_marker(PHASE_B_ATTEMPT, marker_payload, paths.values())
    archive = RawArchive()
    valid = True
    with offline_network_guard(archive):
        try:
            scientific = execute_utility(archive)
        except ProtocolInvalid as error:
            valid = False
            _invalid_completion(archive)
            scientific = {
                "artifact_type": PHASE_B_INVALID_TYPE,
                "phase_b_valid": False,
                "execution_valid": False,
                "invalid_stage": error.phase,
                "reason": error.reason,
                "evidence_reached_before_invalidation": error.evidence,
                "decision": {
                    "repair_utility_survives": False,
                    "cross_backend_material_improvement": False,
                    "default_change_authorized": False,
                },
            }
    load_and_verify_seal(seal_path)
    _verify_authorization_unchanged(authorization)
    _validate_marker(marker)
    return _write_phase_artifact(
        phase="utility",
        valid=valid,
        paths=paths,
        archive=archive,
        scientific_payload=scientific,
        seal=seal,
        marker=marker,
        authorization=authorization,
    )


def configure_official_runtime() -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    seal = actions.add_parser("seal", help="verify and write the outcome-free implementation seal")
    seal.add_argument("--output", type=Path, required=True)
    mechanism = actions.add_parser("mechanism", help="consume the once-only Phase-A namespace")
    mechanism.add_argument("--seal", type=Path, required=True)
    mechanism.add_argument("--output", type=Path, required=True)
    utility = actions.add_parser(
        "utility", help="consume the authorized once-only Phase-B namespace"
    )
    utility.add_argument("--seal", type=Path, required=True)
    utility.add_argument("--phase-a", type=Path, required=True)
    utility.add_argument("--phase-a-raw", type=Path, required=True)
    utility.add_argument("--phase-a-review", type=Path, required=True)
    utility.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # argparse handles --help before this point, so help can never verify, claim, or execute.
    assert_implementation_ready()
    configure_official_runtime()
    if args.action == "seal":
        output = args.output.resolve()
        if output != DEFAULT_SEAL.resolve():
            raise ValueError("seal output must be the sole preregistered path")
        _preflight_absent((output,))
        payload = create_seal()
        digest = _exclusive_write_json(output, payload)
        print(f"saved {output} (sha256={digest})", flush=True)
        return 0
    if args.action == "mechanism":
        written = run_bound_mechanism(args.seal, args.output)
        print(json.dumps(written, sort_keys=True), flush=True)
        return 0
    if args.action == "utility":
        written = run_bound_utility(
            seal_path=args.seal,
            phase_a=args.phase_a,
            phase_a_raw=args.phase_a_raw,
            phase_a_review=args.phase_a_review,
            output=args.output,
        )
        print(json.dumps(written, sort_keys=True), flush=True)
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
