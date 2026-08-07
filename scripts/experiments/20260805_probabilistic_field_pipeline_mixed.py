#!/usr/bin/env python3
"""Run the protected probabilistic-field mechanism and all-dataset comparison matrix.

The top-level process verifies the reviewed task/run lock, evaluates the deterministic synthetic
mechanism controls, and then launches one guarded single-thread CPU worker for every sealed
dataset/seed/arm cell.  Workers may read calibration embedded in compact bundles and Gaussian2D
fields only.  They cannot open source images or external masks.  The calibrated 512-component
cap is deterministic, explicit in every receipt, and inactive outside this frozen experiment.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import datetime as dt
import hashlib
import importlib
import io
import json
import math
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260805_probabilistic_field_pipeline_mixed"
TASK_RELATIVE = Path("experiments/tasks") / f"{TASK_ID}.json"
RUN_RELATIVE = Path("runs") / TASK_ID
DRIVER_RELATIVE = Path("scripts/experiments") / f"{TASK_ID}.py"

EXPECTED_MECHANISM_CELLS = {
    "shape": (
        "center_only",
        "source_footprint",
        "oracle_sigma_surfel",
        "rank_aware_full_covariance",
    ),
    "association": (
        "field_no_association",
        "row_softmax_dustbin",
        "uot_uniform_capacity",
        "uot_field_mass_capacity",
        "uot_shuffled_candidate_negative",
    ),
    "mask": ("none", "hard", "probability"),
    "topology": ("largest_density_mass", "projection_nonlinearity"),
    "schedule": ("all", "progressive_then_full_cleanup"),
    "combined": ("native_controls", "all_candidate_mechanisms"),
}
CALIBRATED_ARMS = EXPECTED_MECHANISM_CELLS["combined"]
IMAGE_SUFFIXES = frozenset({".bmp", ".exr", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
FORBIDDEN_MODULES = (
    "PIL",
    "cv2",
    "imageio",
    "rtgs.data.calibrated",
    "rtgs.data.scene",
    "rtgs.optim.trainer",
    "rtgs.carrier_pipeline",
    "rtgs.lift.beam_fusion",
    "rtgs.lift.carrier_refinement",
    "rtgs.optim.carrier_schedule",
)


class DuplicateKeyError(ValueError):
    """Raised when JSON would otherwise silently overwrite a protocol field."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_source_files() -> tuple[Path, ...]:
    """Return every result-affecting Python source bound by the reviewed task digest."""

    explicit = (
        ROOT / "scripts" / "experiment_contract.py",
        ROOT / DRIVER_RELATIVE,
    )
    package = tuple(sorted((ROOT / "src" / "rtgs").rglob("*.py")))
    files = explicit + package
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("reviewed source binding contains a missing Python file")
    return files


def _source_tree_sha256() -> str:
    """Hash normalized paths and bytes for the exact result-producing source surface."""

    digest = hashlib.sha256()
    for path in _bound_source_files():
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _protocol_sha256(task: Mapping[str, Any]) -> str:
    protocol = {
        key: value for key, value in task.items() if key not in {"protocol_review", "status"}
    }
    return _canonical_sha256(protocol)


def _resolve_repository_path(value: str, *, expected: Path) -> Path:
    candidate = Path(value)
    resolved = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    target = (ROOT / expected).resolve()
    if resolved != target:
        raise ValueError(f"expected {target}, received {resolved}")
    return resolved


@dataclass(frozen=True)
class ExperimentCell:
    """One exact planned unit of outcome access."""

    cell_id: str
    stage: str
    arm: str
    seed: int
    source: str
    factors: dict[str, object]
    prerequisite: str | None = None


def _slug_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("cell factors must be finite")
    return str(value).replace(".", "p")


def _task_configuration(task: Mapping[str, Any]) -> dict[str, Any]:
    configuration = task.get("frozen_configuration")
    if not isinstance(configuration, dict):
        raise ValueError("task is missing frozen_configuration")
    cells = configuration.get("mechanism_cells")
    if not isinstance(cells, dict):
        raise ValueError("task is missing frozen mechanism_cells")
    normalized = {
        key: tuple(value) if isinstance(value, list) else value for key, value in cells.items()
    }
    if normalized != EXPECTED_MECHANISM_CELLS:
        raise ValueError("task mechanism cells do not match the reviewed driver surface")
    if configuration.get("driver") != DRIVER_RELATIVE.as_posix():
        raise ValueError("task driver binding does not name this file")
    return configuration


def _assert_task_contract(task: Mapping[str, Any]) -> None:
    if task.get("task_id") != TASK_ID:
        raise ValueError("task id does not match this driver")
    configuration = _task_configuration(task)
    generation = configuration.get("synthetic_generation")
    if not isinstance(generation, dict):
        raise ValueError("task is missing synthetic_generation")
    if generation.get("camera_count") != 5:
        raise ValueError("this driver requires the frozen five-camera generator")
    if generation.get("train_camera_count") != 4 or generation.get("heldout_camera_count") != 1:
        raise ValueError("this driver requires the frozen four/one camera split")
    source_binding = configuration.get("source_binding")
    if not isinstance(source_binding, dict) or set(source_binding) != {
        "algorithm",
        "scope",
        "sha256",
    }:
        raise ValueError("task is missing the exact source-tree binding")
    if source_binding["algorithm"] != "sha256-length-prefixed-path-and-bytes-v1":
        raise ValueError("task source-binding algorithm does not match this driver")
    if source_binding["scope"] != (
        "scripts/experiment_contract.py, this driver, and every src/rtgs/**/*.py file"
    ):
        raise ValueError("task source-binding scope does not match this driver")
    if source_binding["sha256"] != _source_tree_sha256():
        raise ValueError("reviewed source-tree digest does not match current source bytes")
    expected_command = [
        ".venv/bin/python",
        DRIVER_RELATIVE.as_posix(),
        "--task",
        TASK_RELATIVE.as_posix(),
        "--run",
        RUN_RELATIVE.as_posix(),
    ]
    if task.get("run_command") != expected_command:
        raise ValueError("task run_command does not match this driver")


def compile_cell_plan(task: Mapping[str, Any]) -> tuple[ExperimentCell, ...]:
    """Expand the frozen factorial without opening data or observing an outcome."""

    _assert_task_contract(task)
    configuration = _task_configuration(task)
    generation = configuration["synthetic_generation"]
    seeds = tuple(int(seed) for seed in task["seeds"])
    result: list[ExperimentCell] = []

    for seed, baseline, aspect, noise, arm in product(
        seeds,
        generation["baseline_degrees"],
        generation["aspect_ratios"],
        generation["center_noise_pixels"],
        EXPECTED_MECHANISM_CELLS["shape"],
    ):
        factors = {
            "baseline_degrees": baseline,
            "aspect_ratio": aspect,
            "center_noise_pixels": noise,
            "field_form": "exact_parent_projection",
        }
        result.append(
            ExperimentCell(
                cell_id=(
                    f"shape-s{seed}-b{_slug_number(baseline)}-a{_slug_number(aspect)}-"
                    f"n{_slug_number(noise)}-{arm}"
                ),
                stage="exact_shape_recovery",
                arm=arm,
                seed=seed,
                source="deterministic_synthetic",
                factors=factors,
            )
        )

    for seed, delete_rate, split_rate, arm in product(
        seeds,
        generation["component_delete_rates"],
        generation["component_split_rates"],
        EXPECTED_MECHANISM_CELLS["association"],
    ):
        factors = {
            "component_delete_rate": delete_rate,
            "component_split_rate": split_rate,
            "field_form": "independent_split_merge_delete_jitter",
        }
        result.append(
            ExperimentCell(
                cell_id=(
                    f"association-s{seed}-d{_slug_number(delete_rate)}-"
                    f"p{_slug_number(split_rate)}-{arm}"
                ),
                stage="recomponentized_association",
                arm=arm,
                seed=seed,
                source="deterministic_synthetic",
                factors=factors,
            )
        )

    for seed, false_positive, false_negative, arm in product(
        seeds,
        generation["mask_false_positive_rates"],
        generation["mask_false_negative_rates"],
        EXPECTED_MECHANISM_CELLS["mask"],
    ):
        factors = {
            "mask_false_positive_rate": false_positive,
            "mask_false_negative_rate": false_negative,
        }
        result.append(
            ExperimentCell(
                cell_id=(
                    f"mask-s{seed}-fp{_slug_number(false_positive)}-"
                    f"fn{_slug_number(false_negative)}-{arm}"
                ),
                stage="support_mask_factorial",
                arm=arm,
                seed=seed,
                source="deterministic_synthetic",
                factors=factors,
            )
        )

    for seed, arm in product(seeds, EXPECTED_MECHANISM_CELLS["topology"]):
        result.append(
            ExperimentCell(
                cell_id=f"topology-s{seed}-{arm}",
                stage="topology_factorial",
                arm=arm,
                seed=seed,
                source="deterministic_synthetic",
                factors={"baseline_degrees": 20.0, "aspect_ratio": 16.0},
            )
        )

    for seed, arm in product(seeds, EXPECTED_MECHANISM_CELLS["schedule"]):
        result.append(
            ExperimentCell(
                cell_id=f"schedule-s{seed}-{arm}",
                stage="schedule_factorial",
                arm=arm,
                seed=seed,
                source="deterministic_synthetic",
                factors={
                    "final_cleanup_iterations": configuration["pipeline"][
                        "full_view_cleanup_iterations"
                    ]
                },
            )
        )
        result.append(
            ExperimentCell(
                cell_id=f"half-stability-s{seed}-{arm}",
                stage="independent_half_stability",
                arm=arm,
                seed=seed,
                source="deterministic_synthetic",
                factors={"half_partition": "alternating_original_training_order"},
            )
        )

    for dataset in task["datasets"]:
        for seed, arm in product(seeds, CALIBRATED_ARMS):
            result.append(
                ExperimentCell(
                    cell_id=f"calibrated-{dataset['id']}-s{seed}-{arm}",
                    stage="calibrated_compact_operability",
                    arm=arm,
                    seed=seed,
                    source=dataset["id"],
                    factors={
                        "load_embedded_alpha": True,
                        "external_mask_access": False,
                        "target_component_cap": configuration["calibrated_followup"][
                            "target_component_cap"
                        ],
                    },
                    prerequisite=(
                        "all exact synthetic gates evaluated; failed candidates remain "
                        "diagnostic-only"
                    ),
                )
            )

    identifiers = [cell.cell_id for cell in result]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("compiled experiment cell ids are not unique")
    return tuple(result)


def plan_payload(task: Mapping[str, Any]) -> dict[str, Any]:
    cells = compile_cell_plan(task)
    stage_counts: dict[str, int] = {}
    for cell in cells:
        stage_counts[cell.stage] = stage_counts.get(cell.stage, 0) + 1
    return {
        "schema": "rtgs.probabilistic-field-experiment-plan.v1",
        "task_id": TASK_ID,
        "protocol_sha256": _protocol_sha256(task),
        "outcome_access": "guarded_after_review",
        "result_producer_enabled": True,
        "cell_count": len(cells),
        "stage_counts": stage_counts,
        "cells": [asdict(cell) for cell in cells],
    }


def _validate_run_binding(task_path: Path, run: Path, task: Mapping[str, Any]) -> None:
    if task.get("status") != "ready":
        raise ValueError("task is draft; outcome-producing execution is forbidden")
    review = task.get("protocol_review")
    if not isinstance(review, dict) or review.get("verdict") != "approved":
        raise ValueError("task has no approved distinct prospective review")
    lock_path = run / "task.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"{lock_path} is missing; init-run must bind the task first")
    lock = _strict_json(lock_path)
    seal = ROOT / str(task["data_seal"])
    checks = {
        "task_id": lock.get("task_id") == TASK_ID,
        "task_path": lock.get("task_path") == TASK_RELATIVE.as_posix(),
        "task_sha256": lock.get("task_sha256") == _sha256_file(task_path),
        "protocol_sha256": lock.get("protocol_sha256") == _protocol_sha256(task),
        "protocol_review": lock.get("protocol_review") == review,
        "data_seal_path": lock.get("data_seal_path") == task["data_seal"],
        "data_seal_sha256": seal.is_file() and lock.get("data_seal_sha256") == _sha256_file(seal),
        "command": lock.get("command") == task["run_command"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("run/task binding failed: " + ", ".join(failed))


class NoImageGuard:
    """Live worker boundary denying source-image paths and legacy/image imports."""

    def __init__(self) -> None:
        self.denied_paths = 0
        self.denied_imports = 0
        self.negative_control_denials = 0
        self._probing = False
        self._open = builtins.open
        self._io_open = io.open
        self._os_open = os.open
        self._import = builtins.__import__
        self._import_module = importlib.import_module

    @staticmethod
    def _forbidden_module(name: str) -> bool:
        return any(name == root or name.startswith(f"{root}.") for root in FORBIDDEN_MODULES)

    @staticmethod
    def _forbidden_path(value: object) -> bool:
        if isinstance(value, int):
            return False
        try:
            return Path(os.fspath(value)).suffix.lower() in IMAGE_SUFFIXES
        except TypeError:
            return False

    def _deny_path(self, value: object) -> None:
        if not self._forbidden_path(value):
            return
        if self._probing:
            self.negative_control_denials += 1
        else:
            self.denied_paths += 1
        raise PermissionError("probabilistic-field worker denies every image-file open")

    def _guarded_open(self, value: object, *args: object, **kwargs: object) -> Any:
        self._deny_path(value)
        return self._open(value, *args, **kwargs)

    def _guarded_io_open(self, value: object, *args: object, **kwargs: object) -> Any:
        self._deny_path(value)
        return self._io_open(value, *args, **kwargs)

    def _guarded_os_open(self, value: object, *args: object, **kwargs: object) -> int:
        self._deny_path(value)
        return self._os_open(value, *args, **kwargs)

    def _guarded_import(
        self,
        name: str,
        globals_value: Mapping[str, Any] | None = None,
        locals_value: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        resolved = name
        if level and isinstance(globals_value, Mapping):
            package = globals_value.get("__package__")
            if isinstance(package, str) and package:
                with contextlib.suppress(ImportError, ValueError):
                    resolved = importlib.util.resolve_name("." * level + name, package)
        candidates = (resolved, *(f"{resolved}.{item}" for item in fromlist if item != "*"))
        if any(self._forbidden_module(item) for item in candidates):
            if self._probing:
                self.negative_control_denials += 1
            else:
                self.denied_imports += 1
            raise ImportError("probabilistic-field worker denies image/legacy imports")
        return self._import(name, globals_value, locals_value, fromlist, level)

    def _guarded_import_module(self, name: str, package: str | None = None) -> Any:
        resolved = name
        if name.startswith("."):
            with contextlib.suppress(ImportError, ValueError):
                resolved = importlib.util.resolve_name(name, package)
        if self._forbidden_module(resolved):
            if self._probing:
                self.negative_control_denials += 1
            else:
                self.denied_imports += 1
            raise ImportError("probabilistic-field worker denies image/legacy imports")
        return self._import_module(name, package)

    def __enter__(self) -> NoImageGuard:
        loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        if loaded:
            raise RuntimeError(f"forbidden modules loaded before worker guard: {loaded}")
        builtins.open = self._guarded_open
        io.open = self._guarded_io_open
        os.open = self._guarded_os_open
        builtins.__import__ = self._guarded_import
        importlib.import_module = self._guarded_import_module
        self._probing = True
        try:
            with contextlib.suppress(PermissionError):
                builtins.open(ROOT / "negative-control.png", "rb")
            with contextlib.suppress(ImportError):
                builtins.__import__("PIL.Image")
            with contextlib.suppress(ImportError):
                importlib.import_module("rtgs.data.scene")
        finally:
            self._probing = False
        if self.negative_control_denials != 3:
            self.__exit__()
            raise RuntimeError("worker input-boundary negative controls did not all fire")
        return self

    def __exit__(self, *exc: object) -> None:
        builtins.open = self._open
        io.open = self._io_open
        os.open = self._os_open
        builtins.__import__ = self._import
        importlib.import_module = self._import_module

    def record(self) -> dict[str, object]:
        loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        return {
            "passed": not loaded and self.denied_paths == 0 and self.denied_imports == 0,
            "denied_paths": self.denied_paths,
            "denied_imports": self.denied_imports,
            "negative_control_denials": self.negative_control_denials,
            "forbidden_modules_loaded": loaded,
        }


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text_new(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _failure_payload(
    *,
    phase: str,
    error: BaseException,
    context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the canonical non-overwriting structured failure receipt."""

    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "failed",
        "phase": phase,
        "failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "exception_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "context": dict(context or {}),
    }


def _write_failure_new(
    directory: Path,
    *,
    phase: str,
    error: BaseException,
    context: Mapping[str, object] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "failure.json"
    if not path.exists():
        _write_json_new(
            path,
            _failure_payload(phase=phase, error=error, context=context),
        )
    return path


def _directory_bytes(directory: Path, *, exclude: frozenset[str] = frozenset()) -> int:
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and path.name not in exclude
    )


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform != "darwin" else value)


def _dataset_record(task: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [item for item in task["datasets"] if item["id"] == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"task does not define exactly one dataset {dataset_id!r}")
    return dict(matches[0])


def _split_indices(
    dataset: Any, split: Mapping[str, Any]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    names = [view.view_id for view in dataset.views]
    lookup = {name: index for index, name in enumerate(names)}
    if len(lookup) != len(names) or set(names) != set(split["train"]) | set(split["heldout"]):
        raise ValueError("frozen split does not exactly partition the compact manifest")
    train = tuple(lookup[name] for name in split["train"])
    heldout = tuple(lookup[name] for name in split["heldout"])
    if set(train) & set(heldout) or len(train) + len(heldout) != len(names):
        raise ValueError("frozen compact split overlaps or omits a view")
    return train, heldout


def _cell_relative(dataset_id: str, seed: int, arm: str, warmup: bool) -> Path:
    prefix = "warmups" if warmup else "cells"
    return Path(prefix) / dataset_id / f"seed_{seed}" / arm


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _calibrated_config(task: Mapping[str, Any], *, seed: int, arm: str) -> Any:
    from rtgs.lift.fiber_correspondence import FiberFitConfig
    from rtgs.lift.field_lifter import FieldAssociationConfig, FieldLiftConfig
    from rtgs.lift.field_refit import FieldRefitConfig

    pipeline = task["frozen_configuration"]["pipeline"]
    followup = task["frozen_configuration"]["calibrated_followup"]
    candidate = arm == "all_candidate_mechanisms"
    if arm not in CALIBRATED_ARMS:
        raise ValueError(f"unknown calibrated arm {arm!r}")
    association = None
    if candidate:
        association = FieldAssociationConfig(
            observation_capacity_mode="field_mass",
            track_capacity_mode="field_mass",
            failure_policy=str(pipeline["association_failure_policy"]),
            fit=FiberFitConfig(
                temperatures=tuple(float(item) for item in pipeline["association_temperatures"]),
                residual_variances=tuple(
                    float(item) for item in pipeline["association_residual_variances"]
                ),
                geometry_steps=int(pipeline["association_geometry_steps"]),
                assignment="unbalanced_sinkhorn",
                max_pair_cost=float(pipeline["association_max_pair_cost"]),
                dustbin_cost=float(pipeline["association_dustbin_cost"]),
                track_batch_size=int(followup["max_tracks"]),
                sinkhorn_iterations=80,
                sinkhorn_tolerance=1e-8,
            ),
        )
    refit = FieldRefitConfig(
        iterations=int(followup["refit_iterations"]),
        learning_rate=0.025,
        appearance_start=int(followup["appearance_start"]),
        visibility_refresh=5,
        chunk_size=128,
        view_schedule="progressive" if candidate else "all",
        progressive_start_views=2,
        full_view_cleanup_iterations=(
            int(followup["progressive_full_view_cleanup_iterations"]) if candidate else 0
        ),
    )
    return FieldLiftConfig(
        placement_mode=str(pipeline["placement_mode"]),
        compute_dtype="float64",
        max_tracks=int(followup["max_tracks"]),
        max_train_views=int(followup["max_train_views"]),
        target_component_cap=int(followup["target_component_cap"]),
        depth_samples=int(pipeline["depth_samples"]),
        min_views=int(pipeline["min_views"]),
        background_fraction=0.0,
        mask_mode="probability" if candidate else "hard",
        association=association,
        topology_rounds=int(followup["topology_rounds"]),
        topology_split_mode=("projection_nonlinearity" if candidate else "largest_density_mass"),
        parsimony_per_component=float(pipeline["parsimony_per_component"]),
        validation_sample_cap=int(followup["validation_sample_cap"]),
        seed=seed,
        refit=refit,
    )


def _source_projection_invariants(result: Any) -> dict[str, float]:
    """Measure mean and covariance preservation separately at the exact source seam."""

    import torch

    means, covariances, _depth = result.refit.fiber.source_projection()
    means = means.detach()
    covariances = covariances.detach()
    target_means = result.refit.fiber.source_means2d.detach()
    target_covariances = result.refit.fiber.source_covariances2d.detach()
    covariance_denominator = torch.linalg.matrix_norm(target_covariances, dim=(-2, -1)).clamp_min(
        torch.finfo(covariances.dtype).tiny
    )
    covariance_relative = (
        torch.linalg.matrix_norm(
            covariances - target_covariances,
            dim=(-2, -1),
        )
        / covariance_denominator
    )
    return {
        "source_mean_max_error": float((means - target_means).abs().amax()),
        "source_covariance_max_error": float((covariances - target_covariances).abs().amax()),
        "source_covariance_relative_error": float(covariance_relative.amax()),
    }


def _plan_is_finite_nonnegative(plan: Any) -> bool:
    """Check every realized transport and dustbin field without assuming balanced UOT marginals."""

    import torch

    tensors = [plan.real_mass, plan.track_dustbin_mass, plan.track_capacities]
    if plan.observation_dustbin_mass is not None:
        assert plan.dustbin_dustbin_mass is not None
        assert plan.observation_capacities is not None
        tensors.extend(
            [
                plan.observation_dustbin_mass,
                plan.dustbin_dustbin_mass,
                plan.observation_capacities,
            ]
        )
    return all(bool(torch.isfinite(value).all()) and bool((value >= 0).all()) for value in tensors)


def _association_invariants(association: Any | None) -> dict[str, object]:
    """Summarize finite mass, solver convergence, and retained candidate-gate evidence."""

    if association is None:
        return {
            "transport_plan_count": 0,
            "transport_finite": True,
            "transport_min_real_mass": None,
            "transport_fixed_point_residual_max": None,
            "candidate_gate_violation_mass_max": None,
        }
    plans = tuple(association.plans)
    real_masses = [float(plan.real_mass.sum()) for plan in plans]
    fixed = [
        float(plan.fixed_point_residual) for plan in plans if plan.fixed_point_residual is not None
    ]
    candidate_violations = [
        float(plan.real_mass[~plan.candidate_mask].sum())
        for plan in plans
        if plan.candidate_mask is not None
    ]
    return {
        "transport_plan_count": len(plans),
        "transport_finite": all(_plan_is_finite_nonnegative(plan) for plan in plans),
        "transport_min_real_mass": min(real_masses) if real_masses else None,
        "transport_fixed_point_residual_max": max(fixed) if fixed else None,
        "candidate_gate_violation_mass_max": (
            max(candidate_violations) if len(candidate_violations) == len(plans) else None
        ),
    }


def _split_conservation_invariants() -> dict[str, float]:
    """Exercise the production split constructor and return its two conservation residuals."""

    from rtgs.lift.field_topology import (
        FieldComponent,
        FieldComponentPayload,
        FieldTopologyState,
        SourceAnchor,
        SourceLineage,
        propose_split,
    )

    lineage = SourceLineage(0, 0)
    parent = FieldComponent(
        0,
        FieldComponentPayload(
            source_lineage=(lineage,),
            source_anchor=SourceAnchor(lineage, (12.0, 18.0)),
            depth=2.0,
            cross=(0.0, 0.0),
            log_ray_scale=-2.0,
            density_mass=0.73,
            source_color=(0.2, 0.4, 0.6),
            render_opacity=0.37,
        ),
    )
    proposal = propose_split(
        FieldTopologyState((parent,)),
        parent.stable_id,
        mass_fraction=0.37,
        depth_offsets=(-0.1, 0.1),
    )
    children = proposal.add_components
    child_mass = sum(item.density_mass for item in children)
    child_opacity = 1.0
    for child in children:
        child_opacity *= 1.0 - child.render_opacity
    child_opacity = 1.0 - child_opacity
    return {
        "split_density_mass_error": abs(child_mass - parent.density_mass),
        "split_optical_thickness_error": abs(child_opacity - parent.render_opacity),
    }


def _enforce_result_invariants(
    task: Mapping[str, Any],
    result: Any,
    *,
    association_required: bool,
) -> dict[str, object]:
    gates = task["frozen_configuration"]["invariant_gates"]
    invariants: dict[str, object] = {
        **_source_projection_invariants(result),
        **_association_invariants(result.association),
        **_split_conservation_invariants(),
    }
    failures = []
    if float(invariants["source_mean_max_error"]) > float(gates["source_projection_max_error"]):
        failures.append("source mean projection")
    if float(invariants["source_covariance_relative_error"]) > float(
        gates["source_covariance_relative_error"]
    ):
        failures.append("source covariance projection")
    if float(invariants["split_density_mass_error"]) > float(gates["split_density_mass_tolerance"]):
        failures.append("split density mass")
    if float(invariants["split_optical_thickness_error"]) > float(
        gates["split_optical_thickness_tolerance"]
    ):
        failures.append("split optical thickness")
    if association_required:
        if int(invariants["transport_plan_count"]) == 0:
            failures.append("transport plan missing")
        if not bool(invariants["transport_finite"]):
            failures.append("transport non-finite")
        real_mass = invariants["transport_min_real_mass"]
        if real_mass is None or float(real_mass) < float(gates["minimum_transport_real_mass"]):
            failures.append("transport real mass")
        residual = invariants["transport_fixed_point_residual_max"]
        if residual is None or float(residual) > float(
            gates["transport_fixed_point_residual_tolerance"]
        ):
            failures.append("transport fixed point")
        candidate_violation = invariants["candidate_gate_violation_mass_max"]
        if candidate_violation is None or float(candidate_violation) > float(
            gates["candidate_mass_tolerance"]
        ):
            failures.append("candidate gate")
    if failures:
        raise RuntimeError("hard invariant violation: " + ", ".join(failures))
    return invariants


def _enforce_pipeline_result_invariants(
    task: Mapping[str, Any],
    pipeline: Any,
    *,
    association_required: bool,
) -> dict[str, object]:
    """Enforce and aggregate hard invariants over the primary and every half fit."""

    results = [pipeline.reconstruction]
    if pipeline.half_reconstructions is not None:
        results.extend(pipeline.half_reconstructions)
    per_fit = [
        _enforce_result_invariants(
            task,
            result,
            association_required=association_required,
        )
        for result in results
    ]

    def optional_extreme(key: str, *, minimum: bool = False) -> float | None:
        values = [float(item[key]) for item in per_fit if item[key] is not None]
        if not values:
            return None
        return min(values) if minimum else max(values)

    return {
        "source_mean_max_error": max(float(item["source_mean_max_error"]) for item in per_fit),
        "source_covariance_max_error": max(
            float(item["source_covariance_max_error"]) for item in per_fit
        ),
        "source_covariance_relative_error": max(
            float(item["source_covariance_relative_error"]) for item in per_fit
        ),
        "transport_plan_count": sum(int(item["transport_plan_count"]) for item in per_fit),
        "transport_finite": all(bool(item["transport_finite"]) for item in per_fit),
        "transport_min_real_mass": optional_extreme(
            "transport_min_real_mass",
            minimum=True,
        ),
        "transport_fixed_point_residual_max": optional_extreme(
            "transport_fixed_point_residual_max"
        ),
        "candidate_gate_violation_mass_max": optional_extreme("candidate_gate_violation_mass_max"),
        "split_density_mass_error": max(
            float(item["split_density_mass_error"]) for item in per_fit
        ),
        "split_optical_thickness_error": max(
            float(item["split_optical_thickness_error"]) for item in per_fit
        ),
        "hard_invariant_checked_fit_count": len(per_fit),
    }


def _heldout_fit_access_count(
    task: Mapping[str, Any],
    result: Any,
    fits: Any | None = None,
    *,
    train_view_indices: Sequence[int] | None = None,
    heldout_view_indices: Sequence[int] | None = None,
) -> int:
    """Measure optimized/held-out overlap and fail if the realized split is not exact."""

    if fits is not None:
        if train_view_indices is not None or heldout_view_indices is not None:
            raise ValueError("pass fits or explicit split indices, not both")
        train_view_indices = fits.train_view_indices
        heldout_view_indices = fits.heldout_view_indices
    if train_view_indices is None or heldout_view_indices is None:
        raise ValueError("realized fit isolation requires an explicit train/held-out split")
    optimized = set(result.optimized_view_indices)
    train = set(train_view_indices)
    heldout = set(heldout_view_indices)
    reported_heldout = set(result.heldout_view_indices)
    access_count = len(optimized & heldout)
    expected = int(task["frozen_configuration"]["invariant_gates"]["heldout_fit_access_count"])
    if access_count != expected or not optimized <= train or reported_heldout != heldout:
        raise RuntimeError("held-out view entered fitting or the reporting split changed")
    return access_count


def _pipeline_fit_access_metrics(
    task: Mapping[str, Any],
    pipeline: Any,
    fits: Any,
) -> dict[str, int]:
    """Measure the primary and every realized independent-half fit against its own partition."""

    access_count = _heldout_fit_access_count(task, pipeline.reconstruction, fits)
    checked_fit_count = 1
    halves = pipeline.half_reconstructions
    stability = pipeline.stability
    if halves is None:
        if stability is not None:
            raise RuntimeError("independent-half stability exists without half reconstructions")
    else:
        if stability is None or len(halves) != 2:
            raise RuntimeError("independent-half reconstructions lack exact split evidence")
        half_train_views = (stability.first_train_views, stability.second_train_views)
        original_train = set(fits.train_view_indices)
        first_train, second_train = map(set, half_train_views)
        if first_train & second_train or first_train | second_train != original_train:
            raise RuntimeError("independent-half realized training partitions are not exact")
        for result, train_views in zip(halves, half_train_views, strict=True):
            train_set = set(train_views)
            heldout_views = tuple(index for index in range(fits.n_views) if index not in train_set)
            access_count += _heldout_fit_access_count(
                task,
                result,
                train_view_indices=tuple(train_views),
                heldout_view_indices=heldout_views,
            )
            checked_fit_count += 1
    return {
        "heldout_fit_access_count": access_count,
        "heldout_fit_checked_fit_count": checked_fit_count,
    }


def _calibrated_worker(
    *,
    task: Mapping[str, Any],
    run: Path,
    dataset_id: str,
    seed: int,
    arm: str,
    warmup: bool,
) -> None:
    output = run / _cell_relative(dataset_id, seed, arm, warmup)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibrated cell {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.worker-{os.getpid()}-", dir=output.parent)
    )
    process_started = time.perf_counter()
    guard = NoImageGuard()
    published = False
    phase = "worker_initialization"
    try:
        with guard:
            import torch

            from rtgs.data.compact_views import CompactDataset
            from rtgs.data.field_inputs import SceneFits
            from rtgs.lift.field_lifter import FieldLifter
            from rtgs.lift.probabilistic_pipeline import (
                ProbabilisticFieldPipelineConfig,
                run_probabilistic_field_pipeline,
            )

            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)
            dataset_record = _dataset_record(task, dataset_id)
            compact_manifest = ROOT / dataset_record["compact_manifest"]
            compact_root = compact_manifest.parent
            scope_started = time.perf_counter()
            phase = "compact_loading"
            load_started = time.perf_counter()
            compact = CompactDataset.load(
                compact_root,
                device="cpu",
                load_alpha=bool(
                    task["frozen_configuration"]["calibrated_followup"]["load_embedded_alpha"]
                ),
            )
            train, heldout = _split_indices(compact, task["splits"][dataset_id])
            fits = SceneFits.from_compact_dataset(
                compact,
                train_view_indices=train,
                heldout_view_indices=heldout,
                geometry_is_train_only=False,
            )
            config = _calibrated_config(task, seed=seed, arm=arm)
            load_seconds = time.perf_counter() - load_started
            independent_seed = int(
                task["frozen_configuration"]["calibrated_followup"]["independent_half_seed"]
            )
            phase = "field_fit"
            fit_started = time.perf_counter()
            pipeline_result = None
            if not warmup and seed == independent_seed:
                pipeline_result = run_probabilistic_field_pipeline(
                    fits,
                    ProbabilisticFieldPipelineConfig(
                        lift=config,
                        independent_half_validation=True,
                        minimum_half_views=2,
                        match_radius_fraction=0.10,
                    ),
                )
                result = pipeline_result.reconstruction
                stability = pipeline_result.stability
            else:
                result = FieldLifter(config).fit(fits)
                stability = None
            fit_seconds = time.perf_counter() - fit_started
            fit_access_metrics = (
                _pipeline_fit_access_metrics(task, pipeline_result, fits)
                if pipeline_result is not None
                else {
                    "heldout_fit_access_count": _heldout_fit_access_count(task, result, fits),
                    "heldout_fit_checked_fit_count": 1,
                }
            )
            heldout_metrics = result.semantic_validation.heldout
            placement_heldout = result.placement_semantic_validation.heldout
            if heldout_metrics is None or placement_heldout is None:
                raise RuntimeError("calibrated worker requires reporting-only held-out views")
            invariants = (
                _enforce_pipeline_result_invariants(
                    task,
                    pipeline_result,
                    association_required=arm == "all_candidate_mechanisms",
                )
                if pipeline_result is not None
                else {
                    **_enforce_result_invariants(
                        task,
                        result,
                        association_required=arm == "all_candidate_mechanisms",
                    ),
                    "hard_invariant_checked_fit_count": 1,
                }
            )
            phase = "serialization"
            serialize_started = time.perf_counter()
            result.gaussians_init.save_ply(temporary / "gaussians_init.ply")
            result.gaussians.save_ply(temporary / "gaussians.ply")
            stable_center = None
            stable_censored = False
            stability_payload = None
            if stability is not None:
                stability_payload = _json_safe(asdict(stability))
                stable_center = float(stability.center_median)
                if not math.isfinite(stable_center):
                    stable_center = float(stability.match_radius)
                    stable_censored = True
            summary = {
                "schema_version": 1,
                "task_id": TASK_ID,
                "dataset_id": dataset_id,
                "dataset_role": dataset_record["role"],
                "seed": seed,
                "arm": arm,
                "warmup": warmup,
                "metrics": {
                    "heldout_field_rgb_mse": float(heldout_metrics.rgb_mse),
                    "heldout_field_density_mse": float(heldout_metrics.density_mse),
                    "placement_heldout_rgb_mse": float(placement_heldout.rgb_mse),
                    "placement_heldout_density_mse": float(placement_heldout.density_mse),
                    "refit_wall_seconds": float(result.refit.elapsed_seconds[-1]),
                    "pipeline_wall_seconds": fit_seconds,
                    "final_gaussian_count": result.gaussians.n,
                    "source_projection_max_error": result.refit.source_projection_max_error,
                    "accepted_steps": result.refit.accepted_steps,
                    "independent_half_center_median": stable_center,
                    "input_components_original_mean": statistics.mean(
                        result.diagnostics["target_component_counts_original"]
                    ),
                    "input_components_used_mean": statistics.mean(
                        result.diagnostics["target_component_counts_used"]
                    ),
                    "embedded_alpha_view_fraction": sum(
                        view.alpha is not None for view in compact.views
                    )
                    / compact.n_views,
                    **fit_access_metrics,
                    **invariants,
                },
                "independent_half_stability": stability_payload,
                "independent_half_center_censored_at_match_radius": stable_censored,
                "optimized_view_indices": list(result.optimized_view_indices),
                "heldout_view_indices": list(result.heldout_view_indices),
                "objective_history": list(result.refit.objective_history),
                "active_view_counts": list(result.refit.active_view_counts),
                "refit_elapsed_seconds": list(result.refit.elapsed_seconds),
                "diagnostics": _json_safe(result.diagnostics),
            }
            _write_json_new(temporary / "summary.json", summary)
            _write_json_new(
                temporary / "gaussians.config.json",
                {
                    "task_id": TASK_ID,
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "arm": arm,
                    "warmup": warmup,
                    "field_lift": asdict(config),
                    "training": {"packed": False, "antialiased": False},
                },
            )
            compact_files = [compact_manifest]
            compact_files.extend(compact_root / f"{view.view_id}.rtgsv" for view in compact.views)
            input_bytes = sum(path.stat().st_size for path in compact_files)
            guard_record = guard.record()
            if not guard_record["passed"]:
                raise RuntimeError(f"worker input boundary failed: {guard_record}")
            _write_json_new(
                temporary / "input_boundary_receipt.json",
                {
                    "schema_version": 1,
                    "task_id": TASK_ID,
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "arm": arm,
                    "warmup": warmup,
                    "allowed_modalities": ["calibration", "gaussians2d"],
                    "compact_alpha_loaded": True,
                    "external_mask_access": False,
                    "heldout_training_access": False,
                    "loaded_compact_files": [
                        {
                            "path": path.relative_to(ROOT).as_posix(),
                            "bytes": path.stat().st_size,
                            "sha256": _sha256_file(path),
                        }
                        for path in compact_files
                    ],
                    "input_bytes": input_bytes,
                    "guard": guard_record,
                    "component_cap": result.diagnostics["target_component_cap"],
                    "component_counts_original": result.diagnostics[
                        "target_component_counts_original"
                    ],
                    "component_counts_used": result.diagnostics["target_component_counts_used"],
                    "component_selection_sha256": result.diagnostics[
                        "target_component_selection_sha256"
                    ],
                },
            )
            serialize_seconds = time.perf_counter() - serialize_started
        phase = "publication"
        publish_started = time.perf_counter()
        os.replace(temporary, output)
        published = True
        publish_seconds = time.perf_counter() - publish_started
        scope_seconds = time.perf_counter() - scope_started
        output_bytes = _directory_bytes(output)
        phase = "resource_receipt"
        _write_json_new(
            output / "resource_receipt.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "dataset_id": dataset_id,
                "seed": seed,
                "arm": arm,
                "warmup": warmup,
                "cpu_threads": 1,
                "torch_threads": torch.get_num_threads(),
                "cpu_model": platform.processor() or platform.machine(),
                "cuda_used": False,
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_device_count": int(torch.cuda.device_count()),
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "refit_wall_seconds": float(summary["metrics"]["refit_wall_seconds"]),
                "fit_wall_seconds": fit_seconds,
                "wall_seconds": scope_seconds,
                "process_wall_seconds": time.perf_counter() - process_started,
                "stage_wall_seconds": {
                    "compact_loading": load_seconds,
                    "field_fit": fit_seconds,
                    "serialization": serialize_seconds,
                    "publication": publish_seconds,
                },
                "ru_maxrss_bytes": _peak_rss_bytes(),
            },
        )
    except Exception as error:
        failure_root = output if published else temporary
        _write_failure_new(
            failure_root,
            phase=phase,
            error=error,
            context={
                "dataset_id": dataset_id,
                "seed": seed,
                "arm": arm,
                "warmup": warmup,
            },
        )
        if not published and not output.exists():
            os.replace(temporary, output)
        raise


def _synthetic_fixture(seed: int, aspect_ratio: float = 4.0) -> tuple[Any, tuple[Any, ...]]:
    import torch

    from rtgs.core.camera import Camera
    from rtgs.core.gaussians3d import Gaussians3D

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    count = 12
    means = 0.42 * (torch.rand((count, 3), generator=generator, dtype=torch.float64) - 0.5)
    means[:, 2] *= 0.55
    covariances = []
    for index in range(count):
        angle = 0.31 * index
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = torch.tensor(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        eigenvalues = torch.tensor(
            [0.004 * aspect_ratio, 0.004 / aspect_ratio, 0.003],
            dtype=torch.float64,
        )
        covariances.append(rotation @ torch.diag(eigenvalues) @ rotation.T)
    colors = 0.15 + 0.75 * torch.rand((count, 3), generator=generator, dtype=torch.float64)
    opacity = 0.35 + 0.45 * torch.rand(count, generator=generator, dtype=torch.float64)
    gaussians = Gaussians3D.from_means_covs(
        means,
        torch.stack(covariances),
        colors,
        opacity,
    )
    cameras = []
    for angle in (-18.0, -6.0, 6.0, 18.0, 38.0):
        radians = math.radians(angle)
        eye = torch.tensor(
            [2.6 * math.sin(radians), 0.22, 2.6 * math.cos(radians)],
            dtype=torch.float64,
        )
        cameras.append(Camera.look_at(eye, torch.zeros(3), width=96, height=96))
    return gaussians, tuple(cameras)


def _baseline_cameras(baseline_degrees: float) -> tuple[Any, ...]:
    import torch

    from rtgs.core.camera import Camera

    angles = (
        -1.5 * baseline_degrees,
        -0.5 * baseline_degrees,
        0.5 * baseline_degrees,
        1.5 * baseline_degrees,
        38.0,
    )
    cameras = []
    for angle in angles:
        radians = math.radians(angle)
        eye = torch.tensor(
            [2.6 * math.sin(radians), 0.22, 2.6 * math.cos(radians)],
            dtype=torch.float64,
        )
        cameras.append(Camera.look_at(eye, torch.zeros(3), width=96, height=96))
    return tuple(cameras)


def _analytic_projection(gaussians: Any, camera: Any) -> Any:
    import torch

    from rtgs.core.sh import sh_to_rgb
    from rtgs.lift.field_loss import AnalyticGaussianField2D
    from rtgs.render.projection import project_gaussians_ewa

    projection = project_gaussians_ewa(gaussians, camera)
    colors = sh_to_rgb(gaussians.sh[:, 0])
    density = gaussians.opacity.to(torch.float64)
    return AnalyticGaussianField2D(
        means=projection.means2d,
        covariances=projection.covariances2d,
        density_amplitudes=density,
        rgb_amplitudes=density[:, None] * colors,
    )


def _shape_cell(cell: ExperimentCell) -> dict[str, object]:
    import torch

    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.core.sh import sh_to_rgb
    from rtgs.lift.field_loss import field_l2
    from rtgs.lift.field_observability import (
        solve_projected_covariance,
        triangulate_projected_mean,
    )
    from rtgs.render.projection import EWA_DILATION, project_gaussians_ewa

    started = time.perf_counter()
    baseline = float(cell.factors["baseline_degrees"])
    aspect = float(cell.factors["aspect_ratio"])
    noise = float(cell.factors["center_noise_pixels"])
    gt, _unused = _synthetic_fixture(cell.seed, aspect)
    cameras = _baseline_cameras(baseline)
    train = cameras[:4]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cell.seed + int(100 * baseline + 10 * aspect + 1000 * noise))
    projections = [project_gaussians_ewa(gt, camera) for camera in train]
    recovered_means = []
    recovered_covariances = []
    ranks = []
    for component in range(gt.n):
        centers = torch.stack([projection.means2d[component] for projection in projections])
        if noise:
            centers = centers + noise * torch.randn(
                centers.shape,
                generator=generator,
                dtype=centers.dtype,
            )
        triangulated = triangulate_projected_mean(centers, train)
        mean = triangulated.mean
        projected_covariances = torch.stack(
            [projection.covariances2d[component] for projection in projections]
        )
        if cell.arm == "center_only":
            covariance = torch.eye(3, dtype=torch.float64) * 0.004
            rank = 0
        elif cell.arm == "source_footprint":
            covariance_result = solve_projected_covariance(
                mean,
                train[:1],
                projected_covariances[:1],
                dilation=EWA_DILATION,
                prior_covariance=torch.eye(3, dtype=torch.float64) * 0.004,
                minimum_eigenvalue=1e-8,
            )
            covariance = covariance_result.covariance
            rank = covariance_result.report.rank
        elif cell.arm == "oracle_sigma_surfel":
            mean = gt.means[component]
            covariance = gt.covariance()[component]
            rank = 6
        else:
            covariance_result = solve_projected_covariance(
                mean,
                train,
                projected_covariances,
                dilation=EWA_DILATION,
                prior_covariance=gt.covariance()[component],
                minimum_eigenvalue=1e-8,
            )
            covariance = covariance_result.covariance
            rank = covariance_result.report.rank
        recovered_means.append(mean)
        recovered_covariances.append(covariance)
        ranks.append(rank)
    means = torch.stack(recovered_means)
    covariances = torch.stack(recovered_covariances)
    recovered = Gaussians3D.from_means_covs(
        means,
        covariances,
        sh_to_rgb(gt.sh[:, 0]),
        gt.opacity,
    )
    target = _analytic_projection(gt, cameras[4])
    prediction = _analytic_projection(recovered, cameras[4])
    loss = field_l2(prediction, target, chunk_size=64)
    heldout_truth_projection = project_gaussians_ewa(gt, cameras[4])
    heldout_recovered_projection = project_gaussians_ewa(recovered, cameras[4])
    heldout_covariance_denominator = torch.linalg.matrix_norm(
        heldout_truth_projection.covariances2d,
        dim=(-2, -1),
    ).clamp_min(torch.finfo(torch.float64).tiny)
    heldout_covariance_relative = (
        torch.linalg.matrix_norm(
            heldout_recovered_projection.covariances2d - heldout_truth_projection.covariances2d,
            dim=(-2, -1),
        )
        / heldout_covariance_denominator
    )
    center_rmse = float((means - gt.means).square().sum(dim=-1).mean().sqrt())
    truth_covariance = gt.covariance()
    relative = torch.linalg.matrix_norm(covariances - truth_covariance) / torch.linalg.matrix_norm(
        truth_covariance
    ).clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "heldout_field_density_mse": float(loss.density / (96 * 96)),
        "heldout_field_rgb_mse": float(loss.rgb_numerator / (96 * 96 * 3)),
        "world_center_rmse": center_rmse,
        "covariance_relative_frobenius": float(relative.median()),
        "heldout_covariance_relative_frobenius": float(heldout_covariance_relative.median()),
        "full_rank_fraction": sum(rank == 6 for rank in ranks) / len(ranks),
        "wall_seconds": time.perf_counter() - started,
    }


def _recomponentized_view(
    predicted: Any,
    *,
    split_rate: float,
    delete_rate: float,
    seed: int,
) -> tuple[Any, Any]:
    import torch

    from rtgs.lift.field_loss import AnalyticGaussianField2D

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    means = []
    covariances = []
    densities = []
    rgbs = []
    parents = []
    for parent in range(predicted.n):
        split = float(torch.rand((), generator=generator)) < split_rate
        count = 2 if split else 1
        covariance = predicted.covariances[parent]
        if count == 2:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            axis = eigenvectors[:, 0]
            offset = axis * (0.20 * eigenvalues[0]).sqrt()
            offsets = (-offset, offset)
            child_covariance = covariance - offset[:, None] * offset[None, :]
        else:
            offsets = (torch.zeros(2, dtype=covariance.dtype),)
            child_covariance = covariance
        for offset in offsets:
            if float(torch.rand((), generator=generator)) < delete_rate:
                continue
            jitter = 0.35 * torch.randn(2, generator=generator, dtype=predicted.means.dtype)
            means.append(predicted.means[parent] + offset + jitter)
            covariances.append(child_covariance)
            densities.append(predicted.density_amplitudes[parent] / count)
            rgbs.append(predicted.rgb_amplitudes[parent] / count)
            parents.append(parent)
    if not means:
        means.append(predicted.means[0])
        covariances.append(predicted.covariances[0])
        densities.append(predicted.density_amplitudes[0])
        rgbs.append(predicted.rgb_amplitudes[0])
        parents.append(0)
    return (
        AnalyticGaussianField2D(
            means=torch.stack(means),
            covariances=torch.stack(covariances),
            density_amplitudes=torch.stack(densities),
            rgb_amplitudes=torch.stack(rgbs),
        ),
        torch.tensor(parents, dtype=torch.long),
    )


def _association_cell(
    task: Mapping[str, Any],
    cell: ExperimentCell,
) -> dict[str, object]:
    import torch

    from rtgs.lift.fiber_correspondence import (
        pairwise_bhattacharyya_cost,
        row_softmax_plan,
        unbalanced_sinkhorn_plan,
    )
    from rtgs.lift.field_measurement import association_metrics, normalized_overlap_plan

    started = time.perf_counter()
    gt, cameras = _synthetic_fixture(cell.seed)
    delete_rate = float(cell.factors["component_delete_rate"])
    split_rate = float(cell.factors["component_split_rate"])
    scores = []
    purities = []
    coverages = []
    real_masses = []
    fixed_point_residuals = []
    candidate_violation_masses = []
    finite = True
    for view_index, camera in enumerate(cameras[:4]):
        predicted = _analytic_projection(gt, camera)
        observed, observed_parent_ids = _recomponentized_view(
            predicted,
            split_rate=split_rate,
            delete_rate=delete_rate,
            seed=cell.seed + 1009 * view_index,
        )
        predicted_parent_ids = torch.arange(predicted.n, dtype=torch.long)
        cost = pairwise_bhattacharyya_cost(
            predicted.means,
            predicted.covariances,
            observed.means,
            observed.covariances,
            residual_variance=1.0,
        )
        candidate = cost <= 12.0
        plan = None
        if cell.arm == "field_no_association":
            mass = normalized_overlap_plan(predicted, observed, dustbin=1e-8)
        elif cell.arm == "row_softmax_dustbin":
            plan = row_softmax_plan(
                cost,
                temperature=1.0,
                dustbin_cost=8.0,
                candidate_mask=candidate,
            )
            mass = plan.real_mass
        else:
            track_capacity = (
                predicted.density_amplitudes
                if cell.arm in {"uot_field_mass_capacity", "uot_shuffled_candidate_negative"}
                else torch.ones(predicted.n, dtype=cost.dtype)
            )
            observation_capacity = (
                observed.density_amplitudes
                if cell.arm in {"uot_field_mass_capacity", "uot_shuffled_candidate_negative"}
                else torch.ones(observed.n, dtype=cost.dtype)
            )
            plan_cost = cost
            plan_candidate = candidate
            if cell.arm == "uot_shuffled_candidate_negative":
                # Corrupt the actual track-to-observation geometry/gate while retaining the
                # unshuffled ground-truth labels used for scoring.
                permutation = torch.roll(torch.arange(predicted.n), shifts=3)
                plan_cost = cost[permutation]
                plan_candidate = candidate[permutation]
            plan = unbalanced_sinkhorn_plan(
                plan_cost,
                track_capacities=track_capacity,
                observation_capacities=observation_capacity,
                temperature=1.0,
                marginal_penalty=8.0,
                dustbin_cost=8.0,
                iterations=80,
                tolerance=1e-7,
                candidate_mask=plan_candidate,
            )
            mass = plan.real_mass
            candidate = plan_candidate
        metrics = association_metrics(mass, predicted_parent_ids, observed_parent_ids)
        correct = predicted_parent_ids[:, None] == observed_parent_ids[None, :]
        supported_parents = ((mass * correct).sum(dim=1) > 1e-10).sum()
        coverage = float(supported_parents / predicted.n)
        score = metrics.purity * coverage
        purities.append(metrics.purity)
        coverages.append(coverage)
        scores.append(score)
        real_masses.append(float(mass.sum()))
        candidate_violation_masses.append(float(mass[~candidate].sum()))
        if plan is not None and plan.fixed_point_residual is not None:
            fixed_point_residuals.append(float(plan.fixed_point_residual))
        view_finite = bool(torch.isfinite(mass).all()) and bool((mass >= 0).all())
        if plan is not None:
            view_finite = view_finite and _plan_is_finite_nonnegative(plan)
        finite = finite and view_finite
        if cell.arm.startswith("uot_"):
            gates = task["frozen_configuration"]["invariant_gates"]
            failures = []
            if not view_finite:
                failures.append("transport or dustbin non-finite/negative")
            if float(mass.sum()) < float(gates["minimum_transport_real_mass"]):
                failures.append("transport real mass")
            if (
                plan is None
                or plan.fixed_point_residual is None
                or float(plan.fixed_point_residual)
                > float(gates["transport_fixed_point_residual_tolerance"])
            ):
                failures.append("transport fixed point")
            if candidate_violation_masses[-1] > float(gates["candidate_mass_tolerance"]):
                failures.append("candidate gate")
            if failures:
                raise RuntimeError(
                    f"hard association invariant violation in view {view_index}: "
                    + ", ".join(failures)
                )
    metrics = {
        "track_precision_times_coverage": statistics.mean(scores),
        "association_purity": statistics.mean(purities),
        "association_parent_coverage": statistics.mean(coverages),
        "transport_real_mass": min(real_masses),
        "transport_finite": finite,
        "transport_fixed_point_residual": (
            max(fixed_point_residuals) if fixed_point_residuals else 0.0
        ),
        "candidate_gate_violation_mass": max(candidate_violation_masses),
        "wall_seconds": time.perf_counter() - started,
    }
    return metrics


def _mask_scene(
    cell: ExperimentCell,
    *,
    spurious_components: int,
) -> tuple[Any, int]:
    """Build train fields with nuisance Gaussians and a clean held-out Gaussian field."""

    import torch

    from rtgs.core.observation2d import GaussianObservationField
    from rtgs.core.sh import sh_to_rgb
    from rtgs.data.field_inputs import SceneFits

    gt, cameras = _synthetic_fixture(cell.seed, aspect_ratio=4.0)
    colors = sh_to_rgb(gt.sh[:, 0]).to(torch.float64)
    false_positive = float(cell.factors["mask_false_positive_rate"])
    false_negative = float(cell.factors["mask_false_negative_rate"])
    prevalence = gt.n / (gt.n + spurious_components)

    def posterior(observed_positive: bool) -> float:
        if observed_positive:
            numerator = (1.0 - false_negative) * prevalence
            denominator = numerator + false_positive * (1.0 - prevalence)
        else:
            numerator = false_negative * prevalence
            denominator = numerator + (1.0 - false_positive) * (1.0 - prevalence)
        return numerator / denominator if denominator > 0 else float(observed_positive)

    observations = []
    alphas = []
    for view_index, camera in enumerate(cameras):
        projected = _analytic_projection(gt, camera)
        means = projected.means
        covariances = projected.covariances
        amplitudes = projected.density_amplitudes
        view_colors = colors
        truth = torch.ones(gt.n, dtype=torch.bool)
        if view_index < 4:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(cell.seed + 7919 * (view_index + 1))
            nuisance_means = torch.stack(
                [
                    6.0
                    + (camera.width - 12.0)
                    * torch.rand(spurious_components, generator=generator, dtype=torch.float64),
                    6.0
                    + (camera.height - 12.0)
                    * torch.rand(spurious_components, generator=generator, dtype=torch.float64),
                ],
                dim=-1,
            )
            nuisance_covariances = (
                torch.eye(2, dtype=torch.float64)[None].repeat(spurious_components, 1, 1) * 3.0
            )
            nuisance_amplitudes = 0.45 + 0.35 * torch.rand(
                spurious_components,
                generator=generator,
                dtype=torch.float64,
            )
            nuisance_colors = 0.1 + 0.8 * torch.rand(
                (spurious_components, 3),
                generator=generator,
                dtype=torch.float64,
            )
            means = torch.cat([means, nuisance_means])
            covariances = torch.cat([covariances, nuisance_covariances])
            amplitudes = torch.cat([amplitudes, nuisance_amplitudes])
            view_colors = torch.cat([view_colors, nuisance_colors])
            truth = torch.cat([truth, torch.zeros(spurious_components, dtype=torch.bool)])
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
        axis = eigenvectors[:, :, 0]
        observations.append(
            GaussianObservationField(
                width=camera.width,
                height=camera.height,
                means=means,
                log_scales=0.5 * eigenvalues.log(),
                rotations=torch.atan2(axis[:, 1], axis[:, 0]),
                colors=view_colors,
                amplitudes=amplitudes,
                blend_mode="additive",
                aa_dilation=0.0,
                provider="synthetic_fixture",
                view_id=f"synthetic_mask_{view_index}",
            )
        )
        if view_index == 4:
            alphas.append(torch.ones((camera.height, camera.width), dtype=torch.float64))
            continue
        corrupted = truth.clone()
        positive = torch.nonzero(truth, as_tuple=False).flatten()
        negative = torch.nonzero(~truth, as_tuple=False).flatten()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(cell.seed + 104729 * (view_index + 1))
        fn_count = round(false_negative * positive.numel())
        fp_count = round(false_positive * negative.numel())
        if fn_count:
            corrupted[
                positive[torch.randperm(positive.numel(), generator=generator)[:fn_count]]
            ] = False
        if fp_count:
            corrupted[
                negative[torch.randperm(negative.numel(), generator=generator)[:fp_count]]
            ] = True
        alpha = torch.zeros((camera.height, camera.width), dtype=torch.float64)
        for component, xy in enumerate(means):
            value = (
                float(corrupted[component])
                if cell.arm != "probability"
                else posterior(bool(corrupted[component]))
            )
            x = int(xy[0].floor().clamp(0, camera.width - 1))
            y = int(xy[1].floor().clamp(0, camera.height - 1))
            alpha[
                max(0, y - 1) : min(camera.height, y + 2), max(0, x - 1) : min(camera.width, x + 2)
            ] = torch.maximum(
                alpha[
                    max(0, y - 1) : min(camera.height, y + 2),
                    max(0, x - 1) : min(camera.width, x + 2),
                ],
                alpha.new_tensor(value),
            )
        alphas.append(alpha if cell.arm == "probability" else alpha.bool())
    return (
        SceneFits(
            observations=tuple(observations),
            cameras=cameras,
            view_names=tuple(f"synthetic_mask_{index}" for index in range(len(cameras))),
            alphas=tuple(alphas),
            train_view_indices=(0, 1, 2, 3),
            heldout_view_indices=(4,),
            bounds_hint=(torch.zeros(3, dtype=torch.float64), 5.0),
            geometry_is_train_only=True,
            name=f"synthetic-mask-{cell.seed}",
        ),
        gt.n,
    )


def _mask_cell(task: Mapping[str, Any], cell: ExperimentCell) -> dict[str, object]:
    import torch

    from rtgs.lift.field_lifter import FieldLiftConfig, FieldLifter
    from rtgs.lift.field_refit import FieldRefitConfig

    started = time.perf_counter()
    mask_config = task["frozen_configuration"]["mask_field_test"]
    fits, true_components = _mask_scene(
        cell,
        spurious_components=int(mask_config["spurious_components_per_train_view"]),
    )
    result = FieldLifter(
        FieldLiftConfig(
            placement_mode="fixed_bounded_midpoint",
            compute_dtype="float64",
            max_tracks=int(mask_config["max_tracks"]),
            max_train_views=4,
            depth_samples=8,
            min_views=2,
            background_fraction=0.0,
            mask_mode=str(cell.arm),
            topology_rounds=0,
            validation_sample_cap=int(mask_config["validation_sample_cap"]),
            seed=cell.seed,
            refit=FieldRefitConfig(
                iterations=int(mask_config["refit_iterations"]),
                appearance_start=int(mask_config["refit_iterations"]),
                visibility_refresh=2,
                chunk_size=32,
            ),
        )
    ).fit(fits)
    heldout_access_count = _heldout_fit_access_count(task, result, fits)
    invariants = {
        **_enforce_result_invariants(task, result, association_required=False),
        "hard_invariant_checked_fit_count": 1,
    }
    source_components = result.placement.fiber.source_component_indices
    true_source = source_components < true_components
    support = result.placement.source_support
    true_support = float(support[true_source].sum())
    precision = true_support / max(float(support.sum()), torch.finfo(support.dtype).tiny)
    coverage = true_support / (true_components * len(fits.train_view_indices))
    heldout = result.semantic_validation.heldout
    assert heldout is not None
    opacity_error = float(
        (result.placement.render_opacity - result.placement.render_opacity[0]).abs().amax()
    )
    return {
        "support_precision_times_coverage": precision * coverage,
        "support_precision": precision,
        "support_coverage": coverage,
        "heldout_field_density_mse": heldout.density_mse,
        "heldout_field_rgb_mse": heldout.rgb_mse,
        "render_opacity_max_error": opacity_error,
        "render_opacity_equal": float(opacity_error == 0.0),
        "heldout_fit_access_count": heldout_access_count,
        "heldout_fit_checked_fit_count": 1,
        **invariants,
        "wall_seconds": time.perf_counter() - started,
    }


def _synthetic_scene(seed: int) -> Any:
    import torch

    from rtgs.core.observation2d import GaussianObservationField
    from rtgs.core.sh import sh_to_rgb
    from rtgs.data.field_inputs import SceneFits
    from rtgs.render.projection import project_gaussians_ewa

    gt, cameras = _synthetic_fixture(seed, aspect_ratio=16.0)
    colors = sh_to_rgb(gt.sh[:, 0])
    observations = []
    for index, camera in enumerate(cameras):
        projection = project_gaussians_ewa(gt, camera)
        eigenvalues, eigenvectors = torch.linalg.eigh(projection.covariances2d)
        axis = eigenvectors[:, :, 0]
        observations.append(
            GaussianObservationField(
                width=camera.width,
                height=camera.height,
                means=projection.means2d,
                log_scales=0.5 * eigenvalues.log(),
                rotations=torch.atan2(axis[:, 1], axis[:, 0]),
                colors=colors,
                amplitudes=gt.opacity,
                blend_mode="additive",
                aa_dilation=0.0,
                provider="synthetic_fixture",
                view_id=f"synthetic_{index}",
            )
        )
    return SceneFits(
        observations=tuple(observations),
        cameras=cameras,
        view_names=tuple(f"synthetic_{index}" for index in range(len(cameras))),
        alphas=tuple(None for _ in cameras),
        train_view_indices=(0, 1, 2, 3),
        heldout_view_indices=(4,),
        bounds_hint=(torch.zeros(3), 1.2),
        geometry_is_train_only=True,
        name=f"synthetic-{seed}",
    )


def _pipeline_synthetic_cell(
    task: Mapping[str, Any],
    cell: ExperimentCell,
) -> dict[str, object]:
    from rtgs.lift.field_lifter import FieldLiftConfig, FieldLifter
    from rtgs.lift.field_refit import FieldRefitConfig
    from rtgs.lift.probabilistic_pipeline import (
        ProbabilisticFieldPipelineConfig,
        run_probabilistic_field_pipeline,
    )

    pipeline_config = task["frozen_configuration"]["pipeline"]
    fits = _synthetic_scene(cell.seed)
    progressive = cell.arm == "progressive_then_full_cleanup"
    topology = cell.stage == "topology_factorial"
    refit = FieldRefitConfig(
        iterations=int(pipeline_config["refit_iterations"]),
        appearance_start=int(pipeline_config["appearance_start"]),
        visibility_refresh=5,
        chunk_size=32,
        view_schedule="progressive" if progressive else "all",
        progressive_start_views=2,
        full_view_cleanup_iterations=(
            int(
                cell.factors.get(
                    "final_cleanup_iterations",
                    pipeline_config["full_view_cleanup_iterations"],
                )
            )
            if progressive
            else 0
        ),
    )
    config = FieldLiftConfig(
        placement_mode=str(pipeline_config["placement_mode"]),
        compute_dtype=str(pipeline_config["compute_dtype"]),
        max_tracks=int(pipeline_config["max_tracks"]),
        max_train_views=int(pipeline_config["max_train_views"]),
        target_component_cap=int(pipeline_config["target_component_cap"]),
        depth_samples=int(pipeline_config["depth_samples"]),
        min_views=int(pipeline_config["min_views"]),
        background_fraction=0.0,
        mask_mode="none",
        topology_rounds=int(pipeline_config["topology_rounds"]) if topology else 0,
        topology_split_mode=(
            "projection_nonlinearity"
            if cell.arm == "projection_nonlinearity"
            else "largest_density_mass"
        ),
        parsimony_per_component=float(pipeline_config["parsimony_per_component"]),
        validation_sample_cap=int(pipeline_config["validation_sample_cap"]),
        seed=cell.seed,
        refit=refit,
    )
    started = time.perf_counter()
    pipeline = None
    if cell.stage == "independent_half_stability":
        pipeline = run_probabilistic_field_pipeline(
            fits,
            ProbabilisticFieldPipelineConfig(
                lift=config,
                independent_half_validation=True,
                minimum_half_views=2,
                match_radius_fraction=0.10,
            ),
        )
        result = pipeline.reconstruction
        stability = pipeline.stability
    else:
        result = FieldLifter(config).fit(fits)
        stability = None
    wall = time.perf_counter() - started
    fit_access_metrics = (
        _pipeline_fit_access_metrics(task, pipeline, fits)
        if pipeline is not None
        else {
            "heldout_fit_access_count": _heldout_fit_access_count(task, result, fits),
            "heldout_fit_checked_fit_count": 1,
        }
    )
    heldout = result.semantic_validation.heldout
    assert heldout is not None
    invariants = (
        _enforce_pipeline_result_invariants(task, pipeline, association_required=False)
        if pipeline is not None
        else {
            **_enforce_result_invariants(task, result, association_required=False),
            "hard_invariant_checked_fit_count": 1,
        }
    )
    center = None
    if stability is not None:
        center = stability.center_median
        if not math.isfinite(center):
            center = stability.match_radius
    return {
        "heldout_field_density_mse": heldout.density_mse,
        "heldout_field_rgb_mse": heldout.rgb_mse,
        "refit_wall_seconds": result.refit.elapsed_seconds[-1],
        "pipeline_wall_seconds": wall,
        "final_gaussian_count": result.gaussians.n,
        "independent_half_center_median": center,
        "source_projection_max_error": result.refit.source_projection_max_error,
        "final_active_view_count": result.refit.active_view_counts[-1],
        **fit_access_metrics,
        **invariants,
        "wall_seconds": wall,
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    return float(statistics.median(float(value) for value in values))


def _synthetic_decisions(
    task: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    def selected(stage: str, arm: str, seed: int | None = None) -> list[Mapping[str, Any]]:
        return [
            item
            for item in records
            if item["stage"] == stage
            and item["arm"] == arm
            and (seed is None or int(item["seed"]) == seed)
        ]

    def lookup(
        items: Sequence[Mapping[str, Any]], keys: Sequence[str]
    ) -> dict[tuple[object, ...], Any]:
        return {tuple(item["factors"][key] for key in keys): item for item in items}

    def mask_dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        left_metrics = left["metrics"]
        right_metrics = right["metrics"]
        comparisons = (
            float(left_metrics["support_precision"]) >= float(right_metrics["support_precision"]),
            float(left_metrics["support_coverage"]) >= float(right_metrics["support_coverage"]),
            float(left_metrics["heldout_field_density_mse"])
            <= float(right_metrics["heldout_field_density_mse"]),
            float(left_metrics["heldout_field_rgb_mse"])
            <= float(right_metrics["heldout_field_rgb_mse"]),
        )
        strict = (
            float(left_metrics["support_precision"]) > float(right_metrics["support_precision"])
            or float(left_metrics["support_coverage"]) > float(right_metrics["support_coverage"])
            or float(left_metrics["heldout_field_density_mse"])
            < float(right_metrics["heldout_field_density_mse"])
            or float(left_metrics["heldout_field_rgb_mse"])
            < float(right_metrics["heldout_field_rgb_mse"])
        )
        return all(comparisons) and strict

    gates = task["frozen_configuration"]["invariant_gates"]
    seeds = sorted({int(item["seed"]) for item in records})
    shape_wins = 0
    association_wins = 0
    mask_wins = 0
    topology_wins = 0
    schedule_wins = 0
    for seed in seeds:
        rank = selected("exact_shape_recovery", "rank_aware_full_covariance", seed)
        source = selected("exact_shape_recovery", "source_footprint", seed)
        eligible_rank = [
            item
            for item in rank
            if float(item["factors"]["center_noise_pixels"]) >= 0
            and float(item["metrics"]["full_rank_fraction"]) == 1.0
        ]
        eligible_source = [
            item
            for item in source
            if (
                float(item["factors"]["baseline_degrees"]),
                float(item["factors"]["aspect_ratio"]),
                float(item["factors"]["center_noise_pixels"]),
            )
            in {
                (
                    float(value["factors"]["baseline_degrees"]),
                    float(value["factors"]["aspect_ratio"]),
                    float(value["factors"]["center_noise_pixels"]),
                )
                for value in eligible_rank
            }
        ]
        if (
            eligible_rank
            and _median(
                [float(item["metrics"]["covariance_relative_frobenius"]) for item in eligible_rank]
            )
            < _median(
                [
                    float(item["metrics"]["covariance_relative_frobenius"])
                    for item in eligible_source
                ]
            )
            and _median(
                [
                    float(item["metrics"]["heldout_covariance_relative_frobenius"])
                    for item in eligible_rank
                ]
            )
            < _median(
                [
                    float(item["metrics"]["heldout_covariance_relative_frobenius"])
                    for item in eligible_source
                ]
            )
        ):
            shape_wins += 1

        field_mass = selected("recomponentized_association", "uot_field_mass_capacity", seed)
        row = selected("recomponentized_association", "row_softmax_dustbin", seed)
        native = selected("recomponentized_association", "field_no_association", seed)
        negative = selected("recomponentized_association", "uot_shuffled_candidate_negative", seed)
        association_keys = ("component_delete_rate", "component_split_rate")
        field_lookup = lookup(field_mass, association_keys)
        row_lookup = lookup(row, association_keys)
        native_lookup = lookup(native, association_keys)
        negative_lookup = lookup(negative, association_keys)
        treatment_scores = []
        row_scores = []
        native_scores = []
        negative_scores = []
        for stratum, treatment_item in field_lookup.items():
            comparison_items = (
                treatment_item,
                row_lookup[stratum],
                native_lookup[stratum],
                negative_lookup[stratum],
            )
            matched_coverage = min(
                float(item["metrics"]["association_parent_coverage"]) for item in comparison_items
            )
            scores = [
                float(item["metrics"]["association_purity"]) * matched_coverage
                for item in comparison_items
            ]
            treatment_scores.append(scores[0])
            row_scores.append(scores[1])
            native_scores.append(scores[2])
            negative_scores.append(scores[3])
        treatment = _median(treatment_scores)
        if (
            treatment > _median(row_scores)
            and treatment > _median(native_scores)
            and treatment >= _median(negative_scores) + 0.10
        ):
            association_wins += 1

        probability = [
            item
            for item in selected("support_mask_factorial", "probability", seed)
            if float(item["factors"]["mask_false_positive_rate"])
            + float(item["factors"]["mask_false_negative_rate"])
            > 0
        ]
        mask_keys = ("mask_false_positive_rate", "mask_false_negative_rate")
        hard_lookup = lookup(selected("support_mask_factorial", "hard", seed), mask_keys)
        none_lookup = lookup(selected("support_mask_factorial", "none", seed), mask_keys)
        probability_wins = sum(
            not mask_dominates(
                hard_lookup[tuple(item["factors"][key] for key in mask_keys)],
                item,
            )
            and not mask_dominates(
                none_lookup[tuple(item["factors"][key] for key in mask_keys)],
                item,
            )
            for item in probability
        )
        if probability_wins >= 2:
            mask_wins += 1

        topology_candidate = selected("topology_factorial", "projection_nonlinearity", seed)[0]
        topology_native = selected("topology_factorial", "largest_density_mass", seed)[0]
        if (
            topology_candidate["metrics"]["final_gaussian_count"]
            == topology_native["metrics"]["final_gaussian_count"]
            and float(topology_candidate["metrics"]["heldout_field_density_mse"])
            < float(topology_native["metrics"]["heldout_field_density_mse"])
            and float(topology_candidate["metrics"]["pipeline_wall_seconds"])
            <= 1.2 * float(topology_native["metrics"]["pipeline_wall_seconds"])
        ):
            topology_wins += 1

        progressive = selected("schedule_factorial", "progressive_then_full_cleanup", seed)[0]
        all_views = selected("schedule_factorial", "all", seed)[0]
        density_denominator = max(
            abs(float(all_views["metrics"]["heldout_field_density_mse"])), 1e-12
        )
        rgb_denominator = max(abs(float(all_views["metrics"]["heldout_field_rgb_mse"])), 1e-12)
        if (
            float(progressive["metrics"]["refit_wall_seconds"])
            <= 0.9 * float(all_views["metrics"]["refit_wall_seconds"])
            and abs(
                float(progressive["metrics"]["heldout_field_density_mse"])
                - float(all_views["metrics"]["heldout_field_density_mse"])
            )
            / density_denominator
            <= 0.01
            and abs(
                float(progressive["metrics"]["heldout_field_rgb_mse"])
                - float(all_views["metrics"]["heldout_field_rgb_mse"])
            )
            / rgb_denominator
            <= 0.01
            and int(progressive["metrics"]["final_active_view_count"]) == 4
        ):
            schedule_wins += 1
    transport_records = [
        item
        for item in records
        if item["stage"] == "recomponentized_association" and item["arm"].startswith("uot_")
    ]
    pipeline_records = [
        item
        for item in records
        if item["stage"]
        in {
            "support_mask_factorial",
            "topology_factorial",
            "schedule_factorial",
            "independent_half_stability",
        }
    ]
    all_metrics_finite = all(
        all(
            not isinstance(value, (int, float)) or math.isfinite(float(value))
            for value in item["metrics"].values()
            if value is not None
        )
        for item in records
    )
    transport_fixed_point_max = max(
        float(item["metrics"]["transport_fixed_point_residual"]) for item in transport_records
    )
    candidate_violation_max = max(
        float(item["metrics"]["candidate_gate_violation_mass"]) for item in transport_records
    )
    source_mean_max = max(
        float(item["metrics"]["source_mean_max_error"]) for item in pipeline_records
    )
    source_covariance_relative_max = max(
        float(item["metrics"]["source_covariance_relative_error"]) for item in pipeline_records
    )
    split_density_max = max(
        float(item["metrics"]["split_density_mass_error"]) for item in pipeline_records
    )
    split_optical_max = max(
        float(item["metrics"]["split_optical_thickness_error"]) for item in pipeline_records
    )
    invariants = {
        "all_metrics_finite": all_metrics_finite,
        "transport_finite": all(
            bool(item["metrics"]["transport_finite"]) for item in transport_records
        ),
        "transport_positive_real_mass": all(
            float(item["metrics"]["transport_real_mass"])
            >= float(gates["minimum_transport_real_mass"])
            for item in transport_records
        ),
        "transport_fixed_point": transport_fixed_point_max
        <= float(gates["transport_fixed_point_residual_tolerance"]),
        "candidate_gate_exact_zero": candidate_violation_max
        <= float(gates["candidate_mass_tolerance"]),
        "source_projection": source_mean_max <= float(gates["source_projection_max_error"]),
        "source_covariance": source_covariance_relative_max
        <= float(gates["source_covariance_relative_error"]),
        "split_density_mass": split_density_max <= float(gates["split_density_mass_tolerance"]),
        "split_optical_thickness": split_optical_max
        <= float(gates["split_optical_thickness_tolerance"]),
        "heldout_fit_isolation": all(
            int(item["metrics"]["heldout_fit_access_count"])
            == int(gates["heldout_fit_access_count"])
            for item in pipeline_records
        ),
        "mask_render_opacity_equal": all(
            float(item["metrics"].get("render_opacity_equal", 1.0)) == 1.0
            for item in pipeline_records
            if item["stage"] == "support_mask_factorial"
        ),
        "progressive_final_all_views": all(
            int(item["metrics"]["final_active_view_count"]) == 4
            for item in pipeline_records
            if item["stage"] == "schedule_factorial"
            and item["arm"] == "progressive_then_full_cleanup"
        ),
        "schedule_fresh_process": all(
            bool(item.get("fresh_process"))
            for item in pipeline_records
            if item["stage"] == "schedule_factorial"
        ),
    }
    hard_invariants_passed = all(invariants.values())
    return {
        "invariants": invariants,
        "invariant_measurements": {
            "transport_fixed_point_residual_max": transport_fixed_point_max,
            "candidate_gate_violation_mass_max": candidate_violation_max,
            "source_mean_max_error": source_mean_max,
            "source_covariance_relative_error_max": source_covariance_relative_max,
            "split_density_mass_error_max": split_density_max,
            "split_optical_thickness_error_max": split_optical_max,
        },
        "hard_invariants_passed": hard_invariants_passed,
        "shape": {"seed_wins": shape_wins, "passed": shape_wins >= 2},
        "association": {
            "seed_wins": association_wins,
            "passed": association_wins >= 2
            and all(
                invariants[key]
                for key in invariants
                if key.startswith(("transport_", "dustbin_", "candidate_"))
            ),
        },
        "mask": {"seed_wins": mask_wins, "passed": mask_wins >= 2},
        "topology": {"seed_wins": topology_wins, "passed": topology_wins >= 2},
        "schedule": {"seed_wins": schedule_wins, "passed": schedule_wins >= 2},
        "combined_interpretation": "descriptive_only",
    }


def _synthetic_cell_metrics(
    task: Mapping[str, Any],
    cell: ExperimentCell,
) -> dict[str, object]:
    if cell.stage == "exact_shape_recovery":
        return _shape_cell(cell)
    if cell.stage == "recomponentized_association":
        return _association_cell(task, cell)
    if cell.stage == "support_mask_factorial":
        return _mask_cell(task, cell)
    return _pipeline_synthetic_cell(task, cell)


def _synthetic_cell_worker(
    task: Mapping[str, Any],
    run: Path,
    cell_id: str,
) -> None:
    """Execute one timed schedule cell in its own guarded OS process."""

    matches = [cell for cell in compile_cell_plan(task) if cell.cell_id == cell_id]
    if len(matches) != 1 or matches[0].stage != "schedule_factorial":
        raise ValueError("synthetic-cell worker accepts exactly one frozen schedule cell")
    cell = matches[0]
    output = run / "synthetic" / "fresh_process_cells" / cell.cell_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite synthetic cell {output}")
    output.mkdir(parents=True)
    guard = NoImageGuard()
    started = time.perf_counter()
    try:
        with guard:
            import torch

            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            metrics = _synthetic_cell_metrics(task, cell)
            guard_record = guard.record()
            if not guard_record["passed"]:
                raise RuntimeError(f"synthetic-cell input boundary failed: {guard_record}")
        _write_json_new(
            output / "result.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "cell": asdict(cell),
                "metrics": metrics,
                "fresh_process": True,
                "pid": os.getpid(),
            },
        )
        _write_json_new(
            output / "resource_receipt.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "cell_id": cell.cell_id,
                "wall_seconds": time.perf_counter() - started,
                "ru_maxrss_bytes": _peak_rss_bytes(),
                "cpu_threads": 1,
                "cuda_used": False,
                "guard": guard_record,
            },
        )
    except Exception as error:
        _write_failure_new(
            output,
            phase="fresh_process_synthetic_schedule_cell",
            error=error,
            context={"cell_id": cell.cell_id},
        )
        raise


def _synthetic_worker(task: Mapping[str, Any], run: Path) -> None:
    output = run / "synthetic"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite synthetic worker output {output}")
    output.mkdir(parents=True)
    guard = NoImageGuard()
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    phase = "synthetic_initialization"
    try:
        with guard:
            import torch

            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            for index, cell in enumerate(compile_cell_plan(task), start=1):
                if cell.stage == "calibrated_compact_operability":
                    continue
                phase = f"synthetic_cell:{cell.cell_id}"
                fresh_process = cell.stage == "schedule_factorial"
                if fresh_process:
                    subprocess.run(
                        _internal_command(
                            ROOT / TASK_RELATIVE,
                            run,
                            synthetic_cell_id=cell.cell_id,
                        ),
                        cwd=ROOT,
                        check=True,
                    )
                    child = _strict_json(
                        output / "fresh_process_cells" / cell.cell_id / "result.json"
                    )
                    metrics = child["metrics"]
                else:
                    metrics = _synthetic_cell_metrics(task, cell)
                records.append(
                    {
                        "cell_id": cell.cell_id,
                        "stage": cell.stage,
                        "arm": cell.arm,
                        "seed": cell.seed,
                        "factors": cell.factors,
                        "metrics": metrics,
                        "fresh_process": fresh_process,
                    }
                )
                if index % 50 == 0:
                    print(f"synthetic cells completed: {index}", flush=True)
            guard_record = guard.record()
            if not guard_record["passed"]:
                raise RuntimeError(f"synthetic input boundary failed: {guard_record}")
        decisions = _synthetic_decisions(task, records)
        _write_json_new(
            output / "synthetic_results.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "cell_count": len(records),
                "decisions": decisions,
                "cells": records,
            },
        )
        _write_json_new(
            output / "resource_receipt.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "wall_seconds": time.perf_counter() - started,
                "ru_maxrss_bytes": _peak_rss_bytes(),
                "cpu_threads": 1,
                "cuda_used": False,
                "guard": guard_record,
            },
        )
        if not decisions["hard_invariants_passed"]:
            raise RuntimeError("synthetic hard invariants failed; calibrated execution denied")
    except Exception as error:
        _write_failure_new(
            output,
            phase=phase,
            error=error,
            context={"completed_cells": len(records)},
        )
        raise


def _scaled_camera(camera: Any, maximum_side: int = 256) -> Any:
    from rtgs.core.camera import Camera

    scale = min(1.0, maximum_side / max(camera.width, camera.height))
    width = max(1, round(camera.width * scale))
    height = max(1, round(camera.height * scale))
    return Camera(
        fx=camera.fx * scale,
        fy=camera.fy * scale,
        cx=camera.cx * scale,
        cy=camera.cy * scale,
        width=width,
        height=height,
        R=camera.R,
        t=camera.t,
    )


def _image_from_render(model: Any, camera: Any) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    from rtgs.render.torch_ref import TorchRasterizer

    with torch.no_grad():
        color = TorchRasterizer(row_chunk=32).render(model, camera).color.clamp(0.0, 1.0)
    return Image.fromarray((color.detach().cpu().numpy() * 255.0).round().astype(np.uint8))


def _labeled_panel(image: Any, label: str) -> Any:
    from PIL import Image, ImageDraw

    header = 26
    panel = Image.new("RGB", (image.width, image.height + header), "white")
    panel.paste(image.convert("RGB"), (0, header))
    ImageDraw.Draw(panel).text((6, 6), label, fill="black")
    return panel


def _horizontal_panels(panels: Sequence[Any]) -> Any:
    from PIL import Image

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    result = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        result.paste(panel, (x, 0))
        x += panel.width
    return result


def _orbit_cameras(reference: Any, center: Any, radius: float, *, elevation: bool) -> list[Any]:
    import torch

    from rtgs.core.camera import Camera

    result = []
    for index in range(8):
        angle = 2.0 * math.pi * index / 8
        height = 0.22 * radius * math.sin(2.0 * angle) if elevation else 0.0
        eye = center + center.new_tensor(
            [radius * math.sin(angle), height, radius * math.cos(angle)]
        )
        camera = Camera.look_at(eye, center, width=256, height=256)
        fov_x = 2.0 * math.atan(reference.width / (2.0 * reference.fx))
        camera.fx = 0.5 * camera.width / math.tan(0.5 * fov_x)
        camera.fy = camera.fx
        camera.cx = camera.width / 2
        camera.cy = camera.height / 2
        result.append(camera.to(torch.device("cpu")))
    return result


def _save_dataset_presentation(
    task: Mapping[str, Any],
    source_run: Path,
    publish_root: Path,
    dataset: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, object]:
    import torch
    from PIL import Image

    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.data.compact_views import CompactDataset

    dataset_id = str(dataset["id"])
    directory = publish_root / "datasets" / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    methods = []
    loaded: dict[str, tuple[Any, Any]] = {}
    for arm in CALIBRATED_ARMS:
        cell = source_run / _cell_relative(dataset_id, seed, arm, False)
        initial_target = directory / f"{arm}_gaussians_init.ply"
        final_target = directory / f"{arm}_gaussians.ply"
        shutil.copy2(cell / "gaussians_init.ply", initial_target)
        shutil.copy2(cell / "gaussians.ply", final_target)
        methods.append(
            {
                "name": arm.replace("_", " "),
                "initial": initial_target.name,
                "final": final_target.name,
            }
        )
        loaded[arm] = (Gaussians3D.load_ply(initial_target), Gaussians3D.load_ply(final_target))
    viewer_manifest = directory / "viewer_comparison.json"
    _write_json_new(
        viewer_manifest,
        {"schema": "rtgs.viewer-comparison.v1", "methods": methods},
    )
    compact = CompactDataset.load(
        (ROOT / dataset["compact_manifest"]).parent,
        device="cpu",
        load_alpha=False,
    )
    reference = _scaled_camera(compact.views[0].camera)
    panels = [
        _labeled_panel(_image_from_render(loaded["native_controls"][0], reference), "native init"),
        _labeled_panel(_image_from_render(loaded["native_controls"][1], reference), "native final"),
        _labeled_panel(
            _image_from_render(loaded["all_candidate_mechanisms"][1], reference),
            "candidate final",
        ),
    ]
    contact = _horizontal_panels(panels)
    contact.save(directory / "reconstruction_contact_sheet.png")
    reconstruction_frames = [panel.convert("P", palette=Image.Palette.ADAPTIVE) for panel in panels]
    reconstruction_frames[0].save(
        directory / "reconstruction.gif",
        save_all=True,
        append_images=reconstruction_frames[1:],
        duration=800,
        loop=0,
        optimize=False,
    )
    final_models = [
        ("native", loaded["native_controls"][1]),
        ("candidate", loaded["all_candidate_mechanisms"][1]),
    ]
    joined_means = torch.cat([model.means.to(torch.float64) for _label, model in final_models])
    center = joined_means.median(dim=0).values
    camera_positions = torch.stack(
        [view.camera.position.to(torch.float64) for view in compact.views]
    )
    radius = float(torch.linalg.vector_norm(camera_positions - center, dim=-1).median())
    radius = max(radius, 0.5)
    for name, elevation in (("novel_orbit.gif", False), ("novel_elevation.gif", True)):
        frames = []
        for camera in _orbit_cameras(reference, center, radius, elevation=elevation):
            frame = _horizontal_panels(
                [
                    _labeled_panel(_image_from_render(model, camera), label)
                    for label, model in final_models
                ]
            )
            frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))
        frames[0].save(
            directory / name,
            save_all=True,
            append_images=frames[1:],
            duration=140,
            loop=0,
            optimize=False,
        )
    artifacts = []
    for path in sorted(directory.iterdir()):
        if path.is_file():
            artifacts.append(
                {
                    "path": path.relative_to(publish_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    receipt = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "dataset_id": dataset_id,
        "status": "PASS",
        "source_rgb_or_external_mask_opened": False,
        "renderer": "rtgs.render.torch_ref.TorchRasterizer",
        "representative_seed": seed,
        "artifacts": artifacts,
    }
    _write_json_new(directory / "presentation_receipt.json", receipt)
    return receipt


def _measured_cells(task: Mapping[str, Any], run: Path) -> list[dict[str, Any]]:
    records = []
    for dataset in task["datasets"]:
        for seed in task["seeds"]:
            for arm in CALIBRATED_ARMS:
                cell = run / _cell_relative(str(dataset["id"]), int(seed), arm, False)
                if not cell.is_dir():
                    raise FileNotFoundError(f"missing measured calibrated cell {cell}")
                summary = _strict_json(cell / "summary.json")
                resource_record = _strict_json(cell / "resource_receipt.json")
                boundary = _strict_json(cell / "input_boundary_receipt.json")
                if summary.get("warmup") is not False or not boundary["guard"]["passed"]:
                    raise ValueError(f"invalid measured calibrated cell {cell}")
                summary["metrics"]["peak_rss_bytes"] = resource_record["ru_maxrss_bytes"]
                summary["metrics"]["worker_wall_seconds"] = resource_record["wall_seconds"]
                summary["metrics"]["process_wall_seconds"] = resource_record["process_wall_seconds"]
                summary["metrics"]["input_bytes"] = resource_record["input_bytes"]
                summary["metrics"]["output_bytes"] = resource_record["output_bytes"]
                for stage, value in resource_record["stage_wall_seconds"].items():
                    summary["metrics"][f"stage_{stage}_seconds"] = value
                records.append(
                    {
                        "dataset": dict(dataset),
                        "seed": int(seed),
                        "arm": arm,
                        "cell": cell,
                        "summary": summary,
                        "resource": resource_record,
                        "boundary": boundary,
                    }
                )
    return records


def _history_bundle(
    task: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    records = []
    markers = []
    stages = list(task["stages"])
    stage_labels = {item["id"]: item["label"] for item in stages}
    final_stage = stages[-1]["id"]
    for item in cells:
        summary = item["summary"]
        dataset_id = str(item["dataset"]["id"])
        arm = str(item["arm"])
        seed = int(item["seed"])
        history = [float(value) for value in summary["objective_history"]]
        elapsed = [float(value) for value in summary["refit_elapsed_seconds"]]
        active = [int(value) for value in summary["active_view_counts"]]
        end_step = len(history) - 1
        worker_wall = float(item["resource"]["wall_seconds"])
        refit_offset = max(0.0, worker_wall - elapsed[-1])
        for stage in stages:
            stage_id = stage["id"]
            start_step = 0
            end = end_step if stage_id == final_stage else 0
            start_wall = 0.0
            end_wall = worker_wall if stage_id == final_stage else 0.0
            markers.extend(
                [
                    {
                        "step": start_step,
                        "wall_seconds": start_wall,
                        "stage": stage_id,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "boundary": "start",
                        "label": stage_labels[stage_id],
                    },
                    {
                        "step": end,
                        "wall_seconds": end_wall,
                        "stage": stage_id,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "boundary": "end",
                        "label": stage_labels[stage_id],
                    },
                ]
            )
            records.append(
                {
                    "step": end,
                    "wall_seconds": end_wall,
                    "stage": stage_id,
                    "dataset_id": dataset_id,
                    "arm_id": arm,
                    "seed": seed,
                    "split": "diagnostic",
                    "metric_id": "stage_wall_seconds",
                    "value": worker_wall if stage_id == final_stage else 0.0,
                }
            )
        for step, (objective, refit_elapsed, active_views) in enumerate(
            zip(history, elapsed, active, strict=True)
        ):
            wall = min(worker_wall, refit_offset + refit_elapsed)
            records.extend(
                [
                    {
                        "step": step,
                        "wall_seconds": wall,
                        "stage": final_stage,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "split": "train",
                        "metric_id": "refit_objective",
                        "value": objective,
                    },
                    {
                        "step": step,
                        "wall_seconds": wall,
                        "stage": final_stage,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "split": "diagnostic",
                        "metric_id": "active_view_count",
                        "value": active_views,
                    },
                    {
                        "step": step,
                        "wall_seconds": wall,
                        "stage": final_stage,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "split": "diagnostic",
                        "metric_id": "refit_elapsed_seconds",
                        "value": refit_elapsed,
                    },
                ]
            )
    return {
        "schema_version": 2,
        "records": records,
        "metric_metadata": {
            "stage_wall_seconds": {
                "label": "Stage wall time",
                "unit": "seconds",
                "group": "runtime",
                "direction": "descriptive",
            },
            "refit_objective": {
                "label": "Analytic training-field objective",
                "unit": "normalized objective",
                "group": "convergence",
                "direction": "lower",
            },
            "active_view_count": {
                "label": "Active training views",
                "unit": "views",
                "group": "schedule",
                "direction": "descriptive",
            },
            "refit_elapsed_seconds": {
                "label": "Refit elapsed time",
                "unit": "seconds",
                "group": "runtime",
                "direction": "lower",
            },
        },
        "stage_markers": markers,
    }


def _metric_metadata(metric_id: str) -> dict[str, str]:
    metadata = {
        "heldout_field_rgb_mse": (
            "Held-out capped field RGB-numerator MSE",
            "MSE",
            "quality",
            "lower",
        ),
        "heldout_field_density_mse": (
            "Held-out capped field density MSE",
            "MSE",
            "quality",
            "lower",
        ),
        "placement_heldout_rgb_mse": (
            "Placement held-out RGB-numerator MSE",
            "MSE",
            "quality",
            "lower",
        ),
        "placement_heldout_density_mse": (
            "Placement held-out density MSE",
            "MSE",
            "quality",
            "lower",
        ),
        "refit_wall_seconds": ("Refit wall time", "seconds", "runtime", "lower"),
        "pipeline_wall_seconds": ("Pipeline wall time", "seconds", "runtime", "lower"),
        "worker_wall_seconds": ("Worker wall time", "seconds", "runtime", "lower"),
        "process_wall_seconds": ("Process wall time", "seconds", "runtime", "lower"),
        "input_bytes": ("Sealed input bytes", "bytes", "resources", "descriptive"),
        "output_bytes": ("Serialized raw-output bytes", "bytes", "resources", "descriptive"),
        "stage_compact_loading_seconds": (
            "Compact loading wall time",
            "seconds",
            "runtime",
            "lower",
        ),
        "stage_field_fit_seconds": ("Field fit wall time", "seconds", "runtime", "lower"),
        "stage_serialization_seconds": (
            "Serialization wall time",
            "seconds",
            "runtime",
            "lower",
        ),
        "stage_publication_seconds": (
            "Directory publication wall time",
            "seconds",
            "runtime",
            "lower",
        ),
        "peak_rss_bytes": ("Peak resident memory", "bytes", "resources", "lower"),
        "final_gaussian_count": ("Final Gaussian count", "gaussians", "topology", "descriptive"),
        "source_projection_max_error": (
            "Maximum source-projection error",
            "pixels",
            "guardrails",
            "lower",
        ),
        "source_mean_max_error": (
            "Maximum source-mean error",
            "pixels",
            "guardrails",
            "lower",
        ),
        "source_covariance_max_error": (
            "Maximum source-covariance absolute error",
            "pixel squared",
            "guardrails",
            "lower",
        ),
        "source_covariance_relative_error": (
            "Maximum source-covariance relative error",
            "ratio",
            "guardrails",
            "lower",
        ),
        "transport_plan_count": (
            "Transport plan count",
            "plans",
            "guardrails",
            "descriptive",
        ),
        "transport_finite": (
            "Finite non-negative transport plans",
            "indicator",
            "guardrails",
            "higher",
        ),
        "transport_min_real_mass": (
            "Minimum real transport mass",
            "mass units",
            "guardrails",
            "higher",
        ),
        "transport_fixed_point_residual_max": (
            "Maximum transport fixed-point residual",
            "log-potential units",
            "guardrails",
            "lower",
        ),
        "candidate_gate_violation_mass_max": (
            "Maximum transport mass outside the retained candidate gate",
            "mass units",
            "guardrails",
            "lower",
        ),
        "heldout_fit_access_count": (
            "Held-out views accessed during fitting",
            "views",
            "guardrails",
            "lower",
        ),
        "heldout_fit_checked_fit_count": (
            "Realized fits checked for held-out isolation",
            "fits",
            "guardrails",
            "higher",
        ),
        "hard_invariant_checked_fit_count": (
            "Realized fits checked for all hard invariants",
            "fits",
            "guardrails",
            "higher",
        ),
        "split_density_mass_error": (
            "Split density-mass conservation error",
            "mass units",
            "guardrails",
            "lower",
        ),
        "split_optical_thickness_error": (
            "Split optical-thickness conservation error",
            "opacity units",
            "guardrails",
            "lower",
        ),
        "accepted_steps": ("Accepted continuous steps", "steps", "convergence", "higher"),
        "independent_half_center_median": (
            "Independent-half center median",
            "world units",
            "stability",
            "descriptive",
        ),
        "input_components_original_mean": (
            "Original components per view",
            "components",
            "input",
            "descriptive",
        ),
        "input_components_used_mean": (
            "Retained components per view",
            "components",
            "input",
            "descriptive",
        ),
        "embedded_alpha_view_fraction": (
            "Views with embedded alpha",
            "fraction",
            "input",
            "descriptive",
        ),
    }
    label, unit, group, direction = metadata[metric_id]
    return {"label": label, "unit": unit, "group": group, "direction": direction}


def _dataset_summary(
    task: Mapping[str, Any],
    run: Path,
    dataset: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    *,
    port: int,
) -> dict[str, object]:
    dataset_id = str(dataset["id"])
    metric_ids = sorted(
        {
            metric_id
            for item in cells
            for metric_id, value in item["summary"]["metrics"].items()
            if value is not None
        }
    )
    final_metrics: dict[str, float] = {}
    final_metadata: dict[str, dict[str, str]] = {}
    curves = []
    for metric_id in metric_ids:
        series = []
        for arm in CALIBRATED_ARMS:
            arm_cells = [item for item in cells if item["arm"] == arm]
            points = [
                {"x": int(item["seed"]), "value": float(item["summary"]["metrics"][metric_id])}
                for item in arm_cells
                if item["summary"]["metrics"].get(metric_id) is not None
            ]
            if not points:
                continue
            series.append({"label": arm.replace("_", " "), "points": points})
            key = f"{arm}_{metric_id}"
            final_metrics[key] = _median([point["value"] for point in points])
            base = _metric_metadata(metric_id)
            final_metadata[key] = {
                **base,
                "label": f"{arm.replace('_', ' ')} · {base['label']}",
            }
        if series:
            base = _metric_metadata(metric_id)
            curves.append(
                {
                    "id": metric_id,
                    "title": base["label"],
                    "x_label": "frozen seed",
                    "unit": base["unit"],
                    "direction": base["direction"],
                    "series": series,
                }
            )
    charts = []
    for chart_id, title, metric_id, unit in (
        ("quality", "Held-out capped RGB-numerator error", "heldout_field_rgb_mse", "MSE"),
        ("resources", "Peak resident memory", "peak_rss_bytes", "bytes"),
        ("stage_runtime", "Refit wall time", "refit_wall_seconds", "seconds"),
    ):
        charts.append(
            {
                "id": chart_id,
                "title": title,
                "unit": unit,
                "values": [
                    {
                        "label": arm.replace("_", " "),
                        "value": _median(
                            [
                                float(item["summary"]["metrics"][metric_id])
                                for item in cells
                                if item["arm"] == arm
                            ]
                        ),
                    }
                    for arm in CALIBRATED_ARMS
                ],
            }
        )
    prefix = f"datasets/{dataset_id}"
    artifacts = [
        {"label": "Dataset machine result", "path": f"{prefix}/result.json"},
        {"label": "Orbit comparison manifest", "path": f"{prefix}/viewer_comparison.json"},
        {"label": "Native initialization", "path": f"{prefix}/native_controls_gaussians_init.ply"},
        {"label": "Native final model", "path": f"{prefix}/native_controls_gaussians.ply"},
        {
            "label": "Candidate initialization",
            "path": f"{prefix}/all_candidate_mechanisms_gaussians_init.ply",
        },
        {
            "label": "Candidate final model",
            "path": f"{prefix}/all_candidate_mechanisms_gaussians.ply",
        },
        {
            "label": "Reconstruction contact sheet",
            "path": f"{prefix}/reconstruction_contact_sheet.png",
        },
        {"label": "Reconstruction animation", "path": f"{prefix}/reconstruction.gif"},
        {"label": "Novel orbit", "path": f"{prefix}/novel_orbit.gif"},
        {"label": "Novel elevation", "path": f"{prefix}/novel_elevation.gif"},
        {"label": "Presentation receipt", "path": f"{prefix}/presentation_receipt.json"},
    ]
    viewer = [
        ".venv/bin/rtgs",
        "view",
        "--comparison-manifest",
        f"runs/{TASK_ID}/{prefix}/viewer_comparison.json",
        "--rasterizer",
        "torch",
        "--device",
        "cpu",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-open",
    ]
    alpha_fraction = final_metrics.get("native_controls_embedded_alpha_view_fraction", 0.0)
    return {
        "title": f"{dataset_id}: probabilistic Gaussian-field comparison",
        "summary": (
            "Native-control versus all-candidate development comparison over three frozen seeds. "
            f"Embedded alpha is {'available' if alpha_fraction > 0 else 'absent'}; every view is "
            "bounded by the preregistered deterministic 512-component proxy."
        ),
        "metrics": final_metrics,
        "metric_metadata": final_metadata,
        "charts": charts,
        "curves": curves,
        "artifacts": artifacts,
        "commands": {"viewer": viewer},
        "notes": [
            "Curves show every finite measured metric for every arm/seed; held-out "
            "values are post-fit reporting only.",
            "The component cap is a spatially stratified then global mass-area "
            "approximation, not a complete-field score.",
            "Independent-half distance is stability only and is censored at the match "
            "radius when no mutual match exists.",
        ],
    }


def _environment_record() -> dict[str, object]:
    import numpy
    import torch

    return {
        "schema_version": 1,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "rtgs": "workspace-source",
        },
        "device": {
            "type": "cpu",
            "name": platform.processor() or "CPU",
            "cuda": None,
        },
    }


def _build_aggregate(
    task: Mapping[str, Any],
    source_run: Path,
    publish_root: Path,
) -> None:
    """Build every aggregate artifact under an unpublished staging directory."""

    cells = _measured_cells(task, source_run)
    synthetic = _strict_json(source_run / "synthetic" / "synthetic_results.json")
    representative_seed = int(task["seeds"][0])
    dataset_summaries = {}
    dataset_results = []
    for index, dataset in enumerate(task["datasets"]):
        dataset_id = str(dataset["id"])
        dataset_cells = [item for item in cells if item["dataset"]["id"] == dataset_id]
        presentation = _save_dataset_presentation(
            task,
            source_run,
            publish_root,
            dataset,
            seed=representative_seed,
        )
        result_path = publish_root / "datasets" / dataset_id / "result.json"
        result_payload = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "dataset": dataset,
            "split": task["splits"][dataset_id],
            "synthetic_decisions": synthetic["decisions"],
            "cells": [
                {
                    "seed": item["seed"],
                    "arm": item["arm"],
                    "path": item["cell"].relative_to(source_run).as_posix(),
                    "summary": item["summary"],
                    "resource": item["resource"],
                }
                for item in dataset_cells
            ],
            "presentation": presentation,
        }
        _write_json_new(result_path, result_payload)
        dataset_results.append(
            {"dataset_id": dataset_id, "path": result_path.relative_to(publish_root).as_posix()}
        )
        dataset_summaries[dataset_id] = _dataset_summary(
            task,
            publish_root,
            dataset,
            dataset_cells,
            port=8300 + index,
        )

    first_id = str(task["datasets"][0]["id"])
    first_directory = publish_root / "datasets" / first_id
    for source, target in (
        ("all_candidate_mechanisms_gaussians_init.ply", "gaussians_init.ply"),
        ("all_candidate_mechanisms_gaussians.ply", "gaussians.ply"),
        ("reconstruction_contact_sheet.png", "reconstruction_contact_sheet.png"),
        ("reconstruction.gif", "reconstruction.gif"),
        ("novel_orbit.gif", "novel_orbit.gif"),
        ("novel_elevation.gif", "novel_elevation.gif"),
    ):
        shutil.copy2(first_directory / source, publish_root / target)

    history = _history_bundle(task, cells)
    _write_json_new(publish_root / "training_history.json", history)
    _write_json_new(
        publish_root / "gaussians.config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "frozen_configuration": task["frozen_configuration"],
            "datasets": [item["id"] for item in task["datasets"]],
            "seeds": task["seeds"],
            "arms": list(CALIBRATED_ARMS),
            "representative": {
                "dataset_id": first_id,
                "seed": representative_seed,
                "arm": "all_candidate_mechanisms",
            },
            "training": {"packed": False, "antialiased": False},
        },
    )
    cell_results = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "synthetic": "synthetic/synthetic_results.json",
        "datasets": dataset_results,
        "cells": [
            {
                "dataset_id": item["dataset"]["id"],
                "seed": item["seed"],
                "arm": item["arm"],
                "path": item["cell"].relative_to(source_run).as_posix(),
                "summary": item["summary"],
                "resource": item["resource"],
            }
            for item in cells
        ],
    }
    _write_json_new(publish_root / "cell_results.json", cell_results)
    boundary_paths = [item["cell"] / "input_boundary_receipt.json" for item in cells]
    _write_json_new(
        publish_root / "input_boundary_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "data_seal": task["data_seal"],
            "data_seal_sha256": _sha256_file(ROOT / task["data_seal"]),
            "allowed_modalities": ["calibration", "gaussians2d"],
            "external_images_or_masks_opened": False,
            "heldout_training_access": False,
            "all_worker_guards_passed": all(item["boundary"]["guard"]["passed"] for item in cells),
            "receipts": [
                {
                    "path": path.relative_to(source_run).as_posix(),
                    "sha256": _sha256_file(path),
                }
                for path in boundary_paths
            ],
        },
    )
    resource_paths = [item["cell"] / "resource_receipt.json" for item in cells]
    resource_summaries = []
    for dataset in task["datasets"]:
        for arm in CALIBRATED_ARMS:
            repeats = [
                item["resource"]
                for item in cells
                if item["dataset"]["id"] == dataset["id"] and item["arm"] == arm
            ]
            metrics = {}
            for metric_id in (
                "wall_seconds",
                "process_wall_seconds",
                "refit_wall_seconds",
                "ru_maxrss_bytes",
                "input_bytes",
                "output_bytes",
            ):
                values = [float(item[metric_id]) for item in repeats]
                metrics[metric_id] = {
                    "min": min(values),
                    "median": _median(values),
                    "max": max(values),
                }
            resource_summaries.append(
                {
                    "dataset_id": dataset["id"],
                    "arm": arm,
                    "repeat_count": len(repeats),
                    "metrics": metrics,
                }
            )
    _write_json_new(
        publish_root / "resource_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "protocol": task["resource_protocol"],
            "synthetic": {
                "path": "synthetic/resource_receipt.json",
                "sha256": _sha256_file(source_run / "synthetic" / "resource_receipt.json"),
            },
            "measured_cell_count": len(cells),
            "cuda_used": False,
            "repeat_summaries": resource_summaries,
            "receipts": [
                {
                    "path": path.relative_to(source_run).as_posix(),
                    "sha256": _sha256_file(path),
                }
                for path in resource_paths
            ],
        },
    )
    _write_json_new(publish_root / "environment.json", _environment_record())

    candidate_cells = [item for item in cells if item["arm"] == "all_candidate_mechanisms"]
    half_values = [
        float(item["summary"]["metrics"]["independent_half_center_median"])
        for item in cells
        if item["summary"]["metrics"]["independent_half_center_median"] is not None
    ]
    shape_records = [
        item
        for item in synthetic["cells"]
        if item["stage"] == "exact_shape_recovery" and item["arm"] == "rank_aware_full_covariance"
    ]
    association_records = [
        item
        for item in synthetic["cells"]
        if item["stage"] == "recomponentized_association"
        and item["arm"] == "uot_field_mass_capacity"
    ]
    mask_records = [
        item
        for item in synthetic["cells"]
        if item["stage"] == "support_mask_factorial" and item["arm"] == "probability"
    ]
    final_metrics = {
        "heldout_field_rgb_mse": _median(
            [float(item["summary"]["metrics"]["heldout_field_rgb_mse"]) for item in candidate_cells]
        ),
        "heldout_field_density_mse": _median(
            [
                float(item["summary"]["metrics"]["heldout_field_density_mse"])
                for item in candidate_cells
            ]
        ),
        "world_center_rmse": _median(
            [float(item["metrics"]["world_center_rmse"]) for item in shape_records]
        ),
        "covariance_relative_frobenius": _median(
            [float(item["metrics"]["covariance_relative_frobenius"]) for item in shape_records]
        ),
        "track_precision_times_coverage": _median(
            [
                float(item["metrics"]["track_precision_times_coverage"])
                for item in association_records
            ]
        ),
        "support_precision_times_coverage": _median(
            [float(item["metrics"]["support_precision_times_coverage"]) for item in mask_records]
        ),
        "refit_wall_seconds": _median(
            [float(item["summary"]["metrics"]["refit_wall_seconds"]) for item in candidate_cells]
        ),
        "final_gaussian_count": _median(
            [float(item["summary"]["metrics"]["final_gaussian_count"]) for item in candidate_cells]
        ),
        "independent_half_center_median": _median(half_values),
    }
    task_metric_lookup = {item["id"]: item for item in task["primary_metrics"]}
    root_metadata = {
        metric_id: {
            "label": task_metric_lookup[metric_id]["label"],
            "unit": task_metric_lookup[metric_id]["unit"],
            "group": (
                "quality"
                if metric_id not in {"refit_wall_seconds", "final_gaussian_count"}
                else "runtime/topology"
            ),
            "direction": task_metric_lookup[metric_id]["direction"],
        }
        for metric_id in final_metrics
    }
    root_charts = []
    for chart_id, title, metric_id, unit in (
        (
            "quality",
            "Held-out capped RGB-numerator error by dataset and arm",
            "heldout_field_rgb_mse",
            "MSE",
        ),
        ("resources", "Peak resident memory by dataset and arm", "peak_rss_bytes", "bytes"),
        ("stage_runtime", "Refit wall time by dataset and arm", "refit_wall_seconds", "seconds"),
    ):
        values = []
        for dataset in task["datasets"]:
            for arm in CALIBRATED_ARMS:
                subset = [
                    item
                    for item in cells
                    if item["dataset"]["id"] == dataset["id"] and item["arm"] == arm
                ]
                values.append(
                    {
                        "label": f"{dataset['id']} / {arm}",
                        "value": _median(
                            [float(item["summary"]["metrics"][metric_id]) for item in subset]
                        ),
                    }
                )
        root_charts.append({"id": chart_id, "title": title, "unit": unit, "values": values})
    evidence = [
        {"label": "Producer result", "path": f"benchmarks/results/{TASK_ID}_RESULT.md"},
        {
            "label": "Machine-readable producer result",
            "path": f"benchmarks/results/{TASK_ID}_RESULT.json",
        },
        {"label": "Independent results audit", "path": f"benchmarks/results/{TASK_ID}_AUDIT.md"},
        {
            "label": "Machine-readable independent audit",
            "path": f"benchmarks/results/{TASK_ID}_AUDIT.json",
        },
    ]
    artifacts = [
        {"label": "Representative initialization", "path": "gaussians_init.ply"},
        {"label": "Representative final model", "path": "gaussians.ply"},
        {"label": "Fitting history", "path": "training_history.json"},
        {"label": "Frozen configuration", "path": "gaussians.config.json"},
        {"label": "Input boundary aggregate", "path": "input_boundary_receipt.json"},
        {"label": "Resource aggregate", "path": "resource_receipt.json"},
        {"label": "Run receipt", "path": "run_receipt.json"},
        {"label": "Aggregate commit receipt", "path": "aggregate_commit_receipt.json"},
        {"label": "Environment", "path": "environment.json"},
        {"label": "All cell results", "path": "cell_results.json"},
        {"label": "Synthetic mechanism results", "path": "synthetic/synthetic_results.json"},
        {"label": "Contact sheet", "path": "reconstruction_contact_sheet.png"},
        {"label": "Reconstruction animation", "path": "reconstruction.gif"},
        {"label": "Novel orbit", "path": "novel_orbit.gif"},
        {"label": "Novel elevation", "path": "novel_elevation.gif"},
    ]
    metrics = {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": TASK_ID,
        "summary": (
            "Development-only native-control versus all-candidate comparison over all eleven "
            "sealed Gaussian2D fields, with isolated deterministic synthetic mechanism gates."
        ),
        "decision": "pending_independent_audit",
        "claim_boundary": task["claim_boundary"],
        "metrics": final_metrics,
        "metric_metadata": root_metadata,
        "charts": root_charts,
        "artifacts": artifacts,
        "evidence": evidence,
        "commands": {
            "reproduce": task["run_command"],
            "serve_report": [
                sys.executable,
                "-m",
                "http.server",
                "8765",
                "--directory",
                f"runs/{TASK_ID}",
            ],
            "viewer": dataset_summaries[first_id]["commands"]["viewer"],
        },
        "notes": [
            "Every calibrated view uses at most 512 deterministically selected field "
            "components; complete-field quality is not measured.",
            "Embedded packed alpha is used as hard or probability support where present; "
            "external image masks and source RGB are denied.",
            "All-candidate execution is descriptive; only separately passing synthetic "
            "mechanisms are eligible for interpretation.",
            (
                "Producer outputs remain uninterpreted until the independent results audit "
                "is attached."
            ),
        ],
        "dataset_summaries": dataset_summaries,
    }
    _write_json_new(publish_root / "metrics.json", metrics)
    lock = _strict_json(source_run / "task.lock.json")
    _write_json_new(
        publish_root / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": 0,
            "failure_phase": None,
            "message": "Synthetic controls and all 66 calibrated dataset/seed/arm cells completed.",
        },
    )
    result_payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "pending_independent_audit",
        "synthetic_decisions": synthetic["decisions"],
        "aggregate_metrics": final_metrics,
        "dataset_results": dataset_results,
        "claim_boundary": task["claim_boundary"],
    }
    evidence_staging = publish_root / "producer_evidence"
    _write_json_new(evidence_staging / f"{TASK_ID}_RESULT.json", result_payload)
    decision_lines = [
        f"- {name}: `{value['passed']}` ({value['seed_wins']} / 3 seed wins)"
        for name, value in synthetic["decisions"].items()
        if isinstance(value, dict) and "passed" in value
    ]
    _write_text_new(
        evidence_staging / f"{TASK_ID}_RESULT.md",
        "\n".join(
            [
                "# Probabilistic Gaussian-field all-dataset experiment — producer result",
                "",
                f"- Task: `{TASK_ID}`",
                "- Status: `pending_independent_audit`",
                "- Calibrated measured cells: `66`",
                "- Dataset field sets: `11`",
                "- Per-view teacher cap: `512` components",
                "",
                "## Synthetic mechanism decisions",
                "",
                *decision_lines,
                "",
                "## Boundary",
                "",
                str(task["claim_boundary"]),
                "",
                "These are producer measurements, not audited claims. The result must not be "
                "rendered or interpreted until a distinct independent results audit checks raw "
                "cells, aggregation, input guards, approximations, and viewer artifacts.",
                "",
            ]
        ),
    )


def _publish_aggregate(task: Mapping[str, Any], run: Path) -> None:
    """Stage the full aggregate, validate completeness, then publish canonical outputs."""

    staging = Path(tempfile.mkdtemp(prefix=".aggregate-staging-", dir=run))
    result_targets = {
        suffix: ROOT / "benchmarks" / "results" / f"{TASK_ID}_RESULT.{suffix}"
        for suffix in ("json", "md")
    }
    try:
        collisions = [path for path in result_targets.values() if path.exists()]
        if (run / "aggregate_commit_receipt.json").exists():
            collisions.append(run / "aggregate_commit_receipt.json")
        aggregate_names = (
            "datasets",
            "gaussians_init.ply",
            "gaussians.ply",
            "reconstruction_contact_sheet.png",
            "reconstruction.gif",
            "novel_orbit.gif",
            "novel_elevation.gif",
            "training_history.json",
            "gaussians.config.json",
            "cell_results.json",
            "input_boundary_receipt.json",
            "resource_receipt.json",
            "environment.json",
            "metrics.json",
            "run_receipt.json",
        )
        collisions.extend(run / name for name in aggregate_names if (run / name).exists())
        if collisions:
            raise FileExistsError(
                "refusing to overwrite aggregate outputs: "
                + ", ".join(str(path) for path in collisions)
            )
        _build_aggregate(task, run, staging)
        missing = [name for name in aggregate_names if not (staging / name).exists()]
        if missing:
            raise RuntimeError("aggregate staging is incomplete: " + ", ".join(missing))
        evidence_staging = staging / "producer_evidence"
        for suffix, target in result_targets.items():
            source = evidence_staging / f"{TASK_ID}_RESULT.{suffix}"
            if not source.is_file():
                raise RuntimeError(f"aggregate staging is missing {source.name}")
            os.replace(source, target)
        # Publish the completed receipt last. Before that marker exists, any exception is a
        # canonical failed run and the top-level orchestrator writes the failed receipt.
        for name in aggregate_names:
            if name == "run_receipt.json":
                continue
            os.replace(staging / name, run / name)
        _write_json_new(
            run / "aggregate_commit_receipt.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "status": "committed",
                "staged_entry_count": len(aggregate_names),
                "committed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        os.replace(staging / "run_receipt.json", run / "run_receipt.json")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--inspect-plan",
        action="store_true",
        help="print the outcome-free expanded cell plan; permitted while the task is draft",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-cell-id", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-id", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=CALIBRATED_ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", action="store_true", help=argparse.SUPPRESS)
    return parser


def _internal_command(
    task_path: Path,
    run: Path,
    *,
    synthetic: bool = False,
    synthetic_cell_id: str | None = None,
    dataset_id: str | None = None,
    seed: int | None = None,
    arm: str | None = None,
    warmup: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--task",
        task_path.relative_to(ROOT).as_posix(),
        "--run",
        run.relative_to(ROOT).as_posix(),
    ]
    if synthetic:
        if synthetic_cell_id is not None:
            raise ValueError("synthetic aggregate and cell modes are mutually exclusive")
        command.append("--synthetic-worker")
        return command
    if synthetic_cell_id is not None:
        command.extend(["--synthetic-cell-id", synthetic_cell_id])
        return command
    if dataset_id is None or seed is None or arm is None:
        raise ValueError("calibrated internal command needs dataset, seed, and arm")
    command.extend(
        [
            "--worker",
            "--dataset-id",
            dataset_id,
            "--seed",
            str(seed),
            "--arm",
            arm,
        ]
    )
    if warmup:
        command.append("--warmup")
    return command


def _orchestrate_body(task_path: Path, run: Path, task: Mapping[str, Any]) -> None:
    collisions = [
        path
        for path in (
            run / "synthetic",
            run / "cells",
            run / "warmups",
            run / "metrics.json",
            run / "run_receipt.json",
        )
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "refusing to mix experiment outputs: " + ", ".join(str(path) for path in collisions)
        )
    subprocess.run(
        [
            sys.executable,
            "scripts/experiment_contract.py",
            "validate-data",
            task_path.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        check=True,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    print("[1/69] deterministic synthetic mechanism matrix", flush=True)
    subprocess.run(
        _internal_command(task_path, run, synthetic=True),
        cwd=ROOT,
        env=environment,
        check=True,
    )
    warmup_seed = max(0, min(int(seed) for seed in task["seeds"]) - 1)
    first_dataset = str(task["datasets"][0]["id"])
    print(
        f"[2/69] discarded warmup {first_dataset} seed={warmup_seed} arm=all_candidate_mechanisms",
        flush=True,
    )
    subprocess.run(
        _internal_command(
            task_path,
            run,
            dataset_id=first_dataset,
            seed=warmup_seed,
            arm="all_candidate_mechanisms",
            warmup=True,
        ),
        cwd=ROOT,
        env=environment,
        check=True,
    )
    jobs = []
    for seed_index, seed in enumerate(task["seeds"]):
        datasets = list(task["datasets"])
        if seed_index % 2:
            datasets.reverse()
        arms = CALIBRATED_ARMS[seed_index % 2 :] + CALIBRATED_ARMS[: seed_index % 2]
        for dataset in datasets:
            jobs.extend((str(dataset["id"]), int(seed), arm) for arm in arms)
    for index, (dataset_id, seed, arm) in enumerate(jobs, start=3):
        print(f"[{index}/69] measured {dataset_id} seed={seed} arm={arm}", flush=True)
        subprocess.run(
            _internal_command(
                task_path,
                run,
                dataset_id=dataset_id,
                seed=seed,
                arm=arm,
            ),
            cwd=ROOT,
            env=environment,
            check=True,
        )
    print("[69/69] aggregate models, curves, previews, and producer records", flush=True)
    _publish_aggregate(task, run)
    print(
        "Producer execution complete. Independent results audit is required before render.",
        flush=True,
    )


def _orchestrate(task_path: Path, run: Path, task: Mapping[str, Any]) -> None:
    """Run the producer and always leave a canonical success or failure receipt."""

    try:
        _orchestrate_body(task_path, run, task)
    except Exception as error:
        measured_count = sum(1 for path in (run / "cells").glob("*/*/*") if path.is_dir())
        phase = "aggregation" if measured_count == 66 else "orchestration"
        _write_failure_new(
            run,
            phase=phase,
            error=error,
            context={"measured_cell_count": measured_count},
        )
        receipt_path = run / "run_receipt.json"
        if not receipt_path.exists():
            lock = _strict_json(run / "task.lock.json")
            _write_json_new(
                receipt_path,
                {
                    "schema_version": 1,
                    "task_id": TASK_ID,
                    "status": "failed",
                    "started_at_utc": lock["started_at_utc"],
                    "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "exit_code": 1,
                    "failure_phase": phase,
                    "message": f"{type(error).__name__}: {error}",
                },
            )
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    task_path = _resolve_repository_path(args.task, expected=TASK_RELATIVE)
    run = _resolve_repository_path(args.run, expected=RUN_RELATIVE)
    task = _strict_json(task_path)
    payload = plan_payload(task)
    if args.inspect_plan:
        print(json.dumps(payload, sort_keys=True, indent=2))
        return
    _validate_run_binding(task_path, run, task)
    if args.synthetic_cell_id is not None:
        if (
            args.synthetic_worker
            or args.worker
            or any(value is not None for value in (args.dataset_id, args.seed, args.arm))
            or args.warmup
        ):
            raise ValueError("synthetic-cell worker cannot receive other worker arguments")
        _synthetic_cell_worker(task, run, args.synthetic_cell_id)
        return
    if args.synthetic_worker:
        if (
            args.worker
            or any(value is not None for value in (args.dataset_id, args.seed, args.arm))
            or args.warmup
        ):
            raise ValueError("synthetic worker cannot receive calibrated worker arguments")
        _synthetic_worker(task, run)
        return
    if args.worker:
        if args.dataset_id is None or args.seed is None or args.arm is None:
            raise ValueError("calibrated worker needs dataset id, seed, and arm")
        _calibrated_worker(
            task=task,
            run=run,
            dataset_id=args.dataset_id,
            seed=args.seed,
            arm=args.arm,
            warmup=args.warmup,
        )
        return
    if any(value is not None for value in (args.dataset_id, args.seed, args.arm)) or args.warmup:
        raise ValueError("worker-only arguments cannot be passed to the top-level producer")
    _orchestrate(task_path, run, task)


if __name__ == "__main__":
    main()
