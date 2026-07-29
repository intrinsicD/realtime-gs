#!/usr/bin/env python3
"""Run the preregistered fixed-anchor compact-field placement experiment.

The top-level process only validates the frozen task/run binding and orchestrates fresh CPU
workers. Each worker installs a live no-image/no-legacy/no-beam boundary before importing RTGS,
loads one compact manifest, runs one placement arm, evaluates the immutable compact teachers,
and atomically publishes a self-contained cell directory.

This driver intentionally does not initialize its own run. Before invoking the frozen command,
the task needs a distinct approved prospective review and:

    .venv/bin/python scripts/experiment_contract.py init-run \
      experiments/tasks/20260729_field_sweep_placement_stage_frames00008_00009.json
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260729_field_sweep_placement_stage_frames00008_00009"
TASK_RELATIVE = Path("experiments/tasks") / f"{TASK_ID}.json"
RUN_RELATIVE = Path("runs") / TASK_ID

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
NEGATIVE_CONTROL_COUNT = 5
ARM_MODES = {
    "bounded_midpoint": "fixed_bounded_midpoint",
    "all_view_consensus": "fixed_all_view_consensus",
    "source_excluded_robust": "fixed_source_excluded_robust",
}
ARM_ORDER = tuple(ARM_MODES)
COMPACT_LOADING = {
    "load_alpha": False,
    "embedded_alpha_used_for_reconstruction": False,
}
FIELD_CONFIG = {
    "max_tracks": 128,
    "max_train_views": 8,
    "depth_samples": 32,
    "sweep_anchor_pool_multiplier": 2,
    "sweep_rounds": 3,
    "min_views": 2,
    "robust_view_fraction": 0.60,
    "sweep_refine_ratio": 0.30,
    "sweep_color_sigma": 0.20,
    "sweep_cost_tau": 0.01,
    "background_fraction": 0.0,
    "topology_rounds": 0,
    "validation_sample_cap": 512,
}
REFIT_CONFIG = {
    "iterations": 20,
    "learning_rate": 0.025,
    "appearance_start": 20,
    "sh_degree": 1,
    "density_weight": 1.0,
    "rgb_weight": 0.25,
    "null_prior_weight": 0.05,
    "observability_condition_limit": 1e5,
    "visibility_refresh": 5,
    "gain_ridge": 1e-6,
    "chunk_size": 256,
    "gradient_clip": 10.0,
    "force_source_visible": True,
}


class NoImageGuard:
    """Live denial boundary for image files and forbidden repository modules."""

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
            path = Path(os.fspath(value))
        except TypeError:
            return False
        return path.suffix.lower() in IMAGE_SUFFIXES

    @staticmethod
    def _resolved_name(name: str, globals_value: Mapping[str, Any] | None, level: int) -> str:
        if level <= 0:
            return name
        package = None if globals_value is None else globals_value.get("__package__")
        if not isinstance(package, str) or not package:
            return name
        try:
            return importlib_util.resolve_name("." * level + name, package)
        except (ImportError, ValueError):
            return name

    def _deny_path(self, value: object) -> None:
        if not self._forbidden_path(value):
            return
        if self._probing:
            self.negative_control_denials += 1
        else:
            self.denied_paths += 1
        raise PermissionError("direct-compact experiment denies every image-file open")

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
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        resolved = self._resolved_name(name, globals, level)
        candidates = (
            resolved,
            *(f"{resolved}.{item}" for item in (fromlist or ()) if item != "*"),
        )
        if any(self._forbidden_module(candidate) for candidate in candidates):
            if self._probing:
                self.negative_control_denials += 1
            else:
                self.denied_imports += 1
            raise ImportError("direct-compact experiment denies image/legacy/beam imports")
        return self._import(name, globals, locals, fromlist, level)

    def _guarded_import_module(self, name: str, package: str | None = None) -> Any:
        try:
            resolved = importlib_util.resolve_name(name, package) if name.startswith(".") else name
        except (ImportError, ValueError):
            resolved = name
        if self._forbidden_module(resolved):
            if self._probing:
                self.negative_control_denials += 1
            else:
                self.denied_imports += 1
            raise ImportError("direct-compact experiment denies image/legacy/beam imports")
        return self._import_module(name, package)

    def __enter__(self) -> NoImageGuard:
        forbidden_loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        if forbidden_loaded:
            raise RuntimeError(
                f"forbidden modules loaded before direct-compact boundary: {forbidden_loaded}"
            )
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
            with contextlib.suppress(ImportError):
                importlib.import_module("rtgs.optim.trainer")
            with contextlib.suppress(ImportError):
                importlib.import_module("rtgs.lift.beam_fusion")
        finally:
            self._probing = False
        if self.negative_control_denials != NEGATIVE_CONTROL_COUNT:
            self.__exit__()
            raise RuntimeError("direct-compact negative controls did not all fire")
        return self

    def __exit__(self, *exc: object) -> None:
        builtins.open = self._open
        io.open = self._io_open
        os.open = self._os_open
        builtins.__import__ = self._import
        importlib.import_module = self._import_module

    def record(self) -> dict[str, object]:
        forbidden_loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        passed = (
            not forbidden_loaded
            and self.denied_paths == 0
            and self.denied_imports == 0
            and self.negative_control_denials == NEGATIVE_CONTROL_COUNT
        )
        return {
            "passed": passed,
            "forbidden_modules_at_exit": forbidden_loaded,
            "image_open_attempts": self.denied_paths,
            "forbidden_import_attempts": self.denied_imports,
            "negative_control_denials": self.negative_control_denials,
            "negative_control_expected": NEGATIVE_CONTROL_COUNT,
        }


def _strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{path} repeats JSON key {key!r}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{path} contains non-finite JSON token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_new(path: Path, body: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_data_seal(
    task: Mapping[str, Any],
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Rehash the exact sealed compact bytes immediately before execution."""

    seal_path = ROOT / str(task["data_seal"])
    seal = _strict_json(seal_path)
    if (
        seal.get("schema_version") != 1
        or seal.get("data_slug") != task.get("data_slug")
        or seal.get("input_profile") != "compact"
    ):
        raise ValueError("data seal header does not match the frozen compact task")
    selected_datasets = [
        dataset for dataset in task["datasets"] if dataset_id is None or dataset["id"] == dataset_id
    ]
    if not selected_datasets:
        raise ValueError(f"data seal verification has no dataset {dataset_id!r}")
    compact_prefixes = {f"{dataset['frame_path']}/gaussians2d/" for dataset in selected_datasets}
    calibrations = {str(dataset["calibration"]) for dataset in selected_datasets}
    records = [
        record
        for record in seal.get("files", [])
        if isinstance(record, dict)
        and (
            str(record.get("path")) in calibrations
            or any(str(record.get("path")).startswith(prefix) for prefix in compact_prefixes)
        )
    ]
    sealed_paths = {str(record.get("path")) for record in records}
    expected_paths = set(calibrations)
    for dataset in selected_datasets:
        compact = ROOT / str(dataset["frame_path"]) / "gaussians2d"
        expected_paths.add((compact / "manifest.json").relative_to(ROOT).as_posix())
        expected_paths.update(
            path.relative_to(ROOT).as_posix()
            for path in compact.iterdir()
            if path.is_file() and path.suffix == ".rtgsv"
        )
    if sealed_paths != expected_paths:
        missing = sorted(expected_paths - sealed_paths)
        extra = sorted(sealed_paths - expected_paths)
        raise ValueError(f"data seal selected-file mismatch (missing={missing}, extra={extra})")

    checked: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: str(item["path"])):
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe data-seal path: {relative}")
        path = ROOT / relative
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256_file(path)
        if actual_bytes != record.get("bytes") or actual_sha256 != record.get("sha256"):
            raise ValueError(f"sealed input changed: {relative}")
        checked.append(
            {
                "path": relative.as_posix(),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
            }
        )
    return {
        "data_seal": seal_path.relative_to(ROOT).as_posix(),
        "data_seal_sha256": _sha256_file(seal_path),
        "dataset_id": dataset_id,
        "checked_file_count": len(checked),
        "checked_bytes": sum(int(record["bytes"]) for record in checked),
        "checked_records_sha256": _canonical_sha256(checked),
        "files": checked,
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"{tuple(tensor.shape)}:{tensor.dtype}\0".encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _gaussians_sha256(value: Any) -> str:
    return _canonical_sha256(
        {
            name: _tensor_sha256(getattr(value, name))
            for name in ("means", "quats", "log_scales", "opacity", "sh")
        }
    )


def _peak_rss_bytes() -> int:
    # Linux reports KiB; this repository's official experiments run in the Linux workspace.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _resolve_repository_path(value: str, *, expected: Path) -> Path:
    candidate = Path(value)
    resolved = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    target = (ROOT / expected).resolve()
    if resolved != target:
        raise ValueError(f"expected {target}, received {resolved}")
    return resolved


def _protocol_sha256(task: Mapping[str, Any]) -> str:
    protocol = {
        key: value for key, value in task.items() if key not in {"protocol_review", "status"}
    }
    return _canonical_sha256(protocol)


def _validate_run_binding(task_path: Path, run: Path) -> dict[str, Any]:
    task = _strict_json(task_path)
    if task.get("task_id") != TASK_ID:
        raise ValueError("task id does not match this frozen driver")
    expected_configuration = {
        "driver": TASK_RELATIVE.with_name(f"{TASK_ID}.py")
        .as_posix()
        .replace("experiments/tasks/", "scripts/experiments/"),
        "arms": ARM_MODES,
        "compact_loading": COMPACT_LOADING,
        "field_lift": FIELD_CONFIG,
        "refit": REFIT_CONFIG,
    }
    if task.get("frozen_configuration") != expected_configuration:
        raise ValueError("task frozen_configuration does not match the driver constants")
    if task.get("status") != "ready":
        raise ValueError("task is not ready; obtain and record a distinct prospective review")
    review = task.get("protocol_review")
    if not isinstance(review, dict) or review.get("verdict") != "approved":
        raise ValueError("task does not contain an approved prospective protocol review")
    lock_path = run / "task.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"{lock_path} is missing; run scripts/experiment_contract.py init-run first"
        )
    lock = _strict_json(lock_path)
    expected_command = task.get("run_command")
    seal_path = ROOT / str(task["data_seal"])
    review_artifact = ROOT / str(review["artifact"])
    checks = {
        "task_id": lock.get("task_id") == TASK_ID,
        "task_path": lock.get("task_path") == TASK_RELATIVE.as_posix(),
        "task_sha256": lock.get("task_sha256") == _sha256_file(task_path),
        "protocol_sha256": lock.get("protocol_sha256") == _protocol_sha256(task),
        "protocol_review": lock.get("protocol_review") == review,
        "protocol_review_artifact_sha256": (
            review_artifact.is_file()
            and lock.get("protocol_review_artifact_sha256") == _sha256_file(review_artifact)
        ),
        "data_seal_path": lock.get("data_seal_path") == task["data_seal"],
        "data_seal_sha256": (
            seal_path.is_file() and lock.get("data_seal_sha256") == _sha256_file(seal_path)
        ),
        "command": lock.get("command") == expected_command,
        "official_clean_source": (
            lock.get("development") is True
            or (lock.get("development") is False and lock.get("source_dirty") is False)
        ),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError("run/task lock binding failed: " + ", ".join(failed))
    return task


def _dataset_record(task: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [item for item in task["datasets"] if item["id"] == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"task does not define exactly one dataset {dataset_id!r}")
    return matches[0]


def _cell_relative(dataset_id: str, seed: int, arm: str, warmup: bool) -> Path:
    if warmup:
        return Path("warmups") / dataset_id / arm
    return Path("cells") / dataset_id / f"seed_{seed}" / arm


def _split_indices(
    dataset: Any,
    split: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    names = [view.view_id for view in dataset.views]
    if len(names) != len(set(names)):
        raise ValueError("compact dataset contains duplicate view identifiers")
    lookup = {name: index for index, name in enumerate(names)}
    expected = set(split["train"]) | set(split["heldout"])
    if set(names) != expected:
        raise ValueError("frozen split does not exactly partition the compact manifest")
    train = tuple(lookup[name] for name in split["train"])
    heldout = tuple(lookup[name] for name in split["heldout"])
    return train, heldout


def _worker(
    *,
    task: Mapping[str, Any],
    run: Path,
    dataset_id: str,
    seed: int,
    arm: str,
    warmup: bool,
) -> None:
    if arm not in ARM_MODES:
        raise ValueError(f"unknown arm {arm!r}")
    dataset_record = _dataset_record(task, dataset_id)
    output = run / _cell_relative(dataset_id, seed, arm, warmup)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment cell: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.worker-{os.getpid()}.",
            dir=output.parent,
        )
    )
    guard = NoImageGuard()
    process_started = time.perf_counter()
    published = False
    try:
        with guard:
            import torch

            from rtgs.data.compact_views import CompactDataset
            from rtgs.data.field_inputs import SceneFits
            from rtgs.lift.field_lifter import FieldLiftConfig, FieldLifter
            from rtgs.lift.field_refit import FieldRefitConfig

            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
            torch.manual_seed(seed)

            compact_path = ROOT / dataset_record["frame_path"] / "gaussians2d"
            seal_verification = _verify_data_seal(task, dataset_id)
            scope_started = time.perf_counter()
            dataset = CompactDataset.load(
                compact_path,
                device="cpu",
                load_alpha=COMPACT_LOADING["load_alpha"],
            )
            train, heldout = _split_indices(dataset, task["splits"][dataset_id])
            fits = SceneFits.from_compact_dataset(
                dataset,
                train_view_indices=train,
                heldout_view_indices=heldout,
                geometry_is_train_only=False,
            )
            config = FieldLiftConfig(
                placement_mode=ARM_MODES[arm],
                seed=seed,
                refit=FieldRefitConfig(**REFIT_CONFIG),
                **FIELD_CONFIG,
            )

            fit_started = time.perf_counter()
            result = FieldLifter(config).fit(fits)
            fit_seconds = time.perf_counter() - fit_started

            optimized = set(result.optimized_view_indices)
            if optimized & set(heldout) or not optimized <= set(train):
                raise RuntimeError("held-out view entered field placement/refit")
            placement_heldout = result.placement_semantic_validation.heldout
            final_heldout = result.semantic_validation.heldout
            if placement_heldout is None or final_heldout is None:
                raise RuntimeError("frozen experiment requires held-out compact validation")

            result.gaussians_init.save_ply(temporary / "gaussians_init.ply")
            result.gaussians.save_ply(temporary / "gaussians.ply")
            placement = result.diagnostics["placement"]
            if not isinstance(placement, dict):
                raise TypeError("placement diagnostics must be a dictionary")
            source_lineage_sha256 = _canonical_sha256(
                {
                    "global_views": _tensor_sha256(result.placement.source_global_view_indices),
                    "components": _tensor_sha256(result.placement.fiber.source_component_indices),
                    "xy": _tensor_sha256(result.placement.fiber.source_means2d),
                }
            )
            split_sha256 = _canonical_sha256(
                {
                    "train": task["splits"][dataset_id]["train"],
                    "heldout": task["splits"][dataset_id]["heldout"],
                }
            )
            summary = {
                "schema_version": 1,
                "task_id": TASK_ID,
                "dataset_id": dataset_id,
                "dataset_role": dataset_record["role"],
                "seed": seed,
                "arm": arm,
                "placement_mode": ARM_MODES[arm],
                "warmup": warmup,
                "anchor_sha256": placement["anchor_sha256"],
                "source_lineage_sha256": source_lineage_sha256,
                "split_sha256": split_sha256,
                "optimized_view_indices": list(result.optimized_view_indices),
                "heldout_view_indices": list(result.heldout_view_indices),
                "metrics": {
                    "placement_heldout_rgb_mse": placement_heldout.rgb_mse,
                    "placement_heldout_density_mse": placement_heldout.density_mse,
                    "final_heldout_rgb_mse": final_heldout.rgb_mse,
                    "final_heldout_density_mse": final_heldout.density_mse,
                    "supported_track_fraction": float(placement["supported_track_fraction"]),
                    "source_projection_max_error": (result.refit.source_projection_max_error),
                    "field_lift_seconds": fit_seconds,
                    "initial_gaussians": result.gaussians_init.n,
                    "final_gaussians": result.gaussians.n,
                },
                "placement_validation": asdict(result.placement_semantic_validation),
                "final_validation": asdict(result.semantic_validation),
                "diagnostics": result.diagnostics,
                "gaussians_init_semantic_sha256": _gaussians_sha256(result.gaussians_init),
                "gaussians_semantic_sha256": _gaussians_sha256(result.gaussians),
            }
            _write_json_new(
                temporary / "training_history.json",
                {
                    "objective": list(result.refit.objective_history),
                    "accepted_steps": result.refit.accepted_steps,
                },
            )
            _write_json_new(
                temporary / "gaussians.config.json",
                {
                    "field_lift": asdict(config),
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "arm": arm,
                    "warmup": warmup,
                    "training": {"packed": False, "antialiased": False},
                },
            )
            guard_record = guard.record()
            if not guard_record["passed"]:
                raise RuntimeError(f"direct-compact boundary failed: {guard_record}")
            compact_files = [compact_path / "manifest.json"]
            compact_files.extend(compact_path / f"{view.view_id}.rtgsv" for view in dataset.views)
            input_record = {
                "schema_version": 1,
                "task_id": TASK_ID,
                "dataset_id": dataset_id,
                "seed": seed,
                "arm": arm,
                "warmup": warmup,
                "allowed_modalities": ["calibration", "gaussians2d"],
                "compact_alpha_loaded": COMPACT_LOADING["load_alpha"],
                "embedded_alpha_used_for_reconstruction": (
                    COMPACT_LOADING["embedded_alpha_used_for_reconstruction"]
                ),
                "loaded_compact_files": [
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "bytes": path.stat().st_size,
                    }
                    for path in compact_files
                ],
                "input_bytes": sum(path.stat().st_size for path in compact_files),
                "split_sha256": split_sha256,
                "heldout_training_access": False,
                "seal_verification": seal_verification,
                "guard": guard_record,
            }
            _write_json_new(temporary / "input_boundary_receipt.json", input_record)

        _write_json_new(temporary / "summary.json", summary)
        output_bytes_before_receipt = sum(
            path.stat().st_size for path in temporary.rglob("*") if path.is_file()
        )
        os.replace(temporary, output)
        published = True
        scoped_wall_seconds = time.perf_counter() - scope_started
        process_wall_seconds = time.perf_counter() - process_started
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
                "cuda_used": False,
                "fit_wall_seconds": fit_seconds,
                "wall_seconds": scoped_wall_seconds,
                "process_wall_seconds": process_wall_seconds,
                "ru_maxrss_bytes": _peak_rss_bytes(),
                "output_bytes_before_receipt": output_bytes_before_receipt,
                "scope": (
                    "compact-manifest open through final compact validation and atomic "
                    "cell publication; excludes seal preflight and this self-referential receipt"
                ),
            },
        )
    except Exception:
        failure_root = output if published else temporary
        failure = failure_root / "WORKER_FAILED.txt"
        if not failure.exists():
            failure.write_text(
                "Worker failed; this directory is intentionally retained for diagnosis.\n",
                encoding="utf-8",
            )
        raise


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("geometric mean requires finite non-negative values")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _measured_summaries(task: Mapping[str, Any], run: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset in task["datasets"]:
        for seed in task["seeds"]:
            anchors: set[str] = set()
            lineages: set[str] = set()
            for arm in ARM_ORDER:
                cell = run / _cell_relative(dataset["id"], seed, arm, False)
                summary = _strict_json(cell / "summary.json")
                if summary.get("warmup") is not False:
                    raise ValueError(f"measured cell is marked as warmup: {cell}")
                boundary = _strict_json(cell / "input_boundary_receipt.json")
                if not boundary.get("guard", {}).get("passed"):
                    raise ValueError(f"input boundary did not pass: {cell}")
                resource_receipt = _strict_json(cell / "resource_receipt.json")
                summary["metrics"]["wall_seconds"] = resource_receipt["wall_seconds"]
                summary["metrics"]["peak_rss_bytes"] = resource_receipt["ru_maxrss_bytes"]
                anchors.add(str(summary["anchor_sha256"]))
                lineages.add(str(summary["source_lineage_sha256"]))
                summaries.append(summary)
            if len(anchors) != 1 or len(lineages) != 1:
                raise ValueError(
                    f"matched-arm anchor/lineage parity failed for {dataset['id']} seed {seed}"
                )
    return summaries


def _arm_values(
    summaries: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str,
) -> list[float]:
    return [float(summary["metrics"][metric]) for summary in summaries if summary["arm"] == arm]


def _decision(
    task: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    robust = _arm_values(summaries, "source_excluded_robust", "final_heldout_rgb_mse")
    midpoint = _arm_values(summaries, "bounded_midpoint", "final_heldout_rgb_mse")
    if len(robust) != len(midpoint):
        raise ValueError("paired robust/midpoint cell count changed")
    ratio = _geometric_mean(
        [robust_value / midpoint_value for robust_value, midpoint_value in zip(robust, midpoint)]
    )
    wins_by_dataset: dict[str, int] = {}
    for dataset in task["datasets"]:
        dataset_id = dataset["id"]
        paired = [summary for summary in summaries if summary["dataset_id"] == dataset_id]
        robust_by_seed = {
            int(item["seed"]): float(item["metrics"]["final_heldout_rgb_mse"])
            for item in paired
            if item["arm"] == "source_excluded_robust"
        }
        midpoint_by_seed = {
            int(item["seed"]): float(item["metrics"]["final_heldout_rgb_mse"])
            for item in paired
            if item["arm"] == "bounded_midpoint"
        }
        wins_by_dataset[dataset_id] = sum(
            robust_by_seed[seed] < midpoint_by_seed[seed] for seed in task["seeds"]
        )
    support = min(_arm_values(summaries, "source_excluded_robust", "supported_track_fraction"))
    projection = max(
        float(summary["metrics"]["source_projection_max_error"]) for summary in summaries
    )
    passed = (
        ratio <= 0.95
        and all(wins >= 2 for wins in wins_by_dataset.values())
        and support >= 0.95
        and projection <= 0.0002
    )
    return {
        "producer_rule_passed": passed,
        "robust_to_midpoint_final_rgb_geometric_mean_ratio": ratio,
        "robust_seed_wins_by_dataset": wins_by_dataset,
        "minimum_robust_supported_track_fraction": support,
        "maximum_source_projection_error": projection,
        "audit_status": "pending_independent_audit",
    }


def _publish_aggregate(
    task: Mapping[str, Any],
    run: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    decision = _decision(task, summaries)
    representative = (
        run
        / "cells"
        / task["datasets"][0]["id"]
        / f"seed_{task['seeds'][0]}"
        / "source_excluded_robust"
    )
    for name in ("gaussians_init.ply", "gaussians.ply"):
        target = run / name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite representative model: {target}")
        shutil.copyfile(representative / name, target)

    cell_results = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "cells": list(summaries),
        "producer_decision": decision,
    }
    _write_json_new(run / "cell_results.json", cell_results)
    _write_json_new(
        run / "training_history.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "cells": [
                {
                    "dataset_id": item["dataset_id"],
                    "seed": item["seed"],
                    "arm": item["arm"],
                    "history": _strict_json(
                        run
                        / _cell_relative(
                            str(item["dataset_id"]),
                            int(item["seed"]),
                            str(item["arm"]),
                            False,
                        )
                        / "training_history.json"
                    ),
                }
                for item in summaries
            ],
        },
    )
    _write_json_new(
        run / "gaussians.config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "field_config": FIELD_CONFIG,
            "refit_config": REFIT_CONFIG,
            "arms": ARM_MODES,
            "representative": {
                "dataset_id": task["datasets"][0]["id"],
                "seed": task["seeds"][0],
                "arm": "source_excluded_robust",
            },
            "training": {"packed": False, "antialiased": False},
        },
    )

    boundary_receipts = []
    resource_receipts = []
    for summary in summaries:
        cell = run / _cell_relative(
            str(summary["dataset_id"]),
            int(summary["seed"]),
            str(summary["arm"]),
            False,
        )
        boundary_receipts.append(_strict_json(cell / "input_boundary_receipt.json"))
        resource_receipts.append(_strict_json(cell / "resource_receipt.json"))
    if not all(receipt["guard"]["passed"] for receipt in boundary_receipts):
        raise RuntimeError("one or more measured input boundaries failed")
    _write_json_new(
        run / "input_boundary_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "data_seal": task["data_seal"],
            "data_seal_sha256": _sha256_file(ROOT / task["data_seal"]),
            "allowed_modalities": ["calibration", "gaussians2d"],
            "measured_cell_count": len(boundary_receipts),
            "all_guards_passed": True,
            "heldout_training_access": False,
            "receipts": boundary_receipts,
        },
    )
    _write_json_new(
        run / "resource_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "protocol": task["resource_protocol"],
            "measured_cell_count": len(resource_receipts),
            "cuda_used": False,
            "receipts": resource_receipts,
        },
    )

    arm_quality = {
        arm: _geometric_mean(_arm_values(summaries, arm, "final_heldout_rgb_mse"))
        for arm in ARM_ORDER
    }
    arm_seconds = {
        arm: statistics.median(_arm_values(summaries, arm, "wall_seconds")) for arm in ARM_ORDER
    }
    arm_rss_mib = {
        arm: statistics.median(_arm_values(summaries, arm, "peak_rss_bytes")) / (1024**2)
        for arm in ARM_ORDER
    }
    robust_placement = _geometric_mean(
        _arm_values(summaries, "source_excluded_robust", "placement_heldout_rgb_mse")
    )
    robust_final_rgb = arm_quality["source_excluded_robust"]
    robust_final_density = _geometric_mean(
        _arm_values(summaries, "source_excluded_robust", "final_heldout_density_mse")
    )
    supported = float(decision["minimum_robust_supported_track_fraction"])
    projection = float(decision["maximum_source_projection_error"])
    wall_seconds = statistics.median(
        [float(summary["metrics"]["wall_seconds"]) for summary in summaries]
    )
    audit_qualified = (
        "producer rule supports the frozen development hypothesis"
        if decision["producer_rule_passed"]
        else "producer rule does not support the frozen development hypothesis"
    )
    metrics = {
        "schema_version": 1,
        "report_template_version": 1,
        "task_id": TASK_ID,
        "summary": (f"{audit_qualified}; all values remain pending an independent results audit."),
        "decision": f"pending_independent_audit: {audit_qualified}",
        "claim_boundary": task["claim_boundary"],
        "metrics": {
            "placement_heldout_rgb_mse": robust_placement,
            "final_heldout_rgb_mse": robust_final_rgb,
            "final_heldout_density_mse": robust_final_density,
            "supported_track_fraction": supported,
            "source_projection_max_error": projection,
            "wall_seconds": wall_seconds,
        },
        "metric_metadata": {
            "placement_heldout_rgb_mse": {
                "label": "Robust placement held-out compact RGB MSE",
                "unit": "mse",
                "group": "Quality",
                "direction": "lower",
            },
            "final_heldout_rgb_mse": {
                "label": "Robust final held-out compact RGB MSE",
                "unit": "mse",
                "group": "Quality",
                "direction": "lower",
            },
            "final_heldout_density_mse": {
                "label": "Robust final held-out compact density MSE",
                "unit": "mse",
                "group": "Quality",
                "direction": "lower",
            },
            "supported_track_fraction": {
                "label": "Minimum robust supported-track fraction",
                "unit": "fraction",
                "group": "Guardrails",
                "direction": "higher",
            },
            "source_projection_max_error": {
                "label": "Maximum source-projection error",
                "unit": "pixels",
                "group": "Guardrails",
                "direction": "lower",
            },
            "wall_seconds": {
                "label": "Median measured cell CPU wall time",
                "unit": "seconds",
                "group": "Resources",
                "direction": "descriptive",
            },
        },
        "charts": [
            {
                "id": "quality",
                "title": "Final held-out compact RGB MSE by placement arm",
                "unit": "geometric mean MSE",
                "values": [
                    {"label": arm.replace("_", " "), "value": arm_quality[arm]} for arm in ARM_ORDER
                ],
            },
            {
                "id": "resources",
                "title": "Peak resident memory by placement arm",
                "unit": "median MiB",
                "values": [
                    {"label": arm.replace("_", " "), "value": arm_rss_mib[arm]} for arm in ARM_ORDER
                ],
            },
            {
                "id": "stage_runtime",
                "title": "End-to-end field lift time by placement arm",
                "unit": "median CPU seconds",
                "values": [
                    {"label": arm.replace("_", " "), "value": arm_seconds[arm]} for arm in ARM_ORDER
                ],
            },
        ],
        "artifacts": [
            {"label": "Representative placement model", "path": "gaussians_init.ply"},
            {"label": "Representative final model", "path": "gaussians.ply"},
            {"label": "All objective histories", "path": "training_history.json"},
            {"label": "Frozen model configuration", "path": "gaussians.config.json"},
            {"label": "Input-boundary receipt", "path": "input_boundary_receipt.json"},
            {"label": "Resource receipt", "path": "resource_receipt.json"},
            {"label": "All measured cell results", "path": "cell_results.json"},
        ],
        "evidence": [
            {
                "label": "Producer result note",
                "path": f"benchmarks/results/{TASK_ID}_RESULT.md",
            },
            {
                "label": "Producer result data",
                "path": f"benchmarks/results/{TASK_ID}_RESULT.json",
            },
            {
                "label": "Independent audit note",
                "path": f"benchmarks/results/{TASK_ID}_AUDIT.md",
            },
            {
                "label": "Independent audit data",
                "path": f"benchmarks/results/{TASK_ID}_AUDIT.json",
            },
        ],
        "viewer_command": [
            ".venv/bin/rtgs",
            "view",
            "--gaussians",
            f"runs/{TASK_ID}/gaussians.ply",
            "--initial",
            f"runs/{TASK_ID}/gaussians_init.ply",
            "--no-open",
        ],
        "notes": [
            "No RGB or mask file was opened; compact teachers and embedded cameras were used.",
            "Topology was disabled and all arms shared source anchors within every "
            "scene/seed pair.",
            "Runtime and memory are descriptive CPU reference measurements.",
            "Do not render the canonical report or update claims before independent results audit.",
        ],
    }
    _write_json_new(run / "metrics.json", metrics)

    result_payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "pending_independent_audit",
        "producer_decision": decision,
        "aggregate_metrics": metrics["metrics"],
        "arm_quality": arm_quality,
        "arm_seconds": arm_seconds,
        "arm_peak_rss_mib": arm_rss_mib,
        "cell_results": f"runs/{TASK_ID}/cell_results.json",
        "claim_boundary": task["claim_boundary"],
    }
    result_json = ROOT / "benchmarks/results" / f"{TASK_ID}_RESULT.json"
    result_md = ROOT / "benchmarks/results" / f"{TASK_ID}_RESULT.md"
    _write_json_new(result_json, result_payload)
    win_rows = "\n".join(
        f"- {dataset_id}: {wins} / {len(task['seeds'])} robust wins"
        for dataset_id, wins in decision["robust_seed_wins_by_dataset"].items()
    )
    _write_text_new(
        result_md,
        "\n".join(
            [
                "# Robust compact-field plane-sweep placement — producer result",
                "",
                f"- Task ID: `{TASK_ID}`",
                "- Status: `pending_independent_audit`",
                f"- Producer rule passed: `{decision['producer_rule_passed']}`",
                (
                    "- Robust / midpoint final RGB geometric-mean ratio: "
                    f"`{decision['robust_to_midpoint_final_rgb_geometric_mean_ratio']:.8g}`"
                ),
                (
                    "- Minimum robust supported-track fraction: "
                    f"`{decision['minimum_robust_supported_track_fraction']:.8g}`"
                ),
                (
                    "- Maximum source-projection error: "
                    f"`{decision['maximum_source_projection_error']:.8g}`"
                ),
                "",
                "## Paired seed rule",
                "",
                win_rows,
                "",
                "## Boundary",
                "",
                task["claim_boundary"],
                "",
                "These are producer outputs, not audited claims. A distinct results audit must "
                "verify the cells, guards, aggregation, and viewer handoff before report "
                "rendering.",
                "",
            ]
        ),
    )


def _worker_command(
    *,
    task_path: Path,
    run: Path,
    dataset_id: str,
    seed: int,
    arm: str,
    warmup: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--task",
        task_path.relative_to(ROOT).as_posix(),
        "--run-dir",
        run.relative_to(ROOT).as_posix(),
        "--worker",
        "--dataset-id",
        dataset_id,
        "--seed",
        str(seed),
        "--arm",
        arm,
    ]
    if warmup:
        command.append("--warmup")
    return command


def _orchestrate(task_path: Path, run: Path, task: Mapping[str, Any]) -> None:
    seal_receipt = _verify_data_seal(task)
    print(
        "Verified frozen compact data seal: "
        f"{seal_receipt['checked_file_count']} files, "
        f"{seal_receipt['checked_bytes']} bytes.",
        flush=True,
    )
    existing_roots = [
        run / name
        for name in (
            "cells",
            "warmups",
            "metrics.json",
            "cell_results.json",
            "gaussians_init.ply",
            "gaussians.ply",
        )
        if (run / name).exists()
    ]
    if existing_roots:
        raise FileExistsError(
            "refusing to mix with existing experiment outputs: "
            + ", ".join(str(path) for path in existing_roots)
        )
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    warmup_seed = max(0, min(int(seed) for seed in task["seeds"]) - 1)
    jobs: list[tuple[str, int, str, bool]] = []
    for dataset in task["datasets"]:
        for arm in ARM_ORDER:
            jobs.append((dataset["id"], warmup_seed, arm, True))
    for dataset in task["datasets"]:
        for seed in task["seeds"]:
            # Rotate arm order by seed to counterbalance any OS cache/order effects.
            rotation = task["seeds"].index(seed) % len(ARM_ORDER)
            arm_order = ARM_ORDER[rotation:] + ARM_ORDER[:rotation]
            for arm in arm_order:
                jobs.append((dataset["id"], int(seed), arm, False))

    for index, (dataset_id, seed, arm, warmup) in enumerate(jobs, start=1):
        kind = "warmup" if warmup else "measured"
        print(
            f"[{index}/{len(jobs)}] {kind} {dataset_id} seed={seed} arm={arm}",
            flush=True,
        )
        subprocess.run(
            _worker_command(
                task_path=task_path,
                run=run,
                dataset_id=dataset_id,
                seed=seed,
                arm=arm,
                warmup=warmup,
            ),
            cwd=ROOT,
            env=environment,
            check=True,
        )

    summaries = _measured_summaries(task, run)
    _publish_aggregate(task, run, summaries)
    print(
        "Producer outputs complete. Obtain an independent results audit before rendering "
        "or interpreting the experiment report.",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-id", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARM_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    task_path = _resolve_repository_path(args.task, expected=TASK_RELATIVE)
    run = _resolve_repository_path(args.run_dir, expected=RUN_RELATIVE)
    task = _validate_run_binding(task_path, run)
    if args.worker:
        if args.dataset_id is None or args.seed is None or args.arm is None:
            raise ValueError("internal worker requires dataset id, seed, and arm")
        _worker(
            task=task,
            run=run,
            dataset_id=args.dataset_id,
            seed=args.seed,
            arm=args.arm,
            warmup=args.warmup,
        )
        return
    if any(value is not None for value in (args.dataset_id, args.seed, args.arm)) or args.warmup:
        raise ValueError("worker-only arguments cannot be used by the top-level command")
    _orchestrate(task_path, run, task)


if __name__ == "__main__":
    main()
