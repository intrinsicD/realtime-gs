#!/usr/bin/env python3
"""RTGS-008 provider-neutral full-resolution paper comparison driver.

The development subcommands construct and train one provider/seed cell without reading source
images.  The protected top-level command added below validates the task lock and exact data seal,
runs the complete GaussianImage / StructSplat provider matrix in fresh workers, and publishes the
Bundle Contract v2 producer sources for independent audit:

* ``preflight`` computes train-only feasible initializer counts for all providers;
* ``initialize`` and ``train`` provide bounded mechanism/debug commands;
* ``run`` executes the frozen protected matrix.

Canonical invocation from the repository root::

    .venv/bin/python \
      scripts/experiments/20260801_paper_three_provider_fullres_stage_frame00008.py run \
      --task experiments/tasks/20260801_paper_three_provider_fullres_stage_frame00008.json \
      --run-dir runs/20260801_paper_three_provider_fullres_stage_frame00008

No reconstruction worker reads source RGB or masks. Every worker denies image suffixes,
image-capable loaders, ``SceneData``, and the dense trainer. Presentation is isolated after model
publication and never enters a reconstruction or resource receipt.
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
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260801_paper_three_provider_fullres_stage_frame00008"
DEFAULT_TASK = ROOT / "experiments/tasks" / f"{TASK_ID}.json"
DEFAULT_OUTPUT = ROOT / ".scratch" / TASK_ID / "development"
DEFAULT_RUN = ROOT / "runs" / TASK_ID
ARMS = ("bounded_random", "splat_sfm", "beam_fusion")
ARM_LABELS = {
    "bounded_random": "Bounded Random",
    "splat_sfm": "Splat-SfM",
    "beam_fusion": "Beam Fusion",
}
IMAGE_SUFFIXES = frozenset({".bmp", ".exr", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
IMAGE_FORBIDDEN_MODULES = (
    "PIL",
    "cv2",
    "imageio",
    "rtgs.data.calibrated",
)
DENSE_FORBIDDEN_MODULES = (
    "rtgs.data.scene",
    "rtgs.optim.trainer",
)

TRAIN_CONFIG = {
    "iterations": 10_000,
    "attempts_per_step": 128,
    "proposal_mode": "area_gaussian",
    "schedule_mode": "balanced_cycle",
    "target_mode": "uniform",
    "uniform_fraction": 0.25,
    "device": "cuda:0",
    "lr_means": 1.6e-4,
    "lr_quats": 1.0e-3,
    "lr_scales": 5.0e-3,
    "lr_opacity": 5.0e-2,
    "lr_sh": 2.5e-3,
    "lr_sh_rest": 1.25e-4,
    "point_chunk": 64,
    "gaussian_chunk": 1_024,
    "outer_microbatch": 128,
    "query_component_chunk": 512,
    "teacher_tile_size": 16,
    "evaluation_chunk": 1_024,
    "checkpoints": (
        0,
        100,
        250,
        500,
        1_000,
        2_000,
        3_500,
        5_000,
        6_500,
        8_000,
        10_000,
    ),
    "evaluate_checkpoint_risks": False,
    "sh_degree": 3,
}
DENSITY_CONFIG = {
    "start_iter": 60,
    "stop_iter": 8_000,
    "every": 40,
    "grad_threshold": 2e-4,
    "absgrad": False,
    "split_scale_frac": 0.01,
    "split_factor": 1.6,
    "prune_opacity": 0.005,
    "prune_scale_frac": 0.1,
    "max_gaussians": 100_000,
    "opacity_reset_every": 3_000,
    "opacity_reset_value": 0.011,
    "revised_opacity": True,
}
QUERY_INDEX_CONFIG = {
    "tile_size": 16,
    "max_entries": 16_000_000,
    "max_candidates": 200_000,
    "max_query_pairs": 1_048_576,
}
POINT_RENDER_CONFIG = {
    "backend": "gsplat_microcamera",
    "packed": True,
    "antialiased": False,
    "kernel_support_mode": "hard",
}
CONVERGENCE_CONFIG = {
    "samples_per_view": 1_024,
    "stable_best_multiplier": 1.05,
    "maximum_absolute_tail_relative_change": 0.01,
}
RESOURCE_CONFIG = {
    "device_index": 0,
    "idle_samples": 3,
    "idle_sample_interval_seconds": 0.5,
    "idle_timeout_seconds": 300.0,
    "max_foreign_compute_processes": 0,
    "max_background_device_memory_bytes": 3 * 1024**3,
    "max_background_range_bytes": 128 * 1024**2,
    "max_gpu_utilization_percent": 40,
    "monitor_interval_seconds": 0.25,
}
PRESENTATION_CONFIG = {
    "representative_seed": 801001,
    "calibrated_view_id": "C0014",
    "grid_columns": 3,
    "panel_max_side": 480,
    "animation_frames": 8,
    "animation_downscale": 8,
    "animation_duration_ms": 140,
}


class NvmlProcessSampler:
    """Sample this worker's NVML allocation across the frozen resource scope."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        device_index = int(config["device_index"])
        interval_seconds = float(config["monitor_interval_seconds"])
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        if interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive")
        self.config = dict(config)
        self.device_index = int(device_index)
        self.interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any = None
        self._handle: Any = None
        self._error: BaseException | None = None
        self._samples = 0
        self._process_peak_bytes = 0
        self._background_peak_bytes = 0
        self._background_start_bytes = 0
        self._device_total_bytes = 0
        self._driver_version = ""
        self._idle_guard: dict[str, Any] | None = None
        self._gpu_utilization_peak_percent = 0
        self._foreign_compute_processes: dict[int, int] = {}

    def _snapshot(self) -> dict[str, Any]:
        memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        own_bytes = 0
        foreign = []
        unavailable = getattr(self._pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
        for process in self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
            raw_used = getattr(process, "usedGpuMemory", 0)
            used = 0 if raw_used in {None, unavailable} or int(raw_used) < 0 else int(raw_used)
            if int(process.pid) == os.getpid():
                own_bytes = max(own_bytes, used)
            else:
                foreign.append({"pid": int(process.pid), "used_bytes": used})
        background_bytes = max(0, int(memory.used) - own_bytes)
        return {
            "device_used_bytes": int(memory.used),
            "device_total_bytes": int(memory.total),
            "process_used_bytes": own_bytes,
            "background_device_memory_bytes": background_bytes,
            "gpu_utilization_percent": int(utilization.gpu),
            "memory_utilization_percent": int(utilization.memory),
            "foreign_compute_processes": foreign,
        }

    def _sample_once(self) -> None:
        sample = self._snapshot()
        own_bytes = sample["process_used_bytes"]
        background_bytes = sample["background_device_memory_bytes"]
        self._process_peak_bytes = max(self._process_peak_bytes, own_bytes)
        self._background_peak_bytes = max(self._background_peak_bytes, background_bytes)
        self._gpu_utilization_peak_percent = max(
            self._gpu_utilization_peak_percent,
            sample["gpu_utilization_percent"],
        )
        for process in sample["foreign_compute_processes"]:
            self._foreign_compute_processes[process["pid"]] = max(
                self._foreign_compute_processes.get(process["pid"], 0),
                process["used_bytes"],
            )
        self._samples += 1

    def _wait_for_idle(self) -> dict[str, Any]:
        deadline = time.monotonic() + float(self.config["idle_timeout_seconds"])
        required = int(self.config["idle_samples"])
        accepted: list[dict[str, Any]] = []
        observed = 0
        while time.monotonic() <= deadline:
            sample = self._snapshot()
            observed += 1
            eligible = (
                len(sample["foreign_compute_processes"])
                <= int(self.config["max_foreign_compute_processes"])
                and sample["background_device_memory_bytes"]
                <= int(self.config["max_background_device_memory_bytes"])
                and sample["gpu_utilization_percent"]
                <= int(self.config["max_gpu_utilization_percent"])
            )
            if eligible:
                accepted.append(sample)
                if len(accepted) > required:
                    accepted.pop(0)
                values = [item["background_device_memory_bytes"] for item in accepted]
                stable = len(accepted) == required and max(values) - min(values) <= int(
                    self.config["max_background_range_bytes"]
                )
                if stable:
                    return {
                        "passed": True,
                        "observed_samples": observed,
                        "accepted_samples": accepted,
                        "limits": dict(self.config),
                    }
            else:
                accepted.clear()
            time.sleep(float(self.config["idle_sample_interval_seconds"]))
        raise RuntimeError("GPU quiescence guard timed out before the measured resource scope")

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self._sample_once()
            except BaseException as error:  # pragma: no cover - hardware/runtime dependent
                self._error = error
                self._stop_event.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("NVML sampler already started")
        try:
            import pynvml
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("the frozen resource protocol requires pynvml") from error
        pynvml.nvmlInit()
        try:
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._device_total_bytes = int(memory.total)
            driver = pynvml.nvmlSystemGetDriverVersion()
            self._driver_version = driver.decode() if isinstance(driver, bytes) else str(driver)
            self._idle_guard = self._wait_for_idle()
            self._sample_once()
            self._background_start_bytes = self._background_peak_bytes
            self._thread = threading.Thread(
                target=self._sample_loop,
                name="rtgs-nvml-sampler",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            pynvml.nvmlShutdown()
            raise

    def stop(self) -> dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("NVML sampler was not started")
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        if self._thread.is_alive():
            raise RuntimeError("NVML sampler did not stop")
        try:
            if self._error is not None:
                raise RuntimeError("NVML sampling failed") from self._error
            self._sample_once()
            if self._foreign_compute_processes:
                raise RuntimeError(
                    "foreign CUDA compute process entered the frozen resource measurement scope"
                )
            if self._idle_guard is None:
                raise RuntimeError("GPU quiescence guard was not recorded")
            return {
                "nvml_process_peak_bytes": self._process_peak_bytes,
                "background_device_memory_bytes": self._background_start_bytes,
                "background_device_memory_peak_bytes": self._background_peak_bytes,
                "device_total_bytes": self._device_total_bytes,
                "driver_version": self._driver_version,
                "nvml_sampling_interval_seconds": self.interval_seconds,
                "nvml_samples": self._samples,
                "gpu_utilization_peak_percent": self._gpu_utilization_peak_percent,
                "foreign_compute_processes": [],
                "idle_guard": self._idle_guard,
            }
        finally:
            self._pynvml.nvmlShutdown()
            self._thread = None


class NoImageGuard:
    """Live source-image and dense-training boundary."""

    def __init__(self) -> None:
        self.forbidden_modules = (*IMAGE_FORBIDDEN_MODULES, *DENSE_FORBIDDEN_MODULES)
        self.negative_control_expected = 4
        self.denied_paths = 0
        self.denied_imports = 0
        self.negative_control_denials = 0
        self._probing = False
        self._open = builtins.open
        self._io_open = io.open
        self._os_open = os.open
        self._import = builtins.__import__
        self._import_module = importlib.import_module

    def _forbidden_module(self, name: str) -> bool:
        return any(name == root or name.startswith(f"{root}.") for root in self.forbidden_modules)

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
    def _resolved_name(
        name: str,
        globals_value: Mapping[str, Any] | None,
        level: int,
    ) -> str:
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
        raise PermissionError("compact reconstruction denies every image-file open")

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
            raise ImportError("compact reconstruction denies image/dense-pipeline imports")
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
            raise ImportError("compact reconstruction denies image/dense-pipeline imports")
        return self._import_module(name, package)

    def __enter__(self) -> NoImageGuard:
        loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        if loaded:
            raise RuntimeError(f"forbidden modules loaded before compact boundary: {loaded}")
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
            if self.negative_control_expected == 4:
                with contextlib.suppress(ImportError):
                    importlib.import_module("rtgs.data.calibrated")
                with contextlib.suppress(ImportError):
                    importlib.import_module("rtgs.optim.trainer")
        finally:
            self._probing = False
        if self.negative_control_denials != self.negative_control_expected:
            self.__exit__()
            raise RuntimeError("compact reconstruction negative controls did not all fire")
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
            "passed": (
                not loaded
                and self.denied_paths == 0
                and self.denied_imports == 0
                and self.negative_control_denials == self.negative_control_expected
            ),
            "forbidden_modules_at_exit": loaded,
            "image_open_attempts": self.denied_paths,
            "forbidden_import_attempts": self.denied_imports,
            "negative_control_denials": self.negative_control_denials,
            "negative_control_expected": self.negative_control_expected,
        }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False).encode("utf-8"))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_and_dataset(task_path: Path, dataset_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    task = _load_json(task_path.resolve())
    if task.get("task_id") != TASK_ID or task.get("status") not in {"draft", "ready"}:
        raise ValueError("driver requires the matching draft or ready task")
    matches = [dataset for dataset in task["datasets"] if dataset["id"] == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"task does not contain exactly one dataset {dataset_id!r}")
    return task, matches[0]


def _byte_cap(task: dict[str, Any]) -> int:
    return int(task["frozen_configuration"]["provider_factor"]["byte_cap_per_view"])


def _initializer_settings(task: dict[str, Any]) -> dict[str, Any]:
    frozen = task["frozen_configuration"]["initializer"]
    common_count = frozen["common_starting_count"]
    maximum = frozen["max_starting_gaussians"] if common_count is None else common_count
    return {
        "random_bounds_scale": frozen["random_bounds_scale"],
        "init_opacity": frozen["init_opacity"],
        "max_starting_gaussians": maximum,
        "structural_components_per_view": frozen["structural_components_per_view"],
        "splat_sfm": dict(frozen["splat_sfm"]),
        "beam_fusion": dict(frozen["beam_fusion"]),
    }


def _evidence_status(task: Mapping[str, Any]) -> str:
    return (
        "canonical_producer_awaiting_audit"
        if task["status"] == "ready"
        else "mechanism_only_task_still_draft"
    )


def _subset_inputs(inputs: Any, names: list[str]) -> Any:
    from rtgs.data.reconstruction_inputs import ReconstructionInputs

    lookup = {name: index for index, name in enumerate(inputs.view_names)}
    if len(lookup) != len(inputs.view_names) or any(name not in lookup for name in names):
        raise ValueError("frozen split differs from the compact manifest")
    indices = [lookup[name] for name in names]
    return ReconstructionInputs(
        observations=[inputs.observations[index] for index in indices],
        cameras=[inputs.cameras[index] for index in indices],
        view_names=list(names),
        points=None,
        point_visibility=None,
        bounds_hint=inputs.bounds_hint,
        name=f"{inputs.name}-{'-'.join(names[:1])}-{len(names)}views",
        archive_stats=None,
    )


def _load_split_inputs(
    task: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[Any, Any, Any]:
    from rtgs.data.compact_views import CompactDataset

    directory = (ROOT / dataset["compact_manifest"]).parent
    compact = CompactDataset.load(
        directory,
        device="cpu",
        byte_cap=_byte_cap(task),
        load_alpha=False,
    )
    all_inputs = compact.to_reconstruction_inputs()
    split = task["splits"][dataset["id"]]
    train = _subset_inputs(all_inputs, split["train"])
    heldout = _subset_inputs(all_inputs, split["heldout"])
    if set(train.view_names) & set(heldout.view_names):
        raise RuntimeError("train and held-out compact inputs overlap")
    return compact, train, heldout


def _initialize(args: argparse.Namespace) -> int:
    task, dataset = _task_and_dataset(args.task, args.dataset_id)
    output = args.output.resolve() / args.dataset_id / f"seed_{args.seed}" / "initializations"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite initialization output: {output}")
    output.mkdir(parents=True)
    guard = NoImageGuard()
    with guard:
        import numpy as np
        import torch

        from rtgs.lift.beam_fusion import BeamFusionConfig
        from rtgs.lift.compact_carve import _center_and_extent
        from rtgs.lift.paper_initializers import (
            PaperInitializerConfig,
            build_matched_paper_initializations,
        )
        from rtgs.lift.splat_sfm import SplatSfMConfig

        _compact, train, _heldout = _load_split_inputs(task, dataset)
        common_center, common_extent = _center_and_extent(train, torch.float64)
        initializer_settings = _initializer_settings(task)
        config = PaperInitializerConfig(
            random_seed=args.seed,
            random_bounds_scale=initializer_settings["random_bounds_scale"],
            init_opacity=initializer_settings["init_opacity"],
            max_starting_gaussians=initializer_settings["max_starting_gaussians"],
            structural_components_per_view=initializer_settings["structural_components_per_view"],
            sfm=SplatSfMConfig(
                init_opacity=initializer_settings["init_opacity"],
                **initializer_settings["splat_sfm"],
            ),
            beam=BeamFusionConfig(
                init_opacity=initializer_settings["init_opacity"],
                **initializer_settings["beam_fusion"],
            ),
        )
        started = time.perf_counter()
        result = build_matched_paper_initializations(train, config)
        elapsed = time.perf_counter() - started
        frozen_count = task["frozen_configuration"]["initializer"]["common_starting_count"]
        if frozen_count is not None and result.count != frozen_count:
            raise RuntimeError(
                f"initializer returned {result.count} rows, expected frozen global count "
                f"{frozen_count}"
            )
        models = {
            "bounded_random": result.bounded_random,
            "splat_sfm": result.splat_sfm,
            "beam_fusion": result.beam_fusion,
        }
        for arm, model in models.items():
            arm_directory = output.parent / arm
            arm_directory.mkdir(parents=True)
            model.save_npz(arm_directory / "gaussians_init.npz")
            model.save_ply(arm_directory / "gaussians_init.ply")
        np.savez_compressed(
            output / "initializer_lineage.npz",
            splat_sfm_selected_rows=result.splat_sfm_selected_rows.numpy(),
            beam_fusion_selected_rows=result.beam_fusion_selected_rows.numpy(),
            splat_sfm_track_offsets=result.splat_sfm_result.track_offsets.numpy(),
            splat_sfm_member_view_indices=(result.splat_sfm_result.member_view_indices.numpy()),
            splat_sfm_member_component_indices=(
                result.splat_sfm_result.member_component_indices.numpy()
            ),
            beam_component_offsets=result.beam_fusion_result.component_offsets.numpy(),
            beam_contributor_view_indices=(
                result.beam_fusion_result.contributor_view_indices.numpy()
            ),
            beam_contributor_component_indices=(
                result.beam_fusion_result.contributor_component_indices.numpy()
            ),
            beam_component_weights=result.beam_fusion_result.component_weights.numpy(),
        )
        guard_record = guard.record()
        if not guard_record["passed"]:
            raise RuntimeError(f"initialization input boundary failed: {guard_record}")
        _write_json_atomic(
            output / "receipt.json",
            {
                "schema": "rtgs.paper_three_path_initialization.v1",
                "evidence_status": _evidence_status(task),
                "task_id": TASK_ID,
                "dataset_id": args.dataset_id,
                "seed": args.seed,
                "train_views": train.view_names,
                "heldout_views_excluded": task["splits"][args.dataset_id]["heldout"],
                "config": initializer_settings,
                "elapsed_seconds": elapsed,
                "common_count": result.count,
                "common_training_geometry": {
                    "policy": "camera_axis_fallback_from_frozen_train_cameras",
                    "center": common_center.tolist(),
                    "extent": common_extent,
                },
                "initializer_receipt": result.receipt,
                "input_boundary": guard_record,
                "torch_seed": torch.initial_seed(),
            },
        )
    print(f"three exact-count initializations ({result.count:,}) -> {output.parent}")
    return 0


def _initialize_arm(args: argparse.Namespace) -> int:
    """Construct only the frozen arm needed by one fresh canonical worker."""

    if args.arm not in ARMS:
        raise ValueError(f"unknown arm {args.arm!r}")
    task, dataset = _task_and_dataset(args.task, args.dataset_id)
    output = args.output.resolve() / args.dataset_id / f"seed_{args.seed}" / args.arm
    if output.exists():
        raise FileExistsError(f"refusing to overwrite initialization output: {output}")
    output.mkdir(parents=True)
    guard = NoImageGuard()
    with guard:
        import numpy as np
        import torch

        from rtgs.lift.beam_fusion import BeamFusionConfig
        from rtgs.lift.compact_carve import _center_and_extent
        from rtgs.lift.paper_initializers import (
            PaperInitializerConfig,
            build_frozen_paper_initialization,
        )
        from rtgs.lift.splat_sfm import SplatSfMConfig

        _compact, train, _heldout = _load_split_inputs(task, dataset)
        common_center, common_extent = _center_and_extent(train, torch.float64)
        initializer_settings = _initializer_settings(task)
        frozen_count = task["frozen_configuration"]["initializer"]["common_starting_count"]
        if isinstance(frozen_count, bool) or not isinstance(frozen_count, int):
            raise ValueError("single-arm execution requires a frozen integer common count")
        config = PaperInitializerConfig(
            random_seed=args.seed,
            random_bounds_scale=initializer_settings["random_bounds_scale"],
            init_opacity=initializer_settings["init_opacity"],
            max_starting_gaussians=initializer_settings["max_starting_gaussians"],
            structural_components_per_view=initializer_settings["structural_components_per_view"],
            sfm=SplatSfMConfig(
                init_opacity=initializer_settings["init_opacity"],
                **initializer_settings["splat_sfm"],
            ),
            beam=BeamFusionConfig(
                init_opacity=initializer_settings["init_opacity"],
                **initializer_settings["beam_fusion"],
            ),
        )
        started = time.perf_counter()
        result = build_frozen_paper_initialization(
            train,
            config,
            arm=args.arm,
            count=frozen_count,
        )
        elapsed = time.perf_counter() - started
        result.gaussians.save_npz(output / "gaussians_init.npz")
        result.gaussians.save_ply(output / "gaussians_init.ply")
        np.savez_compressed(
            output / "initializer_lineage.npz",
            **{name: tensor.detach().cpu().numpy() for name, tensor in result.lineage.items()},
        )
        guard_record = guard.record()
        if not guard_record["passed"]:
            raise RuntimeError(f"single-arm initialization boundary failed: {guard_record}")
        _write_json_atomic(
            output / "initializer_receipt.json",
            {
                "schema": "rtgs.paper_single_initialization.v1",
                "evidence_status": _evidence_status(task),
                "task_id": TASK_ID,
                "dataset_id": args.dataset_id,
                "seed": args.seed,
                "arm": args.arm,
                "train_views": train.view_names,
                "heldout_views_excluded": task["splits"][args.dataset_id]["heldout"],
                "elapsed_seconds": elapsed,
                "common_count": result.gaussians.n,
                "common_training_geometry": {
                    "policy": "camera_axis_fallback_from_frozen_train_cameras",
                    "center": common_center.tolist(),
                    "extent": common_extent,
                },
                "initializer_receipt": result.receipt,
                "lineage_keys": sorted(result.lineage),
                "input_boundary": guard_record,
                "torch_seed": torch.initial_seed(),
            },
        )
    print(f"{args.arm} exact-count initialization ({result.gaussians.n:,}) -> {output}")
    return 0


def _query_indexes(inputs: Any) -> list[Any]:
    from rtgs.core.observation2d_cuda import GaussianObservationIndexCuda

    return [
        GaussianObservationIndexCuda.from_field(
            field,
            **QUERY_INDEX_CONFIG,
            device=field.device,
        )
        for field in inputs.observations
    ]


def _sampled_evaluation(
    inputs: Any,
    model: Any,
    *,
    seed: int,
    samples_per_view: int,
    indexes: Sequence[Any] | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    import torch

    from rtgs.render.gsplat_points import GsplatPointRasterizer

    indexes = _query_indexes(inputs) if indexes is None else indexes
    renderer = (
        GsplatPointRasterizer(
            antialiased=POINT_RENDER_CONFIG["antialiased"],
            kernel_support_mode=POINT_RENDER_CONFIG["kernel_support_mode"],
        )
        if renderer is None
        else renderer
    )
    per_view = []
    with torch.no_grad():
        for view_index, (field, camera, index) in enumerate(
            zip(inputs.observations, inputs.cameras, indexes, strict=True)
        ):
            generator = torch.Generator(device=model.means.device).manual_seed(
                seed + 10_000 + view_index
            )
            unit = torch.rand(
                samples_per_view,
                2,
                generator=generator,
                device=model.means.device,
                dtype=model.means.dtype,
            )
            fit_x, fit_y, fit_width, fit_height = field.fit_window
            xy = unit * unit.new_tensor([fit_width, fit_height])
            xy = xy + unit.new_tensor([fit_x, fit_y])
            target = index.query(xy).color
            predicted = renderer.render_points(model, camera, xy).color
            mse = float((predicted - target).square().mean())
            per_view.append(
                {
                    "view_id": inputs.view_names[view_index],
                    "samples": samples_per_view,
                    "xy_sha256": hashlib.sha256(
                        xy.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "uniform_fit_window_mse": mse,
                }
            )
    return {
        "schema": "rtgs.sampled_compact_evaluation.v1",
        "samples_per_view": samples_per_view,
        "equal_view_uniform_fit_window_mse": sum(
            item["uniform_fit_window_mse"] for item in per_view
        )
        / len(per_view),
        "per_view": per_view,
    }


def _convergence_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    stable_best_multiplier: float,
    maximum_absolute_tail_relative_change: float,
) -> dict[str, Any]:
    if not records:
        raise ValueError("convergence summary requires at least one checkpoint")
    ordered = sorted(records, key=lambda item: int(item["step"]))
    steps = [int(item["step"]) for item in ordered]
    if len(steps) != len(set(steps)):
        raise ValueError("convergence checkpoint steps must be unique")
    risks = [float(item["evaluation"]["equal_view_uniform_fit_window_mse"]) for item in ordered]
    if any(not math.isfinite(risk) or risk < 0.0 for risk in risks):
        raise ValueError("convergence risks must be finite and non-negative")
    best_index = min(range(len(risks)), key=risks.__getitem__)
    best = risks[best_index]
    threshold = best * stable_best_multiplier
    stable_step = next(
        (
            steps[index]
            for index in range(len(steps))
            if all(risk <= threshold for risk in risks[index:])
        ),
        None,
    )
    previous = risks[-2] if len(risks) > 1 else risks[-1]
    tail_change = 0.0 if previous == 0.0 else (risks[-1] - previous) / previous
    final_to_best = 1.0 if best == 0.0 and risks[-1] == 0.0 else risks[-1] / max(best, 1e-30)
    return {
        "schema": "rtgs.sampled_compact_convergence.v1",
        "best_step": steps[best_index],
        "best_risk": best,
        "final_step": steps[-1],
        "final_risk": risks[-1],
        "final_to_best_risk_ratio": final_to_best,
        "stable_best_multiplier": stable_best_multiplier,
        "iterations_to_stable_best_band": stable_step,
        "tail_start_step": steps[-2] if len(steps) > 1 else steps[-1],
        "tail_relative_risk_change": tail_change,
        "maximum_absolute_tail_relative_change": maximum_absolute_tail_relative_change,
        "converged_by_frozen_rule": (
            final_to_best <= stable_best_multiplier
            and abs(tail_change) <= maximum_absolute_tail_relative_change
        ),
    }


def _train(args: argparse.Namespace) -> int:
    if args.arm not in ARMS:
        raise ValueError(f"unknown arm {args.arm!r}")
    task, dataset = _task_and_dataset(args.task, args.dataset_id)
    cell = args.output.resolve() / args.dataset_id / f"seed_{args.seed}" / args.arm
    init_path = cell / "gaussians_init.npz"
    if not init_path.is_file():
        raise FileNotFoundError(f"run initialize first: {init_path}")
    final_path = cell / "gaussians.ply"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite trained endpoint: {final_path}")
    checkpoints = cell / "checkpoints"
    checkpoints.mkdir()
    guard = NoImageGuard()
    with guard:
        import torch

        from rtgs.core.gaussians3d import Gaussians3D
        from rtgs.lift.compact_carve import _center_and_extent
        from rtgs.optim.compact_density import ClassicCompactDensityController
        from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer
        from rtgs.optim.density import DensityConfig
        from rtgs.render.gsplat_points import GsplatPointRasterizer

        if getattr(args, "reset_peak_memory_stats", True):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        _compact, train_cpu, heldout_cpu = _load_split_inputs(task, dataset)
        _common_center, common_extent = _center_and_extent(train_cpu, torch.float64)
        train = train_cpu.to("cuda")
        init = Gaussians3D.load_npz(init_path)
        train_config = CompactTrainConfig(
            seed=args.seed,
            extent=common_extent,
            **TRAIN_CONFIG,
        )
        density_config = DensityConfig(**DENSITY_CONFIG)
        trainer = CompactTrainer(
            train_config,
            point_rasterizer=GsplatPointRasterizer(
                absgrad=density_config.absgrad,
                antialiased=POINT_RENDER_CONFIG["antialiased"],
                kernel_support_mode=POINT_RENDER_CONFIG["kernel_support_mode"],
            ),
        )
        train_indexes = _query_indexes(train)

        def checkpoint_callback(snapshot: Any, step: int) -> None:
            snapshot.save_npz(checkpoints / f"gaussians_step_{step:06d}.npz")
            snapshot.save_ply(checkpoints / f"gaussians_step_{step:06d}.ply")

        started = time.perf_counter()
        final, history = trainer.train(
            train,
            init,
            query_backends=train_indexes,
            proposal_query_backends=train_indexes,
            checkpoint_callback=checkpoint_callback,
            topology_controller=ClassicCompactDensityController(
                density_config,
                seed=args.seed,
            ),
            stop_after_step=args.stop_after_step,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        final.save_npz(cell / "gaussians.npz")
        final.save_ply(final_path)
        heldout = heldout_cpu.to("cuda")
        heldout_indexes = _query_indexes(heldout)
        heldout_renderer = GsplatPointRasterizer(
            antialiased=POINT_RENDER_CONFIG["antialiased"],
            kernel_support_mode=POINT_RENDER_CONFIG["kernel_support_mode"],
        )
        convergence_records = []
        for checkpoint_path in sorted(checkpoints.glob("gaussians_step_*.npz")):
            step = int(checkpoint_path.stem.removeprefix("gaussians_step_"))
            snapshot = Gaussians3D.load_npz(checkpoint_path).to("cuda")
            convergence_records.append(
                {
                    "step": step,
                    "evaluation": _sampled_evaluation(
                        heldout,
                        snapshot,
                        seed=args.seed,
                        samples_per_view=CONVERGENCE_CONFIG["samples_per_view"],
                        indexes=heldout_indexes,
                        renderer=heldout_renderer,
                    ),
                }
            )
            del snapshot
        convergence = _convergence_summary(
            convergence_records,
            stable_best_multiplier=CONVERGENCE_CONFIG["stable_best_multiplier"],
            maximum_absolute_tail_relative_change=CONVERGENCE_CONFIG[
                "maximum_absolute_tail_relative_change"
            ],
        )
        sampled = _sampled_evaluation(
            heldout,
            final.to("cuda"),
            seed=args.seed,
            samples_per_view=args.evaluation_samples,
            indexes=heldout_indexes,
            renderer=heldout_renderer,
        )
        guard_record = guard.record()
        if not guard_record["passed"]:
            raise RuntimeError(f"training input boundary failed: {guard_record}")
        _write_json_atomic(cell / "training_history.json", history)
        _write_json_atomic(
            cell / "summary.json",
            {
                "schema": "rtgs.paper_three_path_training.v1",
                "evidence_status": _evidence_status(task),
                "task_id": TASK_ID,
                "dataset_id": args.dataset_id,
                "seed": args.seed,
                "arm": args.arm,
                "train_views": train.view_names,
                "heldout_views": heldout.view_names,
                "initial_gaussians": init.n,
                "final_gaussians": final.n,
                "completed_iterations": history.get(
                    "completed_iterations",
                    TRAIN_CONFIG["iterations"],
                ),
                "elapsed_seconds": elapsed,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "topology_control": history["topology_control"],
                "sampled_heldout_evaluation": sampled,
                "sampled_heldout_convergence": convergence_records,
                "convergence_summary": convergence,
                "train_config": TRAIN_CONFIG,
                "common_training_extent": common_extent,
                "density_config": DENSITY_CONFIG,
                "query_index_config": QUERY_INDEX_CONFIG,
                "point_render_config": POINT_RENDER_CONFIG,
                "input_boundary": guard_record,
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            },
        )
    print(f"{args.arm}: {init.n:,} -> {final.n:,} Gaussians in {elapsed:.1f}s")
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _protocol_sha256(task: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in task.items() if key not in {"protocol_review", "status"}}
    )


def _resolve_exact_path(value: Path, expected: Path, *, label: str) -> Path:
    resolved = value.resolve()
    target = expected.resolve()
    if resolved != target:
        raise ValueError(f"{label} must resolve exactly to {target}, received {resolved}")
    return resolved


def _current_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_binding_passes(task: Mapping[str, Any], lock: Mapping[str, Any]) -> bool:
    binding = task.get("source_binding")
    if not isinstance(binding, dict) or set(binding) != {"reviewed_base_commit", "files"}:
        return False
    base = binding.get("reviewed_base_commit")
    files = binding.get("files")
    if not isinstance(base, str) or not isinstance(files, dict) or not files:
        return False
    current = _current_git_commit()
    if lock.get("source_commit") != current:
        return False
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, current],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        return False
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            return False
        path = ROOT / relative
        try:
            path.resolve(strict=True).relative_to(ROOT.resolve())
        except (FileNotFoundError, ValueError):
            return False
        if _sha256_file(path) != expected:
            return False
    return True


def _validate_run_binding(task_path: Path, run: Path) -> dict[str, Any]:
    task_path = _resolve_exact_path(task_path, DEFAULT_TASK, label="task")
    run = _resolve_exact_path(run, DEFAULT_RUN, label="run directory")
    task = _load_json(task_path)
    if task.get("task_id") != TASK_ID:
        raise ValueError("task id does not match this frozen driver")
    if task.get("status") != "ready":
        raise ValueError("canonical execution requires task status ready")
    review = task.get("protocol_review")
    if not isinstance(review, dict) or review.get("verdict") != "approved":
        raise ValueError("canonical execution requires approved prospective review")
    common_count = task["frozen_configuration"]["initializer"]["common_starting_count"]
    if isinstance(common_count, bool) or not isinstance(common_count, int) or common_count <= 0:
        raise ValueError("canonical execution requires a frozen positive common starting count")
    frozen = task["frozen_configuration"]
    compact = frozen["compact_3dgs"]
    expected_compact = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in TRAIN_CONFIG.items()
    }
    observed_compact = {key: compact.get(key) for key in expected_compact}
    if observed_compact != expected_compact:
        raise ValueError("frozen compact_3dgs configuration differs from driver constants")
    if compact.get("convergence_evaluation") != CONVERGENCE_CONFIG:
        raise ValueError("frozen convergence evaluation differs from driver constants")
    if frozen["classic_density"] != DENSITY_CONFIG:
        raise ValueError("frozen classic_density configuration differs from driver constants")
    if frozen["compact_query_index"] != QUERY_INDEX_CONFIG:
        raise ValueError("frozen compact_query_index differs from driver constants")
    if frozen["point_renderer"] != POINT_RENDER_CONFIG:
        raise ValueError("frozen point_renderer differs from driver constants")
    if frozen.get("resource_measurement") != RESOURCE_CONFIG:
        raise ValueError("frozen resource_measurement differs from driver constants")
    if frozen.get("presentation") != PRESENTATION_CONFIG:
        raise ValueError("frozen presentation differs from driver constants")
    lock_path = run / "task.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"{lock_path} is missing; initialize the protected run through experiment_contract.py"
        )
    lock = _load_json(lock_path)
    review_path = ROOT / str(review.get("artifact"))
    seal_path = ROOT / str(task["data_seal"])
    checks = {
        "task_id": lock.get("task_id") == TASK_ID,
        "task_path": lock.get("task_path") == DEFAULT_TASK.relative_to(ROOT).as_posix(),
        "task_sha256": lock.get("task_sha256") == _sha256_file(task_path),
        "protocol_sha256": lock.get("protocol_sha256") == _protocol_sha256(task),
        "protocol_review": lock.get("protocol_review") == review,
        "protocol_review_artifact_sha256": (
            review_path.is_file()
            and lock.get("protocol_review_artifact_sha256") == _sha256_file(review_path)
        ),
        "data_seal_path": lock.get("data_seal_path") == task["data_seal"],
        "data_seal_sha256": (
            seal_path.is_file() and lock.get("data_seal_sha256") == _sha256_file(seal_path)
        ),
        "command": lock.get("command") == task["run_command"],
        "report_template_version": lock.get("report_template_version") == 2,
        "official_clean_source": (
            lock.get("development") is False
            and lock.get("source_dirty") is False
            and lock.get("source_diff_sha256") == hashlib.sha256(b"").hexdigest()
        ),
        "reviewed_source_binding": _source_binding_passes(task, lock),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("run/task lock binding failed: " + ", ".join(failed))
    return task


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    _write_json_atomic(path, value)


def _dataset_record(task: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [item for item in task["datasets"] if item["id"] == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"task does not contain exactly one dataset {dataset_id!r}")
    return matches[0]


def _cell_relative(dataset_id: str, seed: int, arm: str, *, warmup: bool) -> Path:
    prefix = "warmups" if warmup else "cells"
    return Path(prefix) / dataset_id / f"seed_{seed}" / arm


def _sealed_input_records(
    task: Mapping[str, Any], dataset: Mapping[str, Any]
) -> list[dict[str, Any]]:
    seal = _load_json(ROOT / str(task["data_seal"]))
    manifest_parent = Path(str(dataset["compact_manifest"])).parent
    selected = []
    for item in seal.get("files", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        path = Path(item["path"])
        if path == Path(str(dataset["calibration"])) or path.parent == manifest_parent:
            selected.append(item)
    if not selected:
        raise RuntimeError(f"data seal does not bind files for {dataset['id']}")
    return selected


def _output_file_records(cell: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in cell.rglob("*") if item.is_file()):
        if path.name in {"resource_receipt.json", "cell_receipt.json"} or "previews" in path.parts:
            continue
        records.append(
            {
                "path": path.relative_to(cell).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _validate_resource_record(record: Mapping[str, Any]) -> None:
    required = {
        "wall_seconds",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "nvml_process_peak_bytes",
        "background_device_memory_bytes",
        "background_device_memory_peak_bytes",
        "device_total_bytes",
        "driver_version",
        "torch",
        "torch_cuda",
        "ru_maxrss_bytes",
        "compact_input_bytes",
        "compact_field_bytes",
        "final_model_npz_bytes",
        "final_model_ply_bytes",
        "compact_to_model_compression_ratio",
        "output_bytes",
        "output_files",
        "idle_guard",
        "foreign_compute_processes",
    }
    missing = sorted(required - set(record))
    if missing:
        raise RuntimeError("resource receipt is missing frozen fields: " + ", ".join(missing))
    numeric = required - {
        "driver_version",
        "torch",
        "torch_cuda",
        "output_files",
        "idle_guard",
        "foreign_compute_processes",
    }
    for key in numeric:
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise RuntimeError(f"resource receipt field {key} must be non-negative numeric")
    for key in (
        "compact_field_bytes",
        "final_model_npz_bytes",
        "final_model_ply_bytes",
        "compact_to_model_compression_ratio",
    ):
        if float(record[key]) <= 0.0:
            raise RuntimeError(f"resource receipt field {key} must be positive")
    if record["foreign_compute_processes"] != []:
        raise RuntimeError("resource receipt records foreign CUDA compute interference")
    idle = record["idle_guard"]
    if not isinstance(idle, dict) or idle.get("passed") is not True:
        raise RuntimeError("resource receipt lacks a passing GPU quiescence guard")
    files = record["output_files"]
    if not isinstance(files, list) or not files:
        raise RuntimeError("resource receipt output_files must be a non-empty list")
    if sum(int(item["bytes"]) for item in files) != int(record["output_bytes"]):
        raise RuntimeError("resource receipt output byte accounting is inconsistent")


def _render_cell_previews(
    task: Mapping[str, Any],
    dataset: Mapping[str, Any],
    cell: Path,
    *,
    view_id: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.data.compact_views import CompactDataset
    from rtgs.render.base import get_rasterizer

    compact = CompactDataset.load(
        (ROOT / str(dataset["compact_manifest"])).parent,
        device="cpu",
        byte_cap=_byte_cap(dict(task)),
        load_alpha=False,
    )
    lookup = {view.view_id: view for view in compact.views}
    if view_id not in lookup:
        raise ValueError(f"presentation view {view_id!r} is absent")
    camera = lookup[view_id].camera.to("cuda")
    renderer = get_rasterizer("gsplat", device="cuda", packed=True, antialiased=False)
    preview = cell / "previews"
    preview.mkdir()
    records: dict[str, Any] = {}
    for state, source in (
        ("initial", cell / "gaussians_init.ply"),
        ("final", cell / "gaussians.ply"),
    ):
        model = Gaussians3D.load_ply(source).to("cuda")
        with torch.no_grad():
            color = renderer.render(model, camera).color.clamp(0.0, 1.0)
        output = preview / f"{state}_{view_id}_native.png"
        Image.fromarray((color.cpu().numpy() * 255).round().astype(np.uint8)).save(output)
        records[state] = {
            "path": output.relative_to(cell).as_posix(),
            "width": camera.width,
            "height": camera.height,
            "bytes": output.stat().st_size,
            "sha256": _sha256_file(output),
        }
        del model, color
        torch.cuda.empty_cache()
    return {"view_id": view_id, "view_role": "heldout", "images": records}


def _failure_relative(dataset_id: str, seed: int, arm: str, *, warmup: bool) -> Path:
    kind = "warmups" if warmup else "cells"
    return Path("failures") / kind / dataset_id / f"seed_{seed}" / arm


def _publish_worker_failure(
    temporary: Path,
    run: Path,
    *,
    dataset_id: str,
    seed: int,
    arm: str,
    warmup: bool,
    phase: str,
    error: BaseException,
) -> Path:
    target = run / _failure_relative(dataset_id, seed, arm, warmup=warmup)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite worker failure: {target}") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = [
        {
            "path": path.relative_to(temporary).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in temporary.rglob("*") if item.is_file())
    ]
    _write_json_new(
        temporary / "failure.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "failed",
            "failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "seed": seed,
            "arm": arm,
            "warmup": warmup,
            "failure_phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "partial_files": partial,
        },
    )
    os.replace(temporary, target)
    return target


def _canonical_worker(
    *,
    task_path: Path,
    run: Path,
    dataset_id: str,
    seed: int,
    arm: str,
    warmup: bool,
) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown initializer {arm!r}")
    task_path = _resolve_exact_path(task_path, DEFAULT_TASK, label="task")
    run = _resolve_exact_path(run, DEFAULT_RUN, label="run directory")
    task = _validate_run_binding(task_path, run)
    if not warmup and seed not in task["seeds"]:
        raise ValueError(f"measured seed {seed} is not frozen in the task")
    dataset = _dataset_record(task, dataset_id)
    target = run / _cell_relative(dataset_id, seed, arm, warmup=warmup)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite experiment cell: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{arm}.worker-", dir=target.parent))
    legacy_root = temporary / "legacy"
    process_started = time.perf_counter()
    nvml_sampler = NvmlProcessSampler(task["frozen_configuration"]["resource_measurement"])
    nvml_sampler_started = False
    failure_phase = "resource_preflight"
    try:
        nvml_sampler.start()
        nvml_sampler_started = True
        import torch

        from rtgs.data.compact_views import CompactDataset

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        resource_started = time.perf_counter()
        failure_phase = "load_provider_fields"
        load_started = time.perf_counter()
        compact = CompactDataset.load(
            (ROOT / dataset["compact_manifest"]).parent,
            device="cpu",
            byte_cap=_byte_cap(task),
            load_alpha=False,
        )
        field_semantics = [
            {
                "view_id": view.view_id,
                "provider": view.observation.provider,
                "blend_mode": view.observation.blend_mode,
                "sigma_cutoff": view.observation.sigma_cutoff,
                "support_fade_alpha": view.observation.support_fade_alpha,
                "aa_dilation": view.observation.aa_dilation,
                "n_gaussians": view.observation.n,
                "canvas": [view.observation.width, view.observation.height],
            }
            for view in compact.views
        ]
        factor_key = dataset_id.removeprefix("frame_00008_")
        expected_semantics = task["frozen_configuration"]["provider_factor"].get(factor_key)
        if not isinstance(expected_semantics, dict):
            raise ValueError(f"no frozen provider semantics for {dataset_id}")
        semantic_checks = {
            "provider": expected_semantics["rtgsv_provider"],
            "blend_mode": expected_semantics["blend_mode"],
            "sigma_cutoff": expected_semantics["sigma_cutoff"],
            "support_fade_alpha": expected_semantics["support_fade_alpha"],
            "aa_dilation": expected_semantics["aa_dilation"],
        }
        for view in field_semantics:
            failed = [name for name, expected in semantic_checks.items() if view[name] != expected]
            if view["canvas"] != expected_semantics["canvas"]:
                failed.append("canvas")
            if view["n_gaussians"] > expected_semantics["maximum_capacity_per_view"]:
                failed.append("maximum_capacity_per_view")
            if failed:
                raise RuntimeError(
                    f"{dataset_id}/{view['view_id']} violates frozen semantics: "
                    + ", ".join(failed)
                )
        del compact
        load_seconds = time.perf_counter() - load_started

        failure_phase = "construct_initializations"
        initialize_started = time.perf_counter()
        _initialize_arm(
            argparse.Namespace(
                task=task_path,
                dataset_id=dataset_id,
                seed=seed,
                output=legacy_root,
                arm=arm,
            )
        )
        initialize_seconds = time.perf_counter() - initialize_started
        failure_phase = "compact_3dgs_with_density"
        train_started = time.perf_counter()
        _train(
            argparse.Namespace(
                task=task_path,
                dataset_id=dataset_id,
                seed=seed,
                output=legacy_root,
                arm=arm,
                stop_after_step=None,
                reset_peak_memory_stats=False,
                evaluation_samples=int(
                    task["frozen_configuration"]["compact_3dgs"][
                        "heldout_evaluation_samples_per_view"
                    ]
                ),
            )
        )
        train_call_seconds = time.perf_counter() - train_started
        legacy_base = legacy_root / dataset_id / f"seed_{seed}"
        cell = legacy_base / arm
        summary = _load_json(cell / "summary.json")
        fitting_seconds = float(summary["elapsed_seconds"])
        failure_phase = "heldout_compact_evaluation"
        evaluation_seconds = max(0.0, train_call_seconds - fitting_seconds)
        resource_wall_seconds = time.perf_counter() - resource_started
        nvml_record = nvml_sampler.stop()
        nvml_sampler_started = False
        torch_device_total = int(torch.cuda.get_device_properties(0).total_memory)
        if nvml_record["device_total_bytes"] != torch_device_total:
            raise RuntimeError("NVML and torch disagree on the frozen device memory capacity")
        raw_history = cell / "training_history.json"
        raw_history.rename(cell / "training_history.raw.json")
        input_files = _sealed_input_records(task, dataset)
        _write_json_new(
            cell / "input_boundary_receipt.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "dataset_id": dataset_id,
                "seed": seed,
                "arm": arm,
                "warmup": warmup,
                "allowed_modalities": ["calibration", "gaussians2d"],
                "compact_alpha_loaded": False,
                "embedded_alpha_used_for_reconstruction": False,
                "heldout_training_access": False,
                "source_rgb_or_mask_opened": False,
                "sealed_files": input_files,
                "input_bytes": sum(int(item["bytes"]) for item in input_files),
                "field_semantics": field_semantics,
                "initializer_guard": _load_json(cell / "initializer_receipt.json")[
                    "input_boundary"
                ],
                "training_guard": summary["input_boundary"],
            },
        )
        _write_json_new(
            cell / "gaussians.config.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "dataset_id": dataset_id,
                "seed": seed,
                "arm": arm,
                "warmup": warmup,
                "provider_factor": task["frozen_configuration"]["provider_factor"],
                "initializer": task["frozen_configuration"]["initializer"],
                "compact_3dgs": task["frozen_configuration"]["compact_3dgs"],
                "classic_density": task["frozen_configuration"]["classic_density"],
            },
        )
        compact_input_bytes = sum(int(item["bytes"]) for item in input_files)
        compact_field_bytes = sum(
            int(item["bytes"]) for item in input_files if Path(str(item["path"])).suffix == ".rtgsv"
        )
        final_model_npz_bytes = (cell / "gaussians.npz").stat().st_size
        final_model_ply_bytes = (cell / "gaussians.ply").stat().st_size
        output_files = _output_file_records(cell)
        output_bytes = sum(int(item["bytes"]) for item in output_files)
        resource_record = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "dataset_id": dataset_id,
            "seed": seed,
            "arm": arm,
            "warmup": warmup,
            "scope": (
                "fresh image-free process from sealed compact-manifest load through final model "
                "save and frozen held-out compact sample evaluation; excludes generated "
                "full-resolution previews and this self-referential receipt"
            ),
            "initializer_scope": (
                "construct only the selected frozen initializer arm at the globally matched "
                "count; unused comparator initializers are excluded"
            ),
            "wall_seconds": resource_wall_seconds,
            "process_wall_seconds_before_preview": time.perf_counter() - process_started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "compact_input_bytes": compact_input_bytes,
            "compact_field_bytes": compact_field_bytes,
            "final_model_npz_bytes": final_model_npz_bytes,
            "final_model_ply_bytes": final_model_ply_bytes,
            "compact_to_model_compression_ratio": compact_field_bytes / final_model_npz_bytes,
            "output_bytes": output_bytes,
            "output_files": output_files,
            "device_total_bytes": torch_device_total,
            "device_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            **nvml_record,
        }
        _validate_resource_record(resource_record)
        _write_json_new(cell / "resource_receipt.json", resource_record)
        failure_phase = "fullres_presentation"
        presentation_started = time.perf_counter()
        preview = _render_cell_previews(
            task,
            dataset,
            cell,
            view_id=task["frozen_configuration"]["presentation"]["calibrated_view_id"],
        )
        presentation_seconds = time.perf_counter() - presentation_started
        stage_timing = {
            "load_provider_fields": load_seconds,
            "construct_initializations": initialize_seconds,
            "compact_3dgs_with_density": fitting_seconds,
            "heldout_compact_evaluation": evaluation_seconds,
            "fullres_presentation": presentation_seconds,
        }
        _write_json_new(
            cell / "cell_receipt.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "dataset_id": dataset_id,
                "dataset_role": dataset["role"],
                "seed": seed,
                "arm": arm,
                "warmup": warmup,
                "status": "completed",
                "common_starting_count": task["frozen_configuration"]["initializer"][
                    "common_starting_count"
                ],
                "stage_wall_seconds": stage_timing,
                "preview": preview,
                "summary_sha256": _sha256_file(cell / "summary.json"),
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        os.replace(cell, target)
    except Exception as error:
        if nvml_sampler_started:
            with contextlib.suppress(Exception):
                nvml_sampler.stop()
            nvml_sampler_started = False
        _publish_worker_failure(
            temporary,
            run,
            dataset_id=dataset_id,
            seed=seed,
            arm=arm,
            warmup=warmup,
            phase=failure_phase,
            error=error,
        )
        raise
    finally:
        if nvml_sampler_started:
            with contextlib.suppress(Exception):
                nvml_sampler.stop()
        if target.exists() and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _preflight(task_path: Path, output: Path) -> int:
    task = _load_json(task_path.resolve())
    if task.get("task_id") != TASK_ID or task.get("status") != "draft":
        raise ValueError("initializer preflight requires the matching draft task")
    if task["frozen_configuration"]["initializer"]["common_starting_count"] is not None:
        raise ValueError("common starting count is already frozen")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight output: {output}")
    seed = max(0, min(int(value) for value in task["seeds"]) - 1)
    records = []
    for dataset in task["datasets"]:
        _initialize(
            argparse.Namespace(
                task=task_path,
                dataset_id=dataset["id"],
                seed=seed,
                output=output,
            )
        )
        receipt = _load_json(
            output / dataset["id"] / f"seed_{seed}" / "initializations" / "receipt.json"
        )
        records.append(
            {
                "dataset_id": dataset["id"],
                "seed": seed,
                "common_count_at_cap": receipt["common_count"],
                "initializer_receipt": receipt["initializer_receipt"],
                "receipt_sha256": _sha256_file(
                    output / dataset["id"] / f"seed_{seed}" / "initializations" / "receipt.json"
                ),
            }
        )
    recommended = min(int(item["common_count_at_cap"]) for item in records)
    _write_json_new(
        output / "preflight.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "evidence_status": "train_only_mechanism_preflight_no_downstream_outcomes",
            "seed": seed,
            "records": records,
            "recommended_global_common_starting_count": recommended,
        },
    )
    print(f"global matched initializer recommendation: {recommended:,}")
    return 0


def _environment_record() -> dict[str, Any]:
    import importlib.metadata

    import torch

    packages = {}
    for name in ("numpy", "pillow", "torch", "gsplat", "realtime-gs"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "source-checkout"
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
        },
    }


def _measured_cells(task: Mapping[str, Any], run: Path) -> list[dict[str, Any]]:
    records = []
    for dataset in task["datasets"]:
        for seed in task["seeds"]:
            for arm in ARMS:
                cell = run / _cell_relative(dataset["id"], int(seed), arm, warmup=False)
                if not cell.is_dir():
                    raise FileNotFoundError(f"missing measured cell: {cell}")
                resource_record = _load_json(cell / "resource_receipt.json")
                _validate_resource_record(resource_record)
                records.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "arm": arm,
                        "cell": cell,
                        "summary": _load_json(cell / "summary.json"),
                        "resource": resource_record,
                        "receipt": _load_json(cell / "cell_receipt.json"),
                        "history": _load_json(cell / "training_history.raw.json"),
                    }
                )
    return records


def _history_bundle(
    task: Mapping[str, Any],
    run: Path,
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = []
    markers = []
    stage_labels = {stage["id"]: stage["label"] for stage in task["stages"]}
    iterations = int(task["frozen_configuration"]["compact_3dgs"]["iterations"])
    for item in cells:
        dataset_id = item["dataset"]["id"]
        arm = item["arm"]
        seed = item["seed"]
        timing = item["receipt"]["stage_wall_seconds"]
        boundaries = (
            ("load_provider_fields", 0, 0),
            ("construct_initializations", 0, 0),
            ("compact_3dgs_with_density", 0, iterations),
            ("heldout_compact_evaluation", iterations, iterations),
            ("fullres_presentation", iterations, iterations),
        )
        elapsed = 0.0
        stage_ranges: dict[str, tuple[float, float]] = {}
        for stage, start_step, end_step in boundaries:
            start = elapsed
            elapsed += float(timing[stage])
            stage_ranges[stage] = (start, elapsed)
            markers.extend(
                [
                    {
                        "step": start_step,
                        "wall_seconds": start,
                        "stage": stage,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "boundary": "start",
                        "label": stage_labels[stage],
                    },
                    {
                        "step": end_step,
                        "wall_seconds": elapsed,
                        "stage": stage,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "boundary": "end",
                        "label": stage_labels[stage],
                    },
                ]
            )
            if stage != "compact_3dgs_with_density":
                records.append(
                    {
                        "step": end_step,
                        "wall_seconds": elapsed,
                        "stage": stage,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "split": "diagnostic",
                        "metric_id": "stage_wall_seconds",
                        "value": float(timing[stage]),
                    }
                )
        compact_start, compact_end = stage_ranges["compact_3dgs_with_density"]
        raw_steps = item["history"].get("steps", [])
        cumulative = compact_start
        selected_steps = []
        wall_by_step = {0: compact_start}
        for raw in raw_steps:
            cumulative += float(raw["elapsed_seconds"])
            step = int(raw["step"])
            wall_by_step[step] = min(cumulative, compact_end)
            if step == 1 or step % 100 == 0 or step == iterations:
                selected_steps.append((raw, min(cumulative, compact_end)))
        for raw, wall_seconds in selected_steps:
            records.append(
                {
                    "step": int(raw["step"]),
                    "wall_seconds": wall_seconds,
                    "stage": "compact_3dgs_with_density",
                    "dataset_id": dataset_id,
                    "arm_id": arm,
                    "seed": seed,
                    "split": "train",
                    "metric_id": "sampled_train_loss",
                    "value": float(raw["total_sampled_loss"]),
                }
            )
        for checkpoint in item["summary"].get("sampled_heldout_convergence", []):
            step = int(checkpoint["step"])
            records.append(
                {
                    "step": step,
                    "wall_seconds": wall_by_step.get(step, compact_end),
                    "stage": "compact_3dgs_with_density",
                    "dataset_id": dataset_id,
                    "arm_id": arm,
                    "seed": seed,
                    "split": "validation",
                    "metric_id": "heldout_checkpoint_sampled_j_area",
                    "value": float(checkpoint["evaluation"]["equal_view_uniform_fit_window_mse"]),
                }
            )
        evaluation_end = stage_ranges["heldout_compact_evaluation"][1]
        records.append(
            {
                "step": iterations,
                "wall_seconds": evaluation_end,
                "stage": "heldout_compact_evaluation",
                "dataset_id": dataset_id,
                "arm_id": arm,
                "seed": seed,
                "split": "validation",
                "metric_id": "heldout_sampled_j_area",
                "value": float(
                    item["summary"]["sampled_heldout_evaluation"][
                        "equal_view_uniform_fit_window_mse"
                    ]
                ),
            }
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
            "sampled_train_loss": {
                "label": "Sampled compact training loss",
                "unit": "MSE",
                "group": "quality",
                "direction": "lower",
            },
            "heldout_sampled_j_area": {
                "label": "Held-out sampled compact area risk",
                "unit": "MSE",
                "group": "quality",
                "direction": "lower",
            },
            "heldout_checkpoint_sampled_j_area": {
                "label": "Held-out checkpoint sampled compact area risk",
                "unit": "MSE",
                "group": "convergence",
                "direction": "lower",
            },
        },
        "stage_markers": markers,
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _provider_label(dataset_id: str) -> str:
    return dataset_id.removeprefix("frame_00008_").replace("_", " ")


def _labeled_panel(image: Any, label: str, *, max_side: int) -> Any:
    from PIL import Image, ImageDraw

    source = image.convert("RGB")
    scale = min(1.0, max_side / max(source.size))
    if scale < 1.0:
        source = source.resize(
            (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
            Image.Resampling.LANCZOS,
        )
    header = 28
    panel = Image.new("RGB", (source.width, source.height + header), "white")
    panel.paste(source, (0, header))
    ImageDraw.Draw(panel).text((5, 7), label, fill="black")
    return panel


def _comparison_grid(panels: Sequence[Any], *, columns: int) -> Any:
    from PIL import Image

    if not panels or columns <= 0:
        raise ValueError("comparison grid requires panels and positive columns")
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * height), "white")
    for index, panel in enumerate(panels):
        x = (index % columns) * width + (width - panel.width) // 2
        y = (index // columns) * height + (height - panel.height) // 2
        grid.paste(panel, (x, y))
    return grid


def _gif_frame(image: Any, *, max_side: int = 960) -> Any:
    from PIL import Image

    scale = min(1.0, max_side / max(image.size))
    if scale == 1.0:
        return image
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _scaled_camera(camera: Any, factor: int) -> Any:
    from rtgs.core.camera import Camera

    if factor <= 0 or camera.width % factor or camera.height % factor:
        raise ValueError("presentation camera downscale must exactly divide the native canvas")
    return Camera(
        fx=camera.fx / factor,
        fy=camera.fy / factor,
        cx=camera.cx / factor,
        cy=camera.cy / factor,
        width=camera.width // factor,
        height=camera.height // factor,
        R=camera.R,
        t=camera.t,
    )


def _novel_cameras(
    cameras: Sequence[Any],
    center: Any,
    frames: int,
    *,
    vary_elevation: bool,
) -> list[Any]:
    import torch

    from rtgs.core.camera import Camera

    if not cameras or frames <= 0:
        raise ValueError("novel presentation path requires cameras and positive frame count")
    center = center.to(dtype=torch.float32)
    positions = torch.stack([camera.position for camera in cameras]).to(center)
    offsets = positions - center
    covariance = offsets.T @ offsets / max(offsets.shape[0], 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    axis_x = eigenvectors[:, -1]
    axis_y = eigenvectors[:, -2]
    normal = torch.linalg.cross(axis_x, axis_y)
    camera_down = torch.stack([camera.R[1] for camera in cameras]).to(center).mean(dim=0)
    if torch.dot(normal, camera_down) < 0:
        normal = -normal
        axis_y = -axis_y
    radius = offsets.norm(dim=-1).median().clamp_min(1e-3)
    height = torch.quantile((offsets @ normal).abs(), 0.9) * 0.75
    reference = cameras[len(cameras) // 2]
    fov_x = float(
        torch.rad2deg(2.0 * torch.atan(torch.tensor(reference.width / (2.0 * reference.fx))))
    )
    result = []
    for index in range(frames):
        angle = center.new_tensor(2.0 * torch.pi * (index + 0.5) / frames)
        elevation = height * torch.sin(2.0 * angle) if vary_elevation else height.new_zeros(())
        planar_radius = torch.sqrt((radius.square() - elevation.square()).clamp_min(1e-6))
        eye = center + planar_radius * (torch.cos(angle) * axis_x + torch.sin(angle) * axis_y)
        eye = eye + elevation * normal
        camera = Camera.look_at(
            eye,
            center,
            up=normal,
            fov_x_deg=fov_x,
            width=reference.width,
            height=reference.height,
        )
        camera.fx = reference.fx
        camera.fy = reference.fy
        camera.cx = reference.cx
        camera.cy = reference.cy
        result.append(camera)
    return result


def _save_synchronized_animation(
    output: Path,
    *,
    models: Sequence[tuple[str, Any]],
    cameras: Sequence[Any],
    renderer: Any,
    config: Mapping[str, Any],
) -> None:
    import numpy as np
    import torch
    from PIL import Image

    frames = []
    with torch.no_grad():
        for camera in cameras:
            camera = camera.to("cuda")
            panels = []
            for label, model in models:
                color = renderer.render(
                    model,
                    camera,
                    sh_degree=model.sh_degree,
                ).color.clamp(0.0, 1.0)
                image = Image.fromarray(
                    (color.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
                )
                panels.append(
                    _labeled_panel(
                        image,
                        label,
                        max_side=max(1, int(config["panel_max_side"]) // 2),
                    )
                )
            frames.append(
                _gif_frame(
                    _comparison_grid(panels, columns=int(config["grid_columns"])),
                    max_side=960,
                )
            )
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=int(config["animation_duration_ms"]),
        loop=0,
        optimize=False,
    )


def _render_root_previews(
    task: Mapping[str, Any],
    run: Path,
    methods: Sequence[Mapping[str, str]],
) -> None:
    import torch

    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.lift.compact_carve import _center_and_extent
    from rtgs.render.base import get_rasterizer

    config = task["frozen_configuration"]["presentation"]
    if len(methods) != len(task["datasets"]) * len(ARMS):
        raise RuntimeError("root presentation requires every provider/initializer method")
    guard = NoImageGuard()
    with guard:
        first = task["datasets"][0]
        _compact, train, _heldout = _load_split_inputs(dict(task), dict(first))
        center, _extent = _center_and_extent(train, torch.float64)
        cameras = [
            _scaled_camera(camera, int(config["animation_downscale"])) for camera in train.cameras
        ]
        guard_record = guard.record()
    if not guard_record["passed"]:
        raise RuntimeError("root presentation compact-input boundary failed")

    from PIL import Image

    view_id = str(config["calibrated_view_id"])
    initial_panels = []
    final_panels = []
    for method in methods:
        cell = (run / method["final"]).parent
        for state, panels in (("initial", initial_panels), ("final", final_panels)):
            path = cell / "previews" / f"{state}_{view_id}_native.png"
            with Image.open(path) as source:
                panels.append(
                    _labeled_panel(
                        source.copy(),
                        method["name"],
                        max_side=int(config["panel_max_side"]),
                    )
                )
    initial_grid = _comparison_grid(initial_panels, columns=int(config["grid_columns"]))
    final_grid = _comparison_grid(final_panels, columns=int(config["grid_columns"]))
    final_grid.save(run / "reconstruction_contact_sheet.png")
    reconstruction_frames = [_gif_frame(initial_grid), _gif_frame(final_grid)]
    reconstruction_frames[0].save(
        run / "reconstruction.gif",
        save_all=True,
        append_images=reconstruction_frames[1:],
        duration=900,
        loop=0,
        optimize=False,
    )

    model_records = []
    models = []
    for method in methods:
        path = run / method["final"]
        model_records.append(
            {
                "name": method["name"],
                "path": method["final"],
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        models.append((method["name"], Gaussians3D.load_ply(path).to("cuda")))
    renderer = get_rasterizer("gsplat", device="cuda", packed=True, antialiased=False)
    orbit = _novel_cameras(
        cameras,
        center,
        int(config["animation_frames"]),
        vary_elevation=False,
    )
    elevation = _novel_cameras(
        cameras,
        center,
        int(config["animation_frames"]),
        vary_elevation=True,
    )
    _save_synchronized_animation(
        run / "novel_orbit.gif",
        models=models,
        cameras=orbit,
        renderer=renderer,
        config=config,
    )
    _save_synchronized_animation(
        run / "novel_elevation.gif",
        models=models,
        cameras=elevation,
        renderer=renderer,
        config=config,
    )
    del models, renderer
    torch.cuda.empty_cache()
    artifact_records = []
    for name in (
        "reconstruction_contact_sheet.png",
        "reconstruction.gif",
        "novel_orbit.gif",
        "novel_elevation.gif",
    ):
        path = run / name
        artifact_records.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    _write_json_new(
        run / "presentation_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "PASS",
            "scope": "isolated image-free synchronized nine-method presentation",
            "source_rgb_or_mask_opened": False,
            "compact_input_guard": guard_record,
            "configuration": config,
            "models": model_records,
            "artifacts": artifact_records,
        },
    )


def _metrics_bundle(
    task: Mapping[str, Any],
    run: Path,
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    quality = [
        float(item["summary"]["sampled_heldout_evaluation"]["equal_view_uniform_fit_window_mse"])
        for item in cells
    ]
    counts = [float(item["summary"]["final_gaussians"]) for item in cells]
    peaks = [float(item["resource"]["peak_cuda_allocated_bytes"]) for item in cells]
    walls = [float(item["resource"]["wall_seconds"]) for item in cells]
    convergence_ratios = [
        float(item["summary"]["convergence_summary"]["final_to_best_risk_ratio"]) for item in cells
    ]
    converged = [
        bool(item["summary"]["convergence_summary"]["converged_by_frozen_rule"]) for item in cells
    ]
    compression = [float(item["resource"]["compact_to_model_compression_ratio"]) for item in cells]
    teacher_values = []
    chart_quality = []
    chart_resources = []
    chart_runtime = []
    for dataset in task["datasets"]:
        production = _load_json(ROOT / dataset["production_manifest"])
        teacher_values.append(float(production["summary"]["mean_foreground_psnr_db"]))
        for arm in ARMS:
            group = [
                item
                for item in cells
                if item["dataset"]["id"] == dataset["id"] and item["arm"] == arm
            ]
            label = f"{_provider_label(dataset['id'])} / {ARM_LABELS[arm]}"
            chart_quality.append(
                {
                    "label": label,
                    "value": _median(
                        [
                            float(
                                item["summary"]["sampled_heldout_evaluation"][
                                    "equal_view_uniform_fit_window_mse"
                                ]
                            )
                            for item in group
                        ]
                    ),
                }
            )
            chart_resources.append(
                {
                    "label": label,
                    "value": _median(
                        [float(item["resource"]["peak_cuda_allocated_bytes"]) for item in group]
                    ),
                }
            )
            chart_runtime.append(
                {
                    "label": label,
                    "value": _median([float(item["resource"]["wall_seconds"]) for item in group]),
                }
            )
    artifacts = [
        {"label": "Representative initialization", "path": "gaussians_init.ply"},
        {"label": "Representative final model", "path": "gaussians.ply"},
        {"label": "Fitting history", "path": "training_history.json"},
        {"label": "Effective matrix configuration", "path": "gaussians.config.json"},
        {"label": "Input-boundary aggregate", "path": "input_boundary_receipt.json"},
        {"label": "Resource aggregate", "path": "resource_receipt.json"},
        {"label": "Run receipt", "path": "run_receipt.json"},
        {"label": "Execution environment", "path": "environment.json"},
        {"label": "All cell results", "path": "cell_results.json"},
        {"label": "Nine-method viewer manifest", "path": "viewer_comparison.json"},
        {"label": "Presentation receipt", "path": "presentation_receipt.json"},
        {"label": "Nine-method contact sheet", "path": "reconstruction_contact_sheet.png"},
        {"label": "Initial-to-final comparison", "path": "reconstruction.gif"},
        {"label": "Synchronized novel orbit", "path": "novel_orbit.gif"},
        {"label": "Synchronized novel elevation", "path": "novel_elevation.gif"},
    ]
    evidence = [
        {
            "label": "Result narrative",
            "path": f"benchmarks/results/{TASK_ID}_RESULT.md",
        },
        {
            "label": "Machine-readable result",
            "path": f"benchmarks/results/{TASK_ID}_RESULT.json",
        },
        {
            "label": "Independent audit",
            "path": f"benchmarks/results/{TASK_ID}_AUDIT.md",
        },
        {
            "label": "Machine-readable audit",
            "path": f"benchmarks/results/{TASK_ID}_AUDIT.json",
        },
    ]
    return {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": TASK_ID,
        "summary": (
            "Matched three-provider by three-initializer full-resolution development comparison "
            "on Stage frame 00008."
        ),
        "decision": (
            "Producer measurements are complete and remain uninterpreted until the independent "
            "results audit is attached."
        ),
        "claim_boundary": task["claim_boundary"],
        "metrics": {
            "heldout_sampled_j_area": _median(quality),
            "final_gaussians": _median(counts),
            "fullres_teacher_psnr": sum(teacher_values) / len(teacher_values),
            "peak_cuda_allocated_bytes": _median(peaks),
            "wall_seconds": _median(walls),
            "final_to_best_heldout_risk_ratio": _median(convergence_ratios),
            "converged_by_frozen_rule_fraction": sum(converged) / len(converged),
            "compact_to_model_compression_ratio": _median(compression),
        },
        "metric_metadata": {
            "heldout_sampled_j_area": {
                "label": "Held-out compact sampled area risk",
                "unit": "MSE",
                "group": "quality",
                "direction": "lower",
            },
            "final_gaussians": {
                "label": "Final Gaussian count",
                "unit": "gaussians",
                "group": "topology",
                "direction": "descriptive",
            },
            "fullres_teacher_psnr": {
                "label": "Stage-1 foreground fit PSNR",
                "unit": "dB",
                "group": "input QA",
                "direction": "higher",
            },
            "peak_cuda_allocated_bytes": {
                "label": "Peak CUDA allocated",
                "unit": "bytes",
                "group": "resources",
                "direction": "descriptive",
            },
            "wall_seconds": {
                "label": "Cell wall time",
                "unit": "seconds",
                "group": "runtime",
                "direction": "descriptive",
            },
            "final_to_best_heldout_risk_ratio": {
                "label": "Final / best held-out checkpoint risk",
                "unit": "ratio",
                "group": "convergence",
                "direction": "lower",
            },
            "converged_by_frozen_rule_fraction": {
                "label": "Cells converged by frozen rule",
                "unit": "fraction",
                "group": "convergence",
                "direction": "higher",
            },
            "compact_to_model_compression_ratio": {
                "label": "Compact-field bytes / final model NPZ bytes",
                "unit": "ratio",
                "group": "compression",
                "direction": "descriptive",
            },
        },
        "charts": [
            {
                "id": "quality",
                "title": "Held-out sampled risk",
                "unit": "MSE",
                "values": chart_quality,
            },
            {
                "id": "resources",
                "title": "Peak CUDA allocation",
                "unit": "bytes",
                "values": chart_resources,
            },
            {
                "id": "stage_runtime",
                "title": "End-to-end reconstruction wall time",
                "unit": "seconds",
                "values": chart_runtime,
            },
        ],
        "artifacts": artifacts,
        "evidence": evidence,
        "commands": {
            "reproduce": task["run_command"],
            "serve_report": [
                ".venv/bin/python",
                "-m",
                "http.server",
                "8765",
                "--directory",
                f"runs/{TASK_ID}",
            ],
            "viewer": [
                ".venv/bin/rtgs",
                "view",
                "--comparison-manifest",
                f"runs/{TASK_ID}/viewer_comparison.json",
                "--rasterizer",
                "gsplat",
                "--device",
                "cuda:0",
            ],
        },
        "notes": [
            "All reconstruction cells used calibration and sealed compact fields only; source "
            "RGB and masks were denied.",
            "GaussianImage additive and StructSplat normalized compositor semantics were "
            "preserved rather than homogenized.",
            "RNG seeds, camera schedules, sample counts, and sampling algorithms are matched; "
            "realized compact proposal and evaluation coordinates remain provider-conditioned "
            "because each provider has its own frozen fit windows and distributions.",
            "This is a single outcome-exposed development frame and does not authorize a "
            "general provider ranking.",
        ],
    }


def _publish_aggregate(task: Mapping[str, Any], run: Path) -> None:
    cells = _measured_cells(task, run)
    first = cells[0]["cell"]
    shutil.copy2(first / "gaussians_init.ply", run / "gaussians_init.ply")
    shutil.copy2(first / "gaussians.ply", run / "gaussians.ply")
    _write_json_new(run / "training_history.json", _history_bundle(task, run, cells))
    _write_json_new(
        run / "gaussians.config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "frozen_configuration": task["frozen_configuration"],
            "datasets": [item["id"] for item in task["datasets"]],
            "seeds": task["seeds"],
            "initializers": list(ARMS),
        },
    )
    cell_results = []
    boundary_cells = []
    resource_cells = []
    for item in cells:
        relative = item["cell"].relative_to(run).as_posix()
        boundary_path = item["cell"] / "input_boundary_receipt.json"
        resource_path = item["cell"] / "resource_receipt.json"
        cell_results.append(
            {
                "dataset_id": item["dataset"]["id"],
                "seed": item["seed"],
                "arm": item["arm"],
                "path": relative,
                "summary": item["summary"],
                "receipt": item["receipt"],
                "resource": item["resource"],
            }
        )
        boundary_cells.append(
            {
                "path": boundary_path.relative_to(run).as_posix(),
                "sha256": _sha256_file(boundary_path),
            }
        )
        resource_cells.append(
            {
                "path": resource_path.relative_to(run).as_posix(),
                "sha256": _sha256_file(resource_path),
            }
        )
    warmup_paths = sorted((run / "warmups").rglob("resource_receipt.json"))
    if len(warmup_paths) != int(task["resource_protocol"]["warmup_runs"]):
        raise RuntimeError("warmup resource receipt count differs from the frozen protocol")
    warmup_resources = [
        {"path": path.relative_to(run).as_posix(), "sha256": _sha256_file(path)}
        for path in warmup_paths
    ]
    _write_json_new(
        run / "cell_results.json",
        {"schema_version": 1, "task_id": TASK_ID, "cells": cell_results},
    )
    _write_json_new(
        run / "input_boundary_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "PASS",
            "allowed_modalities": ["calibration", "gaussians2d"],
            "source_rgb_or_mask_opened": False,
            "cell_receipts": boundary_cells,
        },
    )
    _write_json_new(
        run / "resource_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "protocol": task["resource_protocol"],
            "measured_runs_per_provider_initializer": len(task["seeds"]),
            "warmup_runs": len(warmup_resources),
            "measured_cell_count": len(resource_cells),
            "warmup_cell_count": len(warmup_resources),
            "aggregation": "paired median with all raw cell receipts retained",
            "cell_receipts": resource_cells,
            "warmup_cell_receipts": warmup_resources,
            "groups": [
                {
                    "dataset_id": dataset["id"],
                    "arm": arm,
                    "repeats": len(task["seeds"]),
                    "metrics": {
                        metric: {
                            "raw": values,
                            "min": min(values),
                            "median": _median(values),
                            "max": max(values),
                        }
                        for metric in (
                            "nvml_process_peak_bytes",
                            "peak_cuda_allocated_bytes",
                            "peak_cuda_reserved_bytes",
                            "background_device_memory_bytes",
                            "ru_maxrss_bytes",
                            "wall_seconds",
                            "compact_input_bytes",
                            "compact_field_bytes",
                            "final_model_npz_bytes",
                            "final_model_ply_bytes",
                            "compact_to_model_compression_ratio",
                            "output_bytes",
                        )
                        for values in [
                            [
                                float(item["resource"][metric])
                                for item in cells
                                if item["dataset"]["id"] == dataset["id"] and item["arm"] == arm
                            ]
                        ]
                    },
                }
                for dataset in task["datasets"]
                for arm in ARMS
            ],
        },
    )
    seed = int(task["seeds"][0])
    methods = []
    for dataset in task["datasets"]:
        for arm in ARMS:
            cell = run / _cell_relative(dataset["id"], seed, arm, warmup=False)
            methods.append(
                {
                    "name": f"{_provider_label(dataset['id'])} / {ARM_LABELS[arm]}",
                    "initial": (cell / "gaussians_init.ply").relative_to(run).as_posix(),
                    "final": (cell / "gaussians.ply").relative_to(run).as_posix(),
                }
            )
    _write_json_new(
        run / "viewer_comparison.json",
        {"schema": "rtgs.viewer-comparison.v1", "methods": methods},
    )
    _render_root_previews(task, run, methods)
    _write_json_new(run / "metrics.json", _metrics_bundle(task, run, cells))
    lock = _load_json(run / "task.lock.json")
    _write_json_new(
        run / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": 0,
            "failure_phase": None,
            "message": (
                "All frozen provider, initializer, and seed cells completed; producer outputs "
                "await independent audit."
            ),
        },
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
        "worker",
        "--task",
        task_path.relative_to(ROOT).as_posix(),
        "--run-dir",
        run.relative_to(ROOT).as_posix(),
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


def _execution_jobs(task: Mapping[str, Any]) -> list[tuple[str, int, str, bool]]:
    """Return one discarded global warmup and three measured repeats per matrix cell."""

    warmup_seed = max(0, min(int(seed) for seed in task["seeds"]) - 1)
    jobs = [(str(task["datasets"][0]["id"]), warmup_seed, ARMS[0], True)]
    for seed_index, seed in enumerate(task["seeds"]):
        rotation = seed_index % len(ARMS)
        arm_order = ARMS[rotation:] + ARMS[:rotation]
        dataset_order = list(task["datasets"])
        if seed_index % 2:
            dataset_order.reverse()
        for dataset in dataset_order:
            jobs.extend((str(dataset["id"]), int(seed), arm, False) for arm in arm_order)
    return jobs


def _failed_metrics_bundle(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": TASK_ID,
        "summary": "The protected producer stopped before a complete canonical result existed.",
        "decision": "Failed run: inspect diagnostics; do not use partial cells as results.",
        "claim_boundary": task["claim_boundary"],
        "metrics": {},
        "metric_metadata": {},
        "charts": [],
        "artifacts": [
            {"label": "Fitting history", "path": "training_history.json"},
            {"label": "Frozen configuration", "path": "gaussians.config.json"},
            {"label": "Input-boundary receipt", "path": "input_boundary_receipt.json"},
            {"label": "Resource receipt", "path": "resource_receipt.json"},
            {"label": "Run receipt", "path": "run_receipt.json"},
            {"label": "Environment", "path": "environment.json"},
            {"label": "Structured failure", "path": "failure.json"},
        ],
        "evidence": [],
        "commands": {
            "reproduce": task["run_command"],
            "serve_report": [
                ".venv/bin/python",
                "-m",
                "http.server",
                "8765",
                "--directory",
                f"runs/{TASK_ID}",
            ],
            "viewer": None,
        },
        "notes": [
            "This failed bundle is diagnostic and not results-bearing.",
            "Any completed cells remain raw failure context and authorize no comparison claim.",
        ],
    }


def _publish_failed_run(
    task: Mapping[str, Any],
    run: Path,
    *,
    phase: str,
    error: BaseException,
    job: tuple[str, int, str, bool] | None,
) -> None:
    if (run / "run_receipt.json").exists():
        raise FileExistsError("refusing to overwrite an existing run receipt") from error
    lock = _load_json(run / "task.lock.json")
    worker_failures = sorted(run.glob("failures/**/failure.json"))
    completed_boundaries = sorted(run.glob("cells/**/input_boundary_receipt.json"))
    completed_resources = sorted(run.glob("cells/**/resource_receipt.json"))
    _write_json_new(
        run / "failure.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "failed",
            "failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "failure_phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "job": (
                None
                if job is None
                else {
                    "dataset_id": job[0],
                    "seed": job[1],
                    "arm": job[2],
                    "warmup": job[3],
                }
            ),
            "worker_failures": [path.relative_to(run).as_posix() for path in worker_failures],
        },
    )
    if not (run / "training_history.json").exists():
        _write_json_new(
            run / "training_history.json",
            {"schema_version": 2, "records": [], "metric_metadata": {}, "stage_markers": []},
        )
    if not (run / "gaussians.config.json").exists():
        _write_json_new(
            run / "gaussians.config.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "frozen_configuration": task["frozen_configuration"],
                "datasets": [item["id"] for item in task["datasets"]],
                "seeds": task["seeds"],
                "initializers": list(ARMS),
            },
        )
    if not (run / "input_boundary_receipt.json").exists():
        _write_json_new(
            run / "input_boundary_receipt.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "status": "FAILED",
                "allowed_modalities": ["calibration", "gaussians2d"],
                "source_rgb_or_mask_opened": False,
                "completed_cell_receipts": [
                    path.relative_to(run).as_posix() for path in completed_boundaries
                ],
                "worker_failure_receipts": [
                    path.relative_to(run).as_posix() for path in worker_failures
                ],
            },
        )
    if not (run / "resource_receipt.json").exists():
        _write_json_new(
            run / "resource_receipt.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "status": "FAILED",
                "protocol": task["resource_protocol"],
                "completed_cell_receipts": [
                    path.relative_to(run).as_posix() for path in completed_resources
                ],
                "worker_failure_receipts": [
                    path.relative_to(run).as_posix() for path in worker_failures
                ],
            },
        )
    return_code = getattr(error, "returncode", 1)
    exit_code = int(return_code) if isinstance(return_code, int) and return_code != 0 else 1
    _write_json_new(
        run / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "failed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": exit_code,
            "failure_phase": phase,
            "message": f"Protected producer failed: {type(error).__name__}: {error}",
        },
    )
    if (run / "metrics.json").exists():
        os.replace(run / "metrics.json", run / "partial_metrics.json")
    _write_json_new(run / "metrics.json", _failed_metrics_bundle(task))
    rendered = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/experiment_contract.py"),
            "render",
            run.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if rendered.returncode != 0:
        print("failed-run report rendering also failed:\n" + rendered.stderr, file=sys.stderr)


def _orchestrate(task_path: Path, run: Path) -> int:
    task_path = _resolve_exact_path(task_path, DEFAULT_TASK, label="task")
    run = _resolve_exact_path(run, DEFAULT_RUN, label="run directory")
    task = _validate_run_binding(task_path, run)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/experiment_contract.py"),
            "validate-data",
            task_path.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        check=True,
    )
    forbidden = [
        run / name
        for name in (
            "cells",
            "warmups",
            "metrics.json",
            "training_history.json",
            "run_receipt.json",
            "environment.json",
        )
        if (run / name).exists()
    ]
    if forbidden:
        raise FileExistsError(
            "refusing to mix with existing canonical outputs: "
            + ", ".join(str(path) for path in forbidden)
        )
    _write_json_new(run / "environment.json", _environment_record())
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    jobs = _execution_jobs(task)
    failure_phase = "worker_matrix"
    current_job: tuple[str, int, str, bool] | None = None
    try:
        for index, current_job in enumerate(jobs, start=1):
            dataset_id, seed, arm, warmup = current_job
            kind = "warmup" if warmup else "measured"
            print(
                f"[{index}/{len(jobs)}] {kind} {dataset_id} seed={seed} initializer={arm}",
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
        current_job = None
        failure_phase = "aggregate_and_presentation"
        _publish_aggregate(task, run)
    except Exception as error:
        _publish_failed_run(
            task,
            run,
            phase=failure_phase,
            error=error,
            job=current_job,
        )
        raise
    print("Producer matrix complete; obtain independent audit before rendering.", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("initialize")
    train = subparsers.add_parser("train")
    for subparser in (initialize, train):
        subparser.add_argument("--task", type=Path, default=DEFAULT_TASK)
        subparser.add_argument("--dataset-id", default="frame_00008_gaussianimage")
        subparser.add_argument("--seed", type=int, default=801001)
        subparser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--stop-after-step", type=int)
    train.add_argument("--evaluation-samples", type=int, default=4096)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--task", type=Path, default=DEFAULT_TASK)
    preflight.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "preflight")

    run = subparsers.add_parser("run")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--task", type=Path, required=True)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--dataset-id", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--arm", choices=ARMS, required=True)
    worker.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "initialize":
        return _initialize(args)
    if args.command == "train":
        return _train(args)
    if args.command == "preflight":
        return _preflight(args.task, args.output)
    if args.command == "run":
        return _orchestrate(args.task, args.run_dir)
    if args.command == "worker":
        _canonical_worker(
            task_path=args.task,
            run=args.run_dir,
            dataset_id=args.dataset_id,
            seed=args.seed,
            arm=args.arm,
            warmup=args.warmup,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
