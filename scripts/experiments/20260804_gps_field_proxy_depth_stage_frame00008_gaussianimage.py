#!/usr/bin/env python3
"""Run the preregistered GPS-Gaussian field-proxy killing test.

The top-level invocation validates the protected development lock, runs one quality-free warmup
per arm, launches every measured cell in the frozen rotated order, and publishes common Bundle
Contract v2 sources.  Each worker is a fresh process whose NVML/torch peak scope begins before the
first compact archive or external GPS source is opened.  Source RGB, masks, packed alpha,
``SceneData``, and the dense image trainer are denied live.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260804_gps_field_proxy_depth_stage_frame00008_gaussianimage"
TASK_RELATIVE = Path("experiments/tasks") / f"{TASK_ID}.json"
RUN_RELATIVE = Path("runs") / TASK_ID
DEFAULT_TASK = ROOT / TASK_RELATIVE
DEFAULT_RUN = ROOT / RUN_RELATIVE
DEFAULT_GPS_REPO = ROOT / "external/GPS-Gaussian"
DEFAULT_GPS_CHECKPOINT = ROOT / "external/checkpoints/GPS-GS_stage2_state_dict.pt"
DATASET_ID = "frame_00008_gaussianimage"
WARMUP_SEED = 804000
ARMS = (
    "gps_field_proxy",
    "gps_shuffled_field",
    "pair_compact_carve_full",
    "compact_carve",
    "splat_sfm",
    "beam_fusion",
)
DETERMINISTIC_ARMS = (
    "pair_compact_carve_full",
    "compact_carve",
    "splat_sfm",
    "beam_fusion",
)
ARM_LABELS = {
    "gps_field_proxy": "GPS field proxy",
    "gps_shuffled_field": "GPS shuffled field",
    "pair_compact_carve_full": "Pair full-field compact carve",
    "compact_carve": "Compact carve",
    "splat_sfm": "Splat-SfM",
    "beam_fusion": "Beam Fusion",
}
IMAGE_SUFFIXES = frozenset({".bmp", ".exr", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
FORBIDDEN_MODULES = (
    "PIL",
    "cv2",
    "imageio",
    "rtgs.data.calibrated",
    "rtgs.data.scene",
    "rtgs.optim.trainer",
)
RESOURCE_MONITOR = {
    "device_index": 0,
    "monitor_interval_seconds": 0.05,
    "idle_samples": 3,
    "idle_sample_interval_seconds": 0.1,
    "idle_timeout_seconds": 300.0,
    "max_foreign_compute_processes": 0,
    "max_background_device_memory_bytes": 3 * 1024**3,
    "max_background_range_bytes": 128 * 1024**2,
    "max_gpu_utilization_percent": 50,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _protocol_sha256(task: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in task.items() if key not in {"protocol_review", "status"}}
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    _write_json_atomic(path, value)


class NoImageGuard:
    """Deny source-image access and image/dense-pipeline imports inside a worker."""

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
    def _forbidden_path(value: object) -> bool:
        if isinstance(value, int):
            return False
        try:
            return Path(os.fspath(value)).suffix.lower() in IMAGE_SUFFIXES
        except TypeError:
            return False

    @staticmethod
    def _forbidden_module(name: str) -> bool:
        return any(name == root or name.startswith(f"{root}.") for root in FORBIDDEN_MODULES)

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
        globals_value: Mapping[str, Any] | None = None,
        locals_value: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        resolved = name
        if level > 0 and globals_value is not None:
            package = globals_value.get("__package__")
            if isinstance(package, str) and package:
                with contextlib.suppress(ImportError, ValueError):
                    resolved = importlib.util.resolve_name("." * level + name, package)
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
        return self._import(name, globals_value, locals_value, fromlist, level)

    def _guarded_import_module(self, name: str, package: str | None = None) -> Any:
        try:
            resolved = importlib.util.resolve_name(name, package) if name.startswith(".") else name
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
        builtins.open = self._guarded_open
        io.open = self._guarded_io_open
        os.open = self._guarded_os_open
        builtins.__import__ = self._guarded_import
        importlib.import_module = self._guarded_import_module
        self._probing = True

        def probe_builtin_open() -> None:
            with builtins.open(ROOT / "guard-negative.jpg", "rb"):
                pass

        def probe_io_open() -> None:
            with io.open(ROOT / "guard-negative.png", "rb"):  # noqa: UP020
                pass

        probes = (
            probe_builtin_open,
            probe_io_open,
            lambda: os.open(ROOT / "guard-negative.webp", os.O_RDONLY),
            lambda: importlib.import_module("PIL.Image"),
        )
        for probe in probes:
            try:
                probe()
            except (ImportError, PermissionError):
                pass
            else:  # pragma: no cover - guard invariant
                raise RuntimeError("input-boundary negative control unexpectedly succeeded")
        self._probing = False
        if self.negative_control_denials != len(probes):
            raise RuntimeError("input-boundary negative controls were not all denied")
        return self

    def __exit__(self, *_args: object) -> None:
        builtins.open = self._open
        io.open = self._io_open
        os.open = self._os_open
        builtins.__import__ = self._import
        importlib.import_module = self._import_module

    def record(self) -> dict[str, Any]:
        forbidden_loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        return {
            "schema": "rtgs.no_image_guard.v1",
            "passed": (
                not forbidden_loaded
                and self.denied_paths == 0
                and self.denied_imports == 0
                and self.negative_control_denials == 4
            ),
            "negative_control_expected": 4,
            "negative_control_denials": self.negative_control_denials,
            "unexpected_denied_paths": self.denied_paths,
            "unexpected_denied_imports": self.denied_imports,
            "forbidden_modules_loaded": forbidden_loaded,
            "source_rgb_or_mask_opened": False,
        }


class NvmlProcessSampler:
    """Sample this worker's process allocation every 50 ms after a quiescence guard."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.device_index = int(config["device_index"])
        self.interval = float(config["monitor_interval_seconds"])
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any = None
        self._handle: Any = None
        self._error: BaseException | None = None
        self._samples = 0
        self._own_peak = 0
        self._background_peak = 0
        self._background_start = 0
        self._utilization_peak = 0
        self._foreign: dict[int, int] = {}
        self._device_total = 0
        self._driver = ""
        self._idle: dict[str, Any] | None = None

    def _snapshot(self) -> dict[str, Any]:
        memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        unavailable = getattr(self._pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
        own = 0
        foreign = []
        for process in self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
            raw = getattr(process, "usedGpuMemory", 0)
            used = 0 if raw in {None, unavailable} or int(raw) < 0 else int(raw)
            if int(process.pid) == os.getpid():
                own = max(own, used)
            else:
                foreign.append({"pid": int(process.pid), "used_bytes": used})
        return {
            "process_used_bytes": own,
            "background_device_memory_bytes": max(0, int(memory.used) - own),
            "device_total_bytes": int(memory.total),
            "gpu_utilization_percent": int(utilization.gpu),
            "foreign_compute_processes": foreign,
        }

    def _sample_once(self) -> dict[str, Any]:
        sample = self._snapshot()
        self._own_peak = max(self._own_peak, sample["process_used_bytes"])
        self._background_peak = max(self._background_peak, sample["background_device_memory_bytes"])
        self._utilization_peak = max(self._utilization_peak, sample["gpu_utilization_percent"])
        for process in sample["foreign_compute_processes"]:
            pid = process["pid"]
            self._foreign[pid] = max(self._foreign.get(pid, 0), process["used_bytes"])
        self._samples += 1
        return sample

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
                accepted = accepted[-required:]
                background = [item["background_device_memory_bytes"] for item in accepted]
                if len(accepted) == required and max(background) - min(background) <= int(
                    self.config["max_background_range_bytes"]
                ):
                    return {
                        "passed": True,
                        "observed_samples": observed,
                        "accepted_samples": accepted,
                        "limits": dict(self.config),
                    }
            else:
                accepted.clear()
            time.sleep(float(self.config["idle_sample_interval_seconds"]))
        raise RuntimeError("GPU quiescence guard timed out")

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample_once()
            except BaseException as error:  # pragma: no cover - hardware dependent
                self._error = error
                self._stop.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("NVML sampler already started")
        import pynvml

        pynvml.nvmlInit()
        try:
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._device_total = int(memory.total)
            driver = pynvml.nvmlSystemGetDriverVersion()
            self._driver = driver.decode() if isinstance(driver, bytes) else str(driver)
            self._idle = self._wait_for_idle()
            first = self._sample_once()
            self._background_start = first["background_device_memory_bytes"]
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except BaseException:
            pynvml.nvmlShutdown()
            raise

    def stop(self, *, require_clean: bool = True) -> dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("NVML sampler was not started")
        self._stop.set()
        self._thread.join(timeout=max(1.0, 4 * self.interval))
        if self._thread.is_alive():
            raise RuntimeError("NVML sampler did not stop")
        try:
            if self._error is not None:
                raise RuntimeError("NVML sampling failed") from self._error
            self._sample_once()
            if require_clean and self._foreign:
                raise RuntimeError("foreign CUDA compute process entered the measured scope")
            return {
                "nvml_process_peak_bytes": self._own_peak,
                "background_device_memory_bytes": self._background_start,
                "background_device_memory_peak_bytes": self._background_peak,
                "device_total_bytes": self._device_total,
                "driver_version": self._driver,
                "nvml_sampling_interval_seconds": self.interval,
                "nvml_samples": self._samples,
                "gpu_utilization_peak_percent": self._utilization_peak,
                "foreign_compute_processes": [
                    {"pid": pid, "used_bytes": used} for pid, used in sorted(self._foreign.items())
                ],
                "idle_guard": self._idle,
            }
        finally:
            self._pynvml.nvmlShutdown()
            self._thread = None


def _resolve_exact(value: Path, expected: Path, *, label: str) -> Path:
    actual = value.resolve()
    if actual != expected.resolve():
        raise ValueError(f"{label} must be exactly {expected}")
    return actual


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout


def _validate_run_binding(task_path: Path, run: Path) -> dict[str, Any]:
    task_path = _resolve_exact(task_path, DEFAULT_TASK, label="task path")
    run = _resolve_exact(run, DEFAULT_RUN, label="run path")
    task = _load_json(task_path)
    if task.get("task_id") != TASK_ID or task.get("status") != "ready":
        raise ValueError("driver requires the matching ready task")
    review = task.get("protocol_review")
    if not isinstance(review, dict) or review.get("verdict") != "approved":
        raise ValueError("driver requires approved prospective protocol review")
    lock_path = run / "task.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("initialize the run through scripts/experiment_contract.py")
    lock = _load_json(lock_path)
    review_path = ROOT / str(review["artifact"])
    seal_path = ROOT / str(task["data_seal"])
    diff = _git("diff", "--binary", "HEAD").encode()
    source_dirty = bool(_git("status", "--porcelain=v1"))
    checks = {
        "task_id": lock.get("task_id") == TASK_ID,
        "task_path": lock.get("task_path") == TASK_RELATIVE.as_posix(),
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
        "development": lock.get("development") is True,
        "source_dirty": lock.get("source_dirty") is source_dirty,
        "source_diff_sha256": lock.get("source_diff_sha256") == hashlib.sha256(diff).hexdigest(),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError("run/task lock binding failed: " + ", ".join(failed))
    return task


def _dataset(task: Mapping[str, Any]) -> dict[str, Any]:
    matches = [item for item in task["datasets"] if item["id"] == DATASET_ID]
    if len(matches) != 1:
        raise ValueError("task must contain the one frozen dataset")
    return dict(matches[0])


def _runtime_versions(task: Mapping[str, Any]) -> dict[str, str]:
    import pynvml  # noqa: F401
    import scipy

    expected = task["frozen_configuration"]["runtime_dependencies"]
    observed = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "scipy": scipy.__version__,
        "nvidia-ml-py": importlib.metadata.version("nvidia-ml-py"),
    }
    if observed != {key: expected[key] for key in observed}:
        raise RuntimeError(f"frozen runtime dependency mismatch: {observed!r}")
    return observed


def _seed_everything(seed: int, device: str) -> dict[str, Any]:
    import torch

    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError("worker PYTHONHASHSEED was not set before interpreter startup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("worker CUBLAS_WORKSPACE_CONFIG is not frozen")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if str(torch.device(device)) != "cuda:0":
        raise ValueError("frozen experiment requires cuda:0")
    return {
        "seed": seed,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32": False,
        "deterministic_algorithms": True,
    }


def _compact_directory(task: Mapping[str, Any]) -> Path:
    return (ROOT / str(_dataset(task)["compact_manifest"])).parent


def _load_views(task: Mapping[str, Any], names: Sequence[str]) -> Any:
    from rtgs.data.compact_views import CompactDataset

    config = task["frozen_configuration"]["compact_loading"]
    if config != {
        "byte_cap_per_view": 8388608,
        "load_alpha": False,
        "device": "cpu",
        "policy": config["policy"],
    }:
        raise ValueError("frozen compact loading controls changed")
    compact = CompactDataset.load(
        _compact_directory(task),
        device="cpu",
        byte_cap=8_388_608,
        load_alpha=False,
        view_ids=list(names),
    )
    if [view.view_id for view in compact.views] != list(names):
        raise RuntimeError("selected compact view order changed")
    if any(view.alpha is not None for view in compact.views):
        raise RuntimeError("packed alpha was materialized")
    return compact.to_reconstruction_inputs()


def _subset_inputs(inputs: Any, names: Sequence[str]) -> Any:
    from rtgs.data.reconstruction_inputs import ReconstructionInputs

    lookup = {name: index for index, name in enumerate(inputs.view_names)}
    if len(lookup) != inputs.n_views or any(name not in lookup for name in names):
        raise ValueError("requested input subset is missing or duplicated")
    indices = [lookup[name] for name in names]
    return ReconstructionInputs(
        observations=[inputs.observations[index] for index in indices],
        cameras=[inputs.cameras[index] for index in indices],
        view_names=list(names),
        bounds_hint=inputs.bounds_hint,
        name=f"{inputs.name}-{'-'.join(names)}",
    )


def _field_semantics(inputs: Any) -> list[dict[str, Any]]:
    return [
        {
            "view_id": name,
            "provider": field.provider,
            "blend_mode": field.blend_mode,
            "canvas": [field.width, field.height],
            "fit_window": list(field.fit_window),
            "n_gaussians": field.n,
            "dtype": str(field.dtype),
            "device": str(field.device),
        }
        for name, field in zip(inputs.view_names, inputs.observations, strict=True)
    ]


def _dataclass_kwargs(cls: type, source: Mapping[str, Any], *, extras: set[str]) -> dict[str, Any]:
    from dataclasses import fields

    names = {item.name for item in fields(cls)}
    if set(source) != names | extras:
        raise ValueError(
            f"frozen {cls.__name__} keys changed "
            f"(missing={sorted(names - set(source))}, extra={sorted(set(source) - names - extras)})"
        )
    return {name: source[name] for name in names}


def _gps_initializer_config(task: Mapping[str, Any], *, shuffled: bool) -> Any:
    from rtgs.lift.gps_field import FieldProxyConfig, GPSFieldInitializerConfig

    frozen = task["frozen_configuration"]
    proxy_source = frozen["dense_proxy"]
    proxy = FieldProxyConfig(
        resolution=int(proxy_source["resolution"]),
        row_batch=int(proxy_source["row_batch"]),
        tile_size=int(proxy_source["tile_size"]),
        support_threshold=float(proxy_source["support_threshold"]),
        max_index_entries=16_000_000,
        max_candidates_per_tile=200_000,
        max_query_pairs=1_048_576,
    )
    config = frozen["initializer"]
    pair = frozen["pair_selection"]
    return GPSFieldInitializerConfig(
        n_init_3d=int(config["count"]),
        left_view=pair["selected_pair"][0],
        right_view=pair["selected_pair"][1],
        proxy_right_view=(
            pair["shuffled_proxy_right_view"] if shuffled else pair["selected_pair"][1]
        ),
        near=float(config["near"]),
        bounds_scale=float(config["bounds_scale"]),
        minimum_confidence=float(config["minimum_confidence"]),
        maximum_cycle_error_px=float(config["maximum_left_right_consistency_px"]),
        confidence_decay_tau_px=float(config["confidence_decay_tau_px"]),
        disparity_noise_floor_px=float(config["disparity_noise_floor_px"]),
        minimum_axial_sigma_fraction=float(config["minimum_axial_sigma_fraction"]),
        maximum_axial_sigma_fraction=float(config["maximum_axial_sigma_fraction"]),
        minimum_valid_candidate_fraction=float(
            frozen["decision_rule"]["minimum_valid_candidate_fraction"]
        ),
        voxel_size_extent_fraction=float(config["voxel_size_extent_fraction"]),
        color_bin_size=float(config["color_bin_size"]),
        init_opacity=float(config["init_opacity"]),
        sh_degree=int(config["sh_degree"]),
        proxy=proxy,
    )


def _construct_initialization(
    task: Mapping[str, Any],
    train: Any,
    *,
    arm: str,
    seed: int,
    gps_repo: Path,
    gps_checkpoint: Path,
    device: str,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None, Any | None]:
    """Return model, initializer receipt, external receipt, and optional GPS artifacts."""

    if arm in {"gps_field_proxy", "gps_shuffled_field"}:
        from rtgs.depth.gps_gaussian import GPSGaussianConfig, GPSGaussianStereoBackend
        from rtgs.lift.gps_field import GPSFieldProxyInitializer

        backend = GPSGaussianStereoBackend(
            GPSGaussianConfig(repository=gps_repo, checkpoint=gps_checkpoint, device=device)
        )
        initializer = GPSFieldProxyInitializer(
            backend,
            _gps_initializer_config(task, shuffled=arm == "gps_shuffled_field"),
            device=device,
        )
        result, artifacts = initializer.initialize_with_artifacts(train)
        return (
            result.gaussians,
            {
                "schema": "rtgs.gps_field_initialization_cell.v1",
                "diagnostics": result.diagnostics,
                "lineage": {
                    "source_view_indices": result.lineage.source_view_indices,
                    "source_component_indices": result.lineage.source_component_indices,
                    "source_xy": result.lineage.source_xy,
                    "depths": result.depths,
                    "depth_sigmas": result.depth_sigmas,
                    "ray_sigmas": result.ray_sigmas,
                    "scores": result.scores,
                },
            },
            backend.receipt,
            artifacts,
        )

    structural = task["frozen_configuration"]["structural_baselines"]
    if arm in {"pair_compact_carve_full", "compact_carve"}:
        from rtgs.lift.compact_carve import (
            CompactCarveConfig,
            CompactCarveInitializer,
            make_placement_progress_printer,
        )

        source = dict(structural[arm])
        extras = (
            {"views", "source_candidate_policy", "selection", "purpose"}
            if arm == "pair_compact_carve_full"
            else set()
        )
        kwargs = _dataclass_kwargs(CompactCarveConfig, source, extras=extras)
        kwargs["seed"] = seed
        config = CompactCarveConfig(**kwargs)
        inputs = (
            _subset_inputs(train, source["views"]) if arm == "pair_compact_carve_full" else train
        )
        result = CompactCarveInitializer(config).initialize(
            inputs,
            progress_callback=make_placement_progress_printer(every_batches=10, every_seconds=30),
        )
        return (
            result.gaussians,
            {
                "schema": "rtgs.compact_carve_initialization_cell.v1",
                "diagnostics": result.diagnostics,
                "lineage": {
                    "source_view_indices": result.lineage.source_view_indices,
                    "source_component_indices": result.lineage.source_component_indices,
                    "source_xy": result.lineage.source_xy,
                    "depths": result.depths,
                    "depth_sigmas": result.depth_sigmas,
                    "ray_sigmas": result.ray_sigmas,
                    "scores": result.scores,
                },
            },
            None,
            None,
        )

    from rtgs.lift.beam_fusion import BeamFusionConfig
    from rtgs.lift.paper_initializers import (
        PaperInitializerConfig,
        build_frozen_paper_initialization,
    )
    from rtgs.lift.splat_sfm import SplatSfMConfig

    shared = structural["paper_initializer_shared"]
    paper = PaperInitializerConfig(
        random_seed=seed,
        random_bounds_scale=float(shared["random_bounds_scale"]),
        init_opacity=float(shared["init_opacity"]),
        max_starting_gaussians=int(shared["max_starting_gaussians"]),
        structural_components_per_view=int(structural["components_per_view"]),
        sfm=SplatSfMConfig(**structural["splat_sfm"]),
        beam=BeamFusionConfig(**structural["beam_fusion"]),
    )
    result = build_frozen_paper_initialization(
        train,
        paper,
        arm=arm,
        count=int(shared["exact_count"]),
    )
    return (
        result.gaussians,
        {
            "schema": "rtgs.paper_initialization_cell.v1",
            "diagnostics": result.receipt,
            "lineage": result.lineage,
        },
        None,
        None,
    )


def _query_indexes(inputs: Any, task: Mapping[str, Any], device: str) -> list[Any]:
    from rtgs.core.observation2d_cuda import GaussianObservationIndexCuda

    config = task["frozen_configuration"]["compact_refinement"]
    structural = task["frozen_configuration"]["structural_baselines"]
    max_query_pairs = int(structural["compact_carve"]["max_query_pairs"])
    if max_query_pairs != int(structural["pair_compact_carve_full"]["max_query_pairs"]):
        raise ValueError("frozen query-pair caps differ between controls")
    return [
        GaussianObservationIndexCuda.from_field(
            field,
            tile_size=int(config["teacher_tile_size"]),
            max_entries=int(config["max_index_entries_per_view"]),
            max_candidates=int(config["max_candidates_per_tile"]),
            max_query_pairs=max_query_pairs,
            device=device,
        )
        for field in inputs.observations
    ]


def _point_renderer(task: Mapping[str, Any]) -> Any:
    from rtgs.render.gsplat_points import GsplatPointRasterizer

    config = task["frozen_configuration"]["compact_refinement"]
    if config["point_renderer"] != (
        "GsplatPointRasterizer(absgrad=false,antialiased=true,sh_color_activation='hard',"
        "sh_smu1_mu=2/255,kernel_support_mode='hard',visibility_margin_sigma=3.0) for both "
        "training and held-out evaluation"
    ):
        raise ValueError("frozen point-renderer description changed")
    return GsplatPointRasterizer(
        absgrad=False,
        antialiased=True,
        sh_color_activation=str(config["sh_color_activation"]),
        sh_smu1_mu=float(config["sh_smu1_mu"]),
        kernel_support_mode=str(config["kernel_support_mode"]),
        visibility_margin_sigma=float(config["visibility_margin_sigma"]),
    )


def _midpoint_coordinates(field: Any, samples_per_axis: int, *, device: str) -> Any:
    import torch

    fit_x, fit_y, fit_width, fit_height = field.fit_window
    axis = torch.arange(samples_per_axis, dtype=torch.float32) + 0.5
    x = fit_x + axis * (fit_width / samples_per_axis)
    y = fit_y + axis * (fit_height / samples_per_axis)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).to(device)


def _heldout_evaluation(task: Mapping[str, Any], model: Any, *, device: str) -> dict[str, Any]:
    """Load held-out fields only for this call and evaluate the fixed midpoint lattice."""

    import torch

    split = task["splits"][DATASET_ID]
    heldout_cpu = _load_views(task, split["heldout"])
    heldout = heldout_cpu.to(device)
    indexes = _query_indexes(heldout, task, device)
    renderer = _point_renderer(task)
    model_gpu = model.to(device)
    evaluation = task["frozen_configuration"]["heldout_evaluation"]
    side = int(evaluation["samples_per_axis"])
    samples_per_view = int(evaluation["samples_per_view"])
    chunk_size = int(task["frozen_configuration"]["compact_refinement"]["evaluation_chunk"])
    if samples_per_view != side**2:
        raise ValueError("held-out lattice cardinality changed")
    per_view = []
    with torch.inference_mode():
        for name, field, camera, index in zip(
            heldout.view_names,
            heldout.observations,
            heldout.cameras,
            indexes,
            strict=True,
        ):
            xy = _midpoint_coordinates(field, side, device=device)
            digest = hashlib.sha256(xy.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            squared_error_sum = 0.0
            scalar_count = 0
            for start in range(0, xy.shape[0], chunk_size):
                points = xy[start : start + chunk_size]
                target = index.query(points).color
                predicted = renderer.render_points(
                    model_gpu,
                    camera,
                    points,
                    background=torch.zeros(3, device=device),
                    sh_degree=0,
                ).color
                if not bool(torch.isfinite(target).all() and torch.isfinite(predicted).all()):
                    raise RuntimeError("held-out evaluation produced non-finite color")
                squared_error_sum += float((predicted - target).square().double().sum())
                scalar_count += int(target.numel())
            per_view.append(
                {
                    "view_id": name,
                    "samples": samples_per_view,
                    "xy_sha256": digest,
                    "fit_window": list(field.fit_window),
                    "j_area": squared_error_sum / scalar_count,
                }
            )
    coordinate_hashes = [item["xy_sha256"] for item in per_view]
    result = {
        "schema": "rtgs.fixed_midpoint_area_evaluation.v1",
        "samples_per_axis": side,
        "samples_per_view": samples_per_view,
        "equal_view_j_area": sum(item["j_area"] for item in per_view) / len(per_view),
        "coordinate_hashes": coordinate_hashes,
        "per_view": per_view,
    }
    del indexes, heldout, heldout_cpu, renderer, model_gpu
    torch.cuda.synchronize(torch.device(device))
    torch.cuda.empty_cache()
    return result


def _train_config(task: Mapping[str, Any], *, seed: int, extent: float, device: str) -> Any:
    from rtgs.optim.compact_trainer import CompactTrainConfig

    source = task["frozen_configuration"]["compact_refinement"]
    extras = {
        "fixed_topology",
        "heldout_checkpoint_selection",
        "point_renderer",
        "query_backends",
    }
    kwargs = _dataclass_kwargs(CompactTrainConfig, source, extras=extras)
    kwargs.update(seed=seed, extent=extent, device=device, checkpoints=tuple(source["checkpoints"]))
    config = CompactTrainConfig(**kwargs)
    if source["fixed_topology"] is not True or source["heldout_checkpoint_selection"] is not False:
        raise ValueError("frozen topology/held-out checkpoint policy changed")
    return config


def _refine(
    task: Mapping[str, Any],
    train_cpu: Any,
    init: Any,
    *,
    seed: int,
    device: str,
    stop_after_step: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    from rtgs.lift.compact_carve import _center_and_extent
    from rtgs.optim.compact_trainer import CompactTrainer

    train = train_cpu.to(device)
    indexes = _query_indexes(train, task, device)
    _center, extent = _center_and_extent(train_cpu, torch.float64)
    trainer = CompactTrainer(
        _train_config(task, seed=seed, extent=extent, device=device),
        point_rasterizer=_point_renderer(task),
    )
    final, history = trainer.train(
        train,
        init,
        query_backends=indexes,
        proposal_query_backends=indexes,
        stop_after_step=stop_after_step,
    )
    if final.n != init.n:
        raise RuntimeError("fixed-topology refinement changed Gaussian cardinality")
    torch.cuda.synchronize(torch.device(device))
    del indexes, train, trainer
    torch.cuda.empty_cache()
    return final.detach().to("cpu"), history


def _jsonable(value: object) -> object:
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _save_lineage(path: Path, lineage: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    arrays = {}
    for name, value in lineage.items():
        if isinstance(value, torch.Tensor):
            arrays[name] = value.detach().cpu().numpy()
    if arrays:
        np.savez_compressed(path, **arrays)


def _save_gps_preview(path: Path, artifacts: Any) -> None:
    import numpy as np
    import torch.nn.functional as F

    values = {
        "left_proxy_rgb": artifacts.left_proxy_rgb,
        "right_proxy_rgb": artifacts.right_proxy_rgb,
        "left_rectified_rgb": artifacts.left_rectified_rgb,
        "right_rectified_rgb": artifacts.right_rectified_rgb,
        "left_confidence": artifacts.prediction.left_confidence[None],
        "right_confidence": artifacts.prediction.right_confidence[None],
        "left_inverse_depth": artifacts.prediction.left_inverse_depth[None],
        "right_inverse_depth": artifacts.prediction.right_inverse_depth[None],
    }
    downsampled = {
        name: F.interpolate(value[None].float(), size=(128, 128), mode="area")[0].numpy()
        for name, value in values.items()
    }
    np.savez_compressed(path, **downsampled)


def _sealed_input_records(
    task: Mapping[str, Any],
    *,
    warmup: bool,
    external_receipt: Mapping[str, Any] | None,
    gps_checkpoint: Path,
) -> list[dict[str, Any]]:
    dataset = _dataset(task)
    seal_path = ROOT / str(task["data_seal"])
    seal = _load_json(seal_path)
    by_path = {item["path"]: item for item in seal["files"]}
    split = task["splits"][DATASET_ID]
    opened_views = list(split["train"]) + ([] if warmup else list(split["heldout"]))
    compact_dir = Path(str(dataset["compact_manifest"])).parent
    relative_paths = [
        TASK_RELATIVE.as_posix(),
        str(task["data_seal"]),
        str(dataset["calibration"]),
        str(dataset["compact_manifest"]),
        str(dataset["production_manifest"]),
        *(f"{compact_dir.as_posix()}/{name}.rtgsv" for name in opened_views),
    ]
    records = []
    for relative in relative_paths:
        path = ROOT / relative
        record = {
            "category": (
                "task"
                if relative == TASK_RELATIVE.as_posix()
                else "data_seal"
                if relative == task["data_seal"]
                else "calibration"
                if relative == dataset["calibration"]
                else "compact_manifest"
                if relative == dataset["compact_manifest"]
                else "production_manifest"
                if relative == dataset["production_manifest"]
                else "compact_view"
            ),
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        sealed = by_path.get(relative)
        if sealed is not None and (
            sealed["bytes"] != record["bytes"] or sealed["sha256"] != record["sha256"]
        ):
            raise RuntimeError(f"sealed input drift: {relative}")
        records.append(record)
    if external_receipt is not None:
        checkpoint = gps_checkpoint.resolve()
        records.append(
            {
                "category": "gps_sanitized_checkpoint",
                "path": checkpoint.relative_to(ROOT).as_posix(),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256_file(checkpoint),
            }
        )
        for source in external_receipt["imported_sources"]:
            if source["source_sha256"] is None:
                continue
            source_path = Path(source["origins"][0])
            records.append(
                {
                    "category": "gps_external_python_source",
                    "path": source_path.relative_to(ROOT).as_posix(),
                    "bytes": int(source["source_bytes"]),
                    "sha256": source["source_sha256"],
                }
            )
    identities = [record["path"] for record in records]
    if len(identities) != len(set(identities)):
        raise RuntimeError("input byte accounting contains overlapping paths")
    return records


def _output_records(directory: Path) -> list[dict[str, Any]]:
    excluded = {"resource_receipt.json"}
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
        if path.name not in excluded
    ]


def _cell_relative(seed: int, arm: str, *, warmup: bool) -> Path:
    return Path("warmups" if warmup else "cells") / f"seed_{seed}" / arm


def _failure_relative(seed: int, arm: str, *, warmup: bool) -> Path:
    return Path("failures") / ("warmups" if warmup else "cells") / f"seed_{seed}" / arm


def _stage_start(trace: dict[str, dict[str, float]], stage: str, origin: float) -> float:
    now = time.perf_counter() - origin
    trace[stage] = {"start": now, "end": now}
    return now


def _stage_end(trace: dict[str, dict[str, float]], stage: str, origin: float) -> float:
    now = time.perf_counter() - origin
    trace[stage]["end"] = now
    return now


def _run_worker(
    *,
    task_path: Path,
    run: Path,
    gps_repo: Path,
    gps_checkpoint: Path,
    device: str,
    seed: int,
    arm: str,
    warmup: bool,
) -> int:
    """Execute one fresh resource-scoped arm/seed cell."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    target = run / _cell_relative(seed, arm, warmup=warmup)
    failure_target = run / _failure_relative(seed, arm, warmup=warmup)
    if target.exists() or failure_target.exists():
        raise FileExistsError(f"refusing to overwrite cell {seed}/{arm}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{arm}.", dir=target.parent))
    sampler = NvmlProcessSampler(RESOURCE_MONITOR)
    sampler_started = False
    torch_module: Any = None
    origin = time.perf_counter()
    stage_trace: dict[str, dict[str, float]] = {}
    phase = "resource_scope_start"
    external_receipt: dict[str, Any] | None = None
    guard_record: dict[str, Any] | None = None
    guard: NoImageGuard | None = None
    try:
        sampler.start()
        sampler_started = True
        import torch

        torch_module = torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        task = _validate_run_binding(task_path, run)
        if seed != WARMUP_SEED and seed not in task["seeds"]:
            raise ValueError("worker seed is not frozen")
        if warmup != (seed == WARMUP_SEED):
            raise ValueError("warmup flag and seed disagree")
        gps_repo = _resolve_exact(gps_repo, DEFAULT_GPS_REPO, label="GPS repository")
        gps_checkpoint = _resolve_exact(
            gps_checkpoint, DEFAULT_GPS_CHECKPOINT, label="GPS checkpoint"
        )
        versions = _runtime_versions(task)
        determinism = _seed_everything(seed, device)
        guard = NoImageGuard()
        with guard:
            phase = "load_compact_fields"
            _stage_start(stage_trace, phase, origin)
            split = task["splits"][DATASET_ID]
            train = _load_views(task, split["train"])
            semantics = _field_semantics(train)
            if set(train.view_names) & set(split["heldout"]):
                raise RuntimeError("held-out view entered train inputs")
            _stage_end(stage_trace, phase, origin)

            phase = "construct_initialization"
            _stage_start(stage_trace, phase, origin)
            init, initializer_receipt, external_receipt, gps_artifacts = _construct_initialization(
                task,
                train,
                arm=arm,
                seed=seed,
                gps_repo=gps_repo,
                gps_checkpoint=gps_checkpoint,
                device=device,
            )
            if init.n != 3000:
                raise RuntimeError("initializer did not return exactly 3,000 Gaussians")
            _stage_end(stage_trace, phase, origin)

            if warmup:
                phase = "matched_compact_refinement"
                _stage_start(stage_trace, phase, origin)
                _warmup_final, _warmup_history = _refine(
                    task,
                    train,
                    init,
                    seed=seed,
                    device=device,
                    stop_after_step=1,
                )
                _stage_end(stage_trace, phase, origin)
                del _warmup_final, _warmup_history, init, initializer_receipt, gps_artifacts
                training_history = None
                initial_evaluation = None
                final_evaluation = None
                final = None
            else:
                init.save_npz(temporary / "gaussians_init.npz")
                init.save_ply(temporary / "gaussians_init.ply")
                lineage = initializer_receipt.pop("lineage")
                _save_lineage(temporary / "initializer_lineage.npz", lineage)
                if gps_artifacts is not None:
                    _save_gps_preview(temporary / "gps_proxy_preview_128.npz", gps_artifacts)
                del gps_artifacts

                phase = "initial_heldout_evaluation"
                _stage_start(stage_trace, phase, origin)
                initial_evaluation = _heldout_evaluation(task, init, device=device)
                _stage_end(stage_trace, phase, origin)

                phase = "matched_compact_refinement"
                _stage_start(stage_trace, phase, origin)
                final, training_history = _refine(
                    task,
                    train,
                    init,
                    seed=seed,
                    device=device,
                )
                _stage_end(stage_trace, phase, origin)
                if final.n != 3000:
                    raise RuntimeError("refinement did not preserve exactly 3,000 Gaussians")

                phase = "final_heldout_evaluation"
                _stage_start(stage_trace, phase, origin)
                final_evaluation = _heldout_evaluation(task, final, device=device)
                _stage_end(stage_trace, phase, origin)
                if initial_evaluation["coordinate_hashes"] != final_evaluation["coordinate_hashes"]:
                    raise RuntimeError("initial/final held-out coordinates differ")

                phase = "save_and_report"
                _stage_start(stage_trace, phase, origin)
                final.save_npz(temporary / "gaussians.npz")
                final.save_ply(temporary / "gaussians.ply")
                _write_json_new(temporary / "compact_training_history.json", training_history)
                _write_json_new(
                    temporary / "initializer_receipt.json", _jsonable(initializer_receipt)
                )

            input_records = _sealed_input_records(
                task,
                warmup=warmup,
                external_receipt=external_receipt,
                gps_checkpoint=gps_checkpoint,
            )
            guard_record = guard.record()
            if not guard_record["passed"]:
                raise RuntimeError(f"live input boundary failed: {guard_record!r}")
            cell_receipt = {
                "schema": "rtgs.gps_field_proxy_cell.v1",
                "task_id": TASK_ID,
                "dataset_id": DATASET_ID,
                "seed": seed,
                "arm": arm,
                "warmup": warmup,
                "status": "completed",
                "stages_completed": list(stage_trace),
                "stage_trace": stage_trace,
                "heldout_loaded": not warmup,
                "quality_metrics_written": not warmup,
                "train_views": train.view_names,
                "heldout_views": [] if warmup else list(split["heldout"]),
                "field_semantics": semantics,
                "determinism": determinism,
                "runtime_versions": versions,
                "initial_gaussians": 3000,
                "final_gaussians": None if warmup else int(final.n),
                "initial_evaluation": initial_evaluation,
                "final_evaluation": final_evaluation,
                "initializer": None if warmup else _jsonable(initializer_receipt),
                "external_gps": None if external_receipt is None else _jsonable(external_receipt),
                "input_boundary": guard_record,
                "input_records": input_records,
                "input_bytes": sum(int(item["bytes"]) for item in input_records),
            }
            if not warmup:
                _write_json_new(temporary / "cell_receipt.json", cell_receipt)
            else:
                _write_json_new(temporary / "warmup_receipt.json", cell_receipt)
            phase = "save_and_report"
            if phase in stage_trace:
                _stage_end(stage_trace, phase, origin)
                receipt_name = "warmup_receipt.json" if warmup else "cell_receipt.json"
                cell_receipt["stage_trace"] = stage_trace
                _write_json_atomic(temporary / receipt_name, cell_receipt)

        torch.cuda.synchronize(torch.device(device))
        output_records = _output_records(temporary)
        nvml = sampler.stop()
        sampler_started = False
        if nvml["device_total_bytes"] != int(torch.cuda.get_device_properties(0).total_memory):
            raise RuntimeError("NVML and torch disagree on device capacity")
        resource_receipt = {
            "schema": "rtgs.complete_boundary_resource.v1",
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "seed": seed,
            "arm": arm,
            "warmup": warmup,
            "wall_seconds": time.perf_counter() - origin,
            "stage_seconds": {
                name: values["end"] - values["start"] for name, values in stage_trace.items()
            },
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
            "input_bytes": cell_receipt["input_bytes"],
            "output_bytes": sum(int(item["bytes"]) for item in output_records),
            "output_files": output_records,
            "device_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            **nvml,
        }
        _write_json_new(temporary / "resource_receipt.json", resource_receipt)
        os.replace(temporary, target)
        print(f"completed {'warmup' if warmup else 'measured'} {seed}/{arm}", flush=True)
        return 0
    except BaseException as error:
        if guard_record is None and guard is not None:
            guard_record = guard.record()
        failure_nvml: dict[str, Any] | None = None
        if sampler_started:
            try:
                failure_nvml = sampler.stop(require_clean=False)
            except BaseException as sampler_error:
                failure_nvml = {
                    "sampler_stop_error": f"{type(sampler_error).__name__}: {sampler_error}"
                }
        failure = {
            "schema": "rtgs.gps_field_proxy_cell_failure.v1",
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "seed": seed,
            "arm": arm,
            "warmup": warmup,
            "status": "failed",
            "failure_phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_diagnostics": _jsonable(getattr(error, "diagnostics", None)),
            "traceback": traceback.format_exc(),
            "stage_trace": stage_trace,
            "heldout_loaded": any(
                stage in stage_trace
                for stage in {"initial_heldout_evaluation", "final_heldout_evaluation"}
            ),
            "quality_metrics_written": False,
            "input_boundary": guard_record,
            "nvml": failure_nvml,
            "peak_cuda_allocated_bytes": (
                None
                if torch_module is None or not torch_module.cuda.is_available()
                else int(torch_module.cuda.max_memory_allocated())
            ),
            "peak_cuda_reserved_bytes": (
                None
                if torch_module is None or not torch_module.cuda.is_available()
                else int(torch_module.cuda.max_memory_reserved())
            ),
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
            "wall_seconds": time.perf_counter() - origin,
        }
        _write_json_atomic(temporary / "failure.json", failure)
        _write_json_atomic(
            temporary / "resource_receipt.json",
            {
                "schema": "rtgs.complete_boundary_failure_resource.v1",
                "task_id": TASK_ID,
                "dataset_id": DATASET_ID,
                "seed": seed,
                "arm": arm,
                "warmup": warmup,
                "failure_phase": phase,
                "wall_seconds": failure["wall_seconds"],
                "stage_trace": stage_trace,
                "peak_cuda_allocated_bytes": failure["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": failure["peak_cuda_reserved_bytes"],
                "peak_process_rss_bytes": failure["peak_process_rss_bytes"],
                "nvml": failure_nvml,
            },
        )
        failure_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, failure_target)
        print(
            f"failed {'warmup' if warmup else 'measured'} {seed}/{arm}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1


def _execution_jobs(
    task: Mapping[str, Any],
) -> tuple[list[tuple[int, str, bool]], list[tuple[int, str, bool]]]:
    order = task["frozen_configuration"]["execution_order"]
    warmups = [(WARMUP_SEED, arm, True) for arm in order["warmup_arm_order"]]
    measured = [
        (int(seed), arm, False) for seed, arms in order["measured_orders"].items() for arm in arms
    ]
    if [arm for _seed, arm, _warmup in warmups] != list(ARMS):
        raise ValueError("frozen warmup order changed")
    expected = {(seed, arm) for seed in task["seeds"] for arm in ARMS}
    if {(seed, arm) for seed, arm, _warmup in measured} != expected or len(measured) != len(
        expected
    ):
        raise ValueError("frozen measured order is incomplete or duplicated")
    return warmups, measured


def _worker_command(
    *,
    task_path: Path,
    run: Path,
    gps_repo: Path,
    gps_checkpoint: Path,
    device: str,
    seed: int,
    arm: str,
    warmup: bool,
) -> list[str]:
    command = [
        sys.executable,
        os.fspath(Path(__file__).resolve()),
        "--task",
        os.fspath(task_path),
        "--run",
        os.fspath(run),
        "--gps-repo",
        os.fspath(gps_repo),
        "--gps-checkpoint",
        os.fspath(gps_checkpoint),
        "--device",
        device,
        "--worker",
        "--seed",
        str(seed),
        "--arm",
        arm,
    ]
    if warmup:
        command.append("--warmup")
    return command


def _launch_worker(command: Sequence[str], *, seed: int) -> int:
    environment = dict(os.environ)
    environment.update(
        PYTHONHASHSEED=str(seed),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONUNBUFFERED="1",
    )
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    return int(completed.returncode)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median requires at least one value")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def _completed_cells(task: Mapping[str, Any], run: Path) -> list[dict[str, Any]]:
    cells = []
    for seed in task["seeds"]:
        for arm in ARMS:
            directory = run / _cell_relative(seed, arm, warmup=False)
            if not directory.is_dir():
                continue
            receipt = _load_json(directory / "cell_receipt.json")
            resources = _load_json(directory / "resource_receipt.json")
            history = _load_json(directory / "compact_training_history.json")
            if receipt.get("status") != "completed":
                raise RuntimeError(f"completed cell has non-completed receipt: {seed}/{arm}")
            cells.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "directory": directory,
                    "receipt": receipt,
                    "resource": resources,
                    "history": history,
                }
            )
    return cells


def _failed_cells(task: Mapping[str, Any], run: Path, *, warmup: bool) -> list[dict[str, Any]]:
    seeds = [WARMUP_SEED] if warmup else task["seeds"]
    failures = []
    for seed in seeds:
        for arm in ARMS:
            directory = run / _failure_relative(seed, arm, warmup=warmup)
            if directory.is_dir():
                failures.append(_load_json(directory / "failure.json"))
    return failures


def _decision(task: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {(int(cell["seed"]), str(cell["arm"])): cell for cell in cells}
    rule = task["frozen_configuration"]["decision_rule"]
    initial_ratios: dict[str, float] = {}
    final_ratios: dict[str, float] = {}
    shuffled_ratios: dict[str, float] = {}
    initial_wins = 0
    final_wins = 0
    shuffled_wins = 0
    deterministic_complete = True
    gps_complete = True
    shuffled_complete = 0
    for seed in task["seeds"]:
        deterministic = [lookup.get((seed, arm)) for arm in DETERMINISTIC_ARMS]
        correct = lookup.get((seed, "gps_field_proxy"))
        shuffled = lookup.get((seed, "gps_shuffled_field"))
        if any(cell is None for cell in deterministic):
            deterministic_complete = False
        if correct is None:
            gps_complete = False
        if shuffled is not None:
            shuffled_complete += 1
        if correct is not None and all(cell is not None for cell in deterministic):
            gps_initial = float(correct["receipt"]["initial_evaluation"]["equal_view_j_area"])
            gps_final = float(correct["receipt"]["final_evaluation"]["equal_view_j_area"])
            best_initial = min(
                float(cell["receipt"]["initial_evaluation"]["equal_view_j_area"])
                for cell in deterministic
            )
            best_final = min(
                float(cell["receipt"]["final_evaluation"]["equal_view_j_area"])
                for cell in deterministic
            )
            initial_ratios[str(seed)] = gps_initial / max(best_initial, 1e-30)
            final_ratios[str(seed)] = gps_final / max(best_final, 1e-30)
            initial_wins += (
                gps_initial
                <= (1.0 - float(rule["minimum_initial_area_risk_reduction_vs_best_deterministic"]))
                * best_initial
            )
            final_wins += (
                gps_final
                <= float(rule["maximum_final_area_risk_ratio_vs_best_deterministic"]) * best_final
            )
        if correct is not None and shuffled is not None:
            correct_initial = float(correct["receipt"]["initial_evaluation"]["equal_view_j_area"])
            shuffled_initial = float(shuffled["receipt"]["initial_evaluation"]["equal_view_j_area"])
            shuffled_ratios[str(seed)] = shuffled_initial / max(correct_initial, 1e-30)
            shuffled_wins += (
                shuffled_initial
                >= (1.0 + float(rule["minimum_shuffled_control_area_risk_degradation"]))
                * correct_initial
            )
    completion = (
        gps_complete
        and deterministic_complete
        and shuffled_complete >= int(rule["minimum_paired_seed_wins"])
    )
    passed = (
        completion
        and initial_wins >= int(rule["minimum_paired_seed_wins"])
        and final_wins >= int(rule["minimum_paired_seed_wins"])
        and shuffled_wins >= int(rule["minimum_paired_seed_wins"])
    )
    return {
        "schema": "rtgs.gps_field_proxy_decision.v1",
        "completion_requirement_met": completion,
        "gps_cells_complete": gps_complete,
        "deterministic_cells_complete": deterministic_complete,
        "shuffled_cells_complete": shuffled_complete,
        "initial_ratios_to_best_deterministic": initial_ratios,
        "final_ratios_to_best_deterministic": final_ratios,
        "shuffled_to_correct_initial_ratios": shuffled_ratios,
        "initial_seed_wins": initial_wins,
        "final_seed_wins": final_wins,
        "shuffled_seed_wins": shuffled_wins,
        "minimum_paired_seed_wins": int(rule["minimum_paired_seed_wins"]),
        "geometry_pass_opens_field_native_successor": passed,
        "verdict": (
            "field_native_successor_opened"
            if passed
            else "successor_not_opened"
            if completion
            else "inconclusive_incomplete_cells"
        ),
    }


def _history_bundle(task: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = {stage["id"]: stage["label"] for stage in task["stages"]}
    stage_order = [stage["id"] for stage in task["stages"]]
    records: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    for cell in cells:
        seed, arm = int(cell["seed"]), str(cell["arm"])
        trace = cell["receipt"]["stage_trace"]
        step_cursor = 0
        stage_steps: dict[str, tuple[int, int]] = {}
        for stage in stage_order:
            start_step = step_cursor
            end_step = start_step + (250 if stage == "matched_compact_refinement" else 1)
            stage_steps[stage] = (start_step, end_step)
            markers.extend(
                [
                    {
                        "step": start_step,
                        "wall_seconds": float(trace[stage]["start"]),
                        "stage": stage,
                        "dataset_id": DATASET_ID,
                        "arm_id": arm,
                        "seed": seed,
                        "boundary": "start",
                        "label": labels[stage],
                    },
                    {
                        "step": end_step,
                        "wall_seconds": float(trace[stage]["end"]),
                        "stage": stage,
                        "dataset_id": DATASET_ID,
                        "arm_id": arm,
                        "seed": seed,
                        "boundary": "end",
                        "label": labels[stage],
                    },
                ]
            )
            records.append(
                {
                    "step": end_step,
                    "wall_seconds": float(trace[stage]["end"]),
                    "stage": stage,
                    "dataset_id": DATASET_ID,
                    "arm_id": arm,
                    "seed": seed,
                    "split": "diagnostic",
                    "metric_id": "stage_runtime_seconds",
                    "value": float(trace[stage]["end"] - trace[stage]["start"]),
                }
            )
            step_cursor = end_step
        train_start_step, train_end_step = stage_steps["matched_compact_refinement"]
        train_start_time = float(trace["matched_compact_refinement"]["start"])
        train_end_time = float(trace["matched_compact_refinement"]["end"])
        elapsed = train_start_time
        for item in cell["history"]["steps"]:
            elapsed = min(train_end_time, elapsed + float(item["elapsed_seconds"]))
            records.append(
                {
                    "step": train_start_step + int(item["step"]),
                    "wall_seconds": elapsed,
                    "stage": "matched_compact_refinement",
                    "dataset_id": DATASET_ID,
                    "arm_id": arm,
                    "seed": seed,
                    "split": "train",
                    "metric_id": "loss_total",
                    "value": float(item["total_sampled_loss"]),
                }
            )
        if train_start_step + len(cell["history"]["steps"]) != train_end_step:
            raise RuntimeError("compact history does not contain exactly 250 steps")
    return {
        "schema_version": 2,
        "records": records,
        "metric_metadata": {
            "loss_total": {
                "label": "Compact point objective",
                "unit": "MSE",
                "group": "Objective",
                "direction": "lower",
            },
            "stage_runtime_seconds": {
                "label": "Stage runtime",
                "unit": "seconds",
                "group": "Runtime",
                "direction": "descriptive",
            },
        }
        if records
        else {},
        "stage_markers": markers,
    }


def _evidence_entries() -> list[dict[str, str]]:
    return [
        {
            "label": label,
            "path": f"benchmarks/results/{TASK_ID}_{suffix}",
        }
        for label, suffix in (
            ("Result note", "RESULT.md"),
            ("Machine result", "RESULT.json"),
            ("Independent audit", "AUDIT.md"),
            ("Machine audit", "AUDIT.json"),
        )
    ]


def _required_artifacts(*, completed: bool) -> list[dict[str, str]]:
    common = [
        {"label": "Training history", "path": "training_history.json"},
        {"label": "Effective configuration", "path": "gaussians.config.json"},
        {"label": "Input-boundary receipt", "path": "input_boundary_receipt.json"},
        {"label": "Resource receipt", "path": "resource_receipt.json"},
        {"label": "Run receipt", "path": "run_receipt.json"},
        {"label": "Execution environment", "path": "environment.json"},
    ]
    if not completed:
        return common
    return [
        {"label": "Initial Gaussians", "path": "gaussians_init.ply"},
        {"label": "Final Gaussians", "path": "gaussians.ply"},
        *common,
    ]


def _commands(task: Mapping[str, Any], *, completed: bool) -> dict[str, Any]:
    return {
        "reproduce": task["run_command"],
        "serve_report": [
            ".venv/bin/python",
            "-m",
            "http.server",
            "8765",
            "--directory",
            RUN_RELATIVE.as_posix(),
        ],
        "viewer": (
            [
                ".venv/bin/rtgs",
                "view",
                "--gaussians",
                f"{RUN_RELATIVE.as_posix()}/gaussians.ply",
                "--initial",
                f"{RUN_RELATIVE.as_posix()}/gaussians_init.ply",
                "--no-open",
            ]
            if completed
            else None
        ),
    }


def _metric_metadata(task: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    groups = {
        "initial_heldout_j_area": "Quality",
        "final_heldout_j_area": "Quality",
        "valid_depth_candidate_fraction": "Geometry",
        "left_right_consistency_px": "Geometry",
        "peak_cuda_allocated_bytes": "Resources",
        "nvml_process_peak_bytes": "Resources",
        "peak_process_rss_bytes": "Resources",
        "wall_seconds": "Runtime",
        "final_gaussians": "Cardinality",
    }
    return {
        item["id"]: {
            "label": item["label"],
            "unit": item["unit"],
            "group": groups[item["id"]],
            "direction": item["direction"],
        }
        for item in task["primary_metrics"]
    }


def _metrics_bundle(
    task: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    correct = [cell for cell in cells if cell["arm"] == "gps_field_proxy"]
    if not correct:
        raise ValueError("completed metrics require at least one correct GPS cell")
    metrics = {
        "initial_heldout_j_area": _median(
            [cell["receipt"]["initial_evaluation"]["equal_view_j_area"] for cell in correct]
        ),
        "final_heldout_j_area": _median(
            [cell["receipt"]["final_evaluation"]["equal_view_j_area"] for cell in correct]
        ),
        "valid_depth_candidate_fraction": _median(
            [
                cell["receipt"]["initializer"]["diagnostics"]["valid_depth_candidate_fraction"]
                for cell in correct
            ]
        ),
        "left_right_consistency_px": _median(
            [
                cell["receipt"]["initializer"]["diagnostics"][
                    "selected_left_right_consistency_median_px"
                ]
                for cell in correct
            ]
        ),
        "peak_cuda_allocated_bytes": _median(
            [cell["resource"]["peak_cuda_allocated_bytes"] for cell in correct]
        ),
        "nvml_process_peak_bytes": _median(
            [cell["resource"]["nvml_process_peak_bytes"] for cell in correct]
        ),
        "peak_process_rss_bytes": _median(
            [cell["resource"]["peak_process_rss_bytes"] for cell in correct]
        ),
        "wall_seconds": _median([cell["resource"]["wall_seconds"] for cell in correct]),
        "final_gaussians": _median([cell["receipt"]["final_gaussians"] for cell in correct]),
    }
    by_arm = {arm: [cell for cell in cells if cell["arm"] == arm] for arm in ARMS}

    def values(metric: str, source: str) -> list[dict[str, Any]]:
        output = []
        for arm in ARMS:
            rows = by_arm[arm]
            if not rows:
                continue
            if source == "initial":
                samples = [row["receipt"]["initial_evaluation"][metric] for row in rows]
            elif source == "final":
                samples = [row["receipt"]["final_evaluation"][metric] for row in rows]
            elif source == "resource":
                samples = [row["resource"][metric] for row in rows]
            else:
                samples = [row["resource"]["stage_seconds"][metric] for row in rows]
            output.append({"label": ARM_LABELS[arm], "value": _median(samples)})
        return output

    quality_values = []
    for state in ("initial", "final"):
        quality_values.extend(
            {
                "label": f"{item['label']} ({state})",
                "value": item["value"],
            }
            for item in values("equal_view_j_area", state)
        )
    verdict = str(decision["verdict"])
    summary = (
        "The frozen field-proxy geometry gate passed and opens the preregistered field-native "
        "successor."
        if decision["geometry_pass_opens_field_native_successor"]
        else "The frozen field-proxy experiment did not open the field-native successor; see "
        "the paired seed ratios and completion receipt."
    )
    return {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": TASK_ID,
        "summary": summary,
        "decision": verdict,
        "claim_boundary": task["claim_boundary"],
        "metrics": metrics,
        "metric_metadata": _metric_metadata(task),
        "charts": [
            {
                "id": "quality",
                "title": "Fixed held-out midpoint area risk",
                "unit": "MSE",
                "values": quality_values,
            },
            {
                "id": "resources",
                "title": "Complete-boundary process VRAM",
                "unit": "bytes",
                "values": values("nvml_process_peak_bytes", "resource"),
            },
            {
                "id": "stage_runtime",
                "title": "Complete-boundary runtime",
                "unit": "seconds",
                "values": values("wall_seconds", "resource"),
            },
        ],
        "artifacts": _required_artifacts(completed=True),
        "evidence": _evidence_entries(),
        "commands": _commands(task, completed=True),
        "notes": [
            "All quality comparisons use identical 128x128 midpoint lattices and exact "
            "3,000-Gaussian topology.",
            "Memory is descriptive until eager, streamed, decode-on-demand, and point-sampled "
            "RGB controls exist.",
            f"Decision receipt: {json.dumps(decision, sort_keys=True)}",
        ],
    }


def _failed_metrics_bundle(task: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": TASK_ID,
        "summary": message,
        "decision": "run_failed_before_complete_outcomes",
        "claim_boundary": task["claim_boundary"],
        "metrics": {},
        "metric_metadata": {},
        "charts": [],
        "artifacts": _required_artifacts(completed=False),
        "evidence": _evidence_entries(),
        "commands": _commands(task, completed=False),
        "notes": [
            "No missing cell was imputed or silently dropped.",
            "Failure evidence is retained under failures/.",
        ],
    }


def _environment() -> dict[str, Any]:
    import torch

    packages = {
        "realtime-gs": f"git:{_git('rev-parse', 'HEAD').strip()}",
        "torch": torch.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "nvidia-ml-py": importlib.metadata.version("nvidia-ml-py"),
        "gsplat": importlib.metadata.version("gsplat"),
    }
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "cuda": str(torch.version.cuda),
        },
    }


def _publish_sources(
    task: Mapping[str, Any],
    run: Path,
    *,
    cells: Sequence[Mapping[str, Any]],
    completed: bool,
    message: str,
    decision: Mapping[str, Any] | None,
    representative: Mapping[str, Any] | None,
) -> None:
    lock = _load_json(run / "task.lock.json")
    if completed and representative is None:
        raise ValueError("completed publication requires a representative cell")
    if representative is not None:
        source = Path(representative["directory"])
        shutil.copy2(source / "gaussians_init.ply", run / "gaussians_init.ply")
        shutil.copy2(source / "gaussians.ply", run / "gaussians.ply")
    history = _history_bundle(task, cells)
    _write_json_new(run / "training_history.json", history)
    failures = {
        "warmups": _failed_cells(task, run, warmup=True),
        "measured": _failed_cells(task, run, warmup=False),
    }
    _write_json_new(
        run / "gaussians.config.json",
        {
            "schema": "rtgs.gps_field_proxy_aggregate_config.v1",
            "task_id": TASK_ID,
            "frozen_configuration": task["frozen_configuration"],
            "completed_cells": [{"seed": cell["seed"], "arm": cell["arm"]} for cell in cells],
            "failed_cells": failures,
            "decision": decision,
            "representative": (
                None
                if representative is None
                else {"seed": representative["seed"], "arm": representative["arm"]}
            ),
        },
    )
    _write_json_new(
        run / "input_boundary_receipt.json",
        {
            "schema": "rtgs.gps_field_proxy_aggregate_input_boundary.v1",
            "task_id": TASK_ID,
            "allowed_modalities": ["calibration", "gaussians2d"],
            "source_rgb_or_mask_opened": False,
            "compact_alpha_loaded": False,
            "heldout_reporting_only": True,
            "completed_cell_guards": [
                {
                    "seed": cell["seed"],
                    "arm": cell["arm"],
                    "guard": cell["receipt"]["input_boundary"],
                    "input_bytes": cell["receipt"]["input_bytes"],
                }
                for cell in cells
            ],
            "failures": failures,
        },
    )
    _write_json_new(
        run / "resource_receipt.json",
        {
            "schema": "rtgs.gps_field_proxy_aggregate_resource.v1",
            "task_id": TASK_ID,
            "resource_protocol": task["resource_protocol"],
            "monitor_configuration": RESOURCE_MONITOR,
            "cells": [
                {
                    "seed": cell["seed"],
                    "arm": cell["arm"],
                    "resource_path": (Path(cell["directory"]) / "resource_receipt.json")
                    .relative_to(run)
                    .as_posix(),
                    "nvml_process_peak_bytes": cell["resource"]["nvml_process_peak_bytes"],
                    "peak_cuda_allocated_bytes": cell["resource"]["peak_cuda_allocated_bytes"],
                    "peak_cuda_reserved_bytes": cell["resource"]["peak_cuda_reserved_bytes"],
                    "peak_process_rss_bytes": cell["resource"]["peak_process_rss_bytes"],
                    "wall_seconds": cell["resource"]["wall_seconds"],
                }
                for cell in cells
            ],
            "failures": failures,
        },
    )
    _write_json_new(run / "environment.json", _environment())
    metrics = (
        _metrics_bundle(task, cells, decision or {})
        if completed
        else _failed_metrics_bundle(task, message)
    )
    _write_json_new(run / "metrics.json", metrics)
    _write_json_new(
        run / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed" if completed else "failed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": 0 if completed else 1,
            "failure_phase": None if completed else "warmup_or_required_gps_cells",
            "message": message,
        },
    )


def _orchestrate(
    *,
    task_path: Path,
    run: Path,
    gps_repo: Path,
    gps_checkpoint: Path,
    device: str,
) -> int:
    task = _validate_run_binding(task_path, run)
    gps_repo = _resolve_exact(gps_repo, DEFAULT_GPS_REPO, label="GPS repository")
    gps_checkpoint = _resolve_exact(gps_checkpoint, DEFAULT_GPS_CHECKPOINT, label="GPS checkpoint")
    if any((run / name).exists() for name in ("metrics.json", "run_receipt.json")):
        raise FileExistsError("run sources already exist")
    warmups, measured = _execution_jobs(task)
    for seed, arm, warmup in warmups:
        print(f"launching warmup {seed}/{arm}", flush=True)
        code = _launch_worker(
            _worker_command(
                task_path=task_path,
                run=run,
                gps_repo=gps_repo,
                gps_checkpoint=gps_checkpoint,
                device=device,
                seed=seed,
                arm=arm,
                warmup=warmup,
            ),
            seed=seed,
        )
        if code != 0:
            message = f"Warmup {seed}/{arm} failed; frozen protocol halted measured launches."
            _publish_sources(
                task,
                run,
                cells=[],
                completed=False,
                message=message,
                decision=None,
                representative=None,
            )
            return 1
    for seed, arm, warmup in measured:
        print(f"launching measured {seed}/{arm}", flush=True)
        _launch_worker(
            _worker_command(
                task_path=task_path,
                run=run,
                gps_repo=gps_repo,
                gps_checkpoint=gps_checkpoint,
                device=device,
                seed=seed,
                arm=arm,
                warmup=warmup,
            ),
            seed=seed,
        )
    cells = _completed_cells(task, run)
    decision = _decision(task, cells)
    correct = [cell for cell in cells if cell["arm"] == "gps_field_proxy"]
    if not correct:
        message = "No correct-GPS measured cell completed; the run is outcome-incomplete."
        _publish_sources(
            task,
            run,
            cells=cells,
            completed=False,
            message=message,
            decision=decision,
            representative=None,
        )
        return 1
    representative = next(
        (cell for cell in correct if cell["seed"] == task["seeds"][0]), correct[0]
    )
    message = (
        "All frozen workers finished; the field-native successor gate passed."
        if decision["geometry_pass_opens_field_native_successor"]
        else "All scheduled workers were attempted; the field-native successor gate did not pass."
    )
    _publish_sources(
        task,
        run,
        cells=cells,
        completed=True,
        message=message,
        decision=decision,
        representative=representative,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--gps-repo", type=Path, default=DEFAULT_GPS_REPO)
    parser.add_argument("--gps-checkpoint", type=Path, default=DEFAULT_GPS_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker and (args.seed is None or args.arm is None):
        parser.error("internal worker requires --seed and --arm")
    if not args.worker and (args.seed is not None or args.arm is not None or args.warmup):
        parser.error("worker-only arguments were supplied to the orchestrator")
    return args


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _run_worker(
            task_path=args.task,
            run=args.run,
            gps_repo=args.gps_repo,
            gps_checkpoint=args.gps_checkpoint,
            device=args.device,
            seed=args.seed,
            arm=args.arm,
            warmup=args.warmup,
        )
    return _orchestrate(
        task_path=args.task,
        run=args.run,
        gps_repo=args.gps_repo,
        gps_checkpoint=args.gps_checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
