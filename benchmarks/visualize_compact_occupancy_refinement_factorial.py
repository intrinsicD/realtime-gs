"""Post-result-only exact gsplat visualization for the compact factorial iter3.

This utility is intentionally outside the decision-bearing experiment.  It refuses to create
anything until the immutable iter3 RESULT is a PASS, then renders only the RESULT-linked final
PLYs for seed 76801.  Calibrated cameras come exclusively from the strict RGB-free compact
bundle; source RGB is neither loaded nor used as a render input.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import stat
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT / "benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json"
)
EXPECTED_RESULT_SHA256 = "c0a278a8cc41f12632be121b14937f9fc2a2a03cd03716bae96b5bd9d6510116"
TEACHER_BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
RUN_DIR = ROOT / "runs/compact_occupancy_refinement_factorial_iter3_20260717"
OUTPUT_DIR = RUN_DIR / "visualization_seed_76801"
PLAN = OUTPUT_DIR / "PLAN.json"
RECEIPT = OUTPUT_DIR / "RECEIPT.json"
FAILURE = OUTPUT_DIR / "FAILURE.json"

REQUIRED_SEED = 76801
ARMS = ("A", "B", "C", "D")
VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
NATIVE_WIDTH = 5328
NATIVE_HEIGHT = 4608
EXPECTED_BACKEND = "rtgs.render.gsplat_backend.GsplatRasterizer"
PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33")
EXPECTED_PRELOAD_SHA256 = "1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11"
WORKER_TIMEOUT_SECONDS = 1800
THUMBNAIL_WIDTH = 720
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr"})

SOURCE_PATHS = (
    Path("benchmarks/visualize_compact_occupancy_refinement_factorial.py"),
    Path("src/rtgs/__init__.py"),
    Path("src/rtgs/core/__init__.py"),
    Path("src/rtgs/viewer.py"),
    Path("src/rtgs/render/__init__.py"),
    Path("src/rtgs/render/base.py"),
    Path("src/rtgs/render/gsplat_backend.py"),
    Path("src/rtgs/render/point_base.py"),
    Path("src/rtgs/render/torch_points.py"),
    Path("src/rtgs/core/camera.py"),
    Path("src/rtgs/core/gaussians2d.py"),
    Path("src/rtgs/core/gaussians3d.py"),
    Path("src/rtgs/core/observation2d.py"),
    Path("src/rtgs/core/sh.py"),
    Path("src/rtgs/data/__init__.py"),
    Path("src/rtgs/data/reconstruction_inputs.py"),
)


class ProtocolInvalid(RuntimeError):
    """A fail-closed visualization provenance invariant failed."""


def timestamp_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProtocolInvalid(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ProtocolInvalid(f"non-finite JSON constant {value!r} in {source}")

    try:
        value = json.loads(
            source.read_bytes(),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolInvalid(f"cannot strictly load JSON object {source}") from error
    if not isinstance(value, dict):
        raise ProtocolInvalid(f"{source} is not a JSON object")
    canonical_bytes(value)
    return value


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
    encoded = canonical_bytes(dict(payload))
    _exclusive_bytes(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def display_path(path: Path, *, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _ordinary_repo_file(root: Path, relative_text: object, *, label: str) -> Path:
    if not isinstance(relative_text, str):
        raise ProtocolInvalid(f"{label} path is not text")
    relative = PurePosixPath(relative_text)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.as_posix() != relative_text
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in relative_text
    ):
        raise ProtocolInvalid(f"{label} is not a canonical repository-relative path")
    root = root.resolve(strict=True)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as error:
            raise ProtocolInvalid(f"{label} does not exist: {relative_text}") from error
        if stat.S_ISLNK(mode):
            raise ProtocolInvalid(f"{label} path contains a symlink: {relative_text}")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ProtocolInvalid(f"{label} escapes the repository") from error
    if not stat.S_ISREG(os.lstat(resolved).st_mode):
        raise ProtocolInvalid(f"{label} is not an ordinary file")
    return resolved


def _ordinary_file(path: Path, *, label: str) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as error:
        raise ProtocolInvalid(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ProtocolInvalid(f"{label} must be an ordinary non-symlink file")
    return path.resolve(strict=True)


def directory_binding(directory: Path, *, root: Path = ROOT) -> dict[str, Any]:
    try:
        mode = os.lstat(directory).st_mode
    except FileNotFoundError as error:
        raise ProtocolInvalid(f"bound directory does not exist: {directory}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ProtocolInvalid(f"bound path is not an ordinary directory: {directory}")
    records = []
    for path in sorted(directory.rglob("*")):
        item_mode = os.lstat(path).st_mode
        if stat.S_ISLNK(item_mode):
            raise ProtocolInvalid(f"bound directory contains a symlink: {path}")
        if stat.S_ISREG(item_mode):
            records.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not records:
        raise ProtocolInvalid(f"bound directory is empty: {directory}")
    return {
        "path": display_path(directory, root=root),
        "files": records,
        "aggregate_sha256": canonical_hash(records),
    }


def source_binding(*, root: Path = ROOT) -> dict[str, Any]:
    files = {}
    for relative in SOURCE_PATHS:
        path = _ordinary_repo_file(root, relative.as_posix(), label="visualization source")
        files[relative.as_posix()] = sha256_file(path)
    return {"files": files, "aggregate_sha256": canonical_hash(files)}


def preload_binding(path: Path = PRELOAD) -> dict[str, Any]:
    resolved = _ordinary_file(path, label="bound libstdc++ preload")
    digest = sha256_file(resolved)
    if path == PRELOAD and digest != EXPECTED_PRELOAD_SHA256:
        raise ProtocolInvalid("bound libstdc++ preload hash changed")
    return {
        "requested_path": str(path),
        "resolved_path": str(resolved),
        "sha256": digest,
    }


def validate_result_ply_bindings(
    result: Mapping[str, Any],
    *,
    root: Path = ROOT,
    required_seed: int = REQUIRED_SEED,
    arms: Sequence[str] = ARMS,
) -> dict[str, dict[str, Any]]:
    workers = result.get("workers")
    seed_workers = workers.get(str(required_seed)) if isinstance(workers, Mapping) else None
    if not isinstance(seed_workers, Mapping) or set(seed_workers) != set(arms):
        raise ProtocolInvalid("RESULT does not contain exactly four required seed-76801 arms")
    records: dict[str, dict[str, Any]] = {}
    for arm in arms:
        worker = seed_workers[arm]
        final_ply = worker.get("final_ply") if isinstance(worker, Mapping) else None
        if (
            not isinstance(final_ply, Mapping)
            or worker.get("status") != "PASS"
            or worker.get("arm") != arm
            or worker.get("training_seed") != required_seed
            or isinstance(worker.get("n_opt_3d"), bool)
            or worker.get("n_opt_3d") != 835
            or final_ply.get("n_gaussians") != 835
            or isinstance(final_ply.get("bytes"), bool)
            or not isinstance(final_ply.get("bytes"), int)
            or final_ply["bytes"] <= 0
            or not _is_sha256(final_ply.get("sha256"))
        ):
            raise ProtocolInvalid(f"RESULT worker/final PLY schema differs for arm {arm}")
        path = _ordinary_repo_file(root, final_ply.get("path"), label=f"arm {arm} final PLY")
        if path.stat().st_size != final_ply["bytes"] or sha256_file(path) != final_ply["sha256"]:
            raise ProtocolInvalid(f"RESULT-linked final PLY changed for arm {arm}")
        records[arm] = {
            "arm": arm,
            "training_seed": required_seed,
            "path": display_path(path, root=root),
            "sha256": final_ply["sha256"],
            "bytes": final_ply["bytes"],
            "n_gaussians": final_ply["n_gaussians"],
        }
    return records


def validate_result(
    *,
    result_path: Path = RESULT,
    root: Path = ROOT,
    bundle_path: Path = TEACHER_BUNDLE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_file = _ordinary_file(result_path, label="immutable iter3 RESULT")
    before = sha256_file(result_file)
    if (
        result_file == RESULT.resolve()
        and root.resolve() == ROOT.resolve()
        and before != EXPECTED_RESULT_SHA256
    ):
        raise ProtocolInvalid("immutable iter3 RESULT hash differs")
    result = strict_json(result_file)
    after = sha256_file(result_file)
    if before != after:
        raise ProtocolInvalid("iter3 RESULT changed while it was being loaded")
    if (
        result.get("artifact_type") != "compact_occupancy_refinement_factorial_iter3_result_v1"
        or result.get("status") != "PASS"
    ):
        raise ProtocolInvalid("iter3 RESULT is not an immutable PASS result")
    visualization = result.get("visualization")
    if not isinstance(visualization, Mapping) or (
        visualization.get("status") != "DEFERRED_POST_RESULT"
        or visualization.get("required_seed") != REQUIRED_SEED
        or visualization.get("required_arms") != list(ARMS)
        or visualization.get("required_backend") != "gsplat"
        or visualization.get("required_native_resolution") != [NATIVE_WIDTH, NATIVE_HEIGHT]
    ):
        raise ProtocolInvalid("RESULT deferred-visualization contract differs")
    runtime = result.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("preload") != str(PRELOAD)
        or runtime.get("preload_sha256") != EXPECTED_PRELOAD_SHA256
    ):
        raise ProtocolInvalid("RESULT does not bind the required libstdc++ runtime")
    inputs = result.get("inputs")
    expected_bundle = inputs.get("teacher_bundle") if isinstance(inputs, Mapping) else None
    current_bundle = directory_binding(bundle_path, root=root)
    if expected_bundle != current_bundle:
        raise ProtocolInvalid("strict compact camera bundle changed since RESULT")
    plys = validate_result_ply_bindings(result, root=root)
    binding = {
        "path": display_path(result_file, root=root),
        "sha256": before,
        "bytes": result_file.stat().st_size,
        "artifact_type": result["artifact_type"],
        "status": result["status"],
    }
    return result, {"result": binding, "bundle": current_bundle, "plys": plys}


def _camera_record(camera: Any) -> dict[str, Any]:
    return {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "width": int(camera.width),
        "height": int(camera.height),
        "R": camera.R.detach().cpu().reshape(-1).tolist(),
        "t": camera.t.detach().cpu().tolist(),
    }


def load_strict_camera_binding(
    bundle_path: Path = TEACHER_BUNDLE,
) -> tuple[Any, dict[str, Any]]:
    # Lazy import keeps the module itself CPU/CUDA-backend agnostic.
    from rtgs.data.reconstruction_inputs import ReconstructionInputs

    inputs = ReconstructionInputs.load(bundle_path, device="cpu", strict=True)
    if inputs.view_names != list(VIEWS) or len(inputs.cameras) != len(VIEWS):
        raise ProtocolInvalid("strict compact bundle camera order changed")
    records = []
    for view_id, camera in zip(inputs.view_names, inputs.cameras, strict=True):
        record = _camera_record(camera)
        if (record["width"], record["height"]) != (NATIVE_WIDTH, NATIVE_HEIGHT):
            raise ProtocolInvalid(f"compact camera {view_id} is not native 5328x4608")
        records.append({"view_id": view_id, "camera": record})
    binding = {
        "bundle_name": inputs.name,
        "view_count": len(records),
        "views": records,
        "semantic_sha256": canonical_hash(records),
        "source_policy": "ReconstructionInputs.load(strict=True); cameras only; no SceneData/RGB",
    }
    return inputs, binding


def build_plan() -> dict[str, Any]:
    _, result_bindings = validate_result()
    _, cameras = load_strict_camera_binding()
    _, result_bindings_after = validate_result()
    if result_bindings_after != result_bindings:
        raise ProtocolInvalid("RESULT/bundle/PLY bindings changed while cameras were loaded")
    return {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_visualization_plan_v1",
        "timestamp_utc": timestamp_utc(),
        "decision_bearing": False,
        "scope": "post-result exact-render visualization only; no training or metric claim",
        "result": result_bindings["result"],
        "bundle": result_bindings["bundle"],
        "cameras": cameras,
        "plys": result_bindings["plys"],
        "sources": source_binding(),
        "preload": preload_binding(),
        "render_contract": {
            "seed": REQUIRED_SEED,
            "arms": list(ARMS),
            "views": list(VIEWS),
            "native_resolution": [NATIVE_WIDTH, NATIVE_HEIGHT],
            "call": (
                "render_exact_snapshot(model,camera,device='cuda',rasterizer='gsplat',"
                "packed=False,antialiased=False)"
            ),
            "backend": EXPECTED_BACKEND,
            "device": "cuda",
            "rasterizer": "gsplat",
            "packed": False,
            "antialiased": False,
            "worker_isolation": "one fresh bounded subprocess per arm",
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "contact_sheet": {
                "columns": list(ARMS),
                "rows": list(VIEWS),
                "thumbnail_width": THUMBNAIL_WIDTH,
                "individual_pngs_remain_native": True,
            },
        },
        "outputs": {
            "directory": display_path(OUTPUT_DIR),
            "plan": display_path(PLAN),
            "receipt": display_path(RECEIPT),
            "contact_sheet": display_path(OUTPUT_DIR / "CONTACT_SHEET.png"),
            "worker_records": {
                arm: display_path(OUTPUT_DIR / f"worker_{arm}.json") for arm in ARMS
            },
        },
    }


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


class ImageAccessGuard(contextlib.AbstractContextManager["ImageAccessGuard"]):
    """Deny dataset access and image-file access outside the visualization output tree."""

    def __init__(self, allowed_output: Path, *, dataset_root: Path | None = None):
        self.allowed_output = allowed_output.resolve(strict=False)
        self.dataset_root = (
            (ROOT / "dataset").resolve(strict=False)
            if dataset_root is None
            else dataset_root.resolve(strict=False)
        )
        self._original_builtin_open: Any = None
        self._original_io_open: Any = None
        self._original_os_open: Any = None
        self._forbidden = 0
        self._allowed_output_images = 0

    def _check(self, value: Any) -> None:
        if isinstance(value, int):
            return
        try:
            candidate = Path(os.fspath(value)).resolve(strict=False)
        except TypeError:
            return
        in_output = _path_is_within(candidate, self.allowed_output)
        forbidden = _path_is_within(candidate, self.dataset_root) or (
            candidate.suffix.lower() in IMAGE_SUFFIXES and not in_output
        )
        if forbidden:
            self._forbidden += 1
            raise ProtocolInvalid(f"source RGB/dataset access is forbidden: {candidate}")
        if in_output and candidate.suffix.lower() in IMAGE_SUFFIXES:
            self._allowed_output_images += 1

    def __enter__(self) -> ImageAccessGuard:
        self._original_builtin_open = builtins.open
        self._original_io_open = io.open
        self._original_os_open = os.open

        def guarded_builtin(file: Any, *args: Any, **kwargs: Any) -> Any:
            self._check(file)
            return self._original_builtin_open(file, *args, **kwargs)

        def guarded_io(file: Any, *args: Any, **kwargs: Any) -> Any:
            self._check(file)
            return self._original_io_open(file, *args, **kwargs)

        def guarded_os(file: Any, *args: Any, **kwargs: Any) -> Any:
            self._check(file)
            return self._original_os_open(file, *args, **kwargs)

        builtins.open = guarded_builtin
        io.open = guarded_io
        os.open = guarded_os
        return self

    def __exit__(self, *exc_info: object) -> None:
        builtins.open = self._original_builtin_open
        io.open = self._original_io_open
        os.open = self._original_os_open

    def record(self) -> dict[str, Any]:
        return {
            "passed": self._forbidden == 0,
            "source_rgb_or_dataset_open_attempts": self._forbidden,
            "allowed_output_image_open_attempts": self._allowed_output_images,
        }


def _verify_plan_live(plan: Mapping[str, Any], plan_sha256: str) -> None:
    if sha256_file(PLAN) != plan_sha256 or strict_json(PLAN) != plan:
        raise ProtocolInvalid("visualization plan changed")
    _, live = validate_result()
    if (
        plan.get("result") != live["result"]
        or plan.get("bundle") != live["bundle"]
        or plan.get("plys") != live["plys"]
        or plan.get("sources") != source_binding()
        or plan.get("preload") != preload_binding()
    ):
        raise ProtocolInvalid("post-result visualization binding changed")
    _, cameras = load_strict_camera_binding()
    if plan.get("cameras") != cameras:
        raise ProtocolInvalid("strict compact camera semantics changed")


def _tensor_hash(tensor: Any) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor hash requires torch.Tensor")
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _save_native_png(path: Path, color: Any) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    if tuple(color.shape) != (NATIVE_HEIGHT, NATIVE_WIDTH, 3) or not bool(
        torch.isfinite(color).all()
    ):
        raise ProtocolInvalid(f"exact gsplat color has invalid shape/values: {color.shape}")
    array = np.ascontiguousarray(
        (color.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    )
    with path.open("xb") as stream:
        Image.fromarray(array).save(stream, format="PNG", optimize=False)
        stream.flush()
        os.fsync(stream.fileno())
    with Image.open(path) as decoded:
        dimensions = list(decoded.size)
        decoded.verify()
    if dimensions != [NATIVE_WIDTH, NATIVE_HEIGHT] or path.stat().st_size <= 0:
        raise ProtocolInvalid("native PNG round trip differs")
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dimensions": dimensions,
        "color_tensor_sha256": _tensor_hash(color),
    }


def _gsplat_package_binding() -> dict[str, Any]:
    import gsplat

    module_root = Path(gsplat.__file__).resolve().parent
    suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".so", ".pyd"}
    files = []
    for path in sorted(module_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(
                {
                    "path": path.relative_to(module_root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    if not files:
        raise ProtocolInvalid("installed gsplat package binding is empty")
    return {
        "version": importlib.metadata.version("gsplat"),
        "module_file": str(Path(gsplat.__file__).resolve()),
        "module_root": str(module_root),
        "source_file_count": len(files),
        "source_manifest_sha256": canonical_hash(files),
    }


def _loaded_gsplat_extension_binding() -> dict[str, Any]:
    import gsplat.cuda._backend as backend

    extension = getattr(backend, "_C", None)
    path_text = None if extension is None else getattr(extension, "__file__", None)
    if path_text is None:
        raise ProtocolInvalid("gsplat render did not expose a loaded CUDA extension")
    path = Path(path_text).resolve()
    if not path.is_file() or path.suffix not in {".so", ".pyd"}:
        raise ProtocolInvalid("loaded gsplat extension is not a binary module")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _runtime_binding(preload: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    mapped = []
    resolved = preload["resolved_path"]
    if sys.platform == "linux":
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
            if resolved in line:
                mapped.append(line)
    if not mapped:
        raise ProtocolInvalid("bound libstdc++ is absent from the worker process map")
    properties = torch.cuda.get_device_properties(0)
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
    if driver.returncode != 0:
        raise ProtocolInvalid("nvidia-smi runtime binding failed")
    binding = {
        "python": sys.version,
        "executable": sys.executable,
        "executable_resolved": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory": int(properties.total_memory),
        "cuda_uuid": str(properties.uuid),
        "nvidia_smi_driver_device": driver.stdout.strip(),
        "ld_preload_at_startup": os.environ.get("LD_PRELOAD"),
        "preload": dict(preload),
        "preload_proc_maps_sha256": canonical_hash(mapped),
        "process_id": os.getpid(),
    }
    binding["binding_sha256"] = canonical_hash(binding)
    return binding


def run_worker(*, plan_path: Path, plan_sha256: str, arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ProtocolInvalid("unknown visualization arm")
    if os.environ.get("LD_PRELOAD") != str(PRELOAD):
        raise ProtocolInvalid("render worker did not start under the exact bound LD_PRELOAD")
    plan_path = _ordinary_file(plan_path, label="visualization plan")
    if plan_path != PLAN.resolve() or sha256_file(plan_path) != plan_sha256:
        raise ProtocolInvalid("render worker received the wrong plan")
    plan = strict_json(plan_path)
    _verify_plan_live(plan, plan_sha256)
    arm_dir = OUTPUT_DIR / f"arm_{arm}"
    arm_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    guard = ImageAccessGuard(OUTPUT_DIR)
    with guard:
        inputs, cameras = load_strict_camera_binding()
        if cameras != plan["cameras"]:
            raise ProtocolInvalid("worker strict camera binding differs from plan")

        import torch

        from rtgs.core.gaussians3d import Gaussians3D
        from rtgs.viewer import render_exact_snapshot

        if not torch.cuda.is_available():
            raise ProtocolInvalid("exact visualization requires CUDA")
        ply_record = plan["plys"][arm]
        ply_path = _ordinary_repo_file(ROOT, ply_record["path"], label=f"arm {arm} final PLY")
        if sha256_file(ply_path) != ply_record["sha256"]:
            raise ProtocolInvalid(f"arm {arm} PLY changed before rendering")
        model = Gaussians3D.load_ply(ply_path)
        if model.n != ply_record["n_gaussians"]:
            raise ProtocolInvalid(f"arm {arm} loaded Gaussian count differs")
        images = []
        for view_id, camera in zip(VIEWS, inputs.cameras, strict=True):
            snapshot = render_exact_snapshot(
                model,
                camera,
                device="cuda",
                rasterizer="gsplat",
                packed=False,
                antialiased=False,
            )
            if snapshot.backend != EXPECTED_BACKEND or not snapshot.device.startswith("cuda"):
                raise ProtocolInvalid("exact snapshot resolved a non-gsplat/non-CUDA backend")
            png = arm_dir / f"{view_id}.png"
            images.append(
                {
                    "view_id": view_id,
                    **_save_native_png(png, snapshot.color),
                    "backend": snapshot.backend,
                    "device": snapshot.device,
                }
            )
            del snapshot
            torch.cuda.empty_cache()
        extension = _loaded_gsplat_extension_binding()
        package = _gsplat_package_binding()
        runtime = _runtime_binding(plan["preload"])
        if sha256_file(ply_path) != ply_record["sha256"]:
            raise ProtocolInvalid(f"arm {arm} PLY changed during rendering")
    denial = guard.record()
    if not denial["passed"]:
        raise ProtocolInvalid("render worker attempted source RGB/dataset access")
    return {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_visualization_worker_v1",
        "timestamp_utc": timestamp_utc(),
        "status": "PASS",
        "arm": arm,
        "training_seed": REQUIRED_SEED,
        "plan_sha256": plan_sha256,
        "result_sha256": plan["result"]["sha256"],
        "source_aggregate_sha256": plan["sources"]["aggregate_sha256"],
        "ply": dict(plan["plys"][arm]),
        "camera_semantic_sha256": plan["cameras"]["semantic_sha256"],
        "images": images,
        "render_contract": {
            "backend": EXPECTED_BACKEND,
            "device": "cuda",
            "rasterizer": "gsplat",
            "packed": False,
            "antialiased": False,
            "native_resolution": [NATIVE_WIDTH, NATIVE_HEIGHT],
        },
        "rgb_denial": denial,
        "gsplat_package": package,
        "loaded_gsplat_extension": extension,
        "runtime": runtime,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _bounded_failure(error: BaseException, *, stage: str, arm: str | None = None) -> dict[str, Any]:
    return {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_visualization_failure_v1",
        "timestamp_utc": timestamp_utc(),
        "status": "FAIL",
        "stage": stage,
        "arm": arm,
        "failure": {
            "type": type(error).__name__,
            "message": str(error)[:2000],
            "traceback": "".join(traceback.format_exception(error))[-8000:],
        },
    }


def worker_entry(args: argparse.Namespace) -> int:
    arm = str(args.arm)
    output = OUTPUT_DIR / f"worker_{arm}.json"
    try:
        payload = run_worker(
            plan_path=Path(args.plan).resolve(),
            plan_sha256=str(args.plan_sha256),
            arm=arm,
        )
        digest = exclusive_json(output, payload)
        print(json.dumps({"status": "PASS", "worker": str(output), "sha256": digest}))
        return 0
    except BaseException as error:
        payload = _bounded_failure(error, stage="render_worker", arm=arm)
        try:
            digest = exclusive_json(output, payload)
            print(json.dumps({"status": "FAIL", "worker": str(output), "sha256": digest}))
        except BaseException:
            print(json.dumps({"status": "FAIL", "failure": str(error)[:1000]}))
        return 2


def validate_worker_payload(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    arm: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    if (
        payload.get("artifact_type")
        != "compact_occupancy_refinement_factorial_iter3_visualization_worker_v1"
        or payload.get("status") != "PASS"
        or payload.get("arm") != arm
        or payload.get("training_seed") != REQUIRED_SEED
        or payload.get("plan_sha256") != sha256_file(PLAN)
        or payload.get("result_sha256") != plan["result"]["sha256"]
        or payload.get("source_aggregate_sha256") != plan["sources"]["aggregate_sha256"]
        or payload.get("ply") != plan["plys"][arm]
        or payload.get("camera_semantic_sha256") != plan["cameras"]["semantic_sha256"]
        or payload.get("render_contract")
        != {
            "backend": EXPECTED_BACKEND,
            "device": "cuda",
            "rasterizer": "gsplat",
            "packed": False,
            "antialiased": False,
            "native_resolution": [NATIVE_WIDTH, NATIVE_HEIGHT],
        }
    ):
        raise ProtocolInvalid(f"visualization worker {arm} binding differs")
    if (
        payload.get("rgb_denial", {}).get("passed") is not True
        or payload["rgb_denial"].get("source_rgb_or_dataset_open_attempts") != 0
    ):
        raise ProtocolInvalid(f"visualization worker {arm} lacks RGB-denial evidence")
    images = payload.get("images")
    if not isinstance(images, list) or [record.get("view_id") for record in images] != list(VIEWS):
        raise ProtocolInvalid(f"visualization worker {arm} image order differs")
    from PIL import Image

    for record in images:
        path = _ordinary_repo_file(root, record.get("path"), label=f"arm {arm} PNG")
        expected_path = (
            PurePosixPath(plan["outputs"]["directory"]) / f"arm_{arm}" / f"{record['view_id']}.png"
        ).as_posix()
        if (
            record.get("path") != expected_path
            or record.get("dimensions") != [NATIVE_WIDTH, NATIVE_HEIGHT]
            or record.get("backend") != EXPECTED_BACKEND
            or not str(record.get("device", "")).startswith("cuda")
            or not _is_sha256(record.get("sha256"))
            or not _is_sha256(record.get("color_tensor_sha256"))
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ProtocolInvalid(f"visualization worker {arm} PNG binding differs")
        with Image.open(path) as decoded:
            dimensions = list(decoded.size)
            decoded.verify()
        if dimensions != [NATIVE_WIDTH, NATIVE_HEIGHT]:
            raise ProtocolInvalid(f"visualization worker {arm} PNG dimensions differ")
    extension = payload.get("loaded_gsplat_extension")
    extension_path = (
        None if not isinstance(extension, Mapping) else Path(str(extension.get("path")))
    )
    if (
        not isinstance(extension, Mapping)
        or set(extension) != {"path", "bytes", "sha256"}
        or not _is_sha256(extension.get("sha256"))
        or extension_path is None
        or not extension_path.is_absolute()
        or extension_path.resolve() != extension_path
        or not extension_path.is_file()
        or extension_path.suffix not in {".so", ".pyd"}
        or isinstance(extension.get("bytes"), bool)
        or not isinstance(extension.get("bytes"), int)
        or extension_path.stat().st_size != extension["bytes"]
        or sha256_file(extension_path) != extension["sha256"]
    ):
        raise ProtocolInvalid(f"visualization worker {arm} extension binding differs")
    package = payload.get("gsplat_package")
    if (
        not isinstance(package, Mapping)
        or not isinstance(package.get("version"), str)
        or not Path(str(package.get("module_file"))).is_absolute()
        or not Path(str(package.get("module_root"))).is_absolute()
        or isinstance(package.get("source_file_count"), bool)
        or not isinstance(package.get("source_file_count"), int)
        or package["source_file_count"] <= 0
        or not _is_sha256(package.get("source_manifest_sha256"))
    ):
        raise ProtocolInvalid(f"visualization worker {arm} gsplat package binding differs")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ProtocolInvalid(f"visualization worker {arm} runtime is absent")
    runtime_without_hash = dict(runtime)
    runtime_hash = runtime_without_hash.pop("binding_sha256", None)
    if (
        runtime_hash != canonical_hash(runtime_without_hash)
        or runtime.get("ld_preload_at_startup") != plan["preload"]["requested_path"]
        or runtime.get("preload") != plan["preload"]
    ):
        raise ProtocolInvalid(f"visualization worker {arm} runtime hash differs")
    return dict(payload)


def _font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def build_contact_sheet(
    path: Path,
    workers: Mapping[str, Mapping[str, Any]],
    *,
    root: Path = ROOT,
    thumbnail_width: int = THUMBNAIL_WIDTH,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    if set(workers) != set(ARMS) or thumbnail_width < 16:
        raise ProtocolInvalid("contact-sheet arm set or thumbnail width differs")
    thumbnail_height = round(NATIVE_HEIGHT * thumbnail_width / NATIVE_WIDTH)
    margin = 24
    row_label_width = 112
    gap = 12
    header_height = 62
    caption_height = 30
    row_height = thumbnail_height + caption_height + gap
    width = margin * 2 + row_label_width + len(ARMS) * thumbnail_width + (len(ARMS) - 1) * gap
    height = margin * 2 + header_height + len(VIEWS) * row_height
    canvas = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 4),
        "Compact occupancy refinement — exact full-resolution gsplat renders",
        font=_font(22, bold=True),
        fill=(25, 31, 42),
    )
    for column, arm in enumerate(ARMS):
        x = margin + row_label_width + column * (thumbnail_width + gap)
        draw.text(
            (x + thumbnail_width // 2, margin + 29),
            f"Arm {arm}",
            font=_font(19, bold=True),
            fill=(25, 31, 42),
            anchor="ma",
        )
    by_arm = {arm: {record["view_id"]: record for record in workers[arm]["images"]} for arm in ARMS}
    for row, view_id in enumerate(VIEWS):
        y = margin + header_height + row * row_height
        draw.text(
            (margin + row_label_width - 12, y + thumbnail_height // 2),
            view_id,
            font=_font(17, bold=True),
            fill=(55, 65, 81),
            anchor="rm",
        )
        for column, arm in enumerate(ARMS):
            record = by_arm[arm].get(view_id)
            if record is None:
                raise ProtocolInvalid(f"contact sheet lacks arm {arm} view {view_id}")
            source = _ordinary_repo_file(root, record["path"], label="contact source PNG")
            if sha256_file(source) != record["sha256"]:
                raise ProtocolInvalid("contact source PNG changed")
            with Image.open(source) as opened:
                if opened.size != (NATIVE_WIDTH, NATIVE_HEIGHT):
                    raise ProtocolInvalid("contact source is not a native render")
                tile = opened.convert("RGB").resize(
                    (thumbnail_width, thumbnail_height), Image.Resampling.LANCZOS
                )
            x = margin + row_label_width + column * (thumbnail_width + gap)
            canvas.paste(tile, (x, y))
            draw.rectangle(
                (x, y, x + thumbnail_width - 1, y + thumbnail_height - 1),
                outline=(190, 196, 207),
                width=1,
            )
            draw.text(
                (x + thumbnail_width // 2, y + thumbnail_height + 4),
                f"{view_id} · {NATIVE_WIDTH}×{NATIVE_HEIGHT} source",
                font=_font(13),
                fill=(75, 85, 99),
                anchor="ma",
            )
    with path.open("xb") as stream:
        canvas.save(stream, format="PNG", optimize=False)
        stream.flush()
        os.fsync(stream.fileno())
    with Image.open(path) as decoded:
        dimensions = list(decoded.size)
        decoded.verify()
    return {
        "path": display_path(path, root=root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dimensions": dimensions,
        "columns": list(ARMS),
        "rows": list(VIEWS),
        "thumbnail_dimensions": [thumbnail_width, thumbnail_height],
        "source_panels_are_native_resolution": True,
    }


def _worker_command(arm: str, plan_sha256: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--plan",
        str(PLAN),
        "--plan-sha256",
        plan_sha256,
        "--arm",
        arm,
    ]


def _run_parent() -> int:
    # Crucially, every read-only precondition is checked before OUTPUT_DIR is created.
    if os.path.lexists(OUTPUT_DIR):
        raise FileExistsError(f"append-only visualization namespace is consumed: {OUTPUT_DIR}")
    plan = build_plan()
    if not RUN_DIR.is_dir():
        raise ProtocolInvalid("official PASS RESULT lacks its bound run directory")
    OUTPUT_DIR.mkdir(exist_ok=False)
    plan_sha = exclusive_json(PLAN, plan)
    processes = []
    workers: dict[str, dict[str, Any]] = {}
    try:
        for arm in ARMS:
            command = _worker_command(arm, plan_sha)
            environment = dict(os.environ)
            environment["LD_PRELOAD"] = str(PRELOAD)
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=WORKER_TIMEOUT_SECONDS,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise ProtocolInvalid(f"visualization worker {arm} timed out") from error
            worker_path = OUTPUT_DIR / f"worker_{arm}.json"
            processes.append(
                {
                    "arm": arm,
                    "command": command,
                    "timeout_seconds": WORKER_TIMEOUT_SECONDS,
                    "elapsed_seconds": time.perf_counter() - started,
                    "returncode": completed.returncode,
                    "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                }
            )
            if completed.returncode != 0 or not worker_path.is_file():
                raise ProtocolInvalid(
                    f"visualization worker {arm} failed: {completed.stderr[-1000:]}"
                )
            payload = strict_json(worker_path)
            workers[arm] = validate_worker_payload(payload, plan=plan, arm=arm)

        extensions = {
            canonical_hash(worker["loaded_gsplat_extension"]) for worker in workers.values()
        }
        packages = {canonical_hash(worker["gsplat_package"]) for worker in workers.values()}
        if len(extensions) != 1 or len(packages) != 1:
            raise ProtocolInvalid("fresh workers resolved different gsplat package/extension bytes")
        parent_guard = ImageAccessGuard(OUTPUT_DIR)
        with parent_guard:
            contact = build_contact_sheet(OUTPUT_DIR / "CONTACT_SHEET.png", workers)
        denial = parent_guard.record()
        if not denial["passed"]:
            raise ProtocolInvalid("contact-sheet parent attempted source RGB/dataset access")
        _verify_plan_live(plan, plan_sha)
        for arm in ARMS:
            validate_worker_payload(workers[arm], plan=plan, arm=arm)
        pngs = {
            f"{arm}/{record['view_id']}": {
                key: record[key]
                for key in ("path", "sha256", "bytes", "dimensions", "color_tensor_sha256")
            }
            for arm in ARMS
            for record in workers[arm]["images"]
        }
        receipt = {
            "artifact_type": (
                "compact_occupancy_refinement_factorial_iter3_gsplat_visualization_receipt_v1"
            ),
            "timestamp_utc": timestamp_utc(),
            "status": "PASS",
            "decision_bearing": False,
            "scope": "post-result visualization only; no RGB target, refit, metric, or claim",
            "plan": {"path": display_path(PLAN), "sha256": plan_sha},
            "result": plan["result"],
            "sources": plan["sources"],
            "bundle": {
                "path": plan["bundle"]["path"],
                "aggregate_sha256": plan["bundle"]["aggregate_sha256"],
                "camera_semantic_sha256": plan["cameras"]["semantic_sha256"],
            },
            "plys": plan["plys"],
            "pngs": pngs,
            "contact_sheet": contact,
            "render_contract": plan["render_contract"],
            "gsplat": {
                "package": workers[ARMS[0]]["gsplat_package"],
                "loaded_extension": workers[ARMS[0]]["loaded_gsplat_extension"],
                "identical_across_fresh_workers": True,
            },
            "runtime_bindings": {
                arm: {
                    "binding_sha256": workers[arm]["runtime"]["binding_sha256"],
                    "runtime": workers[arm]["runtime"],
                }
                for arm in ARMS
            },
            "worker_records": {
                arm: {
                    "path": display_path(OUTPUT_DIR / f"worker_{arm}.json"),
                    "sha256": sha256_file(OUTPUT_DIR / f"worker_{arm}.json"),
                }
                for arm in ARMS
            },
            "worker_processes": processes,
            "rgb_denial": {
                "render_workers": {arm: workers[arm]["rgb_denial"] for arm in ARMS},
                "contact_sheet_parent": denial,
                "source_rgb_used_for_rendering": False,
            },
            "live_viewer": "SEPARATE_POST_RESULT_STEP_NOT_PERFORMED_BY_THIS_UTILITY",
        }
        digest = exclusive_json(RECEIPT, receipt)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "receipt": str(RECEIPT),
                    "sha256": digest,
                    "contact_sheet": contact,
                },
                indent=2,
            )
        )
        return 0
    except BaseException as error:
        failure = _bounded_failure(error, stage="visualization_parent")
        failure.update(
            {
                "plan_sha256": plan_sha,
                "result_sha256": plan["result"]["sha256"],
                "source_aggregate_sha256": plan["sources"]["aggregate_sha256"],
                "worker_processes": processes,
            }
        )
        if not os.path.lexists(FAILURE):
            exclusive_json(FAILURE, failure)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?", choices=("run", "_worker"), default="run")
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args(argv)
    if args.operation == "_worker" and not all((args.plan, args.plan_sha256, args.arm)):
        parser.error("_worker requires --plan, --plan-sha256, and --arm")
    if args.operation == "run" and any((args.plan, args.plan_sha256, args.arm)):
        parser.error("run accepts no worker-only arguments")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operation == "_worker":
        return worker_entry(args)
    return _run_parent()


if __name__ == "__main__":
    raise SystemExit(main())
