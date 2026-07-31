#!/usr/bin/env python3
"""Produce sealed native-additive compact bundles for the analytic-objective task.

The coordinator launches one fresh worker process per calibrated view.  A worker decodes only its
named RGB/mask pair, performs the frozen native Stage-1 fit, writes one integrity-checked
``.rtgsv`` file, and exits.  The coordinator never retains image tensors and publishes the frame
manifest only after every worker receipt and compact view passes strict verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtgs.data.compact_views import (
    CompactDataset,
    CompactView,
    file_sha256,
    save_compact_view,
    write_compact_dataset_manifest,
)
from rtgs.image2gs.fit import FitConfig, _crop_to_mask, fit_image
from rtgs.image2gs.native_observation import native_gaussians_to_observation

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260730_additive_analytic_objective_stage_frames00008_00009"
DEFAULT_TASK = ROOT / "experiments" / "tasks" / f"{TASK_ID}.json"
PRODUCTION_SCHEMA = "rtgs.additive_native_bundle_production.v1"
RECEIPT_SCHEMA = "rtgs.additive_native_view_production.v1"
PRODUCTION_KEY = "additive_bundle_production"
SOURCE_FILES = (
    Path("scripts/experiments/prepare_additive_native_bundles.py"),
    Path("src/rtgs/core/gaussians2d.py"),
    Path("src/rtgs/core/observation2d.py"),
    Path("src/rtgs/data/calibrated.py"),
    Path("src/rtgs/data/compact_views.py"),
    Path("src/rtgs/image2gs/fit.py"),
    Path("src/rtgs/image2gs/native_observation.py"),
    Path("src/rtgs/image2gs/renderer2d.py"),
    Path("src/rtgs/image2gs/cuda_backend.py"),
    Path("src/rtgs/image2gs/cuda/renderer2d_ext.cpp"),
    Path("src/rtgs/image2gs/cuda/renderer2d_ext.cu"),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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


def _repository_path(value: str, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a safe repository-relative path")
    resolved = (ROOT / relative).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return resolved


def _task(path: Path, *, require_draft: bool = False) -> dict[str, Any]:
    task = _load_json(path.resolve())
    if task.get("task_id") != TASK_ID:
        raise ValueError(f"bundle production is bound to task {TASK_ID}")
    if task.get("status") not in {"draft", "ready"}:
        raise ValueError("additive bundle verification requires a draft or ready task")
    if require_draft and task.get("status") != "draft":
        raise ValueError("additive bundle production is allowed only while the task is draft")
    return task


def _production_config(task: dict[str, Any]) -> dict[str, Any]:
    frozen = task.get("frozen_configuration")
    if not isinstance(frozen, dict):
        raise ValueError("task has no frozen_configuration")
    production = frozen.get(PRODUCTION_KEY)
    expected = {
        "script",
        "device",
        "downscale",
        "seed_base",
        "seed_stride_per_dataset",
        "per_view_process_isolation",
        "field_semantics",
        "fit_config",
    }
    if not isinstance(production, dict) or set(production) != expected:
        raise ValueError(f"{PRODUCTION_KEY} must contain exactly {sorted(expected)}")
    if production["script"] != "scripts/experiments/prepare_additive_native_bundles.py":
        raise ValueError("task names the wrong additive bundle production script")
    if production["device"] != "cuda" or production["downscale"] != 1:
        raise ValueError("production requires full-resolution CUDA fitting")
    if production["per_view_process_isolation"] is not True:
        raise ValueError("production requires fresh per-view worker processes")
    if (
        not isinstance(production["seed_base"], int)
        or isinstance(production["seed_base"], bool)
        or production["seed_base"] < 0
    ):
        raise ValueError("seed_base must be a non-negative integer")
    if (
        not isinstance(production["seed_stride_per_dataset"], int)
        or isinstance(production["seed_stride_per_dataset"], bool)
        or production["seed_stride_per_dataset"] < 100
    ):
        raise ValueError("seed_stride_per_dataset must be an integer of at least 100")
    semantics = production["field_semantics"]
    if semantics != {
        "provider": "native",
        "blend_mode": "additive",
        "sigma_cutoff": 12.0**0.5,
        "support_fade_alpha": 0.0,
        "aa_dilation": 0.0,
    }:
        raise ValueError("field_semantics do not match the native additive adapter")
    fit_config = production["fit_config"]
    if not isinstance(fit_config, dict):
        raise ValueError("fit_config must be an object")
    effective = asdict(FitConfig(**fit_config))
    if effective != fit_config:
        raise ValueError("fit_config must record every effective FitConfig field")
    required_fit = {
        "n_gaussians": 640,
        "max_gaussians": 640,
        "iterations": 100,
        "backend": "native",
        "adaptive_density": False,
        "native_renderer": "cuda",
        "batch_views": False,
        "init_strategy": "gradient",
        "appearance_parameterization": "weight_color_9p",
        "freeze_geometry": False,
        "pool": False,
        "mask_coverage_weight": 0.0,
        "convergence_patience": 0,
    }
    mismatches = {
        key: (fit_config.get(key), expected_value)
        for key, expected_value in required_fit.items()
        if fit_config.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"fit_config violates the frozen production subset: {mismatches}")
    return production


def _datasets(task: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = task.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("task datasets must be a non-empty list")
    result: list[dict[str, Any]] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("task dataset entries must be objects")
        dataset_id = dataset.get("id")
        production_manifest = dataset.get("production_manifest")
        if not isinstance(dataset_id, str) or not isinstance(production_manifest, str):
            raise ValueError("each dataset requires id and production_manifest")
        compact_manifest = _repository_path(
            dataset["compact_manifest"],
            label=f"{dataset_id}.compact_manifest",
        )
        production_path = _repository_path(
            production_manifest,
            label=f"{dataset_id}.production_manifest",
        )
        if compact_manifest.parent != production_path.parent:
            raise ValueError("compact and production manifests must share one directory")
        result.append(dataset)
    return result


def _dataset(task: dict[str, Any], dataset_id: str) -> tuple[int, dict[str, Any]]:
    for index, dataset in enumerate(_datasets(task)):
        if dataset["id"] == dataset_id:
            return index, dataset
    raise ValueError(f"unknown dataset id: {dataset_id}")


def _view_ids(task: dict[str, Any], dataset_id: str) -> list[str]:
    split = task["splits"][dataset_id]
    values = split["train"] + split["heldout"]
    if len(values) != len(set(values)):
        raise ValueError(f"{dataset_id} split contains duplicate views")
    return sorted(values)


def _view_seed(
    production: dict[str, Any],
    *,
    dataset_index: int,
    view_index: int,
) -> int:
    return (
        int(production["seed_base"])
        + dataset_index * int(production["seed_stride_per_dataset"])
        + view_index
    )


def _source_binding() -> dict[str, Any]:
    hashes = {path.as_posix(): file_sha256(ROOT / path) for path in SOURCE_FILES}
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    return {
        "git_revision": revision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "files": hashes,
        "aggregate_sha256": _digest(hashes),
    }


def _fit_contract(production: dict[str, Any]) -> dict[str, Any]:
    return {
        "fit_config": production["fit_config"],
        "downscale": production["downscale"],
        "field_semantics": production["field_semantics"],
        "preprocessing": {
            "rgb": "calibrated_bilinear_undistort",
            "alpha": "calibrated_nearest_undistort_threshold_gt_0.5",
            "fit_window": "tight_foreground_aabb",
            "loss": "native_mask_weighted_mse_on_zeroed_crop",
        },
    }


def _paths(
    task: dict[str, Any],
    dataset_id: str,
    view_id: str,
) -> tuple[Path, Path, Path]:
    _, dataset = _dataset(task, dataset_id)
    output_directory = _repository_path(
        dataset["compact_manifest"],
        label=f"{dataset_id}.compact_manifest",
    ).parent
    return (
        output_directory / f"{view_id}.rtgsv",
        output_directory / "receipts" / f"{view_id}.json",
        output_directory,
    )


def _input_paths(dataset: dict[str, Any], view_id: str) -> tuple[Path, Path, Path]:
    frame = _repository_path(dataset["frame_path"], label=f"{dataset['id']}.frame_path")
    calibration = _repository_path(
        dataset["calibration"],
        label=f"{dataset['id']}.calibration",
    )
    return (
        frame / "rgb" / f"{view_id}.jpg",
        frame / "mask" / f"mask_{view_id}.png",
        calibration,
    )


def _extension_record() -> dict[str, Any]:
    from rtgs.image2gs import cuda_backend

    extension = cuda_backend._EXT
    if extension is None:
        raise RuntimeError("native CUDA renderer extension did not load")
    path = Path(extension.__file__).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _worker(
    task_path: Path,
    dataset_id: str,
    view_id: str,
    expected_seed: int,
) -> int:
    task = _task(task_path, require_draft=True)
    production = _production_config(task)
    dataset_index, dataset = _dataset(task, dataset_id)
    view_ids = _view_ids(task, dataset_id)
    if view_id not in view_ids:
        raise ValueError(f"{view_id} is outside the frozen split")
    view_index = view_ids.index(view_id)
    seed = _view_seed(
        production,
        dataset_index=dataset_index,
        view_index=view_index,
    )
    if seed != expected_seed:
        raise ValueError("worker seed differs from the frozen derivation")
    output_path, receipt_path, _ = _paths(task, dataset_id, view_id)
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError(
            f"refusing to overwrite partial production for {dataset_id}/{view_id}"
        )

    binding = _source_binding()
    fit_contract = _fit_contract(production)
    fit_config_digest = _digest(fit_contract)
    rgb_path, mask_path, calibration_path = _input_paths(dataset, view_id)
    input_record = {
        "rgb": {
            "path": rgb_path.relative_to(ROOT).as_posix(),
            "bytes": rgb_path.stat().st_size,
            "sha256": file_sha256(rgb_path),
        },
        "mask": {
            "path": mask_path.relative_to(ROOT).as_posix(),
            "bytes": mask_path.stat().st_size,
            "sha256": file_sha256(mask_path),
        },
        "calibration": {
            "path": calibration_path.relative_to(ROOT).as_posix(),
            "bytes": calibration_path.stat().st_size,
            "sha256": file_sha256(calibration_path),
        },
    }

    try:
        from rtgs.data.calibrated import load_calibrated_scene

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        scene = load_calibrated_scene(
            _repository_path(dataset["frame_path"], label=f"{dataset_id}.frame_path"),
            calibration_path=calibration_path,
            downscale=production["downscale"],
            test_every=0,
            load_masks=True,
            undistort=True,
            view_ids=[view_id],
        )
        if scene.view_names != [view_id] or scene.masks is None or len(scene.images) != 1:
            raise RuntimeError("worker did not isolate exactly one calibrated RGB/mask view")
        image = scene.images[0]
        mask = scene.masks[0] > 0.5
        camera = scene.cameras[0]
        crop, mask_crop, offset = _crop_to_mask(image, mask)
        fit_window = (
            int(offset[0].item()),
            int(offset[1].item()),
            int(crop.shape[1]),
            int(crop.shape[0]),
        )
        config = FitConfig(**production["fit_config"])
        device = torch.device(production["device"])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        fitted, history = fit_image(
            crop.to(device),
            config,
            seed=seed,
            mask=mask_crop.to(device),
        )
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        fitted = fitted.to("cpu")
        observation = native_gaussians_to_observation(
            fitted,
            canvas_size=(camera.height, camera.width),
            fit_window=fit_window,
            view_id=view_id,
            n_init=config.n_gaussians,
            producer_version=binding["git_revision"],
            producer_source_digest=binding["aggregate_sha256"],
            fit_config_digest=fit_config_digest,
        )
        save_compact_view(
            output_path,
            observation,
            camera,
            calibration_sha256=input_record["calibration"]["sha256"],
            source_rgb_name=rgb_path.name,
            source_rgb_sha256=input_record["rgb"]["sha256"],
            alpha_crop=mask_crop,
            source_mask_name=mask_path.name,
            source_mask_sha256=input_record["mask"]["sha256"],
        )
        loaded = CompactView.load(output_path)
        if not (
            loaded.view_id == view_id
            and loaded.observation.provider == "native"
            and loaded.observation.blend_mode == "additive"
            and loaded.observation.support_fade_alpha == 0.0
            and loaded.observation.aa_dilation == 0.0
            and loaded.observation.fit_config_digest == fit_config_digest
            and loaded.alpha is not None
        ):
            raise RuntimeError(
                "strict compact reload differs from the additive production contract"
            )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS",
            "task_id": TASK_ID,
            "dataset_id": dataset_id,
            "view_id": view_id,
            "view_index": view_index,
            "seed": seed,
            "input": input_record,
            "source_binding": binding,
            "fit_contract": fit_contract,
            "fit_config_digest": fit_config_digest,
            "fit_window": list(fit_window),
            "canvas_size": [camera.height, camera.width],
            "field": {
                "provider": loaded.observation.provider,
                "blend_mode": loaded.observation.blend_mode,
                "sigma_cutoff": loaded.observation.sigma_cutoff,
                "support_fade_alpha": loaded.observation.support_fade_alpha,
                "aa_dilation": loaded.observation.aa_dilation,
                "n_gaussians": loaded.observation.n,
                "n_init": loaded.observation.n_init,
            },
            "output": {
                "path": output_path.relative_to(ROOT).as_posix(),
                "bytes": loaded.bytes,
                "sha256": loaded.sha256,
            },
            "history": history,
            "runtime": {
                "wall_seconds": wall_seconds,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
            "environment": {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "cuda_extension": _extension_record(),
                "determinism": {
                    "seeded_python_numpy_torch_cuda": True,
                    "fresh_process_per_view": True,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "bit_exact_cuda_replay_guaranteed": False,
                    "boundary": (
                        "the native CUDA renderer uses atomic accumulation; inputs, seeds, "
                        "source, binary, and order are frozen, but cross-run bit identity is "
                        "not claimed"
                    ),
                },
            },
        }
        _write_json_atomic(receipt_path, receipt)
        print(
            f"{dataset_id}/{view_id}: PASS ({loaded.bytes} bytes, {wall_seconds:.1f}s)",
            flush=True,
        )
        return 0
    except BaseException as error:
        output_path.unlink(missing_ok=True)
        failure = {
            "schema": RECEIPT_SCHEMA,
            "status": "FAIL",
            "task_id": TASK_ID,
            "dataset_id": dataset_id,
            "view_id": view_id,
            "seed": seed,
            "input": input_record,
            "source_binding": binding,
            "fit_contract": fit_contract,
            "fit_config_digest": fit_config_digest,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(receipt_path, failure)
        raise


def _load_pass_receipt(
    task: dict[str, Any],
    production: dict[str, Any],
    dataset_index: int,
    dataset_id: str,
    view_id: str,
    view_index: int,
) -> dict[str, Any]:
    output_path, receipt_path, _ = _paths(task, dataset_id, view_id)
    receipt = _load_json(receipt_path)
    expected_seed = _view_seed(
        production,
        dataset_index=dataset_index,
        view_index=view_index,
    )
    if not (
        receipt.get("schema") == RECEIPT_SCHEMA
        and receipt.get("status") == "PASS"
        and receipt.get("task_id") == TASK_ID
        and receipt.get("dataset_id") == dataset_id
        and receipt.get("view_id") == view_id
        and receipt.get("view_index") == view_index
        and receipt.get("seed") == expected_seed
        and receipt.get("fit_contract") == _fit_contract(production)
        and receipt.get("fit_config_digest") == _digest(_fit_contract(production))
    ):
        raise ValueError(f"invalid production receipt: {receipt_path}")
    output = receipt.get("output")
    if (
        not isinstance(output, dict)
        or output.get("path") != output_path.relative_to(ROOT).as_posix()
    ):
        raise ValueError(f"receipt output path differs for {dataset_id}/{view_id}")
    if (
        not output_path.is_file()
        or output_path.stat().st_size != output.get("bytes")
        or file_sha256(output_path) != output.get("sha256")
    ):
        raise ValueError(f"receipt output bytes differ for {dataset_id}/{view_id}")
    return receipt


def _existing_view_is_reusable(
    task: dict[str, Any],
    production: dict[str, Any],
    dataset_index: int,
    dataset_id: str,
    view_id: str,
    view_index: int,
) -> bool:
    output_path, receipt_path, _ = _paths(task, dataset_id, view_id)
    if not output_path.exists() and not receipt_path.exists():
        return False
    try:
        receipt = _load_pass_receipt(
            task,
            production,
            dataset_index,
            dataset_id,
            view_id,
            view_index,
        )
        loaded = CompactView.load(output_path)
    except (OSError, ValueError):
        return False
    return (
        receipt["source_binding"] == _source_binding()
        and loaded.observation.provider == "native"
        and loaded.observation.blend_mode == "additive"
        and loaded.observation.support_fade_alpha == 0.0
        and loaded.observation.fit_config_digest == _digest(_fit_contract(production))
    )


def _production_manifest(
    task: dict[str, Any],
    production: dict[str, Any],
    dataset_index: int,
    dataset: dict[str, Any],
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    compact_manifest = _repository_path(
        dataset["compact_manifest"],
        label=f"{dataset['id']}.compact_manifest",
    )
    source_binding = _source_binding()
    if any(receipt["source_binding"] != source_binding for receipt in receipts):
        raise RuntimeError("source binding changed during additive bundle production")
    calibration_records = {receipt["input"]["calibration"]["sha256"] for receipt in receipts}
    if len(calibration_records) != 1:
        raise RuntimeError("view workers did not share one calibration digest")
    manifest: dict[str, Any] = {
        "schema": PRODUCTION_SCHEMA,
        "task_id": TASK_ID,
        "dataset_id": dataset["id"],
        "dataset_index": dataset_index,
        "frame_path": dataset["frame_path"],
        "compact_manifest": {
            "path": compact_manifest.relative_to(ROOT).as_posix(),
            "bytes": compact_manifest.stat().st_size,
            "sha256": file_sha256(compact_manifest),
        },
        "source_binding": source_binding,
        "fit_contract": _fit_contract(production),
        "fit_config_digest": _digest(_fit_contract(production)),
        "seed_policy": {
            "algorithm": "seed_base + dataset_index * seed_stride_per_dataset + view_index",
            "seed_base": production["seed_base"],
            "seed_stride_per_dataset": production["seed_stride_per_dataset"],
            "view_order": [receipt["view_id"] for receipt in receipts],
        },
        "per_view_process_isolation": True,
        "determinism_boundary": {
            "frozen_inputs_config_seeds_source_binary_and_order": True,
            "bit_exact_cuda_replay_guaranteed": False,
            "reason": "native CUDA rendering uses atomic accumulation",
        },
        "views": receipts,
    }
    manifest["semantic_digest"] = _digest(manifest)
    return manifest


def _publish_dataset(
    task: dict[str, Any],
    production: dict[str, Any],
    dataset_index: int,
    dataset: dict[str, Any],
) -> None:
    dataset_id = dataset["id"]
    view_ids = _view_ids(task, dataset_id)
    receipts = [
        _load_pass_receipt(
            task,
            production,
            dataset_index,
            dataset_id,
            view_id,
            view_index,
        )
        for view_index, view_id in enumerate(view_ids)
    ]
    output_directory = _repository_path(
        dataset["compact_manifest"],
        label=f"{dataset_id}.compact_manifest",
    ).parent
    compact_manifest = output_directory / "manifest.json"
    if compact_manifest.exists():
        loaded = CompactDataset.load(output_directory)
        if [view.view_id for view in loaded.views] != view_ids:
            raise ValueError(f"{dataset_id} existing compact manifest has the wrong view order")
    else:
        calibration_sha256 = receipts[0]["input"]["calibration"]["sha256"]
        write_compact_dataset_manifest(
            output_directory,
            name=f"{dataset_id}_additive",
            calibration_sha256=calibration_sha256,
            view_paths=[output_directory / f"{view_id}.rtgsv" for view_id in view_ids],
            bounds_hint=None,
        )
    production_path = _repository_path(
        dataset["production_manifest"],
        label=f"{dataset_id}.production_manifest",
    )
    value = _production_manifest(
        task,
        production,
        dataset_index,
        dataset,
        receipts,
    )
    if production_path.exists():
        if _load_json(production_path) != value:
            raise FileExistsError(
                f"refusing to overwrite changed production manifest: {production_path}"
            )
    else:
        _write_json_atomic(production_path, value)


def _produce(task_path: Path) -> None:
    task = _task(task_path, require_draft=True)
    production = _production_config(task)
    binding = _source_binding()
    print(
        f"source {binding['aggregate_sha256']} · "
        f"{production['fit_config']['n_gaussians']} Gaussians · "
        f"{production['fit_config']['iterations']} iterations",
        flush=True,
    )
    for dataset_index, dataset in enumerate(_datasets(task)):
        dataset_id = dataset["id"]
        view_ids = _view_ids(task, dataset_id)
        for view_index, view_id in enumerate(view_ids):
            if _existing_view_is_reusable(
                task,
                production,
                dataset_index,
                dataset_id,
                view_id,
                view_index,
            ):
                print(f"{dataset_id}/{view_id}: verified existing", flush=True)
                continue
            output_path, receipt_path, output_directory = _paths(task, dataset_id, view_id)
            if output_path.exists():
                raise FileExistsError(f"unverified output blocks production: {output_path}")
            if receipt_path.exists():
                failed = _load_json(receipt_path)
                if failed.get("status") != "FAIL":
                    raise FileExistsError(f"unverified receipt blocks production: {receipt_path}")
                receipt_path.unlink()
            output_directory.mkdir(parents=True, exist_ok=True)
            seed = _view_seed(
                production,
                dataset_index=dataset_index,
                view_index=view_index,
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--task",
                str(task_path.resolve()),
                "--dataset-id",
                dataset_id,
                "--view-id",
                view_id,
                "--seed",
                str(seed),
            ]
            print(
                f"{dataset_id}/{view_id}: fitting ({view_index + 1}/{len(view_ids)}, seed {seed})",
                flush=True,
            )
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker failed for {dataset_id}/{view_id} with {completed.returncode}"
                )
        _publish_dataset(task, production, dataset_index, dataset)
    summary = _verify(task_path)
    print(json.dumps(summary, indent=2), flush=True)


def _verify(task_path: Path) -> dict[str, Any]:
    task = _task(task_path)
    production = _production_config(task)
    binding = _source_binding()
    dataset_summaries: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(_datasets(task)):
        dataset_id = dataset["id"]
        view_ids = _view_ids(task, dataset_id)
        output_directory = _repository_path(
            dataset["compact_manifest"],
            label=f"{dataset_id}.compact_manifest",
        ).parent
        compact = CompactDataset.load(output_directory)
        if [view.view_id for view in compact.views] != view_ids:
            raise ValueError(f"{dataset_id} compact view order differs from the frozen split")
        receipts = [
            _load_pass_receipt(
                task,
                production,
                dataset_index,
                dataset_id,
                view_id,
                view_index,
            )
            for view_index, view_id in enumerate(view_ids)
        ]
        expected_manifest = _production_manifest(
            task,
            production,
            dataset_index,
            dataset,
            receipts,
        )
        production_path = _repository_path(
            dataset["production_manifest"],
            label=f"{dataset_id}.production_manifest",
        )
        if _load_json(production_path) != expected_manifest:
            raise ValueError(f"{dataset_id} production manifest differs from current bytes")
        for view, receipt in zip(compact.views, receipts, strict=True):
            field = view.observation
            if not (
                field.provider == "native"
                and field.blend_mode == "additive"
                and field.support_fade_alpha == 0.0
                and field.aa_dilation == 0.0
                and field.fit_config_digest == receipt["fit_config_digest"]
                and view.alpha is not None
            ):
                raise ValueError(f"{dataset_id}/{view.view_id} field semantics differ")
            rgb_path, mask_path, calibration_path = _input_paths(dataset, view.view_id)
            current_inputs = {
                "rgb": file_sha256(rgb_path),
                "mask": file_sha256(mask_path),
                "calibration": file_sha256(calibration_path),
            }
            stored_inputs = {
                key: receipt["input"][key]["sha256"] for key in ("rgb", "mask", "calibration")
            }
            if current_inputs != stored_inputs:
                raise ValueError(f"{dataset_id}/{view.view_id} source input hashes drifted")
        if any(receipt["source_binding"] != binding for receipt in receipts):
            raise ValueError(f"{dataset_id} source binding differs from current production code")
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "views": len(compact.views),
                "gaussians": sum(view.observation.n for view in compact.views),
                "bytes": sum(view.bytes for view in compact.views),
                "manifest_sha256": file_sha256(output_directory / "manifest.json"),
                "production_manifest_sha256": file_sha256(production_path),
            }
        )
    return {
        "status": "PASS",
        "task_id": TASK_ID,
        "source_binding": binding["aggregate_sha256"],
        "datasets": dataset_summaries,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("produce", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    worker = subparsers.add_parser("_worker")
    worker.add_argument("--task", type=Path, required=True)
    worker.add_argument("--dataset-id", required=True)
    worker.add_argument("--view-id", required=True)
    worker.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "produce":
        _produce(args.task)
        return 0
    if args.command == "verify":
        print(json.dumps(_verify(args.task), indent=2))
        return 0
    if args.command == "_worker":
        return _worker(args.task, args.dataset_id, args.view_id, args.seed)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
