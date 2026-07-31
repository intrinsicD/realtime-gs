#!/usr/bin/env python3
"""Produce the RTGS-007 native full-resolution additive Stage-1 bundles.

Each calibrated view is decoded and fitted in a fresh worker process.  The coordinator publishes
the compact manifest only after every 100k-row field, receipt, and direct-resolution QA render
passes strict reload and source-binding checks.  The larger per-view cap is read from the draft
protocol and passed explicitly; the repository-wide compact default remains unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
from PIL import Image

from rtgs.data.compact_views import (
    CompactDataset,
    CompactView,
    file_sha256,
    save_compact_view,
    write_compact_dataset_manifest,
)
from rtgs.image2gs.fit import FitConfig, _crop_to_mask, fit_image
from rtgs.image2gs.native_observation import native_gaussians_to_observation
from rtgs.image2gs.renderer2d import render_gaussians_2d

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260730_paper_three_path_fullres_stage_frames00008_00009"
DEFAULT_TASK = ROOT / "experiments/tasks" / f"{TASK_ID}.json"
CONFIG_KEY = "stage1_native_fullres"
VIEW_SCHEMA = "rtgs.native_fullres_view_production.v1"
PRODUCTION_SCHEMA = "rtgs.native_fullres_bundle_production.v1"
SOURCE_FILES = (
    Path("scripts/experiments/prepare_native_fullres_paper_bundles.py"),
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
    result = (ROOT / relative).resolve()
    result.relative_to(ROOT.resolve())
    return result


def _task(path: Path, *, require_draft: bool = False) -> dict[str, Any]:
    task = _load_json(path.resolve())
    if task.get("task_id") != TASK_ID:
        raise ValueError(f"production is bound to task {TASK_ID}")
    if task.get("status") not in {"draft", "ready"}:
        raise ValueError("bundle verification requires a draft or ready task")
    if require_draft and task.get("status") != "draft":
        raise ValueError("bundle production is allowed only while the task is draft")
    return task


def _config(task: dict[str, Any]) -> dict[str, Any]:
    frozen = task.get("frozen_configuration")
    value = None if not isinstance(frozen, dict) else frozen.get(CONFIG_KEY)
    expected = {
        "selection_basis",
        "script",
        "device",
        "downscale",
        "seed_base",
        "seed_stride_per_dataset",
        "parallel_workers",
        "per_view_process_isolation",
        "byte_cap",
        "field_semantics",
        "fit_config",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{CONFIG_KEY} must contain exactly {sorted(expected)}")
    if value["script"] != "scripts/experiments/prepare_native_fullres_paper_bundles.py":
        raise ValueError("task names the wrong Stage-1 production script")
    if value["device"] != "cuda" or value["downscale"] != 1:
        raise ValueError("Stage-1 production requires full-resolution CUDA fitting")
    if value["per_view_process_isolation"] is not True:
        raise ValueError("Stage-1 production requires fresh per-view processes")
    for name in ("seed_base", "seed_stride_per_dataset", "parallel_workers", "byte_cap"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if value["seed_stride_per_dataset"] < 100:
        raise ValueError("seed_stride_per_dataset must be at least 100")
    if value["parallel_workers"] > 4:
        raise ValueError("parallel_workers exceeds the bounded production limit")
    if value["field_semantics"] != {
        "provider": "native",
        "blend_mode": "additive",
        "sigma_cutoff": 12.0**0.5,
        "support_fade_alpha": 0.0,
        "aa_dilation": 0.0,
    }:
        raise ValueError("field semantics differ from the native additive adapter")
    fit = value["fit_config"]
    if not isinstance(fit, dict) or asdict(FitConfig(**fit)) != fit:
        raise ValueError("fit_config must record every effective FitConfig field")
    required = {
        "n_gaussians": 100_000,
        "max_gaussians": 100_000,
        "iterations": 2_000,
        "backend": "native",
        "adaptive_density": False,
        "native_renderer": "cuda",
        "batch_views": False,
        "init_strategy": "gradient",
        "appearance_parameterization": "weight_color_9p",
        "freeze_geometry": False,
        "pool": False,
        "convergence_patience": 0,
    }
    mismatches = {
        name: (fit.get(name), expected_value)
        for name, expected_value in required.items()
        if fit.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(f"fit_config violates the frozen production subset: {mismatches}")
    return value


def _datasets(task: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = task.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("task datasets must be a non-empty list")
    return datasets


def _dataset(
    task: dict[str, Any],
    dataset_id: str,
) -> tuple[int, dict[str, Any]]:
    for index, dataset in enumerate(_datasets(task)):
        if dataset.get("id") == dataset_id:
            return index, dataset
    raise ValueError(f"unknown dataset id: {dataset_id}")


def _view_ids(task: dict[str, Any], dataset_id: str) -> list[str]:
    split = task["splits"][dataset_id]
    values = split["train"] + split["heldout"]
    if len(values) != len(set(values)):
        raise ValueError(f"{dataset_id} split contains duplicate views")
    return sorted(values)


def _seed(config: dict[str, Any], dataset_index: int, view_index: int) -> int:
    return (
        int(config["seed_base"])
        + dataset_index * int(config["seed_stride_per_dataset"])
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


def _output_directory(dataset: dict[str, Any]) -> Path:
    return _repository_path(
        dataset["compact_manifest"],
        label=f"{dataset['id']}.compact_manifest",
    ).parent


def _paths(dataset: dict[str, Any], view_id: str) -> tuple[Path, Path, Path, Path]:
    output = _output_directory(dataset)
    return (
        output / f"{view_id}.rtgsv",
        output / "receipts" / f"{view_id}.json",
        output / "qa" / f"{view_id}_field_native_crop.png",
        output / "qa" / f"{view_id}_field_native_full_canvas.png",
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


def _pil_rgb(value: torch.Tensor) -> Image.Image:
    pixels = value.detach().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(pixels, mode="RGB")


def _save_qa(
    rendered: torch.Tensor,
    *,
    crop_path: Path,
    canvas_path: Path,
    canvas_size: tuple[int, int],
    offset: tuple[int, int],
) -> None:
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop = _pil_rgb(rendered)
    crop.save(crop_path)
    height, width = canvas_size
    canvas = Image.new("RGB", (width, height))
    canvas.paste(crop, offset)
    canvas.save(canvas_path)


def _worker(
    task_path: Path,
    dataset_id: str,
    view_id: str,
    expected_seed: int,
) -> int:
    task = _task(task_path, require_draft=True)
    config = _config(task)
    dataset_index, dataset = _dataset(task, dataset_id)
    view_ids = _view_ids(task, dataset_id)
    view_index = view_ids.index(view_id)
    seed = _seed(config, dataset_index, view_index)
    if seed != expected_seed:
        raise ValueError("worker seed differs from the frozen derivation")
    output_path, receipt_path, crop_path, canvas_path = _paths(dataset, view_id)
    if any(path.exists() for path in (output_path, receipt_path, crop_path, canvas_path)):
        raise FileExistsError(
            f"refusing to overwrite partial production for {dataset_id}/{view_id}"
        )

    binding = _source_binding()
    fit_config = FitConfig(**config["fit_config"])
    fit_digest = _digest(
        {
            "fit_config": config["fit_config"],
            "downscale": config["downscale"],
            "field_semantics": config["field_semantics"],
        }
    )
    rgb_path, mask_path, calibration_path = _input_paths(dataset, view_id)
    input_record = {
        "rgb": {
            "name": rgb_path.name,
            "bytes": rgb_path.stat().st_size,
            "sha256": file_sha256(rgb_path),
        },
        "mask": {
            "name": mask_path.name,
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
        scene = load_calibrated_scene(
            _repository_path(dataset["frame_path"], label=f"{dataset_id}.frame_path"),
            calibration_path=calibration_path,
            downscale=1,
            test_every=0,
            load_masks=True,
            undistort=True,
            view_ids=[view_id],
        )
        if scene.view_names != [view_id] or scene.masks is None:
            raise RuntimeError("worker did not isolate one calibrated RGB/mask view")
        image = scene.images[0]
        mask = scene.masks[0] > 0.5
        camera = scene.cameras[0]
        crop, mask_crop, offset_tensor = _crop_to_mask(image, mask)
        offset = (int(offset_tensor[0]), int(offset_tensor[1]))
        fit_window = (offset[0], offset[1], int(crop.shape[1]), int(crop.shape[0]))
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        fitted, history = fit_image(
            crop.to(device),
            fit_config,
            seed=seed,
            mask=mask_crop.to(device),
        )
        rendered = render_gaussians_2d(
            fitted,
            crop.shape[0],
            crop.shape[1],
            renderer="cuda",
        ).clamp(0.0, 1.0)
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        _save_qa(
            rendered,
            crop_path=crop_path,
            canvas_path=canvas_path,
            canvas_size=(camera.height, camera.width),
            offset=offset,
        )
        observation = native_gaussians_to_observation(
            fitted.to("cpu"),
            canvas_size=(camera.height, camera.width),
            fit_window=fit_window,
            view_id=view_id,
            n_init=fit_config.n_gaussians,
            producer_version=binding["git_revision"],
            producer_source_digest=binding["aggregate_sha256"],
            fit_config_digest=fit_digest,
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
            byte_cap=int(config["byte_cap"]),
        )
        loaded = CompactView.load(
            output_path,
            byte_cap=int(config["byte_cap"]),
        )
        if not (
            loaded.observation.n == 100_000
            and loaded.observation.provider == "native"
            and loaded.observation.blend_mode == "additive"
            and loaded.observation.support_fade_alpha == 0.0
            and loaded.observation.aa_dilation == 0.0
            and loaded.observation.fit_config_digest == fit_digest
        ):
            raise RuntimeError("strict compact reload differs from the frozen production contract")
        receipt = {
            "schema": VIEW_SCHEMA,
            "status": "PASS",
            "task_id": TASK_ID,
            "dataset_id": dataset_id,
            "view_id": view_id,
            "view_index": view_index,
            "seed": seed,
            "input": input_record,
            "source_binding": binding,
            "fit_config": config["fit_config"],
            "fit_config_digest": fit_digest,
            "canvas_size": [camera.height, camera.width],
            "fit_window": list(fit_window),
            "foreground_pixels": int(mask_crop.sum()),
            "history": history,
            "runtime": {
                "wall_seconds": wall_seconds,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
            "output": {
                "path": output_path.relative_to(ROOT).as_posix(),
                "bytes": loaded.bytes,
                "sha256": loaded.sha256,
            },
            "qa": {
                "native_crop": {
                    "path": crop_path.relative_to(ROOT).as_posix(),
                    "width": fit_window[2],
                    "height": fit_window[3],
                    "bytes": crop_path.stat().st_size,
                    "sha256": file_sha256(crop_path),
                },
                "native_full_canvas": {
                    "path": canvas_path.relative_to(ROOT).as_posix(),
                    "width": camera.width,
                    "height": camera.height,
                    "bytes": canvas_path.stat().st_size,
                    "sha256": file_sha256(canvas_path),
                },
            },
        }
        _write_json_atomic(receipt_path, receipt)
        print(
            f"{dataset_id}/{view_id}: PASS · {history['final_psnr']:.2f} dB fg · "
            f"{wall_seconds:.1f}s",
            flush=True,
        )
        return 0
    except BaseException as error:
        for path in (output_path, crop_path, canvas_path):
            path.unlink(missing_ok=True)
        failure = {
            "schema": VIEW_SCHEMA,
            "status": "FAIL",
            "task_id": TASK_ID,
            "dataset_id": dataset_id,
            "view_id": view_id,
            "seed": seed,
            "source_binding": binding,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(receipt_path, failure)
        raise


def _load_receipt(
    task: dict[str, Any],
    config: dict[str, Any],
    dataset_index: int,
    dataset: dict[str, Any],
    view_id: str,
    view_index: int,
) -> dict[str, Any]:
    output, receipt_path, crop, canvas = _paths(dataset, view_id)
    receipt = _load_json(receipt_path)
    if not (
        receipt.get("schema") == VIEW_SCHEMA
        and receipt.get("status") == "PASS"
        and receipt.get("task_id") == TASK_ID
        and receipt.get("dataset_id") == dataset["id"]
        and receipt.get("view_id") == view_id
        and receipt.get("view_index") == view_index
        and receipt.get("seed") == _seed(config, dataset_index, view_index)
        and receipt.get("fit_config") == config["fit_config"]
    ):
        raise ValueError(f"invalid production receipt: {receipt_path}")
    for path, record in (
        (output, receipt["output"]),
        (crop, receipt["qa"]["native_crop"]),
        (canvas, receipt["qa"]["native_full_canvas"]),
    ):
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError(f"receipt-bound artifact differs: {path}")
    return receipt


def _reusable(
    task: dict[str, Any],
    config: dict[str, Any],
    dataset_index: int,
    dataset: dict[str, Any],
    view_id: str,
    view_index: int,
) -> bool:
    try:
        receipt = _load_receipt(
            task,
            config,
            dataset_index,
            dataset,
            view_id,
            view_index,
        )
        output, _receipt, _crop, _canvas = _paths(dataset, view_id)
        field = CompactView.load(output, byte_cap=int(config["byte_cap"]))
    except (OSError, ValueError):
        return False
    return (
        receipt["source_binding"] == _source_binding()
        and field.observation.n == config["fit_config"]["n_gaussians"]
    )


def _write_gallery(dataset: dict[str, Any], receipts: list[dict[str, Any]]) -> Path:
    output = _output_directory(dataset)
    cards = []
    for receipt in receipts:
        view = receipt["view_id"]
        psnr = receipt["history"]["final_psnr"]
        crop_name = f"qa/{view}_field_native_crop.png"
        canvas_name = f"qa/{view}_field_native_full_canvas.png"
        cards.append(
            f"""<article><h2>{view} · {psnr:.2f} dB foreground</h2>
<a href="{crop_name}"><img src="{crop_name}" loading="lazy" alt="{view} native crop"></a>
<p><a href="{crop_name}">native 1:1 crop</a> ·
<a href="{canvas_name}">full {receipt["canvas_size"][1]}×{receipt["canvas_size"][0]} canvas</a> ·
100,000 native additive Gaussians</p></article>"""
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{dataset["id"]} native full-resolution 2D Gaussian fields</title>
<style>
body{{font-family:system-ui;background:#111;color:#eee;margin:24px}}
h1{{margin-bottom:4px}} .note{{color:#bbb;margin-top:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}}
article{{background:#1c1c1c;padding:14px;border-radius:8px}} img{{width:100%;height:auto}}
a{{color:#8cc8ff}} h2{{font-size:16px}}
</style></head><body>
<h1>{dataset["id"]}: native full-resolution 2D Gaussian fields</h1>
<p class="note">Direct links preserve original pixels. No StructSplat; 100k native additive
Gaussians, 2,000 updates per calibrated view.</p><main class="grid">
{"".join(cards)}</main></body></html>"""
    path = output / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _publish(
    task: dict[str, Any],
    config: dict[str, Any],
    dataset_index: int,
    dataset: dict[str, Any],
) -> None:
    view_ids = _view_ids(task, dataset["id"])
    receipts = [
        _load_receipt(task, config, dataset_index, dataset, view_id, view_index)
        for view_index, view_id in enumerate(view_ids)
    ]
    binding = _source_binding()
    if any(receipt["source_binding"] != binding for receipt in receipts):
        raise RuntimeError("source binding changed during full-resolution production")
    output = _output_directory(dataset)
    calibration_digest = receipts[0]["input"]["calibration"]["sha256"]
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        write_compact_dataset_manifest(
            output,
            name=f"{dataset['id']}_native_fullres",
            calibration_sha256=calibration_digest,
            view_paths=[output / f"{view_id}.rtgsv" for view_id in view_ids],
            bounds_hint=None,
            byte_cap=int(config["byte_cap"]),
        )
    gallery = _write_gallery(dataset, receipts)
    production = {
        "schema": PRODUCTION_SCHEMA,
        "task_id": TASK_ID,
        "dataset_id": dataset["id"],
        "dataset_index": dataset_index,
        "source_binding": binding,
        "configuration": config,
        "compact_manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": file_sha256(manifest_path),
        },
        "gallery": {
            "path": gallery.relative_to(ROOT).as_posix(),
            "bytes": gallery.stat().st_size,
            "sha256": file_sha256(gallery),
        },
        "views": receipts,
    }
    production["semantic_digest"] = _digest(production)
    production_path = _repository_path(
        dataset["production_manifest"],
        label=f"{dataset['id']}.production_manifest",
    )
    if production_path.exists():
        if _load_json(production_path) != production:
            raise FileExistsError(f"refusing to overwrite changed manifest: {production_path}")
    else:
        _write_json_atomic(production_path, production)


def _produce(task_path: Path) -> None:
    task = _task(task_path, require_draft=True)
    config = _config(task)
    commands: list[list[str]] = []
    for dataset_index, dataset in enumerate(_datasets(task)):
        output = _output_directory(dataset)
        output.mkdir(parents=True, exist_ok=True)
        for view_index, view_id in enumerate(_view_ids(task, dataset["id"])):
            if _reusable(
                task,
                config,
                dataset_index,
                dataset,
                view_id,
                view_index,
            ):
                print(f"{dataset['id']}/{view_id}: verified existing", flush=True)
                continue
            output_path, receipt_path, crop_path, canvas_path = _paths(dataset, view_id)
            if any(path.exists() for path in (output_path, crop_path, canvas_path)):
                raise FileExistsError(f"unverified output blocks production: {output_path}")
            if receipt_path.exists():
                failure = _load_json(receipt_path)
                if failure.get("status") != "FAIL":
                    raise FileExistsError(f"unverified receipt blocks production: {receipt_path}")
                receipt_path.unlink()
            commands.append(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_worker",
                    "--task",
                    str(task_path.resolve()),
                    "--dataset-id",
                    dataset["id"],
                    "--view-id",
                    view_id,
                    "--seed",
                    str(_seed(config, dataset_index, view_index)),
                ]
            )
    print(
        f"{len(commands)} fresh workers · parallel={config['parallel_workers']} · "
        f"{config['fit_config']['n_gaussians']}x{config['fit_config']['iterations']}",
        flush=True,
    )

    def execute(command: list[str]) -> None:
        subprocess.run(command, cwd=ROOT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(config["parallel_workers"])) as pool:
        futures = [pool.submit(execute, command) for command in commands]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    for dataset_index, dataset in enumerate(_datasets(task)):
        _publish(task, config, dataset_index, dataset)
    print(json.dumps(_verify(task_path), indent=2), flush=True)


def _verify(task_path: Path) -> dict[str, Any]:
    task = _task(task_path)
    config = _config(task)
    binding = _source_binding()
    summaries = []
    for dataset_index, dataset in enumerate(_datasets(task)):
        view_ids = _view_ids(task, dataset["id"])
        receipts = [
            _load_receipt(task, config, dataset_index, dataset, view_id, view_index)
            for view_index, view_id in enumerate(view_ids)
        ]
        if any(receipt["source_binding"] != binding for receipt in receipts):
            raise ValueError(f"{dataset['id']} source binding differs")
        compact = CompactDataset.load(
            _output_directory(dataset),
            byte_cap=int(config["byte_cap"]),
            load_alpha=False,
        )
        if [view.view_id for view in compact.views] != view_ids:
            raise ValueError(f"{dataset['id']} compact view order differs")
        production_path = _repository_path(
            dataset["production_manifest"],
            label=f"{dataset['id']}.production_manifest",
        )
        production = _load_json(production_path)
        if production.get("semantic_digest") != _digest(
            {key: value for key, value in production.items() if key != "semantic_digest"}
        ):
            raise ValueError(f"{dataset['id']} production semantic digest differs")
        summaries.append(
            {
                "dataset_id": dataset["id"],
                "views": len(view_ids),
                "gaussians": sum(view.observation.n for view in compact.views),
                "mean_foreground_psnr": sum(
                    receipt["history"]["final_psnr"] for receipt in receipts
                )
                / len(receipts),
                "compact_bytes": sum(view.bytes for view in compact.views),
                "gallery": str(_output_directory(dataset) / "index.html"),
            }
        )
    return {
        "status": "PASS",
        "task_id": TASK_ID,
        "source_binding": binding["aggregate_sha256"],
        "datasets": summaries,
    }


def _parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "produce":
        _produce(args.task)
    elif args.command == "verify":
        print(json.dumps(_verify(args.task), indent=2))
    else:
        return _worker(args.task, args.dataset_id, args.view_id, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
