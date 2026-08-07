#!/usr/bin/env python3
"""Run six independent image-backed Janelle Gaussian2D lifting experiments.

The parent process expands the frozen six-dataset × two-mask-arm × three-seed matrix. Each
measured cell runs in a fresh worker process, loads only optimizer-camera compact fields and
optimizer/validation RGB, lifts a bounded auditable field carrier, refines it with standard
image-supervised 3DGS, and opens the three held-out JPGs only after the endpoint is frozen.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import hmac
import importlib.metadata
import json
import math
import os
import platform
import resource
import secrets
import shutil
import socket
import statistics
import subprocess
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260806_gaussian2d_image_refinement_janelle_frame00008"
TASK_RELATIVE = Path("experiments/tasks") / f"{TASK_ID}.json"
RUN_RELATIVE = Path("runs") / TASK_ID
DRIVER_RELATIVE = Path("scripts/experiments") / f"{TASK_ID}.py"

DATASET_IDS = (
    "gaussians2d",
    "gaussians2d_additive",
    "gaussians2d_gaussianimage_fullres",
    "gaussians2d_native_fullres",
    "gaussians2d_structsplat_mask_contained_fullres",
    "gaussians2d_structsplat_no_boundary_fullres",
)
ARMS = ("masked_pipeline", "unmasked_pipeline")
ARM_LABELS = {
    "masked_pipeline": "Masked lift + RGB refinement",
    "unmasked_pipeline": "Unmasked lift + RGB refinement",
}
STAGE_LABELS = {
    "input_alignment": "Input alignment and split guard",
    "field_lift": "Support-aware probabilistic field lift",
    "rgb_refinement": "Standard image-backed 3DGS refinement",
    "validation_reporting": "Fixed validation convergence reporting",
    "heldout_evaluation": "Final held-out Janelle image evaluation",
    "presentation": "Per-folder report and orbit presentation",
}
EVIDENCE_SUFFIXES = ("RESULT.md", "RESULT.json", "AUDIT.md", "AUDIT.json")
WORKER_SECRET_ENV = "RTGS_JANELLE_WORKER_SECRET"
SOURCE_BINDING_PATTERNS = (
    "src/rtgs/**/*.py",
    "scripts/experiment_contract.py",
    "scripts/check_results_bundle.py",
    DRIVER_RELATIVE.as_posix(),
)
CANONICAL_REVIEW_RELATIVE = Path("experiments/reviews") / f"{TASK_ID}_PROTOCOL_REVIEW.md"
OFFICIAL_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "task_path",
        "task_sha256",
        "protocol_sha256",
        "protocol_review",
        "protocol_review_artifact_sha256",
        "data_seal_path",
        "data_seal_sha256",
        "source_commit",
        "source_dirty",
        "source_diff_sha256",
        "development",
        "started_at_utc",
        "command",
        "report_template_version",
    }
)
MEASURED_CELL_ARTIFACTS = (
    "field_lift.json",
    "gaussians_init.npz",
    "gaussians_init.ply",
    "gaussians.npz",
    "gaussians.ply",
    "heldout_metrics.json",
    "summary.json",
    "training_history.raw.json",
    "validation_metrics.json",
)
WARMUP_CELL_ARTIFACTS = tuple(
    name for name in MEASURED_CELL_ARTIFACTS if name != "heldout_metrics.json"
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.numel() <= 256:
                return tensor.tolist()
            finite = tensor.isfinite() if tensor.is_floating_point() else None
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "finite": None if finite is None else bool(finite.all()),
                "min": None if tensor.numel() == 0 else float(tensor.min()),
                "max": None if tensor.numel() == 0 else float(tensor.max()),
            }
    except ImportError:  # pragma: no cover - repository runtime always has torch
        pass
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_text_exact(path: Path, value: str) -> None:
    """Publish immutable evidence idempotently, rejecting conflicting prior bytes."""

    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"existing canonical evidence differs: {path}")
        return
    _write_text(path, value)


def _write_json_exact(path: Path, value: object) -> None:
    body = json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text_exact(path, body)


def _source_binding_payload() -> dict[str, Any]:
    paths: dict[str, Path] = {}
    for pattern in SOURCE_BINDING_PATTERNS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                relative = path.relative_to(ROOT).as_posix()
                paths[relative] = path
    records = [
        {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for relative, path in sorted(paths.items())
    ]
    return {
        "patterns": list(SOURCE_BINDING_PATTERNS),
        "file_count": len(records),
        "aggregate_sha256": _canonical_sha256(records),
    }


def _verify_source_binding(task: Mapping[str, Any]) -> dict[str, Any]:
    expected = task["frozen_configuration"]["source_binding"]
    actual = _source_binding_payload()
    if actual != expected:
        raise RuntimeError(
            "behavior-bearing source differs from the prospectively reviewed source binding"
        )
    return actual


def _seal_records(task: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    seal = _load_json(ROOT / task["data_seal"])
    records = seal.get("files")
    if not isinstance(records, list):
        raise RuntimeError("data seal files must be a list")
    result = {str(item["path"]): dict(item) for item in records}
    if len(result) != len(records):
        raise RuntimeError("data seal contains duplicate paths")
    return result


def _verify_sealed_paths(
    task: Mapping[str, Any], relative_paths: Sequence[str]
) -> list[dict[str, Any]]:
    sealed = _seal_records(task)
    verified = []
    for relative in relative_paths:
        record = sealed.get(relative)
        if record is None:
            raise RuntimeError(f"input is absent from the frozen data seal: {relative}")
        path = ROOT / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"input bytes differ from the frozen data seal: {relative}")
        verified.append(record)
    return verified


def _verify_full_data_seal(task: Mapping[str, Any]) -> dict[str, Any]:
    records = _seal_records(task)
    verified = _verify_sealed_paths(task, tuple(sorted(records)))
    return {
        "data_seal": task["data_seal"],
        "data_seal_sha256": _sha256_file(ROOT / task["data_seal"]),
        "file_count": len(verified),
        "input_bytes": sum(int(item["bytes"]) for item in verified),
        "verified_at_utc": _utc_now(),
    }


def _official_lock(task_path: Path, run: Path, task: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = run / "task.lock.json"
    lock = _load_json(lock_path)
    if set(lock) != OFFICIAL_LOCK_KEYS:
        raise RuntimeError("official task lock has the wrong keys")
    review = task["protocol_review"]
    if (
        not isinstance(review, Mapping)
        or review.get("verdict") != "approved"
        or review.get("artifact") != CANONICAL_REVIEW_RELATIVE.as_posix()
    ):
        raise RuntimeError("official task requires the canonical approved protocol review")
    review_path = ROOT / CANONICAL_REVIEW_RELATIVE
    if review_path.is_symlink() or not review_path.is_file():
        raise RuntimeError("canonical protocol review artifact is absent or not a regular file")
    review_sha256 = _sha256_file(review_path)
    checks = (
        (lock.get("schema_version") == 2, "lock schema"),
        (lock.get("task_id") == TASK_ID, "task id"),
        (lock.get("task_path") == TASK_RELATIVE.as_posix(), "task path"),
        (lock.get("task_sha256") == _sha256_file(task_path), "task bytes"),
        (
            lock.get("protocol_sha256") == review.get("protocol_sha256"),
            "protocol digest",
        ),
        (lock.get("protocol_review") == review, "protocol review"),
        (
            lock.get("protocol_review_artifact_sha256") == review_sha256,
            "protocol review artifact bytes",
        ),
        (lock.get("data_seal_path") == task["data_seal"], "data seal path"),
        (
            lock.get("data_seal_sha256") == _sha256_file(ROOT / task["data_seal"]),
            "data seal bytes",
        ),
        (lock.get("command") == task["run_command"], "run command"),
    )
    failures = [label for passed, label in checks if not passed]
    if failures:
        raise RuntimeError("official task lock differs: " + ", ".join(failures))
    _verify_source_binding(task)
    return lock


def _task_path(value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve(strict=True)
    expected = (ROOT / TASK_RELATIVE).resolve(strict=True)
    if path != expected:
        raise ValueError(f"expected task {expected}, received {path}")
    return path


def _run_path(value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve(strict=True)
    expected = (ROOT / RUN_RELATIVE).resolve(strict=True)
    if path != expected:
        raise ValueError(f"expected run {expected}, received {path}")
    return path


def _assert_task(task: Mapping[str, Any], *, require_ready: bool = True) -> None:
    if task.get("task_id") != TASK_ID:
        raise ValueError("task id does not match this driver")
    if require_ready and task.get("status") != "ready":
        raise ValueError("protected producer requires a ready task")
    dataset_ids = tuple(item["id"] for item in task["datasets"])
    if dataset_ids != DATASET_IDS:
        raise ValueError("task datasets do not match the owner-selected six-folder order")
    comparator_ids = tuple(item["id"] for item in task["comparators"])
    if comparator_ids != ARMS:
        raise ValueError("task comparators do not match the masked/unmasked driver surface")
    optimizer = tuple(task["frozen_configuration"]["optimizer_views"])
    validation = tuple(task["frozen_configuration"]["validation_views"])
    schema_train = tuple(task["splits"][DATASET_IDS[0]]["train"])
    if len(schema_train) != len((*optimizer, *validation)) or set(schema_train) != set(
        (*optimizer, *validation)
    ):
        raise ValueError("optimizer + validation views must equal the frozen train partition")
    expected_command = [
        ".venv/bin/python",
        DRIVER_RELATIVE.as_posix(),
        "--task",
        TASK_RELATIVE.as_posix(),
        "--run",
        RUN_RELATIVE.as_posix(),
    ]
    if task.get("run_command") != expected_command:
        raise ValueError("task run command does not match this driver")
    frozen = task["frozen_configuration"]
    if tuple(frozen["source_binding"]["patterns"]) != SOURCE_BINDING_PATTERNS:
        raise ValueError("task source-binding patterns do not match this driver")
    rgb = frozen["rgb_refinement"]
    if rgb["internal_checkpoint_evaluation"] is not False:
        raise ValueError("task must disable Trainer's internal checkpoint evaluation")
    if rgb["reset_cuda_peak_stats"] is not False:
        raise ValueError("task must retain process-owned CUDA peak accounting")
    if frozen["cell_receipt_policy"]["measured_cells"] != 36:
        raise ValueError("task cell receipt policy must bind all 36 measured cells")
    policy = frozen["cell_receipt_policy"]
    if (
        policy.get("strict_semantic_bundle_replay") is not True
        or tuple(policy.get("warmup_artifacts", ())) != WARMUP_CELL_ARTIFACTS
        or tuple(policy.get("measured_artifacts", ())) != MEASURED_CELL_ARTIFACTS
        or policy.get("effective_sha256") != _expected_effective_sha256(task)
    ):
        raise ValueError("task cell receipt policy does not match the executable semantics")
    launch = frozen["viewer_launch"]
    if (
        launch.get("after_all_measurement_endpoints") is not True
        or launch.get("retry_reuse_live_viewers") is not True
        or launch.get("require_process_and_http_probe") is not True
        or launch.get("launch_receipt") != "viewer_launch_receipt.json"
    ):
        raise ValueError("task viewer launch policy is not fail-closed and retry-safe")
    smoke = frozen["viewer_smoke"]
    if (
        smoke.get("receipt") != "viewer_smoke.json"
        or smoke.get("schema_version") != 2
        or tuple(smoke.get("required_dataset_ids", ())) != DATASET_IDS
        or smoke.get("require_child_report_http_200") is not True
        or smoke.get("require_webgl2_visible_content") is not True
        or smoke.get("require_orbit_camera_change") is not True
    ):
        raise ValueError("task viewer smoke policy must bind all six child pages and viewers")


def _dataset(task: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [item for item in task["datasets"] if item["id"] == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"task must define exactly one dataset {dataset_id!r}")
    return dict(matches[0])


def _partition(task: Mapping[str, Any], dataset_id: str) -> dict[str, tuple[str, ...]]:
    split = task["splits"][dataset_id]
    optimizer = tuple(task["frozen_configuration"]["optimizer_views"])
    validation = tuple(task["frozen_configuration"]["validation_views"])
    heldout = tuple(split["heldout"])
    if len(split["train"]) != len((*optimizer, *validation)) or set(split["train"]) != set(
        (*optimizer, *validation)
    ):
        raise ValueError(f"{dataset_id}: optimizer/validation partition changed")
    if set(optimizer) & set(validation) or (set(optimizer) | set(validation)) & set(heldout):
        raise ValueError(f"{dataset_id}: optimizer/validation/heldout roles overlap")
    if len(set((*optimizer, *validation, *heldout))) != 26:
        raise ValueError(f"{dataset_id}: camera roles must contain 26 unique ids")
    return {"optimizer": optimizer, "validation": validation, "heldout": heldout}


def _cell_dir(run: Path, dataset_id: str, seed: int, arm: str) -> Path:
    return run / "cells" / dataset_id / f"seed_{seed}" / arm


def _is_masked(arm: str) -> bool:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return arm == "masked_pipeline"


def _field_config(task: Mapping[str, Any], arm: str, seed: int) -> Any:
    from rtgs.lift.field_lifter import FieldLiftConfig
    from rtgs.lift.field_refit import FieldRefitConfig

    frozen = task["frozen_configuration"]["field_lift"]
    refit = FieldRefitConfig(
        iterations=int(frozen["refit_iterations"]),
        learning_rate=0.025,
        appearance_start=int(frozen["appearance_start"]),
        visibility_refresh=5,
        chunk_size=128,
        view_schedule=str(frozen["view_schedule"]),
        full_view_cleanup_iterations=0,
    )
    return FieldLiftConfig(
        placement_mode=str(frozen["placement_mode"]),
        compute_dtype=str(frozen["compute_dtype"]),
        max_tracks=int(frozen["max_tracks"]),
        max_train_views=int(frozen["max_train_views"]),
        target_component_cap=int(frozen["target_component_cap"]),
        depth_samples=int(frozen["depth_samples"]),
        min_views=int(frozen["min_views"]),
        projection_dilation=float(frozen["projection_dilation"]),
        background_fraction=float(frozen["background_fraction"]),
        mask_mode="hard" if _is_masked(arm) else "none",
        association=None,
        topology_rounds=int(frozen["topology_rounds"]),
        validation_sample_cap=int(frozen["validation_sample_cap"]),
        seed=seed,
        refit=refit,
    )


def _train_config(
    task: Mapping[str, Any], arm: str, seed: int, *, iterations: int | None = None
) -> Any:
    from rtgs.optim.density import DensityConfig
    from rtgs.optim.trainer import TrainConfig

    frozen = task["frozen_configuration"]["rgb_refinement"]
    count = int(frozen["iterations"] if iterations is None else iterations)
    density = DensityConfig(
        start_iter=min(int(frozen["densify_start"]), count + 1),
        stop_iter=min(int(frozen["densify_stop"]), count + 1),
        every=int(frozen["densify_every"]),
        grad_threshold=float(frozen["grad_threshold"]),
        absgrad=bool(frozen["absgrad"]),
        prune_opacity=float(frozen["prune_opacity"]),
        prune_scale_frac=float(frozen["prune_scale_frac"]),
        max_gaussians=int(frozen["max_gaussians"]),
    )
    return TrainConfig(
        iterations=count,
        eval_every=min(int(frozen["eval_every"]), count),
        rasterizer=str(frozen["rasterizer"]),
        device=str(frozen["device"]),
        densify=bool(frozen["densify"] and count >= int(frozen["densify_start"])),
        density_strategy=str(frozen["density_strategy"]),
        density=density,
        target_sh_degree=int(frozen["target_sh_degree"]),
        sh_degree_interval=max(1, min(int(frozen["sh_degree_interval"]), count)),
        use_masks=_is_masked(arm),
        random_background=bool(frozen["random_background"]),
        packed=bool(frozen["packed"]),
        antialiased=bool(frozen["antialiased"]),
        stream_scene_from_cpu=bool(frozen["stream_scene_from_cpu"]),
        internal_checkpoint_evaluation=bool(frozen["internal_checkpoint_evaluation"]),
        reset_cuda_peak_stats=bool(frozen["reset_cuda_peak_stats"]),
        checkpoint_policy=str(frozen["checkpoint_policy"]),
        validate_render_finite=bool(frozen["validate_render_finite"]),
        seed=seed,
    )


def _expected_effective_sha256(task: Mapping[str, Any]) -> dict[str, Any]:
    warmup = task["frozen_configuration"]["warmup"]

    def digest(arm: str, seed: int, iterations: int) -> str:
        return _canonical_sha256(
            {
                "field_lift": _json_safe(_field_config(task, arm, seed)),
                "rgb_refinement": _json_safe(_train_config(task, arm, seed, iterations=iterations)),
            }
        )

    measured_iterations = int(task["frozen_configuration"]["rgb_refinement"]["iterations"])
    return {
        "warmup": {
            str(warmup["arm_id"]): {
                str(warmup["seed"]): digest(
                    str(warmup["arm_id"]),
                    int(warmup["seed"]),
                    int(warmup["iterations"]),
                )
            }
        },
        "measured": {
            arm: {str(seed): digest(arm, int(seed), measured_iterations) for seed in task["seeds"]}
            for arm in ARMS
        },
    }


def _float32_gaussians(value: Any) -> Any:
    from rtgs.core.gaussians3d import Gaussians3D

    return Gaussians3D(
        means=value.means.detach().cpu().float(),
        quats=value.quats.detach().cpu().float(),
        log_scales=value.log_scales.detach().cpu().float(),
        opacity=value.opacity.detach().cpu().float(),
        sh=value.sh.detach().cpu().float(),
    )


def _seal_hashes(task: Mapping[str, Any]) -> dict[str, str]:
    seal = _load_json(ROOT / task["data_seal"])
    return {item["path"]: item["sha256"] for item in seal["files"]}


def _load_optimizer_fits(
    task: Mapping[str, Any], dataset_id: str, arm: str
) -> tuple[Any, dict[str, Any]]:
    """Load exactly the optimizer-camera fields; validation/test .rtgsv files stay unopened."""

    import torch

    from rtgs.data.compact_views import CompactView
    from rtgs.data.field_inputs import SceneFits

    dataset = _dataset(task, dataset_id)
    partition = _partition(task, dataset_id)
    manifest_path = ROOT / dataset["compact_manifest"]
    _verify_sealed_paths(
        task,
        (dataset["compact_manifest"], dataset["calibration"]),
    )
    manifest = _load_json(manifest_path)
    records = manifest["views"]
    by_id = {item["view_id"]: item for item in records}
    expected_ids = set((*partition["optimizer"], *partition["validation"], *partition["heldout"]))
    if set(by_id) != expected_ids or len(by_id) != len(records):
        raise RuntimeError(f"{dataset_id}: compact manifest differs from the frozen 26 views")
    calibration_path = ROOT / dataset["calibration"]
    if _sha256_file(calibration_path) != manifest["calibration_sha256"]:
        raise RuntimeError(f"{dataset_id}: compact calibration binding changed")

    cap = int(task["frozen_configuration"]["compact_view_byte_caps"][dataset_id])
    sealed = _seal_hashes(task)
    loaded = []
    loaded_records = []
    for view_id in partition["optimizer"]:
        record = by_id[view_id]
        path = manifest_path.parent / record["path"]
        relative = path.relative_to(ROOT).as_posix()
        if (
            path.stat().st_size != record["bytes"]
            or sealed.get(relative) != record["sha256"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"{dataset_id}/{view_id}: compact byte seal differs")
        view = CompactView.load(path, device="cpu", byte_cap=cap, load_alpha=_is_masked(arm))
        if view.observation.n != record["n_gaussians"] or view.view_id != view_id:
            raise RuntimeError(f"{dataset_id}/{view_id}: compact manifest payload differs")
        rgb_path = Path(dataset["frame_path"]) / "rgb" / f"{view_id}.jpg"
        mask_path = Path(dataset["frame_path"]) / "mask" / f"mask_{view_id}.png"
        if (
            view.source["rgb"]["name"] != rgb_path.name
            or view.source["rgb"]["sha256"] != sealed.get(rgb_path.as_posix())
            or view.source["mask"] is None
            or view.source["mask"]["name"] != mask_path.name
            or view.source["mask"]["sha256"] != sealed.get(mask_path.as_posix())
        ):
            raise RuntimeError(f"{dataset_id}/{view_id}: compact field is not Janelle-source-bound")
        if _is_masked(arm) and view.alpha is None:
            raise RuntimeError(f"{dataset_id}/{view_id}: masked lift requires source alpha")
        if not _is_masked(arm) and view.alpha is not None:
            raise RuntimeError(f"{dataset_id}/{view_id}: unmasked lift decoded alpha")
        loaded.append(view)
        loaded_records.append(
            {
                "view_id": view_id,
                "path": relative,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )

    bounds = manifest.get("bounds_hint")
    bounds_hint = None
    if bounds is not None:
        bounds_hint = (torch.tensor(bounds["center"], dtype=torch.float32), float(bounds["extent"]))
    fits = SceneFits(
        observations=tuple(item.observation for item in loaded),
        cameras=tuple(item.camera for item in loaded),
        view_names=tuple(item.view_id for item in loaded),
        alphas=tuple(item.alpha for item in loaded),
        train_view_indices=tuple(range(len(loaded))),
        heldout_view_indices=(),
        bounds_hint=bounds_hint,
        name=f"{manifest['name']}-{arm}-optimizer-only",
    )
    receipt = {
        "manifest": dataset["compact_manifest"],
        "manifest_sha256": _sha256_file(manifest_path),
        "semantic_digest": manifest["semantic_digest"],
        "provider_by_optimizer_view": [item.observation.provider for item in loaded],
        "blend_mode_by_optimizer_view": [item.observation.blend_mode for item in loaded],
        "source_component_counts_all_views": {
            item["view_id"]: item["n_gaussians"] for item in records
        },
        "source_component_count_total_all_views": sum(item["n_gaussians"] for item in records),
        "loaded_optimizer_views": list(partition["optimizer"]),
        "loaded_optimizer_component_counts": [item.observation.n for item in loaded],
        "loaded_optimizer_compact_sha256": _canonical_sha256(loaded_records),
        "unopened_compact_views": [*partition["validation"], *partition["heldout"]],
        "packed_alpha_decoded": _is_masked(arm),
        "source_rgb_or_mask_file_opened_during_lift": False,
        "byte_cap": cap,
    }
    return fits, receipt


def _verify_janelle_inputs(
    task: Mapping[str, Any], dataset_id: str, view_ids: Sequence[str]
) -> dict[str, Any]:
    dataset = _dataset(task, dataset_id)
    frame = Path(dataset["frame_path"])
    relative_paths = []
    for view_id in view_ids:
        relative_paths.extend(
            (
                (frame / "rgb" / f"{view_id}.jpg").as_posix(),
                (frame / "mask" / f"mask_{view_id}.png").as_posix(),
            )
        )
    verified = _verify_sealed_paths(task, relative_paths)
    return {
        "view_ids": list(view_ids),
        "file_count": len(verified),
        "bytes": sum(int(item["bytes"]) for item in verified),
        "records_sha256": _canonical_sha256(verified),
    }


def _load_training_scene(task: Mapping[str, Any], dataset_id: str) -> tuple[Any, Any, dict]:
    from dataclasses import replace

    from rtgs.data.calibrated import load_calibrated_scene

    dataset = _dataset(task, dataset_id)
    partition = _partition(task, dataset_id)
    view_ids = [*partition["optimizer"], *partition["validation"]]
    input_receipt = _verify_janelle_inputs(task, dataset_id, view_ids)
    scene = load_calibrated_scene(
        ROOT / dataset["frame_path"],
        calibration_path=ROOT / dataset["calibration"],
        downscale=int(task["frozen_configuration"]["image_downscale"]),
        test_every=0,
        load_masks=True,
        undistort=True,
        view_ids=view_ids,
    )
    optimizer_count = len(partition["optimizer"])
    scene = replace(
        scene,
        train_indices=list(range(optimizer_count)),
        test_indices=list(range(optimizer_count, len(view_ids))),
        name=f"{dataset_id}-optimizer-validation",
    )
    scene.validate()
    return scene, replace(scene, masks=None, name=f"{scene.name}-unmasked"), input_receipt


def _verify_camera_alignment(fits: Any, scene: Any) -> dict[str, Any]:
    import torch

    names = {name: index for index, name in enumerate(scene.view_names)}
    if set(fits.view_names) - set(names):
        raise RuntimeError("compact optimizer camera is absent from the Janelle image scene")
    records = []
    for compact_camera, name in zip(fits.cameras, fits.view_names, strict=True):
        image_camera = scene.cameras[names[name]]
        sx = image_camera.width / compact_camera.width
        sy = image_camera.height / compact_camera.height
        if (
            not torch.allclose(compact_camera.R, image_camera.R, atol=1e-6, rtol=0)
            or not torch.allclose(compact_camera.t, image_camera.t, atol=1e-6, rtol=0)
            or abs(sx - sy) > 1e-12
            or abs(compact_camera.fx * sx - image_camera.fx) > 1e-3
            or abs(compact_camera.fy * sy - image_camera.fy) > 1e-3
            or abs(compact_camera.cx * sx - image_camera.cx) > 1e-3
            or abs(compact_camera.cy * sy - image_camera.cy) > 1e-3
        ):
            raise RuntimeError(f"compact/image camera calibration differs for {name}")
        records.append(
            {
                "view_id": name,
                "compact_size": [compact_camera.width, compact_camera.height],
                "image_size": [image_camera.width, image_camera.height],
                "scale": [sx, sy],
                "compact_intrinsics": [
                    compact_camera.fx,
                    compact_camera.fy,
                    compact_camera.cx,
                    compact_camera.cy,
                ],
                "image_intrinsics": [
                    image_camera.fx,
                    image_camera.fy,
                    image_camera.cx,
                    image_camera.cy,
                ],
                "extrinsics_sha256": _canonical_sha256(
                    {
                        "R": compact_camera.R.tolist(),
                        "t": compact_camera.t.tolist(),
                    }
                ),
            }
        )
    return {"records": records, "records_sha256": _canonical_sha256(records)}


def _per_view_metrics(scene: Any, gaussians: Any, renderer: Any, indices: Sequence[int]) -> dict:
    import torch

    from rtgs.core.metrics import image_metrics, psnr

    rows = []
    totals: dict[str, float] = {}
    device = gaussians.means.device
    with torch.no_grad():
        for index in indices:
            image = scene.images[index].to(device)
            camera = scene.cameras[index].to(device)
            mask = None if scene.masks is None else scene.masks[index].to(device)
            output = renderer.render(gaussians, camera)
            values = image_metrics(output.color, image, mask)
            if mask is not None:
                # ``image_metrics(..., mask)`` reports its full-frame diagnostic against a
                # black-matted target. This experiment's declared full-canvas endpoint is the
                # actual Janelle photograph, so retain that distinct quantity explicitly.
                values["psnr_full"] = psnr(output.color.clamp(0, 1), image.clamp(0, 1))
                foreground = mask > 0.5
                predicted = output.alpha > 0.5
                intersection = (foreground & predicted).sum()
                union = (foreground | predicted).sum().clamp_min(1)
                values["alpha_iou"] = float(intersection / union)
                values["alpha_inside"] = float(output.alpha[foreground].mean())
                background = ~foreground
                values["alpha_outside"] = (
                    float(output.alpha[background].mean()) if bool(background.any()) else 0.0
                )
            row = {"view_id": scene.view_names[index], **values}
            rows.append(row)
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value)
    aggregate = {key: value / len(rows) for key, value in totals.items()}
    return {"aggregate": aggregate, "per_view": rows}


def _validation_auc(records: Sequence[Mapping[str, Any]]) -> float:
    points = sorted(
        (float(item["optimizer_wall_seconds"]), float(item["metrics"]["psnr_fg"]))
        for item in records
    )
    if len(points) < 2 or points[-1][0] <= points[0][0]:
        raise ValueError("validation AUC requires at least two increasing-time checkpoints")
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x1 < x0:
            raise ValueError("validation checkpoint time decreased")
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area / (points[-1][0] - points[0][0])


def _start_cuda_measurement(torch_module: Any) -> int | None:
    if not torch_module.cuda.is_available():
        return None
    device_index = int(torch_module.cuda.current_device())
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats(device_index)
    torch_module.cuda.reset_accumulated_memory_stats(device_index)
    torch_module.cuda.synchronize(device_index)
    return device_index


def _freeze_resource_endpoint(
    torch_module: Any,
    device: Any,
    *,
    cell_started: float,
    measurement_started: float,
) -> dict[str, int | float]:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)
        allocated = int(torch_module.cuda.max_memory_allocated(device))
        reserved = int(torch_module.cuda.max_memory_reserved(device))
    else:
        allocated = 0
        reserved = 0
    endpoint = time.perf_counter()
    return {
        "measurement_endpoint_wall_seconds": endpoint - cell_started,
        "measurement_total_wall_seconds": endpoint - measurement_started,
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }


def _artifact_records(output: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    records = []
    for name in names:
        path = output / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"required cell artifact is missing or empty: {path}")
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return records


def _write_cell_receipt(
    *,
    output: Path,
    task: Mapping[str, Any],
    official_binding: Mapping[str, Any],
    summary: Mapping[str, Any],
    compact_receipt: Mapping[str, Any],
    camera_receipt: Mapping[str, Any],
    image_input_receipt: Mapping[str, Any],
) -> None:
    mode = str(official_binding["mode"])
    artifact_names = WARMUP_CELL_ARTIFACTS if mode == "warmup" else MEASURED_CELL_ARTIFACTS
    partition = _partition(task, str(summary["dataset_id"]))
    receipt = {
        "schema": "rtgs.janelle_gaussian2d_image_cell_receipt.v1",
        "task_id": TASK_ID,
        "protocol_sha256": official_binding["protocol_sha256"],
        "task_lock_sha256": official_binding["task_lock_sha256"],
        "data_seal_sha256": official_binding["data_seal_sha256"],
        "source_binding_sha256": task["frozen_configuration"]["source_binding"]["aggregate_sha256"],
        "dataset_id": summary["dataset_id"],
        "arm": summary["arm"],
        "seed": summary["seed"],
        "mode": mode,
        "iterations": official_binding["iterations"],
        "output_path": output.relative_to(ROOT).as_posix(),
        "partition_sha256": _canonical_sha256(partition),
        "effective_sha256": _canonical_sha256(summary["effective"]),
        "input_binding": {
            "manifest_sha256": compact_receipt["manifest_sha256"],
            "compact_optimizer_sha256": compact_receipt["loaded_optimizer_compact_sha256"],
            "camera_records_sha256": camera_receipt["records_sha256"],
            "optimizer_validation_image_sha256": image_input_receipt["records_sha256"],
        },
        "artifacts": _artifact_records(output, artifact_names),
    }
    _write_json(output / "cell_receipt.json", receipt)


def _validate_cell_bundle(
    *,
    run: Path,
    task: Mapping[str, Any],
    lock: Mapping[str, Any],
    dataset_id: str,
    seed: int,
    arm: str,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"warmup", "measured"}:
        raise ValueError("cell mode must be warmup or measured")
    if mode == "warmup":
        output = run / "warmup" / dataset_id / arm
        iterations = int(task["frozen_configuration"]["warmup"]["iterations"])
        artifact_names = WARMUP_CELL_ARTIFACTS
    else:
        output = _cell_dir(run, dataset_id, seed, arm)
        iterations = int(task["frozen_configuration"]["rgb_refinement"]["iterations"])
        artifact_names = MEASURED_CELL_ARTIFACTS
    receipt = _load_json(output / "cell_receipt.json")
    expected_keys = {
        "schema",
        "task_id",
        "protocol_sha256",
        "task_lock_sha256",
        "data_seal_sha256",
        "source_binding_sha256",
        "dataset_id",
        "arm",
        "seed",
        "mode",
        "iterations",
        "output_path",
        "partition_sha256",
        "effective_sha256",
        "input_binding",
        "artifacts",
    }
    if set(receipt) != expected_keys:
        raise RuntimeError(f"cell receipt has the wrong keys: {output}")
    expected_identity = {
        "schema": "rtgs.janelle_gaussian2d_image_cell_receipt.v1",
        "task_id": TASK_ID,
        "protocol_sha256": lock["protocol_sha256"],
        "task_lock_sha256": _sha256_file(run / "task.lock.json"),
        "data_seal_sha256": lock["data_seal_sha256"],
        "source_binding_sha256": task["frozen_configuration"]["source_binding"]["aggregate_sha256"],
        "dataset_id": dataset_id,
        "arm": arm,
        "seed": seed,
        "mode": mode,
        "iterations": iterations,
        "output_path": output.relative_to(ROOT).as_posix(),
        "partition_sha256": _canonical_sha256(_partition(task, dataset_id)),
    }
    for key, expected in expected_identity.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"cell receipt {key} differs for {output}")
    summary = _load_json(output / "summary.json")
    if (
        summary.get("status") != "completed"
        or summary.get("task_id") != TASK_ID
        or summary.get("dataset_id") != dataset_id
        or summary.get("arm") != arm
        or type(summary.get("seed")) is not int
        or summary["seed"] != seed
        or summary.get("warmup") is not (mode == "warmup")
    ):
        raise RuntimeError(f"cell summary identity differs for {output}")
    expected_effective = {
        "field_lift": _json_safe(_field_config(task, arm, seed)),
        "rgb_refinement": _json_safe(_train_config(task, arm, seed, iterations=iterations)),
    }
    if summary.get("effective") != expected_effective:
        raise RuntimeError(f"cell effective configuration differs for {output}")
    if receipt["effective_sha256"] != _canonical_sha256(summary["effective"]):
        raise RuntimeError(f"cell effective configuration digest differs for {output}")
    if mode == "warmup":
        if (
            summary.get("heldout_outcome_access") is not False
            or (output / "heldout_metrics.json").exists()
        ):
            raise RuntimeError(f"warmup accessed held-out outcomes: {output}")
    else:
        metrics = summary.get("metrics")
        expected_metric_ids = {item["id"] for item in task["primary_metrics"]}
        if (
            summary.get("heldout_opened_after_endpoint_saved") is not True
            or summary.get("measurement_endpoint_before_heldout") is not True
            or not isinstance(metrics, dict)
            or set(metrics) != expected_metric_ids
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
        ):
            raise RuntimeError(f"measured cell metric/endpoint semantics differ: {output}")
    records = receipt["artifacts"]
    if not isinstance(records, list) or [item.get("path") for item in records] != list(
        artifact_names
    ):
        raise RuntimeError(f"cell artifact inventory differs for {output}")
    if records != _artifact_records(output, artifact_names):
        raise RuntimeError(f"cell artifact hash differs for {output}")
    field = _load_json(output / "field_lift.json")
    expected_input = {
        "manifest_sha256": field["manifest_sha256"],
        "compact_optimizer_sha256": field["loaded_optimizer_compact_sha256"],
        "camera_records_sha256": summary["input_binding"]["camera_records_sha256"],
        "optimizer_validation_image_sha256": summary["input_binding"][
            "optimizer_validation_image_sha256"
        ],
    }
    if receipt["input_binding"] != expected_input:
        raise RuntimeError(f"cell input binding differs for {output}")
    summary_input = summary.get("input_binding")
    expected_summary_keys = {
        "camera_records_sha256",
        "optimizer_validation_image_sha256",
    }
    if mode == "measured":
        expected_summary_keys.add("heldout_image_sha256")
    if not isinstance(summary_input, dict) or set(summary_input) != expected_summary_keys:
        raise RuntimeError(f"cell summary input binding differs for {output}")
    return summary


def _run_worker(
    *,
    task_path: Path,
    output: Path,
    dataset_id: str,
    seed: int,
    arm: str,
    iterations: int,
    mode: str,
    official_binding: Mapping[str, Any] | None,
) -> int:
    started_utc = _utc_now()
    phase = "input_alignment"
    output.mkdir(parents=True, exist_ok=False)
    cell_started = time.perf_counter()
    stage_intervals: dict[str, dict[str, float]] = {}
    warmup = mode != "measured"
    try:
        import torch

        from rtgs.core.gaussians3d import Gaussians3D
        from rtgs.data.calibrated import load_calibrated_scene
        from rtgs.lift.field_lifter import FieldLifter
        from rtgs.optim.trainer import Trainer
        from rtgs.render.base import get_rasterizer

        task = _load_json(task_path)
        if mode not in {"scratch", "warmup", "measured"}:
            raise ValueError("worker mode is not recognized")
        if mode == "scratch" and official_binding is not None:
            raise ValueError("scratch execution cannot carry an official worker binding")
        if mode != "scratch" and official_binding is None:
            raise ValueError("official execution requires an authenticated worker binding")
        _assert_task(task, require_ready=mode != "scratch")
        if dataset_id not in DATASET_IDS or arm not in ARMS or seed not in task["seeds"]:
            raise ValueError("worker cell is outside the frozen matrix")
        if mode == "measured" and iterations != int(
            task["frozen_configuration"]["rgb_refinement"]["iterations"]
        ):
            raise ValueError("measured worker iterations differ from the frozen endpoint")
        if mode == "warmup" and iterations != int(
            task["frozen_configuration"]["warmup"]["iterations"]
        ):
            raise ValueError("warmup worker iterations differ from the frozen endpoint")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cuda_measurement_device = _start_cuda_measurement(torch)
        measurement_started = time.perf_counter()

        stage_start = time.perf_counter()
        fits, compact_receipt = _load_optimizer_fits(task, dataset_id, arm)
        masked_scene, unmasked_scene, image_input_receipt = _load_training_scene(task, dataset_id)
        camera_receipt = _verify_camera_alignment(fits, masked_scene)
        trainer_source = masked_scene if _is_masked(arm) else unmasked_scene
        partition = _partition(task, dataset_id)
        train_scene = trainer_source.subset(
            list(range(len(partition["optimizer"]))),
            name_suffix="optimizer-only",
        )
        if train_scene.testing_views or train_scene.n_views != len(partition["optimizer"]):
            raise RuntimeError("Trainer scene must contain optimizer cameras only")
        stage_intervals[phase] = {
            "start": 0.0,
            "end": time.perf_counter() - cell_started,
            "seconds": time.perf_counter() - stage_start,
        }

        phase = "field_lift"
        stage_start = time.perf_counter()
        lift_config = _field_config(task, arm, seed)
        lift_result = FieldLifter(lift_config).fit(fits)
        if set(lift_result.optimized_view_indices) - set(fits.train_view_indices):
            raise RuntimeError("field lift accessed a non-optimizer compact view")
        initial = _float32_gaussians(lift_result.gaussians)
        if initial.n <= 0 or not all(
            bool(torch.isfinite(value).all())
            for value in (
                initial.means,
                initial.quats,
                initial.log_scales,
                initial.opacity,
                initial.sh,
            )
        ):
            raise RuntimeError("field lift produced an empty or non-finite initialization")
        initial.save_ply(output / "gaussians_init.ply")
        initial.save_npz(output / "gaussians_init.npz")
        stage_intervals[phase] = {
            "start": stage_start - cell_started,
            "end": time.perf_counter() - cell_started,
            "seconds": time.perf_counter() - stage_start,
        }
        _write_json(
            output / "field_lift.json",
            {
                **compact_receipt,
                "config": lift_config,
                "initial_gaussians": initial.n,
                "optimized_view_indices": lift_result.optimized_view_indices,
                "heldout_view_indices": lift_result.heldout_view_indices,
                "diagnostics": lift_result.diagnostics,
                "semantic_validation": lift_result.semantic_validation,
                "camera_alignment": camera_receipt,
                "optimizer_validation_image_input": image_input_receipt,
                "stage_seconds": stage_intervals[phase]["seconds"],
            },
        )

        phase = "rgb_refinement"
        stage_start = time.perf_counter()
        config = _train_config(task, arm, seed, iterations=iterations)
        if config.internal_checkpoint_evaluation or config.reset_cuda_peak_stats:
            raise RuntimeError(
                "experiment Trainer must use external-only validation and process-owned peaks"
            )
        device = torch.device(config.device)
        if device.type == "cuda" and cuda_measurement_device != device.index:
            raise RuntimeError("CUDA measurement device differs from the Trainer device")
        renderer = get_rasterizer(
            config.rasterizer,
            device=device,
            packed=config.packed,
            antialiased=config.antialiased,
        )
        initial_device = initial.to(device)
        validation_records = []
        validation_render_seconds = 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initial_observer_started = time.perf_counter()
        initial_validation = _per_view_metrics(
            masked_scene,
            initial_device,
            renderer,
            masked_scene.testing_views,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initial_observer_seconds = time.perf_counter() - initial_observer_started
        validation_render_seconds += initial_observer_seconds
        initial_validation_cell_wall_seconds = time.perf_counter() - cell_started
        validation_records.append(
            {
                "step": 0,
                "optimizer_wall_seconds": 0.0,
                "cell_wall_seconds": initial_validation_cell_wall_seconds,
                "metrics": initial_validation["aggregate"],
            }
        )

        def checkpoint(snapshot: Gaussians3D, step: int) -> None:
            nonlocal validation_render_seconds
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            values = _per_view_metrics(
                masked_scene,
                snapshot,
                renderer,
                masked_scene.testing_views,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            validation_render_seconds += time.perf_counter() - before
            validation_records.append(
                {
                    "step": int(step),
                    "optimizer_wall_seconds": None,
                    "cell_wall_seconds": time.perf_counter() - cell_started,
                    "metrics": values["aggregate"],
                }
            )

        trainer_started_cell_wall_seconds = time.perf_counter() - cell_started
        final, history = Trainer(config).train(
            train_scene,
            initial,
            checkpoint_callback=checkpoint,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_seconds = float(history["elapsed"][-1][1])
        elapsed_by_step = {int(step): float(value) for step, value in history["elapsed"]}
        for record in validation_records[1:]:
            step = int(record["step"])
            if step not in elapsed_by_step:
                raise RuntimeError("validation checkpoint lacks a native optimizer timestamp")
            record["optimizer_wall_seconds"] = elapsed_by_step[step]
        callback_observer_seconds = float(history["checkpoint_callback_seconds"])
        validation_observer_seconds = initial_observer_seconds + callback_observer_seconds
        if validation_observer_seconds + 1e-9 < validation_render_seconds:
            raise RuntimeError("validation observer accounting excludes rendered work")
        final = _float32_gaussians(final)
        final.save_ply(output / "gaussians.ply")
        final.save_npz(output / "gaussians.npz")
        endpoint_resource = _freeze_resource_endpoint(
            torch,
            device,
            cell_started=cell_started,
            measurement_started=measurement_started,
        )
        _write_json(output / "training_history.raw.json", history)
        _write_json(output / "validation_metrics.json", {"records": validation_records})
        stage_intervals[phase] = {
            "start": stage_start - cell_started,
            "end": endpoint_resource["measurement_endpoint_wall_seconds"],
            "seconds": endpoint_resource["measurement_endpoint_wall_seconds"]
            - (stage_start - cell_started),
            "optimizer_seconds": optimizer_seconds,
            "observer_seconds": validation_observer_seconds,
            "initial_validation_seconds": initial_observer_seconds,
            "checkpoint_callback_seconds": callback_observer_seconds,
            "validation_render_seconds": validation_render_seconds,
            "trainer_started_cell_wall_seconds": trainer_started_cell_wall_seconds,
        }

        phase = "validation_reporting"
        validation_stage_start = float(endpoint_resource["measurement_endpoint_wall_seconds"])
        validation_auc = _validation_auc(validation_records)
        validation_stage_end = time.perf_counter() - cell_started
        stage_intervals[phase] = {
            "start": validation_stage_start,
            "end": validation_stage_end,
            "seconds": validation_stage_end - validation_stage_start,
        }

        if mode != "measured":
            stage_intervals["heldout_evaluation"] = {
                "start": time.perf_counter() - cell_started,
                "end": time.perf_counter() - cell_started,
                "seconds": 0.0,
                "skipped": True,
            }
            stage_intervals["presentation"] = {
                "start": time.perf_counter() - cell_started,
                "end": time.perf_counter() - cell_started,
                "seconds": 0.0,
            }
            summary = {
                "schema": "rtgs.janelle_gaussian2d_image_warmup.v1",
                "status": "completed",
                "task_id": TASK_ID,
                "dataset_id": dataset_id,
                "arm": arm,
                "seed": seed,
                "warmup": True,
                "started_at_utc": started_utc,
                "finished_at_utc": _utc_now(),
                "optimizer_views": list(partition["optimizer"]),
                "validation_views": list(partition["validation"]),
                "heldout_views_opened": [],
                "heldout_outcome_access": False,
                "initial_gaussians": initial.n,
                "final_gaussians": final.n,
                "input_binding": {
                    "camera_records_sha256": camera_receipt["records_sha256"],
                    "optimizer_validation_image_sha256": image_input_receipt["records_sha256"],
                },
                "stage_intervals": stage_intervals,
                "resource": {
                    "measurement_scope": (
                        "fresh process after CUDA peak reset through final PLY/NPZ save; "
                        "held-out and presentation work excluded"
                    ),
                    "total_cell_wall_seconds": endpoint_resource["measurement_total_wall_seconds"],
                    "field_lift_wall_seconds": stage_intervals["field_lift"]["seconds"],
                    "rgb_refinement_wall_seconds": optimizer_seconds,
                    "validation_observer_seconds": validation_observer_seconds,
                    **endpoint_resource,
                },
                "effective": {
                    "field_lift": lift_config,
                    "rgb_refinement": config,
                },
            }
            _write_json(output / "summary.json", summary)
            if official_binding is not None:
                _write_cell_receipt(
                    output=output,
                    task=task,
                    official_binding=official_binding,
                    summary=_load_json(output / "summary.json"),
                    compact_receipt=compact_receipt,
                    camera_receipt=camera_receipt,
                    image_input_receipt=image_input_receipt,
                )
            print(
                f"WARMUP_COMPLETE dataset={dataset_id} arm={arm} seed={seed} "
                f"heldout_access=false optimizer_seconds={optimizer_seconds:.3f}",
                flush=True,
            )
            return 0

        if official_binding is None:
            raise RuntimeError("held-out evaluation requires an authenticated measured worker")
        phase = "heldout_evaluation"
        stage_start = time.perf_counter()
        dataset = _dataset(task, dataset_id)
        heldout_input_receipt = _verify_janelle_inputs(task, dataset_id, partition["heldout"])
        heldout_scene = load_calibrated_scene(
            ROOT / dataset["frame_path"],
            calibration_path=ROOT / dataset["calibration"],
            downscale=int(task["frozen_configuration"]["image_downscale"]),
            test_every=0,
            load_masks=True,
            undistort=True,
            view_ids=partition["heldout"],
        )
        heldout_scene.train_indices = []
        heldout_scene.test_indices = list(range(heldout_scene.n_views))
        final_device = final.to(device)
        heldout = _per_view_metrics(
            heldout_scene,
            final_device,
            renderer,
            heldout_scene.testing_views,
        )
        _write_json(output / "heldout_metrics.json", heldout)
        stage_intervals[phase] = {
            "start": stage_start - cell_started,
            "end": time.perf_counter() - cell_started,
            "seconds": time.perf_counter() - stage_start,
        }

        phase = "presentation"
        stage_start = time.perf_counter()
        metrics = {
            "heldout_foreground_psnr": heldout["aggregate"]["psnr_fg"],
            "heldout_full_psnr": heldout["aggregate"]["psnr_full"],
            "heldout_crop_ssim": heldout["aggregate"]["ssim_crop"],
            "heldout_alpha_iou": heldout["aggregate"]["alpha_iou"],
            "heldout_exterior_alpha": heldout["aggregate"]["alpha_outside"],
            "validation_auc_foreground_psnr": validation_auc,
            "rgb_refinement_wall_seconds": optimizer_seconds,
            "field_lift_wall_seconds": stage_intervals["field_lift"]["seconds"],
            "validation_observer_seconds": validation_observer_seconds,
            "measurement_endpoint_wall_seconds": endpoint_resource[
                "measurement_endpoint_wall_seconds"
            ],
            "total_cell_wall_seconds": endpoint_resource["measurement_total_wall_seconds"],
            "peak_cuda_allocated_bytes": endpoint_resource["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": endpoint_resource["peak_cuda_reserved_bytes"],
            "peak_rss_bytes": endpoint_resource["peak_rss_bytes"],
            "final_gaussians": final.n,
        }
        stage_intervals[phase] = {
            "start": stage_start - cell_started,
            "end": time.perf_counter() - cell_started,
            "seconds": time.perf_counter() - stage_start,
        }
        summary = {
            "schema": "rtgs.janelle_gaussian2d_image_cell.v1",
            "status": "completed",
            "task_id": TASK_ID,
            "dataset_id": dataset_id,
            "arm": arm,
            "seed": seed,
            "warmup": warmup,
            "started_at_utc": started_utc,
            "finished_at_utc": _utc_now(),
            "optimizer_views": list(partition["optimizer"]),
            "validation_views": list(partition["validation"]),
            "heldout_views": list(partition["heldout"]),
            "heldout_opened_after_endpoint_saved": True,
            "measurement_endpoint_before_heldout": True,
            "masked_field_lift": _is_masked(arm),
            "masked_rgb_refinement": _is_masked(arm),
            "initial_gaussians": initial.n,
            "final_gaussians": final.n,
            "input_binding": {
                "camera_records_sha256": camera_receipt["records_sha256"],
                "optimizer_validation_image_sha256": image_input_receipt["records_sha256"],
                "heldout_image_sha256": heldout_input_receipt["records_sha256"],
            },
            "metrics": metrics,
            "stage_intervals": stage_intervals,
            "resource": {
                "measurement_scope": (
                    "fresh process after CUDA peak reset through final PLY/NPZ save; held-out "
                    "and presentation work excluded"
                ),
                "total_cell_wall_seconds": endpoint_resource["measurement_total_wall_seconds"],
                "field_lift_wall_seconds": stage_intervals["field_lift"]["seconds"],
                "rgb_refinement_wall_seconds": optimizer_seconds,
                "validation_observer_seconds": validation_observer_seconds,
                "heldout_evaluation_wall_seconds": stage_intervals["heldout_evaluation"]["seconds"],
                **endpoint_resource,
            },
            "effective": {
                "field_lift": lift_config,
                "rgb_refinement": config,
            },
        }
        _write_json(output / "summary.json", summary)
        _write_cell_receipt(
            output=output,
            task=task,
            official_binding=official_binding,
            summary=_load_json(output / "summary.json"),
            compact_receipt=compact_receipt,
            camera_receipt=camera_receipt,
            image_input_receipt=image_input_receipt,
        )
        print(
            "CELL_COMPLETE "
            f"dataset={dataset_id} arm={arm} seed={seed} "
            f"fg_psnr={metrics['heldout_foreground_psnr']:.4f} "
            f"gaussians={final.n} optimizer_seconds={optimizer_seconds:.3f}",
            flush=True,
        )
        return 0
    except Exception as error:
        failure = {
            "schema": "rtgs.janelle_gaussian2d_image_cell_failure.v1",
            "status": "failed",
            "task_id": TASK_ID,
            "dataset_id": dataset_id,
            "arm": arm,
            "seed": seed,
            "warmup": warmup,
            "mode": mode,
            "started_at_utc": started_utc,
            "finished_at_utc": _utc_now(),
            "phase": phase,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "stage_intervals": stage_intervals,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        }
        _write_json(output / "failure.json", failure)
        print(
            f"CELL_FAILED dataset={dataset_id} arm={arm} seed={seed} "
            f"phase={phase} error={type(error).__name__}: {error}",
            flush=True,
        )
        return 1


def _worker_ticket_path(run: Path, *, dataset_id: str, seed: int, arm: str, mode: str) -> Path:
    label = f"{mode}__{dataset_id}__seed_{seed}__{arm}.json"
    return run / "worker_tickets" / label


def _write_worker_ticket(
    *,
    run: Path,
    task_path: Path,
    output: Path,
    dataset_id: str,
    seed: int,
    arm: str,
    iterations: int,
    mode: str,
    secret: str,
) -> Path:
    task = _load_json(task_path)
    lock = _official_lock(task_path, run, task)
    body = {
        "schema": "rtgs.janelle_gaussian2d_image_worker_ticket.v1",
        "task_id": TASK_ID,
        "task_path": TASK_RELATIVE.as_posix(),
        "run_path": RUN_RELATIVE.as_posix(),
        "task_lock_sha256": _sha256_file(run / "task.lock.json"),
        "protocol_sha256": lock["protocol_sha256"],
        "protocol_review_artifact_sha256": lock["protocol_review_artifact_sha256"],
        "data_seal_sha256": lock["data_seal_sha256"],
        "dataset_id": dataset_id,
        "arm": arm,
        "seed": seed,
        "iterations": iterations,
        "mode": mode,
        "output_path": output.relative_to(ROOT).as_posix(),
        "nonce": secrets.token_hex(16),
    }
    signature = hmac.new(
        bytes.fromhex(secret),
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    path = _worker_ticket_path(run, dataset_id=dataset_id, seed=seed, arm=arm, mode=mode)
    _write_json(path, {"body": body, "hmac_sha256": signature})
    return path


def _worker_command(ticket: Path) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / DRIVER_RELATIVE),
        "--worker-ticket",
        str(ticket),
    ]


def _run_ticket_worker(ticket_value: str) -> int:
    secret = os.environ.pop(WORKER_SECRET_ENV, None)
    if secret is None or len(secret) != 64:
        raise RuntimeError("worker ticket authentication secret is absent")
    try:
        bytes.fromhex(secret)
    except ValueError as error:
        raise RuntimeError("worker ticket authentication secret is malformed") from error
    ticket = Path(ticket_value).resolve(strict=True)
    run = (ROOT / RUN_RELATIVE).resolve(strict=True)
    try:
        ticket.relative_to((run / "worker_tickets").resolve(strict=True))
    except ValueError as error:
        raise RuntimeError("worker ticket is outside the canonical run") from error
    payload = _load_json(ticket)
    if set(payload) != {"body", "hmac_sha256"} or not isinstance(payload["body"], dict):
        raise RuntimeError("worker ticket payload is malformed")
    body = payload["body"]
    expected_signature = hmac.new(
        bytes.fromhex(secret),
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(str(payload["hmac_sha256"]), expected_signature):
        raise RuntimeError("worker ticket authentication failed")
    task_path = _task_path(str(body.get("task_path")))
    task = _load_json(task_path)
    _assert_task(task)
    lock = _official_lock(task_path, run, task)
    expected_body_keys = {
        "schema",
        "task_id",
        "task_path",
        "run_path",
        "task_lock_sha256",
        "protocol_sha256",
        "protocol_review_artifact_sha256",
        "data_seal_sha256",
        "dataset_id",
        "arm",
        "seed",
        "iterations",
        "mode",
        "output_path",
        "nonce",
    }
    if set(body) != expected_body_keys:
        raise RuntimeError("worker ticket body has the wrong keys")
    dataset_id = str(body["dataset_id"])
    arm = str(body["arm"])
    seed = body["seed"]
    iterations = body["iterations"]
    mode = str(body["mode"])
    if type(seed) is not int or type(iterations) is not int:
        raise RuntimeError("worker ticket numeric identity is not type-strict")
    if mode == "warmup":
        expected_output = run / "warmup" / dataset_id / arm
    elif mode == "measured":
        expected_output = _cell_dir(run, dataset_id, seed, arm)
    else:
        raise RuntimeError("official worker ticket cannot request scratch mode")
    checks = (
        body["schema"] == "rtgs.janelle_gaussian2d_image_worker_ticket.v1",
        body["task_id"] == TASK_ID,
        body["run_path"] == RUN_RELATIVE.as_posix(),
        body["task_lock_sha256"] == _sha256_file(run / "task.lock.json"),
        body["protocol_sha256"] == lock["protocol_sha256"],
        body["protocol_review_artifact_sha256"] == lock["protocol_review_artifact_sha256"],
        body["data_seal_sha256"] == lock["data_seal_sha256"],
        body["output_path"] == expected_output.relative_to(ROOT).as_posix(),
        ticket
        == _worker_ticket_path(run, dataset_id=dataset_id, seed=seed, arm=arm, mode=mode).resolve(
            strict=True
        ),
    )
    if not all(checks):
        raise RuntimeError("worker ticket differs from the canonical locked cell")
    binding = {
        "mode": mode,
        "iterations": iterations,
        "task_lock_sha256": body["task_lock_sha256"],
        "protocol_sha256": body["protocol_sha256"],
        "protocol_review_artifact_sha256": body["protocol_review_artifact_sha256"],
        "data_seal_sha256": body["data_seal_sha256"],
    }
    return _run_worker(
        task_path=task_path,
        output=expected_output,
        dataset_id=dataset_id,
        seed=seed,
        arm=arm,
        iterations=iterations,
        mode=mode,
        official_binding=binding,
    )


def _run_subprocess(command: Sequence[str], *, secret: str) -> int:
    environment = dict(os.environ)
    environment[WORKER_SECRET_ENV] = secret
    process = subprocess.Popen(
        list(command),
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def _median(values: Sequence[int | float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty metric sequence")
    result = float(statistics.median(float(value) for value in values))
    if not math.isfinite(result):
        raise ValueError("aggregate metric is non-finite")
    return result


def _elapsed_at_step(history: Mapping[str, Any], step: int) -> float:
    points = [(0, 0.0), *[(int(item[0]), float(item[1])) for item in history["elapsed"]]]
    points = sorted(dict(points).items())
    if step <= points[0][0]:
        return points[0][1]
    if step >= points[-1][0]:
        return points[-1][1]
    for (left_step, left_time), (right_step, right_time) in zip(points, points[1:]):
        if left_step <= step <= right_step:
            fraction = (step - left_step) / max(right_step - left_step, 1)
            return left_time + fraction * (right_time - left_time)
    raise RuntimeError("training elapsed interpolation failed")


def _callback_elapsed_at_step(history: Mapping[str, Any], step: int) -> float:
    points = [
        (int(item[0]), float(item[1])) for item in history.get("checkpoint_callback_elapsed", [])
    ]
    prior = [value for completed_step, value in points if completed_step <= step]
    return prior[-1] if prior else 0.0


def _history_bundle(task: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]) -> dict:
    records: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, str]] = {
        "input_alignment_pass": {
            "label": "Input alignment gate",
            "unit": "pass",
            "group": "Boundary",
            "direction": "descriptive",
        },
        "initial_gaussians": {
            "label": "Lifted Gaussian count",
            "unit": "gaussians",
            "group": "Capacity",
            "direction": "descriptive",
        },
        "training_loss": {
            "label": "RGB training objective",
            "unit": "loss",
            "group": "Objective",
            "direction": "lower",
        },
        "gaussian_count": {
            "label": "Live Gaussian count",
            "unit": "gaussians",
            "group": "Capacity",
            "direction": "descriptive",
        },
        "validation_psnr_fg": {
            "label": "Validation foreground PSNR",
            "unit": "dB",
            "group": "Validation quality",
            "direction": "higher",
        },
        "validation_psnr_full": {
            "label": "Validation full-canvas PSNR",
            "unit": "dB",
            "group": "Validation quality",
            "direction": "higher",
        },
        "validation_ssim_crop": {
            "label": "Validation crop SSIM",
            "unit": "score",
            "group": "Validation quality",
            "direction": "higher",
        },
        "validation_alpha_iou": {
            "label": "Validation alpha IoU",
            "unit": "score",
            "group": "Validation silhouette",
            "direction": "higher",
        },
        "validation_alpha_outside": {
            "label": "Validation exterior alpha",
            "unit": "mean alpha",
            "group": "Validation silhouette",
            "direction": "lower",
        },
        "validation_auc_foreground_psnr": {
            "label": "Validation foreground PSNR time-normalized AUC",
            "unit": "dB-seconds per second",
            "group": "Convergence",
            "direction": "higher",
        },
        "endpoint_frozen": {
            "label": "Endpoint frozen before held-out evaluation",
            "unit": "pass",
            "group": "Boundary",
            "direction": "descriptive",
        },
        "presentation_artifacts_written": {
            "label": "Cell presentation artifacts written",
            "unit": "pass",
            "group": "Boundary",
            "direction": "descriptive",
        },
    }
    stage_ids = [item["id"] for item in task["stages"]]
    for summary in summaries:
        dataset_id = str(summary["dataset_id"])
        arm = str(summary["arm"])
        seed = int(summary["seed"])
        cell = _cell_dir(ROOT / RUN_RELATIVE, dataset_id, seed, arm)
        raw = _load_json(cell / "training_history.raw.json")
        validation = _load_json(cell / "validation_metrics.json")["records"]
        iterations = int(summary["effective"]["rgb_refinement"]["iterations"])
        rgb_base = 2
        stage_steps = {
            "input_alignment": (0, 1),
            "field_lift": (1, 2),
            "rgb_refinement": (rgb_base, rgb_base + iterations),
            "validation_reporting": (rgb_base + iterations, rgb_base + iterations + 1),
            "heldout_evaluation": (rgb_base + iterations + 1, rgb_base + iterations + 2),
            "presentation": (rgb_base + iterations + 2, rgb_base + iterations + 3),
        }
        intervals = summary["stage_intervals"]
        for stage in stage_ids:
            start_step, end_step = stage_steps[stage]
            interval = intervals[stage]
            markers.extend(
                [
                    {
                        "step": start_step,
                        "wall_seconds": float(interval["start"]),
                        "stage": stage,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "boundary": "start",
                        "label": STAGE_LABELS[stage],
                    },
                    {
                        "step": end_step,
                        "wall_seconds": float(interval["end"]),
                        "stage": stage,
                        "dataset_id": dataset_id,
                        "arm_id": arm,
                        "seed": seed,
                        "boundary": "end",
                        "label": STAGE_LABELS[stage],
                    },
                ]
            )

        def record(
            step: int,
            wall: float,
            stage: str,
            split: str,
            metric: str,
            value: int | float,
            *,
            _dataset_id: str = dataset_id,
            _arm: str = arm,
            _seed: int = seed,
        ) -> None:
            records.append(
                {
                    "step": int(step),
                    "wall_seconds": float(wall),
                    "stage": stage,
                    "dataset_id": _dataset_id,
                    "arm_id": _arm,
                    "seed": _seed,
                    "split": split,
                    "metric_id": metric,
                    "value": float(value),
                }
            )

        record(
            1,
            intervals["input_alignment"]["end"],
            "input_alignment",
            "diagnostic",
            "input_alignment_pass",
            1,
        )
        record(
            2,
            intervals["field_lift"]["end"],
            "field_lift",
            "diagnostic",
            "initial_gaussians",
            summary["initial_gaussians"],
        )
        trainer_start = float(intervals["rgb_refinement"]["trainer_started_cell_wall_seconds"])
        losses = raw["loss"]
        selected_steps = {1, len(losses)} | set(range(10, len(losses) + 1, 10))
        for step in sorted(selected_steps):
            record(
                rgb_base + step,
                trainer_start + _elapsed_at_step(raw, step) + _callback_elapsed_at_step(raw, step),
                "rgb_refinement",
                "train",
                "training_loss",
                losses[step - 1],
            )
        for step, count in raw["n_gaussians"]:
            record(
                rgb_base + int(step),
                trainer_start
                + _elapsed_at_step(raw, int(step))
                + _callback_elapsed_at_step(raw, int(step)),
                "rgb_refinement",
                "diagnostic",
                "gaussian_count",
                count,
            )
        for item in validation:
            step = int(item["step"])
            wall = float(item["cell_wall_seconds"])
            for source, metric in (
                ("psnr_fg", "validation_psnr_fg"),
                ("psnr_full", "validation_psnr_full"),
                ("ssim_crop", "validation_ssim_crop"),
                ("alpha_iou", "validation_alpha_iou"),
                ("alpha_outside", "validation_alpha_outside"),
            ):
                record(
                    rgb_base + step,
                    wall,
                    "rgb_refinement",
                    "validation",
                    metric,
                    item["metrics"][source],
                )
        record(
            stage_steps["validation_reporting"][1],
            intervals["validation_reporting"]["end"],
            "validation_reporting",
            "validation",
            "validation_auc_foreground_psnr",
            summary["metrics"]["validation_auc_foreground_psnr"],
        )
        record(
            stage_steps["heldout_evaluation"][1],
            intervals["heldout_evaluation"]["end"],
            "heldout_evaluation",
            "diagnostic",
            "endpoint_frozen",
            1,
        )
        record(
            stage_steps["presentation"][1],
            intervals["presentation"]["end"],
            "presentation",
            "diagnostic",
            "presentation_artifacts_written",
            1,
        )
    return {
        "schema_version": 2,
        "records": records,
        "metric_metadata": metadata,
        "stage_markers": markers,
    }


def _viewer_command(task: Mapping[str, Any], dataset_id: str, *, port: int) -> list[str]:
    dataset = _dataset(task, dataset_id)
    launch = task["frozen_configuration"]["viewer_launch"]
    return [
        ".venv/bin/rtgs",
        "view",
        "--comparison-manifest",
        f"runs/{TASK_ID}/datasets/{dataset_id}/viewer_comparison.json",
        "--scene",
        dataset["frame_path"],
        "--downscale",
        str(task["frozen_configuration"]["image_downscale"]),
        "--rasterizer",
        "gsplat",
        "--device",
        str(launch["device"]),
        "--host",
        str(launch["host"]),
        "--port",
        str(port),
        "--open",
    ]


def _viewer_port(task: Mapping[str, Any], dataset_id: str) -> int:
    ports = task["frozen_configuration"]["viewer_launch"]["ports"]
    return int(ports[dataset_id])


def _pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _viewer_port_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _write_viewer_launch_receipt(run: Path, records: Sequence[Mapping[str, Any]]) -> None:
    complete = len(records) == len(DATASET_IDS) and all(
        item.get("alive_after_probe") is True and item.get("http_ready_after_probe") is True
        for item in records
    )
    _write_json(
        run / "viewer_launch_receipt.json",
        {
            "schema": "rtgs.janelle_gaussian2d_image_viewer_launch.v2",
            "status": "completed" if complete else "incomplete",
            "launched_after_all_measurement_endpoints": True,
            "updated_at_utc": _utc_now(),
            "records": records,
        },
    )


def _launch_viewers(run: Path, task: Mapping[str, Any]) -> None:
    launch = task["frozen_configuration"]["viewer_launch"]
    if launch.get("after_all_measurement_endpoints") is not True:
        raise RuntimeError("viewer launch must remain after all measurement endpoints")
    receipt_path = run / "viewer_launch_receipt.json"
    prior_by_dataset: dict[str, dict[str, Any]] = {}
    if receipt_path.is_file():
        try:
            prior = _load_json(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError):
            prior = {}
        if prior.get("schema") == "rtgs.janelle_gaussian2d_image_viewer_launch.v2" and isinstance(
            prior.get("records"), list
        ):
            prior_by_dataset = {
                item["dataset_id"]: item
                for item in prior["records"]
                if isinstance(item, dict) and isinstance(item.get("dataset_id"), str)
            }
    records: list[dict[str, Any]] = []
    for dataset_id in DATASET_IDS:
        command = _viewer_command(task, dataset_id, port=_viewer_port(task, dataset_id))
        log_path = run / "datasets" / dataset_id / "viewer.log"
        host = str(launch["host"])
        port = _viewer_port(task, dataset_id)
        previous = prior_by_dataset.get(dataset_id)
        if (
            isinstance(previous, dict)
            and previous.get("command") == command
            and _pid_alive(previous.get("pid"))
            and _viewer_port_ready(host, port)
        ):
            record = {
                **previous,
                "url": f"http://{host}:{port}",
                "alive_after_probe": True,
                "http_ready_after_probe": True,
                "reused": True,
                "error": None,
            }
            records.append(record)
            _write_viewer_launch_receipt(run, records)
            print(
                f"VIEWER_REUSED dataset={dataset_id} pid={record['pid']} url={record['url']}",
                flush=True,
            )
            continue
        process = None
        error_message = None
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except Exception as error:  # preserved in the retry receipt
                error_message = f"{type(error).__name__}: {error}"
        finally:
            log_handle.close()
        deadline = time.monotonic() + float(launch["startup_probe_seconds"])
        while process is not None and process.poll() is None and time.monotonic() < deadline:
            if _viewer_port_ready(host, port):
                break
            time.sleep(0.1)
        alive = process is not None and process.poll() is None
        http_ready = alive and _viewer_port_ready(host, port)
        record = {
            "dataset_id": dataset_id,
            "command": command,
            "pid": None if process is None else process.pid,
            "url": f"http://{host}:{port}",
            "log": log_path.relative_to(run).as_posix(),
            "alive_after_probe": alive,
            "http_ready_after_probe": http_ready,
            "reused": False,
            "error": error_message,
        }
        records.append(record)
        _write_viewer_launch_receipt(run, records)
        print(
            f"VIEWER_LAUNCHED dataset={dataset_id} pid={record['pid']} "
            f"url={record['url']} alive={str(alive).lower()} "
            f"http_ready={str(http_ready).lower()}",
            flush=True,
        )
    _write_viewer_launch_receipt(run, records)
    if not all(item["alive_after_probe"] and item["http_ready_after_probe"] for item in records):
        raise RuntimeError("one or more orbit viewers failed the process/HTTP startup probe")


def _write_viewer_manifest(
    run: Path, task: Mapping[str, Any], dataset_id: str, *, seed: int
) -> Path:
    directory = run / "datasets" / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    methods = []
    for arm in ARMS:
        cell = _cell_dir(run, dataset_id, seed, arm)
        methods.append(
            {
                "name": ARM_LABELS[arm],
                "initial": os.path.relpath(cell / "gaussians_init.ply", directory),
                "final": os.path.relpath(cell / "gaussians.ply", directory),
            }
        )
    path = directory / "viewer_comparison.json"
    _write_json(path, {"schema": "rtgs.viewer-comparison.v1", "methods": methods})
    return path


def _generate_previews(
    run: Path, task: Mapping[str, Any], dataset_id: str, *, seed: int
) -> dict[str, list[str]]:
    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.data.calibrated import load_calibrated_scene
    from rtgs.visualize import save_reconstruction_artifacts

    dataset = _dataset(task, dataset_id)
    partition = _partition(task, dataset_id)
    view_ids = [*partition["optimizer"], *partition["validation"], *partition["heldout"]]
    scene = load_calibrated_scene(
        ROOT / dataset["frame_path"],
        calibration_path=ROOT / dataset["calibration"],
        downscale=int(task["frozen_configuration"]["image_downscale"]),
        test_every=0,
        load_masks=True,
        undistort=True,
        view_ids=view_ids,
    )
    train_count = len(partition["optimizer"]) + len(partition["validation"])
    scene.train_indices = list(range(train_count))
    scene.test_indices = list(range(train_count, scene.n_views))
    outputs: dict[str, list[str]] = {}
    for arm in ARMS:
        cell = _cell_dir(run, dataset_id, seed, arm)
        initial = Gaussians3D.load_ply(cell / "gaussians_init.ply").to("cuda:0")
        final = Gaussians3D.load_ply(cell / "gaussians.ply").to("cuda:0")
        preview_dir = run / "datasets" / dataset_id / "previews" / arm
        saved = save_reconstruction_artifacts(
            scene,
            initial,
            final,
            preview_dir,
            rasterizer="gsplat",
            packed=True,
            antialiased=True,
            max_comparisons=3,
            max_animation_frames=24,
        )
        outputs[arm] = [Path(value).relative_to(run).as_posix() for value in saved.values()]
    return outputs


def _metric_metadata(task: Mapping[str, Any], metric_ids: Sequence[str]) -> dict[str, dict]:
    frozen = {item["id"]: item for item in task["primary_metrics"]}
    result = {}
    for metric_id in metric_ids:
        base = metric_id
        for prefix in (*ARMS, "masked_minus_unmasked"):
            marker = f"{prefix}_"
            if base.startswith(marker):
                base = base[len(marker) :]
                break
        item = frozen[base]
        result[metric_id] = {
            "label": f"{metric_id.replace('_', ' ')}",
            "unit": item["unit"],
            "group": (
                "Quality"
                if base.startswith("heldout") or base.startswith("validation")
                else "Resources"
            ),
            "direction": item["direction"],
        }
    return result


def _dataset_summary(
    run: Path,
    task: Mapping[str, Any],
    dataset_id: str,
    summaries: Sequence[Mapping[str, Any]],
    previews: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    cells = [item for item in summaries if item["dataset_id"] == dataset_id]
    metric_ids = [item["id"] for item in task["primary_metrics"]]
    final_metrics: dict[str, float] = {}
    curves = []
    for metric_id in metric_ids:
        series = []
        for arm in ARMS:
            arm_cells = sorted(
                (item for item in cells if item["arm"] == arm), key=lambda item: item["seed"]
            )
            values = [float(item["metrics"][metric_id]) for item in arm_cells]
            final_metrics[f"{arm}_{metric_id}"] = _median(values)
            series.append(
                {
                    "label": ARM_LABELS[arm],
                    "points": [
                        {"x": int(item["seed"]), "value": float(item["metrics"][metric_id])}
                        for item in arm_cells
                    ],
                }
            )
        curves.append(
            {
                "id": metric_id,
                "title": next(
                    item["label"] for item in task["primary_metrics"] if item["id"] == metric_id
                ),
                "x_label": "seed",
                "unit": next(
                    item["unit"] for item in task["primary_metrics"] if item["id"] == metric_id
                ),
                "direction": next(
                    item["direction"] for item in task["primary_metrics"] if item["id"] == metric_id
                ),
                "series": series,
            }
        )
    validation_curve_specs = (
        ("psnr_fg", "Validation foreground PSNR", "dB", "higher"),
        ("psnr_full", "Validation full-canvas PSNR", "dB", "higher"),
        ("ssim_crop", "Validation foreground-crop SSIM", "score", "higher"),
        ("alpha_iou", "Validation alpha IoU", "score", "higher"),
        ("alpha_outside", "Validation exterior alpha", "mean alpha", "lower"),
    )
    for source, title, unit, direction in validation_curve_specs:
        series = []
        for arm in ARMS:
            for item in sorted(
                (cell for cell in cells if cell["arm"] == arm), key=lambda cell: cell["seed"]
            ):
                validation = _load_json(
                    _cell_dir(run, dataset_id, int(item["seed"]), arm) / "validation_metrics.json"
                )["records"]
                series.append(
                    {
                        "label": f"{ARM_LABELS[arm]} · seed {item['seed']}",
                        "points": [
                            {
                                "x": float(record["optimizer_wall_seconds"]),
                                "value": float(record["metrics"][source]),
                            }
                            for record in validation
                        ],
                    }
                )
        curves.append(
            {
                "id": f"validation_{source}_native_optimizer_time",
                "title": f"{title} convergence",
                "x_label": "native optimizer time excluding validation observers (s)",
                "unit": unit,
                "direction": direction,
                "series": series,
            }
        )
    counts = _load_json(
        _cell_dir(run, dataset_id, int(task["seeds"][0]), ARMS[0]) / "field_lift.json"
    )["source_component_counts_all_views"]
    artifacts = [
        {
            "label": "Masked/unmasked orbit manifest",
            "path": f"datasets/{dataset_id}/viewer_comparison.json",
        },
        {"label": "Per-folder raw aggregate", "path": f"datasets/{dataset_id}/result.json"},
    ]
    for arm in ARMS:
        representative = _cell_dir(run, dataset_id, int(task["seeds"][0]), arm)
        artifacts.extend(
            [
                {
                    "label": f"{ARM_LABELS[arm]} initial PLY",
                    "path": (representative / "gaussians_init.ply").relative_to(run).as_posix(),
                },
                {
                    "label": f"{ARM_LABELS[arm]} final PLY",
                    "path": (representative / "gaussians.ply").relative_to(run).as_posix(),
                },
            ]
        )
        artifacts.extend(
            {"label": f"{ARM_LABELS[arm]} preview {index + 1}", "path": path}
            for index, path in enumerate(previews[arm])
        )
    charts = [
        {
            "id": "quality",
            "title": "Held-out foreground PSNR by arm",
            "unit": "dB",
            "values": [
                {
                    "label": ARM_LABELS[arm],
                    "value": final_metrics[f"{arm}_heldout_foreground_psnr"],
                }
                for arm in ARMS
            ],
        },
        {
            "id": "resources",
            "title": "Peak CUDA allocated by arm",
            "unit": "bytes",
            "values": [
                {
                    "label": ARM_LABELS[arm],
                    "value": final_metrics[f"{arm}_peak_cuda_allocated_bytes"],
                }
                for arm in ARMS
            ],
        },
        {
            "id": "stage_runtime",
            "title": "RGB refinement wall time by arm",
            "unit": "seconds",
            "values": [
                {
                    "label": ARM_LABELS[arm],
                    "value": final_metrics[f"{arm}_rgb_refinement_wall_seconds"],
                }
                for arm in ARMS
            ],
        },
    ]
    _write_json(
        run / "datasets" / dataset_id / "result.json",
        {
            "schema": "rtgs.janelle_gaussian2d_image_dataset_result.v1",
            "dataset_id": dataset_id,
            "source_component_counts": counts,
            "cells": [
                {
                    "seed": item["seed"],
                    "arm": item["arm"],
                    "metrics": item["metrics"],
                    "resource": item["resource"],
                }
                for item in cells
            ],
            "medians": final_metrics,
        },
    )
    return {
        "title": f"{dataset_id}: masked versus unmasked Janelle image refinement",
        "summary": (
            f"Independent experiment over this folder's 26 compact fields "
            f"({sum(counts.values()):,} total source 2D Gaussians), paired across three seeds."
        ),
        "metrics": final_metrics,
        "metric_metadata": _metric_metadata(task, list(final_metrics)),
        "charts": charts,
        "curves": curves,
        "artifacts": artifacts,
        "commands": {
            "viewer": _viewer_command(
                task,
                dataset_id,
                port=_viewer_port(task, dataset_id),
            )
        },
        "notes": [
            (
                "This page is one independent folder experiment; root-level medians are "
                "navigation summaries only."
            ),
            (
                "Seed 80601 was selected prospectively for previews and orbit display, never "
                "by quality."
            ),
            "Held-out views were opened only after every corresponding endpoint was saved.",
        ],
    }


def _environment() -> dict[str, Any]:
    import torch

    packages = {}
    for name in ("torch", "gsplat", "numpy", "Pillow", "realtime-gs"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "editable-or-unversioned"
    if torch.cuda.is_available():
        device = {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "cuda": str(torch.version.cuda),
        }
    else:
        device = {"type": "cpu", "name": platform.processor() or "unknown CPU", "cuda": None}
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": device,
    }


def _nvidia_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "exit_code": process.returncode,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }


def _write_result_evidence(
    task: Mapping[str, Any], dataset_summaries: Mapping[str, Any], summaries: Sequence[Mapping]
) -> None:
    json_path = ROOT / "benchmarks/results" / f"{TASK_ID}_RESULT.json"
    md_path = ROOT / "benchmarks/results" / f"{TASK_ID}_RESULT.md"
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "pending_independent_audit",
        "claim_boundary": task["claim_boundary"],
        "dataset_results": {
            dataset_id: {
                "medians": value["metrics"],
                "raw": f"runs/{TASK_ID}/datasets/{dataset_id}/result.json",
            }
            for dataset_id, value in dataset_summaries.items()
        },
        "cell_count": len(summaries),
        "failed_cell_count": 0,
        "interpretation": (
            "No positive or causal disposition before the independent scientist pass."
        ),
    }
    _write_json_exact(json_path, payload)
    lines = [
        f"# {task['title']}",
        "",
        "Status: **pending independent audit**.",
        "",
        "## Boundary",
        "",
        task["claim_boundary"],
        "",
        "## Raw result units",
        "",
        "| Folder | Masked held-out FG PSNR | Unmasked held-out FG PSNR | Raw |",
        "|---|---:|---:|---|",
    ]
    for dataset_id, value in dataset_summaries.items():
        metrics = value["metrics"]
        lines.append(
            f"| `{dataset_id}` | {metrics['masked_pipeline_heldout_foreground_psnr']:.6f} | "
            f"{metrics['unmasked_pipeline_heldout_foreground_psnr']:.6f} | "
            f"[JSON](../../runs/{TASK_ID}/datasets/{dataset_id}/result.json) |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The producer records the frozen outputs only. The canonical AUDIT files must "
                "recompute the metrics and dispose of every claim before these values are cited."
            ),
            "",
        ]
    )
    _write_text_exact(md_path, "\n".join(lines))


def _write_cell_bundle_receipt(run: Path, task: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    warmup = task["frozen_configuration"]["warmup"]
    identities = [
        {
            "dataset_id": str(warmup["dataset_id"]),
            "seed": int(warmup["seed"]),
            "arm": str(warmup["arm_id"]),
            "mode": "warmup",
        },
        *[
            {
                "dataset_id": dataset_id,
                "seed": int(seed),
                "arm": arm,
                "mode": "measured",
            }
            for dataset_id in DATASET_IDS
            for seed in task["seeds"]
            for arm in ARMS
        ],
    ]
    entries = []
    for identity in identities:
        _validate_cell_bundle(run=run, task=task, lock=lock, **identity)
        if identity["mode"] == "warmup":
            cell = run / "warmup" / identity["dataset_id"] / identity["arm"]
        else:
            cell = _cell_dir(
                run,
                identity["dataset_id"],
                identity["seed"],
                identity["arm"],
            )
        receipt = cell / "cell_receipt.json"
        entries.append(
            {
                **identity,
                "receipt_path": receipt.relative_to(run).as_posix(),
                "receipt_bytes": receipt.stat().st_size,
                "receipt_sha256": _sha256_file(receipt),
            }
        )
    _write_json(
        run / "cell_bundle_receipt.json",
        {
            "schema": "rtgs.janelle_gaussian2d_image_cell_bundle.v1",
            "task_id": TASK_ID,
            "protocol_sha256": lock["protocol_sha256"],
            "task_lock_sha256": _sha256_file(run / "task.lock.json"),
            "data_seal_sha256": lock["data_seal_sha256"],
            "source_binding_sha256": task["frozen_configuration"]["source_binding"][
                "aggregate_sha256"
            ],
            "warmup_cell_count": 1,
            "measured_cell_count": 36,
            "entries": entries,
        },
    )


def _aggregate_run(
    run: Path, task: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock = _load_json(run / "task.lock.json")
    summaries = []
    for dataset_id in DATASET_IDS:
        for seed in task["seeds"]:
            for arm in ARMS:
                summary = _validate_cell_bundle(
                    run=run,
                    task=task,
                    lock=lock,
                    dataset_id=dataset_id,
                    seed=int(seed),
                    arm=arm,
                    mode="measured",
                )
                summaries.append(summary)

    _write_cell_bundle_receipt(run, task, lock)

    representative_seed = int(task["seeds"][0])
    all_previews = {}
    for dataset_id in DATASET_IDS:
        _write_viewer_manifest(run, task, dataset_id, seed=representative_seed)
        print(f"PREVIEW_START dataset={dataset_id}", flush=True)
        all_previews[dataset_id] = _generate_previews(
            run, task, dataset_id, seed=representative_seed
        )
        print(f"PREVIEW_COMPLETE dataset={dataset_id}", flush=True)

    first_cell = _cell_dir(run, DATASET_IDS[0], representative_seed, ARMS[0])
    shutil.copy2(first_cell / "gaussians_init.ply", run / "gaussians_init.ply")
    shutil.copy2(first_cell / "gaussians.ply", run / "gaussians.ply")
    first_preview = run / "datasets" / DATASET_IDS[0] / "previews" / ARMS[0]
    for name in (
        "reconstruction_contact_sheet.png",
        "reconstruction.gif",
        "novel_orbit.gif",
        "novel_elevation.gif",
    ):
        shutil.copy2(first_preview / name, run / name)

    dataset_summaries = {
        dataset_id: _dataset_summary(
            run,
            task,
            dataset_id,
            summaries,
            all_previews[dataset_id],
        )
        for dataset_id in DATASET_IDS
    }
    metric_ids = [item["id"] for item in task["primary_metrics"]]
    root_metrics = {}
    for metric_id in metric_ids:
        folder_medians = []
        for dataset_id in DATASET_IDS:
            values = dataset_summaries[dataset_id]["metrics"]
            folder_medians.extend(
                [
                    values[f"masked_pipeline_{metric_id}"],
                    values[f"unmasked_pipeline_{metric_id}"],
                ]
            )
        root_metrics[metric_id] = _median(folder_medians)

    charts = [
        {
            "id": "quality",
            "title": "Held-out foreground PSNR per folder and arm",
            "unit": "dB",
            "values": [
                {
                    "label": f"{dataset_id} · {ARM_LABELS[arm]}",
                    "value": dataset_summaries[dataset_id]["metrics"][
                        f"{arm}_heldout_foreground_psnr"
                    ],
                }
                for dataset_id in DATASET_IDS
                for arm in ARMS
            ],
        },
        {
            "id": "resources",
            "title": "Peak CUDA allocated per folder and arm",
            "unit": "bytes",
            "values": [
                {
                    "label": f"{dataset_id} · {ARM_LABELS[arm]}",
                    "value": dataset_summaries[dataset_id]["metrics"][
                        f"{arm}_peak_cuda_allocated_bytes"
                    ],
                }
                for dataset_id in DATASET_IDS
                for arm in ARMS
            ],
        },
        {
            "id": "stage_runtime",
            "title": "RGB refinement wall time per folder and arm",
            "unit": "seconds",
            "values": [
                {
                    "label": f"{dataset_id} · {ARM_LABELS[arm]}",
                    "value": dataset_summaries[dataset_id]["metrics"][
                        f"{arm}_rgb_refinement_wall_seconds"
                    ],
                }
                for dataset_id in DATASET_IDS
                for arm in ARMS
            ],
        },
    ]
    evidence = [
        {
            "label": suffix.replace(".", " "),
            "path": f"benchmarks/results/{TASK_ID}_{suffix}",
        }
        for suffix in EVIDENCE_SUFFIXES
    ]
    artifacts = [
        {"label": "Representative initial Gaussians", "path": "gaussians_init.ply"},
        {"label": "Representative final Gaussians", "path": "gaussians.ply"},
        {"label": "Fitting history", "path": "training_history.json"},
        {"label": "Effective configuration", "path": "gaussians.config.json"},
        {"label": "Input-boundary receipt", "path": "input_boundary_receipt.json"},
        {"label": "Resource receipt", "path": "resource_receipt.json"},
        {"label": "Cell-bundle integrity receipt", "path": "cell_bundle_receipt.json"},
        {"label": "Live source/data integrity checks", "path": "integrity_checks.json"},
        {"label": "Orbit-viewer launch receipt", "path": "viewer_launch_receipt.json"},
        {"label": "Run receipt", "path": "run_receipt.json"},
        {"label": "Execution environment", "path": "environment.json"},
        {"label": "Representative contact sheet", "path": "reconstruction_contact_sheet.png"},
        {"label": "Representative calibrated animation", "path": "reconstruction.gif"},
        {"label": "Representative novel orbit", "path": "novel_orbit.gif"},
        {"label": "Representative novel elevation", "path": "novel_elevation.gif"},
    ]
    metrics = {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": TASK_ID,
        "summary": (
            "All six owner-selected Gaussian2D folders completed as independent, image-backed "
            "masked/unmasked Janelle experiments across three seeds; quantitative interpretation "
            "remains pending the canonical independent audit."
        ),
        "decision": "pending_independent_audit",
        "claim_boundary": task["claim_boundary"],
        "metrics": root_metrics,
        "metric_metadata": _metric_metadata(task, list(root_metrics)),
        "charts": charts,
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
            "viewer": _viewer_command(
                task,
                DATASET_IDS[0],
                port=_viewer_port(task, DATASET_IDS[0]),
            ),
        },
        "notes": [
            (
                "Every child dataset page is the scientific comparison unit; root metrics are "
                "the median of the twelve folder-by-arm medians for navigation only."
            ),
            (
                "The previous RTGS-012 analytic Gaussian-field numerator errors are not included "
                "as image quality metrics."
            ),
            (
                "Representative previews/viewers use prospectively selected seed 80601; every "
                "seed remains available in the raw cells."
            ),
        ],
        "dataset_summaries": dataset_summaries,
    }
    _write_json(run / "metrics.json", metrics)
    _write_json(run / "training_history.json", _history_bundle(task, summaries))
    _write_json(
        run / "gaussians.config.json",
        {
            "schema": "rtgs.janelle_gaussian2d_image_config.v1",
            "task_id": TASK_ID,
            "frozen_configuration": task["frozen_configuration"],
            "datasets": DATASET_IDS,
            "arms": ARMS,
            "seeds": task["seeds"],
            "representative_seed": representative_seed,
            "effective_cells": [item["effective"] for item in summaries],
        },
    )
    seal = _load_json(ROOT / task["data_seal"])
    _write_json(
        run / "input_boundary_receipt.json",
        {
            "schema": "rtgs.janelle_gaussian2d_image_input_boundary.v1",
            "status": "passed",
            "task_id": TASK_ID,
            "data_seal": task["data_seal"],
            "data_seal_sha256": _sha256_file(ROOT / task["data_seal"]),
            "sealed_file_count": len(seal["files"]),
            "sealed_input_bytes": sum(item["bytes"] for item in seal["files"]),
            "datasets": list(DATASET_IDS),
            "optimizer_views": task["frozen_configuration"]["optimizer_views"],
            "validation_views": task["frozen_configuration"]["validation_views"],
            "heldout_views": task["splits"][DATASET_IDS[0]]["heldout"],
            "measured_cell_count": len(summaries),
            "cell_bundle_receipt": "cell_bundle_receipt.json",
            "cell_receipts_validated_before_aggregation": True,
            "live_source_binding_verified_at_entry_and_exit": True,
            "complete_data_seal_verified_at_entry_and_exit": True,
            "all_cells_heldout_opened_after_endpoint_saved": all(
                item["heldout_opened_after_endpoint_saved"] for item in summaries
            ),
            "masked_cells": sum(item["masked_rgb_refinement"] for item in summaries),
            "unmasked_cells": sum(not item["masked_rgb_refinement"] for item in summaries),
            "compact_validation_or_heldout_views_opened_for_lifting": False,
            "rgb_or_mask_file_opened_inside_field_lifter": False,
        },
    )
    _write_json(
        run / "resource_receipt.json",
        {
            "schema": "rtgs.janelle_gaussian2d_image_resources.v1",
            "warmup_runs": 1,
            "measured_runs_per_folder_arm": 3,
            "measurement_scope": task["resource_protocol"]["scope"],
            "all_cell_resources_frozen_before_heldout": all(
                item["measurement_endpoint_before_heldout"] for item in summaries
            ),
            "nvidia_inventory": _nvidia_inventory(),
            "cells": [
                {
                    "dataset_id": item["dataset_id"],
                    "arm": item["arm"],
                    "seed": item["seed"],
                    **item["resource"],
                }
                for item in summaries
            ],
        },
    )
    _write_json(run / "environment.json", _environment())
    return dataset_summaries, summaries


def _failure_sources(
    run: Path, task: Mapping[str, Any], message: str, *, phase: str = "experiment_matrix"
) -> None:
    lock = _load_json(run / "task.lock.json")
    _write_json(
        run / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "failed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": _utc_now(),
            "exit_code": 1,
            "failure_phase": phase,
            "message": message,
        },
    )
    _write_json(
        run / "training_history.json",
        {"schema_version": 2, "records": [], "metric_metadata": {}, "stage_markers": []},
    )
    _write_json(
        run / "gaussians.config.json", {"frozen_configuration": task["frozen_configuration"]}
    )
    _write_json(run / "input_boundary_receipt.json", {"status": "failed", "message": message})
    _write_json(run / "resource_receipt.json", {"status": "partial"})
    _write_json(run / "environment.json", _environment())
    _write_json(
        run / "metrics.json",
        {
            "schema_version": 2,
            "report_template_version": 2,
            "task_id": TASK_ID,
            "summary": "The six-folder image-backed matrix did not complete.",
            "decision": "failed",
            "claim_boundary": task["claim_boundary"],
            "metrics": {},
            "metric_metadata": {},
            "charts": [],
            "artifacts": [
                {"label": "History", "path": "training_history.json"},
                {"label": "Config", "path": "gaussians.config.json"},
                {"label": "Boundary", "path": "input_boundary_receipt.json"},
                {"label": "Resources", "path": "resource_receipt.json"},
                {"label": "Run receipt", "path": "run_receipt.json"},
                {"label": "Environment", "path": "environment.json"},
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
            "notes": [message],
        },
    )


def _run_parent(task_path: Path, run: Path) -> int:
    task = _load_json(task_path)
    _assert_task(task)
    lock = _official_lock(task_path, run, task)
    entry_data_check = _verify_full_data_seal(task)
    entry_source_check = _verify_source_binding(task)
    worker_secret = secrets.token_hex(32)
    progress = {
        "schema": "rtgs.janelle_gaussian2d_image_progress.v1",
        "status": "running",
        "started_at_utc": lock["started_at_utc"],
        "completed_cells": [],
        "failed_cells": [],
    }
    _write_json(run / "progress.json", progress)

    warmup_config = task["frozen_configuration"]["warmup"]
    warmup_dir = run / "warmup" / warmup_config["dataset_id"] / warmup_config["arm_id"]
    warmup_identity = {
        "dataset_id": str(warmup_config["dataset_id"]),
        "seed": int(warmup_config["seed"]),
        "arm": str(warmup_config["arm_id"]),
        "mode": "warmup",
    }
    if warmup_dir.exists():
        _validate_cell_bundle(run=run, task=task, lock=lock, **warmup_identity)
        print("WARMUP_RESUME validated=true", flush=True)
    else:
        warmup_ticket = _write_worker_ticket(
            run=run,
            task_path=task_path,
            output=warmup_dir,
            iterations=int(warmup_config["iterations"]),
            secret=worker_secret,
            **{key: warmup_identity[key] for key in ("dataset_id", "seed", "arm", "mode")},
        )
        code = _run_subprocess(
            _worker_command(warmup_ticket),
            secret=worker_secret,
        )
        if code:
            _failure_sources(run, task, "warmup worker failed")
            return code
        _validate_cell_bundle(run=run, task=task, lock=lock, **warmup_identity)

    failures = []
    for dataset_index, dataset_id in enumerate(DATASET_IDS, start=1):
        print(f"DATASET_START {dataset_index}/{len(DATASET_IDS)} dataset={dataset_id}", flush=True)
        for seed in task["seeds"]:
            for arm in ARMS:
                cell = _cell_dir(run, dataset_id, int(seed), arm)
                identity = {"dataset_id": dataset_id, "arm": arm, "seed": int(seed)}
                if cell.exists():
                    try:
                        _validate_cell_bundle(
                            run=run,
                            task=task,
                            lock=lock,
                            mode="measured",
                            **identity,
                        )
                    except Exception as error:
                        failures.append(identity)
                        progress["failed_cells"].append(
                            {**identity, "reason": f"invalid existing cell: {error}"}
                        )
                        _write_json(run / "progress.json", progress)
                        continue
                    print(
                        f"CELL_RESUME dataset={dataset_id} arm={arm} seed={seed} "
                        "receipt_validated=true",
                        flush=True,
                    )
                    progress["completed_cells"].append(identity)
                    _write_json(run / "progress.json", progress)
                    continue
                ticket = _write_worker_ticket(
                    run=run,
                    task_path=task_path,
                    output=cell,
                    iterations=int(task["frozen_configuration"]["rgb_refinement"]["iterations"]),
                    mode="measured",
                    secret=worker_secret,
                    **identity,
                )
                code = _run_subprocess(
                    _worker_command(ticket),
                    secret=worker_secret,
                )
                if code:
                    failures.append(identity)
                    progress["failed_cells"].append(identity)
                else:
                    try:
                        _validate_cell_bundle(
                            run=run,
                            task=task,
                            lock=lock,
                            mode="measured",
                            **identity,
                        )
                    except Exception as error:
                        failures.append(identity)
                        progress["failed_cells"].append(
                            {**identity, "reason": f"post-worker receipt invalid: {error}"}
                        )
                    else:
                        progress["completed_cells"].append(identity)
                _write_json(run / "progress.json", progress)
        representative_seed = int(task["seeds"][0])
        if all(
            (_cell_dir(run, dataset_id, representative_seed, arm) / "gaussians.ply").is_file()
            for arm in ARMS
        ):
            manifest = _write_viewer_manifest(run, task, dataset_id, seed=representative_seed)
            print(
                f"DATASET_COMPLETE {dataset_index}/{len(DATASET_IDS)} dataset={dataset_id} "
                f"viewer_manifest={manifest.relative_to(ROOT)}",
                flush=True,
            )

    if failures:
        message = f"{len(failures)} measured cells failed; no aggregate metrics were imputed"
        _failure_sources(run, task, message)
        progress["status"] = "failed"
        _write_json(run / "progress.json", progress)
        return 1

    try:
        exit_data_check = _verify_full_data_seal(task)
        exit_source_check = _verify_source_binding(task)
        _write_json(
            run / "integrity_checks.json",
            {
                "schema": "rtgs.janelle_gaussian2d_image_integrity.v1",
                "task_id": TASK_ID,
                "entry_data": entry_data_check,
                "exit_data": exit_data_check,
                "entry_source": entry_source_check,
                "exit_source": exit_source_check,
                "source_unchanged": entry_source_check == exit_source_check,
                "data_seal_unchanged": (
                    entry_data_check["data_seal_sha256"] == exit_data_check["data_seal_sha256"]
                ),
            },
        )
        dataset_summaries, summaries = _aggregate_run(run, task)
        _launch_viewers(run, task)
        _write_result_evidence(task, dataset_summaries, summaries)
        progress["status"] = "completed"
        _write_json(run / "progress.json", progress)
        print("RUN_COMPLETE cells=36 datasets=6", flush=True)
        _write_json(
            run / "run_receipt.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "status": "completed",
                "started_at_utc": lock["started_at_utc"],
                "finished_at_utc": _utc_now(),
                "exit_code": 0,
                "failure_phase": None,
                "message": "All six folder experiments and all 36 measured cells completed.",
            },
        )
    except Exception as error:
        message = f"post-matrix publication failed: {type(error).__name__}: {error}"
        _failure_sources(
            run,
            task,
            message,
            phase="post_matrix_publication",
        )
        progress["status"] = "failed"
        progress["post_matrix_failure"] = message
        _write_json(run / "progress.json", progress)
        return 1
    return 0


def _scratch(args: argparse.Namespace) -> int:
    task_path = _task_path(args.task)
    task = _load_json(task_path)
    _assert_task(task, require_ready=False)
    output = Path(args.output).resolve()
    try:
        output.relative_to((ROOT / ".scratch").resolve())
    except ValueError as error:
        raise ValueError("scratch output must remain below .scratch/") from error
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scratch output: {output}")
    return _run_worker(
        task_path=task_path,
        output=output,
        dataset_id=args.dataset_id,
        seed=args.seed,
        arm=args.arm_id,
        iterations=int(args.iterations),
        mode="scratch",
        official_binding=None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        allow_abbrev=False,
    )
    parser.add_argument("--task", default=TASK_RELATIVE.as_posix())
    parser.add_argument("--run", default=RUN_RELATIVE.as_posix())
    parser.add_argument("--worker-ticket", default=None)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dataset-id", choices=DATASET_IDS, default=DATASET_IDS[0])
    parser.add_argument("--arm-id", choices=ARMS, default=ARMS[0])
    parser.add_argument("--seed", type=int, default=80601)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args(argv)

    if args.worker_ticket is not None:
        if args.scratch or args.output is not None or args.iterations is not None:
            parser.error("authenticated worker tickets cannot be combined with scratch controls")
        return _run_ticket_worker(args.worker_ticket)
    if args.scratch:
        if args.output is None or args.iterations is None:
            parser.error("scratch mode requires --output and --iterations")
        return _scratch(args)
    if args.output is not None or args.iterations is not None:
        parser.error("scratch-only arguments were supplied to the parent")
    return _run_parent(_task_path(args.task), _run_path(args.run))


if __name__ == "__main__":
    raise SystemExit(main())
