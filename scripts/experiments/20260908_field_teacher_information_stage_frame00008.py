"""Frozen A/B/C target-information experiment; every fitting cell is a fresh process."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("foreground_psnr", "full_psnr", "boundary_psnr", "crop_lpips", "alpha_iou")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def run_seconds(run: Path) -> float:
    start = dt.datetime.fromisoformat(read_json(run / "task.lock.json")["started_at_utc"])
    return time.time() - start.timestamp()


def source_guard(task_path: Path, task: dict, run: Path) -> None:
    lock = read_json(run / "task.lock.json")
    if lock["task_id"] != task["task_id"] or lock["task_sha256"] != sha256(task_path):
        raise RuntimeError("task bytes differ from the approved task lock")
    if task["status"] != "ready" or task["protocol_review"]["verdict"] != "approved":
        raise RuntimeError("protected execution requires an approved ready task")
    snapshot = read_json(run / "source_snapshot/manifest.json")
    for item in snapshot["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"source changed after snapshot: {item['path']}")


def data_guard(task: dict) -> dict:
    records = read_json(ROOT / task["data_seal"])["files"][:]
    for family in task["field_inputs"].values():
        records.extend(family["files"])
    unique = {item["path"]: item["sha256"] for item in records}
    for name, expected in unique.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"sealed input changed: {name}")
    return {"files_checked": len(unique), "all_match": True}


def access_guard(task: dict, phase: str, arm: str | None = None) -> dict:
    """Deny Python file opens at the loader boundary, and retain attempted violations."""
    frame = (ROOT / task["datasets"][0]["frame_path"]).resolve()
    targets = ROOT / "runs" / task["task_id"] / "targets"
    heldout = set(task["splits"]["frame_00008"]["heldout"])
    receipt = {"phase": phase, "arm": arm, "dataset_opens": [], "denied": []}

    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        if phase == "fit" and path.is_relative_to(targets):
            relative = path.relative_to(targets)
            if relative != Path("metadata.json") and relative.parts[0] != arm:
                receipt["denied"].append(str(path))
                raise PermissionError(f"{phase}/{arm}: forbidden target-cache open {relative}")
        if not path.is_relative_to(frame):
            return
        name = path.stem.removeprefix("mask_")
        forbidden = phase != "evaluate" and name in heldout
        if phase == "fit" and arm != "rgb":
            forbidden |= path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        if forbidden:
            receipt["denied"].append(str(path))
            raise PermissionError(f"{phase}/{arm}: forbidden source open {path.name}")
        receipt["dataset_opens"].append(str(path.relative_to(ROOT)))

    sys.addaudithook(audit)
    return receipt


def camera_record(camera: Camera) -> dict:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "R": camera.R.cpu().tolist(),
        "t": camera.t.cpu().tolist(),
    }


def scaled_camera(camera: Camera, factor: int) -> Camera:
    if camera.width % factor or camera.height % factor:
        raise ValueError("frozen quadrature requires divisible full-resolution dimensions")
    return Camera(
        camera.fx / factor,
        camera.fy / factor,
        camera.cx / factor,
        camera.cy / factor,
        camera.width // factor,
        camera.height // factor,
        camera.R.clone(),
        camera.t.clone(),
    )


def quadrature_sites(width: int, height: int, factor: int) -> torch.Tensor:
    """(H*W,4,2) full-resolution edge-origin sites, ordered y/x then dy/dx."""
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    base = torch.stack((x, y), -1).float().reshape(-1, 1, 2)
    offsets = torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    return (base + offsets[None]) * factor


def image_query(image: torch.Tensor, xy: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    height, width = image.shape[:2]
    source = image[..., None] if image.ndim == 2 else image
    grid = xy.to(source) * source.new_tensor([2 / width, 2 / height]) - 1
    result = F.grid_sample(
        source.permute(2, 0, 1)[None],
        grid[None, None],
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )
    return result[0, :, 0].T


def quadrature_image(image: torch.Tensor, sites: torch.Tensor, camera: Camera) -> torch.Tensor:
    return (
        image_query(image, sites.reshape(-1, 2)).reshape(camera.height, camera.width, 4, -1).mean(2)
    )


def binary_boundary(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    image = mask.float()[None, None]
    kernel = 2 * radius + 1
    dilation = F.max_pool2d(image, kernel, 1, radius)[0, 0] > 0.5
    padded_background = F.pad(1 - image, (radius, radius, radius, radius), value=1)
    erosion = 1 - F.max_pool2d(padded_background, kernel, 1)[0, 0]
    return dilation ^ (erosion > 0.5)


def lpips_model():
    import lpips

    return lpips.LPIPS(net="alex", pretrained=True, pnet_rand=False).eval().to("cuda:0")


def image_scores(prediction, reference, mask, perceptual, alpha=None) -> dict:
    prediction, reference = prediction.clamp(0, 1), reference.clamp(0, 1)
    mask = mask.bool()
    boundary = binary_boundary(mask)
    errors = (prediction - reference).square().mean(-1)

    def psnr(selected):
        if not bool(selected.any()):
            raise RuntimeError("empty frozen metric region")
        return float(-10 * torch.log10(errors[selected].mean().clamp_min(1e-12)))

    where = torch.nonzero(mask)
    y0, x0 = (where.min(0).values - 8).clamp_min(0).tolist()
    y1, x1 = (where.max(0).values + 9).tolist()
    x1, y1 = min(x1, mask.shape[1]), min(y1, mask.shape[0])
    left = prediction[y0:y1, x0:x1].permute(2, 0, 1)[None].to("cuda:0")
    right = reference[y0:y1, x0:x1].permute(2, 0, 1)[None].to("cuda:0")
    with torch.no_grad():
        distance = float(perceptual(left, right, normalize=True))
    result = {
        "foreground_psnr": psnr(mask),
        "full_psnr": psnr(torch.ones_like(mask)),
        "boundary_psnr": psnr(boundary),
        "crop_lpips": distance,
        "foreground_pixels": int(mask.sum()),
        "boundary_pixels": int(boundary.sum()),
        "crop_xyxy": [x0, y0, x1, y1],
    }
    if alpha is not None:
        visible = alpha >= 0.5
        result["alpha_iou"] = float((visible & mask).sum() / (visible | mask).sum())
    if not all(math.isfinite(result[key]) for key in METRICS if key in result):
        raise RuntimeError("nonfinite metric")
    return result


def load_source(task: dict, view: str) -> SceneData:
    from rtgs.data.calibrated import load_calibrated_scene

    scene = load_calibrated_scene(
        ROOT / task["datasets"][0]["frame_path"],
        calibration_path=ROOT / task["datasets"][0]["calibration"],
        downscale=1,
        view_ids=[view],
        test_every=0,
    )
    if scene.masks is None:
        raise RuntimeError(f"missing lossless foreground mask for {view}")
    return scene


def parity_sites(field, index: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(818100 + index)
    canvas = torch.rand(256, 2, generator=rng) * torch.tensor([field.width, field.height])
    x, y, width, height = field.fit_window or (0, 0, field.width, field.height)
    window = torch.rand(256, 2, generator=rng) * torch.tensor([width, height])
    window += torch.tensor([x, y])
    corners = torch.tensor(
        [[x, y], [x + width, y], [x, y + height], [x + width, y + height]], dtype=torch.float32
    )
    boundary = torch.cat([corners, corners - 0.001, corners + 0.001])
    return torch.cat([canvas, window, boundary])


def decode_field(task, arm, view, camera, sites, view_index):
    from rtgs.core.observation2d import GaussianObservationIndex
    from rtgs.core.observation2d_cuda import GaussianObservationIndexCuda
    from rtgs.data.compact_views import CompactView

    family = task["field_inputs"][arm]
    compact = CompactView.load(
        ROOT / family["directory"] / f"{view}.rtgsv", byte_cap=family["byte_cap"], load_alpha=False
    )
    if compact.alpha is not None:
        raise RuntimeError("field alpha was decoded")
    if camera_record(compact.camera) != camera_record(camera):
        raise RuntimeError(f"full camera mismatch for {arm}/{view}")
    field = compact.observation
    config = dict(task["preprocessing"].get("index_config", {}))
    query_chunk = config.pop("query_chunk", config.pop("cuda_query_chunk", 65536))
    index = GaussianObservationIndex(field, **config)
    cuda = GaussianObservationIndexCuda(index, device="cuda:0")
    samples = parity_sites(field, view_index)
    direct = torch.cat(
        [field.query(chunk, component_chunk=4096).color for chunk in samples.split(64)]
    )
    cpu = index.query(samples).color
    gpu = cuda.query(samples).color.cpu()
    direct_error = float((direct - cpu).abs().max())
    cuda_error = float((cpu - gpu).abs().max())
    if direct_error > 1e-5 or cuda_error > 2e-5:
        raise RuntimeError(f"field query parity failed {arm}/{view}: {direct_error}, {cuda_error}")
    flat = sites.reshape(-1, 2)
    decoded = torch.cat([cuda.query(chunk).color.cpu() for chunk in flat.split(query_chunk)])
    target = decoded.reshape(-1, 4, 3).mean(1).clamp(0, 1)
    x, y, width, height = field.fit_window or (0, 0, field.width, field.height)
    factor = task["preprocessing"]["downscale"]
    lower = sites[:, 0] - factor * 0.25
    upper = lower + factor
    intersects = (
        (upper[:, 0] > x)
        & (lower[:, 0] < x + width)
        & (upper[:, 1] > y)
        & (lower[:, 1] < y + height)
    )
    selected = torch.nonzero(intersects)[:128, 0]
    if selected.numel() < 128:
        raise RuntimeError("fewer than128 decoded parity pixels intersect fit window")
    queried = index.query(sites[selected].reshape(-1, 2)).color.reshape(-1, 4, 3).mean(1)
    decode_error = float((queried.clamp(0, 1) - target[selected]).abs().max())
    if decode_error > 2e-5:
        raise RuntimeError("decoded teacher target differs from sampled teacher")
    record = {
        "arm": arm,
        "view_id": view,
        "n_gaussians": field.n,
        "direct_index_max_abs": direct_error,
        "cpu_cuda_max_abs": cuda_error,
        "decoded_sample_max_abs": decode_error,
        "sample_count": len(samples),
        "decoded_sample_pixels": len(selected),
        "camera_exact": True,
        "alpha_decoded": False,
        "compact_sha256": compact.sha256,
    }
    downscaled = scaled_camera(camera, task["preprocessing"]["downscale"])
    return target.reshape(downscaled.height, downscaled.width, 3), record


def prepare(task: dict, run: Path) -> None:
    from rtgs.data.calibrated import _object_bounds

    started = run_seconds(run)
    access = access_guard(task, "prepare")
    perceptual = lpips_model()
    train = task["splits"]["frame_00008"]["train"]
    target_dir = run / "targets"
    target_dir.mkdir()
    full_cameras, full_masks, cameras = [], [], []
    teacher_metrics = {"field_high": [], "field_low": []}
    parity = []
    for view_index, view in enumerate(train):
        print(f"prepare {view_index + 1}/{len(train)} {view}", flush=True)
        source = load_source(task, view)
        camera = source.cameras[0]
        mask = source.masks[0] > 0.5
        image = source.images[0] * mask[..., None]
        output_camera = scaled_camera(camera, task["preprocessing"]["downscale"])
        sites = quadrature_sites(
            output_camera.width, output_camera.height, task["preprocessing"]["downscale"]
        )
        rgb = quadrature_image(image, sites, output_camera).clamp(0, 1)
        out_mask = quadrature_image(mask.float(), sites, output_camera)[..., 0] >= 0.5
        full_cameras.append(camera)
        full_masks.append(mask)
        cameras.append(camera_record(output_camera))
        for arm in ("rgb", "field_high", "field_low"):
            target = rgb
            if arm != "rgb":
                target, record = decode_field(task, arm, view, camera, sites, view_index)
                parity.append(record)
                teacher_metrics[arm].append(
                    {"view_id": view, **image_scores(target, rgb, out_mask, perceptual)}
                )
            directory = target_dir / arm
            directory.mkdir(exist_ok=True)
            np.savez(directory / f"{view}.npz", color=target.numpy())
        masks_dir = target_dir / "initialization_masks"
        masks_dir.mkdir(exist_ok=True)
        np.savez(masks_dir / f"{view}.npz", mask=out_mask.numpy())
    center, extent = _object_bounds(full_cameras, full_masks)
    metadata = {
        "view_ids": train,
        "cameras": cameras,
        "bounds": {"center": center.tolist(), "extent": extent},
    }
    write_json(target_dir / "metadata.json", metadata)
    high = teacher_metrics["field_high"]
    high_psnr = float(np.mean([row["foreground_psnr"] for row in high]))
    high_lpips = float(np.mean([row["crop_lpips"] for row in high]))
    write_json(
        run / "preparation.json",
        {
            "teacher_metrics": teacher_metrics,
            "parity": parity,
            "teacher_qualified": high_psnr >= 30 and high_lpips <= 0.08,
            "high_mean_foreground_psnr": high_psnr,
            "high_mean_crop_lpips": high_lpips,
            "train_view_ids": train,
            "stage_wall_seconds": run_seconds(run) - started,
            "stage_intervals": {"prepare": [started, run_seconds(run)]},
            "access_guard": access,
            "cpu_selftest": selftest(),
            "bounds": metadata["bounds"],
        },
    )


def cached_scene(run: Path, arm: str, with_masks: bool = False) -> SceneData:
    metadata = read_json(run / "targets/metadata.json")
    images, masks = [], []
    for view in metadata["view_ids"]:
        with np.load(run / "targets" / arm / f"{view}.npz") as data:
            images.append(torch.from_numpy(data["color"].copy()))
        if with_masks:
            with np.load(run / "targets/initialization_masks" / f"{view}.npz") as data:
                masks.append(torch.from_numpy(data["mask"].copy()))
    scene = SceneData(
        images,
        [Camera(**record) for record in metadata["cameras"]],
        view_names=metadata["view_ids"],
        masks=masks if with_masks else None,
        train_indices=list(range(len(images))),
        test_indices=[],
        bounds_hint=(torch.tensor(metadata["bounds"]["center"]), metadata["bounds"]["extent"]),
        name="frame_00008",
    )
    scene.validate()
    return scene


def initialize(task: dict, run: Path) -> None:
    started = run_seconds(run)
    access = access_guard(task, "initialize")
    scene = cached_scene(run, "rgb", with_masks=True)
    center, extent = scene.center_and_extent()
    resolution = task["initialization"]["grid_resolution"]
    step = extent / resolution
    coordinates = (torch.arange(resolution).float() + 0.5) * step - extent / 2
    lattice = torch.stack(torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), -1)
    points = lattice.reshape(-1, 3) + center
    occupied = torch.ones(len(points), dtype=torch.bool)
    for camera, mask in zip(scene.cameras, scene.masks):
        dilated = F.max_pool2d(mask.float()[None, None], 3, 1, 1)[0, 0]
        for offset in range(0, len(points), 131072):
            uv, depth = camera.project(points[offset : offset + 131072])
            supported = image_query(dilated, uv, mode="nearest")[:, 0] > 0.5
            occupied[offset : offset + 131072] &= supported & (depth > 0) & camera.in_image(uv)
    volume = occupied.reshape(resolution, resolution, resolution)
    faces = [volume[0], volume[-1], volume[:, 0], volume[:, -1], volume[:, :, 0], volume[:, :, -1]]
    if any(bool(face.any()) for face in faces):
        raise RuntimeError("visual hull touches bounds face; frozen initialization aborted")
    interior = -F.max_pool3d(-volume.float()[None, None], 3, 1, 1)[0, 0] > 0.5
    shell = volume & ~interior
    candidates = points[shell.reshape(-1)]
    if len(candidates) < task["initialization"]["minimum_points"]:
        raise RuntimeError(f"visual hull shell has only{len(candidates)} points")
    if not torch.isfinite(candidates).all():
        raise RuntimeError("visual hull nonfinite geometry")
    directory = run / "initialization"
    directory.mkdir()
    seeds = {}
    for seed in task["seeds"]:
        selected = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed))
        means = candidates[selected[: task["initialization"]["n_max"]]]
        colors = torch.zeros(len(means), 3)
        for camera, image in zip(scene.cameras, scene.images):
            uv, _ = camera.project(means)
            colors += image_query(image, uv)
        colors = (colors / scene.n_views).clamp(0, 1)
        model = Gaussians3D.from_means_covs(
            means,
            torch.eye(3).expand(len(means), 3, 3) * step**2,
            colors,
            torch.full((len(means),), 0.1),
        )
        path = directory / str(seed)
        path.mkdir()
        model.save_npz(path / "gaussians_init.npz")
        model.save_ply(path / "gaussians_init.ply")
        seeds[str(seed)] = {"n_gaussians": model.n, "sha256": sha256(path / "gaussians_init.npz")}
    write_json(
        run / "initialization.json",
        {
            "seeds": seeds,
            "occupied_voxels": int(volume.sum()),
            "shell_voxels": len(candidates),
            "voxel_size": step,
            "bounds": {"center": center.tolist(), "extent": extent},
            "touches_bounds": False,
            "access_guard": access,
            "stage_wall_seconds": run_seconds(run) - started,
            "stage_intervals": {"initialize": [started, run_seconds(run)]},
        },
    )


def warmup() -> None:
    from rtgs.render.base import get_rasterizer

    camera = Camera(32, 32, 16, 16, 32, 32, torch.eye(3), torch.zeros(3)).to("cuda:0")
    means = torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.1, 2.5]], device="cuda:0", requires_grad=True)
    model = Gaussians3D.from_means_covs(
        means,
        torch.eye(3, device="cuda:0").repeat(2, 1, 1) * 0.01,
        torch.full((2, 3), 0.5, device="cuda:0"),
        torch.full((2,), 0.2, device="cuda:0"),
    )
    get_rasterizer("gsplat", device="cuda:0").render(model, camera).color.mean().backward()
    torch.cuda.synchronize()


def train_config(task: dict, seed: int):
    from rtgs.optim.density import DensityConfig
    from rtgs.optim.trainer import TrainConfig

    frozen = dict(task["training"])
    frozen.pop("other_defaults")
    frozen["density"] = DensityConfig(**frozen["density"])
    frozen.setdefault("reset_cuda_peak_stats", False)
    frozen.setdefault("validate_render_finite", True)
    return TrainConfig(**frozen, seed=seed)


def fit(task: dict, run: Path, arm: str, seed: int) -> None:
    from rtgs.optim.trainer import Trainer

    access = access_guard(task, "fit", arm)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no CPU fallback in this protocol")
    warmup()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = run_seconds(run)
    output = run / "cells" / arm / str(seed)
    output.mkdir(parents=True)
    receipt = {
        "arm": arm,
        "seed": seed,
        "rng_seed": seed,
        "status": "running",
        "access_guard": access,
    }
    try:
        scene = cached_scene(run, arm)
        if scene.masks is not None or scene.testing_views:
            raise RuntimeError("fitting cache contains masks or heldout data")
        initial_path = run / "initialization" / str(seed) / "gaussians_init.npz"
        initial = Gaussians3D.load_npz(initial_path)
        expected = read_json(run / "initialization.json")["seeds"][str(seed)]["sha256"]
        receipt["initial_sha256"] = sha256(initial_path)
        if receipt["initial_sha256"] != expected:
            raise RuntimeError("shared initialization digest mismatch")
        shutil.copyfile(initial_path, output / "gaussians_init.npz")
        initial.save_ply(output / "gaussians_init.ply")
        config = train_config(task, seed)
        write_json(
            output / "effective_config.json",
            {
                "train_config": asdict(config),
                "scene_bounds": read_json(run / "targets/metadata.json")["bounds"],
                "train_view_ids": scene.view_names,
                "mask_input": None,
            },
        )
        checkpoints = []

        def checkpoint(model, step):
            for tensor in (model.means, model.quats, model.log_scales, model.opacity, model.sh):
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"nonfinite model at step{step}")
            checkpoints.append({"step": step, "run_seconds": run_seconds(run), "n": model.n})
            print(f"{arm}/{seed} step={step} n={model.n}", flush=True)

        trainer_start = run_seconds(run)
        final, history = Trainer(config).train(scene, initial, checkpoint_callback=checkpoint)
        torch.cuda.synchronize()
        if history["executed_iterations"] != task["training"]["iterations"]:
            raise RuntimeError("trainer did not reach frozen final iteration")
        if not all(math.isfinite(value) for value in history["loss"]):
            raise RuntimeError("nonfinite fitting loss")
        final.save_npz(output / "gaussians.npz")
        final.save_ply(output / "gaussians.ply")
        history["trainer_started_run_seconds"] = trainer_start
        history["checkpoint_observer"] = checkpoints
        write_json(output / "history.json", history)
        receipt.update(
            status="completed", final_count=final.n, final_sha256=sha256(output / "gaussians.npz")
        )
    except BaseException as error:
        receipt.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        torch.cuda.synchronize()
        ended = run_seconds(run)
        receipt.update(
            wall_seconds=ended - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            ru_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ru_maxrss_unit="KiB",
            stage_intervals={"fit": [started, ended]},
        )
        write_json(output / "receipt.json", receipt)


def evaluate(task: dict, run: Path) -> None:
    from rtgs.render.base import get_rasterizer
    from rtgs.visualize import save_reconstruction_artifacts

    for arm, seed in task["execution_order"]["cells"]:
        output = run / "cells" / arm / str(seed)
        if read_json(output / "receipt.json")["status"] != "completed":
            raise RuntimeError("evaluation requires every final model to be saved")
    access = access_guard(task, "evaluate")
    heldout = task["splits"]["frame_00008"]["heldout"]
    images, masks, cameras = [], [], []
    for view in heldout:
        source = load_source(task, view)
        camera = scaled_camera(source.cameras[0], task["preprocessing"]["downscale"])
        sites = quadrature_sites(camera.width, camera.height, task["preprocessing"]["downscale"])
        raw_mask = source.masks[0] > 0.5
        images.append(
            quadrature_image(source.images[0] * raw_mask[..., None], sites, camera).clamp(0, 1)
        )
        masks.append(quadrature_image(raw_mask.float(), sites, camera)[..., 0] >= 0.5)
        cameras.append(camera)
    metadata = read_json(run / "targets/metadata.json")
    # References already contain the frozen premultiplied quadrature; do not mask them twice.
    scene = SceneData(
        images,
        cameras,
        view_names=heldout,
        train_indices=[],
        test_indices=list(range(len(heldout))),
        masks=None,
        bounds_hint=(torch.tensor(metadata["bounds"]["center"]), metadata["bounds"]["extent"]),
    )
    perceptual = lpips_model()
    renderer = get_rasterizer("gsplat", device="cuda:0")
    for arm, seed in task["execution_order"]["cells"]:
        started = run_seconds(run)
        output = run / "cells" / arm / str(seed)
        final = Gaussians3D.load_npz(output / "gaussians.npz").to("cuda:0")
        initial = Gaussians3D.load_npz(output / "gaussians_init.npz").to("cuda:0")
        rows = []
        with torch.no_grad():
            for view, camera, reference, mask in zip(heldout, cameras, images, masks):
                rendered = renderer.render(final, camera.to("cuda:0"))
                rows.append(
                    {
                        "view_id": view,
                        **image_scores(
                            rendered.color.cpu(), reference, mask, perceptual, rendered.alpha.cpu()
                        ),
                    }
                )
        artifacts = save_reconstruction_artifacts(
            scene,
            initial,
            final,
            output,
            rasterizer="gsplat",
            max_comparisons=len(heldout),
            max_animation_frames=24,
        )
        means = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
        write_json(
            output / "evaluation.json",
            {
                "per_view": rows,
                "mean": means,
                "artifacts": artifacts,
                "stage_wall_seconds": run_seconds(run) - started,
                "stage_intervals": {"evaluate": [started, run_seconds(run)]},
                "access_guard": access,
            },
        )
        print(f"evaluated {arm}/{seed}", flush=True)


def selftest() -> dict:
    """CPU mechanics only: edge-origin quadrature, silhouette band and equal-target gradients."""
    from rtgs.core.metrics import ssim
    from rtgs.render.base import get_rasterizer

    torch.manual_seed(53)
    camera = Camera(16, 16, 8, 8, 16, 16, torch.eye(3), torch.zeros(3))
    sites = quadrature_sites(2, 2, 8)
    assert torch.equal(sites[0], torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]]))
    y, x = torch.meshgrid(torch.arange(16) + 0.5, torch.arange(16) + 0.5, indexing="ij")
    ramp = torch.stack((x / 16, y / 16, x * 0 + 0.2), -1)
    sampled = quadrature_image(ramp, sites, scaled_camera(camera, 8))
    assert torch.allclose(sampled[0, 0], torch.tensor([0.25, 0.25, 0.2]))
    uv, _ = camera.project(torch.tensor([[0.2, 0.4, 2.0]]))
    small_uv, _ = scaled_camera(camera, 8).project(torch.tensor([[0.2, 0.4, 2.0]]))
    assert torch.equal(uv / 8, small_uv)
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[4:12, 4:12] = True
    band = binary_boundary(mask, 1)
    assert band.sum() == 64 and not bool(band[7, 7]) and bool(band[3, 4])
    assert binary_boundary(torch.ones_like(mask), 1).sum() == 60
    renderer = get_rasterizer("torch", device="cpu")
    target = torch.rand(16, 16, 3)
    results = []
    for label in ("direct_target", "decoded_identical_target"):
        means = torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.1, 2.5]])
        model = Gaussians3D.from_means_covs(
            means,
            torch.eye(3).repeat(2, 1, 1) * 0.02,
            torch.full((2, 3), 0.5),
            torch.full((2,), 0.2),
        )
        parameters = (model.means, model.quats, model.log_scales, model.opacity, model.sh)
        for parameter in parameters:
            parameter.requires_grad_(True)
        rendered = renderer.render(model, camera).color
        value = target if label == "direct_target" else target.clone()
        loss = 0.8 * (rendered - value).abs().mean() + 0.2 * (1 - ssim(rendered, value))
        gradients = torch.autograd.grad(loss, parameters)
        results.append((loss.detach(), gradients))
    assert torch.equal(results[0][0], results[1][0])
    assert all(torch.equal(left, right) for left, right in zip(results[0][1], results[1][1]))
    fake_task = {
        "task_id": "__field_teacher_guard_selftest__",
        "datasets": [{"frame_path": ".scratch/__field_teacher_guard_selftest__"}],
        "splits": {"frame_00008": {"heldout": ["C0001"]}},
    }
    guarded = access_guard(fake_task, "fit", "field_high")
    probes = [
        ROOT / fake_task["datasets"][0]["frame_path"] / "rgb/C0002.jpg",
        ROOT / fake_task["datasets"][0]["frame_path"] / "mask/mask_C0002.png",
        ROOT / fake_task["datasets"][0]["frame_path"] / "fields/C0001.rtgsv",
        ROOT / "runs" / fake_task["task_id"] / "targets/rgb/C0002.npz",
        ROOT / "runs" / fake_task["task_id"] / "targets/initialization_masks/C0002.npz",
    ]
    for path in probes:
        try:
            path.open("rb")
        except PermissionError:
            pass
        else:
            raise AssertionError(f"input boundary failed for {path}")
    assert len(guarded["denied"]) == len(probes)
    return {
        "quadrature": "passed",
        "projection": "passed",
        "mask_boundary": "passed",
        "equal_target_loss": "exact",
        "equal_target_parameter_gradients": "exact for means,quats,log_scales,opacity,SH",
        "field_worker_input_guard": "five forbidden open probes denied",
    }


def report_module():
    path = Path(__file__).with_name(Path(__file__).stem + "_report.py")
    spec = importlib.util.spec_from_file_location("field_teacher_report", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def coordinate(task_path: Path, task: dict, run: Path) -> None:
    entry = data_guard(task)
    write_json(run / "input_integrity_entry.json", entry)
    split_digest = hashlib.sha256(json.dumps(task["splits"], sort_keys=True).encode()).hexdigest()
    logs = run / "logs"
    logs.mkdir(exist_ok=True)

    def worker(phase, arm=None, seed=None):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            phase,
            "--task",
            str(task_path),
            "--run-dir",
            str(run),
        ]
        name = phase
        if arm is not None:
            command.extend(["--arm", arm, "--seed", str(seed)])
            name += f"_{arm}_{seed}"
        with (logs / f"{name}.log").open("w") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "PYTHONUNBUFFERED": "1",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=3600 if phase == "fit" else None,
                check=True,
            )

    try:
        worker("prepare")
        worker("initialize")
        for arm, seed in task["execution_order"]["cells"]:
            print(f"starting {arm}/{seed}", flush=True)
            worker("fit", arm, seed)
        worker("evaluate")
        source_guard(task_path, task, run)
        write_json(
            run / "input_integrity_exit.json", {**data_guard(task), "split_sha256": split_digest}
        )
        report_module().publish(task, run)
    except BaseException as error:
        failure = {
            "error": str(error),
            "traceback": traceback.format_exc(),
            "run_seconds": run_seconds(run),
            "split_sha256": split_digest,
        }
        write_json(run / "execution_failure.json", failure)
        report_module().publish_failure(task, run, str(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("run", "prepare", "initialize", "fit", "evaluate", "selftest")
    )
    parser.add_argument("--task", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--arm", choices=("rgb", "field_high", "field_low"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "selftest":
        print(json.dumps(selftest()))
        return
    if args.task is None or args.run_dir is None:
        parser.error("--task and --run-dir are required")
    task_path, run = args.task.resolve(), args.run_dir.resolve()
    task = read_json(task_path)
    if run != ROOT / "runs" / task["task_id"]:
        raise ValueError("only the canonical task run root is allowed")
    source_guard(task_path, task, run)
    if args.command == "run":
        coordinate(task_path, task, run)
    elif args.command == "prepare":
        prepare(task, run)
    elif args.command == "initialize":
        initialize(task, run)
    elif args.command == "fit":
        if [args.arm, args.seed] not in task["execution_order"]["cells"]:
            parser.error("fit cell is not registered")
        fit(task, run, args.arm, args.seed)
    elif args.command == "evaluate":
        evaluate(task, run)


if __name__ == "__main__":
    main()
