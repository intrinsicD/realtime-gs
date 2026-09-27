"""RTGS-026 active silhouette-hull floater relocation screen; every cell is a fresh process.

Conditions cross the target family (decoded no_boundary compact fields or photographs) with
opt-in all-mask floater relocation. All calibrated masks (training and held-out) define the
visual hull; colour targets come only from the 24 training views. Colour is scored inside the
two held-out masks; held-out alpha metrics are in-sample because those masks shape the hull.
A conditional ``production`` phase trains the relocation arm on all views after a passing gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
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
DATASET = "frame_00008"
FIELD = "no_boundary"
PHOTOGRAPHS = "photographs"
MASKS = "training_masks"
HULL = "hull"
CONDITIONS = {
    "nb_ms": (FIELD, False),
    "nb_ms_reloc": (FIELD, True),
    "ph_ms": (PHOTOGRAPHS, False),
    "ph_ms_reloc": (PHOTOGRAPHS, True),
}
METRICS = (
    "foreground_psnr",
    "crop_lpips",
    "outside_alpha_mass",
    "floater_fraction",
    "interior_alpha",
    "hull_rejected_fraction",
)
BAND = 3  # output-pixel dilation/erosion radius for held-out alpha bands
PARITY_SITES = 512


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
    unique: dict[str, str] = {}
    for item in records:
        if unique.setdefault(item["path"], item["sha256"]) != item["sha256"]:
            raise RuntimeError(f"conflicting sealed digests for {item['path']}")
    for name, expected in unique.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"sealed input changed: {name}")
    return {"files_checked": len(unique), "all_match": True}


def quadrature_sites(width: int, height: int, factor: int) -> torch.Tensor:
    """(H*W,4,2) edge-origin full-resolution sites; identical to field_targets(supersample=2)."""
    from rtgs.data.field_targets import quadrature_sites as sites

    return sites(width, height, factor, 2)


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


def dilate(mask: torch.Tensor, radius: int = BAND) -> torch.Tensor:
    kernel = 2 * radius + 1
    return F.max_pool2d(mask.float()[None, None], kernel, 1, radius)[0, 0] > 0.5


def erode(mask: torch.Tensor, radius: int = BAND) -> torch.Tensor:
    kernel = 2 * radius + 1
    background = F.pad(1 - mask.float()[None, None], (radius,) * 4, value=1)
    return F.max_pool2d(background, kernel, 1)[0, 0] < 0.5


def mask_scores(prediction, alpha, reference, mask, perceptual=None) -> dict:
    """Colour inside the held-out mask only; alpha outside/inside bands for floaters/coverage."""
    prediction, reference = prediction.clamp(0, 1), reference.clamp(0, 1)
    mask = mask.bool()
    if not bool(mask.any()):
        raise RuntimeError("empty held-out mask")
    errors = (prediction - reference).square().mean(-1)
    outside = ~dilate(mask)
    interior = erode(mask)
    if not bool(outside.any()) or not bool(interior.any()):
        raise RuntimeError("empty frozen alpha band")
    result = {
        "foreground_psnr": float(-10 * torch.log10(errors[mask].mean().clamp_min(1e-12))),
        "outside_alpha_mass": float(alpha[outside].mean()),
        "floater_fraction": float((alpha[outside] >= 0.5).float().mean()),
        "interior_alpha": float(alpha[interior].mean()),
        "foreground_pixels": int(mask.sum()),
        "outside_pixels": int(outside.sum()),
        "interior_pixels": int(interior.sum()),
    }
    where = torch.nonzero(mask)
    y0, x0 = (where.min(0).values - 8).clamp_min(0).tolist()
    y1, x1 = (where.max(0).values + 9).tolist()
    x1, y1 = min(x1, mask.shape[1]), min(y1, mask.shape[0])
    result["crop_xyxy"] = [x0, y0, x1, y1]
    if perceptual is not None:
        weight = mask[..., None].to(prediction)
        left = (prediction * weight)[y0:y1, x0:x1].permute(2, 0, 1)[None]
        right = (reference * weight)[y0:y1, x0:x1].permute(2, 0, 1)[None]
        device = next(perceptual.parameters()).device
        with torch.no_grad():
            result["crop_lpips"] = float(
                perceptual(left.to(device), right.to(device), normalize=True)
            )
    for key in METRICS:
        if key in result and not math.isfinite(result[key]):
            raise RuntimeError(f"nonfinite metric {key}")
    return result


def lpips_model():
    import lpips

    return lpips.LPIPS(net="alex", pretrained=True, pnet_rand=False).eval().to("cuda:0")


def random_initialization(center: torch.Tensor, extent: float, count: int, seed: int) -> dict:
    """Uniform ball of radius extent/2; isotropic scale = mean nearest-neighbour distance."""
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    direction = direction / direction.norm(dim=1, keepdim=True)
    radius = torch.rand(count, 1, generator=generator, dtype=torch.float64) ** (1 / 3)
    means = (center.double() + 0.5 * extent * radius * direction).float()
    nearest = torch.empty(count)
    for start in range(0, count, 2048):
        distances = torch.cdist(means[start : start + 2048], means)
        distances[torch.arange(distances.shape[0]), torch.arange(start, start + len(distances))] = (
            math.inf
        )
        nearest[start : start + 2048] = distances.min(1).values
    return {"means": means, "scale": float(nearest.mean())}


def model_from(means: torch.Tensor, scale: float, colors: torch.Tensor) -> Gaussians3D:
    return Gaussians3D.from_means_covs(
        means,
        torch.eye(3).expand(len(means), 3, 3) * scale**2,
        colors.clamp(0, 1),
        torch.full((len(means),), 0.1),
    )


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


def load_view(task: dict, family: str, view: str, *, load_alpha: bool):
    from rtgs.data.compact_views import CompactView

    spec = task["field_inputs"][family]
    return CompactView.load(
        ROOT / spec["directory"] / f"{view}.rtgsv", byte_cap=spec["byte_cap"], load_alpha=load_alpha
    )


def parity_record(view, seed: int) -> dict:
    """Frozen per-view decoder parity: CUDA vs CPU index and CPU index vs reference scan."""
    from rtgs.data.field_targets import make_query_backend

    field = view.observation
    x, y, width, height = field.fit_window
    generator = torch.Generator().manual_seed(seed)
    sites = torch.rand(PARITY_SITES, 2, generator=generator) * torch.tensor([width, height])
    sites += torch.tensor([x, y])
    reference = field.query(sites).color
    index, _ = make_query_backend(field, "index")
    cuda, _ = make_query_backend(field, "cuda")
    cpu = index.query(sites).color
    gpu = cuda.query(sites).color.cpu()
    index_error = float((cpu - reference).abs().max())
    cuda_error = float((gpu - cpu).abs().max())
    if index_error > 1e-5 or cuda_error > 2e-5:
        raise RuntimeError(f"decoder parity failed: index {index_error}, cuda {cuda_error}")
    return {"index_vs_reference": index_error, "cuda_vs_index": cuda_error}


def _path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_init(directory: Path, model: Gaussians3D) -> dict:
    directory.mkdir(parents=True)
    model.save_npz(directory / "gaussians_init.npz")
    return {"n_gaussians": model.n, "sha256": sha256(directory / "gaussians_init.npz")}


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


def snapshot_source(task: dict, run: Path) -> None:
    """Archive the exact bound source and the environment before the first cell."""
    directory = run / "source_snapshot"
    if directory.exists():
        return
    files = []
    for pattern in task["frozen_configuration"]["source_binding"]["patterns"]:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                relative = path.relative_to(ROOT).as_posix()
                target = directory / "files" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                files.append(
                    {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
                )
    for extra in (task["data_seal"], task["protocol_review"]["artifact"]):
        path = ROOT / extra
        files.append({"path": extra, "bytes": path.stat().st_size, "sha256": sha256(path)})
    diff = subprocess.run(["git", "diff"], cwd=ROOT, capture_output=True, check=True).stdout
    (directory / "tracked.diff").write_bytes(diff)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    (directory / "git-status.txt").write_bytes(status)
    write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "root": str(ROOT),
            "files": files,
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip(),
            "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        },
    )


def write_environment(run: Path) -> None:
    """Record the execution environment; written on every start that lacks it."""
    from importlib.metadata import PackageNotFoundError, version

    if (run / "environment.json").exists():
        return
    packages = {}
    for name in ("torch", "numpy", "gsplat", "lpips", "rtgs"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    write_json(
        run / "environment.json",
        {
            "schema_version": 1,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "device": {
                "type": "cuda",
                "name": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
            },
        },
    )


def preflight() -> dict:
    """Fail before any protected worker if CUDA, gsplat or the LPIPS evaluator is unusable."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no CPU fallback in this protocol")
    import gsplat

    model = lpips_model()
    probe = torch.rand(1, 3, 32, 32, device="cuda:0")
    with torch.no_grad():
        value = float(model(probe, probe.flip(-1), normalize=True))
    if not math.isfinite(value):
        raise RuntimeError("LPIPS preflight produced a nonfinite value")
    del model
    torch.cuda.empty_cache()
    return {"cuda": torch.cuda.get_device_name(0), "gsplat": gsplat.__version__, "lpips": value}


def check_protocol_tables(task: dict) -> None:
    """The driver's condition table must equal the frozen comparators and cell matrix."""
    if [item["id"] for item in task["comparators"]] != list(CONDITIONS):
        raise RuntimeError("driver conditions differ from frozen comparators")
    expected = sorted([c, s] for c in CONDITIONS for s in task["seeds"])
    if sorted(task["execution_order"]["cells"]) != expected:
        raise RuntimeError("execution order must contain every condition/seed exactly once")
    split = task["splits"][DATASET]
    if set(split["train"]) & set(split["heldout"]):
        raise RuntimeError("train and held-out colour views overlap")
    if sorted(task["hull"]["mask_views"]) != sorted(split["train"] + split["heldout"]):
        raise RuntimeError("the hull must use the masks of every calibrated view")


def access_guard(task: dict, phase: str, condition: str | None = None) -> dict:
    """Deny forbidden opens at the Python open boundary and keep the attempted violations.

    prepare may read training photographs/masks and every compact view of the frozen field
    family (held-out views only for their packed alpha). A fitting worker may read only its own
    target family, the soft training masks, the hull cache and metadata. Held-out photographs
    and mask PNGs open only in ``evaluate``; ``production`` reads compact views only.
    """
    frame = (ROOT / task["datasets"][0]["frame_path"]).resolve()
    targets = (ROOT / "runs" / task["task_id"] / "targets").resolve()
    heldout = set(task["splits"][DATASET]["heldout"])
    family = CONDITIONS[condition][0] if condition else None
    receipt = {"phase": phase, "condition": condition, "dataset_opens": [], "denied": []}

    def deny(path: Path, why: str) -> None:
        receipt["denied"].append(str(path))
        raise PermissionError(f"{phase}/{condition}: {why}: {path.name}")

    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if phase == "fit" and path.is_relative_to(targets):
            relative = path.relative_to(targets)
            allowed = {family, MASKS, HULL}
            if relative != Path("metadata.json") and relative.parts[0] not in allowed:
                deny(path, "forbidden target-cache open")
        if not path.is_relative_to(frame):
            return
        stem = path.stem.removeprefix("mask_")
        image = path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        if phase in {"prepare", "initialize"} and image and stem in heldout:
            deny(path, "held-out photograph or mask before evaluation")
        if phase in {"fit", "initialize"}:
            deny(path, "fitting and initialization read only cached targets")
        if phase == "production" and image:
            deny(path, "production reads compact views only")
        receipt["dataset_opens"].append(str(path.relative_to(ROOT)))

    sys.addaudithook(audit)
    return receipt


def prepare(task: dict, run: Path) -> None:
    from rtgs.data.calibrated import _object_bounds
    from rtgs.data.field_targets import decode_alpha_grid, decode_compact_view, downscale_pinhole

    started = run_seconds(run)
    access = access_guard(task, "prepare")
    factor = task["preprocessing"]["downscale"]
    split = task["splits"][DATASET]
    target_dir = run / "targets"
    target_dir.mkdir()
    frame = task["datasets"][0]["frame_path"]
    sealed = {item["path"]: item["sha256"] for item in read_json(ROOT / task["data_seal"])["files"]}
    cameras, masks, records, teacher = [], [], [], []
    for index, view_id in enumerate(split["train"]):
        print(f"prepare train {index + 1}/{len(split['train'])} {view_id}", flush=True)
        source = load_source(task, view_id)
        full_camera = source.cameras[0]
        camera = downscale_pinhole(full_camera, factor)
        sites = quadrature_sites(camera.width, camera.height, factor)
        photo = quadrature_image(source.images[0], sites, camera).clamp(0, 1)
        view = load_view(task, FIELD, view_id, load_alpha=True)
        if camera_record(view.camera) != camera_record(full_camera):
            raise RuntimeError(f"compact/calibrated camera mismatch for {view_id}")
        for kind, path in (
            ("rgb", f"{frame}/rgb/{view_id}.jpg"),
            ("mask", f"{frame}/mask/mask_{view_id}.png"),
        ):
            if view.source[kind]["sha256"] != sealed[path]:
                raise RuntimeError(f"{view_id} was not fitted to the sealed {kind}")
        packed = view.alpha.full_mask((full_camera.height, full_camera.width))
        mismatch = int((packed != (source.masks[0] > 0.5)).sum())
        if mismatch:
            raise RuntimeError(f"packed alpha differs from source mask in {view_id}: {mismatch}")
        soft = decode_alpha_grid(
            view.alpha, (full_camera.height, full_camera.width), downscale=factor
        )
        parity = parity_record(view, 926200 + index)
        decoded = decode_compact_view(view, downscale=factor, supersample=2, backend="cuda")
        clipped = int(((soft > 0) & (decoded.coverage < 1)).sum())
        if clipped:
            raise RuntimeError(f"{view_id}: fit window clips {clipped} mask pixels")
        np.savez(_path(target_dir / MASKS / f"{view_id}.npz"), mask=soft.numpy())
        np.savez(_path(target_dir / PHOTOGRAPHS / f"{view_id}.npz"), color=photo.numpy())
        np.savez(_path(target_dir / FIELD / f"{view_id}.npz"), color=decoded.color.numpy())
        error = (decoded.color - photo).square().mean(-1)[soft >= 0.5].mean().clamp_min(1e-12)
        teacher.append({"view_id": view_id, "foreground_psnr": float(-10 * torch.log10(error))})
        records.append({"view_id": view_id, **parity, "compact_sha256": view.sha256})
        cameras.append(camera)
        masks.append(soft)
    center, extent = _object_bounds(cameras, [m >= 0.5 for m in masks])
    # Held-out views contribute only their packed alpha and camera to the hull; their field
    # colours are neither decoded nor stored, and their photographs/mask PNGs stay closed.
    hull_cameras, hull_masks = [], []
    for view_id in task["hull"]["mask_views"]:
        view = load_view(task, FIELD, view_id, load_alpha=True)
        full = view.camera
        hull_cameras.append(downscale_pinhole(full, factor))
        hull_masks.append(
            decode_alpha_grid(view.alpha, (full.height, full.width), downscale=factor)
        )
        del view
    np.savez(
        _path(target_dir / HULL / "masks.npz"),
        **{view_id: m.numpy() for view_id, m in zip(task["hull"]["mask_views"], hull_masks)},
    )
    metadata = {
        "view_ids": split["train"],
        "cameras": [camera_record(camera) for camera in cameras],
        "hull_view_ids": task["hull"]["mask_views"],
        "hull_cameras": [camera_record(camera) for camera in hull_cameras],
        "bounds": {"center": center.tolist(), "extent": float(extent)},
    }
    write_json(target_dir / "metadata.json", metadata)
    write_json(
        run / "preparation.json",
        {
            "views": records,
            "teacher_foreground_psnr": teacher,
            "teacher_mean_foreground_psnr": float(np.mean([r["foreground_psnr"] for r in teacher])),
            "bounds": metadata["bounds"],
            "access_guard": access,
            "cpu_selftest": selftest(),
            "stage_intervals": {"prepare": [started, run_seconds(run)]},
        },
    )


def load_hull(run: Path, task: dict, device: str = "cpu"):
    from rtgs.optim.silhouette_relocation import SilhouetteHull

    metadata = read_json(run / "targets/metadata.json")
    with np.load(run / "targets" / HULL / "masks.npz") as data:
        masks = [torch.from_numpy(data[v].copy()) for v in metadata["hull_view_ids"]]
    cameras = [Camera(**record) for record in metadata["hull_cameras"]]
    return SilhouetteHull(
        cameras,
        masks,
        dilation_px=task["hull"]["dilation_px"],
        near=task["hull"]["near"],
        device=device,
    )


def cached_scene(run: Path, family: str, view_ids: list[str] | None = None) -> SceneData:
    metadata = read_json(run / "targets/metadata.json")
    names = metadata["view_ids"] if view_ids is None else view_ids
    records = dict(zip(metadata["view_ids"], metadata["cameras"]))
    images, masks = [], []
    for view in names:
        with np.load(run / "targets" / family / f"{view}.npz") as data:
            images.append(torch.from_numpy(data["color"].copy()))
        with np.load(run / "targets" / MASKS / f"{view}.npz") as data:
            masks.append(torch.from_numpy(data["mask"].copy()))
    scene = SceneData(
        images,
        [Camera(**records[view]) for view in names],
        view_names=list(names),
        masks=masks,
        train_indices=list(range(len(images))),
        test_indices=[],
        bounds_hint=(torch.tensor(metadata["bounds"]["center"]), metadata["bounds"]["extent"]),
        name=DATASET,
    )
    scene.validate()
    return scene


def initialize(task: dict, run: Path) -> None:
    started = run_seconds(run)
    access = access_guard(task, "initialize")
    spec = task["initialization"]
    metadata = read_json(run / "targets/metadata.json")
    center = torch.tensor(metadata["bounds"]["center"], dtype=torch.float32)
    extent = float(metadata["bounds"]["extent"])
    hull = load_hull(run, task)
    voxels, voxel = hull.occupied_voxels(center, extent, task["hull"]["grid"])
    grid = task["hull"]["grid"]
    index = torch.round((voxels - center + extent / 2) / voxel - 0.5).long()
    if bool(((index == 0) | (index == grid - 1)).any()):
        raise RuntimeError("the visual hull touches a bound face; frozen protocol aborted")
    if len(voxels) < task["hull"]["minimum_voxels"]:
        raise RuntimeError(f"the visual hull has only {len(voxels)} occupied voxels")
    torch.save({"voxels": voxels, "voxel": voxel}, _path(run / "targets" / HULL / "voxels.pt"))
    records = {}
    for seed in task["seeds"]:
        rand = random_initialization(center, extent, spec["n_points"], seed)
        model = model_from(rand["means"], rand["scale"], torch.full((spec["n_points"], 3), 0.5))
        records[f"random/{seed}"] = _save_init(run / "initialization" / "random" / str(seed), model)
        records[f"random/{seed}"]["scale"] = rand["scale"]
        records[f"random/{seed}"]["hull_rejected_fraction"] = float(
            hull.rejected(rand["means"]).float().mean()
        )
    write_json(
        run / "initialization.json",
        {
            "models": records,
            "hull_occupied_voxels": int(len(voxels)),
            "hull_voxel_size": voxel,
            "bounds": metadata["bounds"],
            "access_guard": access,
            "stage_intervals": {"initialize": [started, run_seconds(run)]},
        },
    )


def train_config(task: dict, seed: int):
    from rtgs.optim.density import DensityConfig
    from rtgs.optim.trainer import TrainConfig

    frozen = dict(task["resolved_training_configs"][str(seed)])
    frozen["density"] = DensityConfig(**frozen["density"])
    config = TrainConfig(**frozen)
    if json.loads(json.dumps(asdict(config))) != task["resolved_training_configs"][str(seed)]:
        raise RuntimeError("resolved training configuration does not round-trip")
    return config


def relocator_for(task: dict, run: Path, seed: int, device: str):
    from rtgs.optim.silhouette_relocation import SilhouetteRelocationConfig, SilhouetteRelocator

    spec = task["relocation"]
    config = SilhouetteRelocationConfig(
        every=spec["every"],
        start=spec["start"],
        stop=spec["stop"],
        dilation_px=task["hull"]["dilation_px"],
        near=task["hull"]["near"],
        jitter_fraction=spec["jitter_fraction"],
        seed=seed,
    )
    cached = torch.load(run / "targets" / HULL / "voxels.pt", weights_only=True)
    return SilhouetteRelocator(
        load_hull(run, task, device), cached["voxels"], cached["voxel"], config
    )


def _train(task, run, output, scene, initial, seed, relocate, receipt):
    from rtgs.optim.trainer import Trainer

    config = train_config(task, seed)
    relocator = relocator_for(task, run, seed, "cuda:0") if relocate else None
    checkpoints = []

    def checkpoint(model, step):
        for tensor in (model.means, model.quats, model.log_scales, model.opacity, model.sh):
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"nonfinite model at step {step}")
        checkpoints.append({"step": step, "run_seconds": run_seconds(run), "n": model.n})
        print(f"{receipt.get('condition')}/{seed} step={step} n={model.n}", flush=True)

    write_json(
        output / "effective_config.json",
        {
            "condition": receipt.get("condition"),
            "relocation": relocate,
            "relocation_config": task["relocation"] if relocate else None,
            "hull": task["hull"],
            "train_config": asdict(config),
            "scene_bounds": read_json(run / "targets/metadata.json")["bounds"],
            "train_view_ids": scene.view_names,
        },
    )
    final, history = Trainer(config).train(
        scene, initial, checkpoint_callback=checkpoint, parameter_step_callback=relocator
    )
    torch.cuda.synchronize()
    if history["executed_iterations"] != config.iterations:
        raise RuntimeError("trainer did not reach the frozen final iteration")
    if not all(math.isfinite(value) for value in history["loss"]):
        raise RuntimeError("nonfinite fitting loss")
    final.save_npz(output / "gaussians.npz")
    final.save_ply(output / "gaussians.ply")
    history["checkpoint_observer"] = checkpoints
    history["relocation_events"] = [] if relocator is None else relocator.events
    write_json(output / "history.json", history)
    return final


def fit(task: dict, run: Path, condition: str, seed: int) -> None:
    family, relocate = CONDITIONS[condition]
    access = access_guard(task, "fit", condition)
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
    output = run / "cells" / condition / str(seed)
    output.mkdir(parents=True)
    receipt = {"condition": condition, "seed": seed, "status": "running", "access_guard": access}
    try:
        scene = cached_scene(run, family)
        key = f"random/{seed}"
        initial_path = run / "initialization" / key / "gaussians_init.npz"
        receipt["initial_sha256"] = sha256(initial_path)
        if (
            receipt["initial_sha256"]
            != read_json(run / "initialization.json")["models"][key]["sha256"]
        ):
            raise RuntimeError("shared initialization digest mismatch")
        initial = Gaussians3D.load_npz(initial_path)
        shutil.copyfile(initial_path, output / "gaussians_init.npz")
        initial.save_ply(output / "gaussians_init.ply")
        final = _train(task, run, output, scene, initial, seed, relocate, receipt)
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
    from rtgs.data.field_targets import decode_compact_view, downscale_pinhole
    from rtgs.render.base import get_rasterizer
    from rtgs.visualize import save_reconstruction_artifacts

    for condition, seed in task["execution_order"]["cells"]:
        if (
            read_json(run / "cells" / condition / str(seed) / "receipt.json")["status"]
            != "completed"
        ):
            raise RuntimeError("evaluation requires every final model to be saved")
    access = access_guard(task, "evaluate")
    factor = task["preprocessing"]["downscale"]
    heldout = task["splits"][DATASET]["heldout"]
    references, masks, cameras, fields = [], [], [], []
    for view_id in heldout:
        source = load_source(task, view_id)
        camera = downscale_pinhole(source.cameras[0], factor)
        sites = quadrature_sites(camera.width, camera.height, factor)
        raw_mask = source.masks[0] > 0.5
        view = load_view(task, FIELD, view_id, load_alpha=True)
        packed = view.alpha.full_mask((source.cameras[0].height, source.cameras[0].width))
        if int((packed != raw_mask).sum()):
            raise RuntimeError(f"held-out packed alpha differs from its source mask: {view_id}")
        references.append(quadrature_image(source.images[0], sites, camera).clamp(0, 1))
        masks.append(quadrature_image(raw_mask.float(), sites, camera)[..., 0] >= 0.5)
        cameras.append(camera)
        fields.append(decode_compact_view(view, downscale=factor, backend="cuda").color)
    hull = load_hull(run, task, "cuda:0")
    preview_scene = SceneData(
        [reference * mask[..., None] for reference, mask in zip(references, masks)],
        cameras,
        view_names=heldout,
        train_indices=[],
        test_indices=list(range(len(heldout))),
        masks=None,
        bounds_hint=None,
    )
    perceptual = lpips_model()
    renderer = get_rasterizer("gsplat", device="cuda:0")
    for condition, seed in task["execution_order"]["cells"]:
        started = run_seconds(run)
        output = run / "cells" / condition / str(seed)
        final = Gaussians3D.load_npz(output / "gaussians.npz").to("cuda:0")
        initial = Gaussians3D.load_npz(output / "gaussians_init.npz").to("cuda:0")
        rejected = float(hull.rejected(final.means).float().mean())
        rows = []
        with torch.no_grad():
            for index, (view_id, camera) in enumerate(zip(heldout, cameras)):
                rendered = renderer.render(final, camera.to("cuda:0"))
                color = rendered.color.cpu()
                alpha = rendered.alpha.cpu().reshape(masks[index].shape)
                row = {
                    "view_id": view_id,
                    **mask_scores(color, alpha, references[index], masks[index], perceptual),
                    "hull_rejected_fraction": rejected,
                }
                errors = (color.clamp(0, 1) - fields[index]).square().mean(-1)[masks[index]]
                row["diagnostic_field_consistency_psnr"] = float(
                    -10 * torch.log10(errors.mean().clamp_min(1e-12))
                )
                rows.append(row)
        artifacts = save_reconstruction_artifacts(
            preview_scene,
            initial,
            final,
            output,
            rasterizer="gsplat",
            max_comparisons=len(heldout),
            max_animation_frames=24,
        )
        write_json(
            output / "evaluation.json",
            {
                "per_view": rows,
                "mean": {key: float(np.mean([row[key] for row in rows])) for key in METRICS},
                "artifacts": artifacts,
                "access_guard": access,
                "stage_intervals": {"evaluate": [started, run_seconds(run)]},
            },
        )
        print(f"evaluated {condition}/{seed}", flush=True)


def production(task: dict, run: Path) -> None:
    """After a passing frozen gate only: train the relocation arm on every view (not evidence)."""
    from rtgs.data.field_targets import decode_alpha_grid, decode_compact_view, downscale_pinhole

    comparison = read_json(run / "comparison.json")
    if comparison["gates"]["h1_relocation"]["verdict"] != "pass":
        raise RuntimeError("production training requires a passing frozen relocation gate")
    access = access_guard(task, "production")
    spec = task["production"]
    factor = task["preprocessing"]["downscale"]
    output = run / "production"
    output.mkdir()
    images, masks, cameras = [], [], []
    for view_id in task["hull"]["mask_views"]:
        view = load_view(task, FIELD, view_id, load_alpha=True)
        full = view.camera
        cameras.append(downscale_pinhole(full, factor))
        images.append(decode_compact_view(view, downscale=factor, backend="cuda").color)
        masks.append(decode_alpha_grid(view.alpha, (full.height, full.width), downscale=factor))
    metadata = read_json(run / "targets/metadata.json")
    scene = SceneData(
        images,
        cameras,
        view_names=list(task["hull"]["mask_views"]),
        masks=masks,
        train_indices=list(range(len(images))),
        test_indices=[],
        bounds_hint=(torch.tensor(metadata["bounds"]["center"]), metadata["bounds"]["extent"]),
        name=DATASET,
    )
    scene.validate()
    seed = spec["seed"]
    warmup()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial = Gaussians3D.load_npz(
        run / "initialization" / "random" / str(seed) / "gaussians_init.npz"
    )
    initial.save_ply(output / "gaussians_init.ply")
    receipt = {
        "condition": "production_nb_ms_reloc_all_views",
        "seed": seed,
        "access_guard": access,
    }
    final = _train(task, run, output, scene, initial, seed, True, receipt)
    write_json(
        output / "receipt.json",
        {
            **receipt,
            "status": "completed",
            "final_count": final.n,
            "final_sha256": sha256(output / "gaussians.npz"),
            "boundary": "All-view production model; no held-out view remains; not evidence.",
        },
    )


def selftest() -> dict:
    """CPU mechanics: quadrature, bands, scores, initializer determinism and the input guard."""
    sites = quadrature_sites(2, 1, 8)
    assert torch.equal(sites[0], torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]]))
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[4:12, 4:12] = True
    assert int(dilate(mask, 1).sum()) == 100 and int(erode(mask, 1).sum()) == 36
    a = random_initialization(torch.zeros(3), 2.0, 256, 7)
    b = random_initialization(torch.zeros(3), 2.0, 256, 7)
    assert torch.equal(a["means"], b["means"])
    fake = {
        "task_id": "__relocation_guard_selftest__",
        "datasets": [{"frame_path": ".scratch/__relocation_guard_selftest__"}],
        "splits": {DATASET: {"heldout": ["C0001"]}},
    }
    guarded = access_guard(fake, "fit", "nb_ms_reloc")
    base = ROOT / fake["datasets"][0]["frame_path"]
    targets = ROOT / "runs" / fake["task_id"] / "targets"
    probes = [
        base / "rgb/C0002.jpg",
        base / "mask/mask_C0002.png",
        base / "gaussians2d_x/C0002.rtgsv",
        targets / PHOTOGRAPHS / "C0002.npz",
    ]
    for path in probes:
        try:
            path.open("rb")
        except PermissionError:
            continue
        raise AssertionError(f"input boundary failed for {path}")
    assert len(guarded["denied"]) == len(probes)
    return {
        "quadrature": "passed",
        "mask_bands": "passed",
        "random_initialization_determinism": "passed",
        "field_worker_input_guard": f"{len(probes)} forbidden open probes denied",
    }


def report_module():
    path = Path(__file__).with_name(Path(__file__).stem + "_report.py")
    spec = importlib.util.spec_from_file_location("relocation_report", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def coordinate(task_path: Path, task: dict, run: Path) -> None:
    split_digest = hashlib.sha256(json.dumps(task["splits"], sort_keys=True).encode()).hexdigest()
    logs = run / "logs"
    logs.mkdir(exist_ok=True)

    def worker(phase, condition=None, seed=None):
        command = [sys.executable, str(Path(__file__).resolve()), phase, "--task", str(task_path)]
        command += ["--run-dir", str(run)]
        name = phase
        if condition is not None:
            command += ["--condition", condition, "--seed", str(seed)]
            name += f"_{condition}_{seed}"
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
                timeout=3600 if phase in {"fit", "production"} else None,
                check=True,
            )

    try:
        snapshot_source(task, run)
        write_environment(run)
        source_guard(task_path, task, run)
        write_json(run / "input_integrity_entry.json", data_guard(task))
        write_json(run / "preflight.json", preflight())
        worker("prepare")
        worker("initialize")
        for condition, seed in task["execution_order"]["cells"]:
            print(f"starting {condition}/{seed}", flush=True)
            try:
                worker("fit", condition, seed)
            except subprocess.TimeoutExpired:
                cell = run / "cells" / condition / str(seed)
                cell.mkdir(parents=True, exist_ok=True)
                write_json(
                    cell / "receipt.json",
                    {
                        "condition": condition,
                        "seed": seed,
                        "status": "timed_out",
                        "timeout_s": 3600,
                    },
                )
                raise
        worker("evaluate")
        source_guard(task_path, task, run)
        write_json(
            run / "input_integrity_exit.json", {**data_guard(task), "split_sha256": split_digest}
        )
        report_module().publish(task, run)
    except BaseException as error:
        write_json(
            run / "execution_failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "split_sha256": split_digest,
            },
        )
        report_module().publish_failure(task, run, str(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("run", "prepare", "initialize", "fit", "evaluate", "production", "selftest"),
    )
    parser.add_argument("--task", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
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
    check_protocol_tables(task)
    if args.command == "run":
        coordinate(task_path, task, run)
        return
    source_guard(task_path, task, run)
    if args.command == "prepare":
        prepare(task, run)
    elif args.command == "initialize":
        initialize(task, run)
    elif args.command == "fit":
        if [args.condition, args.seed] not in task["execution_order"]["cells"]:
            parser.error("fit cell is not registered")
        fit(task, run, args.condition, args.seed)
    elif args.command == "evaluate":
        evaluate(task, run)
    elif args.command == "production":
        production(task, run)


if __name__ == "__main__":
    main()
