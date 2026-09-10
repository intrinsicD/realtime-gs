"""Frozen cached-target residuals and gradients at unchanged saved 3DGS states."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import ssim

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("means", "quats", "scales", "opacities", "sh0", "shN")
COMPONENTS = ("l1", "dssim", "total")
TARGETS = ("rgb", "field_high")


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


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_tensors(model: Gaussians3D) -> dict[str, torch.Tensor]:
    return {
        "means": model.means,
        "quats": model.quats,
        "scales": model.log_scales,
        "opacities": model.opacity,
        "sh": model.sh,
    }


def require_finite(*values: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError("nonfinite diagnostic input, render, loss or gradient")


class RunClock:
    def __init__(self, run: Path):
        start = dt.datetime.fromisoformat(read_json(run / "task.lock.json")["started_at_utc"])
        self.offset = time.time() - start.timestamp()
        self.start = time.perf_counter()

    def __call__(self) -> float:
        return self.offset + time.perf_counter() - self.start


class Deadlines:
    """SIGALRM enforces nested absolute state and task wall deadlines on Linux."""

    def __init__(self, seconds: float):
        self.total_end = time.monotonic() + seconds
        self.state_end = None

    def _arm(self):
        end = self.total_end if self.state_end is None else min(self.total_end, self.state_end)
        signal.setitimer(signal.ITIMER_REAL, max(end - time.monotonic(), 1e-6))

    def _expired(self, _number, _frame):
        raise TimeoutError("frozen state or total diagnostic wall deadline exceeded")

    def __enter__(self):
        self.previous_handler = signal.signal(signal.SIGALRM, self._expired)
        self.previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
        self.entered = time.monotonic()
        self._arm()
        return self

    def __exit__(self, *_error):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)
        if self.previous_timer[0] > 0:
            remaining = max(self.previous_timer[0] - (time.monotonic() - self.entered), 1e-6)
            signal.setitimer(signal.ITIMER_REAL, remaining, self.previous_timer[1])

    @contextlib.contextmanager
    def state(self, seconds: float):
        self.state_end = time.monotonic() + seconds
        self._arm()
        try:
            yield
        finally:
            self.state_end = None
            self._arm()


class InputGuard:
    """A narrow cached-read boundary; never permits writes to previous evidence."""

    def __init__(self, task: dict, root: Path = ROOT):
        self.root = root.resolve()
        self.source_run = (self.root / task["source_run"]).resolve()
        self.dataset = self.root / "dataset"
        self.allowed = {(self.root / item["path"]).resolve() for item in task["cached_inputs"]}
        self.phase = "entry"
        self.receipt = {"allowed_file_count": len(self.allowed), "reads": [], "denied": []}

    def audit(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        lexical = Path(os.path.abspath(os.fsdecode(args[0])))
        path = lexical.resolve()
        in_source = path.is_relative_to(self.source_run) or lexical.is_relative_to(self.source_run)
        raw = path.is_relative_to(self.dataset) or lexical.is_relative_to(self.dataset)
        if not in_source and not raw:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = isinstance(mode, str) and any(mark in mode for mark in ("w", "a", "+", "x"))
        writing |= bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        record = {"path": str(path.relative_to(self.root)), "phase": self.phase}
        if raw or path not in self.allowed or writing:
            self.receipt["denied"].append(record)
            raise PermissionError(f"diagnostic input policy denies {record['path']}")
        self.receipt["reads"].append(record)

    def install(self):
        sys.addaudithook(self.audit)


def source_guard(task_path: Path, task: dict, run: Path) -> dict:
    lock = read_json(run / "task.lock.json")
    if task["status"] != "ready" or task["protocol_review"]["verdict"] != "approved":
        raise RuntimeError("protected diagnostic requires final approved ready protocol")
    if lock["task_id"] != task["task_id"] or lock["task_sha256"] != sha256(task_path):
        raise RuntimeError("task does not match immutable run lock")
    manifest = read_json(run / "source_snapshot/manifest.json")
    files = manifest["files"]
    if Path(__file__).resolve().relative_to(ROOT).as_posix() not in {row["path"] for row in files}:
        raise RuntimeError("executed driver missing from source snapshot")
    for row in files:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"source changed: {row['path']}")
    return {"task_sha256": sha256(task_path), "source_files_checked": len(files), "all_match": True}


def cache_guard(task: dict) -> dict:
    records = task["cached_inputs"]
    expected = {state["model"] for state in task["states"]}
    source = Path(task["source_run"])
    expected.add((source / "targets/metadata.json").as_posix())
    for view in task["splits"]["frame_00008"]["train"]:
        for family in (*TARGETS, "initialization_masks"):
            expected.add((source / "targets" / family / f"{view}.npz").as_posix())
    if len(records) != 76 or {row["path"] for row in records} != expected:
        raise RuntimeError("cached allowlist is not the exact76 frozen inputs")
    for row in records:
        path = ROOT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"cached input changed: {row['path']}")
    return {
        "files_checked": len(records),
        "bytes_checked": sum(row["bytes"] for row in records),
        "all_match": True,
    }


def load_cached(task: dict) -> dict:
    source = ROOT / task["source_run"] / "targets"
    metadata = read_json(source / "metadata.json")
    train = task["splits"]["frame_00008"]["train"]
    if metadata["view_ids"] != train or task["gradient_protocol"]["view_order"] != train:
        raise RuntimeError("camera cache or measurement order differs from frozen training split")
    if set(train) & set(task["splits"]["frame_00008"]["heldout"]):
        raise RuntimeError("training and heldout views overlap")
    cameras = [Camera(**row) for row in metadata["cameras"]]
    if len(cameras) != len(train):
        raise RuntimeError("cached camera count mismatch")
    views = []
    input_hashes = {item["path"]: item["sha256"] for item in task["cached_inputs"]}
    for view, camera in zip(train, cameras):
        row = {"view_id": view, "camera": camera, "input_sha256": {}}
        for family in (*TARGETS, "initialization_masks"):
            path = source / family / f"{view}.npz"
            row["input_sha256"][family] = input_hashes[path.relative_to(ROOT).as_posix()]
            with np.load(path, allow_pickle=False) as archive:
                key = "mask" if family == "initialization_masks" else "color"
                array = archive[key].copy()
            tensor = torch.from_numpy(array)
            if family == "initialization_masks":
                if tensor.dtype != torch.bool or tensor.shape != (camera.height, camera.width):
                    raise RuntimeError("mask cache must be the frozen binary camera-sized bitmap")
            elif tensor.dtype != torch.float32 or tensor.shape != (camera.height, camera.width, 3):
                raise RuntimeError("target cache must be the frozen float32 camera-sized RGB")
            require_finite(tensor)
            if not bool(((tensor >= 0) & (tensor <= 1)).all()):
                raise RuntimeError("cached target values outside frozen range")
            row[family] = tensor
        views.append(row)
    return {"metadata": metadata, "views": views}


def regions(mask: torch.Tensor, radius: int = 3) -> dict[str, torch.Tensor]:
    source = mask.double()[None, None]
    kernel = radius * 2 + 1
    dilation = F.max_pool2d(source, kernel, 1, radius)[0, 0] > 0.5
    outside = F.pad(1 - source, (radius, radius, radius, radius), value=1)
    erosion = (1 - F.max_pool2d(outside, kernel, 1)[0, 0]) > 0.5
    boundary = dilation ^ erosion
    return {"interior": mask & ~boundary, "boundary": boundary, "exterior": ~mask & ~boundary}


def box_mean(image: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Separable complete-image7x7 mean, with reflect padding on both axes."""
    source = image.permute(2, 0, 1)[None]
    horizontal = F.avg_pool2d(
        F.pad(source, (radius, radius, 0, 0), mode="reflect"), (1, radius * 2 + 1), stride=1
    )
    result = F.avg_pool2d(
        F.pad(horizontal, (0, 0, radius, radius), mode="reflect"), (radius * 2 + 1, 1), stride=1
    )
    return result[0].permute(1, 2, 0)


def residual_statistics(photo: torch.Tensor, field: torch.Tensor, mask: torch.Tensor) -> dict:
    photo, field = photo.double(), field.double()
    residual = field - photo
    lowpass = box_mean(residual)
    highpass = residual - lowpass
    contrasts = {}
    for name, image in (("photo", photo), ("field", field)):
        contrasts[name] = (box_mean(image.square()) - box_mean(image).square()).clamp_min(0).sqrt()
    total_absolute = float(residual.abs().sum())
    region_stats = {}
    for name, selection in regions(mask).items():
        count = int(selection.sum())
        values = residual[selection]
        absolute = float(values.abs().sum())
        signed = values.sum(0)
        squared = float(values.square().sum())
        region_stats[name] = {
            "pixel_count": count,
            "channel_value_count": count * 3,
            "signed_sum_per_channel": signed.tolist(),
            "squared_error_sum": squared,
            "signed_mean_per_channel": (signed / count).tolist() if count else None,
            "signed_mean_rgb": float(signed.sum() / (3 * count)) if count else None,
            "mae_region": absolute / (3 * count) if count else None,
            "mse_region": squared / (3 * count) if count else None,
            "absolute_error_sum": absolute,
            "absolute_error_share_full_canvas": absolute / total_absolute
            if count and total_absolute > 0
            else None,
            "full_canvas_l1_contribution": absolute / residual.numel() if count else None,
            "photo_local_contrast": float(contrasts["photo"][selection].mean()) if count else None,
            "field_local_contrast": float(contrasts["field"][selection].mean()) if count else None,
            "residual_lowpass_mse": float(lowpass[selection].square().mean()) if count else None,
            "residual_highpass_mse": float(highpass[selection].square().mean()) if count else None,
        }
    return {
        "regions": region_stats,
        "full_canvas_pixel_count": mask.numel(),
        "full_canvas_absolute_error_sum": total_absolute,
        "full_canvas_mae": total_absolute / residual.numel(),
        "full_canvas_mse": float(residual.square().mean()),
    }


def compare_gradients(photo: torch.Tensor, field: torch.Tensor, near_zero: float = 1e-12) -> dict:
    left, right = (
        photo.detach().cpu().double().reshape(-1),
        field.detach().cpu().double().reshape(-1),
    )
    if left.shape != right.shape:
        raise ValueError("gradient comparisons require identical within-state row identities")
    require_finite(left, right)
    difference = right - left
    a, b = float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(right))
    delta = float(torch.linalg.vector_norm(difference))
    cosine = float(torch.dot(left, right) / (a * b)) if a > near_zero and b > near_zero else None
    return {
        "photo_norm": a,
        "field_norm": b,
        "difference_norm": delta,
        "difference_over_photo_norm": delta / a if a > near_zero else None,
        "cosine": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
        "max_absolute_difference": float(difference.abs().max()) if difference.numel() else 0.0,
        "element_count": left.numel(),
    }


def fresh_parameters(saved: Gaussians3D, kind: str, device: str) -> dict[str, torch.Tensor]:
    working = saved.with_sh_degree(3).to(device)
    alpha = working.opacity
    if kind == "initial":
        alpha = torch.sigmoid(torch.logit(alpha.clamp(1e-4, 1 - 1e-4)))
    parameters = {
        "means": working.means,
        "quats": working.quats,
        "scales": working.log_scales,
        "opacities": alpha,
        "sh0": working.sh[:, :1],
        "shN": working.sh[:, 1:],
    }
    return {name: value.detach().clone().requires_grad_(True) for name, value in parameters.items()}


def assemble(parameters: dict[str, torch.Tensor]) -> Gaussians3D:
    return Gaussians3D(
        means=parameters["means"],
        quats=parameters["quats"],
        log_scales=parameters["scales"],
        opacity=parameters["opacities"],
        sh=torch.cat((parameters["sh0"], parameters["shN"]), dim=1),
    )


def mapped_gradients(raw, parameters) -> dict[str, torch.Tensor]:
    result = {}
    for name, gradient in zip(GROUPS, raw):
        gradient = torch.zeros_like(parameters[name]) if gradient is None else gradient
        if name == "opacities":
            alpha = parameters[name].detach()
            gradient = gradient * alpha * (1 - alpha)
        require_finite(gradient)
        result[name] = gradient.detach()
    return result


def target_route(view: dict, target: str, *, identity: bool = False) -> torch.Tensor:
    if target == "rgb":
        return view["rgb"]
    if target == "field_high":
        return view["rgb"].clone() if identity else view["field_high"]
    raise ValueError("unknown target route")


def evaluate_target(saved, state, view, target, renderer, device, *, identity=False, direct=False):
    """Fresh leaves and render graph per call; masks never enter this objective."""
    parameters = fresh_parameters(saved, state["kind"], device)
    ordered = tuple(parameters[name] for name in GROUPS)
    output = renderer.render(
        assemble(parameters), view["camera"].to(device), sh_degree=state["sh_degree"]
    ).color
    reference = target_route(view, target, identity=identity).to(device)
    require_finite(output, reference)
    l1 = (output - reference).abs().mean()
    dssim = 1 - ssim(output, reference)
    total = 0.8 * l1 + 0.2 * dssim
    require_finite(l1, dssim, total)
    if direct:
        grads = mapped_gradients(torch.autograd.grad(total, ordered, allow_unused=True), parameters)
        result = {"total": {name: value.cpu() for name, value in grads.items()}}
    else:
        left = mapped_gradients(
            torch.autograd.grad(l1, ordered, retain_graph=True, allow_unused=True), parameters
        )
        right = mapped_gradients(torch.autograd.grad(dssim, ordered, allow_unused=True), parameters)
        full = {name: 0.8 * left[name] + 0.2 * right[name] for name in GROUPS}
        result = {
            component: {name: value.cpu() for name, value in gradients.items()}
            for component, gradients in zip(COMPONENTS, (left, right, full))
        }
    return {
        "losses": {
            "l1": float(l1.detach()),
            "dssim": float(dssim.detach()),
            "total": float(total.detach()),
        },
        "gradients": result,
    }


def identity_comparison(left: dict, right: dict, atol=1e-8, rtol=1e-4, loss_atol=1e-7) -> dict:
    comparisons = {}
    passed = True
    for component in left["gradients"]:
        comparisons[component] = {}
        for group in GROUPS:
            a = left["gradients"][component][group].double()
            b = right["gradients"][component][group].double()
            difference = (a - b).abs()
            tolerance = atol + rtol * torch.maximum(a.abs(), b.abs())
            violations = int((difference > tolerance).sum())
            passed &= violations == 0
            comparisons[component][group] = {
                **compare_gradients(a, b),
                "tolerance_violations": violations,
                "max_tolerance_excess": float((difference - tolerance).clamp_min(0).max()),
            }
    losses = {}
    for component in COMPONENTS:
        difference = abs(left["losses"][component] - right["losses"][component])
        losses[component] = {
            "photo_path": left["losses"][component],
            "teacher_path": right["losses"][component],
            "absolute_difference": difference,
        }
        passed &= difference <= loss_atol
    return {
        "passed": bool(passed),
        "losses": losses,
        "comparisons": comparisons,
        "gradient_atol": atol,
        "gradient_rtol": rtol,
        "loss_atol": loss_atol,
        "independent_render_graphs": True,
    }


def gradient_comparisons(left, right, near_zero):
    return {
        component: {
            group: compare_gradients(left[component][group], right[component][group], near_zero)
            for group in GROUPS
        }
        for component in COMPONENTS
    }


def state_diagnostics(task, run, state, cached, renderer, clock, deadlines):
    output = run / "states" / state["id"]
    output.mkdir(parents=True)
    started = clock()
    diagnostic = {
        "state": state,
        "status": "running",
        "rows": [],
        "stage_interval": [started, started],
    }
    try:
        with deadlines.state(task["execution_budget"]["max_state_seconds"]):
            saved = Gaussians3D.load_npz(ROOT / state["model"])
            require_finite(*model_tensors(saved).values())
            if saved.sh_degree != state["sh_degree"]:
                raise RuntimeError("saved SH degree differs from frozen state declaration")
            original = tensor_digest(model_tensors(saved))
            effective = fresh_parameters(saved, state["kind"], "cuda:0")
            alpha = effective["opacities"].detach()
            diagnostic.update(
                {
                    "model_sha256": sha256(ROOT / state["model"]),
                    "saved_tensor_sha256": original,
                    "n_gaussians": saved.n,
                    "parameter_shapes": {
                        name: list(value.shape) for name, value in effective.items()
                    },
                    "effective_parameter_sha256": tensor_digest(effective),
                    "opacity": {
                        "saved_zero_count": int((saved.opacity == 0).sum()),
                        "saved_one_count": int((saved.opacity == 1).sum()),
                        "effective_zero_count": int((alpha == 0).sum()),
                        "effective_one_count": int((alpha == 1).sum()),
                        "local_logit_chain": "g_effective_alpha*alpha*(1-alpha)",
                        "initial_entry_roundtrip": state["kind"] == "initial",
                        "max_saved_to_effective_absolute_difference": float(
                            (alpha.cpu() - saved.opacity).abs().max()
                        ),
                    },
                    "coordinates": task["gradient_protocol"]["coordinates"],
                }
            )
            del effective, alpha
            first = cached["views"][0]
            control_left = evaluate_target(saved, state, first, "rgb", renderer, "cuda:0")
            control_right = evaluate_target(
                saved, state, first, "field_high", renderer, "cuda:0", identity=True
            )
            control = identity_comparison(control_left, control_right)
            diagnostic["identity_control"] = {
                "state_id": state["id"],
                "view_id": first["view_id"],
                **control,
            }
            del control_left, control_right
            write_json(output / "diagnostics.json", diagnostic)
            if not control["passed"]:
                raise RuntimeError(f"equal-target control failed for {state['id']}")
            sums = {target: {component: {} for component in COMPONENTS} for target in TARGETS}
            for ordinal, view in enumerate(cached["views"], 1):
                view_started = clock()
                measured = {
                    target: evaluate_target(saved, state, view, target, renderer, "cuda:0")
                    for target in TARGETS
                }
                for target in TARGETS:
                    for component in COMPONENTS:
                        for group, gradient in measured[target]["gradients"][component].items():
                            if group not in sums[target][component]:
                                sums[target][component][group] = torch.zeros_like(
                                    gradient, dtype=torch.float64
                                )
                            sums[target][component][group].add_(gradient.double())
                comparisons = gradient_comparisons(
                    measured["rgb"]["gradients"],
                    measured["field_high"]["gradients"],
                    task["gradient_protocol"]["near_zero_norm"],
                )
                torch.cuda.synchronize()
                diagnostic["rows"].append(
                    {
                        "view_id": view["view_id"],
                        "input_sha256": view["input_sha256"],
                        "step": ordinal,
                        "wall_seconds": clock(),
                        "view_wall_seconds": clock() - view_started,
                        "losses": {target: measured[target]["losses"] for target in TARGETS},
                        "comparisons": comparisons,
                    }
                )
                diagnostic["stage_interval"][1] = clock()
                write_json(output / "diagnostics.json", diagnostic)
                print(
                    f"{state['id']} view{ordinal}/{len(cached['views'])} {view['view_id']}",
                    flush=True,
                )
                del measured
            arrays = {}
            for target in TARGETS:
                for component in COMPONENTS:
                    for group in GROUPS:
                        sums[target][component][group].div_(len(cached["views"]))
                        arrays[f"{target}__{component}__{group}"] = sums[target][component][
                            group
                        ].numpy()
            np.savez_compressed(output / "mean_gradients.npz", **arrays)
            diagnostic["aggregate"] = gradient_comparisons(
                sums["rgb"], sums["field_high"], task["gradient_protocol"]["near_zero_norm"]
            )
            diagnostic["aggregate_definition"] = (
                "Statistics of arithmetic mean gradient over22views"
            )
            summary = {}
            for component in COMPONENTS:
                summary[component] = {}
                for group in GROUPS:
                    entries = [row["comparisons"][component][group] for row in diagnostic["rows"]]
                    summary[component][group] = {}
                    for key in entries[0]:
                        values = [entry[key] for entry in entries if entry[key] is not None]
                        summary[component][group][key] = {
                            "mean": float(np.mean(values)) if values else None,
                            "defined_count": len(values),
                            "undefined_count": len(entries) - len(values),
                        }
            diagnostic["per_view_statistics_summary"] = summary
            diagnostic["mean_gradients_sha256"] = sha256(output / "mean_gradients.npz")
            diagnostic["saved_state_unchanged"] = original == tensor_digest(model_tensors(saved))
            if not diagnostic["saved_state_unchanged"]:
                raise RuntimeError("saved state tensor changed during no-update diagnostic")
            diagnostic["status"] = "completed"
    except BaseException as error:
        diagnostic.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        diagnostic["stage_interval"][1] = clock()
        write_json(output / "diagnostics.json", diagnostic)


def presentation(task, run, cached):
    from rtgs.data.scene import SceneData
    from rtgs.visualize import save_reconstruction_artifacts

    states = {state["id"]: state for state in task["states"]}
    selected = states[task["presentation_protocol"]["selected_state"]]
    initial = states[task["presentation_protocol"]["initial_state"]]
    final_model = Gaussians3D.load_npz(ROOT / selected["model"])
    initial_model = Gaussians3D.load_npz(ROOT / initial["model"])
    shutil.copyfile(ROOT / selected["model"], run / "gaussians.npz")
    shutil.copyfile(ROOT / initial["model"], run / "gaussians_init.npz")
    final_model.save_ply(run / "gaussians.ply")
    initial_model.save_ply(run / "gaussians_init.ply")
    metadata = cached["metadata"]
    views = cached["views"]
    scene = SceneData(
        images=[view["rgb"] for view in views],
        cameras=[view["camera"] for view in views],
        view_names=[view["view_id"] for view in views],
        train_indices=list(range(len(views))),
        test_indices=[],
        masks=None,
        bounds_hint=(torch.tensor(metadata["bounds"]["center"]), metadata["bounds"]["extent"]),
    )
    artifacts = save_reconstruction_artifacts(
        scene,
        initial_model.to("cuda:0"),
        final_model.to("cuda:0"),
        run,
        rasterizer="gsplat",
        max_comparisons=len(views),
        max_animation_frames=24,
    )
    write_json(
        run / "snapshot_provenance.json",
        {
            "new_fitting": False,
            "selected_state": selected,
            "initial_state": initial,
            "final_npz_exact_copy_sha256": sha256(run / "gaussians.npz"),
            "initial_npz_exact_copy_sha256": sha256(run / "gaussians_init.npz"),
            "preview_views": metadata["view_ids"],
            "artifacts": artifacts,
        },
    )


def synthetic_inputs(device="cpu"):
    torch.manual_seed(6049)
    camera = Camera(32, 32, 16, 16, 32, 32, torch.eye(3), torch.zeros(3))
    model = Gaussians3D.from_means_covs(
        torch.tensor([[-0.12, 0.05, 2.0], [0.1, -0.07, 2.3], [0.06, 0.12, 1.8]]),
        torch.diag(torch.tensor([0.04, 0.01, 0.02])).repeat(3, 1, 1),
        torch.tensor([[0.3, 0.5, 0.7], [0.7, 0.4, 0.2], [0.4, 0.8, 0.3]]),
        torch.tensor([0.0, 0.3, 1.0]),
    ).with_sh_degree(3)
    model.quats *= 1.3
    target = torch.rand(32, 32, 3)
    view = {
        "camera": camera,
        "rgb": target,
        "field_high": target * 0.97 + 0.01,
        "initialization_masks": torch.zeros(32, 32, dtype=torch.bool),
    }
    state = {"kind": "rgb_final", "sh_degree": 3}
    return model.to(device), state, view


def selftest(device="cpu") -> dict:
    from rtgs.render.base import get_rasterizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA selftest requested but CUDA unavailable")
    model, state, view = synthetic_inputs()
    renderer = get_rasterizer("gsplat" if device.startswith("cuda") else "torch", device=device)
    original = tensor_digest(model_tensors(model))
    components = evaluate_target(model, state, view, "rgb", renderer, device)
    direct = evaluate_target(model, state, view, "rgb", renderer, device, direct=True)
    combined = {
        "losses": components["losses"],
        "gradients": {"total": components["gradients"]["total"]},
    }
    full_control = identity_comparison(combined, direct, atol=1e-7, rtol=1e-4)
    assert full_control["passed"], full_control
    repeated = evaluate_target(model, state, view, "field_high", renderer, device, identity=True)
    identity = identity_comparison(components, repeated)
    assert identity["passed"], identity
    assert tensor_digest(model_tensors(model)) == original
    for component in COMPONENTS:
        assert components["gradients"][component]["opacities"][[0, 2]].eq(0).all()
    initial = model.with_sh_degree(0)
    initial_result = evaluate_target(
        initial, {"kind": "initial", "sh_degree": 0}, view, "rgb", renderer, device
    )
    assert all(
        initial_result["gradients"][component]["shN"].eq(0).all() for component in COMPONENTS
    )
    parameters = fresh_parameters(model, "rgb_final", device)
    assert torch.equal(parameters["opacities"].cpu(), model.opacity.cpu())
    assert torch.equal(parameters["quats"].cpu(), model.quats.cpu())
    entry = fresh_parameters(initial, "initial", device)["opacities"]
    expected = torch.sigmoid(torch.logit(initial.opacity.to(device).clamp(1e-4, 1 - 1e-4)))
    assert torch.equal(entry, expected)
    mask = torch.zeros(17, 19, dtype=torch.bool)
    mask[4:13, 5:14] = True
    partition = regions(mask)
    assert sum(part.long() for part in partition.values()).eq(1).all()
    assert int(regions(torch.ones(17, 19, dtype=torch.bool))["boundary"].sum()) == 17 * 19 - 11 * 13
    photo = torch.zeros(17, 19, 3)
    field = torch.full_like(photo, 0.25)
    residual = residual_statistics(photo, field, mask)
    for row in residual["regions"].values():
        assert row["signed_mean_per_channel"] == [0.25] * 3
        assert row["mae_region"] == 0.25 and row["mse_region"] == 0.0625
    assert (
        abs(
            sum(row["absolute_error_share_full_canvas"] for row in residual["regions"].values()) - 1
        )
        < 1e-12
    )
    zero = residual_statistics(photo, photo, torch.zeros_like(mask))
    assert zero["regions"]["interior"]["pixel_count"] == 0
    assert zero["regions"]["interior"]["mae_region"] is None
    assert zero["regions"]["exterior"]["absolute_error_share_full_canvas"] is None
    assert compare_gradients(torch.zeros(3), torch.ones(3))["cosine"] is None
    assert compare_gradients(torch.zeros(3), torch.ones(3))["difference_over_photo_norm"] is None
    fake_root = ROOT / ".scratch/__field_target_gradient_guard_selftest__"
    fake_task = {"source_run": "old_run", "cached_inputs": [{"path": "old_run/allowed.npz"}]}
    guard = InputGuard(fake_task, fake_root)
    for name in ("dataset/rgb/C0001.jpg", "old_run/heldout/C0001.npz", "old_run/unlisted.json"):
        try:
            guard.audit("open", (str(fake_root / name), "rb", os.O_RDONLY))
        except PermissionError:
            pass
        else:
            raise AssertionError("forbidden path admitted")
    try:
        guard.audit("open", (str(fake_root / "old_run/allowed.npz"), "wb", os.O_WRONLY))
    except PermissionError:
        pass
    else:
        raise AssertionError("prior evidence write admitted")
    guard.audit("open", (str(fake_root / "old_run/allowed.npz"), "rb", os.O_RDONLY))
    deadline_passed = False
    try:
        with Deadlines(0.01):
            signal.pause()
    except TimeoutError:
        deadline_passed = True
    assert deadline_passed
    return {
        "device": device,
        "direct_full_gradient_control": full_control,
        "identity_control": identity,
        "inactive_sh": "zero",
        "opacity_chain": "passed",
        "raw_quaternions": "preserved",
        "state_unchanged": True,
        "regions_and_denominators": "passed",
        "input_guard": "four_denials_one_allowed",
        "signal_deadline": "passed",
    }


def warmup(renderer):
    model, state, view = synthetic_inputs()
    evaluate_target(model, state, view, "rgb", renderer, "cuda:0")
    torch.cuda.synchronize()


def report_module():
    path = Path(__file__).with_name(Path(__file__).stem + "_report.py")
    spec = importlib.util.spec_from_file_location("field_target_gradient_report", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resource_receipt(start):
    return {
        "torch_cuda_max_memory_allocated": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
        "torch_cuda_max_memory_reserved": torch.cuda.max_memory_reserved()
        if torch.cuda.is_available()
        else None,
        "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "ru_maxrss_unit": "KiB",
        "wall_seconds": time.perf_counter() - start,
    }


def write_environment(run: Path):
    packages = {}
    for name in ("torch", "gsplat", "numpy", "Pillow", "rtgs"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed as distribution"
    available = torch.cuda.is_available()
    write_json(
        run / "environment.json",
        {
            "schema_version": 1,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "device": {
                "type": "cuda",
                "name": torch.cuda.get_device_name(0) if available else "unavailable",
                "cuda": torch.version.cuda,
            },
        },
    )
    write_json(
        run / "environment_details.json",
        {
            "available": available,
            "capability": list(torch.cuda.get_device_capability(0)) if available else None,
            "source_commit": read_json(run / "task.lock.json")["source_commit"],
            "torch_threads": torch.get_num_threads(),
            "torch_cudnn_version": torch.backends.cudnn.version(),
        },
    )


def write_boundary_resources(task, run, execution, guard):
    boundary = {
        "schema_version": 1,
        "policy": task["cached_input_policy"],
        "allowed_cached_inputs": task["cached_inputs"],
        "guard": guard.receipt,
        "raw_dataset_reads_permitted": False,
        "prior_run_writes_permitted": False,
        "masks": "residual strata only; no fitting-loss or gradient mask",
        "optimizer_steps": 0,
        "topology_updates": 0,
    }
    for phase in ("entry", "exit"):
        path = run / f"input_integrity_{phase}.json"
        boundary[f"integrity_{phase}"] = read_json(path) if path.exists() else None
    write_json(run / "input_boundary_receipt.json", boundary)
    write_json(
        run / "resource_receipt.json",
        {
            "schema_version": 1,
            "scope": task["resource_protocol"]["scope"],
            "resource": execution["resource"],
            "stage_intervals": execution["stage_intervals"],
            "cuda_peak_scope": "after synthetic warmup through diagnostics and presentation",
            "wall_scope": (
                "coordinator entry through completed diagnostic receipts; report generation follows"
            ),
            "state_intervals": {
                path.parent.name: read_json(path)["stage_interval"]
                for path in sorted((run / "states").glob("*/diagnostics.json"))
            },
            "status": execution["status"],
        },
    )


def coordinate(task_path: Path, task: dict, run: Path):
    start = time.perf_counter()
    clock = RunClock(run)
    execution = {
        "status": "running",
        "stage_intervals": {},
        "states": [],
        "optimization_steps": 0,
        "topology_updates": 0,
        "started_run_seconds": clock(),
    }
    guard = InputGuard(task)
    guard.install()
    try:
        with Deadlines(task["execution_budget"]["max_seconds"]) as deadlines:
            write_environment(run)
            source_start = clock()
            entry = {"source": source_guard(task_path, task, run), "cache": cache_guard(task)}
            execution["verification_entry_interval"] = [source_start, clock()]
            write_json(run / "input_integrity_entry.json", entry)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable: frozen diagnostic has no CPU fallback")
            from rtgs.render.base import get_rasterizer

            settings = dict(task["gradient_protocol"]["renderer"])
            backend = settings.pop("backend")
            renderer = get_rasterizer(backend, device="cuda:0", **settings)
            warmup(renderer)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            guard.phase = "residual"
            residual_start = clock()
            cached = load_cached(task)
            residual = {"rows": [], "stage_interval": [residual_start, residual_start]}
            for ordinal, view in enumerate(cached["views"], 1):
                row = {
                    "view_id": view["view_id"],
                    "input_sha256": view["input_sha256"],
                    "step": ordinal,
                    **residual_statistics(
                        view["rgb"], view["field_high"], view["initialization_masks"]
                    ),
                    "wall_seconds": clock(),
                }
                residual["rows"].append(row)
                residual["stage_interval"][1] = clock()
                write_json(run / "residuals.json", residual)
            execution["stage_intervals"]["residual"] = [residual_start, clock()]
            guard.phase = "gradient"
            gradient_start = clock()
            by_id = {state["id"]: state for state in task["states"]}
            for state_id in task["gradient_protocol"]["state_order"]:
                state_diagnostics(task, run, by_id[state_id], cached, renderer, clock, deadlines)
                execution["states"].append(state_id)
                execution["stage_intervals"]["gradient"] = [gradient_start, clock()]
                execution["resource"] = resource_receipt(start)
                write_json(run / "execution.json", execution)
            guard.phase = "presentation"
            presentation_start = clock()
            presentation(task, run, cached)
            execution["stage_intervals"]["presentation"] = [presentation_start, clock()]
            guard.phase = "verification_exit"
            verify_start = clock()
            exit_receipt = {
                "source": source_guard(task_path, task, run),
                "cache": cache_guard(task),
                "split_sha256": hashlib.sha256(
                    json.dumps(task["splits"], sort_keys=True).encode()
                ).hexdigest(),
            }
            write_json(run / "input_integrity_exit.json", exit_receipt)
            execution["verification_exit_interval"] = [verify_start, clock()]
            execution.update(
                status="completed", resource=resource_receipt(start), finished_run_seconds=clock()
            )
            write_json(run / "input_access.json", guard.receipt)
            write_json(run / "execution.json", execution)
            write_boundary_resources(task, run, execution, guard)
            guard.phase = "report"
            report_module().publish(task, run)
    except BaseException as error:
        guard.phase = "failure_exit_verification"
        failure_integrity = {}
        for name, check in (
            ("source", lambda: source_guard(task_path, task, run)),
            ("cache", lambda: cache_guard(task)),
        ):
            try:
                failure_integrity[name] = check()
            except BaseException as integrity_error:
                failure_integrity[name] = {"all_match": False, "error": str(integrity_error)}
        write_json(run / "input_integrity_exit.json", failure_integrity)
        execution.update(
            status="failed",
            error=str(error),
            traceback=traceback.format_exc(),
            resource=resource_receipt(start),
            finished_run_seconds=clock(),
        )
        write_json(run / "execution.json", execution)
        write_json(run / "input_access.json", guard.receipt)
        write_boundary_resources(task, run, execution, guard)
        write_json(
            run / "execution_failure.json",
            {"error": str(error), "traceback": traceback.format_exc()},
        )
        try:
            report_module().publish_failure(task, run, str(error))
        except BaseException as report_error:
            write_json(run / "failure_report_error.json", {"error": str(report_error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "selftest"))
    parser.add_argument("--task", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "selftest":
        print(json.dumps(selftest(args.device), indent=2, allow_nan=False))
        return
    if args.task is None:
        parser.error("--task is required")
    task_path = args.task.resolve()
    task = read_json(task_path)
    run = ROOT / "runs" / task["task_id"]
    if (run / "execution.json").exists() or (run / "residuals.json").exists():
        raise RuntimeError("diagnostic attempt already exists; no automatic overwrite or retry")
    coordinate(task_path, task, run)


if __name__ == "__main__":
    main()
