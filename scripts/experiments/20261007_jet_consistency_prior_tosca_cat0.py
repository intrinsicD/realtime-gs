"""Jet-consistency prior screen on TOSCA cat0 renders; every fitting cell is a fresh process.

    .venv/bin/python scripts/experiments/20261007_jet_consistency_prior_tosca_cat0.py run \
        --task experiments/tasks/20261007_jet_consistency_prior_tosca_cat0.json \
        --run-dir runs/20261007_jet_consistency_prior_tosca_cat0

Arms (PREREG benchmarks/results/20261007_jet_consistency_prior_tosca_cat0_PREREG.md): base
(lambda_jet = 0), jet1 (first-order control, S = 0, lambda 0.1), jet2 (jet prior, lambda 0.1),
descriptive jet2 at lambda 0.01 and 1.0 (seed 9561). Identical Trainer configuration otherwise
(resolved_training_configs in the task). Fitting reads the 128 training views only; held-out
views and the source mesh open only in ``evaluate`` (enforced by an audit hook). Geometry
metrics and the companion-frame export come from ``eval_jet_prior_tosca_cat0.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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
from statistics import mean

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATASET = "tosca_cat0"
CONDITIONS = ("base", "jet1", "jet2", "jet2_l001", "jet2_l1")
SELECTED = ("base", 9561)
PREVIEWS = (
    "reconstruction_contact_sheet.png",
    "reconstruction.gif",
    "novel_orbit.gif",
    "novel_elevation.gif",
)
GEOMETRY = (
    "normal_angle_median",
    "normal_angle_p90",
    "centre_distance_median",
    "centre_distance_p90",
    "centre_distance_over_h_median",
    "zl_err_k10",
)
APPEARANCE = ("heldout_psnr", "heldout_lpips")
FIT_TIMEOUT = 7200


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


def frame(task: dict) -> Path:
    return ROOT / task["datasets"][0]["frame_path"]


def source_guard(task_path: Path, task: dict, run: Path) -> None:
    lock = read_json(run / "task.lock.json")
    if lock["task_id"] != task["task_id"] or lock["task_sha256"] != sha256(task_path):
        raise RuntimeError("task bytes differ from the approved task lock")
    if task["status"] != "ready" or task["protocol_review"]["verdict"] != "approved":
        raise RuntimeError("protected execution requires an approved ready task")
    snapshot = run / "source_snapshot/manifest.json"
    if snapshot.exists():
        for item in read_json(snapshot)["files"]:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"source changed after snapshot: {item['path']}")


def data_guard(task: dict) -> dict:
    records = read_json(ROOT / task["data_seal"])["files"]
    for item in records:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"sealed input changed: {item['path']}")
    provenance = read_json(frame(task) / "source/PROVENANCE.json")
    for name, item in provenance["files"].items():
        if sha256(frame(task) / "source" / name) != item["sha256"]:
            raise RuntimeError(f"mesh source changed: {name}")
    return {"files_checked": len(records) + len(provenance["files"]), "all_match": True}


def access_guard(task: dict, phase: str) -> dict:
    """Deny held-out views and the mesh outside ``evaluate`` at the Python open boundary."""
    root = frame(task).resolve()
    heldout = set(task["splits"][DATASET]["heldout"])
    receipt = {"phase": phase, "dataset_opens": 0, "denied": []}

    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(root):
            return
        relative = path.relative_to(root)
        if phase != "evaluate" and (
            path.stem.removeprefix("mask_") in heldout or relative.parts[0] == "source"
        ):
            receipt["denied"].append(str(relative))
            raise PermissionError(f"{phase}: held-out view or mesh before evaluation: {relative}")
        receipt["dataset_opens"] += 1

    sys.addaudithook(audit)
    return receipt


def load_views(task: dict, views: list[str]) -> SceneData:
    from rtgs.data.calibrated import load_calibrated_scene

    scene = load_calibrated_scene(
        frame(task),
        calibration_path=ROOT / task["datasets"][0]["calibration"],
        view_ids=views,
        test_every=0,
    )
    if scene.masks is None:
        raise RuntimeError("missing masks")
    return scene


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
    diff = subprocess.run(["git", "diff"], cwd=ROOT, capture_output=True, check=True).stdout
    (directory / "tracked.diff").write_bytes(diff)
    write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "files": files,
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip(),
            "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        },
    )


def write_environment(run: Path) -> None:
    from importlib.metadata import PackageNotFoundError, version

    if (run / "environment.json").exists():
        return
    packages = {}
    for name in ("torch", "numpy", "gsplat", "lpips", "trimesh", "rtgs"):
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


def lpips_model():
    import lpips

    return lpips.LPIPS(net="alex", pretrained=True, pnet_rand=False).eval().to("cuda:0")


def check_protocol_tables(task: dict) -> None:
    if [item["id"] for item in task["comparators"]] != list(CONDITIONS):
        raise RuntimeError("driver conditions differ from frozen comparators")
    if set(task["jet_prior_configs"]) != set(CONDITIONS):
        raise RuntimeError("jet prior configs must cover exactly the conditions")
    if set(task["resolved_training_configs"]) != set(CONDITIONS):
        raise RuntimeError("resolved training configs must cover exactly the conditions")
    split = task["splits"][DATASET]
    if set(split["train"]) & set(split["heldout"]):
        raise RuntimeError("train and held-out views overlap")
    for condition, seed in task["execution_order"]["cells"]:
        if str(seed) not in task["resolved_training_configs"][condition]:
            raise RuntimeError(f"missing resolved config for {condition}/{seed}")


def train_config(task: dict, condition: str, seed: int):
    from rtgs.optim.density import DensityConfig
    from rtgs.optim.trainer import TrainConfig

    resolved = task["resolved_training_configs"][condition][str(seed)]
    frozen = dict(resolved)
    frozen["density"] = DensityConfig(**frozen["density"])
    config = TrainConfig(**frozen)
    if json.loads(json.dumps(asdict(config))) != resolved:
        raise RuntimeError("resolved training configuration does not round-trip")
    return config


# --------------------------------------------------------------------------- stages


def prepare(task: dict, run: Path) -> None:
    started = run_seconds(run)
    access = access_guard(task, "prepare")
    scene = load_views(task, task["splits"][DATASET]["train"])
    center, extent = scene.bounds_hint
    write_json(
        run / "preparation.json",
        {
            "train_views": scene.view_names,
            "bounds": {"center": center.tolist(), "extent": float(extent)},
            "render_receipt_sha256": sha256(frame(task) / "render_receipt.json"),
            "access_guard": access,
            "stage_intervals": {"prepare": [started, run_seconds(run)]},
        },
    )


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
        rows = torch.arange(distances.shape[0])
        distances[rows, rows + start] = math.inf
        nearest[start : start + 2048] = distances.min(1).values
    return {"means": means, "scale": float(nearest.mean())}


def initialize(task: dict, run: Path) -> None:
    started = run_seconds(run)
    access = access_guard(task, "initialize")
    bounds = read_json(run / "preparation.json")["bounds"]
    count = task["initialization"]["n_points"]
    records = {}
    for seed in task["seeds"]:
        rand = random_initialization(torch.tensor(bounds["center"]), bounds["extent"], count, seed)
        model = Gaussians3D.from_means_covs(
            rand["means"],
            torch.eye(3).expand(count, 3, 3) * rand["scale"] ** 2,
            torch.full((count, 3), 0.5),
            torch.full((count,), 0.1),
        )
        directory = run / "initialization" / str(seed)
        directory.mkdir(parents=True)
        model.save_npz(directory / "gaussians_init.npz")
        records[str(seed)] = {
            "n_gaussians": model.n,
            "scale": rand["scale"],
            "sha256": sha256(directory / "gaussians_init.npz"),
        }
    write_json(
        run / "initialization.json",
        {
            "models": records,
            "bounds": bounds,
            "access_guard": access,
            "stage_intervals": {"initialize": [started, run_seconds(run)]},
        },
    )


def fit(task: dict, run: Path, condition: str, seed: int) -> None:
    from rtgs.optim.jet_prior import JetPrior
    from rtgs.optim.trainer import Trainer

    access = access_guard(task, "fit")
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
        scene = load_views(task, task["splits"][DATASET]["train"])
        initial_path = run / "initialization" / str(seed) / "gaussians_init.npz"
        receipt["initial_sha256"] = sha256(initial_path)
        if (
            receipt["initial_sha256"]
            != read_json(run / "initialization.json")["models"][str(seed)]["sha256"]
        ):
            raise RuntimeError("shared initialization digest mismatch")
        initial = Gaussians3D.load_npz(initial_path)
        shutil.copyfile(initial_path, output / "gaussians_init.npz")
        initial.save_ply(output / "gaussians_init.ply")
        config = train_config(task, condition, seed)
        checkpoints = []

        def checkpoint(model, step):
            for tensor in (model.means, model.quats, model.log_scales, model.opacity, model.sh):
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"nonfinite model at step {step}")
            checkpoints.append({"step": step, "run_seconds": run_seconds(run), "n": model.n})
            print(f"{condition}/{seed} step={step} n={model.n}", flush=True)

        write_json(
            output / "effective_config.json",
            {
                "condition": condition,
                "train_config": asdict(config),
                "jet_prior": task["jet_prior_configs"][condition],
                "scene_bounds": read_json(run / "preparation.json")["bounds"],
                "train_view_ids": scene.view_names,
            },
        )
        jet = task["jet_prior_configs"][condition]
        prior = JetPrior(jet["weight"], order=jet["order"]) if jet["weight"] else None
        final, history = Trainer(config).train(
            scene, initial, checkpoint_callback=checkpoint, jet_prior=prior
        )
        torch.cuda.synchronize()
        if history["executed_iterations"] != config.iterations:
            raise RuntimeError("trainer did not reach the frozen final iteration")
        if not all(math.isfinite(value) for value in history["loss"]):
            raise RuntimeError("nonfinite fitting loss")
        final.save_npz(output / "gaussians.npz")
        final.save_ply(output / "gaussians.ply")
        terms = history.pop("loss_terms")
        history["jet_regularization"] = [t["jet_regularization"] for t in terms[::50]]
        history.pop("sampled_train_views", None)
        history["checkpoint_observer"] = checkpoints
        write_json(output / "history.json", history)
        receipt.update(
            status="completed",
            final_count=final.n,
            final_sha256=sha256(output / "gaussians.npz"),
            iterations=config.iterations,
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


def appearance(renderer, model, scene: SceneData, perceptual) -> list[dict]:
    """Held-out PSNR inside the mask and masked-crop LPIPS, rendered over white."""
    white = torch.ones(3, device="cuda:0")
    rows = []
    for index, name in enumerate(scene.view_names):
        with torch.no_grad():
            out = renderer.render(model, scene.cameras[index].to("cuda:0"), background=white)
        prediction = out.color.clamp(0, 1).cpu()
        reference = scene.images[index]
        mask = scene.masks[index] > 0.5
        error = (prediction - reference).square().mean(-1)[mask].mean().clamp_min(1e-12)
        where = torch.nonzero(mask)
        y0, x0 = (where.min(0).values - 8).clamp_min(0).tolist()
        y1, x1 = (where.max(0).values + 9).tolist()
        weight = mask[..., None].float()
        left = (prediction * weight)[y0:y1, x0:x1].permute(2, 0, 1)[None].to("cuda:0")
        right = (reference * weight)[y0:y1, x0:x1].permute(2, 0, 1)[None].to("cuda:0")
        with torch.no_grad():
            lp = float(perceptual(left, right, normalize=True))
        rows.append(
            {"view_id": name, "heldout_psnr": float(-10 * torch.log10(error)), "heldout_lpips": lp}
        )
    return rows


def evaluate(task: dict, run: Path) -> None:
    import eval_jet_prior_tosca_cat0 as geometry_eval

    from rtgs.render.base import get_rasterizer
    from rtgs.visualize import save_reconstruction_artifacts

    for condition, seed in task["execution_order"]["cells"]:
        if (
            read_json(run / "cells" / condition / str(seed) / "receipt.json")["status"]
            != "completed"
        ):
            raise RuntimeError("evaluation requires every final model to be saved")
    access = access_guard(task, "evaluate")
    heldout = load_views(task, task["splits"][DATASET]["heldout"])
    preview = SceneData(
        heldout.images,
        heldout.cameras,
        view_names=heldout.view_names,
        masks=None,
        train_indices=[],
        test_indices=list(range(len(heldout.images))),
        bounds_hint=heldout.bounds_hint,
    )
    perceptual = lpips_model()
    renderer = get_rasterizer("gsplat", device="cuda:0")
    for condition, seed in task["execution_order"]["cells"]:
        started = run_seconds(run)
        output = run / "cells" / condition / str(seed)
        final = Gaussians3D.load_npz(output / "gaussians.npz").to("cuda:0")
        initial = Gaussians3D.load_npz(output / "gaussians_init.npz").to("cuda:0")
        rows = appearance(renderer, final, heldout, perceptual)
        geometry = geometry_eval.evaluate_model(final, output)
        artifacts = save_reconstruction_artifacts(
            preview,
            initial,
            final,
            output,
            rasterizer="gsplat",
            max_comparisons=8,
            max_animation_frames=24,
        )
        means = {key: mean(row[key] for row in rows) for key in APPEARANCE}
        write_json(
            output / "evaluation.json",
            {
                "per_view": rows,
                "mean": {**means, **{key: geometry[key] for key in GEOMETRY}},
                "geometry": geometry,
                "artifacts": artifacts,
                "access_guard": access,
                "stage_intervals": {"evaluate": [started, run_seconds(run)]},
            },
        )
        print(
            f"evaluated {condition}/{seed}: {json.dumps(geometry_summary(geometry, means))}",
            flush=True,
        )


def geometry_summary(geometry: dict, means: dict) -> dict:
    return {**{key: round(geometry[key], 6) for key in GEOMETRY}, **means}


# --------------------------------------------------------------------------- decision and report


def gates(task: dict, cells: dict) -> dict:
    """Frozen PREREG rule, per paired seed, jet2 vs base (and jet1 descriptively)."""
    result = {}
    for arm in ("jet2", "jet1"):
        rows = []
        for seed in task["seeds"]:
            a, b = cells[(arm, seed)], cells[("base", seed)]
            normal_gain = b["normal_angle_median"] - a["normal_angle_median"]
            centre_ratio = a["centre_distance_median"] / b["centre_distance_median"]
            psnr_delta = a["heldout_psnr"] - b["heldout_psnr"]
            rows.append(
                {
                    "seed": seed,
                    "normal_median_gain_deg": normal_gain,
                    "centre_median_ratio": centre_ratio,
                    "psnr_delta_db": psnr_delta,
                    "pass": normal_gain >= 4 and centre_ratio <= 0.7 and psnr_delta >= -0.2,
                    "loses_normal": normal_gain < 0,
                    "loses_centre": centre_ratio > 1,
                }
            )
        if all(r["pass"] for r in rows):
            verdict = "pass"
        elif all(r["loses_normal"] for r in rows) or all(r["loses_centre"] for r in rows):
            verdict = "reject"
        else:
            verdict = "inconclusive"
        result[arm] = {"verdict": verdict, "rows": rows}
    result["jet1_descriptive_only"] = True
    return result


def _commands(task: dict, run: Path) -> dict:
    relative = run.relative_to(ROOT).as_posix()
    return {
        "reproduce": task["run_command"],
        "serve_report": [".venv/bin/python", "-m", "http.server", "8765", "--directory", relative],
        "viewer": [
            ".venv/bin/rtgs",
            "view",
            "--gaussians",
            f"{relative}/gaussians.ply",
            "--initial",
            f"{relative}/gaussians_init.ply",
            "--no-open",
            "--port",
            "8879",
        ],
    }


def _receipt(task: dict, run: Path, *, failed: str | None = None) -> None:
    lock = read_json(run / "task.lock.json")
    write_json(
        run / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": task["task_id"],
            "status": "failed" if failed else "completed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": 1 if failed else 0,
            "failure_phase": "execution" if failed else None,
            "message": failed or "All frozen fitting cells and final evaluations completed.",
        },
    )


def history(task: dict, cells: list[dict]) -> dict:
    metadata = {
        "loss_total": {
            "label": "Training objective",
            "unit": "loss",
            "group": "Fitting",
            "direction": "lower",
        },
        "n_gaussians": {
            "label": "Live 3D Gaussians",
            "unit": "count",
            "group": "Capacity",
            "direction": "descriptive",
        },
    }
    records, markers = [], []
    for cell in cells:
        receipt, trace = cell["receipt"], cell["history"]
        total = trace["executed_iterations"]
        common = {"dataset_id": DATASET, "arm_id": cell["condition"], "seed": cell["seed"]}
        for stage in task["stages"]:
            start, end = receipt["stage_intervals"][stage["id"]]
            first = total if stage["id"] == "evaluate" else 0
            last = total if stage["id"] in {"fit", "evaluate"} else 0
            for boundary, stamp, step in [("start", start, first), ("end", end, last)]:
                markers.append(
                    {
                        **common,
                        "stage": stage["id"],
                        "boundary": boundary,
                        "label": stage["label"],
                        "step": step,
                        "wall_seconds": stamp,
                    }
                )
        losses = dict(enumerate(trace["loss"], start=1))
        counts = dict(trace["n_gaussians"])
        observed = {row["step"]: row["run_seconds"] for row in trace["checkpoint_observer"]}
        for step, _elapsed in trace["elapsed"]:
            for metric, values in [("loss_total", losses), ("n_gaussians", counts)]:
                if step in values and step in observed:
                    records.append(
                        {
                            **common,
                            "stage": "fit",
                            "split": "train",
                            "step": step,
                            "wall_seconds": observed[step],
                            "metric_id": metric,
                            "value": values[step],
                        }
                    )
    return {
        "schema_version": 2,
        "records": records,
        "metric_metadata": metadata,
        "stage_markers": markers,
    }


def publish(task: dict, run: Path) -> dict:
    run = run.resolve()
    preparation = read_json(run / "preparation.json")
    initialization = read_json(run / "initialization.json")
    cells, table = [], {}
    for condition, seed in task["execution_order"]["cells"]:
        directory = run / "cells" / condition / str(seed)
        cell = {
            "condition": condition,
            "seed": seed,
            "path": str(directory.relative_to(run)),
            "receipt": read_json(directory / "receipt.json"),
            "history": read_json(directory / "history.json"),
            "evaluation": read_json(directory / "evaluation.json"),
        }
        if cell["receipt"]["status"] != "completed":
            raise ValueError("incomplete fitting cell")
        cell["receipt"]["stage_intervals"] = {
            **preparation["stage_intervals"],
            **initialization["stage_intervals"],
            **cell["receipt"]["stage_intervals"],
            **cell["evaluation"]["stage_intervals"],
        }
        table[(condition, seed)] = cell["evaluation"]["mean"]
        cells.append(cell)
    keys = list(GEOMETRY + APPEARANCE)
    groups = {
        condition: {
            key: mean(c["evaluation"]["mean"][key] for c in cells if c["condition"] == condition)
            for key in keys
        }
        for condition in CONDITIONS
    }
    result_gates = gates(task, table)
    decision = (
        f"PREREG decision (jet2 vs base, every seed): {result_gates['jet2']['verdict']}; "
        f"jet1 under the same rule (descriptive): {result_gates['jet1']['verdict']}. "
        "Development screen on one synthetic object; no default promotion."
    )
    write_json(
        run / "gaussians.config.json",
        {
            "protocol": task,
            "cells": [
                {
                    "condition": c["condition"],
                    "seed": c["seed"],
                    "effective": read_json(run / c["path"] / "effective_config.json"),
                }
                for c in cells
            ],
        },
    )
    write_json(
        run / "comparison.json",
        {
            "groups": groups,
            "gates": result_gates,
            "cells": [
                {
                    "condition": c["condition"],
                    "seed": c["seed"],
                    "metrics": c["evaluation"]["mean"],
                    "geometry": c["evaluation"]["geometry"],
                    "receipt": c["receipt"],
                }
                for c in cells
            ],
        },
    )
    write_json(run / "training_history.json", history(task, cells))
    selected = run / "cells" / SELECTED[0] / str(SELECTED[1])
    for name in ("gaussians_init.ply", "gaussians.ply", *PREVIEWS):
        shutil.copyfile(selected / name, run / name)
    write_json(
        run / "resource_receipt.json",
        {
            "scope": task["resource_protocol"]["scope"],
            "performance_inference": False,
            "cells": [c["receipt"] for c in cells],
        },
    )
    write_json(
        run / "input_boundary_receipt.json",
        {
            "task_id": task["task_id"],
            "split": task["splits"],
            "boundary": task["claim_boundary"],
            "preparation_access": preparation["access_guard"],
            "initialization_access": initialization["access_guard"],
            "cell_receipts": [f"{c['path']}/receipt.json" for c in cells],
            "heldout_role": "Reporting-only after every final fitting model was saved.",
            "mesh_role": "The source mesh is read only by offline rendering and by evaluate.",
        },
    )
    _receipt(task, run)
    metric_values, metadata = {}, {}
    labels = {m["id"]: m for m in task["primary_metrics"] + task["secondary_metrics"]}
    for condition in CONDITIONS:
        for key in keys:
            spec = labels.get(key, {"label": key, "unit": "", "direction": "descriptive"})
            metric_values[f"{condition}_{key}"] = groups[condition][key]
            metadata[f"{condition}_{key}"] = {
                "label": f"{condition}: {spec['label']}",
                "unit": spec["unit"] or "-",
                "group": condition,
                "direction": spec["direction"],
            }
    for spec in task["primary_metrics"]:
        metric_values[spec["id"]] = groups["jet2"][spec["id"]]
        metadata[spec["id"]] = {
            "label": f"jet2 (3-seed mean): {spec['label']}",
            "unit": spec["unit"],
            "group": "primary",
            "direction": spec["direction"],
        }
    summary = (
        f"{len(cells)} cells completed. 3-seed means, normal-angle median: "
        + ", ".join(f"{c} {groups[c]['normal_angle_median']:.2f} deg" for c in CONDITIONS)
        + "; held-out PSNR: "
        + ", ".join(f"{c} {groups[c]['heldout_psnr']:.2f} dB" for c in CONDITIONS)
        + "."
    )
    stem = f"benchmarks/results/{task['task_id']}"
    evidence = [
        {"label": label, "path": f"{stem}_{suffix}"}
        for label, suffix in [
            ("Result note", "RESULT.md"),
            ("Machine result", "RESULT.json"),
            ("Audit", "AUDIT.md"),
            ("Machine audit", "AUDIT.json"),
        ]
    ]
    core = [
        "gaussians_init.ply",
        "gaussians.ply",
        "training_history.json",
        "gaussians.config.json",
        "input_boundary_receipt.json",
        "resource_receipt.json",
        "run_receipt.json",
        "environment.json",
        "comparison.json",
        "preparation.json",
        "initialization.json",
        "source_snapshot/manifest.json",
        *PREVIEWS,
    ]
    artifacts = [{"label": name, "path": name} for name in core]
    for cell in cells:
        for name in (
            "receipt.json",
            "evaluation.json",
            "gaussians.ply",
            "gaussians_splat_frame.ply",
            *PREVIEWS,
        ):
            artifacts.append({"label": f"{cell['path']}/{name}", "path": f"{cell['path']}/{name}"})
    charts = [
        {
            "id": "quality",
            "title": "Normal angle to the mesh, median (3-seed mean)",
            "unit": "deg",
            "values": [{"label": c, "value": groups[c]["normal_angle_median"]} for c in CONDITIONS],
        },
        {
            "id": "resources",
            "title": "Peak allocated CUDA memory (descriptive)",
            "unit": "bytes",
            "values": [
                {
                    "label": f"{c['condition']}/{c['seed']}",
                    "value": c["receipt"]["peak_allocated_bytes"],
                }
                for c in cells
            ],
        },
        {
            "id": "stage_runtime",
            "title": "Recorded stage durations (descriptive)",
            "unit": "seconds",
            "values": [
                {"label": f"{c['condition']}/{c['seed']}/{stage}", "value": b[1] - b[0]}
                for c in cells
                for stage, b in c["receipt"]["stage_intervals"].items()
            ],
        },
    ]
    notes = [
        f"Root preview is the prospectively selected {SELECTED[0]}/{SELECTED[1]} model.",
        "Geometry metrics use splats with opacity > 0.3 against cat0.obj (mesh units).",
        "Held-out PSNR/LPIPS inside the held-out mask, rendered over white (as rendered).",
        "The SplatDiffuseLBO operator metric (lbo_err_k10) is deferred: see the task record.",
        "Shared preparation/initialization intervals recur across series and must not be summed.",
    ]
    write_json(
        run / "metrics.json",
        {
            "schema_version": 2,
            "report_template_version": 2,
            "task_id": task["task_id"],
            "summary": summary,
            "decision": decision,
            "claim_boundary": task["claim_boundary"],
            "metrics": metric_values,
            "metric_metadata": metadata,
            "charts": charts,
            "artifacts": artifacts,
            "evidence": evidence,
            "commands": _commands(task, run),
            "notes": notes,
        },
    )
    result = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "summary": summary,
        "decision": decision,
        "claim_boundary": task["claim_boundary"],
        "groups": groups,
        "gates": result_gates,
        "cells": {f"{c}/{s}": v for (c, s), v in table.items()},
        "command": task["run_command"],
        "source_lock": read_json(run / "task.lock.json"),
        "raw_comparison_path": str(run.relative_to(ROOT) / "comparison.json"),
    }
    write_json(ROOT / f"{stem}_RESULT.json", result)
    return result


def publish_failure(task: dict, run: Path, error: str) -> None:
    _receipt(task, run, failed=error)
    if not (run / "training_history.json").exists():
        write_json(
            run / "training_history.json",
            {"schema_version": 2, "records": [], "metric_metadata": {}, "stage_markers": []},
        )
    for name in ("gaussians.config.json", "input_boundary_receipt.json", "resource_receipt.json"):
        if not (run / name).exists():
            write_json(
                run / name, {"task_id": task["task_id"], "status": "failed", "message": error}
            )
    write_json(
        run / "metrics.json",
        {
            "schema_version": 2,
            "report_template_version": 2,
            "task_id": task["task_id"],
            "summary": "Experiment execution stopped before completion.",
            "decision": error,
            "claim_boundary": task["claim_boundary"],
            "metrics": {},
            "metric_metadata": {},
            "charts": [],
            "artifacts": [],
            "evidence": [],
            "commands": {**_commands(task, run), "viewer": None},
            "notes": ["Failed execution is inspectable but not a results-bearing bundle."],
        },
    )


def companion_busy() -> bool:
    return subprocess.run(["pgrep", "-f", "splat_lbo_cat"], capture_output=True).returncode == 0


def coordinate(task_path: Path, task: dict, run: Path, only: str | None) -> None:
    logs = run / "logs"
    logs.mkdir(exist_ok=True)

    def worker(phase, condition=None, seed=None):
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
        if condition is not None:
            command += ["--condition", condition, "--seed", str(seed)]
            name += f"_{condition}_{seed}"
        if phase in ("fit", "evaluate"):
            while companion_busy():  # never share the machine with the companion CPU job
                print("waiting for the companion job to finish", flush=True)
                time.sleep(180)
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
                timeout=FIT_TIMEOUT if phase == "fit" else None,
                check=True,
            )

    try:
        snapshot_source(task, run)
        write_environment(run)
        source_guard(task_path, task, run)
        write_json(run / "input_integrity_entry.json", data_guard(task))
        if not (run / "preparation.json").exists():
            worker("prepare")
        if not (run / "initialization.json").exists():
            worker("initialize")
        for condition, seed in task["execution_order"]["cells"]:
            receipt = run / "cells" / condition / str(seed) / "receipt.json"
            if receipt.exists() and read_json(receipt)["status"] == "completed":
                continue
            if receipt.exists():
                raise RuntimeError(f"cell {condition}/{seed} already attempted and not completed")
            print(f"starting {condition}/{seed}", flush=True)
            worker("fit", condition, seed)
        worker("evaluate")
        source_guard(task_path, task, run)
        split_digest = hashlib.sha256(
            json.dumps(task["splits"], sort_keys=True).encode()
        ).hexdigest()
        write_json(
            run / "input_integrity_exit.json", {**data_guard(task), "split_sha256": split_digest}
        )
        publish(task, run)
    except BaseException as error:
        write_json(
            run / "execution_failure.json",
            {"error": str(error), "traceback": traceback.format_exc()},
        )
        publish_failure(task, run, str(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("run", "prepare", "initialize", "fit", "evaluate", "publish")
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    task_path, run = args.task.resolve(), args.run_dir.resolve()
    task = read_json(task_path)
    if run != ROOT / "runs" / task["task_id"]:
        raise ValueError("only the canonical task run root is allowed")
    check_protocol_tables(task)
    if args.command == "run":
        coordinate(task_path, task, run, None)
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
    elif args.command == "publish":
        publish(task, run)


if __name__ == "__main__":
    main()
