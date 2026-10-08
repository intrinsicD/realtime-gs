"""Field prior v2.1 on 2DGS surfels, TOSCA cat0 renders; every fitting cell is a fresh process.

    .venv/bin/python scripts/experiments/20261008_jet_field_prior_2dgs_tosca_cat0.py run \
        --task experiments/tasks/20261008_jet_field_prior_2dgs_tosca_cat0.json \
        --run-dir runs/20261008_jet_field_prior_2dgs_tosca_cat0
    # afterwards, the sequential CPU operator stage (b2 and jet, seed 9561, 8 h total), then
    # `publish` again to fold it into the bundle:
    .venv/bin/python scripts/experiments/20261008_jet_field_prior_2dgs_tosca_cat0.py operator \
        --task ... --run-dir ...
    # pre-flight, below .scratch/ only, no task lock (not results-bearing):
    ... smoke   --task ... --run-dir .scratch/<task_id>/smoke   --smoke 600
    ... teacher --task ... --run-dir .scratch/<task_id>/teacher

Arms (PREREG benchmarks/results/20261008_jet_field_prior_2dgs_tosca_cat0_PREREG.md, v2.1): b2
(2DGS, scale-normalized depth distortion + centre-depth consistency from 7500), b2_noanchor
(distortion only), jet / jet_noanchor (the same plus the FieldPrior normal term, lambda 0.1, from
step 7500), jet_c (jet plus the centre term, seed 9561, ablation). Data, splits, seeds and the
seeded random initializations are those of v1 (``20261007_jet_consistency_prior_tosca_cat0``);
its driver supplies the shared guards, loaders and initialization. Geometry comes from
``eval_jet_prior_tosca_cat0.py`` plus the v2.1 additions below.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
v1 = importlib.import_module("20261007_jet_consistency_prior_tosca_cat0")
read_json, write_json, sha256, run_seconds = v1.read_json, v1.write_json, v1.sha256, v1.run_seconds

DATASET = "tosca_cat0"
CONDITIONS = ("b2", "b2_noanchor", "jet", "jet_noanchor", "jet_c")
SELECTED = ("b2", 9561)
RASTERIZER = "gsplat-2dgs"
PREVIEWS = v1.PREVIEWS
GEOMETRY = (
    "normal_angle_median",
    "normal_angle_p90",
    "centre_distance_median",
    "centre_distance_over_hbar0_median",
    "coverage",
    "signed_displacement_median",
    "inward_fraction",
    "collapse_fraction",
    "collapse_fraction_at_activation",
    "cohort_normal_angle_median",
    "cohort_normal_angle_p90",
    "cohort_centre_distance_median",
    "n_subset",
    "n_cohort",
    "flatness_max",
    "zl_err_k10",
)
APPEARANCE = ("heldout_psnr", "heldout_lpips", "heldout_silhouette_iou")
COMPANION = Path.home() / "Documents/SplatDiffuseLBO"
COMPANION_PYTHON = Path.home() / "miniconda3/bin/python3.12"
FIT_TIMEOUT = 7200
LOG_EVERY = 500
SMOKE = None  # pre-flight mode (smoke / teacher): shortened or truncated schedule, no task lock
SMOKE_CELLS = [["b2", 9561], ["jet", 9561]]
TEACHER_CELLS = [["b2", 9561]]


# --------------------------------------------------------------------------- configuration


def check_protocol_tables(task: dict) -> None:
    if [item["id"] for item in task["comparators"]] != list(CONDITIONS):
        raise RuntimeError("driver conditions differ from frozen comparators")
    tables = ("resolved_training_configs", "surfel_regularization_configs", "field_prior_configs")
    for key in tables:
        if set(task[key]) != set(CONDITIONS):
            raise RuntimeError(f"{key} must cover exactly the conditions")
    split = task["splits"][DATASET]
    if set(split["train"]) & set(split["heldout"]):
        raise RuntimeError("train and held-out views overlap")
    for condition, seed in task["execution_order"]["cells"]:
        if str(seed) not in task["resolved_training_configs"][condition]:
            raise RuntimeError(f"missing resolved config for {condition}/{seed}")
        if task["resolved_training_configs"][condition][str(seed)]["rasterizer"] != RASTERIZER:
            raise RuntimeError("every cell must train 2DGS surfels")
        if str(seed) not in task["scene_scale"]["values"]:
            raise RuntimeError(f"missing frozen scene scale for seed {seed}")


def scene_scale(model: Gaussians3D) -> float:
    """Bounding-box diagonal of the initialization centres (the frozen distortion scale s)."""
    means = model.means.detach().double()
    return float((means.max(0).values - means.min(0).values).norm())


def cell_setup(task: dict, condition: str, seed: int):
    """``(TrainConfig, SurfelRegularization, FieldPrior | None, h0_step)`` of one cell."""
    from rtgs.optim.density import DensityConfig
    from rtgs.optim.jet_prior import FieldPrior
    from rtgs.optim.trainer import TrainConfig
    from rtgs.render.gsplat_2dgs_backend import SurfelRegularization

    resolved = task["resolved_training_configs"][condition][str(seed)]
    config = TrainConfig(**{**resolved, "density": DensityConfig(**resolved["density"])})
    if json.loads(json.dumps(asdict(config))) != resolved:
        raise RuntimeError("resolved training configuration does not round-trip")
    surfel = SurfelRegularization(
        **task["surfel_regularization_configs"][condition],
        depth_scale=task["scene_scale"]["values"][str(seed)],
    )
    field = dict(task["field_prior_configs"][condition])
    h0_step = task["collapse_gate"]["h0_step"]
    if SMOKE is not None and SMOKE["mode"] == "teacher":  # same run, stopped at h0_step
        config = replace(config, iterations=h0_step, schedule_iterations=config.iterations)
    elif SMOKE is not None:  # shortened, not results-bearing schedule
        h0_step = SMOKE["iterations"] // 2
        config = replace(
            config,
            iterations=SMOKE["iterations"],
            eval_every=100,
            sh_degree_interval=100,
            density=replace(config.density, start_iter=100, stop_iter=h0_step, every=50),
        )
        surfel = replace(surfel, distortion_start=100, normal_start=h0_step)
        field.update(start=h0_step, log_every=100)
    if field["start"] != h0_step or surfel.normal_start != h0_step:
        raise RuntimeError("the field prior and the anchor must start where h0 is frozen")
    prior = FieldPrior(**field) if field["weight"] else None
    return config, surfel, prior, h0_step


# --------------------------------------------------------------------------- fitting


def warmup() -> None:
    from rtgs.render.base import get_rasterizer

    camera = Camera(32, 32, 16, 16, 32, 32, torch.eye(3), torch.zeros(3)).to("cuda:0")
    model = Gaussians3D.from_means_covs(
        torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.1, 2.5]], device="cuda:0"),
        torch.eye(3, device="cuda:0").repeat(2, 1, 1) * 0.01,
        torch.full((2, 3), 0.5, device="cuda:0"),
        torch.full((2,), 0.2, device="cuda:0"),
    )
    model.means.requires_grad_(True)
    get_rasterizer(RASTERIZER, device="cuda:0").render(model, camera).color.mean().backward()
    torch.cuda.synchronize()


def collapse_fraction(means: torch.Tensor, h0: torch.Tensor) -> float:
    """Fraction of splats with another splat closer than 0.05 h0_i (the PREREG hard gate)."""
    from rtgs.optim.jet_prior import knn_indices

    means = means.detach()
    nearest = (means[knn_indices(means, 1)[:, 0]] - means).norm(dim=-1)
    return float((nearest < 0.05 * h0.to(means.device)).double().mean())


def _normals(model) -> torch.Tensor:
    from rtgs.optim.jet_prior import splat_frames

    return splat_frames(model.quats.detach(), model.log_scales.detach())[:, 2]


def fit(task: dict, run: Path, condition: str, seed: int) -> None:
    from rtgs.optim.jet_prior import mean_neighbour_distance
    from rtgs.optim.trainer import Trainer

    access = v1.access_guard(task, "fit")
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
        scene = v1.load_views(task, task["splits"][DATASET]["train"])
        initial_path = run / "initialization" / str(seed) / "gaussians_init.npz"
        receipt["initial_sha256"] = sha256(initial_path)
        if (
            receipt["initial_sha256"]
            != read_json(run / "initialization.json")["models"][str(seed)]["sha256"]
        ):
            raise RuntimeError("shared initialization digest mismatch")
        initial = Gaussians3D.load_npz(initial_path)
        frozen_scale = task["scene_scale"]["values"][str(seed)]
        if not math.isclose(scene_scale(initial), frozen_scale, rel_tol=1e-6):
            raise RuntimeError("the initialization's bbox diagonal differs from the frozen scale")
        shutil.copyfile(initial_path, output / "gaussians_init.npz")
        initial.save_ply(output / "gaussians_init.ply")
        config, surfel, prior, h0_step = cell_setup(task, condition, seed)
        checkpoints, frozen, diagnostics, previous = [], {}, [], {}

        def checkpoint(model, step):
            finite = (model.means, model.quats, model.log_scales[:, :2], model.opacity, model.sh)
            if not all(torch.isfinite(t).all() for t in finite):
                raise RuntimeError(f"nonfinite model at step {step}")
            row = {"step": step, "n": model.n}
            normals = _normals(model)
            if previous.get("n") == model.n:  # displacement since the last checkpoint
                row["means_displacement_median"] = float(
                    (model.means - previous["means"]).norm(dim=-1).median()
                )
                cosine = (normals * previous["normals"]).sum(-1).abs().clamp(max=1)
                row["normal_rotation_median_deg"] = float(
                    torch.rad2deg(torch.acos(cosine)).median()
                )
            previous.update(n=model.n, means=model.means.clone(), normals=normals)
            if step == h0_step:  # end of densification: the frozen h0 and cohort of every arm
                h0 = mean_neighbour_distance(model.means).clamp_min(1e-12)
                frozen.update(
                    h0=h0.cpu(),
                    cohort=(model.opacity > 0.3).cpu(),
                    means=model.means.cpu(),
                    collapse=collapse_fraction(model.means, h0),
                )
            if prior is not None and prior.stats:
                row["prior"] = dict(prior.stats)
            diagnostics.append(row)
            checkpoints.append({"step": step, "run_seconds": run_seconds(run), "n": model.n})
            print(f"{condition}/{seed} step={step} n={model.n}", flush=True)

        def gradients(params, optimizers, step):
            if step % LOG_EVERY == 0:
                diagnostics.append(
                    {
                        "step": step,
                        **{
                            f"grad_{name}_median": float(params[name].grad.norm(dim=-1).median())
                            for name in ("means", "quats")
                            if params[name].grad is not None
                        },
                        "lr_means": optimizers["means"].param_groups[0]["lr"],
                        "lr_quats": optimizers["quats"].param_groups[0]["lr"],
                    }
                )

        write_json(
            output / "effective_config.json",
            {
                "condition": condition,
                "train_config": asdict(config),
                "surfel_regularization": asdict(surfel),
                "field_prior": task["field_prior_configs"][condition],
                "preflight": SMOKE,
                "scene_bounds": read_json(run / "preparation.json")["bounds"],
                "train_view_ids": scene.view_names,
            },
        )
        final, history = Trainer(config).train(
            scene,
            initial,
            checkpoint_callback=checkpoint,
            jet_prior=prior,
            surfel_regularization=surfel,
            parameter_step_callback=gradients,
        )
        torch.cuda.synchronize()
        if history["executed_iterations"] != config.iterations:
            raise RuntimeError("trainer did not reach the frozen final iteration")
        if not all(math.isfinite(value) for value in history["loss"]):
            raise RuntimeError("nonfinite fitting loss")
        if not torch.isneginf(final.log_scales[:, 2]).all():
            raise RuntimeError("2DGS backend check: the third scale must be -inf for every surfel")
        if frozen["h0"].shape[0] != final.n:
            raise RuntimeError("splat count changed after the h0 freeze")
        if prior is not None and not torch.allclose(prior.h0.cpu(), frozen["h0"], rtol=1e-5):
            raise RuntimeError("the prior's frozen h0 differs from the gate's")
        torch.save(frozen, output / "h0.pt")
        final.save_npz(output / "gaussians.npz")
        final.save_ply(output / "gaussians.ply")
        terms = history.pop("loss_terms")
        for key in ("jet_regularization", "depth_distortion", "normal_consistency"):
            history[key] = [[i + 1, t[key]] for i, t in enumerate(terms) if i % 50 == 49]
        history.pop("sampled_train_views", None)
        history["checkpoint_observer"] = checkpoints
        history["diagnostics"] = diagnostics
        history["field_prior_log"] = None if prior is None else prior.log
        write_json(output / "history.json", history)
        receipt.update(
            status="completed",
            final_count=final.n,
            final_sha256=sha256(output / "gaussians.npz"),
            iterations=config.iterations,
            collapse_fraction=collapse_fraction(final.means, frozen["h0"]),
            collapse_fraction_at_activation=frozen["collapse"],
            h0_median=float(frozen["h0"].median()),
            cohort_count=int(frozen["cohort"].sum()),
            field_prior_refreshes=None if prior is None else prior.refreshes,
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


# --------------------------------------------------------------------------- evaluation


def flatness_max(model: Gaussians3D) -> float:
    """max_i sigma_min / sigma_mid; 0 by construction for surfels (a check of the backend)."""
    sigma = model.log_scales.detach().exp().sort(dim=1, descending=True).values
    return float((sigma[:, 2] / sigma[:, 1]).max())


def mesh_geometry(model: Gaussians3D, frozen: dict, coverage_points: int, seed: int) -> dict:
    """v2.1 fixed-unit geometry: final opacity > 0.3 subset and the frozen step-7500 cohort.

    Signed displacement is (c - p) . n_face at the closest mesh point (positive = outside for
    the outward-oriented mesh); coverage is the fraction of area-sampled mesh points with a
    final-subset centre within 0.5 h0-bar, h0-bar = mean frozen h0 over the cohort.
    """
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.load(v1.ROOT / "dataset/external/tosca_cat0/source/cat0.obj", process=False)
    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    model = model.to("cpu")
    normals = _normals(model).double().numpy()
    centres = model.means.detach().double().numpy()

    def measure(rows):
        closest, distance, face = trimesh.proximity.closest_point(mesh, centres[rows])
        cosine = np.abs((normals[rows] * mesh.face_normals[face]).sum(1)).clip(0, 1)
        angle = np.degrees(np.arccos(cosine))
        signed = ((centres[rows] - closest) * mesh.face_normals[face]).sum(1)
        return angle, distance / diagonal, signed / diagonal

    subset = model.opacity.detach().numpy() > 0.3
    cohort = frozen["cohort"].numpy()
    angle, distance, signed = measure(subset)
    c_angle, c_distance, _ = measure(cohort)
    hbar0 = float(frozen["h0"][frozen["cohort"]].double().mean())
    samples, _ = trimesh.sample.sample_surface(mesh, coverage_points, seed=seed)
    near = cKDTree(centres[subset]).query(samples, distance_upper_bound=0.5 * hbar0)[0]
    return {
        "n_total": model.n,
        "n_subset": int(subset.sum()),
        "n_cohort": int(cohort.sum()),
        "n_subset_and_cohort": int((subset & cohort).sum()),
        "normal_angle_median": float(np.median(angle)),
        "normal_angle_p90": float(np.percentile(angle, 90)),
        "centre_distance_median": float(np.median(distance)),
        "centre_distance_p90": float(np.percentile(distance, 90)),
        "centre_distance_over_hbar0_median": float(np.median(distance) * diagonal / hbar0),
        "signed_displacement_median": float(np.median(signed)),
        "inward_fraction": float((signed < 0).mean()),
        "cohort_normal_angle_median": float(np.median(c_angle)),
        "cohort_normal_angle_p90": float(np.percentile(c_angle, 90)),
        "cohort_centre_distance_median": float(np.median(c_distance)),
        "coverage": float(np.isfinite(near).mean()),
        "coverage_radius_mesh_units": 0.5 * hbar0,
        "bbox_diagonal": diagonal,
        "flatness_max": flatness_max(model),
    }


def appearance(renderer, model, scene: SceneData, perceptual) -> list[dict]:
    """v1 held-out PSNR/LPIPS plus the silhouette IoU (alpha > 0.5 vs mask > 0.5)."""
    rows = v1.appearance(renderer, model, scene, perceptual)
    for index, row in enumerate(rows):
        with torch.no_grad():
            alpha = renderer.render(model, scene.cameras[index].to("cuda:0")).alpha.cpu()
        mask = scene.masks[index] > 0.5
        rendered = alpha > 0.5
        union = (rendered | mask).sum().clamp_min(1)
        row["heldout_silhouette_iou"] = float((rendered & mask).sum() / union)
    return rows


def evaluate(task: dict, run: Path, cells: list) -> None:
    import eval_jet_prior_tosca_cat0 as geometry_eval

    from rtgs.render.base import get_rasterizer
    from rtgs.visualize import save_reconstruction_artifacts

    for condition, seed in cells:
        receipt = read_json(run / "cells" / condition / str(seed) / "receipt.json")
        if receipt["status"] != "completed":
            raise RuntimeError("evaluation requires every final model to be saved")
    access = v1.access_guard(task, "evaluate")
    heldout = v1.load_views(task, task["splits"][DATASET]["heldout"])
    preview = SceneData(
        heldout.images,
        heldout.cameras,
        view_names=heldout.view_names,
        masks=None,
        train_indices=[],
        test_indices=list(range(len(heldout.images))),
        bounds_hint=heldout.bounds_hint,
    )
    perceptual = v1.lpips_model()
    renderer = get_rasterizer(RASTERIZER, device="cuda:0")
    for condition, seed in cells:
        started = run_seconds(run)
        output = run / "cells" / condition / str(seed)
        receipt = read_json(output / "receipt.json")
        final = Gaussians3D.load_npz(output / "gaussians.npz").to("cuda:0")
        initial = Gaussians3D.load_npz(output / "gaussians_init.npz").to("cuda:0")
        rows = appearance(renderer, final, heldout, perceptual)
        exported = geometry_eval.evaluate_model(final, output)  # splat-frame PLY + ZL spectrum
        geometry = mesh_geometry(
            final, torch.load(output / "h0.pt"), task["coverage"]["mesh_points"], seed
        )
        geometry.update(
            zl_err_k10=exported["zl_err_k10"],
            zl_lambda=exported["zl_lambda"],
            splat_frame_ply=exported["splat_frame_ply"],
            collapse_fraction=receipt["collapse_fraction"],
            collapse_fraction_at_activation=receipt["collapse_fraction_at_activation"],
        )
        artifacts = save_reconstruction_artifacts(
            preview,
            initial,
            final,
            output,
            rasterizer=RASTERIZER,
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
        summary = {**{key: round(geometry[key], 6) for key in GEOMETRY}, **means}
        print(f"evaluated {condition}/{seed}: {json.dumps(summary)}", flush=True)


def teacher(task: dict, run: Path) -> None:
    """Go/no-go: on the b2/9561 model at step 7500, trained nu_i vs the field normal n_i of the
    prior exactly as implemented (same h0, pairs, weights, guard), both against the mesh."""
    import trimesh

    from rtgs.optim.jet_prior import field_estimate, radius_pairs

    v1.access_guard(task, "evaluate")
    condition, seed = TEACHER_CELLS[0]
    output = run / "cells" / condition / str(seed)
    model = Gaussians3D.load_npz(output / "gaussians.npz")
    frozen = torch.load(output / "h0.pt")
    spec = task["field_prior_configs"]["jet"]
    h0 = frozen["h0"]
    field = field_estimate(
        model.means,
        model.quats,
        model.log_scales,
        model.opacity,
        h0,
        radius_pairs(model.means, spec["radius"] * h0),
        spec["bandwidth"],
        spec["min_gap"],
        spec["min_support"],
    )
    mesh = trimesh.load(v1.ROOT / "dataset/external/tosca_cat0/source/cat0.obj", process=False)
    _, _, face = trimesh.proximity.closest_point(mesh, model.means.double().numpy())
    truth = mesh.face_normals[face]

    def angles(vectors):
        cosine = np.abs((vectors.double().numpy() * truth).sum(1)).clip(0, 1)
        return np.degrees(np.arccos(cosine))

    trained, teacher_angle = angles(_normals(model)), angles(field["normal"])
    ok = field["ok"].numpy()
    opaque = model.opacity.numpy() > 0.3
    table = {}
    for label, rows in (
        ("opacity>0.3, unguarded (where the prior acts)", opaque & ok),
        ("opacity>0.3, all", opaque),
        ("all splats, unguarded", ok),
    ):
        table[label] = {
            "count": int(rows.sum()),
            "trained_nu_median": float(np.median(trained[rows])),
            "trained_nu_p90": float(np.percentile(trained[rows], 90)),
            "field_n_median": float(np.median(teacher_angle[rows])),
            "field_n_p90": float(np.percentile(teacher_angle[rows], 90)),
        }
    main = table["opacity>0.3, unguarded (where the prior acts)"]
    go = (
        main["field_n_median"] < main["trained_nu_median"] - 3.0
        and main["field_n_p90"] <= main["trained_nu_p90"]
    )
    result = {
        "step": task["collapse_gate"]["h0_step"],
        "n_total": model.n,
        "guarded_fraction_all": float(1 - ok.mean()),
        "guarded_fraction_opacity_gt_0_3": float(1 - ok[opaque].mean()),
        "table": table,
        "rule": "go iff median(field n) < median(trained nu) - 3 deg and p90(field n) <= "
        "p90(trained nu), on opacity > 0.3 splats the guard lets through",
        "go": go,
    }
    write_json(run / "teacher_check.json", result)
    print(json.dumps(result, indent=1))


def companion_busy() -> bool:
    return subprocess.run(["pgrep", "-f", "splat_lbo_cat"], capture_output=True).returncode == 0


GATE_LINE = re.compile(r"^\s+(G15\w*|SP)\s+(PASS|FAIL)\s+(.*)$")


def operator(task: dict, run: Path) -> None:
    """Deferred sequential CPU stage: the SplatDiffuseLBO operator (v2, case a) per model.

    The spectrum counts only if the companion's structural diagnostics pass: its G15a gate
    lines (K.1 = 0, M > 0, zero modes = components, RSS), one connected component, cond(M) and
    no localized modes among k <= 30 (min n90 above the threshold)."""
    stage = task["operator_stage"]
    reference = np.load(COMPANION / "data/cat_reference.npz")["lam_p2"]
    (run / "logs").mkdir(exist_ok=True)
    budget = stage["budget_minutes"] * 60.0
    for condition, seed in stage["cells"]:
        output = run / "cells" / condition / str(seed)
        if (output / "operator.json").exists():
            budget -= read_json(output / "operator.json")["wall_seconds"]
            continue
        while companion_busy():
            print("waiting for another companion job to finish", flush=True)
            time.sleep(180)
        guard_minutes = min(stage["guard_minutes"], budget / 60.0)
        tag = f"{task['task_id']}/{condition}_{seed}"
        ply = output / "gaussians_splat_frame.ply"
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "CAT_PLY": str(ply),
            "CAT_TAG": tag,
            "CAT_GUARD_MIN": f"{guard_minutes:.1f}",
        }
        command = [str(COMPANION_PYTHON), "splat_lbo_cat.py", "v2", "a"]
        log = run / "logs" / f"operator_{condition}_{seed}.log"
        started = time.time()
        record = {"condition": condition, "seed": seed, "command": command, "log": str(log)}
        record["env"] = {k: v for k, v in env.items() if k.startswith("CAT_")}
        record["ply_sha256"] = sha256(ply)
        if guard_minutes <= 0:
            record.update(exit_code=None, completed=False, reason="CPU budget exhausted")
        else:
            with log.open("w") as stream:
                record["exit_code"] = subprocess.run(
                    command,
                    cwd=COMPANION,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=guard_minutes * 60 + 1800,
                ).returncode  # the companion exits 1 when any of its own gates fails
        record["wall_seconds"] = time.time() - started
        budget -= record["wall_seconds"]
        gates = [
            m.groups() for line in log.read_text().splitlines() if (m := GATE_LINE.match(line))
        ]
        record["gate_lines"] = [{"gate": g, "status": s, "text": t} for g, s, t in gates]
        npz = COMPANION / "results" / tag / "cat_v2_subset.npz"
        record.update(completed=npz.exists(), lbo_err_k10=None, structural_ok=False)
        if npz.exists():
            data = np.load(npz)
            lam = data["lam"]
            err = lam[1:31] / reference[1:31] - 1
            structure = {
                "g15a_all_pass": bool(gates)
                and all(s == "PASS" for g, s, _ in gates if g == "G15a"),
                "components": int(data["ncomp"]),
                "zero_modes": int(data["nzero"]),
                "cond_M": float(data["cond"]),
                "min_n90_k1_30": int(data["n90"][1:31].min()),
                "reduced_solve": bool(data["reduced"]),
            }
            limits = stage["structural_limits"]
            structure["ok"] = (
                structure["g15a_all_pass"]
                and structure["components"] == 1
                and structure["zero_modes"] == 1
                and structure["cond_M"] < limits["cond_M_max"]
                and structure["min_n90_k1_30"] > limits["min_n90"]
            )
            raw = float(np.abs(err[:10]).max())
            record.update(
                npz=str(npz),
                npz_sha256=sha256(npz),
                lam=lam.tolist(),
                err=err.tolist(),
                structure=structure,
                structural_ok=structure["ok"],
                lbo_err_k10_raw=raw,
                lbo_err_k10=raw if structure["ok"] else None,
            )
        write_json(output / "operator.json", record)
        print(
            f"operator {condition}/{seed}: {record.get('lbo_err_k10')} ok={record['structural_ok']}"
        )


# --------------------------------------------------------------------------- decision and report


def gates(task: dict, cells: dict) -> dict:
    """Frozen PREREG v2.1 rule: jet vs b2 per seed; any-seed collapse gate; rest descriptive."""
    rule = task["decision_rule"]
    seeds = task["seeds"]

    def compare(arm, control, seed):
        a, b = cells[(arm, seed)], cells[(control, seed)]
        row = {
            "seed": seed,
            "arm": arm,
            "control": control,
            "normal_median_gain_deg": b["normal_angle_median"] - a["normal_angle_median"],
            "normal_p90_gain_deg": b["normal_angle_p90"] - a["normal_angle_p90"],
            "centre_median_delta": a["centre_distance_median"] - b["centre_distance_median"],
            "coverage_delta": a["coverage"] - b["coverage"],
            "collapse_excess": a["collapse_fraction"] - b["collapse_fraction"],
            "psnr_delta_db": a["heldout_psnr"] - b["heldout_psnr"],
        }
        row["collapse_gate_pass"] = row["collapse_excess"] <= rule["collapse_margin"]
        return row

    rows = [compare("jet", "b2", seed) for seed in seeds]
    for row in rows:
        row["pass"] = (
            row["normal_median_gain_deg"] >= rule["normal_gain_deg"]
            and row["normal_p90_gain_deg"] > 0
            and row["centre_median_delta"] <= 0
            and row["coverage_delta"] >= 0
            and row["collapse_gate_pass"]
            and row["psnr_delta_db"] >= rule["psnr_floor_db"]
        )
    collapse_failed_any = not all(r["collapse_gate_pass"] for r in rows)
    if all(r["pass"] for r in rows):
        verdict = "pass"
    elif all(r["normal_median_gain_deg"] < 0 for r in rows) or not any(
        r["collapse_gate_pass"] for r in rows
    ):
        verdict = "reject"
    else:
        verdict = "inconclusive"
    ablation = [compare(c, "b2", s) for c, s in task["execution_order"]["cells"] if c == "jet_c"]
    descriptive = {
        "anchor_alone_b2_noanchor_vs_b2": [compare("b2_noanchor", "b2", s) for s in seeds],
        "anchor_needed_jet_noanchor_vs_jet": [compare("jet_noanchor", "jet", s) for s in seeds],
        "jet_noanchor_vs_b2_noanchor": [compare("jet_noanchor", "b2_noanchor", s) for s in seeds],
        "centre_term_ablation_jet_c_vs_b2": ablation,
    }
    return {
        "verdict": verdict,
        "collapse_gate_failed_any_seed": collapse_failed_any,
        "jet_vs_b2": rows,
        "descriptive": descriptive,
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
        values = dict(cell["evaluation"]["mean"])
        operator_path = directory / "operator.json"
        values["lbo_err_k10"] = (
            read_json(operator_path)["lbo_err_k10"] if operator_path.exists() else None
        )
        table[(condition, seed)] = values
        cells.append(cell)
    keys = list(GEOMETRY + APPEARANCE)
    groups = {
        condition: {
            key: mean(table[(c, s)][key] for c, s in table if c == condition) for key in keys
        }
        for condition in CONDITIONS
    }
    operator_values = {c: v["lbo_err_k10"] for (c, s), v in table.items() if s == 9561}
    result_gates = gates(task, table)
    decision = (
        f"PREREG v2.1 decision (jet vs b2, every seed; collapse gate any-seed): "
        f"{result_gates['verdict']}. Anchor questions and the centre-term ablation descriptive. "
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
            "operator_seed_9561": operator_values,
            "gates": result_gates,
            "cells": [
                {
                    "condition": c["condition"],
                    "seed": c["seed"],
                    "metrics": table[(c["condition"], c["seed"])],
                    "geometry": c["evaluation"]["geometry"],
                    "receipt": c["receipt"],
                    "diagnostics": c["history"]["diagnostics"],
                }
                for c in cells
            ],
        },
    )
    write_json(run / "training_history.json", v1.history(task, cells))
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
    v1._receipt(task, run)
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
        if spec["id"] == "lbo_err_k10":
            value = operator_values.get("jet")
            if value is None:
                continue
            label = "jet seed 9561"
        else:
            value, label = groups["jet"][spec["id"]], "jet (3-seed mean)"
        metric_values[spec["id"]] = value
        metadata[spec["id"]] = {
            "label": f"{label}: {spec['label']}",
            "unit": spec["unit"],
            "group": "primary",
            "direction": spec["direction"],
        }
    summary = (
        f"{len(cells)} cells completed. Seed means, normal-angle median: "
        + ", ".join(f"{c} {groups[c]['normal_angle_median']:.2f} deg" for c in CONDITIONS)
        + "; coverage: "
        + ", ".join(f"{c} {groups[c]['coverage']:.3f}" for c in CONDITIONS)
        + "; collapse fraction: "
        + ", ".join(f"{c} {groups[c]['collapse_fraction']:.4f}" for c in CONDITIONS)
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
        names = ["receipt.json", "evaluation.json", "gaussians.ply", "gaussians_splat_frame.ply"]
        if (run / cell["path"] / "operator.json").exists():
            names.append("operator.json")
        for name in (*names, *PREVIEWS):
            artifacts.append({"label": f"{cell['path']}/{name}", "path": f"{cell['path']}/{name}"})
    charts = [
        {
            "id": "quality",
            "title": "Normal angle to the mesh, median (seed mean)",
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
        "Geometry in fixed units (bbox-diagonal fraction of cat0.obj); opacity > 0.3 final subset "
        "and the frozen step-7500 cohort; the surfel normal is the third rotation axis.",
        "Collapse fraction: share of all splats with another splat closer than 0.05 h0_i, h0 "
        "frozen at step 7500 in every arm; reported at activation and at the end.",
        "Held-out PSNR/LPIPS inside the held-out mask, 2DGS render over white; silhouette IoU "
        "alpha > 0.5 vs mask.",
        "lbo_err_k10: operator stage, b2/jet seed 9561; counts only with structural_ok.",
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
            "commands": v1._commands(task, run),
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
        "operator_seed_9561": operator_values,
        "gates": result_gates,
        "cells": {f"{c}/{s}": v for (c, s), v in table.items()},
        "command": task["run_command"],
        "source_lock": read_json(run / "task.lock.json"),
        "raw_comparison_path": str(run.relative_to(ROOT) / "comparison.json"),
    }
    write_json(ROOT / f"{stem}_RESULT.json", result)
    return result


# --------------------------------------------------------------------------- orchestration


def coordinate(task_path: Path, task: dict, run: Path) -> None:
    logs = run / "logs"
    logs.mkdir(exist_ok=True)

    def worker(phase, condition=None, seed=None):
        command = [sys.executable, str(Path(__file__).resolve()), phase]
        command += ["--task", str(task_path), "--run-dir", str(run)]
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
                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=FIT_TIMEOUT if phase == "fit" else None,
                check=True,
            )

    try:
        v1.snapshot_source(task, run)
        v1.write_environment(run)
        v1.source_guard(task_path, task, run)
        write_json(run / "input_integrity_entry.json", v1.data_guard(task))
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
        v1.source_guard(task_path, task, run)
        split_digest = hashlib.sha256(
            json.dumps(task["splits"], sort_keys=True).encode()
        ).hexdigest()
        write_json(
            run / "input_integrity_exit.json", {**v1.data_guard(task), "split_sha256": split_digest}
        )
        publish(task, run)
    except BaseException as error:
        write_json(
            run / "execution_failure.json",
            {"error": str(error), "traceback": traceback.format_exc()},
        )
        v1.publish_failure(task, run, str(error))
        raise


def initialize(task: dict, run: Path) -> None:
    """v1's seeded initialization, byte-identical (checked against the v1 run when present)."""
    v1.initialize(task, run)
    reference = ROOT / "runs" / task["initialization"]["reused_from"] / "initialization.json"
    if reference.exists():
        ours = read_json(run / "initialization.json")["models"]
        theirs = read_json(reference)["models"]
        for seed in map(str, task["seeds"]):
            if ours[seed]["sha256"] != theirs[seed]["sha256"]:
                raise RuntimeError(f"initialization of seed {seed} differs from v1")


def preflight(task_path: Path, run: Path, args: list[str], cells: list, final: list) -> None:
    """Pre-flight pipeline in a scratch directory, one process per phase (the access guards are
    per-process audit hooks); not results-bearing, no task lock."""
    if run.exists():
        raise ValueError(f"{run} exists; pre-flight runs never overwrite")
    run.mkdir(parents=True)
    write_json(
        run / "task.lock.json",
        {"preflight": True, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
    )
    phases = [["prepare"], ["initialize"]]
    phases += [["fit", "--condition", c, "--seed", str(s)] for c, s in cells]
    for phase in [*phases, final]:
        command = [sys.executable, str(Path(__file__).resolve()), *phase]
        command += ["--task", str(task_path), "--run-dir", str(run), *args]
        print(" ".join(phase), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    global SMOKE
    parser = argparse.ArgumentParser(description=__doc__)
    commands = ("run", "prepare", "initialize", "fit", "evaluate", "publish", "operator")
    commands += ("smoke", "teacher", "teacher-eval")
    parser.add_argument("command", choices=commands)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", type=int, default=None, help="smoke iterations (.scratch/)")
    parser.add_argument("--teacher", action="store_true", help="teacher pre-flight (.scratch/)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    task_path, run = args.task.resolve(), args.run_dir.resolve()
    task = read_json(task_path)
    check_protocol_tables(task)
    if args.command == "teacher":
        args.teacher = True
    allowed = {"prepare", "initialize", "fit", "evaluate", "publish", "operator"}
    if args.teacher:
        if not run.is_relative_to(ROOT / ".scratch"):
            raise ValueError("pre-flight runs write below .scratch/ only")
        SMOKE = {"mode": "teacher"}
        if args.command == "teacher":
            preflight(task_path, run, ["--teacher"], TEACHER_CELLS, ["teacher-eval"])
            return
        allowed = {"prepare", "initialize", "fit", "teacher-eval"}
    elif args.smoke is not None:
        if not run.is_relative_to(ROOT / ".scratch"):
            raise ValueError("pre-flight runs write below .scratch/ only")
        SMOKE = {"mode": "smoke", "iterations": args.smoke}
        if args.command == "smoke":
            preflight(task_path, run, ["--smoke", str(args.smoke)], SMOKE_CELLS, ["evaluate"])
            return
        allowed = {"prepare", "initialize", "fit", "evaluate"}
    elif run != ROOT / "runs" / task["task_id"]:
        raise ValueError("only the canonical task run root is allowed")
    elif args.command == "run":
        coordinate(task_path, task, run)
        return
    else:
        v1.source_guard(task_path, task, run)
    if args.command not in allowed:
        parser.error(f"{args.command} is not available in this mode")
    if args.command == "prepare":
        v1.prepare(task, run)
    elif args.command == "initialize":
        initialize(task, run)
    elif args.command == "fit":
        if [args.condition, args.seed] not in task["execution_order"]["cells"]:
            parser.error("fit cell is not registered")
        fit(task, run, args.condition, args.seed)
    elif args.command == "evaluate":
        evaluate(task, run, SMOKE_CELLS if SMOKE else task["execution_order"]["cells"])
    elif args.command == "teacher-eval":
        teacher(task, run)
    elif args.command == "publish":
        publish(task, run)
    elif args.command == "operator":
        operator(task, run)


if __name__ == "__main__":
    main()
