"""v3 2-jet prior on analytic surfaces: thin driver over the v2.1 driver (one process per phase).

Plan: docs/TASK_jet2_synthetic_sphere_ellipsoid.md. Tasks
``experiments/tasks/20261008_jet2_prior_2dgs_analytic_{sphere,ellipsoid}.json`` share this file.

    .venv/bin/python scripts/experiments/20261008_jet2_prior_2dgs_analytic.py smoke --smoke 600 \
        --task experiments/tasks/<task>.json --run-dir .scratch/<task>/smoke

Everything not cat-specific is the v2.1 driver unchanged (``fit``, guards, seeded
initialization, receipts, pre-flight orchestration). Replaced here: the arms (base / jet1 /
jet2 with ``rtgs.optim.jet2_prior.Jet2Prior``), geometry against the analytic surface
``sum_k x_k^2/a_k^2 = 1`` (closest point, outward normal, Weingarten map), evaluation and the
teacher check with the prior's own estimator. The surface axes come from the task's dataset
record; ``source/reference.json`` (spectra) is evaluation-only, as the access guard enforces.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
v21 = importlib.import_module("20261008_jet_field_prior_2dgs_tosca_cat0")
v1 = v21.v1
v21.__file__ = __file__  # pre-flight / coordinate spawn this driver, not the cat driver

from rtgs.core.gaussians3d import Gaussians3D  # noqa: E402
from rtgs.data.scene import SceneData  # noqa: E402
from rtgs.optim.jet2_prior import Jet2Prior, exact_radius_pairs, jet2_residuals  # noqa: E402
from rtgs.optim.jet_prior import mean_neighbour_distance  # noqa: E402

CONDITIONS = ("base", "jet1", "jet2")
GEOMETRY = (
    "normal_angle_median",
    "normal_angle_p90",
    "centre_distance_median",
    "signed_distance_mean",
    "signed_distance_median",
    "inward_fraction",
    "coverage",
    "cohort_normal_angle_median",
    "cohort_normal_fraction_gt_10deg",
    "cohort_signed_distance_mean",
    "curvature_rel_error_median",
    "curvature_trace_ratio_median",
    "estimator_masked_fraction",
    "collapse_fraction",
    "collapse_fraction_at_activation",
    "n_subset",
    "n_cohort",
)


# --------------------------------------------------------------------------- analytic surface


def closest_point(points: np.ndarray, axes) -> np.ndarray:
    """Closest point on ``sum x_k^2/a_k^2 = 1``.

    Regular branch: ``x_k = a_k^2 p_k/(a_k^2 + t)`` with the root ``t > -c^2`` (``c`` the shortest
    axis) of ``f(t) = sum a_k^2 p_k^2/(a_k^2 + t)^2 - 1``, by bisection. Interior points near the
    plane ``p_c = 0`` have no such root (or one lost to cancellation); there the minimiser is the
    singular branch ``t = -c^2``: ``x_j = a_j^2 p_j/(a_j^2 - c^2)`` for the other axes and
    ``x_c = sign(p_c) c sqrt(1 - sum_j x_j^2/a_j^2)``. Each row takes the closer valid candidate.
    """
    a = np.asarray(axes, dtype=np.float64)
    a2 = a**2
    p = np.asarray(points, dtype=np.float64)
    if (np.linalg.norm(p, axis=1) < 1e-9).any():
        raise ValueError("closest point undefined at the centre")
    lo = np.full(len(p), -a2.min() * (1 - 1e-15))
    hi = np.linalg.norm(p, axis=1) * np.sqrt(a2.max()) + a2.max()
    for _ in range(200):
        t = 0.5 * (lo + hi)
        f = (a2 * p**2 / (a2 + t[:, None]) ** 2).sum(1) - 1
        lo, hi = np.where(f > 0, t, lo), np.where(f > 0, hi, t)
    x = a2 * p / (a2 + 0.5 * (lo + hi)[:, None])
    bad = np.abs((x**2 / a2).sum(1) - 1) > 1e-9
    k = int(np.argmin(a))
    others = np.arange(3) != k
    if bad.any() and np.ptp(a) > 0 and (a[others] > a[k]).all():
        q = p[bad]
        y = np.zeros_like(q)
        y[:, others] = a2[others] * q[:, others] / (a2[others] - a2[k])
        rest = 1 - (y[:, others] ** 2 / a2[others]).sum(1)
        valid = rest >= 0
        y[:, k] = np.where(q[:, k] < 0, -1.0, 1.0) * a[k] * np.sqrt(np.clip(rest, 0, None))
        regular = x[bad]
        regular_ok = np.abs((regular**2 / a2).sum(1) - 1) <= 1e-9
        closer = np.linalg.norm(q - y, axis=1) <= np.linalg.norm(q - regular, axis=1)
        take = valid & (~regular_ok | closer)
        rows = np.flatnonzero(bad)[take]
        x[rows] = y[take]
    residual = np.abs((x**2 / a2).sum(1) - 1).max()
    if residual > 1e-9:
        raise RuntimeError(f"closest-point solve did not converge ({residual})")
    return x


def surface_frame(x: np.ndarray, axes) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normal and Weingarten map ``P Hess(F) P / |grad F|`` (3x3) at surface points."""
    a2 = np.asarray(axes, dtype=np.float64) ** 2
    grad = 2 * x / a2
    norm = np.linalg.norm(grad, axis=1)
    n = grad / norm[:, None]
    proj = np.eye(3) - n[:, :, None] * n[:, None, :]
    weingarten = proj @ np.diag(2 / a2) @ proj / norm[:, None, None]
    return n, weingarten


def surface_samples(axes, count: int, seed: int) -> np.ndarray:
    """Exact area-uniform samples of ``sum x_k^2/a_k^2 = 1``: ``x = a * u`` with ``u`` uniform on
    the unit sphere has area element ``prod(a) |u/a| dOmega``, so ``u`` is accepted with
    probability ``|u/a| min(a)`` (rejection sampling; no mesh, no extra dependency)."""
    a = np.asarray(axes, dtype=np.float64)
    rng = np.random.default_rng(seed)
    kept = []
    while sum(len(k) for k in kept) < count:
        u = rng.standard_normal((2 * count, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        accept = rng.uniform(size=len(u)) < np.linalg.norm(u / a, axis=1) * a.min()
        kept.append(a * u[accept])
    return np.concatenate(kept)[:count]


def surface_area(axes, samples: int = 1_000_000) -> float:
    """``4 pi prod(a) E|u/a|`` over a fixed uniform sample (relative error ~1e-4)."""
    a = np.asarray(axes, dtype=np.float64)
    u = np.random.default_rng(0).standard_normal((samples, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return float(4 * np.pi * a.prod() * np.linalg.norm(u / a, axis=1).mean())


def axes_of(task: dict):
    return task["datasets"][0]["surface_axes"]


# --------------------------------------------------------------------------- prior and arms


class DriverPrior(Jet2Prior):
    """Jet2Prior plus the attributes the v2.1 ``fit`` records (h0 at activation, log)."""

    def __init__(self, *args, log_every: int = 500, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.h0, self.log, self.refreshes, self.log_every = None, [], 0, int(log_every)

    def __call__(self, means, quats, log_scales, step, opacities=None):
        if step >= self.start and self.h0 is None:
            self.h0 = mean_neighbour_distance(means).clamp_min(1e-12)
        if step >= self.start and (step - self.start) % self.refresh_every == 0:
            self.refreshes += 1
        loss = super().__call__(means, quats, log_scales, step, opacities)
        if self.stats and (step - self.start) % self.log_every == 0:
            self.log.append(dict(self.stats))
        return loss


def check_protocol_tables(task: dict) -> None:
    if [item["id"] for item in task["comparators"]] != list(CONDITIONS):
        raise RuntimeError("driver conditions differ from frozen comparators")
    for key in ("resolved_training_configs", "surfel_regularization_configs", "jet2_prior_configs"):
        if set(task[key]) != set(CONDITIONS):
            raise RuntimeError(f"{key} must cover exactly the conditions")
    split = task["splits"][v21.DATASET]
    if set(split["train"]) & set(split["heldout"]):
        raise RuntimeError("train and held-out views overlap")
    for condition, seed in task["execution_order"]["cells"]:
        resolved = task["resolved_training_configs"][condition][str(seed)]
        if resolved["rasterizer"] != v21.RASTERIZER:
            raise RuntimeError("every cell must train 2DGS surfels")
        if str(seed) not in task["scene_scale"]["values"]:
            raise RuntimeError(f"missing frozen scene scale for seed {seed}")


def cell_setup(task: dict, condition: str, seed: int):
    """v2.1 ``cell_setup`` (same configs, same smoke/teacher schedules) with the Jet2Prior."""
    from dataclasses import asdict, replace

    from rtgs.optim.density import DensityConfig
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
    spec = dict(task["jet2_prior_configs"][condition])
    h0_step = task["collapse_gate"]["h0_step"]
    if v21.SMOKE is not None and v21.SMOKE["mode"] == "teacher":
        config = replace(config, iterations=h0_step, schedule_iterations=config.iterations)
    elif v21.SMOKE is not None and v21.SMOKE["mode"] == "smoke":  # shortened schedule
        h0_step = v21.SMOKE["iterations"] // 2
        config = replace(
            config,
            iterations=v21.SMOKE["iterations"],
            eval_every=100,
            sh_degree_interval=100,
            density=replace(config.density, start_iter=100, stop_iter=h0_step, every=50),
        )
        surfel = replace(surfel, distortion_start=100, normal_start=h0_step)
        spec.update(start=h0_step, log_every=100)
    if spec["start"] != h0_step or surfel.normal_start != h0_step:
        raise RuntimeError("the prior and the anchor must start where h0 is frozen")
    # v2.1's fit records task["field_prior_configs"][condition] as effective_config.field_prior:
    # give it the specification that actually executes (smoke overrides included).
    task["field_prior_configs"] = {**task["jet2_prior_configs"], condition: dict(spec)}
    weight = spec.pop("weight")
    return config, surfel, (DriverPrior(weight, **spec) if weight else None), h0_step


def _jet2_spec(task: dict) -> dict:
    spec = dict(task["jet2_prior_configs"]["jet2"])
    return {k: spec[k] for k in ("radius", "min_mass", "min_conditioning")}


# --------------------------------------------------------------------------- geometry


def curvature(model: Gaussians3D, task: dict, closest: np.ndarray) -> dict:
    """The prior's own estimator (order 2, frozen-sig rule at evaluation time) vs the analytic
    Weingarten map, compared through oriented principal curvatures (frame independent)."""
    spec = _jet2_spec(task)
    sig = model.log_scales.detach().max(1).values.exp().clamp_min(1e-12)
    _, _, info = jet2_residuals(
        model.means.detach(),
        model.quats.detach(),
        model.log_scales.detach(),
        model.opacity.detach(),
        sig,
        exact_radius_pairs(model.means, spec["radius"] * sig),
        order=2,
        min_mass=spec["min_mass"],
        min_conditioning=spec["min_conditioning"],
    )
    nu = v21._normals(model).double().numpy()
    n, weingarten = surface_frame(closest, axes_of(task))
    orient = np.sign((nu * n).sum(1))
    estimated = np.sort(orient[:, None] * np.linalg.eigvalsh(info["S"].double().numpy()), 1)
    tangent = np.linalg.eigvalsh(weingarten)  # [0 (normal), k1, k2] up to order
    truth = np.sort(tangent[:, 1:], 1)
    rel = np.linalg.norm(estimated - truth, axis=1) / np.linalg.norm(truth, axis=1)
    ratio = estimated.sum(1) / truth.sum(1)
    return {"ok": info["ok"].numpy(), "rel_error": rel, "trace_ratio": ratio}


def analytic_geometry(model: Gaussians3D, frozen: dict, task: dict, seed: int) -> dict:
    """Fixed absolute units (axes as given). Subset = final opacity > 0.3; cohort = the frozen
    step-7500 cohort. Signed distance > 0 outside. Coverage radius = 0.5 h0-bar (cohort)."""
    from scipy.spatial import cKDTree

    axes = axes_of(task)
    model = model.to("cpu")
    centres = model.means.detach().double().numpy()
    nu = v21._normals(model).double().numpy()
    closest = closest_point(centres, axes)
    n, _ = surface_frame(closest, axes)
    angle = np.degrees(np.arccos(np.abs((nu * n).sum(1)).clip(0, 1)))
    signed = ((centres - closest) * n).sum(1)
    subset = model.opacity.detach().numpy() > 0.3
    cohort = frozen["cohort"].numpy()
    hbar0 = float(frozen["h0"][frozen["cohort"]].double().mean()) if cohort.any() else 0.0
    samples = surface_samples(axes, task["coverage"]["mesh_points"], seed)
    near = (
        cKDTree(centres[subset]).query(samples, distance_upper_bound=0.5 * hbar0)[0]
        if subset.any()
        else np.full(len(samples), np.inf)
    )
    curv = curvature(model, task, closest)
    use = subset & curv["ok"]

    def stat(values, rows, fn):  # None (with the count beside it) instead of a crash on empty sets
        return float(fn(values[rows])) if rows.any() else None

    return {
        "n_total": model.n,
        "n_subset": int(subset.sum()),
        "n_cohort": int(cohort.sum()),
        "n_curvature": int(use.sum()),
        "normal_angle_median": stat(angle, subset, np.median),
        "normal_angle_p90": stat(angle, subset, lambda v: np.percentile(v, 90)),
        "centre_distance_median": stat(np.abs(signed), subset, np.median),
        "signed_distance_mean": stat(signed, subset, np.mean),
        "signed_distance_median": stat(signed, subset, np.median),
        "inward_fraction": stat(signed < 0, subset, np.mean),
        "coverage": float(np.isfinite(near).mean()) if subset.any() else 0.0,
        "coverage_radius": 0.5 * hbar0,
        "cohort_normal_angle_median": stat(angle, cohort, np.median),
        "cohort_normal_angle_p90": stat(angle, cohort, lambda v: np.percentile(v, 90)),
        "normal_fraction_gt_10deg": stat(angle > 10, subset, np.mean),
        "cohort_normal_fraction_gt_10deg": stat(angle > 10, cohort, np.mean),
        "cohort_signed_distance_mean": stat(signed, cohort, np.mean),
        "curvature_rel_error_median": stat(curv["rel_error"], use, np.median),
        "curvature_rel_error_p90": stat(curv["rel_error"], use, lambda v: np.percentile(v, 90)),
        "curvature_trace_ratio_median": stat(curv["trace_ratio"], use, np.median),
        "estimator_masked_fraction": stat(~curv["ok"], subset, np.mean),
        "flatness_max": v21.flatness_max(model),
    }


def evaluate(task: dict, run: Path, cells: list) -> None:
    from rtgs.render.base import get_rasterizer
    from rtgs.visualize import save_reconstruction_artifacts

    for condition, seed in cells:
        if v1.read_json(run / "cells" / condition / str(seed) / "receipt.json")["status"] != (
            "completed"
        ):
            raise RuntimeError("evaluation requires every final model to be saved")
    access = v1.access_guard(task, "evaluate")
    heldout = v1.load_views(task, task["splits"][v21.DATASET]["heldout"])
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
    renderer = get_rasterizer(v21.RASTERIZER, device="cuda:0")
    for condition, seed in cells:
        started = v1.run_seconds(run)
        output = run / "cells" / condition / str(seed)
        receipt = v1.read_json(output / "receipt.json")
        final = Gaussians3D.load_npz(output / "gaussians.npz").to("cuda:0")
        initial = Gaussians3D.load_npz(output / "gaussians_init.npz").to("cuda:0")
        rows = v21.appearance(renderer, final, heldout, perceptual)
        geometry = analytic_geometry(final, torch.load(output / "h0.pt"), task, seed)
        geometry.update(
            collapse_fraction=receipt["collapse_fraction"],
            collapse_fraction_at_activation=receipt["collapse_fraction_at_activation"],
        )
        artifacts = save_reconstruction_artifacts(
            preview,
            initial,
            final,
            output,
            rasterizer=v21.RASTERIZER,
            max_comparisons=8,
            max_animation_frames=24,
        )
        means = {key: mean(row[key] for row in rows) for key in v21.APPEARANCE}
        v1.write_json(
            output / "evaluation.json",
            {
                "per_view": rows,
                "mean": {**means, **{key: geometry[key] for key in GEOMETRY}},
                "geometry": geometry,
                "artifacts": artifacts,
                "access_guard": access,
                "stage_intervals": {"evaluate": [started, v1.run_seconds(run)]},
            },
        )
        print(f"evaluated {condition}/{seed}", flush=True)  # values are read from the bundle


def teacher(task: dict, run: Path, cells: list) -> dict:
    """Base arm at the activation step, every teacher seed: the prior's estimator vs the analytic
    curvature (S = 0 scores 1), trained normals vs analytic normals, and the frozen go rule."""
    v1.access_guard(task, "evaluate")
    rule = task["pilot"]["teacher_go_rule"]
    rows = {}
    for condition, seed in cells:
        output = run / "cells" / condition / str(seed)
        model = Gaussians3D.load_npz(output / "gaussians.npz")
        receipt = v1.read_json(output / "receipt.json")
        geometry = analytic_geometry(model, torch.load(output / "h0.pt"), task, seed)
        all_masked = curvature_masked_all(model, task)
        checks = {
            "curvature_rel_error_median": _le(
                geometry["curvature_rel_error_median"], rule["curvature_rel_error_median_max"]
            ),
            "estimator_masked_fraction": _le(
                geometry["estimator_masked_fraction"], rule["estimator_masked_fraction_max"]
            ),
            "n_curvature": geometry["n_curvature"] >= rule["n_curvature_min"],
            "collapse_fraction_at_activation": _le(
                receipt["collapse_fraction_at_activation"],
                rule["collapse_fraction_at_activation_max"],
            ),
        }
        rows[f"{condition}/{seed}"] = {
            "geometry": geometry,
            "estimator_masked_fraction_all_splats": all_masked,
            "collapse_fraction_at_activation": receipt["collapse_fraction_at_activation"],
            "checks": checks,
            "go": all(checks.values()),
        }
    result = {
        "step": task["collapse_gate"]["h0_step"],
        "rule": rule,
        "rows": rows,
        "go": all(row["go"] for row in rows.values()),
    }
    v1.write_json(run / "teacher_check.json", result)
    return result


def _le(value, limit) -> bool:
    """Missing values never pass."""
    return value is not None and value <= limit


def curvature_masked_all(model: Gaussians3D, task: dict) -> float:
    """Gate rejection over all splats (the prior acts on all, not only on opacity > 0.3)."""
    closest = closest_point(model.means.detach().double().numpy(), axes_of(task))
    return float(1 - curvature(model, task, closest)["ok"].mean())


# --------------------------------------------------------------------------- operator


def ideal_surfels(
    axes,
    count: int,
    seed: int,
    tilt_deg: float = 0.0,
    shift: float = 0.0,
    size_spread: float = 0.0,
) -> Gaussians3D:
    """Surfels at area-uniform surface samples, frame from the analytic normal, tangent sigma
    0.75 sqrt(A/N) (A from ``surface_area``) times ``exp(size_spread z)``, z ~ N(0, 1) per
    surfel, opacity 0.9; optionally every normal tilted by ``tilt_deg`` about a random tangent
    axis and every centre moved by ``-shift`` along it."""
    import math

    from rtgs.core.gaussians3d import rotmat_to_quat

    x = surface_samples(axes, count, seed=seed)
    n, _ = surface_frame(x, axes)
    centres = x - shift * n

    def frame(normal):
        helper = np.where(np.abs(normal[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        t1 = np.cross(normal, helper)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        return t1, np.cross(normal, t1)

    rng = np.random.default_rng(seed + 1)
    t1, t2 = frame(n)
    if tilt_deg:
        phi = rng.uniform(0, 2 * np.pi, count)[:, None]
        axis = np.cos(phi) * t1 + np.sin(phi) * t2
        angle = math.radians(tilt_deg)
        n = math.cos(angle) * n + math.sin(angle) * np.cross(axis, n)
        t1, t2 = frame(n)
    sigma = 0.75 * math.sqrt(surface_area(axes) / count)
    log_sigma = math.log(sigma) + size_spread * rng.standard_normal(count)
    log_scales = np.stack([log_sigma, log_sigma, np.full(count, -np.inf)], 1)
    return Gaussians3D(
        torch.tensor(centres, dtype=torch.float32),
        rotmat_to_quat(torch.tensor(np.stack([t1, t2, n], 2), dtype=torch.float32)),
        torch.tensor(log_scales, dtype=torch.float32),
        torch.full((count,), 0.9),
        torch.zeros(count, 1, 3),
    )


COMPANION_ADAPTER = Path(__file__).resolve().parent / "analytic_operator_companion.py"
EXIT_STATUS = {3: "rss_watchdog"}


def spectrum(task: dict, ply: Path, out: Path, flat: bool = False) -> dict:
    """Companion operator (v2, or v2flat) on one PLY in a fresh process with a hard timeout,
    compared with ``source/reference.json``: relative errors of lambda_1..lambda_10, raw and
    area-normalised (lambda * area / area_ref). Every exit path yields a status."""
    import subprocess
    import time

    guard = task["pilot"]["operator_guard_minutes"]
    command = [str(v21.COMPANION_PYTHON), str(COMPANION_ADAPTER), str(ply)]
    command += [str(a) for a in axes_of(task)] + [str(guard), str(out)] + (["flat"] if flat else [])
    started = time.time()
    with out.with_suffix(".log").open("w") as stream:
        try:
            code = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, timeout=guard * 60 + 600
            ).returncode
        except subprocess.TimeoutExpired:
            code = None
    if out.exists():
        result = v1.read_json(out)
    else:
        status = (
            "killed_timeout" if code is None else EXIT_STATUS.get(code, f"no output (exit {code})")
        )
        result = {"status": status, "peak_rss_bytes": None}
    result.update(exit_code=code, caller_wall_s=time.time() - started)
    reference = v1.read_json(v1.frame(task) / "source/reference.json")

    def compare(block: dict) -> None:
        ref = np.asarray(reference["eigenvalues"][1:11])
        lam = np.asarray(block["lam"][1:11])
        scaled = lam * block["area"] / reference["area"]
        block["rel_error"] = (lam / ref - 1).tolist()
        block["rel_error_area_normalised"] = (scaled / ref - 1).tolist()
        block["max_abs_rel_error"] = float(np.abs(lam / ref - 1).max())
        block["max_abs_rel_error_area_normalised"] = float(np.abs(scaled / ref - 1).max())
        block["area_rel_error"] = block["area"] / reference["area"] - 1

    if result.get("status") == "valid":
        compare(result)
    if "zhou_laehner" in result:  # comparator, computed before the operator's own assembly
        compare(result["zhou_laehner"])
    v1.write_json(out, result)
    return result


def operator(task: dict, run: Path, cells: list) -> None:
    v1.access_guard(task, "evaluate")
    for condition, seed in cells:
        output = run / "cells" / condition / str(seed)
        result = spectrum(task, output / "gaussians.ply", output / "operator.json")
        print(f"operator {condition}/{seed}: {result['status']}", flush=True)


def calibrate(task: dict, run: Path) -> None:
    """Instrument calibration on ideal surfels (frozen rows in task['pilot']['calibration']).
    Coverage samples use their own seed (a shared seed puts sample points on surfel centres)."""
    v1.access_guard(task, "evaluate")
    spec = task["pilot"]["calibration"]
    axes = axes_of(task)
    rows = []
    for row in spec["rows"]:
        model = ideal_surfels(
            axes,
            row["count"],
            spec["seed"],
            row.get("tilt_deg", 0.0),
            size_spread=row.get("size_spread", 0.0),
        )
        directory = run / "calibration" / row["name"]
        directory.mkdir(parents=True)
        model.save_ply(directory / "gaussians.ply")
        frozen = {
            "h0": mean_neighbour_distance(model.means),
            "cohort": torch.ones(model.n, dtype=torch.bool),
        }
        geometry = analytic_geometry(model, frozen, task, spec["coverage_seed"])
        result = spectrum(
            task, directory / "gaussians.ply", directory / "operator.json", row.get("flat", False)
        )
        rows.append({**row, "geometry": geometry, "operator": result})
        print(f"calibration {row['name']}: {result['status']}", flush=True)
    by_name = {row["name"]: row for row in rows}
    rule = spec["instrument_rule"]
    exact, tilted = by_name[rule["exact_row"]]["operator"], by_name[rule["tilted_row"]]["operator"]
    passed = (
        exact.get("status") == "valid"
        and exact["max_abs_rel_error"] <= rule["exact_max_abs_rel_error_max"]
        and tilted.get("status") == "valid"
        and tilted["max_abs_rel_error"] >= rule["tilt_factor_min"] * exact["max_abs_rel_error"]
    )
    v1.write_json(run / "calibration.json", {"rows": rows, "rule": rule, "instrument_pass": passed})


# --------------------------------------------------------------------------- pilot orchestration


def surprises(task: dict, run: Path, cells: list, stage: str) -> list[str]:
    """Frozen surprise triggers (task['pilot']['surprise']); any trigger stops the plan."""
    rule = task["pilot"]["surprise"]
    found = []
    for condition, seed in cells:
        output = run / "cells" / condition / str(seed)
        if stage == "evaluate":
            mean_ = v1.read_json(output / "evaluation.json")["mean"]
            if not _le(mean_["normal_angle_median"], rule["normal_angle_median_max"]):
                found.append(f"{condition}/{seed}: normal median {mean_['normal_angle_median']}")
            if not mean_["heldout_psnr"] >= rule["heldout_psnr_min"]:
                found.append(f"{condition}/{seed}: base not learned (PSNR {mean_['heldout_psnr']})")
            if not mean_["heldout_silhouette_iou"] >= rule["heldout_silhouette_iou_min"]:
                found.append(
                    f"{condition}/{seed}: silhouette IoU {mean_['heldout_silhouette_iou']}"
                )
        else:
            result = v1.read_json(output / "operator.json")
            rss = result.get("peak_rss_bytes")
            if result["status"] == "invalid":
                found.append(f"{condition}/{seed}: operator structural gate G15a failed")
            if result["status"] in ("memory", "rss_watchdog") or (
                rss is not None and rss > rule["operator_rss_max_bytes"]
            ):
                found.append(f"{condition}/{seed}: operator memory {result['status']} rss {rss}")
            if result["status"] in ("error", "killed_timeout") or result["status"].startswith(
                "no "
            ):
                found.append(f"{condition}/{seed}: operator {result['status']}")
    v1.write_json(run / f"surprises_{stage}.json", {"stage": stage, "triggers": found})
    return found


def record(path: Path, task: dict) -> dict:
    """Start/end record: both commits and worktree states, data seals, GPU occupancy."""
    import subprocess

    def git(repo, *args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        ).stdout

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "repository_head": git(v21.ROOT, "rev-parse", "HEAD").strip(),
        "repository_dirty": git(v21.ROOT, "status", "--porcelain", "--untracked-files=no"),
        "companion_head": git(v21.COMPANION, "rev-parse", "HEAD").strip(),
        "companion_dirty": git(v21.COMPANION, "status", "--porcelain", "--untracked-files=no"),
        "data_guard": v1.data_guard(task),
        "gpu": gpu,
    }
    v1.write_json(path, payload)
    return payload


def pilot_main(task_path: Path, task: dict) -> None:
    """Pilot / calibration mode (P3), one surface per invocation, writes below .scratch/ only.

    ``pilot`` orchestrates, each phase a fresh process with a hard timeout:
    start record -> calibrate -> teacher (fits truncated at 7500, teacher-eval, go rule) ->
    pilot fits -> evaluate -> surprise check -> operator -> surprise check -> repeat fit ->
    evaluate -> check -> operator -> check -> end record. A trigger or a failed phase stops.
    """
    import argparse
    import datetime as dt
    import subprocess

    parser = argparse.ArgumentParser()
    commands = ("pilot", "prepare", "initialize", "fit", "evaluate", "operator", "calibrate")
    parser.add_argument("command", choices=(*commands, "teacher-eval"))
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--stage", choices=("teacher", "pilot", "repeat"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.run_dir.resolve()
    if not root.is_relative_to(v21.ROOT / ".scratch"):
        raise ValueError("pilot runs write below .scratch/ only")
    spec = task["pilot"]
    cells = {
        "teacher": spec["teacher_cells"],
        "pilot": spec["cells"],
        "repeat": spec["repeat_cells"],
    }
    v21.SMOKE = {"mode": "teacher" if args.stage == "teacher" else "pilot"}
    if args.command != "pilot":
        run = root
        stage_cells = cells[args.stage] if args.stage else []
        if args.command == "prepare":
            v1.prepare(task, run)
        elif args.command == "initialize":
            v21.initialize(task, run)
        elif args.command == "fit":
            if [args.condition, args.seed] not in stage_cells:
                parser.error("fit cell is not in the frozen cells of this stage")
            v21.fit(task, run, args.condition, args.seed)
        elif args.command == "evaluate":
            evaluate(task, run, stage_cells)
        elif args.command == "teacher-eval":
            teacher(task, run, stage_cells)
        elif args.command == "operator":
            operator(task, run, stage_cells)
        elif args.command == "calibrate":
            calibrate(task, run)
        return

    if root.exists():
        raise ValueError(f"{root} exists; pilot runs never overwrite")
    root.mkdir(parents=True)
    timeouts = spec["timeouts_seconds"]
    start = record(root / "start_record.json", task)
    if start["repository_dirty"] or start["companion_dirty"]:
        raise RuntimeError("pilot runs start from clean worktrees (tracked files)")

    def phase(run: Path, command: str, stage: str | None = None, cell=None) -> None:
        argv = [sys.executable, __file__, command, "--task", str(task_path), "--run-dir", str(run)]
        argv += ["--pilot"] + (["--stage", stage] if stage else [])
        if cell is not None:
            argv += ["--condition", cell[0], "--seed", str(cell[1])]
        print(f"{run.name}: {command} {cell or ''}", flush=True)
        subprocess.run(argv, cwd=v21.ROOT, check=True, timeout=timeouts[command])

    def stage_dir(name: str) -> Path:
        run = root / name
        run.mkdir()
        v1.write_json(
            run / "task.lock.json",
            {"preflight": True, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
        )
        return run

    def stop_if(found: list[str]) -> None:
        if found:
            v1.write_json(root / "stopped.json", {"triggers": found})
            raise SystemExit(f"surprise triggers, plan stopped: {found}")

    phase(stage_dir("calibration"), "calibrate")
    run = stage_dir("teacher")
    for command in ("prepare", "initialize"):
        phase(run, command, "teacher")
    for cell in cells["teacher"]:
        phase(run, "fit", "teacher", cell)
    phase(run, "teacher-eval", "teacher")
    for name in ("pilot", "repeat"):
        run = stage_dir(name)
        for command in ("prepare", "initialize"):
            phase(run, command, name)
        for cell in cells[name]:
            phase(run, "fit", name, cell)
        phase(run, "evaluate", name)
        stop_if(surprises(task, run, cells[name], "evaluate"))
        phase(run, "operator", name)
        stop_if(surprises(task, run, cells[name], "operator"))
    end = record(root / "end_record.json", task)
    if (end["repository_head"], end["companion_head"]) != (
        start["repository_head"],
        start["companion_head"],
    ):
        raise RuntimeError("a commit changed during the pilot")


# --------------------------------------------------------------------------- wiring


def main() -> None:
    task_path = Path(sys.argv[sys.argv.index("--task") + 1])
    task = v1.read_json(task_path)
    dataset = task["datasets"][0]["id"]
    v1.DATASET = v21.DATASET = dataset
    v21.CONDITIONS = CONDITIONS
    v21.GEOMETRY = GEOMETRY
    v21.SMOKE_CELLS = [["base", 9561], ["jet1", 9561], ["jet2", 9561]]
    v21.check_protocol_tables = check_protocol_tables
    v21.cell_setup = cell_setup
    v21.evaluate = evaluate
    v21.teacher = None  # the analytic teacher runs through the pilot mode
    v21.mesh_geometry = None  # cat-only; must never be reached
    v21.operator = None  # cat-only; the analytic spectrum runs through the pilot mode
    v21.publish = None  # official runs need the analytic publish step (P4)
    if "--pilot" in sys.argv:
        pilot_main(task_path, task)
    else:
        v21.main()


if __name__ == "__main__":
    main()
