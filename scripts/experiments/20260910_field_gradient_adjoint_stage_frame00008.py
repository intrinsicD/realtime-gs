"""Common-render adjoint differences, with explicit numerical-repeat evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OLD = load_module(
    Path(__file__).with_name("20260909_field_target_gradient_stage_frame00008.py"),
    "immutable_field_gradient_primitives",
)
ROOT = OLD.ROOT
GROUPS, COMPONENTS = OLD.GROUPS, OLD.COMPONENTS
BASE_COMPONENTS = ("l1", "dssim")
write_json, read_json = OLD.write_json, OLD.read_json


class PhaseGuard(OLD.InputGuard):
    def __init__(self, task, root=ROOT):
        super().__init__(task, root)
        probe = task["gradient_protocol"]["repeatability_probe"]
        state = next(item for item in task["states"] if item["id"] == probe["state"])
        self.probe_allowed = {
            (self.root / state["model"]).resolve(),
            self.source_run / "targets/metadata.json",
            self.source_run / "targets/rgb" / f"{probe['view']}.npz",
        }

    def audit(self, event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if (
                self.phase == "numerical_control"
                and path.is_relative_to(self.source_run)
                and path not in self.probe_allowed
            ):
                self.receipt["denied"].append({"path": str(path), "phase": self.phase})
                raise PermissionError("photo-only numerical gate forbids other cached inputs")
        return super().audit(event, args)


class StorageBudget:
    def __init__(self, run, task):
        self.run = run
        self.maximum = task["execution_budget"]["max_output_bytes"]
        self.preflight_minimum = task["execution_budget"]["min_free_disk_bytes"]

    def check(self, planned=0, preflight=False):
        used = sum(path.stat().st_size for path in self.run.rglob("*") if path.is_file())
        free = shutil.disk_usage(self.run).free
        minimum = self.preflight_minimum if preflight else 2_000_000_000
        if used + planned > self.maximum or free - planned < minimum:
            raise RuntimeError(
                f"frozen storage budget exceeded: output={used},free={free},planned={planned}"
            )
        return {"output_bytes": used, "free_bytes": free, "planned_bytes": planned}

    def save(self, path, arrays):
        # Uncompressed size plus ZIP headers conservatively bounds this next lossless artifact.
        planned = sum(value.nbytes for value in arrays.values()) + 1024 * len(arrays) + 4096
        self.check(planned)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        self.check()
        return {
            "array_path": str(path.relative_to(self.run)),
            "sha256": OLD.sha256(path),
            "bytes": path.stat().st_size,
        }


def source_guard(task_path, task, run):
    result = OLD.source_guard(task_path, task, run)
    paths = {item["path"] for item in read_json(run / "source_snapshot/manifest.json")["files"]}
    if Path(__file__).resolve().relative_to(ROOT).as_posix() not in paths:
        raise RuntimeError("new adjoint driver absent from frozen source manifest")
    return result


def render_graph(saved, state, view, renderer, device):
    parameters = OLD.fresh_parameters(saved, state["kind"], device)
    output = renderer.render(
        OLD.assemble(parameters), view["camera"].to(device), sh_degree=state["sh_degree"]
    ).color
    OLD.require_finite(output)
    return parameters, output


def image_adjoints(image, target):
    leaf = image.detach().clone().requires_grad_(True)
    reference = target.to(leaf)
    l1 = (leaf - reference).abs().mean()
    dssim = 1 - OLD.ssim(leaf, reference)
    (h_l1,) = torch.autograd.grad(l1, leaf, retain_graph=True)
    (h_dssim,) = torch.autograd.grad(dssim, leaf)
    adjoints = {"l1": h_l1.detach(), "dssim": h_dssim.detach()}
    adjoints["total"] = 0.8 * adjoints["l1"] + 0.2 * adjoints["dssim"]
    OLD.require_finite(l1, dssim, *adjoints.values())
    return {
        "losses": {
            "l1": float(l1.detach()),
            "dssim": float(dssim.detach()),
            "total": float((0.8 * l1 + 0.2 * dssim).detach()),
        },
        "adjoints": adjoints,
    }


def vjp(parameters, image, adjoint):
    raw = torch.autograd.grad(
        image,
        tuple(parameters[name] for name in GROUPS),
        grad_outputs=adjoint,
        retain_graph=True,
        allow_unused=True,
    )
    return {name: value.cpu() for name, value in OLD.mapped_gradients(raw, parameters).items()}


def tolerance_stats(a, b, atol=1e-8, rtol=1e-4):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    difference = (a - b).abs()
    tolerance = atol + rtol * torch.maximum(a.abs(), b.abs())
    return {
        **OLD.compare_gradients(a, b),
        "tolerance_violations": int((difference > tolerance).sum()),
        "max_tolerance_excess": float((difference - tolerance).clamp_min(0).max()),
    }


def zero_control(parameters, image, view, state_id):
    photo = image_adjoints(image, OLD.target_route(view, "rgb"))
    identity = image_adjoints(image, OLD.target_route(view, "field_high", identity=True))
    record = {
        "state_id": state_id,
        "view_id": view["view_id"],
        "passed": True,
        "losses": {},
        "adjoints": {},
        "comparisons": {},
        "null_vjp": {},
        "parameter_delta_atol": 1e-8,
        "image_atol": 1e-8,
        "image_rtol": 1e-4,
        "loss_atol": 1e-7,
        "common_render": True,
        "boundary": "null/routing control; not a nonzero-delta accuracy bound",
    }
    arrays = {"render": image.detach().cpu().numpy()}
    for component in COMPONENTS:
        left, right = photo["adjoints"][component], identity["adjoints"][component]
        record["adjoints"][component] = tolerance_stats(left, right)
        loss_delta = abs(photo["losses"][component] - identity["losses"][component])
        record["losses"][component] = {
            "photo_path": photo["losses"][component],
            "teacher_path": identity["losses"][component],
            "absolute_difference": loss_delta,
        }
        record["passed"] &= record["adjoints"][component]["tolerance_violations"] == 0
        record["passed"] &= loss_delta <= 1e-7
        propagated = vjp(parameters, image, right - left)
        null = vjp(parameters, image, torch.zeros_like(left))
        record["comparisons"][component], record["null_vjp"][component] = {}, {}
        for group in GROUPS:
            for category, tensor in (("comparisons", propagated[group]), ("null_vjp", null[group])):
                stats = tolerance_stats(torch.zeros_like(tensor), tensor, atol=1e-8, rtol=0)
                record[category][component][group] = stats
                record["passed"] &= stats["tolerance_violations"] == 0
            arrays[f"delta__{component}__{group}"] = propagated[group].numpy()
            arrays[f"null__{component}__{group}"] = null[group].numpy()
        arrays[f"photo_adjoint__{component}"] = left.cpu().numpy()
        arrays[f"identity_adjoint__{component}"] = right.cpu().numpy()
        arrays[f"delta_adjoint__{component}"] = (right - left).cpu().numpy()
    record["passed"] = bool(record["passed"])
    return record, arrays


def with_total(base):
    output = {
        component: {group: value.double() for group, value in base[component].items()}
        for component in BASE_COMPONENTS
    }
    output["total"] = {
        group: 0.8 * output["l1"][group] + 0.2 * output["dssim"][group] for group in GROUPS
    }
    return output


def repeat_measurement(parameters, image, view):
    adjoints = {
        target: image_adjoints(image, OLD.target_route(view, target))
        for target in ("rgb", "field_high")
    }
    repeated = {kind: [] for kind in ("photo", "delta")}
    for _ in range(2):
        for kind in repeated:
            gradients = {}
            for component in BASE_COMPONENTS:
                left = adjoints["rgb"]["adjoints"][component]
                h = (
                    left
                    if kind == "photo"
                    else adjoints["field_high"]["adjoints"][component] - left
                )
                gradients[component] = vjp(parameters, image, h)
            repeated[kind].append(gradients)
    summary = summarize_repeats(repeated)
    return {
        "losses": {target: value["losses"] for target, value in adjoints.items()},
        "raw": repeated,
        **summary,
    }


def summarize_repeats(repeated):
    full = {kind: [with_total(repeat) for repeat in rows] for kind, rows in repeated.items()}
    means = {
        kind: {
            component: {
                group: (full[kind][0][component][group] + full[kind][1][component][group]) / 2
                for group in GROUPS
            }
            for component in COMPONENTS
        }
        for kind in full
    }
    derived = {
        component: {
            group: means["photo"][component][group] + means["delta"][component][group]
            for group in GROUPS
        }
        for component in COMPONENTS
    }
    comparisons, reliability = {}, {}
    for component in COMPONENTS:
        comparisons[component], reliability[component] = {}, {}
        for group in GROUPS:
            photo, delta = means["photo"][component][group], means["delta"][component][group]
            field = derived[component][group]
            dnorm = float(torch.linalg.vector_norm(delta))
            stats = OLD.compare_gradients(photo, field)
            stats.update(
                direct_delta_norm=dnorm, direct_delta_max_absolute=float(delta.abs().max())
            )
            comparisons[component][group] = stats
            pnoise = float(
                torch.linalg.vector_norm(
                    full["photo"][0][component][group] - full["photo"][1][component][group]
                )
            )
            dnoise = float(
                torch.linalg.vector_norm(
                    full["delta"][0][component][group] - full["delta"][1][component][group]
                )
            )
            reliability[component][group] = {
                "photo_repeat_difference_norm": pnoise,
                "delta_repeat_difference_norm": dnoise,
                "noise_proxy": pnoise + dnoise,
                "effect_resolved_against_observed_repeats": dnorm > 10 * (pnoise + dnoise),
                "reference_resolved": stats["photo_norm"] > 10 * pnoise,
                "field_resolved": stats["field_norm"] > 10 * (pnoise + dnoise),
            }
    return {
        "means": {"rgb": means["photo"], "field_high": derived, "delta": means["delta"]},
        "comparisons": comparisons,
        "reliability": reliability,
    }


def raw_repeat_arrays(repeated):
    return {
        f"{kind}__r{index}__{component}__{group}": gradient.numpy()
        for kind, repeats in repeated.items()
        for index, repeat in enumerate(repeats)
        for component, groups in repeat.items()
        for group, gradient in groups.items()
    }


def repeat_arrays(image=None, adjoints=None, gradients=None):
    arrays = {} if image is None else {"render": image.detach().cpu().numpy()}
    if adjoints is not None:
        arrays.update(
            {
                f"adjoint__{component}": value.detach().cpu().numpy()
                for component, value in adjoints.items()
            }
        )
    if gradients is not None:
        arrays.update(
            {
                f"parameter__{component}__{group}": value.numpy()
                for component, groups in gradients.items()
                for group, value in groups.items()
            }
        )
    return arrays


def pairwise(records, run, *, parameter_gate=False):
    summaries = []
    passed = True
    for i, j in itertools.combinations(range(len(records)), 2):
        with np.load(run / records[i]["array_path"], allow_pickle=False) as archive:
            left = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
        with np.load(run / records[j]["array_path"], allow_pickle=False) as archive:
            right = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
        row = {
            "a": i,
            "b": j,
            "consecutive": j == i + 1,
            "loss_differences": {},
            "adjoints": {},
            "parameters": {},
        }
        if "render" in left:
            row["render_max_abs"] = float((left["render"] - right["render"]).abs().max())
            passed &= row["render_max_abs"] <= 1e-7
        if "losses" in records[i]:
            row["loss_differences"] = {
                c: abs(records[i]["losses"][c] - records[j]["losses"][c]) for c in COMPONENTS
            }
            passed &= all(delta <= 1e-7 for delta in row["loss_differences"].values())
        for component in COMPONENTS:
            key = f"adjoint__{component}"
            if key in left:
                row["adjoints"][component] = tolerance_stats(left[key], right[key])
                passed &= row["adjoints"][component]["tolerance_violations"] == 0
            if f"parameter__{component}__means" in left:
                row["parameters"][component] = {}
                for group in GROUPS:
                    key = f"parameter__{component}__{group}"
                    stats = tolerance_stats(left[key], right[key])
                    stats["old_threshold_violations"] = stats["tolerance_violations"]
                    row["parameters"][component][group] = stats
                    if parameter_gate:
                        passed &= stats["tolerance_violations"] == 0
        summaries.append(row)
    return summaries, bool(passed)


def load_probe(task):
    source = ROOT / task["source_run"] / "targets"
    metadata = read_json(source / "metadata.json")
    view_id = task["gradient_protocol"]["repeatability_probe"]["view"]
    if metadata["view_ids"] != task["splits"]["frame_00008"]["train"]:
        raise RuntimeError("probe camera metadata differs from frozen training split")
    index = metadata["view_ids"].index(view_id)
    with np.load(source / "rgb" / f"{view_id}.npz", allow_pickle=False) as archive:
        photo = torch.from_numpy(archive["color"].copy())
    camera = OLD.Camera(**metadata["cameras"][index])
    if photo.dtype != torch.float32 or photo.shape != (camera.height, camera.width, 3):
        raise RuntimeError("probe photo cache has invalid dtype/shape")
    OLD.require_finite(photo)
    return {"view_id": view_id, "camera": camera, "rgb": photo}


def numerical_control(task, run, renderer, clock, storage, deadlines):
    started = clock()
    probe = task["gradient_protocol"]["repeatability_probe"]
    state = next(row for row in task["states"] if row["id"] == probe["state"])
    record = {
        "state_id": state["id"],
        "view_id": probe["view"],
        "passed": False,
        "stage_interval": [started, started],
        "whole_chain": {"repeats": []},
        "fixed_image": {"repeats": []},
        "fixed_vjp": {"repeats": []},
        "zero_controls": [],
        "ordinary_parameter_repeat_gate": "descriptive only; predecessor remains failed",
    }
    try:
        with deadlines.state(task["execution_budget"]["max_state_seconds"]):
            view = load_probe(task)
            saved = OLD.Gaussians3D.load_npz(ROOT / state["model"])
            unchanged = OLD.tensor_digest(OLD.model_tensors(saved))
            photo_digest = OLD.tensor_digest({"rgb": view["rgb"]})
            for pair, seed in enumerate(probe["seeds"]):
                for placement in ("before", "after"):
                    if placement == "after":
                        for route in ("rgb", "field_high"):
                            parameters, image = render_graph(saved, state, view, renderer, "cuda:0")
                            h = image_adjoints(image, OLD.target_route(view, route, identity=True))
                            gradients = {
                                c: vjp(parameters, image, h["adjoints"][c]) for c in COMPONENTS
                            }
                            receipt = storage.save(
                                run / "numerical_control" / f"whole_{pair}_{route}.npz",
                                repeat_arrays(image, h["adjoints"], gradients),
                            )
                            record["whole_chain"]["repeats"].append(
                                {"seed": seed, "route": route, "losses": h["losses"], **receipt}
                            )
                            del parameters, image, h, gradients
                    parameters, image = render_graph(saved, state, view, renderer, "cuda:0")
                    control, arrays = zero_control(parameters, image, view, state["id"])
                    control.update(
                        placement=placement,
                        pair=pair,
                        **storage.save(
                            run / "numerical_control" / f"zero_{pair}_{placement}.npz", arrays
                        ),
                    )
                    record["zero_controls"].append(control)
                    write_json(run / "numerical_control.json", record)
                    if not control["passed"]:
                        raise RuntimeError("photo-only zero-effect/routing gate failed")
                    del parameters, image, arrays
            parameters, image = render_graph(saved, state, view, renderer, "cuda:0")
            fixed = image.detach().clone()
            for repeat in range(probe["repeats"]):
                h = image_adjoints(fixed, OLD.target_route(view, "rgb"))
                record["fixed_image"]["repeats"].append(
                    {
                        "losses": h["losses"],
                        **storage.save(
                            run / "numerical_control" / f"fixed_image_{repeat}.npz",
                            repeat_arrays(fixed, h["adjoints"]),
                        ),
                    }
                )
            fixed_h = image_adjoints(fixed, OLD.target_route(view, "rgb"))
            if any(not bool(h.abs().any()) for h in fixed_h["adjoints"].values()):
                raise RuntimeError("fixed-adjoint VJP probe requires nonzero photo adjoints")
            record["fixed_vjp"]["fixed_adjoint_norms"] = {
                c: float(torch.linalg.vector_norm(h.detach().double()))
                for c, h in fixed_h["adjoints"].items()
            }
            record["fixed_vjp"]["fixed_inputs"] = storage.save(
                run / "numerical_control/fixed_vjp_inputs.npz",
                repeat_arrays(fixed, fixed_h["adjoints"]),
            )
            for repeat in range(probe["repeats"]):
                gradients = {c: vjp(parameters, image, fixed_h["adjoints"][c]) for c in COMPONENTS}
                record["fixed_vjp"]["repeats"].append(
                    storage.save(
                        run / "numerical_control" / f"fixed_vjp_{repeat}.npz",
                        repeat_arrays(gradients=gradients),
                    )
                )
            passed = True
            for family in ("whole_chain", "fixed_image", "fixed_vjp"):
                summary, gate = pairwise(record[family]["repeats"], run)
                record[family]["pairwise"] = summary
                if family == "fixed_vjp":
                    record[family]["upstream_gate_applicable"] = False
                    record[family]["boundary"] = (
                        "fixed inputs; parameter repeat variation is descriptive"
                    )
                else:
                    record[family]["upstream_gate_passed"] = gate
                passed &= gate
            record["saved_state_unchanged"] = unchanged == OLD.tensor_digest(
                OLD.model_tensors(saved)
            )
            record["photo_unchanged"] = photo_digest == OLD.tensor_digest({"rgb": view["rgb"]})
            record["passed"] = bool(
                passed and record["saved_state_unchanged"] and record["photo_unchanged"]
            )
            if not record["passed"]:
                raise RuntimeError("photo-only render/image-adjoint numerical gate failed")
    except BaseException as error:
        record["error"] = str(error)
        raise
    finally:
        record["stage_interval"][1] = clock()
        write_json(run / "numerical_control.json", record)


def state_diagnostics(task, run, state, cached, renderer, clock, storage, deadlines):
    started = clock()
    directory = run / "states" / state["id"]
    directory.mkdir(parents=True)
    record = {"state": state, "status": "running", "rows": [], "stage_interval": [started, started]}
    try:
        with deadlines.state(task["execution_budget"]["max_state_seconds"]):
            storage.check()
            saved = OLD.Gaussians3D.load_npz(ROOT / state["model"])
            OLD.require_finite(*OLD.model_tensors(saved).values())
            if saved.sh_degree != state["sh_degree"]:
                raise RuntimeError("saved SH degree differs from frozen state")
            before = OLD.tensor_digest(OLD.model_tensors(saved))
            effective = OLD.fresh_parameters(saved, state["kind"], "cuda:0")
            alpha = effective["opacities"].detach().cpu()
            record.update(
                n_gaussians=saved.n,
                model_sha256=OLD.sha256(ROOT / state["model"]),
                saved_tensor_sha256=before,
                effective_parameter_sha256=OLD.tensor_digest(effective),
                parameter_shapes={name: list(value.shape) for name, value in effective.items()},
                opacity={
                    "saved_zero_count": int((saved.opacity == 0).sum()),
                    "saved_one_count": int((saved.opacity == 1).sum()),
                    "effective_zero_count": int((alpha == 0).sum()),
                    "effective_one_count": int((alpha == 1).sum()),
                    "max_saved_to_effective_absolute_difference": float(
                        (alpha - saved.opacity).abs().max()
                    ),
                    "initial_entry_roundtrip": state["kind"] == "initial",
                },
                coordinates=task["gradient_protocol"]["coordinates"],
            )
            del effective
            sums = {}
            for ordinal, view in enumerate(cached["views"], 1):
                storage.check()
                view_start = clock()
                parameters, image = render_graph(saved, state, view, renderer, "cuda:0")
                if ordinal == 1:
                    control, arrays = zero_control(parameters, image, view, state["id"])
                    control.update(storage.save(directory / "identity_arrays.npz", arrays))
                    record["identity_control"] = control
                    write_json(directory / "diagnostics.json", record)
                    if not control["passed"]:
                        raise RuntimeError("state identity-routing gate failed")
                    del arrays
                measured = repeat_measurement(parameters, image, view)
                receipt = storage.save(
                    directory / "views" / f"{view['view_id']}.npz",
                    raw_repeat_arrays(measured["raw"]),
                )
                for target, components in measured["means"].items():
                    for component, groups in components.items():
                        for group, value in groups.items():
                            key = f"{target}__{component}__{group}"
                            if key not in sums:
                                sums[key] = torch.zeros_like(value)
                            sums[key].add_(value)
                torch.cuda.synchronize()
                record["rows"].append(
                    {
                        "view_id": view["view_id"],
                        "step": ordinal,
                        "wall_seconds": clock(),
                        "view_wall_seconds": clock() - view_start,
                        "input_sha256": view["input_sha256"],
                        "losses": measured["losses"],
                        "comparisons": measured["comparisons"],
                        "reliability": measured["reliability"],
                        **receipt,
                    }
                )
                record["stage_interval"][1] = clock()
                write_json(directory / "diagnostics.json", record)
                print(f"{state['id']} view {ordinal}/{len(cached['views'])}", flush=True)
                del parameters, image, measured
            means = {key: value / len(cached["views"]) for key, value in sums.items()}
            saved_arrays = storage.save(
                directory / "mean_gradients.npz",
                {key: value.numpy() for key, value in means.items()},
            )
            record["mean_gradients_sha256"] = saved_arrays["sha256"]
            record["aggregate"] = {c: {} for c in COMPONENTS}
            for component in COMPONENTS:
                for group in GROUPS:
                    delta = means[f"delta__{component}__{group}"]
                    stats = OLD.compare_gradients(
                        means[f"rgb__{component}__{group}"],
                        means[f"field_high__{component}__{group}"],
                    )
                    stats.update(
                        direct_delta_norm=float(torch.linalg.vector_norm(delta)),
                        direct_delta_max_absolute=float(delta.abs().max()),
                    )
                    record["aggregate"][component][group] = stats
            record["aggregate_definition"] = (
                "statistics of view-mean photo/derived-field/direct-delta gradients"
            )
            record["saved_state_unchanged"] = before == OLD.tensor_digest(OLD.model_tensors(saved))
            if not record["saved_state_unchanged"]:
                raise RuntimeError("saved state changed during measurement")
            record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        record["stage_interval"][1] = clock()
        write_json(directory / "diagnostics.json", record)


def selftest(device="cpu"):
    from rtgs.render.base import get_rasterizer

    renderer = get_rasterizer("gsplat" if device.startswith("cuda") else "torch", device=device)
    saved, state, view = OLD.synthetic_inputs()
    view["view_id"] = "synthetic"
    parameters, image = render_graph(saved, state, view, renderer, device)
    control, _ = zero_control(parameters, image, view, "synthetic")
    assert control["passed"]
    measured = repeat_measurement(parameters, image, view)
    independent = {
        target: OLD.evaluate_target(saved, state, view, target, renderer, device)
        for target in ("rgb", "field_high")
    }
    parity = {}
    for component in COMPONENTS:
        parity[component] = {}
        for group in GROUPS:
            expected = (
                independent["field_high"]["gradients"][component][group].double()
                - independent["rgb"]["gradients"][component][group].double()
            )
            stats = tolerance_stats(
                measured["means"]["delta"][component][group], expected, atol=1e-7, rtol=1e-4
            )
            assert stats["tolerance_violations"] == 0, (component, group, stats)
            parity[component][group] = stats
    # Independent full-loss backward validates declared component weighting as well.
    direct = OLD.evaluate_target(saved, state, view, "field_high", renderer, device, direct=True)
    for group in GROUPS:
        stats = tolerance_stats(
            measured["means"]["field_high"]["total"][group],
            direct["gradients"]["total"][group],
            atol=1e-7,
            rtol=1e-4,
        )
        assert stats["tolerance_violations"] == 0, (group, stats)
    logits = torch.logit(saved.opacity.to(device)).detach().requires_grad_(True)
    logit_parameters = OLD.fresh_parameters(saved, state["kind"], device)
    logit_parameters["opacities"] = torch.sigmoid(logits)
    prediction = renderer.render(
        OLD.assemble(logit_parameters), view["camera"].to(device), sh_degree=state["sh_degree"]
    ).color
    photo = view["rgb"].to(device)
    logit_loss = 0.8 * (prediction - photo).abs().mean() + 0.2 * (1 - OLD.ssim(prediction, photo))
    (direct_logit,) = torch.autograd.grad(logit_loss, logits)
    opacity_control = tolerance_stats(
        measured["means"]["rgb"]["total"]["opacities"], direct_logit, atol=1e-7, rtol=1e-4
    )
    assert opacity_control["tolerance_violations"] == 0, opacity_control
    return {
        "device": device,
        "common_render_control": control,
        "distinct_target_parity": parity,
        "full_loss_linearity": "passed",
        "opacity_chain_control": opacity_control,
        "no_capture_inputs": True,
    }


def publisher():
    return load_module(
        Path(__file__).with_name(Path(__file__).stem + "_report.py"), "adjoint_report"
    )


def coordinate(task_path, task, run):
    start = time.perf_counter()
    clock = OLD.RunClock(run)
    execution = {
        "status": "running",
        "stage_intervals": {},
        "states": [],
        "optimization_steps": 0,
        "topology_updates": 0,
        "started_run_seconds": clock(),
    }
    guard, storage = PhaseGuard(task), StorageBudget(run, task)
    guard.install()
    write_json(run / "execution.json", execution)
    try:
        with OLD.Deadlines(task["execution_budget"]["max_seconds"]) as deadlines:
            OLD.write_environment(run)
            execution["storage_preflight"] = storage.check(preflight=True)
            verify_start = clock()
            write_json(
                run / "input_integrity_entry.json",
                {"source": source_guard(task_path, task, run), "cache": OLD.cache_guard(task)},
            )
            execution["verification_entry_interval"] = [verify_start, clock()]
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no fallback")
            from rtgs.render.base import get_rasterizer

            settings = dict(task["gradient_protocol"]["renderer"])
            renderer = get_rasterizer(settings.pop("backend"), device="cuda:0", **settings)
            saved, state, view = OLD.synthetic_inputs()
            parameters, image = render_graph(saved, state, view, renderer, "cuda:0")
            h = image_adjoints(image, view["rgb"])
            vjp(parameters, image, h["adjoints"]["total"])
            del parameters, image, h
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            guard.phase = "numerical_control"
            numerical_control(task, run, renderer, clock, storage, deadlines)
            execution["stage_intervals"]["numerical_control"] = read_json(
                run / "numerical_control.json"
            )["stage_interval"]
            guard.phase = "residual"
            residual_start = clock()
            cached = OLD.load_cached(task)
            residual = {"rows": [], "stage_interval": [residual_start, residual_start]}
            for ordinal, view in enumerate(cached["views"], 1):
                residual["rows"].append(
                    {
                        "view_id": view["view_id"],
                        "step": ordinal,
                        "input_sha256": view["input_sha256"],
                        **OLD.residual_statistics(
                            view["rgb"], view["field_high"], view["initialization_masks"]
                        ),
                        "wall_seconds": clock(),
                    }
                )
                residual["stage_interval"][1] = clock()
                write_json(run / "residuals.json", residual)
            execution["stage_intervals"]["residual"] = residual["stage_interval"]
            guard.phase = "gradient"
            gradient_start = clock()
            by_id = {state["id"]: state for state in task["states"]}
            for state_id in task["gradient_protocol"]["state_order"]:
                state_diagnostics(
                    task, run, by_id[state_id], cached, renderer, clock, storage, deadlines
                )
                execution["states"].append(state_id)
                execution["stage_intervals"]["gradient"] = [gradient_start, clock()]
                execution["resource"] = OLD.resource_receipt(start)
                write_json(run / "execution.json", execution)
            guard.phase = "presentation"
            presentation_start = clock()
            storage.check()
            OLD.presentation(task, run, cached)
            storage.check()
            execution["stage_intervals"]["presentation"] = [presentation_start, clock()]
            guard.phase = "verification_exit"
            verify_start = clock()
            write_json(
                run / "input_integrity_exit.json",
                {
                    "source": source_guard(task_path, task, run),
                    "cache": OLD.cache_guard(task),
                    "split_sha256": hashlib.sha256(
                        json.dumps(task["splits"], sort_keys=True).encode()
                    ).hexdigest(),
                },
            )
            execution["verification_exit_interval"] = [verify_start, clock()]
            execution.update(
                status="completed",
                resource=OLD.resource_receipt(start),
                finished_run_seconds=clock(),
            )
            write_json(run / "execution.json", execution)
            write_json(run / "input_access.json", guard.receipt)
            OLD.write_boundary_resources(task, run, execution, guard)
            guard.phase = "report"
            publisher().publish(task, run)
    except BaseException as error:
        guard.phase = "failure_exit_verification"
        integrity = {}
        for name, check in (
            ("source", lambda: source_guard(task_path, task, run)),
            ("cache", lambda: OLD.cache_guard(task)),
        ):
            try:
                integrity[name] = check()
            except BaseException as issue:
                integrity[name] = {"all_match": False, "error": str(issue)}
        write_json(run / "input_integrity_exit.json", integrity)
        execution.update(
            status="failed",
            error=str(error),
            traceback=traceback.format_exc(),
            resource=OLD.resource_receipt(start),
            finished_run_seconds=clock(),
        )
        write_json(run / "execution.json", execution)
        write_json(run / "input_access.json", guard.receipt)
        OLD.write_boundary_resources(task, run, execution, guard)
        write_json(
            run / "execution_failure.json",
            {"error": str(error), "traceback": traceback.format_exc()},
        )
        try:
            publisher().publish_failure(task, run, str(error))
        except BaseException as issue:
            write_json(run / "failure_report_error.json", {"error": str(issue)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "selftest"))
    parser.add_argument("--task", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "selftest":
        print(json.dumps(selftest(args.device), indent=2, allow_nan=False))
        return
    if args.task is None:
        parser.error("--task required")
    task_path = args.task.resolve()
    task = read_json(task_path)
    run = ROOT / "runs" / task["task_id"]
    if any(
        (run / name).exists()
        for name in ("execution.json", "numerical_control.json", "residuals.json")
    ):
        raise RuntimeError("run attempt already exists; no automatic retry/overwrite")
    coordinate(task_path, task, run)


if __name__ == "__main__":
    main()
