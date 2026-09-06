#!/usr/bin/env python3
"""Run RTGS-016's prospectively reviewed, compact-only constraint comparison.

Preparation is a separate image-capable program. This coordinator launches fresh
CPU workers; each worker loads only its permitted compact teachers. Frozen source,
review and run-lock checks apply before any reconstruction or held-out access.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260906_tomography_source_constraints_haelyn_dome"
TASK_PATH = ROOT / "experiments/tasks" / f"{TASK_ID}.json"
RUN_PATH = ROOT / "runs" / TASK_ID
ARMS = ("hard", "soft", "free", "beam_reference")
PROCESS_STARTED = time.perf_counter()


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    return value


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def configs(task: dict, arm: str, seed: int, *, dataset_id: str, warmup: bool = False):
    from rtgs.lift.beam_fusion import BeamFusionConfig
    from rtgs.lift.carrier_refinement import CarrierRepairConfig
    from rtgs.lift.field_lifter import FieldLiftConfig
    from rtgs.lift.field_refit import FieldRefitConfig
    from rtgs.optim.compact_trainer import CompactTrainConfig

    frozen = task["frozen_configuration"]
    common = dict(frozen["field_refit"])
    treatments = common.pop("source_constraints")
    mode, weight = treatments.get(arm, treatments["hard"])
    refit = FieldRefitConfig(**common, source_constraint=mode, source_anchor_weight=weight)
    native = CompactTrainConfig(**frozen["native_refinement"], seed=seed)
    if warmup:
        warm = frozen["warmup"]
        refit = replace(refit, iterations=warm["proxy_iterations"], appearance_start=0)
        native = replace(
            native, iterations=warm["native_iterations"], checkpoints=(0, warm["native_iterations"])
        )
    field = FieldLiftConfig(
        **frozen["field_lift"],
        seed=seed,
        refit=refit,
        mask_mode="none" if dataset_id == "haelyn_unmasked" else "hard",
    )
    beam = BeamFusionConfig(**frozen["beam_reference"]["beam"])
    repair = CarrierRepairConfig(**frozen["beam_reference"]["repair"])
    if warmup:
        repair = replace(repair, covariance_steps=frozen["warmup"]["proxy_iterations"])
    return field, native, beam, repair


def validate_binding(task_path: Path, run: Path) -> dict:
    if task_path.resolve() != TASK_PATH or run.resolve() != RUN_PATH:
        raise ValueError("task and run must use the one canonical RTGS-016 path")
    contract = module(ROOT / "scripts/experiment_contract.py", "_rtgs016_contract")
    task, _lock, errors = contract._locked_task(run, root=ROOT)
    if errors:
        raise ValueError("invalid run binding: " + "; ".join(errors))
    if task["task_id"] != TASK_ID or tuple(d["id"] for d in task["comparators"]) != ARMS:
        raise ValueError("protocol identity or comparator order changed")
    if errors := contract.verify_source_binding(task, root=ROOT):
        raise ValueError("; ".join(errors))
    expected = task["frozen_configuration"]["effective_configuration"]
    actual = effective_configuration(task)
    if safe(actual) != expected:
        raise ValueError("effective configuration differs from the frozen protocol")
    return task


def effective_configuration(task: dict) -> dict:
    return {
        dataset: {
            arm: {
                str(seed): hashlib.sha256(
                    json.dumps(
                        safe([asdict(c) for c in configs(task, arm, seed, dataset_id=dataset)]),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
                for seed in task["seeds"]
            }
            for arm in ARMS
        }
        for dataset in task["frozen_configuration"]["phase_order"]
    }


def boundary(task: dict, dataset: dict, readable: set[str]):
    """Reuse the tested live import/open guard and restrict view/reference reads."""
    base = module(
        ROOT / "scripts/experiments/20260729_field_sweep_placement_stage_frames00008_00009.py",
        "_rtgs016_import_guard",
    )
    # Beam is the explicit practical reference. All workers retain the four
    # image/SceneData/RGB-trainer probes; its import is the positive control.
    base.FORBIDDEN_MODULES = tuple(
        x
        for x in base.FORBIDDEN_MODULES
        if x not in {"rtgs.lift.beam_fusion", "rtgs.lift.carrier_refinement"}
    )
    base.NEGATIVE_CONTROL_COUNT = 4
    compact = (ROOT / dataset["compact_manifest"]).parent.resolve()
    reference = (ROOT / task["frozen_configuration"]["source_reference"]["path"]).parent.resolve()

    class Boundary(base.NoImageGuard):
        def __init__(self):
            super().__init__()
            self.opened_compact = []
            self.phase = "fit"

        def _forbidden_path(self, value):
            if super()._forbidden_path(value):
                return True
            if isinstance(value, int):
                return False
            try:
                path = Path(os.fspath(value)).resolve()
            except TypeError:
                return False
            if path.parent == reference:
                return True
            if path.suffix == ".rtgsv":
                if path.parent != compact or path.stem not in readable:
                    return True
                self.opened_compact.append({"view_id": path.stem, "phase": self.phase})
            return False

    return Boundary()


def _float32(model):
    import torch

    from rtgs.core.gaussians3d import Gaussians3D

    if model.n == 0 or not all(
        torch.isfinite(getattr(model, k)).all()
        for k in ("means", "quats", "log_scales", "opacity", "sh")
    ):
        raise RuntimeError("empty or non-finite saved Gaussian state")
    return Gaussians3D(
        **{
            k: getattr(model, k).detach().cpu().float()
            for k in ("means", "quats", "log_scales", "opacity", "sh")
        }
    )


def check_placement(diagnostics: dict) -> None:
    reason = diagnostics.get("placement_fallback_reason")
    if reason is not None:
        raise RuntimeError(f"frozen compact_carve placement used fallback: {reason}")


def check_proxy(refit) -> None:
    for name in ("objective_history", "elapsed_seconds"):
        if not all(math.isfinite(float(value)) for value in getattr(refit, name)):
            raise RuntimeError(f"non-finite proxy {name}")


def check_windows(data) -> None:
    for view in data.views:
        field = view.observation
        if field.fit_window != (0, 0, field.width, field.height):
            raise RuntimeError("frozen full-canvas teacher fit window changed")


def worker(task: dict, run: Path, dataset_id: str, arm: str, seed: int, warmup: bool) -> None:
    run = run.resolve()
    started = PROCESS_STARTED
    if warmup:
        frozen_warmup = task["frozen_configuration"]["warmup"]
        if (dataset_id, arm, seed) != (
            frozen_warmup["dataset_id"],
            frozen_warmup["arm_id"],
            frozen_warmup["seed"],
        ):
            raise ValueError("warmup cell differs from the frozen warmup")
    if arm not in ARMS or seed not in task["seeds"]:
        raise ValueError("unfrozen cell")
    dataset = next(d for d in task["datasets"] if d["id"] == dataset_id)
    output = run / ("warmup" if warmup else "cells") / dataset_id / f"seed_{seed}" / arm
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cell {output}")
    output.mkdir(parents=True)
    validation = task["frozen_configuration"]["validation_views"][dataset_id]
    train = [v for v in task["splits"][dataset_id]["train"] if v not in validation]
    heldout = task["splits"][dataset_id]["heldout"]
    readable = set(train + validation)
    guard = boundary(task, dataset, readable)
    compact_path = (ROOT / dataset["compact_manifest"]).parent
    selected_paths = [compact_path / "manifest.json"] + [
        compact_path / f"{v}.rtgsv" for v in train + validation
    ]
    selected_entry = {str(p): sha(p) for p in selected_paths}
    try:
        with guard:
            import torch

            from rtgs.data.compact_views import CompactDataset
            from rtgs.data.field_inputs import SceneFits
            from rtgs.lift import field_lifter as lifting
            from rtgs.lift.beam_fusion import fuse_gaussian_beams
            from rtgs.lift.carrier_refinement import repair_beam_carriers
            from rtgs.optim.compact_trainer import CompactTrainer

            torch.set_num_threads(1)
            torch.manual_seed(seed)
            field_cfg, native_cfg, beam_cfg, repair_cfg = configs(
                task, arm, seed, dataset_id=dataset_id, warmup=warmup
            )
            masked = dataset_id != "haelyn_unmasked"
            path = (ROOT / dataset["compact_manifest"]).parent
            cap = task["frozen_configuration"]["converter"]["view_byte_cap"]
            load_alpha = masked and arm != "beam_reference"
            training = CompactDataset.load(
                path, byte_cap=cap, load_alpha=load_alpha, view_ids=train
            )
            validation_data = CompactDataset.load(
                path, byte_cap=cap, load_alpha=False, view_ids=validation
            )
            check_windows(training)
            check_windows(validation_data)
            inputs = training.to_reconstruction_inputs()
            evaluator = CompactTrainer(native_cfg)
            evaluation_seconds = 0.0
            records = []
            markers = []
            diagnostics = {}
            models = {}
            stage_start = 0.0

            def marker(stage, which, step, elapsed):
                markers.append(
                    {
                        "stage": stage,
                        "boundary": which,
                        "step": step,
                        "wall_seconds": elapsed,
                        "label": next(s["label"] for s in task["stages"] if s["id"] == stage),
                    }
                )

            def observe(model, stage, step, save_name):
                nonlocal evaluation_seconds
                model = _float32(model)
                model.save_ply(output / save_name)
                models[save_name] = sha(output / save_name)
                if save_name == "gaussians_init.ply" and arm in {"soft", "free"} and not warmup:
                    baseline = output.parent / "hard" / save_name
                    if not baseline.is_file() or sha(baseline) != models[save_name]:
                        raise RuntimeError("causal initialization differs before refit")
                tick = time.perf_counter()
                if not warmup:
                    result = evaluator.evaluate(validation_data.to_reconstruction_inputs(), model)
                    records.append(
                        {
                            "stage": stage,
                            "step": step,
                            "split": "validation",
                            "metric_id": "native_teacher_mse",
                            "value": result["J_pixel"],
                            "wall_seconds": time.perf_counter() - started,
                        }
                    )
                evaluation_seconds += time.perf_counter() - tick
                return model

            marker("placement", "start", 0, 0.0)
            if arm == "beam_reference":
                beam = fuse_gaussian_beams(inputs, beam_cfg)
                init = observe(beam.gaussians, "placement", 0, "gaussians_init.ply")
                stage_start = time.perf_counter() - started
                marker("placement", "end", 0, stage_start)
                marker("field_refit", "start", 0, stage_start)
                repaired = repair_beam_carriers(beam, inputs, repair_cfg)
                proxy = observe(
                    repaired.gaussians,
                    "field_refit",
                    repair_cfg.covariance_steps,
                    "gaussians_proxy.ply",
                )
                diagnostics["repair"] = repaired.diagnostics
                proxy_steps = repair_cfg.covariance_steps
                source_drift = None
            else:
                fits = SceneFits.from_compact_dataset(training, geometry_is_train_only=True)
                original_place = lifting._place

                def timed_place(*args, **kwargs):
                    nonlocal stage_start
                    placed = original_place(*args, **kwargs)
                    check_placement(placed.diagnostics)
                    initial = placed.fiber.as_gaussians(
                        colors=placed.source_colors, opacity=placed.render_opacity
                    ).detach()
                    observe(initial, "placement", 0, "gaussians_init.ply")
                    stage_start = time.perf_counter() - started
                    marker("placement", "end", 0, stage_start)
                    marker("field_refit", "start", 0, stage_start)
                    return placed

                lifting._place = timed_place
                try:
                    lifted = lifting.FieldLifter(field_cfg).fit(fits)
                finally:
                    lifting._place = original_place
                check_placement(lifted.diagnostics)
                check_proxy(lifted.refit)
                proxy_steps = field_cfg.refit.iterations
                proxy = observe(lifted.gaussians, "field_refit", proxy_steps, "gaussians_proxy.ply")
                source_drift = float(lifted.refit.fiber.source_mean_offset.square().mean().sqrt())
                diagnostics["field"] = lifted.diagnostics
                diagnostics["proxy_objective"] = list(lifted.refit.objective_history)
                diagnostics["proxy_elapsed_seconds"] = list(lifted.refit.elapsed_seconds)
                init = _float32(lifted.gaussians_init)
                # Compare file-level native states, not labels or component counts alone.
                check = output / "initial_parity.ply"
                init.save_ply(check)
                if sha(check) != models["gaussians_init.ply"]:
                    raise RuntimeError("placement observer changed the initial state")
                check.unlink()
            proxy_endpoint = time.perf_counter() - started
            marker("field_refit", "end", proxy_steps, proxy_endpoint)
            marker("native_refine", "start", proxy_steps, proxy_endpoint)

            def checkpoint(model, step):
                if step != native_cfg.iterations:
                    observe(
                        model, "native_refine", proxy_steps + step, f"checkpoint_{step:04d}.ply"
                    )

            final, native_history = CompactTrainer(native_cfg).train(
                inputs,
                proxy,
                checkpoint_callback=checkpoint,
            )
            final = observe(
                final, "native_refine", proxy_steps + native_cfg.iterations, "gaussians.ply"
            )
            endpoint = time.perf_counter() - started
            marker("native_refine", "end", proxy_steps + native_cfg.iterations, endpoint)
            selected_exit = {str(p): sha(p) for p in selected_paths}
            if selected_exit != selected_entry:
                raise RuntimeError("selected compact inputs changed during fitting")
            if "source_binding" in task["frozen_configuration"]:
                contract = module(ROOT / "scripts/experiment_contract.py", "_rtgs016_worker_exit")
                if errors := contract.verify_source_binding(task, root=ROOT):
                    raise RuntimeError("source changed during fitting: " + "; ".join(errors))
            # Held-out teacher archives cannot open until the final artifact exists.
            if not (output / "gaussians.ply").is_file():
                raise RuntimeError("missing saved endpoint")
            heldout_metrics = {}
            if not warmup:
                readable.update(heldout)
                guard.phase = "reporting_after_saved_endpoint"
                held = CompactDataset.load(path, byte_cap=cap, load_alpha=False, view_ids=heldout)
                check_windows(held)
                teacher = held.to_reconstruction_inputs()
                for label, model in (("initial", init), ("proxy", proxy), ("final", final)):
                    heldout_metrics[label] = evaluator.evaluate(teacher, model)
            guard_record = guard.record()
            guard_record["opened_compact"] = guard.opened_compact
            guard_record["optimizer_views"] = train
            guard_record["validation_views"] = validation
            guard_record["heldout_views"] = heldout
            guard_record["load_alpha"] = load_alpha
            guard_record["mask_mode"] = field_cfg.mask_mode if arm != "beam_reference" else "none"
            guard_record["embedded_alpha_use"] = "source_support_gating" if load_alpha else "none"
            if not guard_record["passed"]:
                raise RuntimeError("compact input guard failed")
            for item in records + markers:
                item.update(dataset_id=dataset_id, arm_id=arm, seed=seed)
            write_new(
                output / "summary.json",
                {
                    "schema_version": 1,
                    "task_id": TASK_ID,
                    "status": "completed",
                    "dataset_id": dataset_id,
                    "arm_id": arm,
                    "seed": seed,
                    "warmup": warmup,
                    "initial_gaussians": init.n,
                    "final_gaussians": final.n,
                    "initial_sha256": models["gaussians_init.ply"],
                    "models": models,
                    "records": records,
                    "stage_markers": markers,
                    "heldout_metrics": heldout_metrics,
                    "source_mean_drift": source_drift,
                    "placement_fallback_reason": None
                    if arm == "beam_reference"
                    else lifted.diagnostics.get("placement_fallback_reason"),
                    "proxy_accepted_steps": None
                    if arm == "beam_reference"
                    else lifted.refit.accepted_steps,
                    "worker_endpoint_seconds": endpoint,
                    "endpoint_excluding_validation_seconds": endpoint - evaluation_seconds,
                    "validation_observer_seconds": evaluation_seconds,
                    "worker_total_seconds_including_heldout": time.perf_counter() - started,
                    "peak_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "effective": [asdict(c) for c in (field_cfg, native_cfg, beam_cfg, repair_cfg)],
                    "diagnostics": diagnostics,
                    "native_history": native_history,
                    "input_boundary": guard_record,
                    "selected_input_binding": {"entry": selected_entry, "exit": selected_exit},
                },
            )
    except BaseException as error:
        write_new(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "wall_seconds": time.perf_counter() - started,
                "task_id": TASK_ID,
                "dataset_id": dataset_id,
                "arm_id": arm,
                "seed": seed,
                "warmup": warmup,
                "records": [
                    dict(r, dataset_id=dataset_id, arm_id=arm, seed=seed)
                    for r in locals().get("records", [])
                ],
                "stage_markers": [
                    dict(r, dataset_id=dataset_id, arm_id=arm, seed=seed)
                    for r in locals().get("markers", [])
                ],
            },
        )
        raise


def verify_external_seal(task: dict) -> dict:
    seal = task["frozen_configuration"]["external_evaluation"]["reference_files_seal"]
    path = ROOT / seal["path"]
    if sha(path) != seal["sha256"]:
        raise ValueError("external evaluation seal changed")
    payload = json.loads(path.read_text())
    for entry in payload["files"]:
        item = ROOT / entry["path"]
        if item.stat().st_size != entry["bytes"] or sha(item) != entry["sha256"]:
            raise ValueError(f"external reference changed: {entry['path']}")
    return payload


def coordinator(task: dict, run: Path) -> None:
    run = run.resolve()
    # Refusing a previous attempt must never rewrite its existing evidence.
    if any(
        (run / name).exists() for name in ("cells", "warmup", "metrics.json", "run_receipt.json")
    ):
        raise FileExistsError("existing outputs require an explicit audit; nothing overwritten")
    progress = {"phase": "input_validation"}
    try:
        _coordinate(task, run, progress)
    except BaseException as error:
        report = module(Path(__file__).with_name(TASK_ID + "_report.py"), "_rtgs016_failed_report")
        report.failure(task, run, error, progress["phase"])
        raise


def _coordinate(task: dict, run: Path, progress: dict) -> None:
    contract = module(ROOT / "scripts/experiment_contract.py", "_rtgs016_coordinator_contract")
    report = module(Path(__file__).with_name(TASK_ID + "_report.py"), "_rtgs016_report")
    if errors := contract.verify_data_seal(task, root=ROOT):
        raise ValueError("; ".join(errors))
    verify_external_seal(task)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    env.pop("PYTHONDONTWRITEBYTECODE", None)

    def launch(dataset, arm, seed, warmup=False):
        progress["phase"] = f"{'warmup' if warmup else 'cell'}/{dataset}/{arm}/{seed}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--task",
            str(TASK_PATH),
            "--run",
            str(run),
            "--cell",
            dataset,
            arm,
            str(seed),
        ]
        if warmup:
            command.append("--warmup")
        tick = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        out = run / ("warmup" if warmup else "cells") / dataset / f"seed_{seed}" / arm
        write_new(
            out / "process_receipt.json",
            {
                "argv": command,
                "exit_code": result.returncode,
                "subprocess_wall_seconds": time.perf_counter() - tick,
                "source_binding": task["frozen_configuration"]["source_binding"],
                "data_seal_sha256": sha(ROOT / task["data_seal"]),
            },
        )
        if result.returncode:
            raise RuntimeError(f"cell failed with exit {result.returncode}: {out}")
        print(f"completed {dataset} {arm} seed={seed} warmup={warmup}", flush=True)

    warm = task["frozen_configuration"]["warmup"]
    launch(warm["dataset_id"], warm["arm_id"], warm["seed"], True)
    for dataset in task["frozen_configuration"]["phase_order"]:
        for arm in ARMS:
            for seed in task["seeds"]:
                launch(dataset, arm, seed)
        for seed in task["seeds"]:
            hashes = [
                json.loads(
                    (run / "cells" / dataset / f"seed_{seed}" / arm / "summary.json").read_text()
                )["initial_sha256"]
                for arm in ARMS[:3]
            ]
            if len(set(hashes)) != 1:
                raise RuntimeError("causal arms did not start from identical native states")
        # Reporting runs in its own image-capable process after the phase endpoints.
        progress["phase"] = f"external_report/{dataset}"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name(TASK_ID + "_report.py")),
                "--task",
                str(TASK_PATH),
                "--run",
                str(run),
                "--phase",
                dataset,
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    progress["phase"] = "exit_integrity"
    validate_binding(TASK_PATH, run)
    if errors := contract.verify_data_seal(task, root=ROOT):
        raise ValueError("; ".join(errors))
    verify_external_seal(task)
    progress["phase"] = "aggregate"
    report.aggregate(task, run)
    print(
        "Producer artifacts saved; distinct results audit and browser smoke remain required.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cell", nargs=3, metavar=("DATASET", "ARM", "SEED"))
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    args.run = args.run.resolve()
    task = validate_binding(args.task, args.run)
    if args.cell:
        worker(task, args.run, args.cell[0], args.cell[1], int(args.cell[2]), args.warmup)
    elif args.warmup:
        parser.error("--warmup requires a frozen cell")
    else:
        coordinator(task, args.run)


if __name__ == "__main__":
    main()
