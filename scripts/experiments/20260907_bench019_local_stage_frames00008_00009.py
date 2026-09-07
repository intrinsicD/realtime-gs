#!/usr/bin/env python3
"""Two-gate local BENCH-019 development experiment.

The canonical command is frozen in the matching experiments/tasks JSON.
Initialize through experiment_contract.py, then invoke this file's run command. Preparation
publishes inputs and pauses until the independently finalized StructSplat protocol exists.
Worker modes are private subprocess boundaries of that frozen command.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260907_bench019_local_stage_frames00008_00009"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "structsplat" / "src"))


def read_json(path: Path) -> dict[str, Any]:
    from rtgs.bench019 import load_json_object

    return load_json_object(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        temporary = path.with_name(path.name + f".pending-{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, path)


def canonical_run(task: dict[str, Any], run: Path, *, root: Path = ROOT) -> Path:
    if task.get("task_id") != TASK_ID:
        raise ValueError("unexpected experiment task identity")
    expected = root.resolve() / "runs" / TASK_ID
    if run.resolve() != expected or run.is_symlink():
        raise ValueError("run must be the task's canonical non-symlink root")
    return expected


def verify_file_inventory(base: Path, inventory: dict[str, str]) -> None:
    if not inventory:
        raise ValueError("bound inventory must be nonempty")
    for relative, digest in inventory.items():
        candidate = base / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe bound path: {relative}")
        if (
            any(
                part.is_symlink() for part in [candidate, *candidate.parents] if part != base.parent
            )
            or not candidate.is_file()
        ):
            raise ValueError(f"missing/nonordinary bound file: {relative}")
        if not candidate.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"bound file escapes its root: {relative}")
        if sha256(candidate) != digest:
            raise ValueError(f"bound file changed: {relative}")


def verify_external_sources(repository: dict[str, Any]) -> None:
    base = Path(repository["root"])
    patterns = repository["patterns"]
    if not patterns or any(Path(p).is_absolute() or ".." in Path(p).parts for p in patterns):
        raise ValueError("external source patterns must be nonempty and relative")
    observed = {
        path.relative_to(base).as_posix()
        for pattern in patterns
        for path in base.glob(pattern)
        if path.is_file()
    }
    if observed != set(repository["files"]):
        raise ValueError("external behavior-bearing source inventory changed")
    verify_file_inventory(base, repository["files"])


def guard(task_path: Path, run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts import experiment_contract as contract

    task = read_json(task_path)
    canonical_run(task, run)
    lock = read_json(run / "task.lock.json")
    if lock["task_sha256"] != sha256(task_path):
        raise ValueError("ready task bytes changed after initialization")
    if lock["protocol_sha256"] != contract.protocol_sha256(task):
        raise ValueError("locked protocol changed")
    for repository in (ROOT, ROOT.parent / "structsplat"):
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository, text=True
        )
        if dirty.strip():
            raise ValueError(f"source worktree is not clean: {repository.name}")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if current_commit != lock["source_commit"]:
        raise ValueError("RTGS source commit differs from initialized run")
    problems = contract.validate_task(task, task_path)
    problems += contract.verify_data_seal(task)
    if sha256(ROOT / task["data_seal"]) != lock["data_seal_sha256"]:
        problems.append("locked raw seal bytes changed")
    problems += contract.verify_source_binding(task)
    review_path = ROOT / task["protocol_review"]["artifact"]
    if sha256(review_path) != lock["protocol_review_artifact_sha256"]:
        problems.append("initial prospective review artifact changed")
    for repository in task.get("external_source_bindings", []):
        verify_external_sources(repository)
    if problems:
        raise ValueError("preflight failed: " + "; ".join(problems))
    return task, lock


def environment() -> dict[str, Any]:
    import torch

    packages = {}
    for name in ("torch", "numpy", "gsplat", "Pillow", "realtime-gs", "structsplat"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "source-tree"
    return {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "device": {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
        },
    }


def verify_environment(task: dict[str, Any]) -> dict[str, Any]:
    observed = environment()
    policy = task["environment_policy"]
    if Path(sys.executable).absolute() != Path(policy["python"]).absolute():
        raise ValueError("Python executable differs from frozen environment")
    for package in ("torch", "gsplat"):
        if observed["packages"][package] != policy[package]:
            raise ValueError(f"{package} version differs from frozen environment")
    if observed["device"] != {"type": "cuda", "name": policy["gpu"], "cuda": policy["cuda_build"]}:
        raise ValueError("CUDA device/build differs from frozen environment")
    return observed


class Resources:
    """Bounded per-process measurements; foreign allocations never substitute for ours."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.nvml: Any = None
        self.started = time.perf_counter()

    def __enter__(self) -> Resources:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("explicit CUDA execution unavailable; no CPU fallback")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._sample()
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
        except Exception as error:
            self.errors.append(f"NVML unavailable: {type(error).__name__}: {error}")
        return self

    def _sample(self) -> None:
        try:
            processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            own = []
            foreign = []
            for process in processes:
                used = getattr(process, "usedGpuMemory", None)
                available = isinstance(used, int) and 0 <= used < (1 << 60)
                record = {"pid": int(process.pid), "bytes": int(used) if available else None}
                (own if int(process.pid) == os.getpid() else foreign).append(record)
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.samples.append(
                {
                    "wall_seconds": time.perf_counter() - self.started,
                    "own_processes": own,
                    "foreign_compute_processes": foreign,
                    "device_used_bytes": int(memory.used),
                    "device_total_bytes": int(memory.total),
                }
            )
        except Exception as error:
            self.errors.append(f"NVML sample unavailable: {type(error).__name__}: {error}")

    def _monitor(self) -> None:
        while not self.stop.wait(self.interval):
            self._sample()

    def __exit__(self, *_: object) -> None:
        import torch

        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval * 3))
            self._sample()
        try:
            torch.cuda.synchronize()
            self.elapsed = time.perf_counter() - self.started
            self.allocated = torch.cuda.max_memory_allocated()
            self.reserved = torch.cuda.max_memory_reserved()
            self.rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        finally:
            if self.nvml is not None:
                try:
                    self.nvml.nvmlShutdown()
                except Exception as error:
                    self.errors.append(f"NVML shutdown unavailable: {error}")

    def receipt(self) -> dict[str, Any]:
        own_bytes = [
            p["bytes"] for s in self.samples for p in s["own_processes"] if p["bytes"] is not None
        ]
        contended = any(s["foreign_compute_processes"] for s in self.samples)
        return {
            "schema": "rtgs.bench019.local.resources.v1",
            "pid": os.getpid(),
            "wall_seconds": self.elapsed,
            "peak_cuda_allocated_bytes": self.allocated,
            "peak_cuda_reserved_bytes": self.reserved,
            "ru_maxrss_bytes": self.rss,
            "nvml_process_peak_bytes": max(own_bytes) if own_bytes else None,
            "nvml_errors": self.errors,
            "samples": self.samples,
            "contended": contended,
            "timing_scope": "exploratory; no speed claim"
            if contended or self.errors
            else "uncontended worker; lazy imports/JIT/cache setup included; reporting excluded",
        }


def worker_environment(task: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    policy = task["environment_policy"]
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT / "src"), str(ROOT.parent / "structsplat" / "src"), str(ROOT)]
            ),
            "CUDA_HOME": policy["cuda_home"],
            "CXX": policy["compiler"],
            "TORCH_CUDA_ARCH_LIST": policy["architecture"],
            "TORCH_EXTENSIONS_DIR": policy["extension_cache"],
            "MAX_JOBS": str(policy["max_jobs"]),
            "MKL_THREADING_LAYER": "GNU",
            "OMP_NUM_THREADS": str(policy["cpu_threads"]),
            "MKL_NUM_THREADS": str(policy["cpu_threads"]),
            "OPENBLAS_NUM_THREADS": str(policy["cpu_threads"]),
        }
    )
    return env


def dispatch(
    task: dict[str, Any],
    task_path: Path,
    run: Path,
    mode: str,
    frame: str,
    family: str,
    *,
    seed: int = 0,
    replicate: str = "primary",
) -> None:
    label = f"{mode}-{frame}-{family}-{seed}-{replicate}"
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        mode,
        "--task",
        str(task_path),
        "--run-dir",
        str(run),
        "--frame",
        frame,
        "--family",
        family,
        "--seed",
        str(seed),
        "--replicate",
        replicate,
    ]
    log = run / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"start {label}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            env=worker_environment(task),
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=task["execution_limits"]["worker_timeout_seconds"],
        )
    if completed.returncode:
        raise RuntimeError(f"{label} failed with exit {completed.returncode}; see {log}")
    print(f"finished {label}", flush=True)


def input_inventory(run: Path) -> dict[str, str]:
    return {
        p.relative_to(run).as_posix(): sha256(p)
        for p in sorted((run / "inputs").rglob("*"))
        if p.is_file()
    }


def verify_phase2(task: dict[str, Any], run: Path) -> dict[str, Any]:
    from rtgs.bench019 import protocol_identity

    protocol_path = run / "protocol" / "structsplat.frozen.json"
    protocol = read_json(protocol_path)
    protocol_identity(protocol, protocol_base=protocol_path.parent, allow_review=False)
    if Path(protocol["downstream"]["outcome_root"]).resolve() != run / "downstream":
        raise ValueError("Struct protocol does not bind the reserved downstream root")
    task_record = protocol["downstream"]["task_manifest"]
    if task_record["sha256"] != sha256(ROOT / "experiments" / "tasks" / f"{TASK_ID}.json"):
        raise ValueError("Struct protocol does not bind the immutable RTGS task")
    frozen_inputs = read_json(run / "generated_inputs.json")
    verify_file_inventory(run, frozen_inputs["files"])
    if frozen_inputs["files"] != input_inventory(run):
        raise ValueError("generated input inventory differs from the frozen publication")
    downstream = protocol["downstream"]
    expected_artifacts = {
        "dataset_manifest": run / "generated_inputs.json",
        "environment": run / "environment.json",
        "schedule_config": run / "gaussians.config.json",
    }
    for name, path in expected_artifacts.items():
        if downstream[name]["sha256"] != sha256(path):
            raise ValueError(f"Struct protocol does not bind {name}")
    if (
        downstream["seeds"] != task["seeds"]
        or downstream["initializers"] != ["field_sweep"]
        or downstream["command"] != task["run_command"]
        or Path(downstream["working_directory"]).resolve() != ROOT
    ):
        raise ValueError("Struct downstream schedule differs from ready RTGS task")
    frames = {frame["id"]: frame for capture in protocol["captures"] for frame in capture["frames"]}
    if set(frames) != {d["id"] for d in task["datasets"]}:
        raise ValueError("Struct frame inventory differs from RTGS task")
    for frame_id, frame in frames.items():
        if frame["split"] != task["splits"][frame_id]:
            raise ValueError("Struct split differs from RTGS task")
        families = {family["id"]: family for family in frame["families"]}
        if set(families) != {c["id"] for c in task["comparators"]}:
            raise ValueError("Struct family inventory differs from RTGS task")
        for family_id, family in families.items():
            for key, filename in (
                ("field_manifest", "manifest.json"),
                ("stage1_metrics", "stage1_metrics.json"),
            ):
                expected = run / "inputs" / frame_id / family_id / filename
                if family[key]["sha256"] != sha256(expected):
                    raise ValueError("Struct protocol does not bind generated family bytes")
    return protocol


def cell_directory(run: Path, frame: str, family: str, seed: int, replicate: str) -> Path:
    parent = "warmup" if replicate == "warmup" else "downstream"
    return run / parent / frame / family / f"seed_{seed}_{replicate}"


def execute_worker(args: argparse.Namespace) -> None:
    task, _lock = guard(args.task, args.run_dir)
    if args.frame not in task["splits"] or args.family not in {
        c["id"] for c in task["comparators"]
    }:
        raise ValueError("worker frame/family not frozen")
    validate_worker_selection(task, args)
    verify_environment(task)
    if args.mode == "input-worker":
        from rtgs.bench019_local_inputs import produce_frame_family

        output = args.run_dir / "inputs" / args.frame / args.family
        with Resources(task["resource_monitor"]["interval_seconds"]) as monitor:
            produce_frame_family(task, args.frame, args.family, output)
        write_json(output / "resource_receipt.json", monitor.receipt())
    else:
        from rtgs.bench019_local_downstream import evaluate_cell, run_cell

        verify_phase2(task, args.run_dir)
        output = cell_directory(args.run_dir, args.frame, args.family, args.seed, args.replicate)
        if args.mode == "cell-worker":
            with Resources(task["resource_monitor"]["interval_seconds"]) as monitor:
                run_cell(
                    task,
                    args.frame,
                    args.family,
                    args.run_dir / "inputs" / args.frame / args.family,
                    args.seed,
                    output,
                )
            write_json(output / "resource_receipt.json", monitor.receipt())
        elif args.mode == "evaluate-worker":
            evaluate_cell(
                task,
                args.frame,
                args.family,
                args.seed,
                output / "gaussians.npz",
                output / "evaluation",
            )
        elif args.mode == "preview-worker":
            save_previews(task, args.run_dir, args.frame, output)
        else:
            raise ValueError("unknown worker mode")
    guard(args.task, args.run_dir)


def validate_worker_selection(task: dict[str, Any], args: argparse.Namespace) -> None:
    if args.mode == "input-worker":
        if args.seed != 0 or args.replicate != "primary":
            raise ValueError("input worker accepts no downstream seed/replicate")
        return
    if args.mode == "preview-worker":
        if (
            args.family != "native_additive"
            or args.seed != task["seeds"][0]
            or args.replicate != "primary"
        ):
            raise ValueError("preview worker differs from frozen representative")
        return
    if args.replicate == "warmup":
        if args.seed != task["resource_monitor"]["warmup_seed"] or args.mode != "cell-worker":
            raise ValueError("warmup seed/mode differs from frozen policy")
    elif args.replicate == "aa":
        policy = task["aa_replay"]
        if (
            args.seed != policy["seed"]
            or args.frame != policy["frame_id"]
            or args.family not in (policy["family_id"], policy["additional_family_id"])
        ):
            raise ValueError("A/A worker outside frozen policy")
    elif args.replicate != "primary" or args.seed not in task["seeds"]:
        raise ValueError("primary worker outside frozen seeds")


def save_previews(task: dict[str, Any], run: Path, frame: str, cell: Path) -> None:
    from rtgs.bench019_local_downstream import _load_scene
    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.visualize import save_reconstruction_artifacts

    output = run if frame == task["datasets"][0]["id"] else run / "previews" / frame
    started = time.perf_counter()
    scene = _load_scene(task, frame, split="heldout")
    initial = Gaussians3D.load_npz(cell / "checkpoint_0000.npz").to("cuda:0")
    final = Gaussians3D.load_npz(cell / "gaussians.npz").to("cuda:0")
    artifacts = save_reconstruction_artifacts(
        scene,
        initial,
        final,
        output,
        rasterizer="gsplat",
        max_comparisons=3,
        max_animation_frames=16,
    )
    write_json(
        output / "preview_receipt.json",
        {
            "scope": "diagnostic only; frozen native primary seed19001",
            "frame_id": frame,
            "cell": str(cell.relative_to(run)),
            "seconds": time.perf_counter() - started,
            "artifacts": {
                name: {"path": str(Path(path).relative_to(run)), "sha256": sha256(Path(path))}
                for name, path in artifacts.items()
            },
        },
    )


def aa_check(task: dict[str, Any], run: Path) -> None:
    policy = task["aa_replay"]
    records = []
    for family in (policy["family_id"], policy["additional_family_id"]):
        directories = [
            cell_directory(run, policy["frame_id"], family, policy["seed"], replicate)
            for replicate in ("primary", "aa")
        ]
        base, replay = [
            read_json(directory / "evaluation" / "evaluation.json")["metrics"]
            for directory in directories
        ]
        receipts = [read_json(directory / "receipt.json") for directory in directories]
        configurations = [read_json(directory / "config.json") for directory in directories]
        if configurations[0] != configurations[1] or receipts[0]["fields"] != receipts[1]["fields"]:
            raise RuntimeError("A/A configuration or field binding mismatch")
        for metric, tolerance in (
            ("heldout_foreground_psnr", policy["foreground_psnr_absolute_tolerance_db"]),
            ("heldout_alpha_iou", policy["alpha_iou_absolute_tolerance"]),
        ):
            delta = abs(float(base[metric]) - float(replay[metric]))
            if not math.isfinite(delta) or delta > tolerance:
                raise RuntimeError(f"A/A {family} {metric} failed: {delta} > {tolerance}")
            records.append(
                {
                    "family": family,
                    "metric": metric,
                    "absolute_delta": delta,
                    "tolerance": tolerance,
                }
            )
        if any(
            receipt["metrics"]["final_gaussians"] != task["downstream_config"]["n_init_3d"]
            for receipt in receipts
        ):
            raise RuntimeError("A/A count mismatch")
    write_json(run / "aa_replay.json", {"status": "passed", "checks": records})


def run_experiment(args: argparse.Namespace) -> None:
    task, lock = guard(args.task, args.run_dir)
    run = args.run_dir
    write_json(run / "environment.json", verify_environment(task))
    write_json(run / "gaussians.config.json", task)
    for dataset in task["datasets"]:
        for family in task["comparators"]:
            dispatch(task, args.task, run, "input-worker", dataset["id"], family["id"])
    write_json(
        run / "generated_inputs.json",
        {
            "schema": "rtgs.bench019.local.generated_inputs.v1",
            "task_sha256": lock["task_sha256"],
            "files": input_inventory(run),
        },
    )
    write_json(
        run / "phase1_ready.json",
        {
            "state": "awaiting_independent_structsplat_protocol",
            "outcome_root": str(run / "downstream"),
            "generated_inputs_sha256": sha256(run / "generated_inputs.json"),
        },
    )
    print("phase1 complete; awaiting independently finalized StructSplat protocol", flush=True)
    deadline = time.monotonic() + task["execution_limits"]["phase2_review_timeout_seconds"]
    while not (run / "protocol" / "structsplat.frozen.json").is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "independent phase2 protocol was not supplied before the frozen timeout"
            )
        time.sleep(2.0)
    verify_phase2(task, run)
    warmup = task["resource_monitor"]["warmup_seed"]
    for dataset in task["datasets"]:
        for family in task["comparators"]:
            dispatch(
                task,
                args.task,
                run,
                "cell-worker",
                dataset["id"],
                family["id"],
                seed=warmup,
                replicate="warmup",
            )
    policy = task["aa_replay"]
    primaries_done = set()
    for family in (policy["family_id"], policy["additional_family_id"]):
        for replicate in ("primary", "aa"):
            dispatch(
                task,
                args.task,
                run,
                "cell-worker",
                policy["frame_id"],
                family,
                seed=policy["seed"],
                replicate=replicate,
            )
            dispatch(
                task,
                args.task,
                run,
                "evaluate-worker",
                policy["frame_id"],
                family,
                seed=policy["seed"],
                replicate=replicate,
            )
        primaries_done.add((policy["frame_id"], family, policy["seed"]))
    aa_check(task, run)
    for dataset in task["datasets"]:
        for seed in task["seeds"]:
            for family in task["comparators"]:
                key = (dataset["id"], family["id"], seed)
                if key in primaries_done:
                    continue
                dispatch(task, args.task, run, "cell-worker", *key[:2], seed=seed)
                dispatch(task, args.task, run, "evaluate-worker", *key[:2], seed=seed)
    guard(args.task, run)
    verify_phase2(task, run)
    for dataset in task["datasets"]:
        dispatch(
            task,
            args.task,
            run,
            "preview-worker",
            dataset["id"],
            "native_additive",
            seed=task["seeds"][0],
        )
    publish_result_sources(task, lock, run)


def publish_result_sources(task: dict[str, Any], lock: dict[str, Any], run: Path) -> None:
    from rtgs.bench019_local_report import publish_result_sources as publish

    publish(task, run, run / "protocol" / "structsplat.frozen.json")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "mode", choices=["run", "input-worker", "cell-worker", "evaluate-worker", "preview-worker"]
    )
    value.add_argument("--task", type=Path, required=True)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--frame")
    value.add_argument("--family")
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--replicate", choices=["primary", "aa", "warmup"], default="primary")
    return value


def main() -> int:
    args = parser().parse_args()
    args.task = args.task.resolve()
    args.run_dir = args.run_dir.absolute()
    try:
        if args.mode == "run":
            run_experiment(args)
        else:
            execute_worker(args)
        return 0
    except Exception as error:
        # Resolve the canonical identity before publishing even an early preflight failure.
        try:
            # The current task may itself be missing or malformed: the fixed canonical
            # path and initialized lock are sufficient to receipt that first failure.
            run = canonical_run({"task_id": TASK_ID}, args.run_dir)
            if run.is_dir():
                failure = {
                    "status": "failed",
                    "mode": args.mode,
                    "frame": args.frame,
                    "family": args.family,
                    "seed": args.seed,
                    "replicate": args.replicate,
                    "exception": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                label = f"{args.mode}-{args.frame}-{args.family}-{args.seed}-{args.replicate}"
                write_json(run / "failures" / f"{label}.json", failure)
                if args.mode == "run":
                    lock = read_json(run / "task.lock.json")
                    write_json(
                        run / "run_receipt.json",
                        {
                            "schema_version": 1,
                            "task_id": TASK_ID,
                            "status": "failed",
                            "started_at_utc": lock["started_at_utc"],
                            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "exit_code": 1,
                            "failure_phase": args.mode,
                            "message": str(error),
                        },
                    )
                    from rtgs.bench019_local_report import publish_failure_sources

                    try:
                        failure_task = read_json(args.task)
                    except (ValueError, OSError):
                        failure_task = None
                    publish_failure_sources(failure_task, run, error)
        except Exception as publishing_error:
            print(f"failure receipt unavailable: {publishing_error}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
