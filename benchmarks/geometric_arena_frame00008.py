#!/usr/bin/env python3
"""Compare dynamic gsplat storage with the geometric Stage-3 arena on Janelle."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from benchmarks import pool_structure_wse_10k_frame00008 as prior
except ModuleNotFoundError:
    import pool_structure_wse_10k_frame00008 as prior

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
RAW_SCENE = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
INITIALIZATION = (
    ROOT / "runs/pool_structure_wse_frame00008_20260724/models/"
    "pool-structure-wse/gaussians_init.npz"
)
INITIALIZATION_SHA256 = "f0e41c4c57289f08c8b7101898c1f06192e0b2085b10bb877d8a315e97971abb"

DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_PREREG.md"
DEFAULT_OUT = ROOT / "runs/geometric_arena_frame00008_20260724"
VIEWER_MANIFEST = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_VIEWER.json"
RESULT_NOTE = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_RESULT.md"
AUDIT_JSON = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_AUDIT.json"
AUDIT_NOTE = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_AUDIT.md"

ARMS = ("dynamic-a", "geometric", "dynamic-b")
STORAGE_POLICY = {
    "dynamic-a": "dynamic",
    "geometric": "geometric",
    "dynamic-b": "dynamic",
}
COLORS = {
    "dynamic-a": "#68d5c4",
    "geometric": "#ffca6e",
    "dynamic-b": "#9fa8ff",
}
CHECKPOINT_STEPS = (2_000, 4_000, 6_000, 8_000, 10_000)
TRAIN_VIEW = "C0014"
HELDOUT_VIEW = "C1004"

SOURCE_FILES = (
    "benchmarks/geometric_arena_frame00008.py",
    "benchmarks/pool_structure_wse_10k_frame00008.py",
    "benchmarks/new_variants_frame00008.py",
    "benchmarks/results/20260724_geometric_arena_frame00008_PREREG.md",
    "src/rtgs/core/gaussians3d.py",
    "src/rtgs/core/metrics.py",
    "src/rtgs/data/calibrated.py",
    "src/rtgs/optim/arena.py",
    "src/rtgs/optim/trainer.py",
    "src/rtgs/optim/strategies.py",
    "src/rtgs/optim/density.py",
    "src/rtgs/render/gsplat_backend.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _command_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _repository_binding() -> dict[str, Any]:
    hashes = {relative: _sha256(ROOT / relative) for relative in SOURCE_FILES}
    status = _command_output("git", "status", "--porcelain=v1")
    tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT)
    return {
        "git_revision": _command_output("git", "rev-parse", "HEAD"),
        "git_branch": _command_output("git", "branch", "--show-current"),
        "git_status_lines": status.splitlines(),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_files": hashes,
        "source_aggregate_sha256": _canonical_hash(hashes),
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("gsplat", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    allocator = None
    if torch.cuda.is_available():
        allocator = torch.cuda.get_allocator_backend()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "allocator_backend": allocator,
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "packages": packages,
        "ld_preload": os.environ.get("LD_PRELOAD"),
    }


def _gpu_process_snapshot() -> dict[str, Any]:
    command = (
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    try:
        output = _command_output(*command)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"command": list(command), "available": False, "error": str(error), "rows": []}
    return {
        "command": list(command),
        "available": True,
        "rows": [line for line in output.splitlines() if line.strip()],
    }


def _scene():
    scene = load_calibrated_scene(
        RAW_SCENE,
        calibration_path=CALIBRATION,
        downscale=16,
        max_images=8,
        test_every=8,
        load_masks=True,
        undistort=True,
    )
    expected = ["C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039", "C1004"]
    if scene.view_names != expected:
        raise RuntimeError(f"calibrated view order drifted: {scene.view_names}")
    if scene.training_views != list(range(7)) or scene.testing_views != [7]:
        raise RuntimeError("frozen seven-train/one-test split drifted")
    if scene.masks is None:
        raise RuntimeError("complete masks are required")
    return scene


def _config(arm: str) -> TrainConfig:
    return dataclasses.replace(
        prior._train_config(),
        gaussian_storage_policy=STORAGE_POLICY[arm],
        profile_density_events=True,
        arena_growth_factor=2.0,
        arena_initial_capacity=None,
    )


def _field_hashes(model: Gaussians3D) -> dict[str, str]:
    return {
        name: prior._tensor_hash(getattr(model, name))
        for name in ("means", "quats", "log_scales", "opacity", "sh")
    }


def _worker(arm: str, out: Path, protocol: Path) -> int:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not (out / "plan.json").is_file():
        raise FileNotFoundError("worker requires the parent plan.json")
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    if _sha256(protocol) != plan["protocol"]["sha256"]:
        raise RuntimeError("protocol changed before worker execution")
    current_sources = {relative: _sha256(ROOT / relative) for relative in SOURCE_FILES}
    if current_sources != plan["repository"]["source_files"]:
        raise RuntimeError("bound source changed before worker execution")
    if _sha256(INITIALIZATION) != INITIALIZATION_SHA256:
        raise RuntimeError("initialization hash drifted")

    model_dir = out / "models" / arm
    if model_dir.exists():
        raise FileExistsError(f"refusing to overwrite worker output: {model_dir}")
    model_dir.mkdir(parents=True)
    scene = _scene()
    initial = Gaussians3D.load_npz(INITIALIZATION)
    if initial.n != 422:
        raise RuntimeError(f"frozen initialization count drifted: {initial.n}")
    initial_artifacts = prior._save_model(model_dir / "gaussians_init", initial)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    capture = prior._CheckpointCapture(CHECKPOINT_STEPS)
    config = _config(arm)
    returned, history = Trainer(config).train(
        scene,
        initial.to("cuda"),
        checkpoint_callback=capture,
    )
    returned_cpu = returned.to("cpu")
    if tuple(sorted(capture.snapshots)) != CHECKPOINT_STEPS:
        raise RuntimeError(f"{arm} checkpoint set drifted")
    if not prior._all_fields_equal(returned_cpu, capture.snapshots[10_000]):
        raise RuntimeError(f"{arm} returned final differs from captured 10k state")

    history_path = model_dir / "training_history.json"
    _write_json(history_path, history)
    checkpoints = {}
    for step in CHECKPOINT_STEPS:
        model = capture.snapshots[step]
        checkpoints[str(step)] = {
            "step": step,
            "n_gaussians": model.n,
            "field_sha256": _field_hashes(model),
            "artifacts": prior._save_model(
                model_dir / "checkpoints" / f"gaussians_step_{step:05d}",
                model,
            ),
        }
    final_artifacts = prior._save_model(model_dir / "gaussians_final", returned_cpu)
    receipt = {
        "schema": "rtgs.geometric_arena_frame00008.worker.v1",
        "completed_utc": dt.datetime.now(dt.UTC).isoformat(),
        "arm": arm,
        "storage_policy": STORAGE_POLICY[arm],
        "config": dataclasses.asdict(config),
        "initial_artifacts": initial_artifacts,
        "training_history": _artifact(history_path),
        "checkpoints": checkpoints,
        "final_n_gaussians": returned_cpu.n,
        "final_field_sha256": _field_hashes(returned_cpu),
        "final_artifacts": final_artifacts,
        "gpu_processes_after": _gpu_process_snapshot(),
    }
    _write_json(model_dir / "worker.json", receipt)
    print(f"[worker-complete] {arm}: N={returned_cpu.n}", flush=True)
    return 0


def _history_record(history: dict[str, Any]) -> dict[str, Any]:
    event_seconds = [
        float(item["event_seconds"]) for item in history["density_stats"] if "event_seconds" in item
    ]
    total = float(history["elapsed"][-1][1])
    event_total = sum(event_seconds)
    return {
        "native_elapsed_seconds": total,
        "density_event_seconds": event_seconds,
        "density_event_total_seconds": event_total,
        "approx_non_density_seconds": total - event_total,
        "peak_allocated_gib": float(history["peak_vram_gb"]),
        "peak_reserved_gib": float(history["peak_vram_reserved_gb"]),
        "cuda_memory_stats": history["cuda_memory_stats"],
        "storage_diagnostics": history["storage_diagnostics"],
        "sampled_train_views_sha256": prior._canonical_hash(history["sampled_train_views"]),
        "density_trajectory": [
            {
                "iteration": int(item["iteration"]),
                "n_before": int(item["n_before"]),
                "n_after": int(item["n_after"]),
            }
            for item in history["density_stats"]
        ],
    }


def _comparison_sheet(
    path: Path,
    scene,
    checkpoints: dict[str, dict[int, Gaussians3D]],
    initial: Gaussians3D,
    view_index: int,
    renderer,
) -> None:
    target = scene.images[view_index]
    if scene.masks is not None:
        target = target * scene.masks[view_index][..., None]
    camera = scene.cameras[view_index].to("cuda")
    rows = []
    with torch.no_grad():
        for arm in ARMS:
            panels = [prior.common._labeled_panel(target, f"{arm} · target")]
            init_render = renderer.render(initial.to("cuda"), camera).color.clamp(0, 1)
            panels.append(prior.common._labeled_panel(init_render, f"{arm} · init"))
            for step in CHECKPOINT_STEPS:
                render = renderer.render(
                    checkpoints[arm][step].to("cuda"),
                    camera,
                ).color.clamp(0, 1)
                panels.append(prior.common._labeled_panel(render, f"{arm} · {step // 1000}k"))
            rows.append(prior.common._horizontal_sheet(panels))
    prior.common._vertical_sheet(rows).save(path)


def _write_viewer_manifest(out: Path) -> None:
    methods = []
    for arm in ARMS:
        initial = out / "models" / arm / "gaussians_init.ply"
        for step in CHECKPOINT_STEPS:
            final = out / "models" / arm / "checkpoints" / f"gaussians_step_{step:05d}.ply"
            methods.append(
                {
                    "name": f"{arm} @ {step // 1000}k",
                    "initial": os.path.relpath(initial, VIEWER_MANIFEST.parent),
                    "final": os.path.relpath(final, VIEWER_MANIFEST.parent),
                }
            )
    _write_json(
        VIEWER_MANIFEST,
        {"schema": "rtgs.viewer-comparison.v1", "methods": methods},
    )


def _decision(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    controls = ("dynamic-a", "dynamic-b")
    histories = {arm: arms[arm]["performance"] for arm in ARMS}
    trajectories = [histories[arm]["density_trajectory"] for arm in ARMS]
    view_hashes = [histories[arm]["sampled_train_views_sha256"] for arm in ARMS]
    dynamic_final_counts = [arms[arm]["final_n_gaussians"] for arm in controls]
    validity = {
        "sampled_views_match": len(set(view_hashes)) == 1,
        "event_trajectories_match": trajectories[0] == trajectories[1] == trajectories[2],
        "dynamic_final_counts_match": len(set(dynamic_final_counts)) == 1,
    }
    final_metrics = {arm: arms[arm]["checkpoints"]["10000"]["metrics"] for arm in ARMS}

    def envelope(metric_path: tuple[str, str], margin: float) -> tuple[float, float]:
        values = [float(final_metrics[arm][metric_path[0]][metric_path[1]]) for arm in controls]
        return min(values) - margin, max(values) + margin

    train_low, train_high = envelope(("train", "psnr_fg"), 0.02)
    test_low, test_high = envelope(("test", "psnr_fg"), 0.02)
    alpha_low, alpha_high = envelope(("test", "alpha_iou"), 0.002)
    geometric_final = final_metrics["geometric"]
    correctness = {
        "count_matches": arms["geometric"]["final_n_gaussians"] == dynamic_final_counts[0],
        "train_psnr_in_envelope": train_low
        <= float(geometric_final["train"]["psnr_fg"])
        <= train_high,
        "test_psnr_in_envelope": test_low <= float(geometric_final["test"]["psnr_fg"]) <= test_high,
        "test_alpha_iou_in_envelope": alpha_low
        <= float(geometric_final["test"]["alpha_iou"])
        <= alpha_high,
    }
    correctness_pass = all(validity.values()) and all(correctness.values())
    dynamic_event_fastest = min(histories[arm]["density_event_total_seconds"] for arm in controls)
    dynamic_total_fastest = min(histories[arm]["native_elapsed_seconds"] for arm in controls)
    dynamic_non_density_worst = max(
        histories[arm]["approx_non_density_seconds"] for arm in controls
    )
    dynamic_allocated_worst = max(histories[arm]["peak_allocated_gib"] for arm in controls)
    dynamic_reserved_worst = max(histories[arm]["peak_reserved_gib"] for arm in controls)
    geometric = histories["geometric"]
    performance = {
        "density_event_ratio_to_faster_dynamic": (
            geometric["density_event_total_seconds"] / dynamic_event_fastest
        ),
        "native_elapsed_ratio_to_faster_dynamic": (
            geometric["native_elapsed_seconds"] / dynamic_total_fastest
        ),
        "non_density_ratio_to_worse_dynamic": (
            geometric["approx_non_density_seconds"] / dynamic_non_density_worst
        ),
        "peak_allocated_ratio_to_worse_dynamic": (
            geometric["peak_allocated_gib"] / dynamic_allocated_worst
        ),
        "peak_reserved_ratio_to_worse_dynamic": (
            geometric["peak_reserved_gib"] / dynamic_reserved_worst
        ),
    }
    gates = {
        "density_event_win": performance["density_event_ratio_to_faster_dynamic"] <= 0.80,
        "end_to_end_win": performance["native_elapsed_ratio_to_faster_dynamic"] <= 0.98,
        "non_density_noninferior": performance["non_density_ratio_to_worse_dynamic"] <= 1.02,
        "peak_allocated_noninferior": (
            performance["peak_allocated_ratio_to_worse_dynamic"] <= 1.10
        ),
        "peak_reserved_noninferior": (performance["peak_reserved_ratio_to_worse_dynamic"] <= 1.10),
    }
    dynamic_ab_psnr_delta = abs(
        float(final_metrics["dynamic-a"]["test"]["psnr_fg"])
        - float(final_metrics["dynamic-b"]["test"]["psnr_fg"])
    )
    if not correctness_pass:
        disposition = "REJECT_CURRENT_ARENA_CORRECTNESS"
    elif gates["density_event_win"]:
        disposition = "RETAIN_OPT_IN_AUTHORIZE_SCALING_STUDY"
    else:
        disposition = "NEGATIVE_SYSTEMS_RESULT_KEEP_DYNAMIC_DEFAULT"
    return {
        "validity": validity,
        "correctness": correctness,
        "correctness_pass": correctness_pass,
        "performance": performance,
        "gates": gates,
        "dynamic_ab_heldout_psnr_delta_db": dynamic_ab_psnr_delta,
        "dynamic_ab_timing_descriptive": dynamic_ab_psnr_delta > 0.02,
        "disposition": disposition,
        "default_change_authorized": False,
    }


def _index_html(summary: dict[str, Any], out: Path) -> str:
    arms = summary["arms"]
    decision = summary["decision"]
    metric_rows = []
    for arm in ARMS:
        for step in CHECKPOINT_STEPS:
            record = arms[arm]["checkpoints"][str(step)]
            metrics = record["metrics"]
            metric_rows.append(
                "<tr>"
                f'<td><span class="dot" style="--c:{COLORS[arm]}"></span>{arm}</td>'
                f"<td>{step // 1000}k</td><td>{record['n_gaussians']:,}</td>"
                f"<td>{metrics['train']['psnr_fg']:.4f}</td>"
                f"<td>{metrics['test']['psnr_fg']:.4f}</td>"
                f"<td>{metrics['test']['alpha_iou']:.5f}</td></tr>"
            )
    performance_rows = []
    for arm in ARMS:
        perf = arms[arm]["performance"]
        storage = perf["storage_diagnostics"]
        capacity = "dynamic" if storage is None else f"{storage['capacity']:,}"
        migrations = "—" if storage is None else str(storage["migration_count"])
        performance_rows.append(
            "<tr>"
            f'<td><span class="dot" style="--c:{COLORS[arm]}"></span>{arm}</td>'
            f"<td>{perf['native_elapsed_seconds']:.4f}</td>"
            f"<td>{perf['density_event_total_seconds']:.5f}</td>"
            f"<td>{perf['approx_non_density_seconds']:.4f}</td>"
            f"<td>{perf['peak_allocated_gib'] * 1024:.2f}</td>"
            f"<td>{perf['peak_reserved_gib'] * 1024:.2f}</td>"
            f"<td>{capacity}</td><td>{migrations}</td></tr>"
        )
    cards = []
    for arm in ARMS:
        for step in CHECKPOINT_STEPS:
            record = arms[arm]["checkpoints"][str(step)]
            visual = ROOT / record["visuals"]["heldout"]["path"]
            ply = ROOT / record["artifacts"]["ply"]["path"]
            visual_rel = os.path.relpath(visual, out)
            ply_rel = os.path.relpath(ply, out)
            cards.append(
                '<article class="card">'
                f"<header><strong>{arm} · {step // 1000}k</strong>"
                f"<span>{record['metrics']['test']['psnr_fg']:.3f} dB</span></header>"
                f'<a href="{html.escape(visual_rel)}"><img loading="lazy" '
                f'src="{html.escape(visual_rel)}" alt="{arm} at {step} steps"></a>'
                f'<footer><a href="{html.escape(ply_rel)}">PLY</a></footer></article>'
            )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Janelle geometric Stage-3 arena experiment</title>
  <style>
    :root {{ color-scheme:dark; --bg:#090d12; --panel:#121a24; --line:#2a3749;
      --text:#eff5fc; --muted:#9eacbf; --accent:#68d5c4; --warm:#ffca6e; }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:radial-gradient(circle at 15% 0,
      #213449,var(--bg) 40rem); color:var(--text); font:15px/1.5 system-ui,sans-serif }}
    a {{ color:var(--accent) }} .shell {{ width:min(1500px,calc(100% - 32px)); margin:auto }}
    .hero {{ padding:48px 0 20px }} .eyebrow {{ color:var(--accent); text-transform:uppercase;
      font-size:12px; font-weight:800; letter-spacing:.14em }} h1 {{ font-size:clamp(32px,5vw,58px);
      line-height:1.03; margin:7px 0 }} .lead,.muted {{ color:var(--muted) }}
    .decision {{ padding:14px 16px; background:#201c12; border:1px solid #675834;
      border-radius:12px }} nav {{ position:sticky; top:0; z-index:3; display:flex; gap:15px;
      flex-wrap:wrap; padding:12px 0; background:rgba(9,13,18,.92); backdrop-filter:blur(12px) }}
    section {{ margin:28px 0 }} .panel {{ overflow:auto; padding:17px; border:1px solid var(--line);
      background:rgba(18,26,36,.93); border-radius:14px }} table {{ width:100%;
      border-collapse:collapse; font-variant-numeric:tabular-nums }} th,td {{ padding:9px 11px;
      border-bottom:1px solid var(--line); text-align:right; white-space:nowrap }}
    th:first-child,td:first-child {{ text-align:left }} .dot {{ display:inline-block; width:10px;
      height:10px; margin-right:8px; border-radius:50%; background:var(--c) }}
    .sheet {{ width:100%; display:block; border-radius:10px; background:#000 }}
    .grid {{ display:grid; grid-template-columns:repeat(5,minmax(220px,1fr)); gap:10px;
      overflow-x:auto }} .card {{ min-width:220px; border:1px solid var(--line);
      background:var(--panel); border-radius:12px; overflow:hidden }} .card header {{
      display:flex; justify-content:space-between; padding:10px 12px; gap:8px }}
    .card header span,.card footer {{ color:var(--muted); font-size:12px }}
    .card img {{ display:block; width:100%; background:#000 }} .card footer {{ padding:8px 12px }}
    .links {{ display:flex; flex-wrap:wrap; gap:15px }} footer.page {{ padding:25px 0 42px;
      border-top:1px solid var(--line); color:var(--muted) }}
  </style>
</head>
<body><main class="shell">
  <section class="hero"><div class="eyebrow">realtime-gs · storage-only systems experiment</div>
    <h1>Geometric Stage-3 Gaussian arena</h1>
    <p class="lead">Fresh-process dynamic A → geometric arena → dynamic B comparison on the
    audited pool+structure+WSE Janelle initialization. Every arm uses the same 10k training
    schedule and reporting-only held-out camera C1004.</p>
    <p class="decision"><strong>Preregistered disposition:</strong>
      {html.escape(decision["disposition"])}. Independent audit remains authoritative.</p>
  </section>
  <nav><a href="#performance">Performance</a><a href="#metrics">Quality</a>
    <a href="#visuals">Visuals</a><a href="summary.json">summary.json</a>
    <a href="../../benchmarks/results/20260724_geometric_arena_frame00008_RESULT.md">result</a>
    <a href="../../benchmarks/results/20260724_geometric_arena_frame00008_AUDIT.md">audit</a>
    <a href="../../benchmarks/results/20260724_geometric_arena_frame00008_VIEWER.json">viewer</a>
  </nav>
  <section id="performance"><h2>Timing and memory</h2><div class="panel"><table>
    <thead><tr><th>Arm</th><th>Native 10k s</th><th>Density events s</th>
      <th>Approx. other s</th><th>Peak allocated MiB</th><th>Peak reserved MiB</th>
      <th>Final capacity</th><th>Migrations</th></tr></thead>
    <tbody>{"".join(performance_rows)}</tbody></table>
    <p class="muted">Density events use explicit CUDA synchronization. “Approx. other” is native
      elapsed minus their measured total; checkpoint serialization is excluded.</p></div></section>
  <section><h2>Preregistered gates</h2><div class="panel"><pre>{
        html.escape(json.dumps(decision, indent=2))
    }</pre></div></section>
  <section id="metrics"><h2>Exact saved-checkpoint metrics</h2><div class="panel"><table>
    <thead><tr><th>Arm</th><th>Step</th><th>N</th><th>Train FG dB</th>
      <th>Held-out FG dB</th><th>Held-out α-IoU</th></tr></thead>
    <tbody>{"".join(metric_rows)}</tbody></table></div></section>
  <section><h2>Cross-arm checkpoint sheets</h2><div class="panel">
    <h3>Held-out C1004</h3><a href="checkpoint_heldout_C1004.png">
      <img class="sheet" src="checkpoint_heldout_C1004.png" alt="Held-out checkpoints"></a>
    <h3>Training C0014</h3><a href="checkpoint_train_C0014.png">
      <img class="sheet" src="checkpoint_train_C0014.png" alt="Training checkpoints"></a>
  </div></section>
  <section id="visuals"><h2>Held-out checkpoint details</h2>
    <div class="grid">{"".join(cards)}</div></section>
  <section><h2>Provenance</h2><div class="panel links">
    <a href="../../benchmarks/results/20260724_geometric_arena_frame00008_PREREG.md">
      frozen protocol</a>
    <a href="plan.json">plan</a>
    <a href="../pool_structure_wse_10k_frame00008_20260724/index.html">parent experiment</a>
  </div></section>
  <footer class="page">Generated from exact saved artifacts. Visuals are diagnostic; frozen
    metrics and the independent audit determine the interpretation.</footer>
</main></body></html>"""


def _result_markdown(summary: dict[str, Any]) -> str:
    arms = summary["arms"]
    lines = [
        "# Geometric Stage-3 arena — Janelle `frame_00008`",
        "",
        "This note reports the frozen single-scene storage experiment. The adjacent independent "
        "audit controls the final interpretation.",
        "",
        "| arm | final N | native 10k s | density events s | peak allocated MiB | "
        "held-out FG PSNR | α-IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        perf = arms[arm]["performance"]
        metrics = arms[arm]["checkpoints"]["10000"]["metrics"]["test"]
        lines.append(
            f"| {arm} | {arms[arm]['final_n_gaussians']} | "
            f"{perf['native_elapsed_seconds']:.6f} | "
            f"{perf['density_event_total_seconds']:.6f} | "
            f"{perf['peak_allocated_gib'] * 1024:.3f} | "
            f"{metrics['psnr_fg']:.6f} | {metrics['alpha_iou']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen decision reduction",
            "",
            f"- Disposition: `{summary['decision']['disposition']}`",
            f"- Storage correctness: `{summary['decision']['correctness_pass']}`",
            f"- Density-event material win: `{summary['decision']['gates']['density_event_win']}`",
            f"- End-to-end win: `{summary['decision']['gates']['end_to_end_win']}`",
            "- Default change authorized: `False`",
            "",
            "## Evidence boundary",
            "",
            "One scene, one seed, one device, and two dynamic timing controls do not establish a "
            "general performance claim. Held-out C1004 was reporting-only. The arena currently "
            "covers `gsplat-default`; it does not test fixed-budget MCMC relocation.",
            "",
            f"- Results page: `{summary['index_html']['path']}`",
            f"- Viewer manifest: `{summary['viewer_manifest']['path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _parent(out: Path, protocol: Path) -> int:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {out}")
    if not protocol.is_file():
        raise FileNotFoundError(f"frozen protocol is missing: {protocol}")
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen experiment requires CUDA")
    if _sha256(INITIALIZATION) != INITIALIZATION_SHA256:
        raise RuntimeError("frozen initialization hash drifted")

    scene = _scene()
    repository = _repository_binding()
    out.mkdir(parents=True)
    input_records = {}
    for name, image, mask in zip(scene.view_names, scene.images, scene.masks, strict=True):
        rgb = prior._input_path(name, "rgb")
        mask_path = prior._input_path(name, "mask")
        input_records[name] = {
            "rgb": _artifact(rgb),
            "mask": _artifact(mask_path),
            "loaded_rgb_tensor_sha256": prior._tensor_hash(image),
            "loaded_mask_tensor_sha256": prior._tensor_hash(mask),
        }
    plan = {
        "schema": "rtgs.geometric_arena_frame00008.plan.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "scope": (
            "single-scene, single-seed, fresh-process storage comparison; C1004 "
            "reporting-only; no default or general performance claim"
        ),
        "protocol": _artifact(protocol),
        "repository": repository,
        "environment": _environment(),
        "gpu_processes_before": _gpu_process_snapshot(),
        "initialization": _artifact(INITIALIZATION),
        "scene": {
            "raw_path": str(RAW_SCENE),
            "calibration": _artifact(CALIBRATION),
            "view_names": scene.view_names,
            "train_indices": scene.training_views,
            "test_indices": scene.testing_views,
            "resolution": [scene.cameras[0].width, scene.cameras[0].height],
            "inputs": input_records,
        },
        "arms": list(ARMS),
        "execution_order": list(ARMS),
        "storage_policy": STORAGE_POLICY,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "training_by_arm": {arm: dataclasses.asdict(_config(arm)) for arm in ARMS},
        "worker_command": [
            sys.executable,
            "benchmarks/geometric_arena_frame00008.py",
            "--worker-arm",
            "<arm>",
            "--out",
            str(out),
            "--protocol",
            str(protocol),
        ],
        "index_html_required": True,
    }
    plan_path = out / "plan.json"
    _write_json(plan_path, plan)

    logs = out / "logs"
    logs.mkdir()
    for arm in ARMS:
        print(f"[launch] {arm}", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-arm",
            arm,
            "--out",
            str(out),
            "--protocol",
            str(protocol),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = logs / f"{arm}.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        print(completed.stdout, end="", flush=True)
        if completed.returncode:
            raise RuntimeError(f"worker {arm} failed; see {log_path}")

    renderer = get_rasterizer(
        "gsplat",
        device=torch.device("cuda"),
        packed=False,
        antialiased=True,
    )
    initial = Gaussians3D.load_npz(INITIALIZATION)
    checkpoints: dict[str, dict[int, Gaussians3D]] = {}
    arm_records: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        model_dir = out / "models" / arm
        worker_path = model_dir / "worker.json"
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        history_path = ROOT / worker["training_history"]["path"]
        history = json.loads(history_path.read_text(encoding="utf-8"))
        checkpoints[arm] = {}
        checkpoint_records = {}
        train_frames = []
        heldout_frames = []
        for step in CHECKPOINT_STEPS:
            record = worker["checkpoints"][str(step)]
            npz_path = ROOT / record["artifacts"]["npz"]["path"]
            model = Gaussians3D.load_npz(npz_path)
            checkpoints[arm][step] = model
            metrics = prior._evaluate(scene, model, renderer)
            visual_dir = model_dir / "checkpoint_visuals" / f"step_{step:05d}"
            train_visual = visual_dir / f"train_{TRAIN_VIEW}.png"
            heldout_visual = visual_dir / f"heldout_{HELDOUT_VIEW}.png"
            prior._triptych(
                train_visual,
                scene,
                model,
                scene.view_names.index(TRAIN_VIEW),
                renderer,
                f"{arm} · {step // 1000}k · train",
            )
            prior._triptych(
                heldout_visual,
                scene,
                model,
                scene.view_names.index(HELDOUT_VIEW),
                renderer,
                f"{arm} · {step // 1000}k · held-out",
            )
            train_frames.append(train_visual)
            heldout_frames.append(heldout_visual)
            checkpoint_records[str(step)] = {
                **record,
                "metrics": metrics,
                "visuals": {
                    "train": _artifact(train_visual),
                    "heldout": _artifact(heldout_visual),
                },
            }
        train_gif = model_dir / f"checkpoint_train_{TRAIN_VIEW}.gif"
        heldout_gif = model_dir / f"checkpoint_heldout_{HELDOUT_VIEW}.gif"
        prior._progress_gif(train_gif, train_frames)
        prior._progress_gif(heldout_gif, heldout_frames)
        arm_records[arm] = {
            "storage_policy": STORAGE_POLICY[arm],
            "worker": _artifact(worker_path),
            "initial_artifacts": worker["initial_artifacts"],
            "training_history": worker["training_history"],
            "performance": _history_record(history),
            "checkpoints": checkpoint_records,
            "final_n_gaussians": worker["final_n_gaussians"],
            "final_field_sha256": worker["final_field_sha256"],
            "final_artifacts": worker["final_artifacts"],
            "progress_visuals": {
                "train_gif": _artifact(train_gif),
                "heldout_gif": _artifact(heldout_gif),
            },
            "worker_gpu_processes_after": worker["gpu_processes_after"],
        }

    field_differences = {}
    final_models = {
        arm: Gaussians3D.load_npz(ROOT / arm_records[arm]["final_artifacts"]["npz"]["path"])
        for arm in ARMS
    }
    for control in ("dynamic-a", "dynamic-b"):
        fields = {}
        for name in ("means", "quats", "log_scales", "opacity", "sh"):
            left = getattr(final_models["geometric"], name)
            right = getattr(final_models[control], name)
            fields[name] = {
                "torch_equal": bool(torch.equal(left, right)),
                "max_abs": float((left - right).abs().max()) if left.shape == right.shape else None,
            }
        field_differences[f"geometric_vs_{control}"] = fields

    train_sheet = out / f"checkpoint_train_{TRAIN_VIEW}.png"
    heldout_sheet = out / f"checkpoint_heldout_{HELDOUT_VIEW}.png"
    _comparison_sheet(
        train_sheet,
        scene,
        checkpoints,
        initial,
        scene.view_names.index(TRAIN_VIEW),
        renderer,
    )
    _comparison_sheet(
        heldout_sheet,
        scene,
        checkpoints,
        initial,
        scene.view_names.index(HELDOUT_VIEW),
        renderer,
    )
    _write_viewer_manifest(out)

    summary = {
        "schema": "rtgs.geometric_arena_frame00008.result.v1",
        "status": "complete",
        "completed_utc": dt.datetime.now(dt.UTC).isoformat(),
        "scope": plan["scope"],
        "plan": _artifact(plan_path),
        "protocol": plan["protocol"],
        "repository": repository,
        "environment": plan["environment"],
        "gpu_processes_before": plan["gpu_processes_before"],
        "gpu_processes_after": _gpu_process_snapshot(),
        "initialization": plan["initialization"],
        "scene": plan["scene"],
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "arms": arm_records,
        "final_field_differences": field_differences,
        "comparison_visuals": {
            "train_checkpoints": _artifact(train_sheet),
            "heldout_checkpoints": _artifact(heldout_sheet),
        },
        "viewer_manifest": _artifact(VIEWER_MANIFEST),
    }
    summary["decision"] = _decision(arm_records)
    index_path = out / "index.html"
    # Bind expected result/audit targets without mutating the page after the scientist pass.
    provisional = {**summary, "index_html": {"path": str(index_path.relative_to(ROOT))}}
    index_path.write_text(_index_html(provisional, out), encoding="utf-8")
    summary["index_html"] = _artifact(index_path)
    summary_path = out / "summary.json"
    _write_json(summary_path, summary)
    RESULT_NOTE.write_text(_result_markdown(summary), encoding="utf-8")

    current_sources = {relative: _sha256(ROOT / relative) for relative in SOURCE_FILES}
    if current_sources != repository["source_files"]:
        raise RuntimeError("bound source changed during execution")
    if _sha256(protocol) != plan["protocol"]["sha256"]:
        raise RuntimeError("frozen protocol changed during execution")
    if _sha256(INITIALIZATION) != INITIALIZATION_SHA256:
        raise RuntimeError("initialization changed during execution")
    for view_name, record in input_records.items():
        if _sha256(prior._input_path(view_name, "rgb")) != record["rgb"]["sha256"]:
            raise RuntimeError(f"RGB input changed during execution: {view_name}")
        if _sha256(prior._input_path(view_name, "mask")) != record["mask"]["sha256"]:
            raise RuntimeError(f"mask input changed during execution: {view_name}")

    print(f"[complete] {summary_path.relative_to(ROOT)}", flush=True)
    print(f"[index] {index_path.relative_to(ROOT)}", flush=True)
    print(f"[result] {RESULT_NOTE.relative_to(ROOT)}", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--worker-arm", choices=ARMS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = args.out.expanduser().resolve()
    protocol = args.protocol.expanduser().resolve()
    if args.worker_arm is not None:
        return _worker(args.worker_arm, out, protocol)
    return _parent(out, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
