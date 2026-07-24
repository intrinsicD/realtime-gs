#!/usr/bin/env python3
"""Run the three pooled Janelle variants for 10k steps with 2k checkpoint samples.

The exact audited stage-2 initializations from ``pool_structure_wse_frame00008_20260724`` are
reused. Each arm starts a fresh 10,000-step refinement schedule and saves complete states, exact
train/held-out metrics, and calibrated renders at 2k, 4k, 6k, 8k, and 10k. A self-contained
``index.html`` is a required, summary-bound result artifact.
"""

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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

try:
    from benchmarks import new_variants_frame00008 as common
except ModuleNotFoundError:
    import new_variants_frame00008 as common

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
RAW_SCENE = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
PARENT_RUN = ROOT / "runs/pool_structure_wse_frame00008_20260724"
PARENT_SUMMARY = PARENT_RUN / "summary.json"
PARENT_SUMMARY_SHA256 = "83c832b920a4603937112f4ff177ca8ac4d420dc58e72e97e847e7c896e176eb"

DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260724_pool_structure_wse_10k_frame00008_PREREG.md"
DEFAULT_OUT = ROOT / "runs/pool_structure_wse_10k_frame00008_20260724"
VIEWER_MANIFEST = ROOT / "benchmarks/results/20260724_pool_structure_wse_10k_frame00008_VIEWER.json"

ARMS = (
    "pool-gradient",
    "pool-structure-density",
    "pool-structure-wse",
)
CHECKPOINT_STEPS = (2_000, 4_000, 6_000, 8_000, 10_000)
TRAIN_ITERATIONS = 10_000
EVAL_EVERY = 100
TRAIN_VIEW = "C0014"
HELDOUT_VIEW = "C1004"

SOURCE_FILES = (
    "benchmarks/pool_structure_wse_10k_frame00008.py",
    "benchmarks/new_variants_frame00008.py",
    "src/rtgs/core/gaussians3d.py",
    "src/rtgs/core/metrics.py",
    "src/rtgs/data/calibrated.py",
    "src/rtgs/optim/trainer.py",
    "src/rtgs/optim/density.py",
    "src/rtgs/optim/strategies.py",
    "src/rtgs/render/gsplat_backend.py",
)

COLORS = {
    "pool-gradient": "#63d7c6",
    "pool-structure-density": "#ffca6e",
    "pool-structure-wse": "#9fa8ff",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
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


def _input_path(view_name: str, kind: str) -> Path:
    if kind == "rgb":
        return RAW_SCENE / "rgb" / f"{view_name}.jpg"
    candidates = (
        RAW_SCENE / "mask" / f"mask_{view_name}.png",
        RAW_SCENE / "mask" / f"mask_{view_name}.jpg",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("gsplat", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
        "ld_preload": os.environ.get("LD_PRELOAD"),
    }


def _train_config() -> TrainConfig:
    return dataclasses.replace(
        common._train_config("final"),
        iterations=TRAIN_ITERATIONS,
        schedule_iterations=TRAIN_ITERATIONS,
        eval_every=EVAL_EVERY,
    )


def _all_fields_equal(left: Gaussians3D, right: Gaussians3D) -> bool:
    return all(
        torch.equal(getattr(left, name), getattr(right, name))
        for name in ("means", "quats", "log_scales", "opacity", "sh")
    )


class _CheckpointCapture:
    """Retain only the preregistered snapshots, detached on CPU."""

    def __init__(self, steps: tuple[int, ...]) -> None:
        self.steps = frozenset(steps)
        self.snapshots: dict[int, Gaussians3D] = {}

    def __call__(self, snapshot: Gaussians3D, step: int) -> None:
        if step not in self.steps:
            return
        if step in self.snapshots:
            raise RuntimeError(f"duplicate checkpoint callback at step {step}")
        self.snapshots[step] = snapshot.to("cpu")


def _save_model(path_stem: Path, model: Gaussians3D) -> dict[str, Any]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    npz = path_stem.with_suffix(".npz")
    ply = path_stem.with_suffix(".ply")
    model.save_npz(npz)
    model.save_ply(ply)
    return {"npz": _artifact(npz), "ply": _artifact(ply)}


def _evaluate(scene, model: Gaussians3D, renderer) -> dict[str, dict[str, float]]:
    device_model = model.to("cuda")
    return {
        "train": Trainer.evaluate_metrics(
            scene,
            device_model,
            renderer,
            indices=scene.training_views,
        ),
        "test": Trainer.evaluate_metrics(
            scene,
            device_model,
            renderer,
            indices=scene.testing_views,
        ),
    }


def _triptych(path: Path, scene, model: Gaussians3D, view_index: int, renderer, label: str) -> None:
    target = scene.images[view_index]
    if scene.masks is not None:
        target = target * scene.masks[view_index][..., None]
    camera = scene.cameras[view_index].to("cuda")
    with torch.no_grad():
        prediction = renderer.render(model.to("cuda"), camera).color.clamp(0.0, 1.0).cpu()
    error = (prediction - target).abs().mul(4.0).clamp(0.0, 1.0)
    panels = [
        common._labeled_panel(target, f"{label} · target"),
        common._labeled_panel(prediction, f"{label} · render"),
        common._labeled_panel(error, f"{label} · |error|×4"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    common._horizontal_sheet(panels).save(path)


def _progress_gif(path: Path, frames: list[Path]) -> None:
    images = []
    for frame in frames:
        with Image.open(frame) as image:
            images.append(image.convert("RGB").copy())
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=900,
        loop=0,
        optimize=False,
    )


def _checkpoint_sheet(
    path: Path,
    scene,
    snapshots: dict[str, dict[int, Gaussians3D]],
    view_index: int,
    renderer,
    initializations: dict[str, Gaussians3D],
) -> None:
    target = scene.images[view_index]
    if scene.masks is not None:
        target = target * scene.masks[view_index][..., None]
    camera = scene.cameras[view_index].to("cuda")
    rows = []
    with torch.no_grad():
        for arm in ARMS:
            panels = [common._labeled_panel(target, f"{arm} · target")]
            initial = renderer.render(initializations[arm].to("cuda"), camera).color.clamp(0, 1)
            panels.append(common._labeled_panel(initial, f"{arm} · init"))
            for step in CHECKPOINT_STEPS:
                render = renderer.render(snapshots[arm][step].to("cuda"), camera).color.clamp(0, 1)
                panels.append(common._labeled_panel(render, f"{arm} · {step // 1000}k"))
            rows.append(common._horizontal_sheet(panels))
    common._vertical_sheet(rows).save(path)


def _trajectory_svg(
    records: dict[str, dict[str, Any]],
    initial_metrics: dict[str, dict[str, dict[str, float]]],
) -> str:
    width, height = 1180, 500
    margin_left, margin_top, panel_width, panel_height = 72, 62, 480, 330
    panel_gap = 95
    steps = (0, *CHECKPOINT_STEPS)

    def values(arm: str, split: str) -> list[float]:
        result = [float(initial_metrics[arm][split]["psnr_fg"])]
        result.extend(
            float(records[arm]["checkpoints"][str(step)]["metrics"][split]["psnr_fg"])
            for step in CHECKPOINT_STEPS
        )
        return result

    panels = []
    for panel_index, (split, title) in enumerate(
        (("train", "Train foreground PSNR"), ("test", "Held-out C1004 foreground PSNR"))
    ):
        x0 = margin_left + panel_index * (panel_width + panel_gap)
        all_values = [value for arm in ARMS for value in values(arm, split)]
        low, high = min(all_values), max(all_values)
        padding = max((high - low) * 0.12, 0.1)
        low -= padding
        high += padding

        def x(step: int, origin: float = x0) -> float:
            return origin + panel_width * step / TRAIN_ITERATIONS

        def y(value: float, lower: float = low, upper: float = high) -> float:
            return margin_top + panel_height * (upper - value) / (upper - lower)

        parts = [
            f'<text x="{x0}" y="30" class="title">{html.escape(title)}</text>',
            (
                f'<rect x="{x0}" y="{margin_top}" width="{panel_width}" '
                f'height="{panel_height}" class="plot"/>'
            ),
        ]
        for tick in range(6):
            step = tick * 2_000
            px = x(step)
            parts.append(
                f'<line x1="{px}" y1="{margin_top}" x2="{px}" '
                f'y2="{margin_top + panel_height}" class="grid"/>'
            )
            parts.append(
                f'<text x="{px}" y="{margin_top + panel_height + 25}" '
                f'class="tick" text-anchor="middle">{step // 1000}k</text>'
            )
        for tick in range(5):
            value = low + (high - low) * tick / 4
            py = y(value)
            parts.append(
                f'<line x1="{x0}" y1="{py}" x2="{x0 + panel_width}" y2="{py}" class="grid"/>'
            )
            parts.append(
                f'<text x="{x0 - 9}" y="{py + 4}" class="tick" text-anchor="end">{value:.1f}</text>'
            )
        for arm in ARMS:
            points = " ".join(
                f"{x(step):.2f},{y(value):.2f}"
                for step, value in zip(steps, values(arm, split), strict=True)
            )
            color = COLORS[arm]
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for step, value in zip(steps, values(arm, split), strict=True):
                parts.append(
                    f'<circle cx="{x(step):.2f}" cy="{y(value):.2f}" r="4" fill="{color}"/>'
                )
        panels.append("".join(parts))

    legend = []
    for index, arm in enumerate(ARMS):
        x = margin_left + index * 280
        legend.append(
            f'<line x1="{x}" y1="455" x2="{x + 28}" y2="455" '
            f'stroke="{COLORS[arm]}" stroke-width="4"/>'
            f'<text x="{x + 38}" y="460" class="legend">{html.escape(arm)}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="PSNR trajectories">'
        "<style>"
        ".bg{fill:#10151e}.plot{fill:#090c11;stroke:#2a3447}.grid{stroke:#273044;"
        "stroke-width:1}.title,.legend{fill:#eaf1fb;font:600 16px system-ui,sans-serif}"
        ".tick{fill:#9eabc0;font:12px system-ui,sans-serif}"
        "</style>"
        f'<rect width="{width}" height="{height}" class="bg"/>'
        + "".join(panels)
        + "".join(legend)
        + "</svg>"
    )


def _index_html(
    arm_records: dict[str, dict[str, Any]],
    initial_metrics: dict[str, dict[str, dict[str, float]]],
    out: Path,
) -> str:
    table_rows = []
    for arm in ARMS:
        initial = initial_metrics[arm]
        table_rows.append(
            "<tr>"
            f'<td><span class="swatch" style="--c:{COLORS[arm]}"></span>{arm}</td>'
            "<td>init</td>"
            f"<td>{arm_records[arm]['init_n_gaussians']:,}</td>"
            f"<td>{initial['train']['psnr_fg']:.4f}</td>"
            f"<td>{initial['test']['psnr_fg']:.4f}</td>"
            f"<td>{initial['test']['alpha_iou']:.5f}</td>"
            "<td>—</td></tr>"
        )
        for step in CHECKPOINT_STEPS:
            record = arm_records[arm]["checkpoints"][str(step)]
            metrics = record["metrics"]
            table_rows.append(
                "<tr>"
                f'<td><span class="swatch" style="--c:{COLORS[arm]}"></span>{arm}</td>'
                f"<td>{step // 1000}k</td>"
                f"<td>{record['n_gaussians']:,}</td>"
                f"<td>{metrics['train']['psnr_fg']:.4f}</td>"
                f"<td>{metrics['test']['psnr_fg']:.4f}</td>"
                f"<td>{metrics['test']['alpha_iou']:.5f}</td>"
                f"<td>{metrics['test']['alpha_outside']:.5f}</td></tr>"
            )

    arm_sections = []
    for arm in ARMS:
        cards = []
        for step in CHECKPOINT_STEPS:
            record = arm_records[arm]["checkpoints"][str(step)]
            metrics = record["metrics"]["test"]
            visual = record["visuals"]["heldout"]["path"]
            visual_rel = os.path.relpath(ROOT / visual, out)
            ply_rel = os.path.relpath(ROOT / record["artifacts"]["ply"]["path"], out)
            cards.append(
                '<article class="checkpoint">'
                f"<header><strong>{step // 1000}k</strong>"
                f"<span>{metrics['psnr_fg']:.3f} dB · α-IoU {metrics['alpha_iou']:.4f}</span>"
                "</header>"
                f'<a href="{html.escape(visual_rel)}"><img loading="lazy" '
                f'src="{html.escape(visual_rel)}" alt="{html.escape(arm)} at {step} steps"></a>'
                f'<footer><a href="{html.escape(ply_rel)}">PLY</a></footer>'
                "</article>"
            )
        arm_sections.append(
            f'<section class="arm"><h2><span class="swatch" '
            f'style="--c:{COLORS[arm]}"></span>{html.escape(arm)}</h2>'
            f'<div class="checkpoint-grid">{"".join(cards)}</div></section>'
        )

    final_values = {
        arm: arm_records[arm]["checkpoints"][str(TRAIN_ITERATIONS)]["metrics"]["test"]["psnr_fg"]
        for arm in ARMS
    }
    highest_arm = max(final_values, key=final_values.__getitem__)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Janelle pooled variants · 10k checkpoint study</title>
  <style>
    :root {{ color-scheme:dark; --bg:#090c11; --panel:#121821; --line:#293448;
      --text:#edf3fb; --muted:#9aa8bc; --accent:#63d7c6; --warm:#ffca6e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 15% -5%,#203145 0,var(--bg) 38rem);
      color:var(--text); font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif; }}
    a {{ color:var(--accent); }} code {{ color:#d8e4f4; }}
    .shell {{ width:min(1500px,calc(100% - 32px)); margin:auto; }}
    .hero {{ padding:48px 0 22px; }} .eyebrow {{ color:var(--accent); font-size:12px;
      font-weight:800; letter-spacing:.14em; text-transform:uppercase; }}
    h1 {{ margin:7px 0 12px; font-size:clamp(32px,5vw,60px); line-height:1.02; }}
    .lead {{ color:var(--muted); max-width:980px; font-size:17px; }}
    .notice {{ border:1px solid #665530; background:#211c12; color:#ead8ab; padding:13px 15px;
      border-radius:11px; max-width:1100px; }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px;
      margin:24px 0; }} .summary div {{ background:rgba(18,24,33,.9); border:1px solid var(--line);
      border-radius:12px; padding:14px 16px; display:flex; flex-direction:column; }}
    .summary span,.summary small {{ color:var(--muted); }} .summary strong {{ color:var(--warm);
      font-size:25px; }}
    nav {{ position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:14px;
      padding:12px 0; background:rgba(9,12,17,.91); backdrop-filter:blur(14px);
      border-bottom:1px solid var(--line); }}
    section {{ margin:28px 0; }} h2 {{ margin:0 0 12px; }}
    .panel {{ background:rgba(18,24,33,.9); border:1px solid var(--line); border-radius:14px;
      padding:18px; overflow:auto; }}
    .hero-image {{ display:block; width:100%; background:#000; border-radius:10px; }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:right;
      white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }}
    .swatch {{ display:inline-block; width:10px; height:10px; background:var(--c);
      border-radius:50%; margin-right:8px; }}
    .checkpoint-grid {{ display:grid; grid-template-columns:repeat(5,minmax(220px,1fr)); gap:10px;
      overflow-x:auto; }} .checkpoint {{ min-width:220px; background:var(--panel);
      border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
    .checkpoint header {{ display:flex; justify-content:space-between; gap:8px;
      padding:10px 12px; }}
    .checkpoint header span,.checkpoint footer {{ color:var(--muted); font-size:12px; }}
    .checkpoint img {{ display:block; width:100%; background:#000; }}
    .checkpoint footer {{ padding:8px 12px; }}
    .links {{ display:flex; flex-wrap:wrap; gap:14px; }}
    .page-footer {{ color:var(--muted); border-top:1px solid var(--line); padding:24px 0 40px; }}
    @media(max-width:760px) {{ .checkpoint-grid {{ grid-template-columns:1fr; }}
      th,td {{ padding:7px; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">realtime-gs · calibrated checkpoint study</div>
      <h1>Pooled variants over a fresh 10k schedule</h1>
      <p class="lead">Three audited Janelle <code>frame_00008</code> initializations, each trained
      from step zero for 10,000 iterations. Complete states and exact metrics are sampled at
      2k, 4k, 6k, 8k, and 10k; <code>C1004</code> remains reporting-only.</p>
      <p class="notice"><strong>Scope:</strong> this page reports observed values and visual
      diagnostics. It is generated before the independent scientist pass and does not itself
      authorize a winner, default, timing, or generalization claim.</p>
      <div class="summary">
        <div><span>Variants</span><strong>3</strong><small>same parent initializations</small></div>
        <div><span>Schedule</span><strong>10k</strong>
          <small>fresh optimizer trajectory</small></div>
        <div><span>Saved states</span><strong>15</strong><small>five per arm</small></div>
        <div><span>Highest observed 10k test FG</span>
          <strong>{final_values[highest_arm]:.3f}</strong>
          <small>{html.escape(highest_arm)} · not a selection claim</small></div>
      </div>
    </section>
    <nav>
      <a href="#trajectories">Trajectories</a><a href="#metrics">Metrics</a>
      <a href="#visuals">Visual checkpoints</a>
      <a href="summary.json">summary.json</a>
      <a href="../../benchmarks/results/20260724_pool_structure_wse_10k_frame00008_VIEWER.json">
        viewer manifest</a>
      <a href="../../benchmarks/results/20260724_pool_structure_wse_10k_frame00008_RESULT.md">
        result note</a>
      <a href="../../benchmarks/results/20260724_pool_structure_wse_10k_frame00008_AUDIT.json">
        audit JSON</a>
    </nav>
    <section id="trajectories">
      <h2>Foreground PSNR trajectories</h2>
      <div class="panel"><img class="hero-image" src="quality_trajectory.svg"
        alt="Train and held-out PSNR over 10k steps"></div>
    </section>
    <section>
      <h2>Cross-arm checkpoint sheets</h2>
      <div class="panel"><h3>Held-out · C1004</h3>
        <a href="checkpoint_heldout_C1004.png"><img class="hero-image"
          src="checkpoint_heldout_C1004.png" alt="Held-out checkpoint comparison"></a>
        <h3>Train · C0014</h3>
        <a href="checkpoint_train_C0014.png"><img class="hero-image"
          src="checkpoint_train_C0014.png" alt="Train checkpoint comparison"></a>
      </div>
    </section>
    <section id="metrics">
      <h2>Exact checkpoint metrics</h2>
      <div class="panel"><table>
        <thead><tr><th>Variant</th><th>Step</th><th>N</th><th>Train FG dB</th>
          <th>Held-out FG dB</th><th>Held-out α-IoU</th><th>Outside α</th></tr></thead>
        <tbody>{"".join(table_rows)}</tbody>
      </table></div>
    </section>
    <section id="visuals">
      <h2>Held-out checkpoint details</h2>
      {"".join(arm_sections)}
    </section>
    <section>
      <h2>Provenance</h2>
      <div class="panel links">
        <a href="../../benchmarks/results/20260724_pool_structure_wse_10k_frame00008_PREREG.md">
          frozen protocol</a>
        <a href="../pool_structure_wse_frame00008_20260724/summary.json">parent 2k summary</a>
        <a href="stage1_contact_sheet.png">parent stage-1 sheet</a>
      </div>
    </section>
    <footer class="page-footer">Generated from exact saved checkpoint metrics and renders.
      All assets use relative links and remain inspectable offline.</footer>
  </main>
</body>
</html>
"""


def _write_viewer_manifest(out: Path) -> None:
    methods = []
    for arm in ARMS:
        initial = out / "models" / arm / "gaussians_init.ply"
        for step in CHECKPOINT_STEPS:
            checkpoint = out / "models" / arm / "checkpoints" / f"gaussians_step_{step:05d}.ply"
            methods.append(
                {
                    "name": f"{arm} @ {step // 1000}k",
                    "initial": os.path.relpath(initial, VIEWER_MANIFEST.parent),
                    "final": os.path.relpath(checkpoint, VIEWER_MANIFEST.parent),
                }
            )
    _write_json(
        VIEWER_MANIFEST,
        {
            "schema": "rtgs.viewer-comparison.v1",
            "methods": methods,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = args.out.expanduser().resolve()
    protocol = args.protocol.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {out}")
    if not protocol.is_file():
        raise FileNotFoundError(f"frozen protocol is missing: {protocol}")
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen experiment requires CUDA")
    if _sha256(PARENT_SUMMARY) != PARENT_SUMMARY_SHA256:
        raise RuntimeError("audited parent summary hash drifted")

    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if parent.get("status") != "complete" or tuple(parent["reconstruction"]) != ARMS:
        raise RuntimeError("audited parent result structure drifted")

    out.mkdir(parents=True)
    try:
        scene = load_calibrated_scene(
            RAW_SCENE,
            calibration_path=CALIBRATION,
            downscale=16,
            max_images=8,
            test_every=8,
            load_masks=True,
            undistort=True,
        )
        if scene.masks is None:
            raise RuntimeError("complete masks are required")
        expected_views = [
            "C0001",
            "C0008",
            "C0014",
            "C0021",
            "C0026",
            "C0031",
            "C0039",
            "C1004",
        ]
        if scene.view_names != expected_views:
            raise RuntimeError(f"calibrated view order drifted: {scene.view_names}")
        if scene.training_views != list(range(7)) or scene.testing_views != [7]:
            raise RuntimeError("frozen seven-train/one-test split drifted")

        input_records = {}
        for name, image, mask in zip(scene.view_names, scene.images, scene.masks, strict=True):
            rgb = _input_path(name, "rgb")
            mask_path = _input_path(name, "mask")
            input_records[name] = {
                "rgb": _artifact(rgb),
                "mask": _artifact(mask_path),
                "loaded_rgb_tensor_sha256": _tensor_hash(image),
                "loaded_mask_tensor_sha256": _tensor_hash(mask),
            }

        repository = _repository_binding()
        config = _train_config()
        parent_initial_records = {
            arm: parent["reconstruction"][arm]["artifacts"]["initial_npz"] for arm in ARMS
        }
        plan = {
            "schema": "rtgs.pool_structure_wse_10k_frame00008.plan.v1",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "scope": (
                "single-scene, single-seed 10k development trajectories from exact audited "
                "initializations; held-out C1004 reporting only; no default/performance claim"
            ),
            "protocol": _artifact(protocol),
            "repository": repository,
            "environment": _environment(),
            "parent": {
                "summary": _artifact(PARENT_SUMMARY),
                "initial_npz": parent_initial_records,
            },
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
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "training": dataclasses.asdict(config),
            "schedule_note": (
                "fresh 10k schedule from parent initializations; the 2k checkpoint is not a "
                "continuation or expected replay of the parent 2k endpoint"
            ),
            "index_html_required": True,
        }
        plan_path = out / "plan.json"
        _write_json(plan_path, plan)

        parent_stage1 = ROOT / parent["comparison_visuals"]["stage1"]["path"]
        stage1_copy = out / "stage1_contact_sheet.png"
        shutil.copy2(parent_stage1, stage1_copy)

        renderer = get_rasterizer(
            "gsplat",
            device=torch.device("cuda"),
            packed=False,
            antialiased=True,
        )
        initializations: dict[str, Gaussians3D] = {}
        snapshots: dict[str, dict[int, Gaussians3D]] = {}
        initial_metrics: dict[str, dict[str, dict[str, float]]] = {}
        arm_records: dict[str, dict[str, Any]] = {}

        for arm in ARMS:
            print(f"[load-init] {arm}", flush=True)
            parent_record = parent_initial_records[arm]
            parent_path = ROOT / parent_record["path"]
            if (
                not parent_path.is_file()
                or parent_path.stat().st_size != parent_record["bytes"]
                or _sha256(parent_path) != parent_record["sha256"]
            ):
                raise RuntimeError(f"parent initialization artifact drifted: {arm}")
            initial = Gaussians3D.load_npz(parent_path)
            initializations[arm] = initial
            initial_metrics[arm] = _evaluate(scene, initial, renderer)
            model_dir = out / "models" / arm
            _save_model(model_dir / "gaussians_init", initial)

            capture = _CheckpointCapture(CHECKPOINT_STEPS)
            print(f"[refine-10k] {arm}", flush=True)
            returned, history = Trainer(config).train(
                scene,
                initial.to("cuda"),
                checkpoint_callback=capture,
            )
            returned_cpu = returned.to("cpu")
            if tuple(sorted(capture.snapshots)) != CHECKPOINT_STEPS:
                raise RuntimeError(
                    f"{arm} checkpoint set drifted: {tuple(sorted(capture.snapshots))}"
                )
            if not _all_fields_equal(returned_cpu, capture.snapshots[TRAIN_ITERATIONS]):
                raise RuntimeError(f"{arm} returned final differs from captured 10k state")
            snapshots[arm] = capture.snapshots

            history_path = model_dir / "training_history.json"
            _write_json(history_path, history)
            checkpoint_records = {}
            train_frames = []
            heldout_frames = []
            for step in CHECKPOINT_STEPS:
                model = capture.snapshots[step]
                checkpoint_dir = model_dir / "checkpoints"
                artifacts = _save_model(
                    checkpoint_dir / f"gaussians_step_{step:05d}",
                    model,
                )
                metrics = _evaluate(scene, model, renderer)
                visual_dir = model_dir / "checkpoint_visuals" / f"step_{step:05d}"
                train_visual = visual_dir / f"train_{TRAIN_VIEW}.png"
                heldout_visual = visual_dir / f"heldout_{HELDOUT_VIEW}.png"
                _triptych(
                    train_visual,
                    scene,
                    model,
                    scene.view_names.index(TRAIN_VIEW),
                    renderer,
                    f"{arm} · {step // 1000}k · train",
                )
                _triptych(
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
                    "step": step,
                    "n_gaussians": model.n,
                    "metrics": metrics,
                    "artifacts": artifacts,
                    "visuals": {
                        "train": _artifact(train_visual),
                        "heldout": _artifact(heldout_visual),
                    },
                }
                print(
                    f"  {step // 1000}k N={model.n} "
                    f"train={metrics['train']['psnr_fg']:.4f} "
                    f"test={metrics['test']['psnr_fg']:.4f}",
                    flush=True,
                )

            train_gif = model_dir / f"checkpoint_train_{TRAIN_VIEW}.gif"
            heldout_gif = model_dir / f"checkpoint_heldout_{HELDOUT_VIEW}.gif"
            _progress_gif(train_gif, train_frames)
            _progress_gif(heldout_gif, heldout_frames)
            final_artifacts = _save_model(model_dir / "gaussians_final", returned_cpu)
            arm_records[arm] = {
                "parent_initial_npz": parent_record,
                "init_n_gaussians": initial.n,
                "init_metrics": initial_metrics[arm],
                "training_history": _artifact(history_path),
                "checkpoints": checkpoint_records,
                "final_step": TRAIN_ITERATIONS,
                "final_n_gaussians": returned_cpu.n,
                "final_artifacts": final_artifacts,
                "progress_visuals": {
                    "train_gif": _artifact(train_gif),
                    "heldout_gif": _artifact(heldout_gif),
                },
            }
            del returned, returned_cpu
            torch.cuda.empty_cache()

        train_sheet = out / f"checkpoint_train_{TRAIN_VIEW}.png"
        heldout_sheet = out / f"checkpoint_heldout_{HELDOUT_VIEW}.png"
        _checkpoint_sheet(
            train_sheet,
            scene,
            snapshots,
            scene.view_names.index(TRAIN_VIEW),
            renderer,
            initializations,
        )
        _checkpoint_sheet(
            heldout_sheet,
            scene,
            snapshots,
            scene.view_names.index(HELDOUT_VIEW),
            renderer,
            initializations,
        )

        trajectory_path = out / "quality_trajectory.svg"
        trajectory_path.write_text(
            _trajectory_svg(arm_records, initial_metrics),
            encoding="utf-8",
        )
        _write_viewer_manifest(out)

        index_path = out / "index.html"
        index_path.write_text(
            _index_html(arm_records, initial_metrics, out),
            encoding="utf-8",
        )

        current_source_hashes = {relative: _sha256(ROOT / relative) for relative in SOURCE_FILES}
        if current_source_hashes != repository["source_files"]:
            raise RuntimeError("bound experiment source changed during execution")
        if _sha256(protocol) != plan["protocol"]["sha256"]:
            raise RuntimeError("frozen protocol changed during execution")
        if _sha256(PARENT_SUMMARY) != PARENT_SUMMARY_SHA256:
            raise RuntimeError("parent summary changed during execution")
        for view_name, record in input_records.items():
            if _sha256(_input_path(view_name, "rgb")) != record["rgb"]["sha256"]:
                raise RuntimeError(f"RGB input changed during execution: {view_name}")
            if _sha256(_input_path(view_name, "mask")) != record["mask"]["sha256"]:
                raise RuntimeError(f"mask input changed during execution: {view_name}")

        summary = {
            "schema": "rtgs.pool_structure_wse_10k_frame00008.result.v1",
            "status": "complete",
            "completed_utc": dt.datetime.now(dt.UTC).isoformat(),
            "scope": plan["scope"],
            "plan": _artifact(plan_path),
            "protocol": plan["protocol"],
            "repository": repository,
            "environment": plan["environment"],
            "parent": plan["parent"],
            "scene": plan["scene"],
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "training": plan["training"],
            "arms": arm_records,
            "comparison_visuals": {
                "stage1_parent": _artifact(stage1_copy),
                "train_checkpoints": _artifact(train_sheet),
                "heldout_checkpoints": _artifact(heldout_sheet),
                "quality_trajectory": _artifact(trajectory_path),
            },
            "viewer_manifest": _artifact(VIEWER_MANIFEST),
            "index_html": _artifact(index_path),
            "timing_policy": (
                "elapsed and allocation fields describe a contended, unrepeated local execution "
                "and are not performance evidence"
            ),
        }
        summary_path = out / "summary.json"
        _write_json(summary_path, summary)
        print(f"[complete] {summary_path.relative_to(ROOT)}", flush=True)
        print(f"[index] {index_path.relative_to(ROOT)}", flush=True)
        print(f"[viewer] {VIEWER_MANIFEST.relative_to(ROOT)}", flush=True)
        return 0
    except BaseException as error:
        _write_json(
            out / "failure.json",
            {
                "schema": "rtgs.pool_structure_wse_10k_frame00008.failure.v1",
                "failed_utc": dt.datetime.now(dt.UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
