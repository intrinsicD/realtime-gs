#!/usr/bin/env python3
# ruff: noqa: E501
"""Render compact StructSplat teachers and compare them with their source RGB views.

The compact ``.rtgsv`` files hold fitted 2D StructSplat fields, not 3D Gaussian sets.  This
utility reconstructs each field with the independent StructSplat reference renderer at its
native fitted-window resolution, reproduces the calibrated RGB preprocessing used during dataset
conversion, and writes a static browser gallery plus machine-readable provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from rtgs.data.calibrated import _resize_image, _undistort
from rtgs.data.compact_views import CompactDataset, CompactView, file_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPACT_ROOT = ROOT / "dataset/2025_03_07_stage_with_fabric"
DEFAULT_SOURCE_ROOT = ROOT / "external/dataset/2025_03_07_stage_with_fabric"
DEFAULT_STRUCTSPLAT_ROOT = ROOT / "external/structsplat"
DEFAULT_OUT = ROOT / "runs/structsplat_teacher_gallery_20260721"
DISPLAY_ERROR_SCALE = 4.0


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _git_output(repository: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _structsplat_source_binding(structsplat_root: Path) -> dict[str, Any]:
    source_root = structsplat_root / "src/structsplat"
    suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    files = sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.suffix in suffixes
    )
    digest = hashlib.sha256()
    manifest: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        value = _sha256(path)
        manifest[relative] = value
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "repository_root": str(structsplat_root),
        "module_root": str(source_root),
        "git_revision": _git_output(structsplat_root, "rev-parse", "HEAD"),
        "git_status": _git_output(structsplat_root, "status", "--short"),
        "source_digest": digest.hexdigest(),
        "source_file_count": len(files),
        "source_manifest_digest": hashlib.sha256(_canonical_json(manifest)).hexdigest(),
    }


def _calibration_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["camera_id"]).upper(): record for record in payload["cameras"]}


def _load_preprocessed_rgb(
    view: CompactView,
    *,
    source_frame: Path,
    calibration_record: dict[str, Any],
) -> tuple[torch.Tensor, Path]:
    source_record = view.source["rgb"]
    source_path = source_frame / "rgb" / source_record["name"]
    if not source_path.is_file():
        raise FileNotFoundError(f"missing source RGB for {view.view_id}: {source_path}")
    if _sha256(source_path) != source_record["sha256"]:
        raise RuntimeError(f"source RGB digest mismatch for {view.view_id}")

    mask_record = view.source["mask"]
    if mask_record is not None:
        mask_path = source_frame / "mask" / mask_record["name"]
        if not mask_path.is_file() or _sha256(mask_path) != mask_record["sha256"]:
            raise RuntimeError(f"source mask digest mismatch for {view.view_id}")

    intrinsics = calibration_record["intrinsics"]
    width, height = (int(value) for value in intrinsics["resolution"])
    matrix = intrinsics["camera_matrix"]
    fx, fy = float(matrix[0]), float(matrix[4])
    cx, cy = float(matrix[2]) + 0.5, float(matrix[5]) + 0.5
    distortion = [float(value) for value in intrinsics.get("distortion_coefficients", [])]
    if (width, height) != (view.camera.width, view.camera.height):
        raise RuntimeError(f"calibration dimensions differ from compact camera for {view.view_id}")
    image = _resize_image(source_path, width, height)
    image = _undistort(image, fx, fy, cx, cy, distortion).contiguous()
    return image, source_path


def _render_structsplat(field, *, chunk: int) -> torch.Tensor:
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    if field.epsilon != 1e-8:
        raise ValueError(
            "the external StructSplat reference renderer fixes epsilon at 1e-8; "
            f"archive declares {field.epsilon}"
        )
    _, _, fit_width, fit_height = field.fit_window
    live = GaussianField(
        field.local_means(),
        field.log_scales,
        field.rotations,
        field.colors,
        color_grads=field.color_grads,
        filter_variance=field.filter_variance,
    )
    with torch.no_grad():
        rendered = render_field(
            live.means,
            live.conics(field.aa_dilation),
            live.colors,
            live.radii(field.sigma_cutoff, field.aa_dilation),
            fit_height,
            fit_width,
            chunk=chunk,
            mode=field.blend_mode,
            opacities=field.amplitudes,
            scales=live.effective_scales(0.0),
            rotations=live.rotations,
            support_fade=field.support_fade_alpha > 0.0,
            support_fade_alpha=field.support_fade_alpha,
            sigma_cutoff=field.sigma_cutoff,
            color_grads=live.color_grads,
        )
    if rendered.shape != (fit_height, fit_width, 3) or not bool(torch.isfinite(rendered).all()):
        raise RuntimeError("StructSplat returned an invalid reconstruction")
    return rendered.cpu()


def _resize_hwc(value: torch.Tensor, width: int) -> torch.Tensor:
    if value.shape[1] <= width:
        return value
    height = max(1, round(value.shape[0] * width / value.shape[1]))
    return F.interpolate(
        value.permute(2, 0, 1).unsqueeze(0),
        size=(height, width),
        mode="area",
    )[0].permute(1, 2, 0)


def _save_jpeg(path: Path, value: torch.Tensor, *, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (value.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(array).save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=True,
        progressive=True,
    )


def _error_heatmap(error: torch.Tensor) -> torch.Tensor:
    """Map mean absolute RGB error to a fixed black-blue-red-yellow display scale."""
    x = (error * DISPLAY_ERROR_SCALE).clamp(0.0, 1.0)
    red = (2.2 * x - 0.35).clamp(0.0, 1.0)
    green = (2.5 * x - 1.3).clamp(0.0, 1.0)
    blue = (1.8 * x).clamp(0.0, 1.0) * (1.0 - 0.65 * x)
    return torch.stack([red, green, blue], dim=-1)


def _psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-30))


def _query_parity(field, rendered: torch.Tensor, *, samples: int, seed: int) -> dict[str, float]:
    if samples <= 0:
        return {"samples": 0, "max_abs_error": 0.0, "mean_abs_error": 0.0}
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    generator = torch.Generator().manual_seed(seed)
    xs = torch.randint(fit_width, (samples,), generator=generator)
    ys = torch.randint(fit_height, (samples,), generator=generator)
    xy = torch.stack([xs + fit_x, ys + fit_y], dim=-1).to(field.dtype) + 0.5
    with torch.no_grad():
        queried = field.query(xy, component_chunk=128).color
    absolute = (queried - rendered[ys, xs]).abs()
    return {
        "samples": samples,
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
    }


def _relative(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def _render_view(
    view: CompactView,
    *,
    frame_name: str,
    source_frame: Path,
    calibration_record: dict[str, Any],
    out: Path,
    chunk: int,
    preview_width: int,
    jpeg_quality: int,
    parity_samples: int,
) -> dict[str, Any]:
    field = view.observation
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    started = time.perf_counter()
    source_rgb, source_path = _load_preprocessed_rgb(
        view,
        source_frame=source_frame,
        calibration_record=calibration_record,
    )
    target = source_rgb[fit_y : fit_y + fit_height, fit_x : fit_x + fit_width]
    if target.shape != (fit_height, fit_width, 3):
        raise RuntimeError(f"source crop shape mismatch for {frame_name}/{view.view_id}")
    if view.alpha is None:
        foreground = torch.ones(fit_height, fit_width, dtype=torch.bool)
    else:
        foreground = view.alpha.crop_mask()
    if foreground.shape != (fit_height, fit_width) or not bool(foreground.any()):
        raise RuntimeError(f"invalid compact alpha for {frame_name}/{view.view_id}")

    render_started = time.perf_counter()
    rendered = _render_structsplat(field, chunk=chunk)
    render_seconds = time.perf_counter() - render_started
    parity_seed = int(view.sha256[:8], 16) & 0x7FFF_FFFF
    parity = _query_parity(field, rendered, samples=parity_samples, seed=parity_seed)

    clamped = rendered.clamp(0.0, 1.0)
    mask3 = foreground[..., None]
    foreground_difference = rendered[foreground] - target[foreground]
    foreground_clamped_difference = clamped[foreground] - target[foreground]
    foreground_mse = float(foreground_difference.square().mean())
    foreground_clamped_mse = float(foreground_clamped_difference.square().mean())
    foreground_mae = float(foreground_clamped_difference.abs().mean())
    render_out_of_range_fraction = float(((rendered < 0.0) | (rendered > 1.0)).float().mean())
    masked_difference = (clamped - target) * mask3
    masked_crop_mse = float(masked_difference.square().mean())
    display_error = (clamped - target).abs().mean(dim=-1) * foreground

    native_dir = out / "images/native" / frame_name
    preview_dir = out / "images/preview" / frame_name
    target_native = native_dir / f"{view.view_id}_original_undistorted_crop.jpg"
    render_native = native_dir / f"{view.view_id}_structsplat_reconstruction.jpg"
    target_preview = preview_dir / f"{view.view_id}_original_foreground.jpg"
    render_preview = preview_dir / f"{view.view_id}_reconstruction_foreground.jpg"
    error_preview = preview_dir / f"{view.view_id}_absolute_error_x4.jpg"

    _save_jpeg(target_native, target, quality=jpeg_quality)
    _save_jpeg(render_native, clamped, quality=jpeg_quality)
    _save_jpeg(
        target_preview,
        _resize_hwc(target * mask3, preview_width),
        quality=max(80, jpeg_quality - 5),
    )
    _save_jpeg(
        render_preview,
        _resize_hwc(clamped * mask3, preview_width),
        quality=max(80, jpeg_quality - 5),
    )
    heatmap = _error_heatmap(display_error)
    _save_jpeg(
        error_preview,
        _resize_hwc(heatmap, preview_width),
        quality=max(80, jpeg_quality - 5),
    )

    del source_rgb, target, rendered, clamped, foreground_difference
    return {
        "frame": frame_name,
        "view_id": view.view_id,
        "compact_path": _relative(view.path, out),
        "compact_sha256": view.sha256,
        "source_path": _relative(source_path, out),
        "source_rgb_sha256": view.source["rgb"]["sha256"],
        "fit_window": [fit_x, fit_y, fit_width, fit_height],
        "n_gaussians": field.n,
        "n_init": field.n_init,
        "renderer": {
            "provider": "external/structsplat",
            "mode": field.blend_mode,
            "chunk": chunk,
            "epsilon": field.epsilon,
            "sigma_cutoff": field.sigma_cutoff,
            "support_fade_alpha": field.support_fade_alpha,
            "aa_dilation": field.aa_dilation,
        },
        "metrics": {
            "foreground_pixels": int(foreground.sum()),
            "foreground_fraction_of_fit_window": float(foreground.float().mean()),
            "foreground_psnr_db": _psnr(foreground_mse),
            "foreground_clamped_psnr_db": _psnr(foreground_clamped_mse),
            "foreground_clamped_mae": foreground_mae,
            "masked_fit_window_psnr_db": _psnr(masked_crop_mse),
            "render_out_of_display_range_fraction": render_out_of_range_fraction,
            "archive_query_vs_structsplat_render": parity,
        },
        "timing": {
            "structsplat_render_seconds": render_seconds,
            "total_view_seconds": time.perf_counter() - started,
        },
        "images": {
            "original_native": _relative(target_native, out),
            "reconstruction_native": _relative(render_native, out),
            "original_preview": _relative(target_preview, out),
            "reconstruction_preview": _relative(render_preview, out),
            "error_preview": _relative(error_preview, out),
        },
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for frame in sorted({record["frame"] for record in records}):
        selected = [record for record in records if record["frame"] == frame]
        frames[frame] = {
            "views": len(selected),
            "mean_foreground_clamped_psnr_db": _mean(
                [item["metrics"]["foreground_clamped_psnr_db"] for item in selected]
            ),
            "min_foreground_clamped_psnr_db": min(
                item["metrics"]["foreground_clamped_psnr_db"] for item in selected
            ),
            "max_foreground_clamped_psnr_db": max(
                item["metrics"]["foreground_clamped_psnr_db"] for item in selected
            ),
            "mean_structsplat_render_seconds": _mean(
                [item["timing"]["structsplat_render_seconds"] for item in selected]
            ),
        }
    return {
        "views": len(records),
        "frames": frames,
        "mean_foreground_clamped_psnr_db": _mean(
            [record["metrics"]["foreground_clamped_psnr_db"] for record in records]
        ),
        "max_archive_query_vs_structsplat_render_abs_error": max(
            record["metrics"]["archive_query_vs_structsplat_render"]["max_abs_error"]
            for record in records
        ),
        "total_structsplat_render_seconds": sum(
            record["timing"]["structsplat_render_seconds"] for record in records
        ),
    }


def _card(record: dict[str, Any]) -> str:
    images = record["images"]
    metrics = record["metrics"]
    frame = html.escape(record["frame"])
    view_id = html.escape(record["view_id"])
    psnr = metrics["foreground_clamped_psnr_db"]
    mae = metrics["foreground_clamped_mae"]
    parity = metrics["archive_query_vs_structsplat_render"]["max_abs_error"]
    fit = record["fit_window"]
    return f"""
      <article class="card" data-frame="{frame}" data-view="{view_id.lower()}"
               data-psnr="{psnr:.9f}">
        <header>
          <div><span class="frame">{frame}</span><h2>{view_id}</h2></div>
          <div class="metric"><strong>{psnr:.2f} dB</strong><span>foreground PSNR</span></div>
        </header>
        <div class="panels">
          <figure><a href="{html.escape(images["original_native"])}"><img loading="lazy"
            src="{html.escape(images["original_preview"])}" alt="{view_id} original"></a>
            <figcaption>Original · calibrated + foreground mask</figcaption></figure>
          <figure><a href="{html.escape(images["reconstruction_native"])}"><img loading="lazy"
            src="{html.escape(images["reconstruction_preview"])}"
            alt="{view_id} StructSplat reconstruction"></a>
            <figcaption>StructSplat · {record["n_gaussians"]:,} 2D Gaussians</figcaption></figure>
          <figure><img loading="lazy" src="{html.escape(images["error_preview"])}"
            alt="{view_id} absolute error">
            <figcaption>Absolute RGB error · fixed ×{DISPLAY_ERROR_SCALE:g} scale</figcaption></figure>
        </div>
        <details><summary>Provenance and exact assets</summary>
          <dl>
            <dt>Fit window</dt><dd>x={fit[0]}, y={fit[1]}, {fit[2]}×{fit[3]}</dd>
            <dt>Foreground MAE</dt><dd>{mae:.5f}</dd>
            <dt>Renderer parity</dt><dd>max |Δ| {parity:.3g}</dd>
            <dt>Native render</dt><dd><a href="{html.escape(images["reconstruction_native"])}">open clamped exact raster</a></dd>
            <dt>Prepared original</dt><dd><a href="{html.escape(images["original_native"])}">open calibrated crop</a></dd>
            <dt>Raw original</dt><dd><a href="{html.escape(record["source_path"])}">open source JPEG</a></dd>
            <dt>Compact field</dt><dd><a href="{html.escape(record["compact_path"])}">open .rtgsv archive</a></dd>
          </dl>
        </details>
      </article>"""


def _write_index(out: Path, manifest: dict[str, Any]) -> None:
    records = manifest["views"]
    summary = manifest["summary"]
    frame_buttons = "".join(
        f'<button data-filter="{html.escape(frame)}">{html.escape(frame)} '
        f"<span>{values['views']}</span></button>"
        for frame, values in summary["frames"].items()
    )
    frame_summaries = "".join(
        f"<div><span>{html.escape(frame)}</span><strong>"
        f"{values['mean_foreground_clamped_psnr_db']:.2f} dB</strong>"
        f"<small>{values['min_foreground_clamped_psnr_db']:.2f}–"
        f"{values['max_foreground_clamped_psnr_db']:.2f} dB</small></div>"
        for frame, values in summary["frames"].items()
    )
    cards = "\n".join(_card(record) for record in records)
    generated = html.escape(manifest["generated_at"])
    source_revision = html.escape(manifest["structsplat"]["git_revision"][:12])
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StructSplat compact teacher gallery</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0a0d12; --panel:#121721; --line:#263043;
      --muted:#9daac0; --text:#eef3fb; --accent:#77d6c7; --warm:#ffce73; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 20% -10%,#1c2938 0,#0a0d12 36rem);
      color:var(--text); font:15px/1.45 Inter,ui-sans-serif,system-ui,sans-serif; }}
    a {{ color:var(--accent); }}
    .shell {{ width:min(1800px,calc(100% - 32px)); margin:0 auto; }}
    .hero {{ padding:48px 0 24px; }}
    .eyebrow {{ color:var(--accent); font-weight:700; letter-spacing:.13em;
      text-transform:uppercase; font-size:12px; }}
    h1 {{ margin:7px 0 12px; font-size:clamp(30px,5vw,58px); line-height:1.02; max-width:1050px; }}
    .lead {{ color:var(--muted); max-width:1000px; font-size:17px; }}
    .notice {{ border:1px solid #5f5130; background:#201c13; color:#e8d8af; padding:12px 14px;
      border-radius:10px; max-width:1100px; }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px;
      margin:24px 0; }}
    .summary>div {{ background:rgba(18,23,33,.82); border:1px solid var(--line); border-radius:12px;
      padding:14px 16px; display:flex; flex-direction:column; }}
    .summary span,.summary small {{ color:var(--muted); }}
    .summary strong {{ font-size:24px; color:var(--warm); }}
    .controls {{ position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:8px;
      align-items:center; padding:12px 0; backdrop-filter:blur(15px); background:rgba(10,13,18,.88); }}
    button,select,input {{ border:1px solid var(--line); background:#141a25; color:var(--text);
      border-radius:8px; padding:9px 11px; font:inherit; }}
    button.active {{ border-color:var(--accent); color:var(--accent); }}
    button span {{ color:var(--muted); margin-left:5px; }}
    input {{ min-width:180px; }}
    #cards {{ display:grid; gap:18px; padding:10px 0 60px; }}
    .card {{ background:rgba(18,23,33,.9); border:1px solid var(--line); border-radius:14px;
      overflow:hidden; box-shadow:0 16px 50px rgba(0,0,0,.2); }}
    .card>header {{ display:flex; justify-content:space-between; gap:20px; align-items:center;
      padding:14px 17px; border-bottom:1px solid var(--line); }}
    .card h2 {{ display:inline; margin:0 0 0 9px; font-size:20px; }}
    .frame {{ color:var(--muted); font-size:12px; border:1px solid var(--line); border-radius:20px;
      padding:4px 8px; }}
    .metric {{ text-align:right; display:flex; flex-direction:column; }}
    .metric strong {{ color:var(--warm); font-size:20px; }} .metric span {{ color:var(--muted); font-size:11px; }}
    .panels {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; background:var(--line); }}
    figure {{ margin:0; background:#05070a; min-width:0; }}
    figure img {{ display:block; width:100%; aspect-ratio:1.9/1; object-fit:contain; background:#000; }}
    figcaption {{ background:var(--panel); padding:9px 12px; color:#c6d0df; font-size:12px; }}
    details {{ padding:10px 17px 14px; color:var(--muted); }} summary {{ cursor:pointer; }}
    dl {{ display:grid; grid-template-columns:max-content 1fr; gap:5px 15px; }} dt {{ color:#d7dfeb; }} dd {{ margin:0; }}
    footer {{ color:var(--muted); border-top:1px solid var(--line); padding:24px 0 40px; }}
    [hidden] {{ display:none!important; }}
    @media(max-width:850px) {{ .panels {{ grid-template-columns:1fr; }} figure img {{ aspect-ratio:auto; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">realtime-gs · compact 2D teacher viewer</div>
      <h1>Originals vs StructSplat reconstructions</h1>
      <p class="lead">All {summary["views"]} compact views were decoded from <code>dataset/2025_03_07_stage_with_fabric</code>
      and rasterized at their native fitted-window resolution by <code>external/structsplat</code>
      revision <code>{source_revision}</code>. Originals come from <code>external/dataset</code> and
      receive the same calibrated bilinear undistortion used when the compact fields were fitted.</p>
      <p class="notice"><strong>Scope:</strong> these are 2D observation-teacher reconstructions.
      They are not the 5,000-Gaussian 3D compact-placement initialization or a refined 3DGS model.
      Main comparison panels apply the stored foreground mask to both sides; native links expose
      the unmasked fitted crop. Browser JPEGs are display assets; metrics use the float raster.</p>
      <div class="summary">
        <div><span>All views</span><strong>{summary["mean_foreground_clamped_psnr_db"]:.2f} dB</strong><small>mean foreground PSNR</small></div>
        {frame_summaries}
        <div><span>Renderer parity</span><strong>{summary["max_archive_query_vs_structsplat_render_abs_error"]:.2g}</strong><small>maximum sampled |Δ|</small></div>
      </div>
    </section>
    <nav class="controls" aria-label="Gallery controls">
      <button class="active" data-filter="all">All <span>{summary["views"]}</span></button>{frame_buttons}
      <input id="search" type="search" placeholder="Camera, e.g. C0021" aria-label="Search camera">
      <select id="sort" aria-label="Sort cards"><option value="camera">Camera order</option>
        <option value="quality-low">Lowest PSNR first</option><option value="quality-high">Highest PSNR first</option></select>
      <a href="manifest.json">manifest.json</a>
    </nav>
    <section id="cards">{cards}</section>
  </main>
  <footer><div class="shell">Generated {generated}. Error maps use one fixed ×{DISPLAY_ERROR_SCALE:g}
    mean-absolute-RGB scale across all views.</div></footer>
  <script>
    const cards = [...document.querySelectorAll('.card')];
    const container = document.querySelector('#cards');
    let frame = 'all';
    function apply() {{
      const query = document.querySelector('#search').value.trim().toLowerCase();
      for (const card of cards) card.hidden = !((frame === 'all' || card.dataset.frame === frame)
        && (!query || card.dataset.view.includes(query)));
    }}
    document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
      document.querySelectorAll('[data-filter]').forEach(item => item.classList.remove('active'));
      button.classList.add('active'); frame = button.dataset.filter; apply();
    }}));
    document.querySelector('#search').addEventListener('input', apply);
    document.querySelector('#sort').addEventListener('change', event => {{
      const mode = event.target.value;
      cards.sort((a,b) => mode === 'quality-low' ? +a.dataset.psnr - +b.dataset.psnr
        : mode === 'quality-high' ? +b.dataset.psnr - +a.dataset.psnr
        : (a.dataset.frame + a.dataset.view).localeCompare(b.dataset.frame + b.dataset.view));
      cards.forEach(card => container.appendChild(card));
    }});
  </script>
</body>
</html>
"""
    (out / "index.html").write_text(page, encoding="utf-8")


def _record_path(out: Path, frame: str, view_id: str) -> Path:
    return out / "records" / frame / f"{view_id}.json"


def _record_complete(record: dict[str, Any], out: Path, view: CompactView) -> bool:
    return record.get("compact_sha256") == view.sha256 and all(
        (out / value).is_file() for value in record.get("images", {}).values()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-root", type=Path, default=DEFAULT_COMPACT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--structsplat-root", type=Path, default=DEFAULT_STRUCTSPLAT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frames", nargs="+", default=None)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=93)
    parser.add_argument("--parity-samples", type=int, default=256)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compact_root = args.compact_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    structsplat_root = args.structsplat_root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if args.chunk <= 0 or args.preview_width <= 0 or args.parity_samples < 0 or args.threads <= 0:
        raise ValueError("chunk, preview width, and threads must be positive; parity samples >= 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("JPEG quality must be in [1,100]")
    module_root = structsplat_root / "src"
    if not (module_root / "structsplat/render.py").is_file():
        raise FileNotFoundError(f"StructSplat source checkout is incomplete: {structsplat_root}")
    sys.path.insert(0, str(module_root))
    import structsplat

    imported = Path(structsplat.__file__).resolve()
    if module_root not in imported.parents:
        raise RuntimeError(f"imported StructSplat from {imported}, expected below {module_root}")
    if out.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume to reuse verified records: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    calibration_path = source_root / "calibration_dome.json"
    compact_calibration_path = compact_root / "calibration_dome.json"
    if not calibration_path.is_file() or not compact_calibration_path.is_file():
        raise FileNotFoundError("source and compact calibration files are required")
    if _sha256(calibration_path) != _sha256(compact_calibration_path):
        raise RuntimeError("source and compact calibration files differ")
    calibration = _calibration_records(calibration_path)
    frame_dirs = sorted(path for path in compact_root.glob("frame_*") if path.is_dir())
    if args.frames is not None:
        requested = set(args.frames)
        frame_dirs = [path for path in frame_dirs if path.name in requested]
        missing = requested - {path.name for path in frame_dirs}
        if missing:
            raise ValueError(f"requested compact frames do not exist: {sorted(missing)}")
    if not frame_dirs:
        raise RuntimeError("no compact frame directories found")

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    total_views = sum(CompactDataset.load(path).n_views for path in frame_dirs)
    completed = 0
    for frame_dir in frame_dirs:
        compact = CompactDataset.load(frame_dir)
        source_frame = source_root / frame_dir.name
        if not source_frame.is_dir():
            raise FileNotFoundError(f"missing source frame: {source_frame}")
        for view in compact.views:
            completed += 1
            record_path = _record_path(out, frame_dir.name, view.view_id)
            if args.resume and record_path.is_file():
                existing = json.loads(record_path.read_text(encoding="utf-8"))
                if _record_complete(existing, out, view):
                    records.append(existing)
                    print(
                        f"[{completed:02d}/{total_views}] reuse {frame_dir.name}/{view.view_id}",
                        flush=True,
                    )
                    continue
            print(
                f"[{completed:02d}/{total_views}] render {frame_dir.name}/{view.view_id}",
                flush=True,
            )
            if view.view_id not in calibration:
                raise RuntimeError(f"missing calibration record for {view.view_id}")
            record = _render_view(
                view,
                frame_name=frame_dir.name,
                source_frame=source_frame,
                calibration_record=calibration[view.view_id],
                out=out,
                chunk=args.chunk,
                preview_width=args.preview_width,
                jpeg_quality=args.jpeg_quality,
                parity_samples=args.parity_samples,
            )
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append(record)
            print(
                f"  {record['metrics']['foreground_clamped_psnr_db']:.2f} dB | "
                f"render {record['timing']['structsplat_render_seconds']:.2f}s | "
                f"parity {record['metrics']['archive_query_vs_structsplat_render']['max_abs_error']:.3g}",
                flush=True,
            )

    records.sort(key=lambda record: (record["frame"], record["view_id"]))
    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    binding = _structsplat_source_binding(structsplat_root)
    producer_digests = sorted(
        {
            view.observation.producer_source_digest
            for frame_dir in frame_dirs
            for view in CompactDataset.load(frame_dir).views
        }
    )
    manifest = {
        "schema": "rtgs.structsplat_teacher_gallery.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "diagnostic_scope": (
            "2D compact observation-teacher reconstruction only; no 3D placement or refinement"
        ),
        "command": command,
        "repository": {
            "root": str(ROOT),
            "git_revision": _git_output(ROOT, "rev-parse", "HEAD"),
            "git_status": _git_output(ROOT, "status", "--short"),
        },
        "structsplat": {
            **binding,
            "imported_module": str(imported),
            "producer_source_digests_from_archives": producer_digests,
            "current_source_matches_archive_producer": producer_digests
            == [binding["source_digest"]],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "inputs": {
            "compact_root": str(compact_root),
            "source_root": str(source_root),
            "calibration_sha256": _sha256(calibration_path),
            "frames": [path.name for path in frame_dirs],
        },
        "display": {
            "jpeg_quality": args.jpeg_quality,
            "preview_width": args.preview_width,
            "error_scale": DISPLAY_ERROR_SCALE,
            "note": "metrics are computed from float tensors before clamping/JPEG encoding",
        },
        "summary": _summary(records),
        "wall_seconds": time.perf_counter() - started,
        "views": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_index(out, manifest)
    print(f"gallery: {out / 'index.html'}", flush=True)
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
