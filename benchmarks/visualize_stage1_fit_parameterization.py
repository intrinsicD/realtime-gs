"""Replay the N78 terminal 2D fits with the GaussianImage gsplat CUDA kernel.

This is a visualization-only adapter.  It does not refit either arm and it does
not lift the 2D Gaussians into the repository's 3D alpha-compositing backend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

EXPECTED_RAW_SHA256 = "028c93f350b30b61debebd5bf0706ff128f2c54faaee04614d1ee12191a3aeb7"
FINAL_STEP = 120
BLOCK_NAMES = {0: "Appearance-only", 1: "Joint Stage 1"}
ARM_NAMES = {0: "Current 9p", 1: "Candidate 8p"}
BG = (248, 249, 251)
INK = (31, 41, 55)
MUTED = (75, 85, 99)
BORDER = (209, 213, 219)
BLUE = (30, 94, 152)
ORANGE = (190, 108, 31)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("benchmarks/results/20260716T101608Z_cpu_stage1_fit_parameterization_RAW.npz"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "benchmarks/results/20260716T101608Z_cpu_stage1_fit_parameterization_GSPLAT_REPLAY"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    delta = reference.astype(np.float64) - estimate.astype(np.float64)
    mse = float(np.mean(delta * delta))
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _rgb_image(array: np.ndarray, scale: int) -> Image.Image:
    values = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(values)
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _paste_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    array: np.ndarray,
    xy: tuple[int, int],
    *,
    scale: int,
    caption: str,
    accent: tuple[int, int, int] = BORDER,
    caption_size: int = 15,
) -> None:
    image = _rgb_image(array, scale)
    x, y = xy
    canvas.paste(image, (x, y))
    draw.rectangle(
        (x - 1, y - 1, x + image.width, y + image.height),
        outline=accent,
        width=2,
    )
    draw.text(
        (x + image.width // 2, y + image.height + 5),
        caption,
        font=_font(caption_size),
        fill=MUTED,
        anchor="ma",
    )


def _load_terminal(raw_path: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    checkpoint_fields = (
        "fit_index",
        "step",
        "built_xy",
        "built_chol",
        "built_amplitude",
        "target",
        "render",
        "clamped_render",
        "psnr",
    )
    fit_fields = (
        "fit_index",
        "block_index",
        "seed_position",
        "seed",
        "local_view",
        "original_view",
        "arm_code",
    )
    with np.load(raw_path, allow_pickle=False) as archive:
        checkpoint = {
            field: archive[f"scientific/checkpoint/{field}"] for field in checkpoint_fields
        }
        fit = {field: archive[f"scientific/fit/{field}"] for field in fit_fields}

    terminal_rows = np.flatnonzero(checkpoint["step"] == FINAL_STEP)
    if terminal_rows.size != 108:
        raise RuntimeError(f"expected 108 terminal rows, found {terminal_rows.size}")

    fit_row_by_index = {
        int(fit_index): row for row, fit_index in enumerate(fit["fit_index"].tolist())
    }
    records: list[dict[str, Any]] = []
    for checkpoint_row in terminal_rows.tolist():
        fit_index = int(checkpoint["fit_index"][checkpoint_row])
        fit_row = fit_row_by_index[fit_index]
        record: dict[str, Any] = {
            "checkpoint_row": checkpoint_row,
            "fit_index": fit_index,
        }
        for field in fit_fields[1:]:
            record[field] = int(fit[field][fit_row])
        records.append(record)

    records.sort(
        key=lambda record: (
            record["block_index"],
            record["seed_position"],
            record["local_view"],
            record["arm_code"],
        )
    )
    return records, checkpoint


def _render_all(
    records: list[dict[str, Any]],
    checkpoint: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    try:
        import gsplat
        from gsplat import project_gaussians_2d_covariance, rasterize_gaussians_plus
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "the installed GaussianImage gsplat fork could not load; on this machine run with "
            "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        ) from error

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this replay requires a CUDA device")

    replays: dict[int, np.ndarray] = {}
    frame_metrics: list[dict[str, Any]] = []
    clamped_parity_psnr: list[float] = []
    clamped_parity_abs: list[np.ndarray] = []
    unclamped_parity_psnr: list[float] = []
    unclamped_parity_abs: list[np.ndarray] = []
    with torch.no_grad():
        for record in records:
            row = record["checkpoint_row"]
            xy = torch.from_numpy(checkpoint["built_xy"][row]).to(device) - 0.5
            chol = torch.from_numpy(checkpoint["built_chol"][row]).to(device)
            covariance = torch.stack(
                (
                    chol[:, 0].square(),
                    chol[:, 0] * chol[:, 1],
                    chol[:, 1].square() + chol[:, 2].square(),
                ),
                dim=-1,
            )
            amplitude = torch.from_numpy(checkpoint["built_amplitude"][row]).to(device)
            opacity = torch.ones((xy.shape[0], 1), dtype=torch.float32, device=device)
            height, width = checkpoint["target"][row].shape[:2]
            tile_bounds = ((width + 15) // 16, (height + 15) // 16, 1)
            projected = project_gaussians_2d_covariance(
                xy,
                covariance,
                height,
                width,
                tile_bounds,
                clip_thresh=0.01,
                clip_coe=math.sqrt(12.0),
                radius_clip=0.0,
                isprint=False,
            )
            rendered = rasterize_gaussians_plus(
                *projected,
                amplitude,
                opacity,
                height,
                width,
                16,
                16,
                background=torch.zeros(3, dtype=torch.float32, device=device),
                radius_clip=0.0,
                isprint=False,
            )
            replay_unclamped = rendered.cpu().numpy()
            replay = np.clip(replay_unclamped, 0.0, 1.0)
            native_unclamped = checkpoint["render"][row]
            native = checkpoint["clamped_render"][row]
            target = checkpoint["target"][row]
            absolute = np.abs(replay.astype(np.float64) - native.astype(np.float64))
            unclamped_absolute = np.abs(
                replay_unclamped.astype(np.float64) - native_unclamped.astype(np.float64)
            )
            replay_parity_psnr = _psnr(native, replay)
            unclamped_replay_parity_psnr = _psnr(native_unclamped, replay_unclamped)
            replay_target_psnr = _psnr(target, replay)
            replays[record["fit_index"]] = replay
            clamped_parity_psnr.append(replay_parity_psnr)
            clamped_parity_abs.append(absolute)
            unclamped_parity_psnr.append(unclamped_replay_parity_psnr)
            unclamped_parity_abs.append(unclamped_absolute)
            frame_metrics.append(
                {
                    **{key: int(value) for key, value in record.items()},
                    "official_target_psnr_db": float(checkpoint["psnr"][row]),
                    "gsplat_target_psnr_db": replay_target_psnr,
                    "gsplat_vs_native_clamped_psnr_db": replay_parity_psnr,
                    "gsplat_vs_native_clamped_max_abs": float(absolute.max()),
                    "gsplat_vs_native_clamped_mean_abs": float(absolute.mean()),
                    "gsplat_vs_native_unclamped_psnr_db": unclamped_replay_parity_psnr,
                    "gsplat_vs_native_unclamped_max_abs": float(unclamped_absolute.max()),
                    "gsplat_vs_native_unclamped_mean_abs": float(unclamped_absolute.mean()),
                }
            )

    all_clamped_absolute = np.stack(clamped_parity_abs)
    all_unclamped_absolute = np.stack(unclamped_parity_abs)
    parity = {
        "frame_count": len(clamped_parity_psnr),
        "display_clamped": {
            "psnr_db": {
                "minimum": float(np.min(clamped_parity_psnr)),
                "median": float(np.median(clamped_parity_psnr)),
                "mean": float(np.mean(clamped_parity_psnr)),
                "maximum": float(np.max(clamped_parity_psnr)),
            },
            "absolute_error": {
                "maximum": float(all_clamped_absolute.max()),
                "mean": float(all_clamped_absolute.mean()),
            },
        },
        "unclamped": {
            "psnr_db": {
                "minimum": float(np.min(unclamped_parity_psnr)),
                "median": float(np.median(unclamped_parity_psnr)),
                "mean": float(np.mean(unclamped_parity_psnr)),
                "maximum": float(np.max(unclamped_parity_psnr)),
            },
            "absolute_error": {
                "maximum": float(all_unclamped_absolute.max()),
                "mean": float(all_unclamped_absolute.mean()),
            },
        },
    }
    renderer = {
        "package": "gsplat",
        "package_version": importlib.metadata.version("gsplat"),
        "module_path": str(Path(gsplat.__file__).resolve()),
        "api": ["project_gaussians_2d_covariance", "rasterize_gaussians_plus"],
        "semantics": "2D additive CUDA sum; built amplitude with unit opacity",
        "is_repository_3d_gsplat_backend": False,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "ld_preload": os.environ.get("LD_PRELOAD"),
    }
    return replays, frame_metrics, {"parity": parity, "renderer": renderer}


def _paired_records(
    records: list[dict[str, Any]],
) -> dict[tuple[int, int, int], dict[int, dict[str, Any]]]:
    pairs: dict[tuple[int, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (record["block_index"], record["seed"], record["local_view"])
        pairs[key][record["arm_code"]] = record
    if len(pairs) != 54 or any(set(pair) != {0, 1} for pair in pairs.values()):
        raise RuntimeError("terminal fit table does not contain 54 complete arm pairs")
    return dict(pairs)


def _draw_overview(
    path: Path,
    pairs: dict[tuple[int, int, int], dict[int, dict[str, Any]]],
    checkpoint: dict[str, np.ndarray],
    replays: dict[int, np.ndarray],
) -> None:
    scale = 4
    image_size = 48 * scale
    label_width = 210
    gap = 16
    margin = 28
    row_height = image_size + 48
    top = 142
    column_count = 5
    width = margin * 2 + label_width + column_count * image_size + (column_count - 1) * gap
    height = top + 6 * row_height + 32
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 24), "N78 terminal fits — CUDA gsplat replay", font=_font(28, bold=True), fill=INK
    )
    draw.text(
        (margin, 64),
        "Fixed local view 0 for every seed · 48×48 · 150 Gaussians · step 120",
        font=_font(17),
        fill=MUTED,
    )
    draw.text(
        (margin, 91),
        "Post-hoc 2D additive replay; brighter error pixels are larger (absolute RGB error ×4).",
        font=_font(16),
        fill=MUTED,
    )
    headers = ("Target", "Current 9p", "Candidate 8p", "Current error", "Candidate error")
    for column, header in enumerate(headers):
        x = margin + label_width + column * (image_size + gap) + image_size // 2
        color = BLUE if column in (1, 3) else ORANGE if column in (2, 4) else INK
        draw.text((x, 122), header, font=_font(16, bold=True), fill=color, anchor="ma")

    selected = [pair for key, pair in sorted(pairs.items()) if key[2] == 0]
    for row_index, pair in enumerate(selected):
        current = pair[0]
        candidate = pair[1]
        current_row = current["checkpoint_row"]
        candidate_row = candidate["checkpoint_row"]
        target = checkpoint["target"][current_row]
        if not np.array_equal(target, checkpoint["target"][candidate_row]):
            raise RuntimeError("paired arms do not share an exact target")
        current_replay = replays[current["fit_index"]]
        candidate_replay = replays[candidate["fit_index"]]
        y = top + row_index * row_height
        draw.rectangle(
            (margin, y, margin + 5, y + image_size), fill=BLUE if row_index < 3 else ORANGE
        )
        draw.multiline_text(
            (margin + 18, y + 45),
            (
                f"{BLOCK_NAMES[current['block_index']]}\n"
                f"seed {current['seed']}\nview {current['original_view']}"
            ),
            font=_font(16, bold=True),
            fill=INK,
            spacing=7,
        )
        images = (
            target,
            current_replay,
            candidate_replay,
            np.clip(np.abs(current_replay - target) * 4.0, 0.0, 1.0),
            np.clip(np.abs(candidate_replay - target) * 4.0, 0.0, 1.0),
        )
        captions = (
            "reference",
            f"{_psnr(target, current_replay):.2f} dB",
            f"{_psnr(target, candidate_replay):.2f} dB",
            "×4",
            "×4",
        )
        accents = (BORDER, BLUE, ORANGE, BLUE, ORANGE)
        for column, (image, caption, accent) in enumerate(
            zip(images, captions, accents, strict=True)
        ):
            x = margin + label_width + column * (image_size + gap)
            _paste_tile(canvas, draw, image, (x, y), scale=scale, caption=caption, accent=accent)
    canvas.save(path, optimize=True)


def _draw_all_views(
    path: Path,
    pairs: dict[tuple[int, int, int], dict[int, dict[str, Any]]],
    checkpoint: dict[str, np.ndarray],
    replays: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    scale = 3
    image_size = 48 * scale
    label_width = 215
    gap = 8
    margin = 26
    tile_height = image_size + 28
    seed_header = 56
    block_height = seed_header + 3 * tile_height + 28
    top = 118
    width = margin * 2 + label_width + 9 * image_size + 8 * gap
    height = top + 6 * block_height + 24
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 22),
        "N78 terminal fits — all 54 source images",
        font=_font(28, bold=True),
        fill=INK,
    )
    draw.text(
        (margin, 62),
        "Target and both fitted arms for every seed/view · post-hoc GaussianImage "
        "gsplat 2D CUDA replay",
        font=_font(16),
        fill=MUTED,
    )
    draw.text(
        (margin, 87),
        "Tile captions are original-view IDs and per-image PSNR; seed headers report "
        "the 9-view mean.",
        font=_font(15),
        fill=MUTED,
    )

    summaries: list[dict[str, Any]] = []
    seed_keys = sorted({(key[0], key[1]) for key in pairs})
    for seed_index, (block_index, seed) in enumerate(seed_keys):
        y0 = top + seed_index * block_height
        views = [pairs[key] for key in sorted(pairs) if key[:2] == (block_index, seed)]
        current_scores = []
        candidate_scores = []
        for pair in views:
            current = pair[0]
            candidate = pair[1]
            target = checkpoint["target"][current["checkpoint_row"]]
            current_scores.append(_psnr(target, replays[current["fit_index"]]))
            candidate_scores.append(_psnr(target, replays[candidate["fit_index"]]))
        current_mean = float(np.mean(current_scores))
        candidate_mean = float(np.mean(candidate_scores))
        summaries.append(
            {
                "block_index": block_index,
                "block": BLOCK_NAMES[block_index],
                "seed": seed,
                "view_count": len(views),
                "current_mean_gsplat_psnr_db": current_mean,
                "candidate_mean_gsplat_psnr_db": candidate_mean,
                "candidate_minus_current_mean_gsplat_psnr_db": candidate_mean - current_mean,
            }
        )
        draw.rectangle(
            (margin, y0 + 4, margin + 6, y0 + 43), fill=BLUE if block_index == 0 else ORANGE
        )
        draw.text(
            (margin + 18, y0 + 3),
            f"{BLOCK_NAMES[block_index]} · seed {seed}",
            font=_font(19, bold=True),
            fill=INK,
        )
        draw.text(
            (margin + 18, y0 + 29),
            f"mean PSNR: current {current_mean:.3f} dB · candidate {candidate_mean:.3f} dB "
            f"· Δ {candidate_mean - current_mean:+.3f} dB",
            font=_font(14),
            fill=MUTED,
        )
        rows_y = [y0 + seed_header + i * tile_height for i in range(3)]
        draw.text((margin + 8, rows_y[0] + 55), "Target", font=_font(17, bold=True), fill=INK)
        draw.text(
            (margin + 8, rows_y[1] + 48),
            "Current 9p\ngsplat",
            font=_font(17, bold=True),
            fill=BLUE,
            spacing=5,
        )
        draw.text(
            (margin + 8, rows_y[2] + 48),
            "Candidate 8p\ngsplat",
            font=_font(17, bold=True),
            fill=ORANGE,
            spacing=5,
        )
        for column, pair in enumerate(views):
            current = pair[0]
            candidate = pair[1]
            target = checkpoint["target"][current["checkpoint_row"]]
            current_replay = replays[current["fit_index"]]
            candidate_replay = replays[candidate["fit_index"]]
            x = margin + label_width + column * (image_size + gap)
            view = current["original_view"]
            _paste_tile(canvas, draw, target, (x, rows_y[0]), scale=scale, caption=f"view {view}")
            _paste_tile(
                canvas,
                draw,
                current_replay,
                (x, rows_y[1]),
                scale=scale,
                caption=f"v{view} · {current_scores[column]:.2f} dB",
                accent=BLUE,
            )
            _paste_tile(
                canvas,
                draw,
                candidate_replay,
                (x, rows_y[2]),
                scale=scale,
                caption=f"v{view} · {candidate_scores[column]:.2f} dB",
                accent=ORANGE,
            )
    canvas.save(path, optimize=True)
    return summaries


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = _parse_args()
    raw_path = args.raw.resolve()
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_sha256 = _sha256(raw_path)
    if raw_sha256 != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            f"raw archive SHA-256 mismatch: expected {EXPECTED_RAW_SHA256}, got {raw_sha256}"
        )

    records, checkpoint = _load_terminal(raw_path)
    device = torch.device(args.device)
    replays, frame_metrics, replay_info = _render_all(records, checkpoint, device)
    pairs = _paired_records(records)
    overview_path = output_prefix.with_name(f"{output_prefix.name}_OVERVIEW.png")
    all_views_path = output_prefix.with_name(f"{output_prefix.name}_ALL_VIEWS.png")
    json_path = output_prefix.with_suffix(".json")
    _draw_overview(overview_path, pairs, checkpoint, replays)
    summaries = _draw_all_views(all_views_path, pairs, checkpoint, replays)

    script_path = Path(__file__).resolve()
    manifest = {
        "artifact_type": "stage1_fit_parameterization_gsplat_visualization",
        "valid": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "post-hoc visualization replay; no refit, lift, timing, or default claim",
        "source": {
            "raw_path": str(raw_path),
            "raw_sha256": raw_sha256,
            "terminal_step": FINAL_STEP,
            "terminal_fit_count": len(records),
            "paired_source_image_count": len(pairs),
        },
        "adapter": {
            "script_path": str(script_path),
            "script_sha256": _sha256(script_path),
            "coordinate_mapping": "saved pixel-center xy minus 0.5 pixels",
            "covariance_mapping": "[l00^2, l00*l10, l10^2+l11^2]",
            "appearance_mapping": "saved built_amplitude with unit opacity",
            "support": (
                "projection clip_coe=sqrt(12); gsplat raster support threshold remains 1/255"
            ),
        },
        "chart_contract": {
            "question": (
                "How do both N78 terminal fits compare with their targets under a CUDA "
                "gsplat replay?"
            ),
            "family": "faceted small-multiple image comparison",
            "grain": "108 terminal fits from 54 paired source images across six seeds",
            "overview_selection": "local_view == 0 for every seed; fixed before visualization",
            "all_views_selection": "all 54 paired source images",
            "palette": (
                "native RGB; blue current labels; orange candidate labels; RGB "
                "absolute-error panels"
            ),
            "upscaling": "nearest-neighbor only",
        },
        **replay_info,
        "seed_summaries": summaries,
        "frames": frame_metrics,
        "outputs": {
            "overview_png": str(overview_path),
            "overview_sha256": _sha256(overview_path),
            "all_views_png": str(all_views_path),
            "all_views_sha256": _sha256(all_views_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_revision": _git_revision(),
            "working_tree_was_assumed_clean": False,
            "command": [str(Path(sys.executable).resolve()), *sys.argv],
        },
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({"json": str(json_path), **manifest["outputs"], **manifest["parity"]}, indent=2)
    )


if __name__ == "__main__":
    main()
