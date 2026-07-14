"""Reconstruction artifacts for visual inspection and regression review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.render.base import get_rasterizer


def save_reconstruction_artifacts(
    scene: SceneData,
    initial: Gaussians3D,
    final: Gaussians3D,
    output_dir: str | Path,
    rasterizer: str = "auto",
    max_comparisons: int = 12,
    max_animation_frames: int = 48,
) -> dict[str, str]:
    """Save reference/init/final/error views, a contact sheet, and a turntable GIF.

    The GIF follows the calibrated capture cameras.  For object-centric captures this is a more
    trustworthy first inspection than inventing an orbit axis that may not match dataset world
    coordinates.
    """
    output = Path(output_dir)
    render_dir = output / "renders"
    for name in ("reference", "initial", "final", "error", "comparison"):
        (render_dir / name).mkdir(parents=True, exist_ok=True)

    device = final.means.device
    scene = scene.to(device)
    initial = initial.to(device)
    renderer = get_rasterizer(rasterizer, device=device)
    comparisons: list[Image.Image] = []
    final_frames: list[Image.Image] = []
    preferred = scene.testing_views + [
        index for index in scene.training_views if index not in scene.testing_views
    ]
    comparison_indices = _evenly_spaced(preferred, max_comparisons)
    animation_indices = _evenly_spaced(list(range(scene.n_views)), max_animation_frames)
    render_indices = comparison_indices | animation_indices
    names = scene.view_names or [f"view_{i:04d}" for i in range(scene.n_views)]

    with torch.no_grad():
        for index, (camera, target) in enumerate(zip(scene.cameras, scene.images)):
            if index not in render_indices:
                continue
            init_image = renderer.render(initial, camera).color.clamp(0, 1)
            final_image = renderer.render(final, camera).color.clamp(0, 1)
            reference = target
            if scene.masks is not None:
                reference = reference * scene.masks[index].to(reference)[..., None]
            error = (final_image - reference).abs().mul(4).clamp(0, 1)
            safe_name = Path(names[index]).stem
            images = {
                "reference": _pil(reference),
                "initial": _pil(init_image),
                "final": _pil(final_image),
                "error": _pil(error),
            }
            for kind, image in images.items():
                image.save(render_dir / kind / f"{safe_name}.png")
            if index in comparison_indices:
                comparison = _labeled_row(images, safe_name)
                comparison.save(render_dir / "comparison" / f"{safe_name}.png")
                comparisons.append(comparison)
            if index in animation_indices:
                final_frames.append(images["final"])

    contact_path = output / "reconstruction_contact_sheet.png"
    _contact_sheet(comparisons, columns=2).save(contact_path)
    gif_path = output / "reconstruction.gif"
    gif_frames = [_resize_for_gif(frame, max_side=640) for frame in final_frames]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    return {"contact_sheet": str(contact_path), "turntable": str(gif_path)}


def _pil(image: torch.Tensor) -> Image.Image:
    array = (image.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _labeled_row(images: dict[str, Image.Image], view_name: str) -> Image.Image:
    labels = ["reference", "initial", "final", "error x4"]
    panels = [images["reference"], images["initial"], images["final"], images["error"]]
    width, height = panels[0].size
    header = 24
    result = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(result)
    for index, (label, panel) in enumerate(zip(labels, panels)):
        result.paste(panel, (index * width, header))
        draw.text((index * width + 4, 5), f"{view_name} — {label}", fill="black")
    return result


def _contact_sheet(images: list[Image.Image], columns: int) -> Image.Image:
    if not images:
        raise ValueError("cannot build a contact sheet without images")
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    return sheet


def _evenly_spaced(indices: list[int], count: int) -> set[int]:
    if len(indices) <= count:
        return set(indices)
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return {indices[int(position)] for position in positions}


def _resize_for_gif(image: Image.Image, max_side: int) -> Image.Image:
    scale = min(1.0, max_side / max(image.size))
    if scale == 1.0:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)
