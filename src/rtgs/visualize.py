"""Reconstruction artifacts for visual inspection and regression review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.render.base import get_rasterizer


def save_reconstruction_artifacts(
    scene: SceneData,
    initial: Gaussians3D,
    final: Gaussians3D,
    output_dir: str | Path,
    rasterizer: str = "auto",
    packed: bool = False,
    antialiased: bool = False,
    max_comparisons: int = 12,
    max_animation_frames: int = 48,
) -> dict[str, str]:
    """Save calibrated comparisons plus calibrated-path and novel-orbit animations."""
    output = Path(output_dir)
    render_dir = output / "renders"
    for name in ("reference", "initial", "final", "error", "comparison"):
        (render_dir / name).mkdir(parents=True, exist_ok=True)

    device = final.means.device
    scene = scene.to(device)
    initial = initial.to(device)
    renderer = get_rasterizer(rasterizer, device=device, packed=packed, antialiased=antialiased)
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

    novel_path = output / "novel_orbit.gif"
    orbit_frames: list[Image.Image] = []
    with torch.no_grad():
        for camera in _novel_orbit_cameras(scene, max_animation_frames):
            image = renderer.render(final, camera, sh_degree=final.sh_degree).color.clamp(0, 1)
            orbit_frames.append(_resize_for_gif(_pil(image), max_side=640))
    orbit_frames[0].save(
        novel_path,
        save_all=True,
        append_images=orbit_frames[1:],
        duration=100,
        loop=0,
        optimize=False,
    )

    elevation_path = output / "novel_elevation.gif"
    elevation_frames: list[Image.Image] = []
    with torch.no_grad():
        for camera in _novel_orbit_cameras(scene, max_animation_frames, vary_elevation=True):
            image = renderer.render(final, camera, sh_degree=final.sh_degree).color.clamp(0, 1)
            elevation_frames.append(_resize_for_gif(_pil(image), max_side=640))
    elevation_frames[0].save(
        elevation_path,
        save_all=True,
        append_images=elevation_frames[1:],
        duration=100,
        loop=0,
        optimize=False,
    )
    return {
        "contact_sheet": str(contact_path),
        "turntable": str(gif_path),
        "novel_orbit": str(novel_path),
        "novel_elevation": str(elevation_path),
    }


def _novel_orbit_cameras(
    scene: SceneData, frames: int, *, vary_elevation: bool = False
) -> list[Camera]:
    """Create an object-centric orbit in the capture rig's dominant camera plane.

    The path uses the calibrated cameras to infer radius, plane, image size, and intrinsics.
    Its angles are offset by half a frame so none is an exact training camera pose.
    """
    if frames <= 0:
        raise ValueError("novel orbit requires at least one frame")
    center, _ = scene.center_and_extent()
    positions = torch.stack([camera.position for camera in scene.cameras]).to(center)
    offsets = positions - center
    covariance = offsets.T @ offsets / max(offsets.shape[0], 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    axis_x = eigenvectors[:, -1]
    axis_y = eigenvectors[:, -2]
    normal = torch.linalg.cross(axis_x, axis_y)
    # Choose a stable handedness consistent with the capture cameras' down direction.
    camera_down = torch.stack([camera.R[1] for camera in scene.cameras]).to(center).mean(dim=0)
    if torch.dot(normal, camera_down) < 0:
        normal = -normal
        axis_y = -axis_y
    radius = offsets.norm(dim=-1).median().clamp_min(1e-3)
    height = torch.quantile((offsets @ normal).abs(), 0.9) * 0.75
    reference = scene.cameras[len(scene.cameras) // 2]
    fov_x = float(
        torch.rad2deg(2.0 * torch.atan(torch.tensor(reference.width / (2.0 * reference.fx))))
    )
    cameras = []
    for index in range(frames):
        angle = center.new_tensor(2.0 * torch.pi * (index + 0.5) / frames)
        elevation = height * torch.sin(2.0 * angle) if vary_elevation else height.new_zeros(())
        planar_radius = torch.sqrt((radius.square() - elevation.square()).clamp_min(1e-6))
        eye = center + planar_radius * (torch.cos(angle) * axis_x + torch.sin(angle) * axis_y)
        eye = eye + elevation * normal
        camera = Camera.look_at(
            eye,
            center,
            up=normal,
            fov_x_deg=fov_x,
            width=reference.width,
            height=reference.height,
        )
        camera.fx = reference.fx
        camera.fy = reference.fy
        camera.cx = reference.cx
        camera.cy = reference.cy
        cameras.append(camera.to(center.device))
    return cameras


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
