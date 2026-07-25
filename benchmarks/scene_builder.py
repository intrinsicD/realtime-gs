"""Resolution-parameterized compact-teacher scene construction shared by the GPU harnesses.

``beam_surfel_init.build_scene`` is fixed at downscale 32 and its source hash is bound to already
published results, so it is left untouched.  This module is the same construction with the
resolution, view selection, and device made explicit, for runs that are not at the CPU screen's
scale.

Teachers are exact ``GaussianObservationField`` queries at downscaled pixel centres, masked by the
view's packed alpha — no source RGB is read, which keeps the compact-bundle contract intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rtgs.core.camera import Camera
from rtgs.data.compact_views import CompactDataset
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.scene import SceneData

DEFAULT_TRAIN_VIEWS = (0, 3, 6, 9, 12, 15, 18, 21)
DEFAULT_HELDOUT_VIEWS = (1, 13, 25)


@dataclass(frozen=True)
class SceneSplit:
    """A built scene plus the local indices of each split."""

    scene: SceneData
    train_local: list[int]
    heldout_local: list[int]
    extrapolative_local: int


def build_scene_at(
    dataset: CompactDataset,
    *,
    downscale: int,
    train_views: tuple[int, ...] = DEFAULT_TRAIN_VIEWS,
    heldout_views: tuple[int, ...] = DEFAULT_HELDOUT_VIEWS,
    extrapolative_view: int | None = None,
    query_chunk: int = 8192,
    component_chunk: int = 2048,
) -> SceneSplit:
    """Build a train + held-out scene from compact teachers at an explicit downscale."""
    if downscale <= 0:
        raise ValueError("downscale must be positive")
    overlap = set(train_views) & set(heldout_views)
    if overlap:
        raise ValueError(f"train and held-out views overlap: {sorted(overlap)}")
    if extrapolative_view is None:
        extrapolative_view = heldout_views[-1]
    if extrapolative_view not in heldout_views:
        raise ValueError("extrapolative_view must be one of the held-out views")

    order = list(train_views) + list(heldout_views)
    images, masks, cameras, names = [], [], [], []
    for global_index in order:
        view = dataset.views[global_index]
        observation = view.observation
        fit_x, fit_y, width, height = observation.fit_window
        ds_width, ds_height = width // downscale, height // downscale
        if ds_width <= 0 or ds_height <= 0:
            raise ValueError(f"downscale {downscale} collapses view {view.view_id}")
        ys = (torch.arange(ds_height, dtype=torch.float32) + 0.5) * downscale + fit_y
        xs = (torch.arange(ds_width, dtype=torch.float32) + 0.5) * downscale + fit_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        colors = []
        for start in range(0, xy.shape[0], query_chunk):
            colors.append(
                observation.query(
                    xy[start : start + query_chunk], component_chunk=component_chunk
                ).color
            )
        image = torch.cat(colors).reshape(ds_height, ds_width, 3).clamp(0, 1)
        if view.alpha is None:
            raise RuntimeError(
                f"view {view.view_id} has no packed alpha; the masked protocol needs it"
            )
        alpha = view.alpha.crop_mask("cpu").float()
        alpha = (
            alpha[: ds_height * downscale, : ds_width * downscale]
            .reshape(ds_height, downscale, ds_width, downscale)
            .mean(dim=(1, 3))
        )
        images.append((image * alpha[..., None]).float())
        masks.append(alpha)
        camera = view.camera
        cameras.append(
            Camera(
                fx=camera.fx / downscale,
                fy=camera.fy / downscale,
                cx=(camera.cx - fit_x) / downscale,
                cy=(camera.cy - fit_y) / downscale,
                width=ds_width,
                height=ds_height,
                R=camera.R,
                t=camera.t,
            )
        )
        names.append(view.view_id)

    train_local = list(range(len(train_views)))
    heldout_local = list(range(len(train_views), len(order)))
    scene = SceneData(
        images=images,
        cameras=cameras,
        view_names=names,
        masks=masks,
        train_indices=train_local,
        test_indices=heldout_local,
        bounds_hint=dataset.bounds_hint,
        name=f"{dataset.name}-ds{downscale}",
    )
    scene.validate()
    return SceneSplit(
        scene=scene,
        train_local=train_local,
        heldout_local=heldout_local,
        extrapolative_local=len(train_views) + list(heldout_views).index(extrapolative_view),
    )


def train_inputs_at(
    dataset: CompactDataset,
    train_views: tuple[int, ...] = DEFAULT_TRAIN_VIEWS,
) -> ReconstructionInputs:
    """Reconstruction inputs restricted to the training views (what the initializer may see)."""
    complete = dataset.to_reconstruction_inputs()
    return ReconstructionInputs(
        observations=[complete.observations[i] for i in train_views],
        cameras=[complete.cameras[i] for i in train_views],
        view_names=[complete.view_names[i] for i in train_views],
        bounds_hint=complete.bounds_hint,
        name=f"{complete.name}-train",
    )


__all__ = [
    "DEFAULT_HELDOUT_VIEWS",
    "DEFAULT_TRAIN_VIEWS",
    "SceneSplit",
    "build_scene_at",
    "train_inputs_at",
]
