"""Typed, image-free inputs for field-level 2D-to-3D lifting.

``SceneFits`` is the boundary between offline per-view fitting and the field-lift stage.  It
keeps calibrated observation fields, optional lossless alpha, optional geometric priors, and an
explicit train/held-out split together.  In particular, constructing it from a
``CompactDataset`` does not pass through ``to_reconstruction_inputs()``, because that legacy
adapter intentionally drops packed alpha.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import CompactDataset, PackedAlpha
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.scene import SceneData

AlphaData: TypeAlias = PackedAlpha | torch.Tensor | None


def _split_indices(
    count: int,
    train: Sequence[int] | None,
    heldout: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    universe = set(range(count))
    train_tuple = None if train is None else tuple(int(index) for index in train)
    heldout_tuple = None if heldout is None else tuple(int(index) for index in heldout)
    if train_tuple is None and heldout_tuple is None:
        train_tuple = tuple(range(count))
        heldout_tuple = ()
    elif train_tuple is None:
        assert heldout_tuple is not None
        train_tuple = tuple(index for index in range(count) if index not in set(heldout_tuple))
    elif heldout_tuple is None:
        heldout_tuple = tuple(index for index in range(count) if index not in set(train_tuple))

    assert train_tuple is not None and heldout_tuple is not None
    if not train_tuple:
        raise ValueError("SceneFits requires at least one training view")
    if len(set(train_tuple)) != len(train_tuple) or len(set(heldout_tuple)) != len(heldout_tuple):
        raise ValueError("SceneFits split indices must be unique")
    train_set, heldout_set = set(train_tuple), set(heldout_tuple)
    if train_set & heldout_set:
        raise ValueError("SceneFits train and held-out indices must be disjoint")
    if train_set | heldout_set != universe:
        raise ValueError("SceneFits train and held-out indices must partition every view")
    return train_tuple, heldout_tuple


def _optional_per_view(
    values: Sequence[torch.Tensor | None] | None,
    count: int,
    *,
    name: str,
) -> tuple[torch.Tensor | None, ...] | None:
    if values is None:
        return None
    result = tuple(values)
    if len(result) != count:
        raise ValueError(f"{name} must contain one entry per view")
    return result


def _legacy_observation(
    gaussian: Gaussians2D,
    camera: Camera,
    *,
    view_name: str,
) -> GaussianObservationField:
    """Convert the accumulated legacy representation into an additive frozen field."""

    tensors = (gaussian.xy, gaussian.chol, gaussian.color, gaussian.weight)
    if gaussian.n <= 0:
        raise ValueError("legacy fits must contain at least one Gaussian per view")
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("legacy Gaussian tensors must be floating point")
    if any(
        tensor.device != gaussian.xy.device or tensor.dtype != gaussian.xy.dtype
        for tensor in tensors
    ):
        raise ValueError("legacy Gaussian tensors must share dtype and device")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("legacy Gaussian tensors must be finite")
    if bool((gaussian.chol[:, (0, 2)] <= 0).any()):
        raise ValueError("legacy Gaussian Cholesky diagonals must be positive")
    if bool((gaussian.weight < 0).any()):
        raise ValueError("legacy Gaussian weights must be non-negative")

    covariance = gaussian.covariance()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if bool((eigenvalues <= 0).any()):
        raise ValueError("legacy Gaussian covariance must be positive definite")
    first_axis = eigenvectors[:, :, 0]
    rotations = torch.atan2(first_axis[:, 1], first_axis[:, 0])
    return GaussianObservationField(
        width=camera.width,
        height=camera.height,
        means=gaussian.xy,
        log_scales=0.5 * eigenvalues.log(),
        rotations=rotations,
        colors=gaussian.color,
        amplitudes=gaussian.weight,
        blend_mode="additive",
        aa_dilation=0.0,
        view_id=view_name,
        n_init=gaussian.n,
        provider="synthetic_fixture",
    )


@dataclass(slots=True)
class SceneFits:
    """Ordered compact fields and all optional image-free Stage-2 evidence."""

    observations: tuple[GaussianObservationField, ...]
    cameras: tuple[Camera, ...]
    view_names: tuple[str, ...]
    alphas: tuple[AlphaData, ...]
    train_view_indices: tuple[int, ...]
    heldout_view_indices: tuple[int, ...]
    depth_priors: tuple[torch.Tensor | None, ...] | None = None
    depth_confidences: tuple[torch.Tensor | None, ...] | None = None
    neighbors: tuple[tuple[int, ...], ...] | None = None
    points: torch.Tensor | None = None
    point_visibility: tuple[torch.Tensor, ...] | None = None
    bounds_hint: tuple[torch.Tensor, float] | None = None
    geometry_is_train_only: bool = False
    name: str = "scene"

    def __post_init__(self) -> None:
        self.observations = tuple(self.observations)
        self.cameras = tuple(self.cameras)
        self.view_names = tuple(self.view_names)
        self.alphas = tuple(self.alphas)
        self.train_view_indices, self.heldout_view_indices = _split_indices(
            len(self.observations),
            self.train_view_indices,
            self.heldout_view_indices,
        )
        self.depth_priors = _optional_per_view(
            self.depth_priors,
            self.n_views,
            name="depth_priors",
        )
        self.depth_confidences = _optional_per_view(
            self.depth_confidences,
            self.n_views,
            name="depth_confidences",
        )
        if self.point_visibility is not None:
            self.point_visibility = tuple(self.point_visibility)
        if self.neighbors is not None:
            self.neighbors = tuple(tuple(int(item) for item in row) for row in self.neighbors)
        self.validate()

    @property
    def n_views(self) -> int:
        return len(self.observations)

    @property
    def training_observations(self) -> tuple[GaussianObservationField, ...]:
        return tuple(self.observations[index] for index in self.train_view_indices)

    def validate(self) -> None:
        if not self.observations:
            raise ValueError("SceneFits requires at least one observation")
        if (
            len(self.cameras) != self.n_views
            or len(self.view_names) != self.n_views
            or len(self.alphas) != self.n_views
        ):
            raise ValueError("SceneFits view-valued fields must have equal length")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SceneFits name must be non-empty")
        if not isinstance(self.geometry_is_train_only, bool):
            raise TypeError("geometry_is_train_only must be a bool")
        if len(set(self.view_names)) != self.n_views:
            raise ValueError("SceneFits view names must be unique")

        _split_indices(
            self.n_views,
            self.train_view_indices,
            self.heldout_view_indices,
        )
        for index, (name, observation, camera, alpha) in enumerate(
            zip(
                self.view_names,
                self.observations,
                self.cameras,
                self.alphas,
                strict=True,
            )
        ):
            if observation.view_id is not None and observation.view_id != name:
                raise ValueError(f"observation {index} view id does not match SceneFits ordering")
            if observation.width != camera.width or observation.height != camera.height:
                raise ValueError(f"observation {index} canvas does not match its camera")
            if isinstance(alpha, PackedAlpha):
                fit_x, fit_y, fit_width, fit_height = observation.fit_window
                if alpha.origin != (fit_x, fit_y) or alpha.shape != (
                    fit_height,
                    fit_width,
                ):
                    raise ValueError(f"packed alpha {index} does not match the observation window")
            elif torch.is_tensor(alpha):
                if alpha.shape != (camera.height, camera.width):
                    raise ValueError(f"tensor alpha {index} must match the full camera canvas")
                if alpha.is_floating_point() and not bool(torch.isfinite(alpha).all()):
                    raise ValueError(f"tensor alpha {index} must be finite")
            elif alpha is not None:
                raise TypeError("SceneFits alpha entries must be PackedAlpha, tensors, or None")

        if (self.depth_priors is None) != (self.depth_confidences is None):
            raise ValueError("depth priors and confidences must be supplied together")
        if self.depth_priors is not None and self.depth_confidences is not None:
            for index, (observation, prior, confidence) in enumerate(
                zip(
                    self.observations,
                    self.depth_priors,
                    self.depth_confidences,
                    strict=True,
                )
            ):
                if (prior is None) != (confidence is None):
                    raise ValueError(f"depth prior and confidence presence differs in view {index}")
                if prior is None:
                    continue
                assert confidence is not None
                if prior.shape != (observation.n,) or confidence.shape != (observation.n,):
                    raise ValueError(
                        "per-view depth prior/confidence must match observation cardinality"
                    )
                if (
                    not prior.is_floating_point()
                    or not confidence.is_floating_point()
                    or prior.device != observation.device
                    or confidence.device != observation.device
                    or prior.dtype != observation.dtype
                    or confidence.dtype != observation.dtype
                ):
                    raise ValueError(
                        "depth prior/confidence must share observation dtype and device"
                    )
                if (
                    not bool(torch.isfinite(prior).all())
                    or not bool(torch.isfinite(confidence).all())
                    or bool((prior <= 0).any())
                    or bool(((confidence < 0) | (confidence > 1)).any())
                ):
                    raise ValueError(
                        "depths must be positive finite and confidence must lie in [0,1]"
                    )

        if self.neighbors is not None:
            if len(self.neighbors) != self.n_views:
                raise ValueError("neighbors must contain one entry per view")
            for view, row in enumerate(self.neighbors):
                if (
                    len(set(row)) != len(row)
                    or view in row
                    or any(index < 0 or index >= self.n_views for index in row)
                ):
                    raise ValueError("neighbor rows must contain unique, valid non-self views")

        if self.points is not None and (
            self.points.ndim != 2
            or self.points.shape[1] != 3
            or not self.points.is_floating_point()
            or not bool(torch.isfinite(self.points).all())
        ):
            raise ValueError("points must be finite floating point with shape (M,3)")
        if self.point_visibility is not None:
            if self.points is None or len(self.point_visibility) != self.n_views:
                raise ValueError("point_visibility requires points and one entry per view")
            for indices in self.point_visibility:
                if indices.ndim != 1 or indices.dtype not in {torch.int32, torch.int64}:
                    raise ValueError("point visibility entries must be integer vectors")
                if indices.numel() and (
                    int(indices.min()) < 0 or int(indices.max()) >= self.points.shape[0]
                ):
                    raise ValueError("point visibility index is out of range")
        if self.bounds_hint is not None:
            center, extent = self.bounds_hint
            if (
                center.shape != (3,)
                or not center.is_floating_point()
                or not bool(torch.isfinite(center).all())
                or not math.isfinite(float(extent))
                or extent <= 0
            ):
                raise ValueError("bounds_hint must contain a finite center and positive extent")

    @classmethod
    def from_compact_dataset(
        cls,
        dataset: CompactDataset,
        *,
        train_view_indices: Sequence[int] | None = None,
        heldout_view_indices: Sequence[int] | None = None,
        depth_priors: Sequence[torch.Tensor | None] | None = None,
        depth_confidences: Sequence[torch.Tensor | None] | None = None,
        neighbors: Sequence[Sequence[int]] | None = None,
        geometry_is_train_only: bool = False,
    ) -> SceneFits:
        """Preserve compact views, including each exact ``PackedAlpha`` object."""

        train, heldout = _split_indices(
            dataset.n_views,
            train_view_indices,
            heldout_view_indices,
        )
        return cls(
            observations=tuple(view.observation for view in dataset.views),
            cameras=tuple(view.camera for view in dataset.views),
            view_names=tuple(view.view_id for view in dataset.views),
            alphas=tuple(view.alpha for view in dataset.views),
            train_view_indices=train,
            heldout_view_indices=heldout,
            depth_priors=None if depth_priors is None else tuple(depth_priors),
            depth_confidences=(None if depth_confidences is None else tuple(depth_confidences)),
            neighbors=(None if neighbors is None else tuple(tuple(row) for row in neighbors)),
            bounds_hint=dataset.bounds_hint,
            geometry_is_train_only=geometry_is_train_only,
            name=dataset.name,
        )

    @classmethod
    def from_reconstruction_inputs(
        cls,
        inputs: ReconstructionInputs,
        *,
        alphas: Sequence[AlphaData] | None = None,
        train_view_indices: Sequence[int] | None = None,
        heldout_view_indices: Sequence[int] | None = None,
        depth_priors: Sequence[torch.Tensor | None] | None = None,
        depth_confidences: Sequence[torch.Tensor | None] | None = None,
        neighbors: Sequence[Sequence[int]] | None = None,
        geometry_is_train_only: bool = False,
    ) -> SceneFits:
        train, heldout = _split_indices(
            inputs.n_views,
            train_view_indices,
            heldout_view_indices,
        )
        return cls(
            observations=tuple(inputs.observations),
            cameras=tuple(inputs.cameras),
            view_names=tuple(inputs.view_names),
            alphas=(
                tuple(None for _ in range(inputs.n_views)) if alphas is None else tuple(alphas)
            ),
            train_view_indices=train,
            heldout_view_indices=heldout,
            depth_priors=None if depth_priors is None else tuple(depth_priors),
            depth_confidences=(None if depth_confidences is None else tuple(depth_confidences)),
            neighbors=(None if neighbors is None else tuple(tuple(row) for row in neighbors)),
            points=inputs.points,
            point_visibility=(
                None if inputs.point_visibility is None else tuple(inputs.point_visibility)
            ),
            bounds_hint=inputs.bounds_hint,
            geometry_is_train_only=geometry_is_train_only,
            name=inputs.name,
        )

    @classmethod
    def from_legacy(
        cls,
        gaussians2d: Sequence[Gaussians2D],
        scene: SceneData,
        *,
        view_indices: Sequence[int] | None = None,
        depth_priors: Sequence[torch.Tensor | None] | None = None,
        depth_confidences: Sequence[torch.Tensor | None] | None = None,
        neighbors: Sequence[Sequence[int]] | None = None,
        geometry_is_train_only: bool = False,
    ) -> SceneFits:
        """Adapt legacy accumulated fits without retaining any source RGB tensor."""

        scene.validate()
        fits = tuple(gaussians2d)
        if view_indices is None:
            if len(fits) == scene.n_views:
                selected = tuple(range(scene.n_views))
            elif len(fits) == len(scene.training_views):
                selected = tuple(scene.training_views)
            else:
                raise ValueError(
                    "legacy fit count must match all views or the scene training split"
                )
        else:
            selected = tuple(int(index) for index in view_indices)
        if (
            len(selected) != len(fits)
            or len(set(selected)) != len(selected)
            or any(index < 0 or index >= scene.n_views for index in selected)
        ):
            raise ValueError("legacy view_indices must uniquely align every supplied fit")

        source_names = (
            scene.view_names
            if scene.view_names is not None
            else [f"view-{index:04d}" for index in range(scene.n_views)]
        )
        names = tuple(source_names[index] for index in selected)
        cameras = tuple(scene.cameras[index] for index in selected)
        observations = tuple(
            _legacy_observation(fit, camera, view_name=name)
            for fit, camera, name in zip(fits, cameras, names, strict=True)
        )
        original_train = set(scene.training_views)
        original_heldout = set(scene.testing_views)
        local_train = tuple(
            local for local, original in enumerate(selected) if original in original_train
        )
        local_heldout = tuple(
            local for local, original in enumerate(selected) if original in original_heldout
        )
        # A caller may intentionally adapt a subset not covered by an explicit legacy split.
        unclassified = set(range(len(selected))) - set(local_train) - set(local_heldout)
        local_train = (*local_train, *sorted(unclassified))
        local_train, local_heldout = _split_indices(
            len(selected),
            local_train,
            local_heldout,
        )
        alphas: tuple[AlphaData, ...] = (
            tuple(None for _ in selected)
            if scene.masks is None
            else tuple(scene.masks[index].detach().clone() for index in selected)
        )
        return cls(
            observations=observations,
            cameras=cameras,
            view_names=names,
            alphas=alphas,
            train_view_indices=local_train,
            heldout_view_indices=local_heldout,
            depth_priors=None if depth_priors is None else tuple(depth_priors),
            depth_confidences=(None if depth_confidences is None else tuple(depth_confidences)),
            neighbors=(None if neighbors is None else tuple(tuple(row) for row in neighbors)),
            points=scene.points,
            point_visibility=(
                None
                if scene.point_visibility is None
                else tuple(scene.point_visibility[index] for index in selected)
            ),
            bounds_hint=scene.bounds_hint,
            geometry_is_train_only=geometry_is_train_only,
            name=scene.name,
        )

    def to(self, device: torch.device | str) -> SceneFits:
        alphas: list[AlphaData] = []
        for alpha in self.alphas:
            alphas.append(alpha.to(device) if torch.is_tensor(alpha) else alpha)
        hint = None
        if self.bounds_hint is not None:
            hint = (self.bounds_hint[0].to(device), self.bounds_hint[1])
        return SceneFits(
            observations=tuple(field.to(device) for field in self.observations),
            cameras=tuple(camera.to(device) for camera in self.cameras),
            view_names=self.view_names,
            alphas=tuple(alphas),
            train_view_indices=self.train_view_indices,
            heldout_view_indices=self.heldout_view_indices,
            depth_priors=(
                None
                if self.depth_priors is None
                else tuple(
                    None if value is None else value.to(device) for value in self.depth_priors
                )
            ),
            depth_confidences=(
                None
                if self.depth_confidences is None
                else tuple(
                    None if value is None else value.to(device) for value in self.depth_confidences
                )
            ),
            neighbors=self.neighbors,
            points=None if self.points is None else self.points.to(device),
            point_visibility=(
                None
                if self.point_visibility is None
                else tuple(indices.to(device) for indices in self.point_visibility)
            ),
            bounds_hint=hint,
            geometry_is_train_only=self.geometry_is_train_only,
            name=self.name,
        )


__all__ = ["AlphaData", "SceneFits"]
