"""Initialization-only compact-view evaluation for image-free 3D placement.

This is the missing measurement behind the "weak initialization" question. It renders a candidate
3D initialization through each calibrated camera and compares it to that view's exact frozen 2D
teacher render — the RGB-free surrogate for the source image — *before* any 3DGS optimization. A
denser all-Gaussian + merge placement can therefore be scored against the balanced top-K on equal,
quantitative footing, without attributing quality recovered by photometric densification to
placement.

The comparison stays inside the RGB-free contract: the target is the compact 2D field's own
render (via the exact CSR observation query), never a decoded source image.  Everything here is
CPU-first and deterministic; the dense renderer and metrics are the repository's existing anchors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import masked_psnr, psnr, ssim
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    ObservationQueryBackend,
)
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import (
    CompactCarveConfig,
    CompactCarveInitializer,
    CompactInitializationResult,
)
from rtgs.lift.merge import merge_by_voxel

_SSIM_WINDOW = 11


@dataclass(frozen=True)
class InitViewMetrics:
    """Init-only compact-view metrics for a single calibrated view."""

    view_index: int
    view_name: str
    psnr: float  # full fit-window render-vs-teacher PSNR (includes matched background)
    foreground_psnr: float  # restricted to teacher-supported pixels
    ssim: float  # 0.0 when the fit window is smaller than the SSIM window
    teacher_coverage_fraction: float


@dataclass(frozen=True)
class InitEvaluation:
    """Aggregate init-only compact-view evaluation of one 3D initialization."""

    n_gaussians: int
    mean_psnr: float
    mean_foreground_psnr: float
    mean_ssim: float
    per_view: tuple[InitViewMetrics, ...]

    def as_dict(self) -> dict:
        return {
            "n_gaussians": self.n_gaussians,
            "mean_psnr": self.mean_psnr,
            "mean_foreground_psnr": self.mean_foreground_psnr,
            "mean_ssim": self.mean_ssim,
            "per_view": [
                {
                    "view_index": view.view_index,
                    "view_name": view.view_name,
                    "psnr": view.psnr,
                    "foreground_psnr": view.foreground_psnr,
                    "ssim": view.ssim,
                    "teacher_coverage_fraction": view.teacher_coverage_fraction,
                }
                for view in self.per_view
            ],
        }


@dataclass(frozen=True)
class InitEvaluationTarget:
    """One immutable compact teacher target reusable across candidate evaluations."""

    view_index: int
    view_name: str
    fit_window: tuple[int, int, int, int]
    teacher: torch.Tensor
    foreground: torch.Tensor


@dataclass(frozen=True)
class InitEvaluationProgress:
    """Low-overhead progress record emitted around each calibrated-view evaluation."""

    phase: str  # "view_start" or "view_complete"
    view_index: int
    view_name: str
    completed_views: int
    total_views: int
    elapsed_seconds: float
    view_seconds: float | None = None
    visible_gaussians: int | None = None


InitEvaluationProgressCallback = Callable[[InitEvaluationProgress], None]


def crop_camera_to_fit_window(camera: Camera, fit_window: tuple[int, int, int, int]) -> Camera:
    """Return the exact pinhole camera for a rectangular pixel crop.

    Rendering this camera evaluates the same pixel centers as rendering the full camera and
    slicing ``fit_window`` afterward, without paying for pixels that are never scored.
    """
    fit_x, fit_y, fit_width, fit_height = fit_window
    if fit_x < 0 or fit_y < 0 or fit_width <= 0 or fit_height <= 0:
        raise ValueError("fit_window must be a positive rectangle inside the camera")
    if fit_x + fit_width > camera.width or fit_y + fit_height > camera.height:
        raise ValueError("fit_window must remain inside the camera")
    return Camera(
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx - fit_x,
        cy=camera.cy - fit_y,
        width=fit_width,
        height=fit_height,
        R=camera.R,
        t=camera.t,
    )


def render_teacher_image(
    field: GaussianObservationField,
    *,
    backend: ObservationQueryBackend | None = None,
    row_batch: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render the exact 2D teacher over its fit window in bounded row tiles.

    Returns ``(color, coverage)`` where ``color`` is an ``(H_fit, W_fit, 3)`` image clamped to the
    displayable range and ``coverage`` is the ``(H_fit, W_fit)`` normalized-renderer weight sum
    (its positive support is the foreground mask). ``backend`` defaults to a fresh CSR index so the
    render is local and fast on production-sized fields; pass a shared index to avoid rebuilding.
    """
    if row_batch <= 0:
        raise ValueError("row_batch must be positive")
    if backend is None:
        backend = GaussianObservationIndex(field)
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    color = torch.zeros(fit_height, fit_width, 3, dtype=field.dtype)
    coverage = torch.zeros(fit_height, fit_width, dtype=field.dtype)
    xs = torch.arange(fit_width, dtype=field.dtype) + (fit_x + 0.5)
    for start in range(0, fit_height, row_batch):
        stop = min(start + row_batch, fit_height)
        ys = torch.arange(start, stop, dtype=field.dtype) + (fit_y + 0.5)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        query = backend.query(xy)
        color[start:stop] = query.color.reshape(stop - start, fit_width, 3)
        coverage[start:stop] = query.weight_sum.reshape(stop - start, fit_width)
    return color.clamp(0.0, 1.0), coverage


def prepare_evaluation_targets(
    inputs: ReconstructionInputs,
    *,
    backends: Sequence[ObservationQueryBackend] | None = None,
    row_batch: int = 64,
) -> tuple[InitEvaluationTarget, ...]:
    """Render invariant compact teachers once for reuse across candidate initializations."""
    if backends is not None and len(backends) != inputs.n_views:
        raise ValueError("backends must contain one query backend per view")
    targets = []
    for index in range(inputs.n_views):
        field = inputs.observations[index]
        backend = None if backends is None else backends[index]
        teacher, coverage = render_teacher_image(field, backend=backend, row_batch=row_batch)
        targets.append(
            InitEvaluationTarget(
                view_index=index,
                view_name=inputs.view_names[index],
                fit_window=field.fit_window,
                teacher=teacher,
                foreground=coverage > 0,
            )
        )
    return tuple(targets)


def _validate_evaluation_targets(
    inputs: ReconstructionInputs,
    targets: Sequence[InitEvaluationTarget],
) -> None:
    if len(targets) != inputs.n_views:
        raise ValueError("targets must contain one prepared target per view")
    for index, target in enumerate(targets):
        field = inputs.observations[index]
        if (
            target.view_index != index
            or target.view_name != inputs.view_names[index]
            or target.fit_window != field.fit_window
        ):
            raise ValueError(f"prepared target {index} does not match the reconstruction inputs")
        fit_width, fit_height = field.fit_window[2:]
        if target.teacher.shape != (fit_height, fit_width, 3):
            raise ValueError(f"prepared target {index} has an invalid teacher shape")
        if (
            target.foreground.shape != (fit_height, fit_width)
            or target.foreground.dtype != torch.bool
        ):
            raise ValueError(f"prepared target {index} has an invalid foreground mask")


def evaluate_initialization(
    inputs: ReconstructionInputs,
    gaussians: Gaussians3D,
    *,
    backends: list[ObservationQueryBackend] | None = None,
    rasterizer=None,
    background: torch.Tensor | None = None,
    row_batch: int = 64,
    progress_callback: InitEvaluationProgressCallback | None = None,
    targets: Sequence[InitEvaluationTarget] | None = None,
) -> InitEvaluation:
    """Score a 3D initialization against every view's exact 2D teacher render.

    ``backends`` optionally supplies one shared observation query backend per view (a CSR index),
    reused for the teacher render.  ``rasterizer`` defaults to the pure-PyTorch CPU reference
    :class:`rtgs.render.torch_ref.TorchRasterizer`; the compact target and the 3D render therefore
    both stay on the CPU correctness anchor. ``targets`` can supply teachers prepared once by
    :func:`prepare_evaluation_targets` when several candidates share the same compact inputs.
    """
    if backends is not None and len(backends) != inputs.n_views:
        raise ValueError("backends must contain one query backend per view")
    if targets is not None:
        _validate_evaluation_targets(inputs, targets)
    if rasterizer is None:
        from rtgs.render.torch_ref import TorchRasterizer

        rasterizer = TorchRasterizer()

    per_view: list[InitViewMetrics] = []
    started = perf_counter() if progress_callback is not None else None
    for index in range(inputs.n_views):
        field = inputs.observations[index]
        fit_window = field.fit_window
        camera = crop_camera_to_fit_window(inputs.cameras[index], fit_window).to(
            gaussians.means.device
        )
        backend = None if backends is None else backends[index]
        if progress_callback is not None:
            assert started is not None
            view_started = perf_counter()
            progress_callback(
                InitEvaluationProgress(
                    phase="view_start",
                    view_index=index,
                    view_name=inputs.view_names[index],
                    completed_views=index,
                    total_views=inputs.n_views,
                    elapsed_seconds=view_started - started,
                )
            )
        if targets is None:
            teacher, coverage = render_teacher_image(field, backend=backend, row_batch=row_batch)
            foreground = (coverage > 0).to(teacher.dtype)
        else:
            teacher = targets[index].teacher
            foreground = targets[index].foreground.to(teacher.dtype)
        render_output = rasterizer.render(gaussians, camera, background=background)
        crop = render_output.color.detach().cpu().clamp(0.0, 1.0).to(teacher.dtype)
        fit_width, fit_height = fit_window[2], fit_window[3]
        full_psnr = psnr(crop, teacher)
        if float(foreground.sum()) > 0:
            foreground_psnr = masked_psnr(crop, teacher, foreground)
        else:
            foreground_psnr = full_psnr
        if min(fit_height, fit_width) >= _SSIM_WINDOW:
            view_ssim = float(ssim(crop, teacher))
        else:
            view_ssim = 0.0
        per_view.append(
            InitViewMetrics(
                view_index=index,
                view_name=inputs.view_names[index],
                psnr=full_psnr,
                foreground_psnr=foreground_psnr,
                ssim=view_ssim,
                teacher_coverage_fraction=float(foreground.mean()),
            )
        )
        if progress_callback is not None:
            assert started is not None
            completed = perf_counter()
            visible = render_output.visible
            progress_callback(
                InitEvaluationProgress(
                    phase="view_complete",
                    view_index=index,
                    view_name=inputs.view_names[index],
                    completed_views=index + 1,
                    total_views=inputs.n_views,
                    elapsed_seconds=completed - started,
                    view_seconds=completed - view_started,
                    visible_gaussians=None if visible is None else int(visible.numel()),
                )
            )

    count = max(len(per_view), 1)
    return InitEvaluation(
        n_gaussians=gaussians.n,
        mean_psnr=sum(view.psnr for view in per_view) / count,
        mean_foreground_psnr=sum(view.foreground_psnr for view in per_view) / count,
        mean_ssim=sum(view.ssim for view in per_view) / count,
        per_view=tuple(per_view),
    )


def merge_initialization(
    result: CompactInitializationResult,
    voxel_size: float,
    *,
    color_bin_size: float | None = None,
    opacity_mode: str = "union",
    weight_by_score: bool = True,
) -> tuple[Gaussians3D, torch.Tensor]:
    """Deduplicate a dense per-view initialization with the voxel-hash moment merge.

    Returns ``(merged, group)`` where ``group`` maps each input Gaussian to its merged output
    row.  Composed with :attr:`CompactInitializationResult.lineage`, ``group`` is the cross-view
    correspondence byproduct: inputs sharing a cluster are the same surface patch seen from
    different cameras. Confidence weighting uses the placement score by default.
    """
    weights = result.scores.clamp_min(torch.finfo(result.gaussians.means.dtype).tiny)
    return merge_by_voxel(
        result.gaussians,
        voxel_size,
        opacity_mode=opacity_mode,
        component_weights=weights if weight_by_score else None,
        color_bin_size=color_bin_size,
        return_group=True,
    )


def dense_merged_initialization(
    inputs: ReconstructionInputs,
    config: CompactCarveConfig,
    *,
    merge_voxel_size: float,
    backends: list[ObservationQueryBackend] | None = None,
    color_bin_size: float | None = None,
    opacity_mode: str = "union",
    weight_by_score: bool = True,
) -> tuple[Gaussians3D, CompactInitializationResult, torch.Tensor]:
    """Lift every supported 2D Gaussian, then merge to a deduplicated dense initialization.

    ``config`` must enable ``select_all_eligible``; otherwise the caller would silently get the
    sparse top-K placement.  Returns ``(merged_gaussians, dense_result, group)``.
    """
    if not config.select_all_eligible:
        raise ValueError("dense_merged_initialization requires config.select_all_eligible=True")
    result = CompactCarveInitializer(config).initialize(inputs, backends=backends)
    merged, group = merge_initialization(
        result,
        merge_voxel_size,
        color_bin_size=color_bin_size,
        opacity_mode=opacity_mode,
        weight_by_score=weight_by_score,
    )
    return merged, result, group


__all__ = [
    "InitEvaluation",
    "InitEvaluationProgress",
    "InitEvaluationProgressCallback",
    "InitEvaluationTarget",
    "InitViewMetrics",
    "crop_camera_to_fit_window",
    "dense_merged_initialization",
    "evaluate_initialization",
    "merge_initialization",
    "prepare_evaluation_targets",
    "render_teacher_image",
]
