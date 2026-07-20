"""RGB-free fixed-topology refinement against frozen 2D Gaussian fields.

This module is deliberately separate from :mod:`rtgs.optim.trainer`: its supervision
surface is a list of compact observation fields and calibrated cameras, never dense RGB
images.  Both teacher and student are evaluated only at explicit image-plane points.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    GaussianPixelProposal,
    GaussianPointProposal,
    ObservationQuery,
    ObservationQueryBackend,
    ObservationSamples,
    fixed_attempt_mean,
)
from rtgs.core.sh import DEFAULT_SMU1_MU
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.render.base import DEFAULT_VISIBILITY_MARGIN_SIGMA
from rtgs.render.point_base import PointRasterizer, PointRenderOutput
from rtgs.render.torch_points import TorchPointRasterizer

ProposalMode = Literal[
    "pixel_uniform",
    "pixel_gaussian",
    "area_uniform",
    "area_gaussian",
]
ScheduleMode = Literal["iid", "balanced_cycle"]
TargetMode = Literal["uniform", "proposal_attempt"]

PROPOSAL_MODES: tuple[ProposalMode, ...] = (
    "pixel_uniform",
    "pixel_gaussian",
    "area_uniform",
    "area_gaussian",
)
SCHEDULE_MODES: tuple[ScheduleMode, ...] = ("iid", "balanced_cycle")
TARGET_MODES: tuple[TargetMode, ...] = ("uniform", "proposal_attempt")


class CompactTopologyController(Protocol):
    """Opt-in research hook for compact score collection and topology edits.

    The default compact trainer never constructs or calls this surface. Implementations
    own persistent identities and must return a complete replacement parameter dictionary
    after any post-step edit while preserving the optimizer bindings.
    """

    @property
    def persistent_ids(self) -> torch.Tensor:
        """Return one unique persistent integer identity per current parameter row."""
        ...

    def bind(
        self,
        params: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        extent: float,
        n_views: int,
        attempts_per_step: int,
    ) -> None:
        """Bind the initial parameters and immutable training dimensions."""
        ...

    def needs_compositing_color_basis(self, step: int) -> bool:
        """Whether this one-based step needs the graph-valued compositor basis."""
        ...

    def observe_pre_backward(
        self,
        *,
        step: int,
        view_index: int,
        output: PointRenderOutput,
        point_loss: torch.Tensor,
        active: torch.Tensor,
        attempts: int,
    ) -> None:
        """Contract graph diagnostics before the ordinary loss backward."""
        ...

    def observe_post_backward(
        self,
        *,
        step: int,
        view_index: int,
        output: PointRenderOutput,
        width: int,
        height: int,
    ) -> None:
        """Collect screen-space gradients immediately after ordinary backward."""
        ...

    def after_step(
        self,
        *,
        step: int,
        params: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        snapshot: Gaussians3D,
    ) -> dict[str, torch.Tensor]:
        """Observe the post-Adam state and optionally return topology-edited params."""
        ...

    def history_record(self) -> dict:
        """Return a JSON-compatible controller receipt after training."""
        ...


_GROUP_ORDER = ("means", "quats", "scales", "opacities", "sh0", "shN")
_FAMILY_NAMES = {
    "means": "means",
    "quats": "quaternions",
    "scales": "log_scales",
    "opacities": "opacity_logits",
    "sh0": "sh0",
    "shN": "shN",
}
_AREA_OFFSETS = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))


@dataclass(frozen=True)
class CompactTrainConfig:
    """Configuration for fixed-attempt compact point training.

    The defaults are CPU-first.  Official experiments override ``teacher_tile_size`` and
    checkpoint values explicitly so their complete execution contract remains visible at
    the call site.
    """

    iterations: int = 120
    attempts_per_step: int = 128
    proposal_mode: ProposalMode = "pixel_gaussian"
    schedule_mode: ScheduleMode = "iid"
    target_mode: TargetMode = "uniform"
    uniform_fraction: float = 0.25
    seed: int = 0
    extent: float | None = None
    device: str = "cpu"

    lr_means: float = 1.6e-4
    lr_quats: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3
    lr_sh_rest: float = 2.5e-3 / 20.0

    point_chunk: int = 32
    gaussian_chunk: int = 64
    outer_microbatch: int = 32
    query_component_chunk: int = 64
    teacher_tile_size: int = 16
    evaluation_chunk: int = 256
    checkpoints: tuple[int, ...] = (0, 30, 60, 120)
    evaluate_checkpoint_risks: bool = True

    sh_degree: int = 0
    sh_color_activation: str = "hard"
    sh_smu1_mu: float = DEFAULT_SMU1_MU
    kernel_support_mode: str = "hard"
    visibility_margin_sigma: float = DEFAULT_VISIBILITY_MARGIN_SIGMA

    max_views: int = 64
    max_fitted_pixels_per_view: int = 50_000_000
    max_components_per_view: int = 2_000_000
    max_index_entries_per_view: int = 16_000_000
    max_candidates_per_tile: int = 200_000
    max_manifest_bytes: int = 8_388_608
    max_teacher_archives: int = 64
    max_teacher_archive_bytes: int = 268_435_456
    max_total_teacher_archive_bytes: int = 2_147_483_648
    max_zip_members: int = 64
    max_member_uncompressed_bytes: int = 268_435_456
    max_archive_uncompressed_bytes: int = 1_073_741_824

    def __post_init__(self) -> None:
        if self.proposal_mode not in PROPOSAL_MODES:
            choices = ", ".join(PROPOSAL_MODES)
            raise ValueError(f"unknown compact proposal mode '{self.proposal_mode}' ({choices})")
        if self.schedule_mode not in SCHEDULE_MODES:
            choices = ", ".join(SCHEDULE_MODES)
            raise ValueError(f"unknown compact schedule mode '{self.schedule_mode}' ({choices})")
        if self.target_mode not in TARGET_MODES:
            choices = ", ".join(TARGET_MODES)
            raise ValueError(f"unknown compact target mode '{self.target_mode}' ({choices})")
        if self.target_mode == "proposal_attempt" and self.proposal_mode != "area_gaussian":
            raise ValueError("proposal_attempt target requires proposal_mode='area_gaussian'")
        integer_fields = (
            "iterations",
            "attempts_per_step",
            "point_chunk",
            "gaussian_chunk",
            "outer_microbatch",
            "query_component_chunk",
            "teacher_tile_size",
            "evaluation_chunk",
            "max_views",
            "max_fitted_pixels_per_view",
            "max_components_per_view",
            "max_index_entries_per_view",
            "max_candidates_per_tile",
            "max_manifest_bytes",
            "max_teacher_archives",
            "max_teacher_archive_bytes",
            "max_total_teacher_archive_bytes",
            "max_zip_members",
            "max_member_uncompressed_bytes",
            "max_archive_uncompressed_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not math.isfinite(self.uniform_fraction) or not 0.0 < self.uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be finite and in (0,1]")
        if self.extent is not None and (not math.isfinite(self.extent) or self.extent <= 0.0):
            raise ValueError("extent must be finite and positive when supplied")
        for name in (
            "lr_means",
            "lr_quats",
            "lr_scales",
            "lr_opacity",
            "lr_sh",
            "lr_sh_rest",
            "sh_smu1_mu",
            "visibility_margin_sigma",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.sh_degree, bool) or not isinstance(self.sh_degree, int):
            raise ValueError("sh_degree must be a non-negative integer")
        if self.sh_degree < 0 or self.sh_degree > 3:
            raise ValueError("sh_degree must be in [0,3]")
        checkpoints = tuple(self.checkpoints)
        object.__setattr__(self, "checkpoints", checkpoints)
        if (
            not checkpoints
            or checkpoints[0] != 0
            or checkpoints[-1] != self.iterations
            or tuple(sorted(set(checkpoints))) != checkpoints
        ):
            raise ValueError("checkpoints must be unique, sorted, and include 0 and iterations")
        if not isinstance(self.evaluate_checkpoint_risks, bool):
            raise TypeError("evaluate_checkpoint_risks must be bool")


def step_sample_seed(seed: int, step: int) -> int:
    """Return a stable proposal seed independent of mode and earlier RNG consumption."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    payload = f"rtgs.compact-point.sample.v1\0{seed}\0{step}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def build_view_schedule(
    n_views: int,
    iterations: int,
    seed: int,
    *,
    mode: ScheduleMode = "iid",
) -> tuple[int, ...]:
    """Generate a complete view schedule without sharing the proposal RNG stream.

    ``iid`` retains the original schedule byte-for-byte. ``balanced_cycle`` uses an
    independently seeded permutation for each cycle, so every full cycle visits every view
    once and extending ``iterations`` preserves the existing prefix.
    """
    if isinstance(n_views, bool) or not isinstance(n_views, int) or n_views <= 0:
        raise ValueError("n_views must be a positive integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if mode not in SCHEDULE_MODES:
        choices = ", ".join(SCHEDULE_MODES)
        raise ValueError(f"unknown compact schedule mode '{mode}' ({choices})")
    if mode == "iid":
        payload = f"rtgs.compact-point.views.v1\0{seed}".encode()
        view_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(view_seed)
        return tuple(
            int(value) for value in torch.randint(n_views, (iterations,), generator=generator)
        )

    schedule: list[int] = []
    cycle = 0
    while len(schedule) < iterations:
        payload = f"rtgs.compact-point.views-balanced.v1\0{seed}\0{cycle}".encode()
        cycle_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
            (1 << 63) - 1
        )
        generator = torch.Generator(device="cpu").manual_seed(cycle_seed)
        schedule.extend(int(value) for value in torch.randperm(n_views, generator=generator))
        cycle += 1
    return tuple(schedule[:iterations])


def _update_tensor_digest(digest: hashlib._Hash, name: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(detached.dtype).encode())
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode())
    digest.update(detached.numpy().tobytes(order="C"))


def observation_digest(inputs: ReconstructionInputs) -> str:
    """Hash all frozen teacher values and semantics in deterministic view order."""
    digest = hashlib.sha256()
    for view, (name, field) in enumerate(zip(inputs.view_names, inputs.observations, strict=True)):
        metadata = {
            "view": view,
            "name": name,
            "width": field.width,
            "height": field.height,
            "blend_mode": field.blend_mode,
            "epsilon": field.epsilon,
            "sigma_cutoff": field.sigma_cutoff,
            "support_fade_alpha": field.support_fade_alpha,
            "aa_dilation": field.aa_dilation,
            "view_id": field.view_id,
            "fit_window": field.fit_window,
            "n_init": field.n_init,
            "provider": field.provider,
            "producer_version": field.producer_version,
            "producer_source_digest": field.producer_source_digest,
            "fit_config_digest": field.fit_config_digest,
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        for tensor_name in ("means", "log_scales", "rotations", "colors", "amplitudes"):
            _update_tensor_digest(digest, tensor_name, getattr(field, tensor_name))
        for tensor_name in ("color_grads", "filter_variance"):
            tensor = getattr(field, tensor_name)
            digest.update(tensor_name.encode())
            if tensor is None:
                digest.update(b"none")
            else:
                _update_tensor_digest(digest, tensor_name, tensor)
        digest.update(b"mean_residuals")
        if field.mean_residuals is None:
            digest.update(b"\x00")
        else:
            digest.update(b"\x01")
            _update_tensor_digest(digest, "mean_residuals", field.mean_residuals)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _gaussians_sha256(gaussians: Gaussians3D) -> str:
    digest = hashlib.sha256()
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        digest.update(name.encode())
        digest.update(_tensor_sha256(getattr(gaussians, name)).encode())
    return digest.hexdigest()


def _archive_stats_record(
    inputs: ReconstructionInputs,
    config: CompactTrainConfig,
    bundle_path: str | Path | None,
) -> dict | None:
    """Validate the strict loader's retained central-directory evidence."""
    stats = inputs.archive_stats
    if stats is None:
        if bundle_path is not None:
            raise ValueError("bundle_path requires inputs loaded with strict archive preflight")
        return None
    if stats.manifest_bytes > config.max_manifest_bytes:
        raise ValueError("bundle manifest exceeds compact trainer safety cap")
    if len(stats.teacher_archives) > config.max_teacher_archives:
        raise ValueError("teacher archive count exceeds compact trainer safety cap")
    if len(stats.teacher_archives) != inputs.n_views:
        raise ValueError("strict archive evidence does not match observation view count")
    archives = []
    for archive in stats.archives:
        if archive.compressed_bytes > config.max_teacher_archive_bytes:
            raise ValueError("bundle archive exceeds compact trainer compressed-byte cap")
        if archive.member_count > config.max_zip_members:
            raise ValueError("bundle archive exceeds compact trainer ZIP-member cap")
        if archive.max_member_uncompressed_bytes > config.max_member_uncompressed_bytes:
            raise ValueError("bundle archive member exceeds compact trainer uncompressed cap")
        if archive.uncompressed_bytes > config.max_archive_uncompressed_bytes:
            raise ValueError("bundle archive exceeds compact trainer uncompressed-byte cap")
        archives.append(
            {
                "relative_path": archive.relative_path,
                "compressed_bytes": archive.compressed_bytes,
                "member_count": archive.member_count,
                "uncompressed_bytes": archive.uncompressed_bytes,
                "max_member_uncompressed_bytes": archive.max_member_uncompressed_bytes,
            }
        )
    if stats.total_compressed_bytes > config.max_total_teacher_archive_bytes:
        raise ValueError("bundle archives exceed compact trainer aggregate compressed-byte cap")
    return {
        "bundle_path": (
            None if bundle_path is None else str(Path(bundle_path).resolve(strict=True))
        ),
        "manifest_bytes": stats.manifest_bytes,
        "teacher_archive_count": len(stats.teacher_archives),
        "archives": archives,
        "total_compressed_bytes": stats.total_compressed_bytes,
        "total_uncompressed_bytes": stats.total_uncompressed_bytes,
    }


def preflight_observations(
    inputs: ReconstructionInputs,
    config: CompactTrainConfig,
    bundle_path: str | Path | None = None,
) -> dict:
    """Fail closed on view, field, index-overlap, and optional archive budgets."""
    inputs.validate()
    if inputs.n_views > config.max_views:
        raise ValueError("observation view count exceeds compact trainer safety cap")
    views = []
    for index, (name, field) in enumerate(zip(inputs.view_names, inputs.observations, strict=True)):
        fitted_pixels = field.fit_window[2] * field.fit_window[3]
        if fitted_pixels > config.max_fitted_pixels_per_view:
            raise ValueError(f"view {index} fitted pixels exceed compact trainer safety cap")
        if field.n > config.max_components_per_view:
            raise ValueError(f"view {index} components exceed compact trainer safety cap")
        entries = (
            GaussianObservationIndex.estimate_entries(field, tile_size=config.teacher_tile_size)
            if field.device.type == "cpu"
            else _estimate_device_index_entries(
                field,
                tile_size=config.teacher_tile_size,
                component_chunk=config.query_component_chunk,
            )
        )
        if entries > config.max_index_entries_per_view:
            raise ValueError(f"view {index} index entries exceed compact trainer safety cap")
        views.append(
            {
                "view_index": index,
                "view_name": name,
                "fitted_pixels": fitted_pixels,
                "components": field.n,
                "estimated_index_entries": entries,
            }
        )
    archive = _archive_stats_record(inputs, config, bundle_path)
    return {"n_views": inputs.n_views, "views": views, "archive": archive}


def _validate_proposal_fields(
    teachers: Sequence[GaussianObservationField],
    proposals: Sequence[GaussianObservationField],
    view_names: Sequence[str],
) -> None:
    """Require an explicit density field to preserve each teacher's sampling geometry.

    The proposal amplitude may differ (for example, anchor amplitude times a compact
    center-occupancy scalar), and proposal colors are deliberately unconstrained because they
    are never supervision. Geometry and support remain exact so density changes cannot silently
    alter coordinates, domains, or kernel semantics.
    """
    if len(proposals) != len(teachers):
        raise ValueError("proposal_fields must contain exactly one field per teacher view")
    if len(view_names) != len(teachers):
        raise ValueError("view_names must align with teacher fields")
    tensor_names = ("means", "log_scales", "rotations", "filter_variance")
    scalar_names = (
        "width",
        "height",
        "fit_window",
        "sigma_cutoff",
        "support_fade_alpha",
        "aa_dilation",
        "view_id",
    )
    for view_index, (view_name, teacher, proposal) in enumerate(
        zip(view_names, teachers, proposals, strict=True)
    ):
        if not isinstance(proposal, GaussianObservationField):
            raise TypeError(f"proposal field {view_index} must be GaussianObservationField")
        if proposal.device != teacher.device or proposal.dtype != teacher.dtype:
            raise ValueError(f"proposal field {view_index} must share the teacher device and dtype")
        if proposal.n != teacher.n:
            raise ValueError(f"proposal field {view_index} component count must match its teacher")
        for name in scalar_names:
            if getattr(proposal, name) != getattr(teacher, name):
                raise ValueError(
                    f"proposal field {view_index} {name} differs from teacher {view_name!r}"
                )
        for name in tensor_names:
            proposal_value = getattr(proposal, name)
            teacher_value = getattr(teacher, name)
            if proposal_value is None or teacher_value is None:
                equal = proposal_value is None and teacher_value is None
            else:
                equal = torch.equal(proposal_value, teacher_value)
            if not equal:
                raise ValueError(
                    f"proposal field {view_index} {name} differs from teacher {view_name!r}"
                )


def _preflight_proposal_fields(
    fields: Sequence[GaussianObservationField],
    view_names: Sequence[str],
    config: CompactTrainConfig,
) -> dict:
    """Apply component and index budgets before constructing proposal indices."""
    if len(fields) != len(view_names):
        raise ValueError("proposal fields and view names must have the same length")
    views = []
    for index, (name, field) in enumerate(zip(view_names, fields, strict=True)):
        fitted_pixels = field.fit_window[2] * field.fit_window[3]
        if fitted_pixels > config.max_fitted_pixels_per_view:
            raise ValueError(
                f"proposal view {index} fitted pixels exceed compact trainer safety cap"
            )
        if field.n > config.max_components_per_view:
            raise ValueError(f"proposal view {index} components exceed compact trainer safety cap")
        entries = (
            GaussianObservationIndex.estimate_entries(field, tile_size=config.teacher_tile_size)
            if field.device.type == "cpu"
            else _estimate_device_index_entries(
                field,
                tile_size=config.teacher_tile_size,
                component_chunk=config.query_component_chunk,
            )
        )
        if entries > config.max_index_entries_per_view:
            raise ValueError(
                f"proposal view {index} index entries exceed compact trainer safety cap"
            )
        views.append(
            {
                "view_index": index,
                "view_name": name,
                "fitted_pixels": fitted_pixels,
                "components": field.n,
                "estimated_index_entries": entries,
            }
        )
    return {"n_views": len(fields), "views": views, "archive": None}


def _estimate_device_index_entries(
    field: GaussianObservationField,
    *,
    tile_size: int,
    component_chunk: int,
) -> int:
    """Exactly count tile overlaps in bounded chunks on the field's current device.

    Arithmetic is promoted to float64 to match the CPU reference's conversion of tensor
    scalars to Python floats.  Only one scalar subtotal per chunk crosses back to the host;
    teacher tensors are never copied wholesale for preflight.
    """
    GaussianObservationIndex._validate_limits(tile_size, None, None)
    if isinstance(component_chunk, bool) or not isinstance(component_chunk, int):
        raise ValueError("component_chunk must be a positive integer")
    if component_chunk <= 0:
        raise ValueError("component_chunk must be a positive integer")

    tiles_x = math.ceil(field.width / tile_size)
    tiles_y = math.ceil(field.height / tile_size)
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    fit_right = fit_x + fit_width
    fit_bottom = fit_y + fit_height
    total = 0
    with torch.no_grad():
        for start in range(0, field.n, component_chunk):
            end = min(start + component_chunk, field.n)
            centers = field.support_centers(slice(start, end)).to(torch.float64)
            radii = field.radii()[start:end].to(torch.float64)
            left = (centers[:, 0] - radii[:, 0]).clamp_min(fit_x)
            right = (centers[:, 0] + radii[:, 0]).clamp_max(fit_right)
            top = (centers[:, 1] - radii[:, 1]).clamp_min(fit_y)
            bottom = (centers[:, 1] + radii[:, 1]).clamp_max(fit_bottom)
            active = (field.amplitudes[start:end] > 0.0) & (left < fit_right)
            active &= (right >= fit_x) & (top < fit_bottom) & (bottom >= fit_y)
            tile_x0 = torch.floor(left / tile_size).clamp_min(0)
            tile_x1 = torch.floor(right / tile_size).clamp_max(tiles_x - 1)
            tile_y0 = torch.floor(top / tile_size).clamp_min(0)
            tile_y1 = torch.floor(bottom / tile_size).clamp_max(tiles_y - 1)
            overlaps = (tile_x1 - tile_x0 + 1) * (tile_y1 - tile_y0 + 1)
            total += int(overlaps.masked_fill(~active, 0).sum().item())
    return total


def _compact_working_inputs(
    inputs: ReconstructionInputs,
    device: torch.device,
) -> ReconstructionInputs:
    """Move only the teacher/camera working set needed by compact refinement.

    Sparse global points, visibility lists, and the bounds-center tensor are acquisition and
    initialization inputs.  Compact refinement does not query them.  The caller resolves the
    scalar extent from the original inputs so omitting them here does not change LR semantics.
    Existing teacher objects are retained when already on ``device`` so injected query-backend
    identity remains valid.
    """
    observations = [
        field if field.device == device else field.to(device) for field in inputs.observations
    ]
    cameras = [
        camera if camera.R.device == device and camera.t.device == device else camera.to(device)
        for camera in inputs.cameras
    ]
    return ReconstructionInputs(
        observations=observations,
        cameras=cameras,
        view_names=list(inputs.view_names),
        points=None,
        point_visibility=None,
        bounds_hint=None,
        name=inputs.name,
        archive_stats=inputs.archive_stats,
    )


class _QueryChunkAdapter:
    """Freeze the teacher component chunk even when proposal helpers use their default."""

    def __init__(self, backend: ObservationQueryBackend, component_chunk: int):
        self.backend = backend
        self.component_chunk = component_chunk

    def query(self, xy: torch.Tensor, component_chunk: int = 4096) -> ObservationQuery:
        del component_chunk
        return self.backend.query(xy, component_chunk=self.component_chunk)

    def query_weight_sum(self, xy: torch.Tensor, component_chunk: int = 4096) -> torch.Tensor:
        del component_chunk
        return self.backend.query_weight_sum(xy, component_chunk=self.component_chunk)


def _index_diagnostic(backend: ObservationQueryBackend, view_index: int) -> dict:
    if isinstance(backend, GaussianObservationIndex):
        stats = backend.stats
        return {
            "view_index": view_index,
            "kind": type(backend).__name__,
            "nonempty_tiles": stats.nonempty_tiles,
            "total_entries": stats.total_entries,
            "estimated_entries": backend.estimated_entries,
            "max_candidates_per_tile": stats.max_candidates,
        }
    return {
        "view_index": view_index,
        "kind": type(backend).__name__,
        "nonempty_tiles": None,
        "total_entries": None,
        "max_candidates_per_tile": None,
    }


def _prepare_backends(
    inputs: ReconstructionInputs,
    config: CompactTrainConfig,
    query_backends: Sequence[ObservationQueryBackend] | None,
    *,
    role: str = "teacher",
) -> tuple[list[ObservationQueryBackend], list[_QueryChunkAdapter], list[dict]]:
    if query_backends is not None:
        raw = list(query_backends)
        if len(raw) != inputs.n_views:
            raise ValueError(f"{role} query backends must contain exactly one backend per view")
        for view, (field, backend) in enumerate(zip(inputs.observations, raw, strict=True)):
            indexed_field = getattr(backend, "field", field)
            if indexed_field is not field:
                raise ValueError(f"{role} query backend {view} is bound to a different field")
    elif inputs.observations[0].device.type == "cpu":
        raw = [
            GaussianObservationIndex(
                field,
                config.teacher_tile_size,
                max_entries=config.max_index_entries_per_view,
                max_candidates=config.max_candidates_per_tile,
            )
            for field in inputs.observations
        ]
    else:
        raw = list(inputs.observations)
    diagnostics = [_index_diagnostic(backend, view) for view, backend in enumerate(raw)]
    for backend, item in zip(raw, diagnostics, strict=True):
        if isinstance(backend, GaussianObservationIndex):
            if backend.tile_size != config.teacher_tile_size:
                raise ValueError(
                    f"injected {role} index tile size differs from compact trainer config"
                )
            if backend.stats.total_entries > config.max_index_entries_per_view:
                raise ValueError(f"constructed {role} index exceeds entry safety cap")
        candidates = item["max_candidates_per_tile"]
        if candidates is not None and candidates > config.max_candidates_per_tile:
            raise ValueError(f"constructed {role} index exceeds per-tile candidate safety cap")
    adapted = [_QueryChunkAdapter(backend, config.query_component_chunk) for backend in raw]
    return raw, adapted, diagnostics


def _resolved_extent(
    inputs: ReconstructionInputs,
    init: Gaussians3D,
    explicit: float | None,
) -> tuple[float, str]:
    if explicit is not None:
        return float(explicit), "explicit"
    if inputs.bounds_hint is not None:
        return float(inputs.bounds_hint[1]), "bounds_hint"
    means = init.means.detach()
    if init.n >= 20:
        lower = torch.quantile(means, 0.01, dim=0)
        upper = torch.quantile(means, 0.99, dim=0)
    else:
        lower = means.amin(dim=0)
        upper = means.amax(dim=0)
    center = 0.5 * (lower + upper)
    radial = torch.linalg.vector_norm(means - center, dim=1)
    radius = torch.quantile(radial, 0.99) if init.n >= 20 else radial.amax()
    return max(2.2 * float(radius), 1e-3), "initial_mean_cloud"


def _slice_samples(samples: ObservationSamples, start: int, end: int) -> ObservationSamples:
    return ObservationSamples(
        xy=samples.xy[start:end],
        proposal_component_ids=samples.proposal_component_ids[start:end],
        proposal_density=samples.proposal_density[start:end],
        joint_density=samples.joint_density[start:end],
        target_density=samples.target_density[start:end],
        importance=samples.importance[start:end],
        inside_fit_window=samples.inside_fit_window[start:end],
        active=samples.active[start:end],
        risk_measure=samples.risk_measure,
    )


def _retarget_samples(samples: ObservationSamples, target_mode: TargetMode) -> ObservationSamples:
    """Retain proposal attempts while changing only their fixed-attempt target measure.

    The proposal-attempt target is the active marginal proposal submeasure itself. Active
    attempts therefore carry unit importance, while rejected/null attempts remain explicit
    zero-loss states. No active-count or estimated acceptance normalization is applied.
    """
    if target_mode == "uniform":
        return samples
    if target_mode != "proposal_attempt":
        raise ValueError(f"unknown compact target mode {target_mode!r}")
    active_weight = samples.active.to(samples.proposal_density.dtype)
    return ObservationSamples(
        xy=samples.xy,
        proposal_component_ids=samples.proposal_component_ids,
        proposal_density=samples.proposal_density,
        joint_density=samples.joint_density,
        target_density=samples.proposal_density * active_weight,
        importance=active_weight,
        inside_fit_window=samples.inside_fit_window,
        active=samples.active,
        risk_measure=samples.risk_measure,
    )


def _proposal_normalizer_record(
    view_index: int,
    view_name: str,
    proposal: GaussianPixelProposal | GaussianPointProposal,
) -> dict:
    if isinstance(proposal, GaussianPointProposal):
        kind = "continuous_analytic_mass"
        value = proposal.total_mass
    else:
        kind = "discrete_rectangle_envelope_mass"
        value = proposal.envelope_mass
    return {
        "view_index": view_index,
        "view_name": view_name,
        "components": proposal.field.n,
        "normalizer_kind": kind,
        "normalizer": float(value.detach()),
    }


def _peak_rss_bytes() -> int:
    # Linux reports KiB while macOS reports bytes. The repository's frozen runs use Linux.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _clone_optimizer_metadata(value: object) -> object:
    """Copy a param-group value without silently weakening exact comparisons."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_metadata(item) for item in value)
    if isinstance(value, list):
        return [_clone_optimizer_metadata(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_optimizer_metadata(item) for key, item in value.items()}
    return value


def _optimizer_metadata_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and torch.equal(left, right)
        )
    if isinstance(left, tuple):
        assert isinstance(right, tuple)
        return len(left) == len(right) and all(
            _optimizer_metadata_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, list):
        assert isinstance(right, list)
        return len(left) == len(right) and all(
            _optimizer_metadata_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        assert isinstance(right, dict)
        return tuple(left) == tuple(right) and all(
            _optimizer_metadata_equal(left[key], right[key]) for key in left
        )
    return bool(left == right)


def _tensor_bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare finite tensor values byte-for-byte, including signed zero."""
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return False
    return torch.equal(
        left.detach().contiguous().view(torch.uint8),
        right.detach().contiguous().view(torch.uint8),
    )


def _validated_controller_ids(
    controller: CompactTopologyController,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    ids = controller.persistent_ids
    if not isinstance(ids, torch.Tensor):
        raise RuntimeError("topology controller persistent IDs must be a tensor")
    if ids.ndim != 1 or ids.dtype != torch.long:
        raise RuntimeError("topology controller persistent IDs must be a LongTensor")
    if tuple(params) != _GROUP_ORDER:
        raise RuntimeError("topology controller changed parameter group order")
    if ids.device != params["means"].device:
        raise RuntimeError("topology controller persistent IDs changed device")
    if ids.numel() != params["means"].shape[0]:
        raise RuntimeError("topology controller persistent IDs have the wrong count")
    if ids.numel() != torch.unique(ids).numel():
        raise RuntimeError("topology controller persistent IDs are not unique")
    return ids.detach().clone()


def _adam_clock_value(value: object, *, expected_step: int) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise RuntimeError("Adam scalar clock is not a finite scalar tensor")
        scalar = float(value.item())
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("Adam scalar clock has an unsupported representation")
    else:
        scalar = float(value)
    if not math.isfinite(scalar) or scalar != expected_step:
        raise RuntimeError("Adam group clocks are not aligned")
    return int(scalar)


def _validate_adam_moment(
    moment: object,
    parameter: torch.Tensor,
    *,
    group_name: str,
    moment_name: str,
) -> torch.Tensor:
    if not isinstance(moment, torch.Tensor):
        raise RuntimeError(f"optimizer {moment_name} for {group_name} must be a tensor")
    if moment.shape != parameter.shape:
        raise RuntimeError(f"optimizer {moment_name} for {group_name} has the wrong shape")
    if moment.device != parameter.device:
        raise RuntimeError(f"optimizer {moment_name} for {group_name} changed device")
    if moment.dtype != parameter.dtype:
        raise RuntimeError(f"optimizer {moment_name} for {group_name} changed dtype")
    if not bool(torch.isfinite(moment).all()):
        raise RuntimeError(f"optimizer {moment_name} for {group_name} is non-finite")
    return moment


def _capture_topology_optimizer_boundary(
    params: dict[str, torch.Tensor],
    optimizers: dict[str, torch.optim.Optimizer],
    persistent_ids: torch.Tensor,
    *,
    step: int,
) -> dict:
    """Bind the exact pre-edit Adam state needed for a fail-closed row audit."""
    if tuple(optimizers) != _GROUP_ORDER:
        raise RuntimeError("topology controller changed optimizer group order")
    records = {}
    for name in _GROUP_ORDER:
        parameter = params[name]
        optimizer = optimizers[name]
        if not isinstance(optimizer, torch.optim.Adam):
            raise RuntimeError(f"optimizer group {name} is not Adam")
        if len(optimizer.param_groups) != 1:
            raise RuntimeError(f"optimizer group {name} changed param-group count")
        group = optimizer.param_groups[0]
        if len(group["params"]) != 1 or group["params"][0] is not parameter:
            raise RuntimeError("topology controller broke optimizer parameter binding")
        if group.get("name") != name:
            raise RuntimeError(f"optimizer group {name} changed its name")
        if len(optimizer.state) != 1 or parameter not in optimizer.state:
            raise RuntimeError(f"optimizer group {name} has unexpected parameter state")
        state = optimizer.state[parameter]
        if tuple(state) != ("step", "exp_avg", "exp_avg_sq"):
            raise RuntimeError(f"optimizer group {name} has unexpected Adam state fields")
        clock = state["step"]
        _adam_clock_value(clock, expected_step=step)
        moments = {}
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = _validate_adam_moment(
                state[moment_name],
                parameter,
                group_name=name,
                moment_name=moment_name,
            )
            moments[moment_name] = moment.detach().clone()
        records[name] = {
            "optimizer": optimizer,
            "parameter_shape": tuple(parameter.shape),
            "parameter_device": parameter.device,
            "parameter_dtype": parameter.dtype,
            "group_metadata": tuple(
                (key, _clone_optimizer_metadata(value))
                for key, value in group.items()
                if key != "params"
            ),
            "state_fields": tuple(state),
            "clock": _clone_optimizer_metadata(clock),
            "moments": moments,
        }
    return {
        "persistent_ids": persistent_ids,
        "records": records,
    }


def _validate_topology_optimizer_boundary(
    before: dict,
    params: dict[str, torch.Tensor],
    optimizers: dict[str, torch.optim.Optimizer],
    persistent_ids: torch.Tensor,
    *,
    step: int,
) -> dict:
    """Prove that topology surgery preserved every surviving Adam row exactly."""
    if tuple(params) != _GROUP_ORDER:
        raise RuntimeError("topology controller changed parameter group order")
    if tuple(optimizers) != _GROUP_ORDER:
        raise RuntimeError("topology controller changed optimizer group order")
    before_ids = before["persistent_ids"]
    before_id_to_row = {
        int(identity): row for row, identity in enumerate(before_ids.detach().cpu().tolist())
    }
    after_ids = persistent_ids.detach().cpu().tolist()
    survivor_after_rows = [
        row for row, identity in enumerate(after_ids) if int(identity) in before_id_to_row
    ]
    survivor_before_rows = [before_id_to_row[int(after_ids[row])] for row in survivor_after_rows]
    newborn_rows = [
        row for row, identity in enumerate(after_ids) if int(identity) not in before_id_to_row
    ]
    removed_count = len(before_id_to_row) - len(survivor_before_rows)

    for name in _GROUP_ORDER:
        record = before["records"][name]
        parameter = params[name]
        if optimizers[name] is not record["optimizer"]:
            raise RuntimeError(f"topology controller replaced optimizer group {name}")
        if (
            tuple(parameter.shape[1:]) != tuple(record["parameter_shape"][1:])
            or parameter.shape[0] != persistent_ids.numel()
        ):
            raise RuntimeError(f"topology controller returned a wrong shape for {name}")
        if parameter.device != record["parameter_device"]:
            raise RuntimeError(f"topology controller changed parameter device for {name}")
        if parameter.dtype != record["parameter_dtype"]:
            raise RuntimeError(f"topology controller changed parameter dtype for {name}")
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f"optimizer group {name} became non-finite")

        optimizer = optimizers[name]
        if len(optimizer.param_groups) != 1:
            raise RuntimeError(f"optimizer group {name} changed param-group count")
        group = optimizer.param_groups[0]
        if len(group["params"]) != 1 or group["params"][0] is not parameter:
            raise RuntimeError("topology controller broke optimizer parameter binding")
        metadata = tuple((key, value) for key, value in group.items() if key != "params")
        frozen_metadata = record["group_metadata"]
        if tuple(key for key, _ in metadata) != tuple(key for key, _ in frozen_metadata) or any(
            not _optimizer_metadata_equal(current, frozen)
            for (_, current), (_, frozen) in zip(metadata, frozen_metadata, strict=True)
        ):
            raise RuntimeError(f"topology controller changed optimizer metadata for {name}")
        if len(optimizer.state) != 1 or parameter not in optimizer.state:
            raise RuntimeError(f"optimizer group {name} has unexpected parameter state")
        state = optimizer.state[parameter]
        if tuple(state) != record["state_fields"]:
            raise RuntimeError(f"optimizer group {name} changed Adam state fields")
        _adam_clock_value(state["step"], expected_step=step)
        if not _optimizer_metadata_equal(state["step"], record["clock"]):
            raise RuntimeError(f"topology controller changed Adam scalar clock for {name}")

        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = _validate_adam_moment(
                state[moment_name],
                parameter,
                group_name=name,
                moment_name=moment_name,
            )
            frozen = record["moments"][moment_name]
            if survivor_after_rows:
                after_index = torch.tensor(
                    survivor_after_rows,
                    dtype=torch.long,
                    device=moment.device,
                )
                before_index = torch.tensor(
                    survivor_before_rows,
                    dtype=torch.long,
                    device=frozen.device,
                )
                if not _tensor_bits_equal(moment[after_index], frozen[before_index]):
                    raise RuntimeError(
                        f"topology controller changed surviving {moment_name} moments for {name}"
                    )
            if newborn_rows:
                newborn_index = torch.tensor(
                    newborn_rows,
                    dtype=torch.long,
                    device=moment.device,
                )
                newborn = moment[newborn_index]
                if not _tensor_bits_equal(newborn, torch.zeros_like(newborn)):
                    raise RuntimeError(
                        f"topology controller created non-exact-zero newborn {moment_name} moments "
                        f"for {name}"
                    )

    return {
        "schema": "rtgs.compact_topology_optimizer_boundary.v1",
        "step": step,
        "rows_before": int(before_ids.numel()),
        "rows_after": int(persistent_ids.numel()),
        "survivor_rows": len(survivor_after_rows),
        "removed_rows": removed_count,
        "newborn_rows": len(newborn_rows),
        "optimizer_group_order": list(_GROUP_ORDER),
        "group_metadata_preserved": True,
        "parameter_layout_finite": True,
        "moment_layout_finite": True,
        "survivor_moments_bit_preserved": True,
        "newborn_moments_exact_zero": True,
        "scalar_clocks_unchanged": True,
    }


class CompactTrainer:
    """Optimize a fixed 3D Gaussian set against RGB-free observation fields."""

    def __init__(
        self,
        config: CompactTrainConfig | None = None,
        *,
        point_rasterizer: PointRasterizer | None = None,
    ) -> None:
        self.config = config or CompactTrainConfig()
        self.point_rasterizer = point_rasterizer

    def _renderer(self) -> PointRasterizer:
        cfg = self.config
        return self.point_rasterizer or TorchPointRasterizer(
            point_chunk=cfg.point_chunk,
            gaussian_chunk=cfg.gaussian_chunk,
            sh_color_activation=cfg.sh_color_activation,
            sh_smu1_mu=cfg.sh_smu1_mu,
            kernel_support_mode=cfg.kernel_support_mode,
            visibility_margin_sigma=cfg.visibility_margin_sigma,
        )

    @staticmethod
    def _build(params: dict[str, torch.nn.Parameter]) -> Gaussians3D:
        return Gaussians3D(
            means=params["means"],
            quats=params["quats"],
            log_scales=params["scales"],
            opacity=torch.sigmoid(params["opacities"]),
            sh=torch.cat([params["sh0"], params["shN"]], dim=1),
        )

    def _proposal(
        self,
        field: GaussianObservationField,
        backend: ObservationQueryBackend,
    ) -> GaussianPixelProposal | GaussianPointProposal:
        if self.config.proposal_mode.startswith("pixel_"):
            return GaussianPixelProposal(field, backend)
        return GaussianPointProposal(field, backend)

    def _effective_uniform_fraction(self) -> float:
        if self.config.proposal_mode.endswith("_uniform"):
            return 1.0
        return self.config.uniform_fraction

    def _validate_samples(self, samples: ObservationSamples) -> None:
        cfg = self.config
        if samples.xy.shape[0] != cfg.attempts_per_step:
            raise RuntimeError("proposal changed the fixed attempt count")
        expected_risk = (
            "discrete_pixels" if cfg.proposal_mode.startswith("pixel_") else "continuous_area"
        )
        if samples.risk_measure != expected_risk:
            raise RuntimeError("proposal returned the wrong risk measure")
        tensors = (
            samples.xy,
            samples.proposal_density,
            samples.joint_density,
            samples.target_density,
            samples.importance,
        )
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("proposal returned non-finite values")
        if bool((samples.active & ~samples.inside_fit_window).any()):
            raise RuntimeError("active proposal attempt lies outside the fitted window")
        invalid = ~samples.inside_fit_window
        for name in ("joint_density", "target_density", "proposal_density", "importance"):
            if invalid.any() and bool((getattr(samples, name)[invalid] != 0.0).any()):
                raise RuntimeError(f"invalid proposal rows have nonzero {name}")
        inactive = ~samples.active
        for name in ("joint_density", "target_density", "importance"):
            if inactive.any() and bool((getattr(samples, name)[inactive] != 0.0).any()):
                raise RuntimeError(f"inactive proposal rows have nonzero {name}")
        if cfg.target_mode == "proposal_attempt":
            if samples.active.any() and not bool(
                torch.equal(
                    samples.importance[samples.active],
                    torch.ones_like(samples.importance[samples.active]),
                )
            ):
                raise RuntimeError("proposal-attempt target has non-unit active importance")
            if samples.active.any() and not bool(
                torch.equal(
                    samples.target_density[samples.active],
                    samples.proposal_density[samples.active],
                )
            ):
                raise RuntimeError("proposal-attempt target differs from active proposal density")
            limit = 1.0 + 1e-5
        else:
            limit = 1.0 / self._effective_uniform_fraction() + 1e-5
        if float(samples.importance.max()) > limit:
            raise RuntimeError("proposal importance exceeds its target-specific bound")

    def train(
        self,
        inputs: ReconstructionInputs,
        init: Gaussians3D,
        *,
        query_backends: Sequence[ObservationQueryBackend] | None = None,
        proposal_fields: Sequence[GaussianObservationField] | None = None,
        proposal_query_backends: Sequence[ObservationQueryBackend] | None = None,
        bundle_path: str | Path | None = None,
        checkpoint_callback: Callable[[Gaussians3D, int], None] | None = None,
        topology_controller: CompactTopologyController | None = None,
        stop_after_step: int | None = None,
    ) -> tuple[Gaussians3D, dict]:
        """Run compact optimization and return ``(refined, history)``.

        ``topology_controller`` and ``stop_after_step`` are opt-in research seams. Their
        ``None`` defaults preserve the established fixed-topology path.
        """
        cfg = self.config
        device = torch.device(cfg.device)
        if stop_after_step is not None and (
            isinstance(stop_after_step, bool)
            or not isinstance(stop_after_step, int)
            or not 1 <= stop_after_step <= cfg.iterations
        ):
            raise ValueError("stop_after_step must be an integer in [1, iterations]")
        if init.n <= 0:
            raise ValueError("compact training requires at least one 3D Gaussian")
        for name in ("means", "quats", "log_scales", "opacity", "sh"):
            if not bool(torch.isfinite(getattr(init, name)).all()):
                raise ValueError(f"initial {name} contains non-finite values")

        if query_backends is not None and any(
            field.device != device for field in inputs.observations
        ):
            raise ValueError(
                "injected query backends require inputs already on the configured device"
            )
        explicit_proposal_surface = (
            proposal_fields is not None or proposal_query_backends is not None
        )
        source_proposal_fields = (
            list(inputs.observations) if proposal_fields is None else list(proposal_fields)
        )
        _validate_proposal_fields(
            inputs.observations,
            source_proposal_fields,
            inputs.view_names,
        )
        if proposal_query_backends is not None and any(
            field.device != device for field in source_proposal_fields
        ):
            raise ValueError(
                "injected proposal query backends require proposal fields already on the "
                "configured device"
            )
        preflight = preflight_observations(inputs, cfg, bundle_path=bundle_path)
        proposal_preflight = (
            _preflight_proposal_fields(source_proposal_fields, inputs.view_names, cfg)
            if explicit_proposal_surface
            else preflight
        )
        working_inputs = _compact_working_inputs(inputs, device)
        _, adapted_backends, index_diagnostics = _prepare_backends(
            working_inputs, cfg, query_backends
        )
        teacher_before = observation_digest(working_inputs)

        if explicit_proposal_surface:
            working_proposal_fields = [
                field if field.device == device else field.to(device)
                for field in source_proposal_fields
            ]
            _validate_proposal_fields(
                working_inputs.observations,
                working_proposal_fields,
                working_inputs.view_names,
            )
            proposal_inputs = ReconstructionInputs(
                observations=working_proposal_fields,
                cameras=working_inputs.cameras,
                view_names=list(working_inputs.view_names),
                points=None,
                point_visibility=None,
                bounds_hint=None,
                name=f"{working_inputs.name}-proposal",
                archive_stats=None,
            )
            _, adapted_proposal_backends, proposal_index_diagnostics = _prepare_backends(
                proposal_inputs,
                cfg,
                proposal_query_backends,
                role="proposal",
            )
            proposal_before = observation_digest(proposal_inputs)
        else:
            working_proposal_fields = list(working_inputs.observations)
            proposal_inputs = working_inputs
            adapted_proposal_backends = adapted_backends
            proposal_index_diagnostics = index_diagnostics
            proposal_before = teacher_before

        init = init.with_sh_degree(cfg.sh_degree).to(device)
        extent, extent_source = _resolved_extent(inputs, init, cfg.extent)
        params: dict[str, torch.nn.Parameter] = {
            "means": torch.nn.Parameter(init.means.detach().clone()),
            "quats": torch.nn.Parameter(init.quats.detach().clone()),
            "scales": torch.nn.Parameter(init.log_scales.detach().clone()),
            "opacities": torch.nn.Parameter(
                torch.logit(init.opacity.detach().clamp(1e-4, 1.0 - 1e-4)).clone()
            ),
            "sh0": torch.nn.Parameter(init.sh[:, :1].detach().clone()),
            "shN": torch.nn.Parameter(init.sh[:, 1:].detach().clone()),
        }
        initial_params = {name: value.detach().clone() for name, value in params.items()}
        lrs = {
            "means": cfg.lr_means * extent,
            "quats": cfg.lr_quats,
            "scales": cfg.lr_scales,
            "opacities": cfg.lr_opacity,
            "sh0": cfg.lr_sh,
            "shN": cfg.lr_sh_rest,
        }
        optimizers = {
            name: torch.optim.Adam(
                [{"params": [params[name]], "lr": lrs[name], "name": name}],
                betas=(0.9, 0.999),
                eps=1e-15,
                weight_decay=0.0,
                amsgrad=False,
                maximize=False,
                foreach=False,
                fused=False,
            )
            for name in _GROUP_ORDER
        }
        if topology_controller is not None:
            topology_controller.bind(
                params,
                optimizers,
                extent=extent,
                n_views=working_inputs.n_views,
                attempts_per_step=cfg.attempts_per_step,
            )
            initial_ids = _validated_controller_ids(topology_controller, params)
            expected_ids = torch.arange(init.n, device=device, dtype=torch.long)
            if not torch.equal(initial_ids, expected_ids):
                raise RuntimeError(
                    "topology controller initial persistent IDs must match input row order"
                )
        means_gamma = 0.01 ** (1.0 / cfg.iterations)
        renderer = self._renderer()
        proposals = [
            self._proposal(field, backend)
            for field, backend in zip(
                working_proposal_fields,
                adapted_proposal_backends,
                strict=True,
            )
        ]
        view_schedule = build_view_schedule(
            working_inputs.n_views,
            cfg.iterations,
            cfg.seed,
            mode=cfg.schedule_mode,
        )
        schedule_sha = hashlib.sha256(
            json.dumps(list(view_schedule), separators=(",", ":")).encode()
        ).hexdigest()
        risk_measure = proposals[0].risk_measure
        proposal_normalizers = [
            _proposal_normalizer_record(view_index, view_name, proposal)
            for view_index, (view_name, proposal) in enumerate(
                zip(working_inputs.view_names, proposals, strict=True)
            )
        ]
        history: dict = {
            "schema": "rtgs.compact_train_history.v2",
            "proposal_mode": cfg.proposal_mode,
            "schedule_mode": cfg.schedule_mode,
            "target_mode": cfg.target_mode,
            "risk_measure": risk_measure,
            "seed": cfg.seed,
            "iterations": cfg.iterations,
            "attempts_per_step": cfg.attempts_per_step,
            "uniform_fraction": self._effective_uniform_fraction(),
            "extent": extent,
            "extent_source": extent_source,
            "n_init_3d": init.n,
            "n_opt_3d": init.n,
            "view_schedule": list(view_schedule),
            "view_schedule_sha256": schedule_sha,
            "view_visit_counts": [
                sum(view_index == scheduled for scheduled in view_schedule)
                for view_index in range(working_inputs.n_views)
            ],
            "teacher_digest_before": teacher_before,
            "teacher_digest_after": None,
            "proposal_field_source": ("explicit" if explicit_proposal_surface else "teacher"),
            "proposal_digest_before": proposal_before,
            "proposal_digest_after": None,
            "preflight": preflight,
            "index_diagnostics": index_diagnostics,
            "proposal_preflight": proposal_preflight,
            "proposal_index_diagnostics": proposal_index_diagnostics,
            "proposal_normalizers": proposal_normalizers,
            "proposal_view_diagnostics": [],
            "steps": [],
            "checkpoint_risk_evaluation_enabled": cfg.evaluate_checkpoint_risks,
            "checkpoint_risk_evaluation_call_count": 0,
            "checkpoint_callback_call_count": 0,
            "checkpoint_snapshot_count": 0,
            "proposal_branch_counts": {
                "uniform": 0,
                "gaussian": 0,
                "gaussian_accepted": 0,
                "gaussian_rejected": 0,
            },
            "proposal_invariants": {
                "active_implies_inside_fit_window": True,
                "invalid_zero_joint_density": True,
                "invalid_zero_target_density": True,
                "invalid_zero_proposal_density": True,
                "invalid_zero_importance": True,
                "no_null_resampling": True,
                "inactive_zero_joint_target_importance": True,
                "importance_within_uniform_floor": True,
            },
            "checkpoints": [],
            "optimizer_steps": {},
            "parameter_motion": {},
            "optimizer_group_motion": {},
            "elapsed_seconds": None,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        if stop_after_step is not None:
            history["planned_view_visit_counts"] = list(history["view_visit_counts"])
            prefix = view_schedule[:stop_after_step]
            history["view_visit_counts"] = [
                sum(view_index == scheduled for scheduled in prefix)
                for view_index in range(working_inputs.n_views)
            ]
            history["stop_after_step"] = stop_after_step
        if topology_controller is not None:
            history["topology_control_enabled"] = True

        checkpoint_set = set(cfg.checkpoints)

        def record_checkpoint(step: int) -> None:
            snapshot = self._build(params).detach()
            history["checkpoint_snapshot_count"] += 1
            if cfg.evaluate_checkpoint_risks:
                metrics = self._evaluate_prepared(
                    working_inputs,
                    snapshot,
                    adapted_backends,
                    renderer,
                )
                history["checkpoint_risk_evaluation_call_count"] += 1
            else:
                metrics = None
            history["checkpoints"].append(
                {
                    "step": step,
                    "snapshot_sha256": _gaussians_sha256(snapshot),
                    "evaluation": metrics,
                }
            )
            if checkpoint_callback is not None:
                with torch.no_grad():
                    checkpoint_callback(snapshot.detach(), step)
                history["checkpoint_callback_call_count"] += 1

        started = time.perf_counter()
        if 0 in checkpoint_set:
            record_checkpoint(0)
        completed_iterations = 0
        for step_index, view_index in enumerate(view_schedule):
            step = step_index + 1
            step_started = time.perf_counter()
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device=device).manual_seed(
                step_sample_seed(cfg.seed, step_index)
            )
            samples = proposals[view_index].sample(
                cfg.attempts_per_step,
                uniform_fraction=self._effective_uniform_fraction(),
                generator=generator,
            )
            samples = _retarget_samples(samples, cfg.target_mode)
            self._validate_samples(samples)
            uniform_attempts = samples.proposal_component_ids == -1
            gaussian_attempts = ~uniform_attempts
            if bool((uniform_attempts & ~samples.active).any()):
                raise RuntimeError("uniform proposal branch produced an inactive attempt")
            branch_counts = history["proposal_branch_counts"]
            branch_counts["uniform"] += int(uniform_attempts.sum())
            branch_counts["gaussian"] += int(gaussian_attempts.sum())
            branch_counts["gaussian_accepted"] += int((gaussian_attempts & samples.active).sum())
            branch_counts["gaussian_rejected"] += int((gaussian_attempts & ~samples.active).sum())

            detached_loss = 0.0
            visible_count: int | None = None
            visible_indices: torch.Tensor | None = None
            rendered_pairs = 0
            teacher_calls = 0
            student_calls = 0
            for start in range(0, cfg.attempts_per_step, cfg.outer_microbatch):
                end = min(start + cfg.outer_microbatch, cfg.attempts_per_step)
                chunk_samples = _slice_samples(samples, start, end)
                with torch.no_grad():
                    target = adapted_backends[view_index].query(chunk_samples.xy).color
                teacher_calls += 1
                if not bool(torch.isfinite(target).all()):
                    raise RuntimeError("compact teacher query contains non-finite color")
                render_kwargs = {
                    "background": torch.zeros(3, device=device, dtype=init.means.dtype),
                    "sh_degree": cfg.sh_degree,
                }
                if topology_controller is not None:
                    render_kwargs["collect_compositing_color_basis"] = (
                        topology_controller.needs_compositing_color_basis(step)
                    )
                output = renderer.render_points(
                    self._build(params),
                    working_inputs.cameras[view_index],
                    chunk_samples.xy,
                    **render_kwargs,
                )
                student_calls += 1
                if not bool(torch.isfinite(output.color).all()):
                    raise RuntimeError("compact point render contains non-finite color")
                current_visible = 0 if output.visible is None else int(output.visible.numel())
                if visible_count is None:
                    visible_count = current_visible
                elif visible_count != current_visible:
                    raise RuntimeError("global visible set changed across one point microbatch")
                if topology_controller is not None:
                    current_indices = (
                        torch.empty(0, dtype=torch.long, device=device)
                        if output.visible is None
                        else output.visible.detach()
                    )
                    if visible_indices is None:
                        visible_indices = current_indices.clone()
                    elif not torch.equal(visible_indices, current_indices):
                        raise RuntimeError(
                            "global visible ordering changed across one point microbatch"
                        )
                rendered_pairs += (end - start) * current_visible
                point_loss = (output.color - target).square().mean(dim=-1)
                if not bool(torch.isfinite(point_loss).all()):
                    raise RuntimeError("compact point loss contains non-finite values")
                if topology_controller is not None:
                    topology_controller.observe_pre_backward(
                        step=step,
                        view_index=view_index,
                        output=output,
                        point_loss=point_loss.detach(),
                        active=chunk_samples.active.detach(),
                        attempts=cfg.attempts_per_step,
                    )
                    if output.compositing_color_basis is not None:
                        raise RuntimeError(
                            "topology controller did not clear compositing color basis"
                        )
                chunk_mean = fixed_attempt_mean(point_loss, chunk_samples)
                scaled_loss = chunk_mean * ((end - start) / cfg.attempts_per_step)
                detached_loss += float(scaled_loss.detach())
                # A fresh anchor is required for each immediate microbatch backward: an
                # all-invisible render is otherwise a constant with no autograd graph.
                scaled_loss = scaled_loss + sum(
                    parameter.sum() * 0.0 for parameter in params.values()
                )
                scaled_loss.backward()
                if topology_controller is not None:
                    camera = working_inputs.cameras[view_index]
                    topology_controller.observe_post_backward(
                        step=step,
                        view_index=view_index,
                        output=output,
                        width=camera.width,
                        height=camera.height,
                    )

            gradient_max = {}
            for name, parameter in params.items():
                if parameter.grad is None:
                    raise RuntimeError(f"optimizer group {name} has no aligned gradient")
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"optimizer group {name} has non-finite gradients")
                gradient_max[_FAMILY_NAMES[name]] = (
                    float(parameter.grad.detach().abs().max()) if parameter.numel() else 0.0
                )
            group_lrs_used = {
                name: float(optimizers[name].param_groups[0]["lr"]) for name in _GROUP_ORDER
            }
            for name in _GROUP_ORDER:
                optimizers[name].step()
            optimizers["means"].param_groups[0]["lr"] *= means_gamma

            for name, parameter in params.items():
                if topology_controller is None and parameter.shape[0] != init.n:
                    raise RuntimeError("compact trainer changed 3D Gaussian cardinality")
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError(f"optimizer group {name} became non-finite")
                state_step = optimizers[name].state[parameter].get("step")
                clock = (
                    int(state_step.item())
                    if isinstance(state_step, torch.Tensor)
                    else int(state_step)
                )
                if clock != step:
                    raise RuntimeError("Adam group clocks are not aligned")

            weights = samples.importance.detach().double()
            weight_sum = float(weights.sum())
            weight_square_sum = float(weights.square().sum())
            ess = 0.0 if weight_square_sum == 0.0 else weight_sum**2 / weight_square_sum
            history["steps"].append(
                {
                    "step": step,
                    "view_index": view_index,
                    "view_name": working_inputs.view_names[view_index],
                    "sample_seed": step_sample_seed(cfg.seed, step_index),
                    "xy_sha256": _tensor_sha256(samples.xy),
                    "active_sha256": _tensor_sha256(samples.active),
                    "inside_fit_window_sha256": _tensor_sha256(samples.inside_fit_window),
                    "proposal_density_sha256": _tensor_sha256(samples.proposal_density),
                    "joint_density_sha256": _tensor_sha256(samples.joint_density),
                    "target_density_sha256": _tensor_sha256(samples.target_density),
                    "importance_sha256": _tensor_sha256(samples.importance),
                    "proposal_component_ids_sha256": _tensor_sha256(samples.proposal_component_ids),
                    "attempts": cfg.attempts_per_step,
                    "active_count": int(samples.active.sum()),
                    "null_count": int((~samples.active).sum()),
                    "invalid_count": int((~samples.inside_fit_window).sum()),
                    "uniform_attempt_count": int(uniform_attempts.sum()),
                    "gaussian_attempt_count": int(gaussian_attempts.sum()),
                    "gaussian_accepted_count": int((gaussian_attempts & samples.active).sum()),
                    "gaussian_rejected_count": int((gaussian_attempts & ~samples.active).sum()),
                    "visible_count": 0 if visible_count is None else visible_count,
                    "sampled_loss": detached_loss,
                    "importance_max": float(samples.importance.max()),
                    "importance_ess": ess,
                    "importance_ess_per_attempt": ess / cfg.attempts_per_step,
                    "rendered_point_gaussian_pairs": rendered_pairs,
                    "teacher_query_attempts": cfg.attempts_per_step,
                    "student_query_attempts": cfg.attempts_per_step,
                    "teacher_query_calls": teacher_calls,
                    "student_query_calls": student_calls,
                    "group_lrs_used": group_lrs_used,
                    "gradient_max": gradient_max,
                    "cardinality": int(params["means"].shape[0]),
                    "elapsed_seconds": time.perf_counter() - step_started,
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )
            if topology_controller is not None:
                pre_topology_n = int(params["means"].shape[0])
                pre_topology_ids = _validated_controller_ids(
                    topology_controller,
                    params,
                )
                optimizer_boundary = _capture_topology_optimizer_boundary(
                    params,
                    optimizers,
                    pre_topology_ids,
                    step=step,
                )
                params = topology_controller.after_step(
                    step=step,
                    params=params,
                    optimizers=optimizers,
                    snapshot=self._build(params).detach(),
                )
                ids = _validated_controller_ids(topology_controller, params)
                current_n = int(ids.numel())
                boundary_receipt = _validate_topology_optimizer_boundary(
                    optimizer_boundary,
                    params,
                    optimizers,
                    ids,
                    step=step,
                )
                history["steps"][-1]["cardinality_after_topology"] = current_n
                history["steps"][-1]["topology_changed"] = current_n != pre_topology_n
                history["steps"][-1]["topology_optimizer_boundary"] = boundary_receipt
            if step in checkpoint_set:
                record_checkpoint(step)
            completed_iterations = step
            if stop_after_step is not None and step == stop_after_step:
                break

        teacher_after = observation_digest(working_inputs)
        history["teacher_digest_after"] = teacher_after
        if teacher_after != teacher_before:
            raise RuntimeError("compact teacher fields changed during optimization")
        proposal_after = observation_digest(proposal_inputs)
        history["proposal_digest_after"] = proposal_after
        if proposal_after != proposal_before:
            raise RuntimeError("compact proposal fields changed during optimization")
        for view_index, view_name in enumerate(working_inputs.view_names):
            view_steps = [
                record for record in history["steps"] if record["view_index"] == view_index
            ]
            attempts = sum(record["attempts"] for record in view_steps)
            active = sum(record["active_count"] for record in view_steps)
            null = sum(record["null_count"] for record in view_steps)
            gaussian_attempts = sum(record["gaussian_attempt_count"] for record in view_steps)
            gaussian_accepted = sum(record["gaussian_accepted_count"] for record in view_steps)
            history["proposal_view_diagnostics"].append(
                {
                    "view_index": view_index,
                    "view_name": view_name,
                    "steps": len(view_steps),
                    "attempts": attempts,
                    "active_count": active,
                    "null_count": null,
                    "active_fraction": None if attempts == 0 else active / attempts,
                    "null_fraction": None if attempts == 0 else null / attempts,
                    "gaussian_attempt_count": gaussian_attempts,
                    "gaussian_accepted_count": gaussian_accepted,
                    "gaussian_acceptance_fraction": (
                        None if gaussian_attempts == 0 else gaussian_accepted / gaussian_attempts
                    ),
                }
            )
        final = self._build(params).detach()
        if topology_controller is None and final.n != init.n:
            raise RuntimeError("compact trainer changed 3D Gaussian cardinality")
        if topology_controller is not None and final.n != int(
            topology_controller.persistent_ids.numel()
        ):
            raise RuntimeError("final topology count differs from persistent identities")
        history["n_opt_3d"] = final.n
        current_ids = (
            torch.arange(init.n, device=device, dtype=torch.long)
            if topology_controller is None
            else topology_controller.persistent_ids
        )
        surviving_original = current_ids < init.n
        current_original_rows = surviving_original.nonzero(as_tuple=True)[0]
        original_ids = current_ids[surviving_original]
        for name, parameter in params.items():
            current = parameter.detach()[current_original_rows]
            initial = initial_params[name][original_ids]
            delta = current - initial
            motion = {
                "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
                "l2": float(torch.linalg.vector_norm(delta)) if delta.numel() else 0.0,
            }
            history["optimizer_group_motion"][name] = motion
            if delta.numel():
                history["parameter_motion"][_FAMILY_NAMES[name]] = motion
            state_step = optimizers[name].state[parameter]["step"]
            history["optimizer_steps"][name] = int(state_step.item())
        if topology_controller is not None:
            original_set = set(int(value) for value in original_ids.detach().cpu().tolist())
            history["surviving_original_ids"] = sorted(original_set)
            history["removed_original_ids"] = sorted(set(range(init.n)) - original_set)
            newborn_rows = (~surviving_original).nonzero(as_tuple=True)[0]
            history["newborn_parameter_summary"] = {}
            for name, parameter in params.items():
                newborn = parameter.detach()[newborn_rows]
                history["newborn_parameter_summary"][name] = {
                    "rows": int(newborn.shape[0]),
                    "max_abs": float(newborn.abs().max()) if newborn.numel() else 0.0,
                    "l2": float(torch.linalg.vector_norm(newborn)) if newborn.numel() else 0.0,
                }
            history["persistent_ids"] = current_ids.detach().cpu().tolist()
            controller_record = topology_controller.history_record()
            history["topology_control"] = controller_record
            current_id_to_row = {
                int(identity): row
                for row, identity in enumerate(current_ids.detach().cpu().tolist())
            }
            family_ids = {
                "clone": [],
                "split_child_0": [],
                "split_child_1": [],
            }
            lineage = (
                controller_record.get("lineage", []) if isinstance(controller_record, dict) else []
            )
            for item in lineage:
                if not isinstance(item, dict) or "birth_id" not in item:
                    raise RuntimeError("topology controller returned invalid lineage evidence")
                birth_id = int(item["birth_id"])
                if birth_id < init.n:
                    raise RuntimeError(
                        "topology controller assigned newborn lineage to an original"
                    )
                operator = item.get("operator")
                child_ordinal = item.get("child_ordinal")
                if operator == "clone" and child_ordinal == 0:
                    family = "clone"
                elif operator == "split" and child_ordinal == 0:
                    family = "split_child_0"
                elif operator == "split" and child_ordinal == 1:
                    family = "split_child_1"
                else:
                    raise RuntimeError("topology controller returned invalid newborn lineage")
                family_ids[family].append(birth_id)
            flattened_family_ids = [
                identity for values in family_ids.values() for identity in values
            ]
            current_newborn_ids = [
                int(identity)
                for identity in current_ids[~surviving_original].detach().cpu().tolist()
            ]
            if sorted(flattened_family_ids) != sorted(current_newborn_ids):
                raise RuntimeError(
                    "topology controller lineage does not cover current newborn identities"
                )
            history["newborn_parameter_summary_by_lineage"] = {}
            for family, identities in family_ids.items():
                rows = torch.tensor(
                    [current_id_to_row[identity] for identity in identities],
                    dtype=torch.long,
                    device=device,
                )
                family_summary = {}
                for name, parameter in params.items():
                    values = parameter.detach()[rows]
                    family_summary[name] = {
                        "rows": int(values.shape[0]),
                        "max_abs": float(values.abs().max()) if values.numel() else 0.0,
                        "l2": (float(torch.linalg.vector_norm(values)) if values.numel() else 0.0),
                    }
                history["newborn_parameter_summary_by_lineage"][family] = family_summary
        if stop_after_step is not None or topology_controller is not None:
            history["completed_iterations"] = completed_iterations
        history["elapsed_seconds"] = time.perf_counter() - started
        history["peak_rss_bytes"] = _peak_rss_bytes()
        return final, history

    def evaluate(
        self,
        inputs: ReconstructionInputs,
        gaussians: Gaussians3D,
        *,
        query_backends: Sequence[ObservationQueryBackend] | None = None,
        bundle_path: str | Path | None = None,
    ) -> dict:
        """Stream exact pixel risk and frozen four-offset area quadrature."""
        cfg = self.config
        device = torch.device(cfg.device)
        if query_backends is not None and any(
            field.device != device for field in inputs.observations
        ):
            raise ValueError("injected query backends require inputs on the configured device")
        preflight_observations(inputs, cfg, bundle_path=bundle_path)
        working_inputs = _compact_working_inputs(inputs, device)
        _, adapted, _ = _prepare_backends(working_inputs, cfg, query_backends)
        return self._evaluate_prepared(
            working_inputs,
            gaussians.to(device),
            adapted,
            self._renderer(),
        )

    def _evaluate_prepared(
        self,
        inputs: ReconstructionInputs,
        gaussians: Gaussians3D,
        query_backends: Sequence[ObservationQueryBackend],
        renderer: PointRasterizer,
    ) -> dict:
        pixel = self._evaluate_measure(
            inputs,
            gaussians,
            query_backends,
            renderer,
            offsets=((0.5, 0.5),),
            name="pixel",
        )
        area = self._evaluate_measure(
            inputs,
            gaussians,
            query_backends,
            renderer,
            offsets=_AREA_OFFSETS,
            name="area",
        )
        return {
            "schema": "rtgs.compact_point_evaluation.v1",
            "J_pixel": pixel["equal_view_mse"],
            "J_area": area["equal_view_mse"],
            "pixel": pixel,
            "area": area,
        }

    def _evaluate_measure(
        self,
        inputs: ReconstructionInputs,
        gaussians: Gaussians3D,
        query_backends: Sequence[ObservationQueryBackend],
        renderer: PointRasterizer,
        *,
        offsets: tuple[tuple[float, float], ...],
        name: str,
    ) -> dict:
        cfg = self.config
        per_view = []
        view_means = []
        totals = {
            "teacher_below_zero": 0,
            "teacher_above_one": 0,
            "prediction_below_zero": 0,
            "prediction_above_one": 0,
            "scalar_count": 0,
        }
        background = torch.zeros(3, device=gaussians.means.device, dtype=gaussians.means.dtype)
        with torch.no_grad():
            for view_index, (field, camera, backend) in enumerate(
                zip(inputs.observations, inputs.cameras, query_backends, strict=True)
            ):
                started = time.perf_counter()
                fit_x, fit_y, fit_width, fit_height = field.fit_window
                pixel_count = fit_width * fit_height
                sse = 0.0
                scalar_count = 0
                counts = {key: 0 for key in totals if key != "scalar_count"}
                for offset_x, offset_y in offsets:
                    for start in range(0, pixel_count, cfg.evaluation_chunk):
                        end = min(start + cfg.evaluation_chunk, pixel_count)
                        linear = torch.arange(
                            start,
                            end,
                            device=gaussians.means.device,
                            dtype=torch.long,
                        )
                        pixel_x = torch.remainder(linear, fit_width) + fit_x
                        pixel_y = torch.div(linear, fit_width, rounding_mode="floor") + fit_y
                        xy = torch.stack(
                            [
                                pixel_x.to(field.dtype) + offset_x,
                                pixel_y.to(field.dtype) + offset_y,
                            ],
                            dim=-1,
                        )
                        teacher = backend.query(xy).color
                        prediction = renderer.render_points(
                            gaussians,
                            camera,
                            xy,
                            background=background,
                            sh_degree=cfg.sh_degree,
                        ).color
                        if not bool(torch.isfinite(teacher).all()) or not bool(
                            torch.isfinite(prediction).all()
                        ):
                            raise RuntimeError("compact evaluation produced non-finite values")
                        error = prediction.double() - teacher.double()
                        sse += float(error.square().sum(dtype=torch.float64))
                        scalar_count += int(error.numel())
                        counts["teacher_below_zero"] += int((teacher < 0.0).sum())
                        counts["teacher_above_one"] += int((teacher > 1.0).sum())
                        counts["prediction_below_zero"] += int((prediction < 0.0).sum())
                        counts["prediction_above_one"] += int((prediction > 1.0).sum())
                mse = sse / scalar_count
                record = {
                    "view_index": view_index,
                    "view_name": inputs.view_names[view_index],
                    "sse": sse,
                    "scalar_count": scalar_count,
                    "mse": mse,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                for key, value in counts.items():
                    record[f"{key}_count"] = value
                    record[f"{key}_fraction"] = value / scalar_count
                    totals[key] += value
                totals["scalar_count"] += scalar_count
                per_view.append(record)
                view_means.append(mse)
        aggregate = {
            "risk_measure": name,
            "offsets": [list(offset) for offset in offsets],
            "equal_view_mse": sum(view_means) / len(view_means),
            "per_view": per_view,
            "scalar_count": totals["scalar_count"],
        }
        for key in (
            "teacher_below_zero",
            "teacher_above_one",
            "prediction_below_zero",
            "prediction_above_one",
        ):
            aggregate[f"{key}_count"] = totals[key]
            aggregate[f"{key}_fraction"] = totals[key] / totals["scalar_count"]
        return aggregate


__all__ = [
    "CompactTrainConfig",
    "CompactTrainer",
    "PROPOSAL_MODES",
    "SCHEDULE_MODES",
    "TARGET_MODES",
    "ProposalMode",
    "ScheduleMode",
    "TargetMode",
    "build_view_schedule",
    "observation_digest",
    "preflight_observations",
    "step_sample_seed",
]
