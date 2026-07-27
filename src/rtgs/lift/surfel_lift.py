"""Closed-form surfel construction of covariance, opacity, and colour from 2D captures.

This module implements ADR-XXXX (`docs/ADR-XXXX-surfel-lift.md`).  Beam fusion produces
accurate *means* whose rendered state is nonetheless empty, because the information that
lives in the non-positional parameters is destroyed between Stage 1 and Stage 2.  Here
that information is rebuilt from the per-view 2D Gaussian captures alone.

Governing constraints, restated because they are load-bearing for every function below:

1. **No optimization.**  Every step is closed-form or a bounded box-constrained least-squares
   solve.  There is no autograd graph and no render loss anywhere in this file (ADR §A4).
2. **Image-free.**  Inputs are the lifted means, the per-view 2D Gaussian parameters, the
   camera calibration, the beam-fusion contributor lineage, and — for §3 only — the
   Stage-1 per-view alpha.  Source RGB is never read.
3. **SPD by construction.**  Covariance leaves this module as ``(quaternion, log sigma)``;
   no raw matrix is ever fitted, so a non-SPD result is structurally impossible.
4. **Fixed order: rotation -> scale -> opacity -> colour.**  Each stage consumes the previous
   one.  Solving opacity around a wrong covariance converges to the opacity that best
   *compensates* the covariance error and hides it from every later diagnostic.

The five kernels are separately callable and separately testable:
:func:`estimate_surfel_rotations` (§1), :func:`estimate_surfel_scales` (§2),
:func:`solve_transmittance_opacity` (§3), :func:`estimate_surfel_colors` (§4), and
:func:`build_surfel_lift`, which runs them in the mandated order and collects the §5 flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D, rotmat_to_quat
from rtgs.core.sh import rgb_to_sh

_EPS = 1e-30


@dataclass(frozen=True)
class SurfelLiftConfig:
    """Every knob in ADR-XXXX.  ``rho`` is the only intended sweep axis (E7)."""

    # §1 rotation
    frame_neighbors: int = 12
    normal_disagree_deg: float = 30.0
    # §2 scale
    rho: float = 0.1
    max_incidence_deg: float = 70.0
    aspect_cap: float = 3.0
    scale_disagree_ratio: float = 3.0
    # §5 degenerate cases
    line_aspect_ratio: float = 8.0
    isolation_factor: float = 3.0
    single_view_rho_factor: float = 2.0
    # §3 opacity
    solve_opacity: bool = True
    alpha_clip: float = 0.995
    pixel_stride: int = 4
    opacity_min: float = 1e-3
    opacity_max: float = 0.99
    nnls_iterations: int = 300
    support_sigma: float = 3.0
    merge_candidate_opacity: float = 5e-3
    # The §3-off value.  0.10 is the incumbent beam-fusion ``init_opacity``, so the E8 contrast
    # is "solve versus the pipeline default", not "solve versus a second new constant".
    fallback_opacity: float = 0.1
    # implementation
    neighbor_chunk: int = 4096
    sh_degree: int = 3

    def __post_init__(self) -> None:
        for name in ("frame_neighbors", "pixel_stride", "nnls_iterations", "neighbor_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.frame_neighbors < 3:
            raise ValueError("frame_neighbors must be at least three to span a local plane")
        if isinstance(self.sh_degree, bool) or not isinstance(self.sh_degree, int):
            raise TypeError("sh_degree must be an integer")
        if self.sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        for name in (
            "rho",
            "aspect_cap",
            "scale_disagree_ratio",
            "line_aspect_ratio",
            "isolation_factor",
            "single_view_rho_factor",
            "support_sigma",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.aspect_cap < 1.0:
            raise ValueError("aspect_cap must be at least one")
        if not 0.0 < self.max_incidence_deg < 90.0:
            raise ValueError("max_incidence_deg must lie strictly inside (0, 90)")
        if not 0.0 <= self.normal_disagree_deg <= 90.0:
            raise ValueError("normal_disagree_deg must lie in [0, 90]")
        if not 0.0 < self.alpha_clip < 1.0:
            raise ValueError("alpha_clip must lie in (0,1)")
        if not 0.0 < self.opacity_min <= self.opacity_max < 1.0:
            raise ValueError("opacity bounds must satisfy 0 < opacity_min <= opacity_max < 1")
        for name in ("merge_candidate_opacity", "fallback_opacity"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0,1)")
        if not isinstance(self.solve_opacity, bool):
            raise TypeError("solve_opacity must be bool")


@dataclass(frozen=True)
class SurfelFrames:
    """§1 output: a local surface frame per primitive plus its planarity and cross-checks."""

    normals: torch.Tensor  # (N,3) unit, sign-disambiguated toward the contributing cameras
    tangents: torch.Tensor  # (N,3,2) unit, orthonormal to normals
    coherence: torch.Tensor  # (N,) c_hat = 3 (l2 - l3) / (l1 + l2 + l3), in [0,1]
    spacing: torch.Tensor  # (N,) mean distance to the three nearest neighbours
    knn_radius: torch.Tensor  # (N,) distance to the k-th neighbour
    axis_normal_angle_deg: torch.Tensor  # (N,) angle to the 2D-axis null-space normal; nan if K<2
    n_contributors: torch.Tensor  # (N,) int64


@dataclass(frozen=True)
class SurfelScales:
    """§2 output: fused tangential sigmas, the normal-axis prior, and the fusion diagnostics."""

    sigma_t1: torch.Tensor  # (N,)
    sigma_t2: torch.Tensor  # (N,)
    sigma_normal: torch.Tensor  # (N,)
    used_views: torch.Tensor  # (N,) int64, contributors surviving the incidence gate
    scale_ratio: torch.Tensor  # (N,) max/min tangential candidate ratio across views
    median_aspect_2d: torch.Tensor  # (N,) median 2D fit aspect over contributors


@dataclass(frozen=True)
class OpacitySolve:
    """§3 output: the box-constrained NNLS solution and its residual bookkeeping."""

    opacity: torch.Tensor  # (N,)
    residual_before: float
    residual_after: float
    target_norm: float
    n_equations: int
    n_nonzeros: int
    n_at_upper_bound: int
    n_at_lower_bound: int
    iterations: int
    linearization_valid_fraction: float


@dataclass(frozen=True)
class SurfelLiftResult:
    """The complete lifted initialization plus every §5 flag and audit number."""

    gaussians: Gaussians3D
    frames: SurfelFrames
    scales: SurfelScales
    opacity_solve: OpacitySolve | None
    flags: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Lineage access
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Lineage:
    """Contributor lineage flattened into aligned float64 CSR arrays."""

    offsets: torch.Tensor  # (N+1,) int64
    view_indices: torch.Tensor  # (M,) int64
    component_indices: torch.Tensor  # (M,) int64
    depths: torch.Tensor  # (M,) float64
    primitive_of_link: torch.Tensor  # (M,) int64


def _lineage(beam) -> _Lineage:
    offsets = beam.component_offsets.detach().to(torch.int64).cpu()
    view_indices = beam.contributor_view_indices.detach().to(torch.int64).cpu()
    component_indices = beam.contributor_component_indices.detach().to(torch.int64).cpu()
    depths = beam.contributor_depths.detach().to(torch.float64).cpu()
    n = int(offsets.numel() - 1)
    if n <= 0:
        raise ValueError("beam lineage carries no components")
    counts = offsets[1:] - offsets[:-1]
    if bool((counts <= 0).any()):
        raise ValueError("every component must carry at least one contributor")
    if not (view_indices.numel() == component_indices.numel() == depths.numel()):
        raise ValueError("beam contributor arrays disagree in length")
    if int(offsets[-1]) != view_indices.numel():
        raise ValueError("beam offsets do not span the contributor arrays")
    primitive_of_link = torch.repeat_interleave(torch.arange(n, dtype=torch.int64), counts)
    return _Lineage(
        offsets=offsets,
        view_indices=view_indices,
        component_indices=component_indices,
        depths=depths,
        primitive_of_link=primitive_of_link,
    )


def _segment_reduce(
    values: torch.Tensor,
    primitive_of_link: torch.Tensor,
    n: int,
    *,
    how: str,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce per-link ``values`` to per-primitive values without a Python loop over links.

    ``median`` sorts each segment; ties follow the lower middle element, which makes an
    even-length track's fused value deterministic and conservative (ADR §2.2 fusion rule).
    """
    if valid is None:
        valid = torch.ones(values.shape[0], dtype=torch.bool)
    keys = primitive_of_link[valid]
    payload = values[valid].to(torch.float64)
    out = torch.zeros(n, dtype=torch.float64)
    if how == "sum":
        return out.index_add_(0, keys, payload)
    if how == "max":
        out = torch.full((n,), -math.inf, dtype=torch.float64)
        return out.index_reduce_(0, keys, payload, "amax", include_self=True)
    if how == "min":
        out = torch.full((n,), math.inf, dtype=torch.float64)
        return out.index_reduce_(0, keys, payload, "amin", include_self=True)
    if how != "median":
        raise ValueError("how must be one of sum, max, min, median")
    # Sort by value (stable), then by key (stable): each segment becomes contiguous and
    # internally ascending, so the median is a gather at the segment's lower middle slot.
    counts = torch.zeros(n, dtype=torch.int64).index_add_(
        0, keys, torch.ones_like(keys, dtype=torch.int64)
    )
    starts = torch.zeros(n, dtype=torch.int64)
    starts[1:] = counts.cumsum(0)[:-1]
    order = torch.argsort(payload, stable=True)
    order = order[torch.argsort(keys[order], stable=True)]
    ordered_keys = keys[order]
    within = torch.arange(order.numel(), dtype=torch.int64) - starts[ordered_keys]
    hit = within == ((counts - 1) // 2)[ordered_keys]
    result = torch.zeros(n, dtype=torch.float64)
    result[ordered_keys[hit]] = payload[order][hit]
    return result


# --------------------------------------------------------------------------------------------
# §1 Rotation
# --------------------------------------------------------------------------------------------


def _pca_neighbourhoods(
    points: torch.Tensor, k: int, chunk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunk-invariant kNN PCA: descending eigenvalues, eigenvectors, spacing, kNN radius."""
    n = points.shape[0]
    k = min(k, n - 1)
    eigenvalues = torch.zeros(n, 3, dtype=torch.float64)
    eigenvectors = torch.zeros(n, 3, 3, dtype=torch.float64)
    spacing = torch.zeros(n, dtype=torch.float64)
    radius = torch.zeros(n, dtype=torch.float64)
    spacing_k = min(3, k)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = points[start:stop]
        # See rtgs.lift.surfel_init: the matmul expansion is not chunk invariant and loses
        # precision to cancellation for nearby points.
        distances = torch.cdist(block, points, compute_mode="donot_use_mm_for_euclid_dist")
        rows = torch.arange(start, stop)
        distances[rows - start, rows] = torch.inf
        nearest, indices = distances.topk(k, largest=False)
        spacing[start:stop] = nearest[:, :spacing_k].mean(dim=-1)
        radius[start:stop] = nearest[:, -1]
        neighbourhood = points[indices]
        centered = neighbourhood - neighbourhood.mean(dim=1, keepdim=True)
        covariance = centered.transpose(1, 2) @ centered / float(k)
        values, vectors = torch.linalg.eigh(covariance)  # ascending
        eigenvalues[start:stop] = values.clamp_min(0.0).flip(-1)  # descending l1 >= l2 >= l3
        eigenvectors[start:stop] = vectors.flip(-1)
    return eigenvalues, eigenvectors, spacing, radius


def _orthonormalize(normals: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """Re-orthonormalize ``(e1, e2)`` against ``normals`` (ADR §1 tangent frame)."""
    t1 = e1 - (e1 * normals).sum(-1, keepdim=True) * normals
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    t2 = e2 - (e2 * normals).sum(-1, keepdim=True) * normals - (e2 * t1).sum(-1, keepdim=True) * t1
    t2 = t2 / t2.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.stack([t1, t2], dim=-1)


def _observation_axes(observation) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,2) unit major-axis directions and (N,) major/minor aspect of a 2D capture."""
    variances = observation.effective_variances().to(torch.float64)
    theta = observation.rotations.to(torch.float64)
    cos, sin = theta.cos(), theta.sin()
    major_is_first = variances[:, 0] >= variances[:, 1]
    axis_x = torch.where(major_is_first, cos, -sin)
    axis_y = torch.where(major_is_first, sin, cos)
    axis = torch.stack([axis_x, axis_y], dim=-1)
    hi = variances.amax(dim=-1).clamp_min(_EPS)
    lo = variances.amin(dim=-1).clamp_min(_EPS)
    return axis, (hi / lo).sqrt()


def estimate_surfel_rotations(
    means: torch.Tensor,
    inputs,
    beam,
    config: SurfelLiftConfig | None = None,
) -> SurfelFrames:
    """§1: normals from local PCA over the lifted means, with the 2D-axis cross-check.

    The normal is the smallest-variance neighbourhood direction, *not* the mean contributing
    ray: a ray estimates the viewing direction, and its error is maximal at grazing incidence,
    exactly where covariance matters most.  The secondary normal built from the back-projected
    2D major axes is a cross-check and is deliberately never fused into the PCA normal —
    averaging two differently-wrong normals produces a confidently wrong one.
    """
    config = config or SurfelLiftConfig()
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N,3)")
    points = means.detach().to(torch.float64).cpu()
    n = points.shape[0]
    if n <= 3:
        raise ValueError("surfel rotation estimation needs more than three primitives")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("means contain non-finite values")

    eigenvalues, eigenvectors, spacing, radius = _pca_neighbourhoods(
        points, config.frame_neighbors, config.neighbor_chunk
    )
    normals = eigenvectors[:, :, 2]  # smallest eigenvalue direction
    total = eigenvalues.sum(dim=-1).clamp_min(_EPS)
    coherence = (3.0 * (eigenvalues[:, 1] - eigenvalues[:, 2]) / total).clamp(0.0, 1.0)

    lineage = _lineage(beam)
    cameras = inputs.cameras
    positions = torch.stack(
        [camera.position.detach().to(torch.float64).cpu() for camera in cameras]
    )
    link_to_camera = positions[lineage.view_indices] - points[lineage.primitive_of_link]
    link_to_camera = link_to_camera / link_to_camera.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    mean_to_camera = torch.zeros(n, 3, dtype=torch.float64).index_add_(
        0, lineage.primitive_of_link, link_to_camera
    )
    flip = (normals * mean_to_camera).sum(-1) < 0
    normals = torch.where(flip[:, None], -normals, normals)
    tangents = _orthonormalize(normals, eigenvectors[:, :, 0], eigenvectors[:, :, 1])

    counts = lineage.offsets[1:] - lineage.offsets[:-1]
    axis_angle = _axis_normal_disagreement(points, normals, inputs, lineage, counts)

    return SurfelFrames(
        normals=normals,
        tangents=tangents,
        coherence=coherence,
        spacing=spacing.clamp_min(1e-12),
        knn_radius=radius.clamp_min(1e-12),
        axis_normal_angle_deg=axis_angle,
        n_contributors=counts,
    )


def _axis_normal_disagreement(
    points: torch.Tensor,
    normals: torch.Tensor,
    inputs,
    lineage: _Lineage,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Angle between the PCA normal and the null space of the back-projected 2D major axes.

    The major axis of a 2D capture is the image-space projection of a surface tangent, so with
    K >= 2 views the surface normal is the smallest right singular vector of the stacked
    tangents.  Primitives with K < 2 get ``nan``: the cross-check does not exist for them.
    """
    n = points.shape[0]
    tangents = torch.zeros(lineage.view_indices.numel(), 3, dtype=torch.float64)
    for view, (camera, observation) in enumerate(zip(inputs.cameras, inputs.observations)):
        rows = (lineage.view_indices == view).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        axis2d, _ = _observation_axes(observation)
        axis = axis2d[lineage.component_indices[rows]]
        rotation = camera.R.detach().to(torch.float64).cpu()
        # World direction of an image-x / image-y displacement at the contributor's own depth.
        world_x = rotation[0, :]
        world_y = rotation[1, :]
        depth = lineage.depths[rows][:, None]
        direction = (
            axis[:, 0:1] * world_x[None, :] / float(camera.fx)
            + axis[:, 1:2] * world_y[None, :] / float(camera.fy)
        ) * depth
        tangents[rows] = direction / direction.norm(dim=-1, keepdim=True).clamp_min(_EPS)

    # Sum of outer products per primitive: the null space of the stacked tangents is the
    # eigenvector of the smallest eigenvalue of that scatter matrix.
    outer = tangents[:, :, None] * tangents[:, None, :]
    scatter = torch.zeros(n, 3, 3, dtype=torch.float64).index_add_(
        0, lineage.primitive_of_link, outer
    )
    values, vectors = torch.linalg.eigh(scatter)
    axis_normal = vectors[:, :, 0]
    cosine = (axis_normal * normals).sum(-1).abs().clamp(0.0, 1.0)
    angle = torch.rad2deg(torch.arccos(cosine))
    return torch.where(counts >= 2, angle, torch.full_like(angle, float("nan")))


# --------------------------------------------------------------------------------------------
# §2 Scale
# --------------------------------------------------------------------------------------------


def _covariances_2d(observation) -> torch.Tensor:
    """(N,2,2) rendered pixel covariance of a 2D capture, in the fit window's pixel units."""
    variances = observation.effective_variances().to(torch.float64)
    theta = observation.rotations.to(torch.float64)
    cos, sin = theta.cos(), theta.sin()
    off = cos * sin * (variances[:, 0] - variances[:, 1])
    xx = cos.square() * variances[:, 0] + sin.square() * variances[:, 1]
    yy = sin.square() * variances[:, 0] + cos.square() * variances[:, 1]
    row0 = torch.stack([xx, off], dim=-1)
    row1 = torch.stack([off, yy], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def estimate_surfel_scales(
    means: torch.Tensor,
    frames: SurfelFrames,
    inputs,
    beam,
    config: SurfelLiftConfig | None = None,
) -> tuple[SurfelScales, dict[str, torch.Tensor]]:
    """§2: tangential extent back-projected from the 2D footprints; the normal axis is a prior.

    Stage 1 *measured* the world-space tangential footprint in pixels.  Recovering it is a
    back-projection, not an estimate.  The normal axis is not recoverable from two views even
    in principle, so ``rho`` supplies it as a declared prior — that is the covariance-
    intersection lesson, and it is why ``rho`` is the only sweep axis in E7.
    """
    config = config or SurfelLiftConfig()
    points = means.detach().to(torch.float64).cpu()
    n = points.shape[0]
    lineage = _lineage(beam)
    m = lineage.view_indices.numel()

    sigma_axis = torch.zeros(m, 2, dtype=torch.float64)
    cos_incidence = torch.zeros(m, dtype=torch.float64)
    aspect2d = torch.zeros(m, dtype=torch.float64)
    valid = torch.zeros(m, dtype=torch.bool)
    cos_gate = math.cos(math.radians(config.max_incidence_deg))

    normals = frames.normals
    tangent_basis = frames.tangents  # (N,3,2)

    for view, (camera, observation) in enumerate(zip(inputs.cameras, inputs.observations)):
        rows = (lineage.view_indices == view).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        primitive = lineage.primitive_of_link[rows]
        depth = lineage.depths[rows]
        component = lineage.component_indices[rows]

        covariance2d = _covariances_2d(observation)[component]
        _, view_aspect = _observation_axes(observation)
        aspect2d[rows] = view_aspect[component]

        focal = math.sqrt(float(camera.fx) * float(camera.fy))
        if focal <= 0:
            raise ValueError("camera focal lengths must be positive")
        # §2.1 local perspective Jacobian J ~ (f / z) I : pixels^2 -> world units^2.
        world_scale = (depth / focal).square()[:, None, None]
        sigma_world_plane = covariance2d * world_scale

        rotation = camera.R.detach().to(torch.float64).cpu()
        camera_x = rotation[0, :]
        camera_y = rotation[1, :]
        position = camera.position.detach().to(torch.float64).cpu()

        to_camera = position[None, :] - points[primitive]
        to_camera = to_camera / to_camera.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        cosine = (normals[primitive] * to_camera).sum(-1).abs().clamp(0.0, 1.0)
        cos_incidence[rows] = cosine
        keep = cosine >= cos_gate
        valid[rows] = keep

        # Foreshortening: the footprint along the *projected normal* image direction is
        # compressed by cos(theta).  Undo it on that axis only.
        normal_image = torch.stack(
            [
                (normals[primitive] * camera_x[None, :]).sum(-1),
                (normals[primitive] * camera_y[None, :]).sum(-1),
            ],
            dim=-1,
        )
        norm = normal_image.norm(dim=-1, keepdim=True)
        degenerate = norm.squeeze(-1) < 1e-9
        unit = torch.where(
            degenerate[:, None],
            torch.tensor([1.0, 0.0], dtype=torch.float64).expand_as(normal_image),
            normal_image / norm.clamp_min(_EPS),
        )
        perpendicular = torch.stack([-unit[:, 1], unit[:, 0]], dim=-1)
        basis = torch.stack([unit, perpendicular], dim=-2)  # (R,2,2) rows are the axes
        in_basis = basis @ sigma_world_plane @ basis.transpose(-1, -2)
        correction = torch.ones_like(in_basis)
        correction[:, 0, 0] = 1.0 / cosine.clamp_min(math.cos(math.radians(89.0))).square()
        correction[:, 0, 1] = 1.0 / cosine.clamp_min(math.cos(math.radians(89.0)))
        correction[:, 1, 0] = correction[:, 0, 1]
        corrected = in_basis * correction
        sigma_image_world = basis.transpose(-1, -2) @ corrected @ basis

        # Embed in world (the image plane's world basis is the camera's own x/y axes), then
        # take the 2x2 block in the shared tangent frame from §1.
        embed = torch.stack([camera_x, camera_y], dim=-1)  # (3,2)
        sigma_world = embed[None] @ sigma_image_world @ embed.transpose(-1, -2)[None]
        tangent = tangent_basis[primitive]  # (R,3,2)
        sigma_tangent = tangent.transpose(-1, -2) @ sigma_world @ tangent
        sigma_axis[rows] = torch.stack(
            [sigma_tangent[:, 0, 0].clamp_min(0.0), sigma_tangent[:, 1, 1].clamp_min(0.0)],
            dim=-1,
        ).sqrt()

    used = _segment_reduce(
        torch.ones(m, dtype=torch.float64), lineage.primitive_of_link, n, how="sum", valid=valid
    ).to(torch.int64)
    # Primitives whose every observation was rejected by the incidence gate fall back to their
    # unrejected candidates; a surfel with no scale evidence at all is worse than a grazing one.
    rescue = used == 0
    if bool(rescue.any()):
        valid = valid | rescue[lineage.primitive_of_link]
        used = _segment_reduce(
            torch.ones(m, dtype=torch.float64),
            lineage.primitive_of_link,
            n,
            how="sum",
            valid=valid,
        ).to(torch.int64)

    log_axis = sigma_axis.clamp_min(_EPS).log()
    fused = torch.stack(
        [
            _segment_reduce(
                log_axis[:, axis], lineage.primitive_of_link, n, how="median", valid=valid
            )
            for axis in (0, 1)
        ],
        dim=-1,
    ).exp()

    ratios = []
    for axis in (0, 1):
        hi = _segment_reduce(
            log_axis[:, axis], lineage.primitive_of_link, n, how="max", valid=valid
        )
        lo = _segment_reduce(
            log_axis[:, axis], lineage.primitive_of_link, n, how="min", valid=valid
        )
        ratios.append((hi - lo).clamp_min(0.0).exp())
    scale_ratio = torch.maximum(*ratios)
    scale_ratio = torch.where(used >= 2, scale_ratio, torch.ones_like(scale_ratio))

    median_aspect = _segment_reduce(aspect2d, lineage.primitive_of_link, n, how="median")

    sigma_t1, sigma_t2 = fused[:, 0].clone(), fused[:, 1].clone()
    geometric = (sigma_t1 * sigma_t2).clamp_min(_EPS).sqrt()

    # §2.3 aspect cap: rotation gradients are degenerate at l1 ~ l2 and stiff at extreme
    # aspect; both freeze wrong orientations.  The cap is relaxed by the coarse-to-fine
    # schedule during training, never here.
    limit = math.sqrt(config.aspect_cap)
    sigma_t1 = sigma_t1.clamp(geometric / limit, geometric * limit)
    sigma_t2 = sigma_t2.clamp(geometric / limit, geometric * limit)
    geometric = (sigma_t1 * sigma_t2).clamp_min(_EPS).sqrt()

    # §2.3 anisotropy modulation by the PCA coherence.
    coherence = frames.coherence
    line_like = median_aspect > config.line_aspect_ratio
    blend = torch.where(line_like, torch.zeros_like(coherence), coherence)
    sigma_t1 = blend * sigma_t1 + (1.0 - blend) * geometric
    sigma_t2 = blend * sigma_t2 + (1.0 - blend) * geometric
    geometric = (sigma_t1 * sigma_t2).clamp_min(_EPS).sqrt()

    # §2.3 / §5: the normal axis is a prior, widened where depth confidence is absent.
    isolated = frames.knn_radius > config.isolation_factor * frames.spacing.median()
    single_view = frames.n_contributors <= 1
    rho = torch.full_like(geometric, float(config.rho))
    rho = torch.where(single_view, rho * config.single_view_rho_factor, rho)
    rho = torch.where(isolated, torch.ones_like(rho), rho)
    sigma_normal = (rho * geometric).clamp_min(_EPS)
    sigma_t1 = torch.where(isolated, geometric, sigma_t1)
    sigma_t2 = torch.where(isolated, geometric, sigma_t2)

    scales = SurfelScales(
        sigma_t1=sigma_t1,
        sigma_t2=sigma_t2,
        sigma_normal=sigma_normal,
        used_views=used,
        scale_ratio=scale_ratio,
        median_aspect_2d=median_aspect,
    )
    flags = {
        "flag_single_view": single_view,
        "flag_isolated": isolated,
        "flag_line_like": line_like,
        "flag_scale_disagree": scale_ratio > config.scale_disagree_ratio,
    }
    return scales, flags


# --------------------------------------------------------------------------------------------
# §3 Opacity
# --------------------------------------------------------------------------------------------


def _gaussians_from_frames(
    means: torch.Tensor,
    frames: SurfelFrames,
    scales: SurfelScales,
    opacity: torch.Tensor,
    sh: torch.Tensor,
) -> Gaussians3D:
    """Assemble ``R = [t1 t2 n]`` into a quaternion; scales stay in log space (SPD by design)."""
    rotation = torch.cat([frames.tangents, frames.normals[:, :, None]], dim=-1)
    flip = torch.linalg.det(rotation) < 0
    rotation = rotation.clone()
    rotation[flip, :, 0] *= -1.0
    quats = rotmat_to_quat(rotation)
    sigma = torch.stack([scales.sigma_t1, scales.sigma_t2, scales.sigma_normal], dim=-1)
    log_scales = sigma.clamp_min(_EPS).log()
    dtype = means.dtype
    return Gaussians3D(
        means=means.detach().clone(),
        quats=quats.to(dtype),
        log_scales=log_scales.to(dtype),
        opacity=opacity.to(dtype),
        sh=sh.to(dtype),
    )


@dataclass(frozen=True)
class AlphaTarget:
    """One fitted view's Stage-1 alpha composite, addressed in full-canvas pixel coordinates."""

    view_index: int
    alpha: torch.Tensor  # (H,W) in [0,1]
    origin: tuple[int, int]  # (x, y) of alpha[0,0] in the full canvas


def _assemble_transmittance_system(
    gaussians: Gaussians3D,
    cameras: list[Camera],
    targets: list[AlphaTarget],
    config: SurfelLiftConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Stack ``sum_i alpha_i G_i(x) = -log(1 - A(x))`` over views and a strided pixel subsample.

    Returns ``(row, col, value, target, valid_fraction)`` in COO form.  The subsample is every
    ``pixel_stride``-th pixel in x and y; the system is massively overdetermined, so the stride
    is an implementation default and not a tunable.
    """
    from rtgs.render.projection import project_gaussians_ewa

    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    row_base = 0
    inside_bound = 0
    total_terms = 0
    stride = config.pixel_stride
    support = config.support_sigma

    for entry in targets:
        camera = cameras[entry.view_index]
        alpha = entry.alpha.detach().to(torch.float64)
        height, width = alpha.shape
        ys = torch.arange(0, height, stride)
        xs = torch.arange(0, width, stride)
        sub = alpha[ys][:, xs]
        pixel_x = (xs.to(torch.float64) + 0.5) + float(entry.origin[0])
        pixel_y = (ys.to(torch.float64) + 0.5) + float(entry.origin[1])
        n_rows = sub.numel()

        clipped = sub.clamp(0.0, config.alpha_clip)
        target_rows.append(-torch.log1p(-clipped).reshape(-1))

        projection = project_gaussians_ewa(gaussians.detach(), camera, dilation=0.0)
        mean2d = projection.means2d.to(torch.float64)
        cov2d = projection.covariances2d.to(torch.float64)
        depth = projection.depth.to(torch.float64)
        determinant = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
        visible = (depth > 0) & (determinant > 1e-18)
        index = visible.nonzero(as_tuple=True)[0]
        if index.numel() == 0:
            row_base += n_rows
            continue
        mean2d, cov2d = mean2d[index], cov2d[index]
        determinant = determinant[index]
        inverse = (
            torch.stack(
                [
                    torch.stack([cov2d[:, 1, 1], -cov2d[:, 0, 1]], dim=-1),
                    torch.stack([-cov2d[:, 1, 0], cov2d[:, 0, 0]], dim=-1),
                ],
                dim=-2,
            )
            / determinant[:, None, None]
        )
        radius_x = support * cov2d[:, 0, 0].clamp_min(0.0).sqrt()
        radius_y = support * cov2d[:, 1, 1].clamp_min(0.0).sqrt()

        # Bucket the strided grid so each primitive only touches its own footprint.
        lo_x = torch.searchsorted(pixel_x, mean2d[:, 0] - radius_x)
        hi_x = torch.searchsorted(pixel_x, mean2d[:, 0] + radius_x, right=True)
        lo_y = torch.searchsorted(pixel_y, mean2d[:, 1] - radius_y)
        hi_y = torch.searchsorted(pixel_y, mean2d[:, 1] + radius_y, right=True)
        count_x = (hi_x - lo_x).clamp_min(0)
        count_y = (hi_y - lo_y).clamp_min(0)
        counts = count_x * count_y
        touching = counts > 0
        if not bool(touching.any()):
            row_base += n_rows
            continue
        keep = touching.nonzero(as_tuple=True)[0]
        primitive = index[keep]
        flat_counts = counts[keep]
        repeat = torch.repeat_interleave(torch.arange(keep.numel()), flat_counts)
        offset = torch.arange(int(flat_counts.sum())) - torch.repeat_interleave(
            torch.cat([torch.zeros(1, dtype=torch.int64), flat_counts.cumsum(0)[:-1]]), flat_counts
        )
        local_x = offset % count_x[keep][repeat]
        local_y = offset // count_x[keep][repeat]
        grid_x = lo_x[keep][repeat] + local_x
        grid_y = lo_y[keep][repeat] + local_y

        delta = torch.stack(
            [pixel_x[grid_x] - mean2d[keep][repeat, 0], pixel_y[grid_y] - mean2d[keep][repeat, 1]],
            dim=-1,
        )
        conic = inverse[keep][repeat]
        power = 0.5 * (
            delta[:, 0] * (conic[:, 0, 0] * delta[:, 0] + conic[:, 0, 1] * delta[:, 1])
            + delta[:, 1] * (conic[:, 1, 0] * delta[:, 0] + conic[:, 1, 1] * delta[:, 1])
        )
        weight = torch.exp(-power.clamp_min(0.0))
        alive = weight > 1e-6
        if not bool(alive.any()):
            row_base += n_rows
            continue
        rows.append(row_base + grid_y[alive] * xs.numel() + grid_x[alive])
        cols.append(primitive[repeat][alive])
        values.append(weight[alive])
        inside_bound += int((weight[alive] <= 0.7).sum())
        total_terms += int(alive.sum())
        row_base += n_rows

    if not rows:
        raise ValueError("the transmittance system is empty: no primitive covers any sample")
    valid_fraction = inside_bound / max(total_terms, 1)
    return (
        torch.cat(rows),
        torch.cat(cols),
        torch.cat(values),
        torch.cat(target_rows),
        valid_fraction,
    )


def solve_transmittance_opacity(
    gaussians: Gaussians3D,
    cameras: list[Camera],
    targets: list[AlphaTarget],
    config: SurfelLiftConfig | None = None,
) -> OpacitySolve:
    """§3: log-transmittance linearization followed by a box-constrained NNLS.

    ``log T(x) = sum_i log(1 - a_i G_i(x)) ~ -sum_i a_i G_i(x)`` is accurate for
    ``a_i G_i <~ 0.7`` and, crucially, *order-free* — which is what makes it usable as a
    closed-form target at all.  The solve is a projected-gradient descent on
    ``0.5 ||G a - b||^2`` over ``a in [opacity_min, opacity_max]``; that is exactly the
    objective ADR §3 decomposes into per-tile Lawson-Hanson solves, and the decomposition is a
    parallelization strategy rather than a different problem.
    """
    config = config or SurfelLiftConfig()
    row, col, value, target, valid_fraction = _assemble_transmittance_system(
        gaussians, cameras, targets, config
    )
    n = gaussians.n
    n_rows = int(target.numel())

    def forward(alpha: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(n_rows, dtype=torch.float64)
        return out.index_add_(0, row, value * alpha[col])

    def transpose(residual: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(n, dtype=torch.float64)
        return out.index_add_(0, col, value * residual[row])

    # Power iteration for the Lipschitz constant of the normal operator.
    probe = torch.ones(n, dtype=torch.float64)
    for _ in range(24):
        probe = transpose(forward(probe))
        norm = probe.norm().clamp_min(_EPS)
        probe = probe / norm
    lipschitz = float((transpose(forward(probe)) * probe).sum().clamp_min(1e-12))
    step = 1.0 / lipschitz

    alpha = torch.full((n,), 0.5 * (config.opacity_min + config.opacity_max), dtype=torch.float64)
    residual_before = float((forward(alpha) - target).square().sum())
    momentum = alpha.clone()
    theta = 1.0
    for _ in range(config.nnls_iterations):
        gradient = transpose(forward(momentum) - target)
        updated = (momentum - step * gradient).clamp(config.opacity_min, config.opacity_max)
        theta_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * theta * theta))
        momentum = updated + ((theta - 1.0) / theta_next) * (updated - alpha)
        alpha, theta = updated, theta_next
    residual_after = float((forward(alpha) - target).square().sum())

    return OpacitySolve(
        opacity=alpha,
        residual_before=residual_before,
        residual_after=residual_after,
        target_norm=float(target.square().sum()),
        n_equations=n_rows,
        n_nonzeros=int(value.numel()),
        n_at_upper_bound=int((alpha >= config.opacity_max - 1e-9).sum()),
        n_at_lower_bound=int((alpha <= config.opacity_min + 1e-9).sum()),
        iterations=config.nnls_iterations,
        linearization_valid_fraction=valid_fraction,
    )


# --------------------------------------------------------------------------------------------
# §4 Colour / SH
# --------------------------------------------------------------------------------------------


def estimate_surfel_colors(
    means: torch.Tensor,
    frames: SurfelFrames,
    inputs,
    beam,
    config: SurfelLiftConfig | None = None,
) -> torch.Tensor:
    """§4: SH-0 is the ``quality * cos(theta)``-weighted mean of the contributing 2D colours.

    Bands >= 1 are initialized to zero.  With K = 2-5 directional samples a band-1 fit is
    underdetermined and fits noise, and cross-view colour variation confounds specularity with
    misfit inseparably at that K.  Zeroing is also what the coarse-to-fine SH schedule expects.
    """
    config = config or SurfelLiftConfig()
    points = means.detach().to(torch.float64).cpu()
    n = points.shape[0]
    lineage = _lineage(beam)
    m = lineage.view_indices.numel()

    colors = torch.zeros(m, 3, dtype=torch.float64)
    weights = torch.zeros(m, dtype=torch.float64)
    for view, (camera, observation) in enumerate(zip(inputs.cameras, inputs.observations)):
        rows = (lineage.view_indices == view).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        primitive = lineage.primitive_of_link[rows]
        colors[rows] = observation.colors.to(torch.float64).cpu()[lineage.component_indices[rows]]
        position = camera.position.detach().to(torch.float64).cpu()
        to_camera = position[None, :] - points[primitive]
        to_camera = to_camera / to_camera.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        # quality_v is the Stage-1 per-gaussian fit quality where captured; this provider does
        # not record one, so it is 1 and the weight is the incidence cosine alone.
        weights[rows] = (frames.normals[primitive] * to_camera).sum(-1).abs().clamp(0.0, 1.0)

    weights = weights.clamp_min(1e-6)
    weighted = torch.zeros(n, 3, dtype=torch.float64).index_add_(
        0, lineage.primitive_of_link, colors * weights[:, None]
    )
    total = torch.zeros(n, dtype=torch.float64).index_add_(0, lineage.primitive_of_link, weights)
    mean_color = (weighted / total.clamp_min(_EPS)[:, None]).clamp(0.0, 1.0)

    bands = (config.sh_degree + 1) ** 2
    sh = torch.zeros(n, bands, 3, dtype=torch.float64)
    sh[:, 0] = rgb_to_sh(mean_color)
    return sh


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def build_surfel_lift(
    inputs,
    beam,
    config: SurfelLiftConfig | None = None,
    *,
    alpha_targets: list[AlphaTarget] | None = None,
) -> SurfelLiftResult:
    """Run §1 -> §2 -> §3 -> §4 in the mandated order and collect the §5 flags.

    ``alpha_targets`` supplies the Stage-1 alpha composites §3 solves against.  When it is
    ``None`` (a capture bundle without packed alpha) the §3 solve is skipped and the fallback
    opacity is used, which is recorded in the diagnostics as ``opacity_source``: §3 cannot be
    silently approximated from something that is not an alpha.
    """
    config = config or SurfelLiftConfig()
    means = beam.gaussians.means.detach().cpu()

    frames = estimate_surfel_rotations(means, inputs, beam, config)
    scales, flags = estimate_surfel_scales(means, frames, inputs, beam, config)
    sh = estimate_surfel_colors(means, frames, inputs, beam, config)

    n = means.shape[0]
    flags["flag_normal_disagree"] = (
        torch.nan_to_num(frames.axis_normal_angle_deg, nan=0.0) > config.normal_disagree_deg
    )

    provisional = _gaussians_from_frames(
        means,
        frames,
        scales,
        torch.full((n,), float(config.fallback_opacity), dtype=torch.float64),
        sh,
    )

    solve: OpacitySolve | None = None
    opacity_source = "fallback_constant"
    opacity = torch.full((n,), float(config.fallback_opacity), dtype=torch.float64)
    saturated_fraction = None
    if config.solve_opacity and alpha_targets:
        saturated = [
            float((entry.alpha >= config.alpha_clip).to(torch.float64).mean())
            for entry in alpha_targets
        ]
        saturated_fraction = saturated
        solve = solve_transmittance_opacity(provisional, inputs.cameras, alpha_targets, config)
        opacity = solve.opacity.clone()
        opacity_source = "transmittance_nnls"
        # §5 single-view primitives have no depth confidence: take the capture alpha directly
        # rather than a solved value the geometry cannot support.
        single = flags["flag_single_view"]
        if bool(single.any()):
            opacity[single] = float(config.fallback_opacity)

    flags["merge_candidate"] = opacity <= config.merge_candidate_opacity
    gaussians = _gaussians_from_frames(means, frames, scales, opacity, sh)

    def summary(values: torch.Tensor) -> dict[str, float]:
        finite = values[torch.isfinite(values)].to(torch.float64)
        if finite.numel() == 0:
            return {"p10": float("nan"), "median": float("nan"), "p90": float("nan")}
        return {
            "p10": float(finite.quantile(0.10)),
            "median": float(finite.median()),
            "p90": float(finite.quantile(0.90)),
        }

    diagnostics: dict[str, object] = {
        "n_components": n,
        "rho": config.rho,
        "aspect_cap": config.aspect_cap,
        "max_incidence_deg": config.max_incidence_deg,
        "opacity_source": opacity_source,
        "alpha_views": 0 if not alpha_targets else len(alpha_targets),
        "saturated_alpha_fraction_per_view": saturated_fraction,
        "coherence": summary(frames.coherence),
        "spacing": summary(frames.spacing),
        "sigma_t1": summary(scales.sigma_t1),
        "sigma_t2": summary(scales.sigma_t2),
        "sigma_normal": summary(scales.sigma_normal),
        "sigma_tangent_over_spacing": summary(
            (scales.sigma_t1 * scales.sigma_t2).sqrt() / frames.spacing
        ),
        "contributors": summary(frames.n_contributors.to(torch.float64)),
        "used_views": summary(scales.used_views.to(torch.float64)),
        "axis_normal_angle_deg": summary(frames.axis_normal_angle_deg),
        "opacity": summary(opacity),
        "flag_fractions": {
            name: float(mask.to(torch.float64).mean()) for name, mask in flags.items()
        },
    }
    if solve is not None:
        diagnostics["opacity_solve"] = {
            "residual_before": solve.residual_before,
            "residual_after": solve.residual_after,
            "target_norm": solve.target_norm,
            "relative_residual": solve.residual_after / max(solve.target_norm, 1e-12),
            "n_equations": solve.n_equations,
            "n_nonzeros": solve.n_nonzeros,
            "n_at_upper_bound": solve.n_at_upper_bound,
            "n_at_lower_bound": solve.n_at_lower_bound,
            "linearization_valid_fraction": solve.linearization_valid_fraction,
        }

    return SurfelLiftResult(
        gaussians=gaussians,
        frames=frames,
        scales=scales,
        opacity_solve=solve,
        flags=flags,
        diagnostics=diagnostics,
    )


__all__ = [
    "AlphaTarget",
    "OpacitySolve",
    "SurfelFrames",
    "SurfelLiftConfig",
    "SurfelLiftResult",
    "SurfelScales",
    "build_surfel_lift",
    "estimate_surfel_colors",
    "estimate_surfel_rotations",
    "estimate_surfel_scales",
    "solve_transmittance_opacity",
]
