"""Reconcile a localization covariance with the covariance a renderer actually needs.

Correspondence-based initializers (``splat_sfm``, ``beam_fusion``) return, for every 3D
component, the covariance of its *position estimate*: beam fusion's covariance-intersection
rule fuses per-view beam precisions, so the result answers "how well is this component
localized?".  Rendering asks a different question — "how much surface does this primitive
cover?" — and the two answers differ systematically in both scale and orientation:

* **Scale.** A precision mean is dominated by the *largest* precision, i.e. by the
  *sharpest* 2D Gaussian a component happened to match.  Every fused component therefore
  inherits the tightest observation in its track rather than the typical one.  On the
  checked-in Janelle bundles the fused sigma is 1.66-1.75x the smallest contributor
  footprint but only 0.44-0.55x the median contributor footprint.
* **Decimation.** Seeding, signature dedupe, and voxel NMS reduce ~40,000 fitted 2D
  Gaussians to ~800 3D components, but the covariance still describes the un-decimated
  observation scale.  The largest fused axis measures only ~0.18-0.26x the distance to the
  component's own nearest neighbours, so the components cannot touch, let alone tile.
* **Orientation.** With cameras on a dome the contributing baseline spans ~160 degrees, so
  no direction is systematically under-triangulated and the residual ~4:1 anisotropy is set
  by heterogeneous per-view footprints.  Measured alignment between the fused short axis and
  a local surface normal is 0.53 (mean |cos|), indistinguishable from the 0.5 random
  baseline: the ellipsoid carries no surface information.

The consequences are exactly the observed failure mode.  Gaussians separated by ~5x their own
sigma accumulate almost no alpha (~6.6 overlapping components are needed to cross 0.5 at
opacity 0.10), and each covers too few pixels to earn a screen-space gradient above the
densification threshold, so the primitives neither render nor densify.  Recovery by gradient
descent is also rate-limited rather than free: scales are optimized as ``log_scales`` under
Adam, whose normalized step moves a log-scale by at most the learning rate, so closing the
measured ~3.1x median scale deficit needs at least ``ln(3.1) / 5e-3 ~ 227`` consecutive
maximal steps -- a floor, not a typical time, and the whole budget of a short schedule.

This module rebuilds the covariance and opacity from the requirement the renderer imposes,
using the localization estimate as a *bound* rather than as the answer:

1. **Local surface frame** — a chunked kNN PCA over the component centers gives a normal
   (smallest-variance direction), two tangents, the local spacing, and a planarity score.
   Orientation comes from the reconstructed surface, not from the camera arc.
2. **Tangential extent** — ``sigma_t = max(coverage_sigma_ratio * spacing, resolution_floor)``.
   The first term is a partition-of-unity condition: isotropic Gaussians of standard
   deviation ``sigma`` on a hexagonal lattice of spacing ``d`` sum to a field whose relative
   ripple (:func:`hexagonal_cover_ripple`) collapses as ``sigma/d`` grows — 81.5% at 0.30,
   13.3% at 0.40, 1.25% at 0.50, 0.31% at 0.55.  ``0.5`` is therefore the smallest ratio that
   covers a surface smoothly, and it is derived rather than tuned.  The second term keeps a
   primitive no finer than what the images resolve.  The rule is budget-adaptive: the cover
   term dominates a decimated set, the resolution term dominates a dense one, and neither is
   a fitted multiplier.
3. **Normal extent** — the measured local out-of-plane spread, raised to the localization
   uncertainty along that same normal (never claim the surface is thinner than the position is
   known), then capped at ``flatness * sigma_t``.  The cap is applied **last** and therefore
   wins: a component whose position is known only to within half its own footprint is a blob,
   not a surface element, and is kept surfel-shaped rather than allowed to round out.
4. **Opacity** — with ``S = 2 pi sigma_t^2 / area_per_component`` kernel units overlapping a
   typical surface point, the accumulated alpha of independent layers is
   ``1 - (1 - o)^S``; inverting for a target gives ``o = 1 - (1 - target_alpha)^(1/S)``.
   Opacity becomes per-component and self-consistent with the chosen extent instead of a
   fixed 0.10.

This is a *rendering-consistent surface-element* covariance, not a recovered physical
matter distribution: the honest physical inputs are the measured local surface frame and
thickness plus the localization bound.  ``reconcile_covariances`` is generic over any
``Gaussians3D`` point set; ``beam_contributor_footprints`` supplies the resolution floor from
beam fusion's own contributor lineage.  Opt-in, CPU-first, deterministic, no RGB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rtgs.core.gaussians3d import Gaussians3D, rotmat_to_quat

# Area of one hexagonal cell with nearest-neighbour spacing d is _HEX_CELL_AREA * d^2.
_HEX_CELL_AREA = math.sqrt(3.0) / 2.0


def hexagonal_cover_ripple(sigma_over_spacing: float, *, shells: int = 14) -> float:
    """Peak-to-trough relative ripple of a hexagonal Gaussian cover, by direct summation.

    Evaluated at the three extremal points of the hexagonal cell (a lattice point, an edge
    midpoint, and the deep hole) and normalized by the cover's mean.  This is the exact
    quantity behind :attr:`SurfelInitConfig.coverage_sigma_ratio`; a leading-shell closed form
    understates it by roughly a third, so the summation is the oracle.
    """
    if not math.isfinite(sigma_over_spacing) or sigma_over_spacing <= 0:
        raise ValueError("sigma_over_spacing must be finite and positive")
    if isinstance(shells, bool) or not isinstance(shells, int) or shells <= 0:
        raise ValueError("shells must be a positive integer")
    sigma = float(sigma_over_spacing)
    basis = torch.tensor(
        [[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]],
        dtype=torch.float64,
    )
    span = torch.arange(-shells, shells + 1, dtype=torch.float64)
    grid_a, grid_b = torch.meshgrid(span, span, indexing="ij")
    centers = torch.stack([grid_a.reshape(-1), grid_b.reshape(-1)], dim=-1) @ basis
    probes = torch.stack(
        [torch.zeros(2, dtype=torch.float64), basis[0] / 2.0, (basis[0] + basis[1]) / 3.0]
    )
    offsets = probes[:, None, :] - centers[None, :, :]
    values = torch.exp(-offsets.square().sum(dim=-1) / (2.0 * sigma**2)).sum(dim=-1)
    return float((values.max() - values.min()) / values.mean())


@dataclass(frozen=True)
class SurfelInitConfig:
    """Controls for the local surface frame, the cover condition, and the opacity rule."""

    frame_neighbors: int = 12
    spacing_neighbors: int = 3
    coverage_sigma_ratio: float = 0.5
    target_alpha: float = 0.9
    min_opacity: float = 0.02
    max_opacity: float = 0.95
    flatness: float = 0.5
    isotropic: bool = False
    use_resolution_floor: bool = True
    use_coverage_opacity: bool = True
    fixed_opacity: float = 0.1
    min_planarity: float = 0.0
    neighbor_chunk: int = 4096
    min_sigma_world: float = 1e-5
    max_sigma_world: float | None = None

    def __post_init__(self) -> None:
        for name in ("frame_neighbors", "spacing_neighbors", "neighbor_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.frame_neighbors < 3:
            raise ValueError("frame_neighbors must be at least three to span a local plane")
        if self.spacing_neighbors > self.frame_neighbors:
            raise ValueError("spacing_neighbors must not exceed frame_neighbors")
        for name in ("coverage_sigma_ratio", "flatness", "min_sigma_world", "fixed_opacity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.target_alpha < 1.0:
            raise ValueError("target_alpha must be in (0,1)")
        if not 0.0 < self.min_opacity <= self.max_opacity < 1.0:
            raise ValueError("opacity bounds must satisfy 0 < min_opacity <= max_opacity < 1")
        if not 0.0 <= self.min_planarity < 1.0:
            raise ValueError("min_planarity must be in [0,1)")
        if self.fixed_opacity >= 1.0:
            raise ValueError("fixed_opacity must be below one")
        if self.max_sigma_world is not None and (
            not math.isfinite(self.max_sigma_world) or self.max_sigma_world <= 0
        ):
            raise ValueError("max_sigma_world must be finite and positive when supplied")
        for name in ("isotropic", "use_resolution_floor", "use_coverage_opacity"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


def effective_cover_sigma(gaussians: Gaussians3D) -> torch.Tensor:
    """(N,) equal-area isotropic equivalent of each primitive's widest cross-section.

    The two largest principal axes span the cross-section a viewer sees when looking down the
    smallest one; ``sqrt(sigma_1 * sigma_2)`` is the isotropic radius of equal area.  This lets
    the cover and opacity rules be applied to a primitive set whose covariance came from
    somewhere else, which is what the attribution controls need.
    """
    sigmas = torch.linalg.eigvalsh(gaussians.covariance().detach().to(torch.float64))
    sigmas = sigmas.clamp_min(0.0).sqrt()
    return (sigmas[:, 1] * sigmas[:, 2]).clamp_min(1e-30).sqrt()


def coverage_opacity(
    sigma_tangent: torch.Tensor,
    spacing: torch.Tensor,
    config: SurfelInitConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-primitive opacity that makes a cover of the given extent reach ``target_alpha``.

    With ``S = 2 pi sigma^2 / cell_area`` kernel units over a typical surface point, stacking
    ``S`` independent layers of opacity ``o`` accumulates ``1 - (1 - o)^S``.  Returns the
    clamped opacity and the raw overlap ``S``.
    """
    config = config or SurfelInitConfig()
    if sigma_tangent.shape != spacing.shape:
        raise ValueError("sigma_tangent and spacing must have the same shape")
    sigma = sigma_tangent.detach().to(torch.float64)
    cell_area = (_HEX_CELL_AREA * spacing.detach().to(torch.float64).square()).clamp_min(1e-30)
    overlap = (2.0 * math.pi * sigma.square() / cell_area).clamp_min(1e-6)
    opacity = 1.0 - (1.0 - config.target_alpha) ** (1.0 / overlap)
    return opacity.clamp(config.min_opacity, config.max_opacity), overlap


@dataclass(frozen=True)
class LocalSurfaceFrames:
    """Per-component local surface geometry measured from the component centers alone."""

    normals: torch.Tensor  # (N,3) unit, smallest-variance neighbourhood direction
    tangents: torch.Tensor  # (N,3,2) unit, orthonormal to normals
    spacing: torch.Tensor  # (N,) mean distance to the nearest ``spacing_neighbors``
    out_of_plane_sigma: torch.Tensor  # (N,) sqrt of the smallest neighbourhood eigenvalue
    planarity: torch.Tensor  # (N,) 1 - sqrt(lambda_0 / lambda_2), in [0,1]


@dataclass(frozen=True)
class SurfelInitResult:
    """Reconciled initialization plus the diagnostics needed to audit the rule."""

    gaussians: Gaussians3D
    frames: LocalSurfaceFrames
    sigma_tangent: torch.Tensor  # (N,)
    sigma_normal: torch.Tensor  # (N,)
    overlap: torch.Tensor  # (N,) expected kernel units covering a local surface point
    resolution_floor_active: torch.Tensor  # (N,) bool
    diagnostics: dict[str, object]


def _validate_points(means: torch.Tensor, config: SurfelInitConfig) -> torch.Tensor:
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N,3)")
    if means.shape[0] <= config.spacing_neighbors:
        raise ValueError(
            "surfel reconciliation needs more components than spacing_neighbors: "
            f"{means.shape[0]} <= {config.spacing_neighbors}"
        )
    if means.is_cuda:
        raise ValueError("surfel reconciliation is CPU-first; move means to CPU")
    points = means.detach().to(torch.float64)
    if not bool(torch.isfinite(points).all()):
        raise ValueError("means contain non-finite values")
    return points


def estimate_local_surface_frames(
    means: torch.Tensor,
    config: SurfelInitConfig | None = None,
) -> LocalSurfaceFrames:
    """Measure a local surface frame, spacing, and planarity from the centers by chunked kNN.

    The neighbourhood covariance is taken about the neighbourhood mean, so the frame
    describes the local surface patch rather than the offset of one component from it.
    """
    config = config or SurfelInitConfig()
    points = _validate_points(means, config)
    n = points.shape[0]
    frame_k = min(config.frame_neighbors, n - 1)
    spacing_k = min(config.spacing_neighbors, frame_k)

    normals = torch.zeros(n, 3, dtype=torch.float64)
    tangents = torch.zeros(n, 3, 2, dtype=torch.float64)
    spacing = torch.zeros(n, dtype=torch.float64)
    out_of_plane = torch.zeros(n, dtype=torch.float64)
    planarity = torch.zeros(n, dtype=torch.float64)

    for start in range(0, n, config.neighbor_chunk):
        stop = min(start + config.neighbor_chunk, n)
        block = points[start:stop]
        # ``donot_use_mm_for_euclid_dist`` forces the direct pairwise form. The default lets
        # torch pick a matmul expansion (|x|^2 - 2x.y + |y|^2) for larger blocks, so the same
        # point pair yields different last bits depending on ``neighbor_chunk``. That breaks the
        # chunk invariance this function promises, and the expansion also loses precision to
        # cancellation for nearby points (~1.7e-11 on a 0.07-spaced float64 lattice, from a
        # geometry term this path then takes a mean and an eigendecomposition of).
        distances = torch.cdist(block, points, compute_mode="donot_use_mm_for_euclid_dist")
        rows = torch.arange(start, stop)
        distances[rows - start, rows] = torch.inf
        nearest, indices = distances.topk(frame_k, largest=False)
        spacing[start:stop] = nearest[:, :spacing_k].mean(dim=-1)
        neighbourhood = points[indices]  # (B,k,3)
        centered = neighbourhood - neighbourhood.mean(dim=1, keepdim=True)
        covariance = centered.transpose(1, 2) @ centered / float(frame_k)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)  # ascending
        eigenvalues = eigenvalues.clamp_min(0.0)
        normals[start:stop] = eigenvectors[:, :, 0]
        tangents[start:stop] = eigenvectors[:, :, 1:3]
        out_of_plane[start:stop] = eigenvalues[:, 0].sqrt()
        planarity[start:stop] = 1.0 - eigenvalues[:, 0].sqrt() / eigenvalues[:, 2].sqrt().clamp_min(
            1e-30
        )

    return LocalSurfaceFrames(
        normals=normals,
        tangents=tangents,
        spacing=spacing.clamp_min(1e-12),
        out_of_plane_sigma=out_of_plane,
        planarity=planarity.clamp(0.0, 1.0),
    )


def _oriented_rotations(frames: LocalSurfaceFrames) -> torch.Tensor:
    """Right-handed (N,3,3) rotations whose columns are ``(t1, t2, normal)``."""
    rotations = torch.cat([frames.tangents, frames.normals[:, :, None]], dim=-1)
    flip = torch.linalg.det(rotations) < 0
    rotations = rotations.clone()
    rotations[flip, :, 0] *= -1.0
    return rotations


def reconcile_covariances(
    gaussians: Gaussians3D,
    config: SurfelInitConfig | None = None,
    *,
    resolution_floor: torch.Tensor | None = None,
    frames: LocalSurfaceFrames | None = None,
) -> SurfelInitResult:
    """Replace localization covariance and fixed opacity with cover-consistent values.

    ``resolution_floor`` is an optional ``(N,)`` per-component world-space sigma below which a
    primitive would be finer than the images resolve; :func:`beam_contributor_footprints`
    derives it from beam fusion's contributor lineage.  Means, SH, and count never change.
    """
    config = config or SurfelInitConfig()
    points = _validate_points(gaussians.means, config)
    n = points.shape[0]
    frames = frames or estimate_local_surface_frames(gaussians.means, config)
    if frames.spacing.shape[0] != n:
        raise ValueError("supplied frames do not match the gaussian count")

    cover_sigma = config.coverage_sigma_ratio * frames.spacing
    floor = torch.zeros(n, dtype=torch.float64)
    if resolution_floor is not None:
        if resolution_floor.shape != (n,):
            raise ValueError("resolution_floor must have shape (N,)")
        floor_values = resolution_floor.detach().to(torch.float64)
        if not bool(torch.isfinite(floor_values).all()) or bool((floor_values < 0).any()):
            raise ValueError("resolution_floor must be finite and non-negative")
        if config.use_resolution_floor:
            floor = floor_values
    sigma_tangent = torch.maximum(cover_sigma, floor).clamp_min(config.min_sigma_world)
    if config.max_sigma_world is not None:
        sigma_tangent = sigma_tangent.clamp_max(config.max_sigma_world)
    floor_active = floor > cover_sigma

    # Localization uncertainty resolved along the estimated normal raises the measured
    # out-of-plane spread -- the surface may not be claimed thinner than its position is
    # known -- but the surfel flatness cap below is applied last and overrides it.
    covariance = gaussians.covariance().detach().to(torch.float64)
    normal_variance = torch.einsum(
        "ni,nij,nj->n", frames.normals, covariance, frames.normals
    ).clamp_min(0.0)
    sigma_normal = torch.maximum(frames.out_of_plane_sigma, normal_variance.sqrt())
    sigma_normal = sigma_normal.clamp(min=config.min_sigma_world, max=None)
    sigma_normal = torch.minimum(sigma_normal, config.flatness * sigma_tangent)
    sigma_normal = sigma_normal.clamp_min(config.min_sigma_world)

    unoriented = frames.planarity < config.min_planarity
    if config.isotropic:
        unoriented = torch.ones_like(unoriented)
    scales = torch.stack([sigma_tangent, sigma_tangent, sigma_normal], dim=-1)
    scales[unoriented] = sigma_tangent[unoriented, None].expand(-1, 3)
    rotations = _oriented_rotations(frames)
    quats = rotmat_to_quat(rotations)
    identity = quats.new_zeros(n, 4)
    identity[:, 0] = 1.0
    quats = torch.where(unoriented[:, None], identity, quats)

    derived_opacity, overlap = coverage_opacity(sigma_tangent, frames.spacing, config)
    opacity = (
        derived_opacity
        if config.use_coverage_opacity
        else torch.full((n,), float(config.fixed_opacity), dtype=torch.float64)
    )

    dtype = gaussians.means.dtype
    reconciled = Gaussians3D(
        means=gaussians.means.detach().clone(),
        quats=quats.to(dtype),
        log_scales=scales.log().to(dtype),
        opacity=opacity.to(dtype),
        sh=gaussians.sh.detach().clone(),
    )

    def summarize(values: torch.Tensor) -> dict[str, float]:
        return {
            "p10": float(values.quantile(0.10)),
            "median": float(values.median()),
            "p90": float(values.quantile(0.90)),
            "mean": float(values.mean()),
        }

    previous = torch.linalg.eigvalsh(covariance).clamp_min(0.0).sqrt()
    diagnostics: dict[str, object] = {
        "n_components": n,
        "coverage_sigma_ratio": config.coverage_sigma_ratio,
        "hexagonal_cover_ripple": hexagonal_cover_ripple(config.coverage_sigma_ratio),
        "isotropic": config.isotropic,
        "resolution_floor_supplied": resolution_floor is not None,
        "resolution_floor_used": bool(config.use_resolution_floor and resolution_floor is not None),
        "n_resolution_floor_active": int(floor_active.sum()),
        "coverage_opacity": config.use_coverage_opacity,
        "n_unoriented_fallback": int(unoriented.sum()),
        "spacing": summarize(frames.spacing),
        "planarity": summarize(frames.planarity),
        "sigma_tangent": summarize(sigma_tangent),
        "sigma_normal": summarize(sigma_normal),
        "sigma_tangent_over_spacing": summarize(sigma_tangent / frames.spacing),
        "prior_sigma_max": summarize(previous[:, 2]),
        "prior_sigma_max_over_spacing": summarize(previous[:, 2] / frames.spacing),
        "sigma_tangent_over_prior_sigma_max": summarize(
            sigma_tangent / previous[:, 2].clamp_min(1e-30)
        ),
        "overlap_kernel_units": summarize(overlap),
        "opacity": summarize(opacity),
    }
    return SurfelInitResult(
        gaussians=reconciled,
        frames=frames,
        sigma_tangent=sigma_tangent,
        sigma_normal=sigma_normal,
        overlap=overlap,
        resolution_floor_active=floor_active,
        diagnostics=diagnostics,
    )


def beam_contributor_footprints(
    inputs,
    beam,
    *,
    reducer: str = "median",
) -> torch.Tensor:
    """World-space transverse footprint implied by each component's matched 2D Gaussians.

    For contributor ``k`` of component ``c`` the fitted 2D standard deviation in pixels is
    carried to world scale through its own implied camera depth and focal length,
    ``sigma_px * depth / f``.  ``reducer`` selects ``median`` (the typical observation, the
    default resolution floor), ``min``, or ``max`` over a component's contributors.  The
    median follows ``torch.median`` and takes the lower of two middle values, which is
    deterministic and biases an even-length track's floor conservatively downward.
    """
    if reducer not in ("median", "min", "max"):
        raise ValueError("reducer must be one of median, min, max")
    offsets = beam.component_offsets
    n = int(offsets.numel() - 1)
    view_indices = beam.contributor_view_indices
    component_indices = beam.contributor_component_indices
    depths = beam.contributor_depths.to(torch.float64)
    if view_indices.numel() != component_indices.numel() or view_indices.numel() != depths.numel():
        raise ValueError("beam contributor arrays disagree in length")

    sigma_px = [
        (0.5 * observation.effective_variances().to(torch.float64).sum(dim=-1)).sqrt()
        for observation in inputs.observations
    ]
    focal = torch.tensor(
        [math.sqrt(float(camera.fx) * float(camera.fy)) for camera in inputs.cameras],
        dtype=torch.float64,
    )
    if bool((focal <= 0).any()):
        raise ValueError("camera focal lengths must be positive")

    link_sigma = torch.zeros(view_indices.numel(), dtype=torch.float64)
    for view in range(len(sigma_px)):
        rows = (view_indices == view).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        link_sigma[rows] = sigma_px[view][component_indices[rows]] * depths[rows] / focal[view]

    footprints = torch.zeros(n, dtype=torch.float64)
    starts = offsets[:-1].tolist()
    stops = offsets[1:].tolist()
    for component, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        if stop <= start:
            raise ValueError(f"component {component} has no contributors")
        values = link_sigma[start:stop]
        if reducer == "median":
            footprints[component] = values.median()
        elif reducer == "min":
            footprints[component] = values.min()
        else:
            footprints[component] = values.max()
    if not bool(torch.isfinite(footprints).all()):
        raise ValueError("contributor footprints contain non-finite values")
    return footprints


__all__ = [
    "LocalSurfaceFrames",
    "SurfelInitConfig",
    "SurfelInitResult",
    "beam_contributor_footprints",
    "coverage_opacity",
    "effective_cover_sigma",
    "estimate_local_surface_frames",
    "hexagonal_cover_ripple",
    "reconcile_covariances",
]
