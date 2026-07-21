"""Tomographic Gaussian beam fusion: a density-based RGB-free 3D initializer.

The per-view 2D Gaussian fields are *projections* of an unknown 3D radiance density, so 3D
initialization can be posed as reconstruction-from-projections instead of correspondence search.
Gaussians make that tractable in closed form:

1. **Back-projection**: each 2D splat becomes an analytic elongated 3D Gaussian *beam* — its 2D
   covariance lifted to the ray-orthogonal tangent plane at an implied depth, plus a long
   along-ray variance covering the working depth range. No voxel grid.
2. **Pair seeding**: for splats in two views, the closest points between their center rays give
   implied depths in closed form; pairs are gated by transverse ray distance (measured against
   both beams' tangent footprints) and color agreement. The gate is the first-order Gaussian
   product weight, so negligible cross terms are never materialized.
3. **Fusion by covariance intersection**: surviving beams are fused with the CI rule
   ``Lambda = mean_k Lambda_k``, ``m = Lambda^{-1} mean_k(Lambda_k m_k)``. A naive Gaussian
   product (``Lambda = sum_k Lambda_k``) is *rejected by design*: the views are correlated
   observations of one physical splat, and the product double-counts every axis observed by more
   than one view (two orthogonal views of an isotropic blob would fuse to half its variance).
   CI is the standard consistent rule for correlated sources: exact on directions all views
   share, conservative (never overconfident) elsewhere.
4. **Greedy fold-in**: each fused component projects into the remaining views and absorbs the
   nearest splat within a pixel gate and color gate (at most one per view); contributors are the
   correspondence byproduct.
5. **Reduction**: exact contributor-signature dedupe, then per-voxel non-maximum suppression by
   weight — selection, not moment matching, so fused covariances survive.

Every output Gaussian is a positive-weight CI fusion of at least ``min_views`` beams with bounded
SPD covariance and complete contributor lineage; unmatched splats are reported per view for
densification.  Association emerges from density overlap rather than discrete matching — this is
deliberately a different family from both the consensus scoring (which drifts) and the SfM-style
discrete matcher.  CPU-first, deterministic, no RGB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import _center_and_extent, _ray_box, _validate_cpu_inputs
from rtgs.lift.inverse_projection_fiber import _camera_ray_geometry


@dataclass(frozen=True)
class BeamFusionConfig:
    """Controls for beam back-projection, pair seeding, CI fusion, and reduction."""

    min_views: int = 2
    near: float = 0.05
    bounds_scale: float = 0.5
    transverse_gate_sigma: float = 3.0
    max_color_distance: float = 0.35
    color_sigma: float = 0.25
    fold_in_gate_sigma: float = 3.0
    weight_floor: float = 1e-4
    nms_voxel_size: float | None = None
    min_sigma_world: float = 1e-4
    max_sigma_world: float | None = None
    init_opacity: float = 0.1
    source_chunk: int = 1024
    pair_limit: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.min_views, bool) or not isinstance(self.min_views, int):
            raise TypeError("min_views must be an integer")
        if self.min_views < 2:
            raise ValueError("min_views must be at least two")
        if isinstance(self.source_chunk, bool) or not isinstance(self.source_chunk, int):
            raise TypeError("source_chunk must be an integer")
        if self.source_chunk <= 0:
            raise ValueError("source_chunk must be positive")
        if self.pair_limit is not None and (
            isinstance(self.pair_limit, bool)
            or not isinstance(self.pair_limit, int)
            or self.pair_limit <= 0
        ):
            raise ValueError("pair_limit must be a positive integer or None")
        for name in (
            "near",
            "bounds_scale",
            "transverse_gate_sigma",
            "max_color_distance",
            "color_sigma",
            "fold_in_gate_sigma",
            "min_sigma_world",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_floor) or self.weight_floor < 0:
            raise ValueError("weight_floor must be finite and non-negative")
        if self.nms_voxel_size is not None and (
            not math.isfinite(self.nms_voxel_size) or self.nms_voxel_size <= 0
        ):
            raise ValueError("nms_voxel_size must be finite and positive when supplied")
        if self.max_sigma_world is not None and (
            not math.isfinite(self.max_sigma_world) or self.max_sigma_world <= 0
        ):
            raise ValueError("max_sigma_world must be finite and positive when supplied")
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError("init_opacity must be in (0,1)")


@dataclass(frozen=True)
class BeamFusionResult:
    """CI-fused 3D initialization with complete contributor provenance."""

    gaussians: Gaussians3D
    component_offsets: torch.Tensor  # (C+1,), int64 CSR offsets into contributor arrays
    contributor_view_indices: torch.Tensor  # (M,), int64
    contributor_component_indices: torch.Tensor  # (M,), int64
    component_weights: torch.Tensor  # (C,)
    unmatched_per_view: tuple[int, ...]
    diagnostics: dict[str, object]

    @property
    def n_components(self) -> int:
        return int(self.component_offsets.numel() - 1)


@dataclass(frozen=True)
class _BeamView:
    """Per-view float64 geometry shared by seeding, lifting, and fold-in."""

    means2d: torch.Tensor  # (N,2)
    covariances2d: torch.Tensor  # (N,2,2)
    colors: torch.Tensor  # (N,3)
    amplitudes: torch.Tensor  # (N,)
    sigmas_px: torch.Tensor  # (N,)
    origin: torch.Tensor  # (3,) world
    dirs_world: torch.Tensor  # (N,3), unit camera depth
    rays_cam: torch.Tensor  # (N,3), unit camera depth (camera frame)
    basis_cam: torch.Tensor  # (N,3,3), orthonormal camera-frame basis, column 2 = ray unit
    depth_lo: torch.Tensor  # (N,)
    depth_hi: torch.Tensor  # (N,)


def _component_covariances_2d(field: GaussianObservationField) -> torch.Tensor:
    variances = field.effective_variances().to(torch.float64)
    theta = field.rotations.to(torch.float64)
    cos, sin = theta.cos(), theta.sin()
    off = cos * sin * (variances[:, 0] - variances[:, 1])
    row0 = torch.stack(
        [cos.square() * variances[:, 0] + sin.square() * variances[:, 1], off], dim=-1
    )
    row1 = torch.stack(
        [off, sin.square() * variances[:, 0] + cos.square() * variances[:, 1]], dim=-1
    )
    return torch.stack([row0, row1], dim=-2)


def _prepare_views(inputs: ReconstructionInputs, config: BeamFusionConfig) -> list[_BeamView]:
    center, extent = _center_and_extent(inputs, torch.float64)
    half = extent * config.bounds_scale
    lower, upper = center - half, center + half
    views: list[_BeamView] = []
    for field, camera in zip(inputs.observations, inputs.cameras, strict=True):
        means2d = field.native_means(dtype=torch.float64)
        covariances = _component_covariances_2d(field)
        origin, dirs_world = camera.pixel_rays(means2d)
        rays_cam, basis_cam = _camera_ray_geometry(camera, means2d)
        t0, t1 = _ray_box(origin.expand_as(dirs_world), dirs_world, lower, upper)
        t0 = t0.clamp_min(config.near)
        views.append(
            _BeamView(
                means2d=means2d,
                covariances2d=covariances,
                colors=field.colors.to(torch.float64),
                amplitudes=field.amplitudes.to(torch.float64),
                sigmas_px=(0.5 * (covariances[:, 0, 0] + covariances[:, 1, 1])).sqrt(),
                origin=origin.to(torch.float64),
                dirs_world=dirs_world.to(torch.float64),
                rays_cam=rays_cam.to(torch.float64),
                basis_cam=basis_cam.to(torch.float64),
                depth_lo=t0,
                depth_hi=torch.maximum(t1, t0),
            )
        )
    return views


def _beam_precisions(
    view: _BeamView,
    camera,
    indices: torch.Tensor,
    depths: torch.Tensor,
    ray_half_length: torch.Tensor,
) -> torch.Tensor:
    """World-frame beam precision matrices for ``indices`` at implied camera depths.

    The 2D covariance is lifted to the ray-orthogonal tangent plane through the inverse of the
    projected tangent Jacobian (the fiber's exact construction) and inverted; the along-ray
    precision is ``1 / L^2`` for beam half-length ``L``.  The camera-frame quadratic form pulls
    back to world through the world-to-camera rotation.
    """
    rays = view.rays_cam[indices]
    basis = view.basis_cam[indices]
    points_cam = depths[:, None] * rays
    z = points_cam[:, 2].clamp_min(1e-8)
    jacobian = torch.zeros(indices.numel(), 2, 3, dtype=torch.float64)
    jacobian[:, 0, 0] = camera.fx / z
    jacobian[:, 0, 2] = -camera.fx * points_cam[:, 0] / z.square()
    jacobian[:, 1, 1] = camera.fy / z
    jacobian[:, 1, 2] = -camera.fy * points_cam[:, 1] / z.square()
    tangent_projection = jacobian @ basis[:, :, :2]
    inverse_projection = torch.linalg.inv(tangent_projection)
    tangent_cov = (
        inverse_projection @ view.covariances2d[indices] @ inverse_projection.transpose(-1, -2)
    )
    tangent_cov = 0.5 * (tangent_cov + tangent_cov.transpose(-1, -2))
    tangent_precision = torch.linalg.inv(tangent_cov)
    block = torch.zeros(indices.numel(), 3, 3, dtype=torch.float64)
    block[:, :2, :2] = tangent_precision
    block[:, 2, 2] = 1.0 / ray_half_length.square().clamp_min(1e-12)
    precision_cam = basis @ block @ basis.transpose(-1, -2)
    rotation = camera.R.to(torch.float64)
    return rotation.transpose(-1, -2) @ precision_cam @ rotation


def _ray_closest_points(
    o_u: torch.Tensor,
    d1: torch.Tensor,
    o_v: torch.Tensor,
    d2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form per-pair depths ``(s, t)`` and midpoint distance for two ray families."""
    w = o_v - o_u
    a = (d1 * d1).sum(dim=1)
    c = (d2 * d2).sum(dim=1)
    b = d1 @ d2.T
    e = d1 @ w
    f = d2 @ w
    denom = a[:, None] * c[None, :] - b.square()
    parallel = denom.abs() <= 1e-12 * a[:, None] * c[None, :]
    safe = torch.where(parallel, torch.ones_like(denom), denom)
    s = (c[None, :] * e[:, None] - b * f[None, :]) / safe
    t = (b * e[:, None] - a[:, None] * f[None, :]) / safe
    points_u = o_u[None, None, :] + s[..., None] * d1[:, None, :]
    points_v = o_v[None, None, :] + t[..., None] * d2[None, :, :]
    distance = (points_u - points_v).norm(dim=-1)
    distance = torch.where(parallel, torch.full_like(distance, torch.inf), distance)
    return s, t, distance


def _ci_fuse(precisions: torch.Tensor, means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-weight covariance intersection of ``(K,3,3)`` precisions and ``(K,3)`` means."""
    fused_precision = precisions.mean(dim=0)
    information = (precisions @ means[:, :, None]).mean(dim=0)
    fused_mean = torch.linalg.solve(fused_precision, information)[:, 0]
    return fused_precision, fused_mean


def fuse_gaussian_beams(
    inputs: ReconstructionInputs,
    config: BeamFusionConfig | None = None,
) -> BeamFusionResult:
    """Run tomographic beam fusion and return the CI-fused 3D initialization."""
    if config is None:
        config = BeamFusionConfig()
    _validate_cpu_inputs(inputs)
    if config.min_views > inputs.n_views:
        raise ValueError("beam fusion min_views exceeds the number of views")

    views = _prepare_views(inputs, config)
    n_views = inputs.n_views
    _, extent = _center_and_extent(inputs, torch.float64)
    max_sigma = (
        config.max_sigma_world if config.max_sigma_world is not None else 0.5 * float(extent)
    )
    nms_voxel = (
        config.nms_voxel_size
        if config.nms_voxel_size is not None
        else max(1e-6, 0.01 * float(extent))
    )

    pairs = [(u, v) for u in range(n_views) for v in range(u + 1, n_views)]
    positions = torch.stack([camera.position for camera in inputs.cameras]).to(torch.float64)
    pairs.sort(key=lambda pair: (float((positions[pair[0]] - positions[pair[1]]).norm()), pair))
    if config.pair_limit is not None:
        pairs = pairs[: config.pair_limit]

    # Stage 1-2: seed candidate components from gated pair intersections.
    seeds: list[tuple[int, int, int, int, float, float, float]] = []  # u,i,v,j,s,t,gate_weight
    for u, v in pairs:
        view_u, view_v = views[u], views[v]
        n_u, n_v = view_u.means2d.shape[0], view_v.means2d.shape[0]
        if n_u == 0 or n_v == 0:
            continue
        for start in range(0, n_u, config.source_chunk):
            stop = min(start + config.source_chunk, n_u)
            s, t, distance = _ray_closest_points(
                view_u.origin,
                view_u.dirs_world[start:stop],
                view_v.origin,
                view_v.dirs_world,
            )
            # First-order transverse footprints at the implied depths (pixels -> world).
            focal_u = math.sqrt(inputs.cameras[u].fx * inputs.cameras[u].fy)
            focal_v = math.sqrt(inputs.cameras[v].fx * inputs.cameras[v].fy)
            sigma_u = view_u.sigmas_px[start:stop, None] * s.clamp_min(1e-8) / focal_u
            sigma_v = view_v.sigmas_px[None, :] * t.clamp_min(1e-8) / focal_v
            gate = distance.square() / (sigma_u.square() + sigma_v.square()).clamp_min(1e-18)
            color_distance = torch.cdist(view_u.colors[start:stop], view_v.colors)
            valid = (
                (s >= view_u.depth_lo[start:stop, None])
                & (s <= view_u.depth_hi[start:stop, None])
                & (t >= view_v.depth_lo[None, :])
                & (t <= view_v.depth_hi[None, :])
                & (gate <= config.transverse_gate_sigma**2)
                & (color_distance <= config.max_color_distance)
            )
            for local_i, j in zip(*valid.nonzero(as_tuple=True), strict=True):
                i = int(local_i) + start
                weight = float(
                    0.5
                    * (view_u.amplitudes[i] + view_v.amplitudes[int(j)])
                    * math.exp(-0.5 * float(gate[local_i, j]))
                    * math.exp(
                        -0.5 * float(color_distance[local_i, j]) ** 2 / config.color_sigma**2
                    )
                )
                seeds.append((u, i, v, int(j), float(s[local_i, j]), float(t[local_i, j]), weight))
    if not seeds:
        raise ValueError(
            "beam fusion produced no gated pair intersections; loosen the transverse/color "
            "gates or supply views with more overlap"
        )
    seeds.sort(key=lambda seed: (-seed[6], seed[0], seed[1], seed[2], seed[3]))

    # Stage 3-4: CI-fuse each seed pair, then greedily fold in the remaining views.
    components: list[dict] = []
    for u, i, v, j, s, t, gate_weight in seeds:
        contributors = {u: (i, s), v: (j, t)}
        for w in range(n_views):
            if w in contributors:
                continue
            view_w = views[w]
            # Provisional midpoint from the seed pair locates the projection into view w.
            provisional = 0.5 * (
                views[u].origin
                + s * views[u].dirs_world[i]
                + views[v].origin
                + t * views[v].dirs_world[j]
            )
            uv, depth = inputs.cameras[w].project(provisional[None, :])
            if float(depth[0]) <= config.near:
                continue
            offsets = (view_w.means2d - uv[0]).norm(dim=1)
            candidate = int(offsets.argmin())
            pixel_gate = config.fold_in_gate_sigma * float(view_w.sigmas_px[candidate])
            color_gap = float((view_w.colors[candidate] - views[u].colors[i]).norm())
            if float(offsets[candidate]) <= pixel_gate and color_gap <= config.max_color_distance:
                contributors[w] = (candidate, float(depth[0]))
        if len(contributors) < config.min_views:
            continue
        member_views = sorted(contributors)
        precisions = []
        means = []
        for view_index in member_views:
            component, depth = contributors[view_index]
            view = views[view_index]
            index_tensor = torch.tensor([component], dtype=torch.long)
            depth_tensor = torch.tensor([depth], dtype=torch.float64)
            half_length = (
                0.5
                * (view.depth_hi[component] - view.depth_lo[component]).clamp_min(1e-6)
                * view.dirs_world[component].norm()
            )
            precisions.append(
                _beam_precisions(
                    view,
                    inputs.cameras[view_index],
                    index_tensor,
                    depth_tensor,
                    half_length[None],
                )[0]
            )
            means.append(view.origin + depth * view.dirs_world[component])
        fused_precision, fused_mean = _ci_fuse(torch.stack(precisions), torch.stack(means))
        signature = tuple((view_index, contributors[view_index][0]) for view_index in member_views)
        components.append(
            {
                "signature": signature,
                "weight": gate_weight * len(member_views),
                "precision": fused_precision,
                "mean": fused_mean,
            }
        )
    if not components:
        raise ValueError(
            "beam fusion rejected every seed at the min_views fold-in gate; loosen the gates "
            "or lower min_views"
        )

    # Stage 5: exact signature dedupe, then per-voxel non-maximum suppression by weight.
    by_signature: dict[tuple, dict] = {}
    for component in components:
        existing = by_signature.get(component["signature"])
        if existing is None or component["weight"] > existing["weight"]:
            by_signature[component["signature"]] = component
    unique = sorted(by_signature.values(), key=lambda item: (-item["weight"], item["signature"]))
    kept: list[dict] = []
    occupied: dict[tuple[int, int, int], float] = {}
    for component in unique:
        if component["weight"] < config.weight_floor:
            continue
        key = tuple(int(x) for x in torch.floor(component["mean"] / nms_voxel).tolist())
        if key in occupied:
            continue
        occupied[key] = component["weight"]
        kept.append(component)
    if not kept:
        raise ValueError("beam fusion NMS removed every component; lower weight_floor")

    means = torch.stack([component["mean"] for component in kept])
    covariances = torch.linalg.inv(torch.stack([component["precision"] for component in kept]))
    covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    eigenvalues = eigenvalues.clamp(min=config.min_sigma_world**2, max=max_sigma**2)
    covariances = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)

    colors = []
    offsets = [0]
    contributor_views: list[int] = []
    contributor_components: list[int] = []
    matched: dict[int, set[int]] = {view: set() for view in range(n_views)}
    for component in kept:
        color_sum = torch.zeros(3, dtype=torch.float64)
        amplitude_sum = 0.0
        for view_index, splat_index in component["signature"]:
            amplitude = float(views[view_index].amplitudes[splat_index])
            color_sum += amplitude * views[view_index].colors[splat_index]
            amplitude_sum += amplitude
            contributor_views.append(view_index)
            contributor_components.append(splat_index)
            matched[view_index].add(splat_index)
        offsets.append(len(contributor_views))
        colors.append(color_sum / max(amplitude_sum, 1e-12))
    unmatched = tuple(
        int(inputs.observations[view].n - len(matched[view])) for view in range(n_views)
    )

    dtype = inputs.observations[0].dtype
    gaussians = Gaussians3D.from_means_covs(
        means=means.to(dtype),
        covs=covariances.to(dtype),
        colors=torch.stack(colors).clamp(0.0, 1.0).to(dtype),
        opacity=torch.full((len(kept),), config.init_opacity, dtype=dtype),
        sh_degree=0,
    )
    lengths = torch.tensor([len(component["signature"]) for component in kept], dtype=torch.long)
    diagnostics: dict[str, object] = {
        "n_views": n_views,
        "n_view_pairs": len(pairs),
        "n_seeds": len(seeds),
        "n_components_prefilter": len(components),
        "n_components": len(kept),
        "component_view_histogram": {
            int(length): int((lengths == length).sum()) for length in lengths.unique(sorted=True)
        },
        "nms_voxel_size": nms_voxel,
        "max_sigma_world": max_sigma,
        "unmatched_per_view": list(unmatched),
    }
    return BeamFusionResult(
        gaussians=gaussians,
        component_offsets=torch.tensor(offsets, dtype=torch.long),
        contributor_view_indices=torch.tensor(contributor_views, dtype=torch.long),
        contributor_component_indices=torch.tensor(contributor_components, dtype=torch.long),
        component_weights=torch.tensor(
            [component["weight"] for component in kept], dtype=torch.float64
        ),
        unmatched_per_view=unmatched,
        diagnostics=diagnostics,
    )


__all__ = [
    "BeamFusionConfig",
    "BeamFusionResult",
    "fuse_gaussian_beams",
]
