"""Calibrated structure-from-splats: an SfM-style RGB-free 3D initialization from 2D Gaussians.

Classical incremental SfM runs feature detection, descriptor matching with epipolar RANSAC,
track building, triangulation, and bundle adjustment.  Here the cameras are already calibrated, so
pose estimation disappears and the epipolar geometry is exact.  What remains is the structure half,
re-derived for 2D Gaussian primitives instead of keypoints:

1. **Feature** = one fitted 2D Gaussian (center, full 2x2 covariance, color, amplitude).
2. **Pairwise matching**: for splats ``i`` (view ``u``) and ``j`` (view ``v``), the closest points
   between their center rays give implied depths ``(s, t)`` in closed form; reprojecting ray ``i``'s
   closest point into ``v`` measures the epipolar residual. Candidates are gated by that residual
   (normalized by the candidate's own pixel sigma), color distance, and world-size consistency
   ``sigma_px * z / f`` (the same physical patch must imply the same metric size from both views).
   Mutual-best selection plus a SIFT-style ratio test keeps only distinctive matches.
3. **Tracks**: union-find over accepted pair matches, processed in ascending cost with a strict
   one-splat-per-view invariant (a conflicting edge is skipped, never merged).
4. **Center triangulation**: batched calibrated DLT
   (:func:`rtgs.lift.gaussian_correspondence.triangulate_centers_dlt`) with cheirality,
   per-view reprojection, and triangulation-angle gates — SfM's geometric verification.
5. **Covariance triangulation**: each view constrains the 3D covariance linearly,
   ``vech(Sigma2D_v) = A_v vech(Sigma3D)`` with ``A_v`` built from the projection Jacobian
   ``J_v = J_cam(X) R_v``; three equations per view, six unknowns, so two or more views determine
   ``Sigma3D`` by least squares. The solution is projected to SPD with bounded eigenvalues.

The output is deliberately well-defined: every returned Gaussian is backed by an epipolar-verified
multi-view track with bounded reprojection error and triangulation angle; covariances are the
projection-consistent least-squares SPD solutions; colors are amplitude-weighted track means; and
the mapping from every input splat to its track (or to "unmatched") is returned. Unmatched splats
are expected — 2D Gaussian fits are a per-view segmentation, not repeatable detections — and are
left to downstream densification by design.  CPU-first and deterministic; no RGB is read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import _center_and_extent, _ray_box, _validate_cpu_inputs
from rtgs.lift.gaussian_correspondence import triangulate_centers_dlt

_VECH_INDEX = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


@dataclass(frozen=True)
class SplatSfMConfig:
    """Controls for calibrated splat matching, track building, and triangulation."""

    min_views: int = 2
    near: float = 0.05
    bounds_scale: float = 0.5
    max_epipolar_sigma: float = 3.0
    max_color_distance: float = 0.35
    max_size_log_ratio: float = 1.0
    cost_epipolar_weight: float = 1.0
    cost_color_weight: float = 1.0
    cost_size_weight: float = 0.5
    ratio_test: float = 0.8
    max_reprojection_px: float = 3.0
    min_triangulation_angle_deg: float = 2.0
    min_sigma_world: float = 1e-4
    max_sigma_world: float | None = None
    init_opacity: float = 0.1
    source_chunk: int = 1024
    seed_pair_limit: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.min_views, bool) or not isinstance(self.min_views, int):
            raise TypeError("min_views must be an integer")
        if self.min_views < 2:
            raise ValueError("min_views must be at least two")
        if isinstance(self.source_chunk, bool) or not isinstance(self.source_chunk, int):
            raise TypeError("source_chunk must be an integer")
        if self.source_chunk <= 0:
            raise ValueError("source_chunk must be positive")
        if self.seed_pair_limit is not None and (
            isinstance(self.seed_pair_limit, bool)
            or not isinstance(self.seed_pair_limit, int)
            or self.seed_pair_limit <= 0
        ):
            raise ValueError("seed_pair_limit must be a positive integer or None")
        positive = (
            "near",
            "bounds_scale",
            "max_epipolar_sigma",
            "max_color_distance",
            "max_size_log_ratio",
            "max_reprojection_px",
            "min_sigma_world",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("cost_epipolar_weight", "cost_color_weight", "cost_size_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < self.ratio_test <= 1.0:
            raise ValueError("ratio_test must be in (0,1]")
        if not 0.0 <= self.min_triangulation_angle_deg < 90.0:
            raise ValueError("min_triangulation_angle_deg must be in [0,90)")
        if self.max_sigma_world is not None and (
            not math.isfinite(self.max_sigma_world) or self.max_sigma_world <= 0
        ):
            raise ValueError("max_sigma_world must be finite and positive when supplied")
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError("init_opacity must be in (0,1)")


@dataclass(frozen=True)
class SplatSfMResult:
    """Track-backed 3D initialization with complete per-splat provenance."""

    gaussians: Gaussians3D
    track_offsets: torch.Tensor  # (T+1,), int64 CSR offsets into member arrays
    member_view_indices: torch.Tensor  # (M,), int64
    member_component_indices: torch.Tensor  # (M,), int64
    track_reprojection_error: torch.Tensor  # (T,), max member reprojection in pixels
    track_triangulation_angle_deg: torch.Tensor  # (T,)
    track_covariance_residual: torch.Tensor  # (T,), relative linear-fit residual
    unmatched_per_view: tuple[int, ...]
    diagnostics: dict[str, object]

    @property
    def n_tracks(self) -> int:
        return int(self.track_offsets.numel() - 1)


@dataclass(frozen=True)
class _ViewSplats:
    """Per-view float64 splat geometry shared by matching and triangulation."""

    means2d: torch.Tensor  # (N,2)
    covariances2d: torch.Tensor  # (N,2,2)
    colors: torch.Tensor  # (N,3)
    amplitudes: torch.Tensor  # (N,)
    sigmas_px: torch.Tensor  # (N,), RMS pixel sigma
    ray_origin: torch.Tensor  # (3,)
    ray_dirs: torch.Tensor  # (N,3), unit camera depth
    depth_lo: torch.Tensor  # (N,)
    depth_hi: torch.Tensor  # (N,)
    focal: float


def _component_covariances_2d(field: GaussianObservationField) -> torch.Tensor:
    variances = field.effective_variances().to(torch.float64)
    theta = field.rotations.to(torch.float64)
    cos = theta.cos()
    sin = theta.sin()
    row0 = torch.stack(
        [
            cos.square() * variances[:, 0] + sin.square() * variances[:, 1],
            cos * sin * (variances[:, 0] - variances[:, 1]),
        ],
        dim=-1,
    )
    row1 = torch.stack(
        [
            cos * sin * (variances[:, 0] - variances[:, 1]),
            sin.square() * variances[:, 0] + cos.square() * variances[:, 1],
        ],
        dim=-1,
    )
    return torch.stack([row0, row1], dim=-2)


def _prepare_views(
    inputs: ReconstructionInputs,
    config: SplatSfMConfig,
) -> list[_ViewSplats]:
    center, extent = _center_and_extent(inputs, torch.float64)
    half = extent * config.bounds_scale
    lower = center - half
    upper = center + half
    views: list[_ViewSplats] = []
    for field, camera in zip(inputs.observations, inputs.cameras, strict=True):
        means2d = field.native_means(dtype=torch.float64)
        covariances = _component_covariances_2d(field)
        origin, directions = camera.pixel_rays(means2d)
        t0, t1 = _ray_box(origin.expand_as(directions), directions, lower, upper)
        t0 = t0.clamp_min(config.near)
        views.append(
            _ViewSplats(
                means2d=means2d,
                covariances2d=covariances,
                colors=field.colors.to(torch.float64),
                amplitudes=field.amplitudes.to(torch.float64),
                sigmas_px=(0.5 * (covariances[:, 0, 0] + covariances[:, 1, 1])).sqrt(),
                ray_origin=origin.to(torch.float64),
                ray_dirs=directions.to(torch.float64),
                depth_lo=t0,
                depth_hi=torch.maximum(t1, t0),
                focal=math.sqrt(camera.fx * camera.fy),
            )
        )
    return views


def _pair_order(inputs: ReconstructionInputs, limit: int | None) -> list[tuple[int, int]]:
    positions = torch.stack([camera.position for camera in inputs.cameras]).to(torch.float64)
    pairs = [(u, v) for u in range(inputs.n_views) for v in range(u + 1, inputs.n_views)]
    distances = [float((positions[u] - positions[v]).norm()) for u, v in pairs]
    order = sorted(range(len(pairs)), key=lambda k: (distances[k], pairs[k]))
    selected = [pairs[k] for k in order]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _match_pair(
    u: int,
    v: int,
    view_u: _ViewSplats,
    view_v: _ViewSplats,
    camera_v,
    config: SplatSfMConfig,
) -> list[tuple[float, int, int, int, int]]:
    """Score all epipolar-consistent (i, j) candidates and keep mutual, distinctive matches."""
    n_u = view_u.means2d.shape[0]
    n_v = view_v.means2d.shape[0]
    if n_u == 0 or n_v == 0:
        return []
    o_u = view_u.ray_origin
    o_v = view_v.ray_origin
    w = o_v - o_u
    d2 = view_v.ray_dirs
    c = (d2 * d2).sum(dim=1)
    f_j = d2 @ w

    best_cost = torch.full((n_u,), torch.inf, dtype=torch.float64)
    second_cost = torch.full((n_u,), torch.inf, dtype=torch.float64)
    best_j = torch.full((n_u,), -1, dtype=torch.long)
    reverse_best_cost = torch.full((n_v,), torch.inf, dtype=torch.float64)
    reverse_best_i = torch.full((n_v,), -1, dtype=torch.long)

    for start in range(0, n_u, config.source_chunk):
        stop = min(start + config.source_chunk, n_u)
        d1 = view_u.ray_dirs[start:stop]
        a = (d1 * d1).sum(dim=1)
        e = d1 @ w
        b = d1 @ d2.T
        denom = a[:, None] * c[None, :] - b.square()
        parallel = denom.abs() <= 1e-12 * a[:, None] * c[None, :]
        safe = torch.where(parallel, torch.ones_like(denom), denom)
        s = (c[None, :] * e[:, None] - b * f_j[None, :]) / safe
        t = (b * e[:, None] - a[:, None] * f_j[None, :]) / safe

        points_u = o_u[None, None, :] + s[..., None] * d1[:, None, :]
        uv, depth_v = camera_v.project(points_u.reshape(-1, 3))
        residual = (uv.reshape(stop - start, n_v, 2) - view_v.means2d[None, :, :]).norm(dim=-1)
        epipolar = residual / view_v.sigmas_px[None, :]
        color_distance = torch.cdist(view_u.colors[start:stop], view_v.colors)
        size_u = view_u.sigmas_px[start:stop, None] * s / view_u.focal
        size_v = view_v.sigmas_px[None, :] * t / view_v.focal
        size_ratio = (size_u.clamp_min(1e-12).log() - size_v.clamp_min(1e-12).log()).abs()

        valid = (
            ~parallel
            & (s >= view_u.depth_lo[start:stop, None])
            & (s <= view_u.depth_hi[start:stop, None])
            & (t >= config.near)
            & (depth_v.reshape(stop - start, n_v) > config.near)
            & (epipolar <= config.max_epipolar_sigma)
            & (color_distance <= config.max_color_distance)
            & (size_ratio <= config.max_size_log_ratio)
        )
        cost = (
            config.cost_epipolar_weight * epipolar
            + config.cost_color_weight * color_distance
            + config.cost_size_weight * size_ratio
        )
        cost = torch.where(valid, cost, torch.full_like(cost, torch.inf))

        top = torch.topk(cost, k=min(2, n_v), dim=1, largest=False)
        best_cost[start:stop] = top.values[:, 0]
        best_j[start:stop] = top.indices[:, 0]
        if n_v > 1:
            second_cost[start:stop] = top.values[:, 1]

        column_best = cost.min(dim=0)
        improved = column_best.values < reverse_best_cost
        reverse_best_cost = torch.where(improved, column_best.values, reverse_best_cost)
        reverse_best_i = torch.where(improved, column_best.indices + start, reverse_best_i)

    matches: list[tuple[float, int, int, int, int]] = []
    for i in range(n_u):
        cost_value = float(best_cost[i])
        if not math.isfinite(cost_value):
            continue
        j = int(best_j[i])
        if int(reverse_best_i[j]) != i:
            continue
        second = float(second_cost[i])
        if math.isfinite(second) and cost_value > config.ratio_test * second:
            continue
        matches.append((cost_value, u, i, v, j))
    return matches


class _TrackUnionFind:
    """Union-find whose components maintain the strict one-splat-per-view invariant."""

    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}
        self.members: dict[tuple[int, int], dict[int, int]] = {}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        if node not in self.parent:
            self.parent[node] = node
            self.members[node] = {node[0]: node[1]}
            return node
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, first: tuple[int, int], second: tuple[int, int]) -> bool:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return False
        members_a = self.members[root_a]
        members_b = self.members[root_b]
        if len(members_a) < len(members_b):
            root_a, root_b = root_b, root_a
            members_a, members_b = members_b, members_a
        if any(view in members_a for view in members_b):
            return False  # would place two splats from one view in a single track
        members_a.update(members_b)
        self.parent[root_b] = root_a
        del self.members[root_b]
        return True

    def components(self) -> list[list[tuple[int, int]]]:
        grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for node in self.parent:
            grouped.setdefault(self.find(node), []).append(node)
        tracks = [sorted(nodes) for nodes in grouped.values()]
        tracks.sort()
        return tracks


def _triangulate_covariances(
    points_world: torch.Tensor,
    member_mask: torch.Tensor,
    member_cov2d: torch.Tensor,
    cameras,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``vech(Sigma2D_v) = A_v vech(Sigma3D)`` by masked batched least squares.

    ``points_world`` is ``(T,3)``, ``member_mask`` is ``(T,V)`` and ``member_cov2d`` is
    ``(T,V,2,2)``. Returns the raw symmetric solutions ``(T,3,3)`` (not yet SPD-projected) and the
    relative linear residual per track.
    """
    count, n_views = member_mask.shape
    device = points_world.device
    jacobians = torch.zeros(count, n_views, 2, 3, dtype=torch.float64, device=device)
    for view_index, camera in enumerate(cameras):
        cam_points = camera.world_to_cam(points_world.to(torch.float64))
        z = cam_points[:, 2].clamp_min(1e-8)
        j_cam = torch.zeros(count, 2, 3, dtype=torch.float64, device=device)
        j_cam[:, 0, 0] = camera.fx / z
        j_cam[:, 0, 2] = -camera.fx * cam_points[:, 0] / z.square()
        j_cam[:, 1, 1] = camera.fy / z
        j_cam[:, 1, 2] = -camera.fy * cam_points[:, 1] / z.square()
        jacobians[:, view_index] = j_cam @ camera.R.to(torch.float64)

    design = torch.zeros(count, n_views, 3, 6, dtype=torch.float64, device=device)
    for column, (p, q) in enumerate(_VECH_INDEX):
        basis = torch.zeros(3, 3, dtype=torch.float64, device=device)
        basis[p, q] = 1.0
        basis[q, p] = 1.0
        projected = jacobians @ basis @ jacobians.transpose(-1, -2)
        design[:, :, 0, column] = projected[:, :, 0, 0]
        design[:, :, 1, column] = projected[:, :, 0, 1]
        design[:, :, 2, column] = projected[:, :, 1, 1]

    targets = torch.stack(
        [
            member_cov2d[:, :, 0, 0],
            member_cov2d[:, :, 0, 1],
            member_cov2d[:, :, 1, 1],
        ],
        dim=-1,
    )
    row_mask = member_mask[:, :, None].to(torch.float64)
    design = (design * row_mask[..., None]).reshape(count, n_views * 3, 6)
    targets = (targets * row_mask).reshape(count, n_views * 3)

    solution = torch.linalg.pinv(design) @ targets[..., None]
    solution = solution[..., 0]
    fitted = (design @ solution[..., None])[..., 0]
    residual = (fitted - targets).norm(dim=1) / targets.norm(dim=1).clamp_min(1e-12)

    covariances = torch.zeros(count, 3, 3, dtype=torch.float64, device=device)
    for column, (p, q) in enumerate(_VECH_INDEX):
        covariances[:, p, q] = solution[:, column]
        covariances[:, q, p] = solution[:, column]
    return covariances, residual


def _spd_project(
    covariances: torch.Tensor,
    min_sigma: float,
    max_sigma: float,
) -> torch.Tensor:
    symmetric = 0.5 * (covariances + covariances.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.clamp(min=min_sigma**2, max=max_sigma**2)
    return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)


def structure_from_splats(
    inputs: ReconstructionInputs,
    config: SplatSfMConfig | None = None,
) -> SplatSfMResult:
    """Run calibrated splat-SfM and return the track-backed 3D initialization."""
    if config is None:
        config = SplatSfMConfig()
    _validate_cpu_inputs(inputs)
    if config.min_views > inputs.n_views:
        raise ValueError("splat SfM min_views exceeds the number of views")

    views = _prepare_views(inputs, config)
    pairs = _pair_order(inputs, config.seed_pair_limit)
    edges: list[tuple[float, int, int, int, int]] = []
    for u, v in pairs:
        edges.extend(_match_pair(u, v, views[u], views[v], inputs.cameras[v], config))
    edges.sort(key=lambda edge: (edge[0], edge[1], edge[2], edge[3], edge[4]))

    union = _TrackUnionFind()
    accepted_edges = 0
    for _cost, u, i, v, j in edges:
        if union.union((u, i), (v, j)):
            accepted_edges += 1
    tracks = [members for members in union.components() if len(members) >= config.min_views]
    if not tracks:
        raise ValueError(
            "splat SfM found no multi-view tracks; loosen the matching gates or supply "
            "views with more overlap"
        )

    n_views = inputs.n_views
    count = len(tracks)
    image_points = torch.zeros(count, n_views, 2, dtype=torch.float64)
    member_mask = torch.zeros(count, n_views, dtype=torch.bool)
    member_cov2d = torch.zeros(count, n_views, 2, 2, dtype=torch.float64)
    member_color = torch.zeros(count, n_views, 3, dtype=torch.float64)
    member_weight = torch.zeros(count, n_views, dtype=torch.float64)
    for track_index, members in enumerate(tracks):
        for view_index, component in members:
            image_points[track_index, view_index] = views[view_index].means2d[component]
            member_mask[track_index, view_index] = True
            member_cov2d[track_index, view_index] = views[view_index].covariances2d[component]
            member_color[track_index, view_index] = views[view_index].colors[component]
            member_weight[track_index, view_index] = views[view_index].amplitudes[component]

    triangulation = triangulate_centers_dlt(
        inputs.cameras,
        image_points,
        observation_mask=member_mask,
        require_in_image=False,
        min_views=config.min_views,
        min_depth=config.near,
        max_reprojection_error=config.max_reprojection_px,
    )

    centers = torch.stack([camera.position for camera in inputs.cameras]).to(torch.float64)
    directions = torch.nn.functional.normalize(
        triangulation.points_world[:, None, :] - centers[None, :, :], dim=-1
    )
    pair_cos = torch.ones(count, dtype=torch.float64)
    for first in range(n_views):
        for second in range(first + 1, n_views):
            both = member_mask[:, first] & member_mask[:, second]
            if not bool(both.any()):
                continue
            cosine = (directions[:, first] * directions[:, second]).sum(dim=-1)
            pair_cos = torch.where(both, torch.minimum(pair_cos, cosine), pair_cos)
    angles_deg = torch.rad2deg(torch.arccos(pair_cos.clamp(-1.0, 1.0)))

    keep = triangulation.valid & (angles_deg >= config.min_triangulation_angle_deg)
    if not bool(keep.any()):
        raise ValueError(
            "splat SfM rejected every track at the triangulation gates; check camera "
            "baselines or loosen max_reprojection_px/min_triangulation_angle_deg"
        )
    kept_indices = keep.nonzero(as_tuple=True)[0]
    tracks = [tracks[int(index)] for index in kept_indices]
    points_world = triangulation.points_world[kept_indices]
    member_mask = member_mask[kept_indices]
    member_cov2d = member_cov2d[kept_indices]
    member_color = member_color[kept_indices]
    member_weight = member_weight[kept_indices]
    reprojection = triangulation.max_observed_reprojection_error[kept_indices]
    angles_deg = angles_deg[kept_indices]

    covariances, covariance_residual = _triangulate_covariances(
        points_world, member_mask, member_cov2d, inputs.cameras
    )
    _, extent = _center_and_extent(inputs, torch.float64)
    max_sigma = (
        config.max_sigma_world if config.max_sigma_world is not None else 0.5 * float(extent)
    )
    covariances = _spd_project(covariances, config.min_sigma_world, max_sigma)

    weight = member_weight * member_mask.to(torch.float64)
    weight_sum = weight.sum(dim=1).clamp_min(1e-12)
    colors = (member_color * weight[..., None]).sum(dim=1) / weight_sum[:, None]

    dtype = inputs.observations[0].dtype
    gaussians = Gaussians3D.from_means_covs(
        means=points_world.to(dtype),
        covs=covariances.to(dtype),
        colors=colors.clamp(0.0, 1.0).to(dtype),
        opacity=torch.full((len(tracks),), config.init_opacity, dtype=dtype),
        sh_degree=0,
    )

    offsets = [0]
    member_views: list[int] = []
    member_components: list[int] = []
    matched: dict[int, set[int]] = {view: set() for view in range(n_views)}
    for members in tracks:
        for view_index, component in members:
            member_views.append(view_index)
            member_components.append(component)
            matched[view_index].add(component)
        offsets.append(len(member_views))
    unmatched = tuple(
        int(inputs.observations[view].n - len(matched[view])) for view in range(n_views)
    )

    track_lengths = torch.tensor([len(members) for members in tracks], dtype=torch.long)
    diagnostics: dict[str, object] = {
        "n_views": n_views,
        "n_view_pairs": len(pairs),
        "n_pair_matches": len(edges),
        "n_union_edges": accepted_edges,
        "n_tracks_prefilter": count,
        "n_tracks": len(tracks),
        "track_length_histogram": {
            int(length): int((track_lengths == length).sum())
            for length in track_lengths.unique(sorted=True)
        },
        "mean_reprojection_px": float(reprojection.mean()),
        "max_reprojection_px": float(reprojection.max()),
        "mean_triangulation_angle_deg": float(angles_deg.mean()),
        "mean_covariance_residual": float(covariance_residual.mean()),
        "max_sigma_world": max_sigma,
        "unmatched_per_view": list(unmatched),
    }
    return SplatSfMResult(
        gaussians=gaussians,
        track_offsets=torch.tensor(offsets, dtype=torch.long),
        member_view_indices=torch.tensor(member_views, dtype=torch.long),
        member_component_indices=torch.tensor(member_components, dtype=torch.long),
        track_reprojection_error=reprojection.to(dtype),
        track_triangulation_angle_deg=angles_deg.to(dtype),
        track_covariance_residual=covariance_residual.to(dtype),
        unmatched_per_view=unmatched,
        diagnostics=diagnostics,
    )


__all__ = [
    "SplatSfMConfig",
    "SplatSfMResult",
    "structure_from_splats",
]
