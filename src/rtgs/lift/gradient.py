"""Variant A: lift per-image 2D gaussians, then optimize them along their rays.

Implements the staged idea: each fitted 2D gaussian is lifted to a 3D gaussian whose
mean is pinned to the pixel's viewing ray (lateral position frozen — the "on the ray"
constraint) and whose along-ray extent starts at a *footprint-matched* thickness (the
"epsilon along the missing dimension", scaled by ``ray_thickness``). Then a multi-view
photometric optimization moves each gaussian to find:

  * position along the ray  (depth ``t``, always),
  * rotation                (``optimize_rotation``),
  * scale                   (``optimize_scale``),

with **color / SH / opacity frozen** (trusted from the 2D fit). Finally redundant
gaussians from overlapping views are fused by moment-matched merging (``merge``) so the
downstream 3DGS stage does not start with N-views of duplicated geometry.

Why not a literal epsilon: the covariance is stored as log-scales and the rotation is
recovered by eigendecomposition, so a razor-thin disk (a) underflows / goes singular and
(b) has an ill-defined rotation and near-zero gradient to rotate into the true surface.
``ray_thickness`` is therefore clamped to a small positive fraction of the footprint. See
docs/RESEARCH.md §"Missing-dimension covariance" and docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import ssim
from rtgs.data.scene import SceneData
from rtgs.lift.base import bilinear_sample, eigvals_2x2, lift_view_at_depth
from rtgs.lift.depth import AlignedDepthPrior
from rtgs.lift.merge import merge_by_voxel
from rtgs.lift.surface import (
    OrientedPointTargets,
    local_plane_loss,
    shortest_axis_normal_loss,
    validate_oriented_point_targets,
)
from rtgs.render.base import get_rasterizer

# Floor on ray_thickness: a disk thinner than this (relative to its lateral footprint)
# makes the eigendecomposition rotation ill-conditioned and the covariance near-singular.
_MIN_THICKNESS = 0.05
_DEPTH_ANCHOR_MODES = {
    "legacy",
    "normalized",
    "valid_uniform",
    "confidence",
    "confidence_shuffled",
    "thresholded",
}
_PHOTOMETRIC_SUPERVISION_MODES = {
    "all",
    "leave_one_source_out",
    "matched_nonself_dropout",
}


def _build_photometric_exclusions(
    source_view_ids: torch.Tensor,
    target_views: list[int],
    mode: str,
    seed: int,
) -> tuple[dict[int, torch.Tensor], list[dict[str, object]]]:
    """Build compact frozen exclusions without consuming the optimizer RNG."""
    if mode not in _PHOTOMETRIC_SUPERVISION_MODES:
        raise ValueError(
            "photometric_supervision_mode must be one of "
            f"{sorted(_PHOTOMETRIC_SUPERVISION_MODES)}, got {mode!r}"
        )
    if source_view_ids.ndim != 1 or source_view_ids.numel() == 0:
        raise ValueError("source_view_ids must be a non-empty 1D tensor")
    if source_view_ids.dtype != torch.long or bool((source_view_ids < 0).any()):
        raise ValueError("source_view_ids must contain non-negative int64 indices")
    if not target_views or len(set(target_views)) != len(target_views):
        raise ValueError("target_views must contain unique view indices")
    if any(view < 0 for view in target_views):
        raise ValueError("target view indices must be non-negative")

    n_view_bins = max(int(source_view_ids.max()) + 1, max(target_views) + 1)
    dropout_gen = torch.Generator(device=source_view_ids.device).manual_seed(seed + 2_000_003)
    matched_exclusions: dict[int, torch.Tensor] = {}
    if mode == "matched_nonself_dropout":
        source_views = set(torch.unique(source_view_ids).tolist())
        if source_views != set(target_views):
            raise ValueError(
                "balanced matched non-self dropout requires target_views to equal the retained "
                "source-view set"
            )
        grouped_donors = []
        target_slots = []
        max_source_count = 0
        for target_view in target_views:
            source_indices = torch.nonzero(source_view_ids == target_view, as_tuple=False).squeeze(
                -1
            )
            source_count = int(source_indices.numel())
            max_source_count = max(max_source_count, source_count)
            order = torch.randperm(
                source_count, generator=dropout_gen, device=source_indices.device
            )
            grouped_donors.append(source_indices[order])
            target_slots.append(
                torch.full(
                    (source_count,), target_view, dtype=torch.long, device=source_view_ids.device
                )
            )
        if max_source_count > source_view_ids.numel() - max_source_count:
            raise ValueError(
                "balanced matched non-self dropout requires the largest source group to contain "
                "at most half of retained primitives"
            )
        donors = torch.roll(torch.cat(grouped_donors), shifts=-max_source_count)
        slots = torch.cat(target_slots)
        if bool((source_view_ids[donors] == slots).any()):
            raise AssertionError("matched non-self assignment contains an own-source donor")
        exposure = torch.bincount(donors, minlength=source_view_ids.numel())
        if not bool((exposure == 1).all()):
            raise AssertionError("matched non-self assignment does not use every primitive once")
        matched_exclusions = {
            target_view: donors[slots == target_view] for target_view in target_views
        }
    exclusions: dict[int, torch.Tensor] = {}
    diagnostics = []
    for target_view in target_views:
        own = source_view_ids == target_view
        own_count = int(own.sum())
        excluded = source_view_ids.new_empty(0)
        if mode == "leave_one_source_out":
            excluded = torch.nonzero(own, as_tuple=False).squeeze(-1)
        elif mode == "matched_nonself_dropout":
            excluded = matched_exclusions[target_view]
        if excluded.numel() >= source_view_ids.numel():
            raise ValueError(
                f"photometric supervision mode {mode!r} leaves target view {target_view} "
                "with no rendered primitives"
            )
        excluded_source_counts = torch.bincount(source_view_ids[excluded], minlength=n_view_bins)
        diagnostics.append(
            {
                "target_view": target_view,
                "total_count": int(source_view_ids.numel()),
                "own_source_count": own_count,
                "rendered_count": int(source_view_ids.numel() - excluded.numel()),
                "excluded_count": int(excluded.numel()),
                "excluded_own_count": int(own[excluded].sum()),
                "excluded_source_counts": tuple(excluded_source_counts.tolist()),
                "global_exclusion_exposure_min": (1 if mode == "matched_nonself_dropout" else None),
                "global_exclusion_exposure_max": (1 if mode == "matched_nonself_dropout" else None),
            }
        )
        exclusions[target_view] = excluded
    return exclusions, diagnostics


def _build_photometric_keep_masks(
    source_view_ids: torch.Tensor,
    target_views: list[int],
    mode: str,
    seed: int,
) -> tuple[dict[int, torch.Tensor], list[dict[str, object]]]:
    """Materialize keep masks for exact diagnostics and small controlled tests."""
    exclusions, diagnostics = _build_photometric_exclusions(
        source_view_ids, target_views, mode, seed
    )
    masks = {}
    for target_view, excluded in exclusions.items():
        keep = torch.ones_like(source_view_ids, dtype=torch.bool)
        keep[excluded] = False
        masks[target_view] = keep
    return masks, diagnostics


def _resolve_anchor_weights(
    valid_prior: torch.Tensor,
    sampled_confidence: torch.Tensor,
    mode: str,
    shuffle_generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Resolve per-view anchor weights after confidence sampling and ray filtering."""
    confidence_weight = torch.where(
        valid_prior, sampled_confidence, torch.zeros_like(sampled_confidence)
    )
    weights = valid_prior.to(sampled_confidence) if mode == "valid_uniform" else confidence_weight
    multiset_exact: bool | None = None
    location_changed: bool | None = None
    if mode == "confidence_shuffled":
        valid_indices = torch.nonzero(valid_prior, as_tuple=False).squeeze(-1)
        valid_weights = confidence_weight[valid_indices]
        if valid_indices.numel() > 1:
            permutation = torch.randperm(
                valid_indices.numel(), generator=shuffle_generator, device=valid_indices.device
            )
            shuffled = valid_weights[permutation]
            has_distinct_weights = bool(torch.unique(valid_weights).numel() > 1)
            if has_distinct_weights and torch.equal(shuffled, valid_weights):
                shuffled = torch.roll(valid_weights, shifts=1)
            weights = confidence_weight.clone()
            weights[valid_indices] = shuffled
            location_changed = not torch.equal(shuffled, valid_weights)
        else:
            location_changed = False
        multiset_exact = torch.equal(
            weights[valid_prior].sort().values, confidence_weight[valid_prior].sort().values
        )
        if not multiset_exact:
            raise AssertionError("sampled confidence shuffle changed the valid weight multiset")

    if mode not in {"valid_uniform", "confidence", "confidence_shuffled"}:
        return weights, {}
    confidence_valid = confidence_weight[valid_prior].double().sort().values
    resolved_valid = weights[valid_prior].double().sort().values
    return weights, {
        "valid_count": int(valid_prior.sum()),
        "valid_indices": tuple(torch.nonzero(valid_prior, as_tuple=False).squeeze(-1).tolist()),
        "distinct_confidence_count": int(torch.unique(confidence_valid).numel()),
        "confidence_sum": float(confidence_valid.sum()),
        "confidence_square_sum": float(confidence_valid.square().sum()),
        "resolved_sum": float(resolved_valid.sum()),
        "resolved_square_sum": float(resolved_valid.square().sum()),
        "invalid_nonzero_count": int(torch.count_nonzero(weights[~valid_prior])),
        "shuffle_multiset_exact": multiset_exact,
        "shuffle_location_changed": location_changed,
    }


def bounded_ray_anchor_loss(
    raw_t: torch.Tensor,
    raw_t_init: torch.Tensor,
    anchor_fraction: torch.Tensor,
    anchor_weight: torch.Tensor,
    mode: str,
    beta: float = 0.05,
    confidence_threshold: float = 0.5,
    normalized_scale: float = 1.0,
) -> torch.Tensor:
    """Regularize bounded-ray depth in legacy or confidence-aware coordinates.

    ``legacy`` reproduces the historical raw-logit L2 penalty around the jittered
    initialization. The other modes use an unjittered target in normalized ray-fraction
    space so the force does not depend on the length of the ray/AABB interval.
    """
    if mode == "legacy":
        return (raw_t - raw_t_init).square().mean()
    per_ray = normalized_scale * F.smooth_l1_loss(
        torch.sigmoid(raw_t), anchor_fraction, reduction="none", beta=beta
    )
    if mode == "normalized":
        return per_ray.mean()
    weights = anchor_weight
    if mode == "thresholded":
        weights = ((weights > 0) & (weights >= confidence_threshold)).to(per_ray)
    return (per_ray * weights).sum() / weights.sum().clamp_min(1e-8)


def world_position_consistency_loss(
    means: torch.Tensor,
    position_pairs: torch.Tensor,
    scene_extent: float,
    beta: float = 0.05,
) -> torch.Tensor:
    """Robust position-only consistency for fixed cross-view primitive pairs.

    The residual follows the preregistered MAC-Splat-grounded form: world-coordinate
    L1 discrepancy normalized by scene extent, followed by standard Huber loss to zero.
    Pair weights are uniform and the reduction is a mean over edges.
    """
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    if position_pairs.ndim != 2 or position_pairs.shape[1] != 2 or position_pairs.shape[0] == 0:
        raise ValueError("position_pairs must have non-empty shape (E, 2)")
    if not math.isfinite(scene_extent) or scene_extent <= 0:
        raise ValueError("scene_extent must be finite and positive")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("position_consistency_beta must be finite and positive")
    delta = (means[position_pairs[:, 0]] - means[position_pairs[:, 1]]).abs().sum(dim=-1)
    residual = delta / scene_extent
    return F.huber_loss(residual, torch.zeros_like(residual), reduction="mean", delta=beta)


def _validate_position_pairs(
    position_pairs: torch.Tensor,
    source_view_ids: torch.Tensor,
) -> torch.Tensor:
    """Validate and freeze canonical retained-index pairs on the lifter device."""
    if not isinstance(position_pairs, torch.Tensor):
        raise TypeError("position_pairs must be a torch.Tensor")
    if position_pairs.dtype != torch.long:
        raise ValueError("position_pairs must use torch.int64 indices")
    if position_pairs.ndim != 2 or position_pairs.shape[1] != 2 or position_pairs.shape[0] == 0:
        raise ValueError("position_pairs must have non-empty shape (E, 2)")
    if position_pairs.requires_grad or position_pairs.grad_fn is not None:
        raise ValueError("position_pairs must be detached")

    pairs = position_pairs.detach().to(device=source_view_ids.device).clone()
    if bool((pairs < 0).any()) or bool((pairs >= source_view_ids.numel()).any()):
        raise ValueError("position_pairs contain an out-of-range retained primitive index")
    if not bool((pairs[:, 0] < pairs[:, 1]).all()):
        raise ValueError("position_pairs must be canonical with the lower retained index first")
    if bool((source_view_ids[pairs[:, 0]] == source_view_ids[pairs[:, 1]]).any()):
        raise ValueError("position_pairs must connect primitives from different source views")
    if torch.unique(pairs, dim=0).shape[0] != pairs.shape[0]:
        raise ValueError("position_pairs must be unique")
    return pairs


class GradientLifter:
    """Lift 2D gaussians and optimize depth (+rotation +scale) by multi-view descent."""

    def __init__(
        self,
        iterations: int = 150,
        lr: float = 0.1,
        lr_rotation: float = 5e-3,
        lr_scale: float = 5e-3,
        rasterizer: str = "auto",
        min_weight: float = 0.05,
        depth_jitter: float = 0.15,
        ray_thickness: float = 1.0,
        init_opacity: float = 0.1,
        optimize_rotation: bool = True,
        optimize_scale: bool = True,
        max_log_scale_delta: float = 1.0,
        depth_prior_lambda: float = 1e-3,
        scale_prior_lambda: float = 1e-4,
        ssim_lambda: float = 0.0,
        merge: bool = True,
        merge_voxel_frac: float = 0.01,
        sh_degree: int = 0,
        seed: int = 0,
        depth_anchor_mode: str = "legacy",
        depth_anchor_beta: float = 0.05,
        depth_confidence_threshold: float = 0.5,
        photometric_supervision_mode: str = "all",
        position_consistency_lambda: float = 0.0,
        position_consistency_beta: float = 0.05,
        plane_consistency_lambda: float = 0.0,
        normal_consistency_lambda: float = 0.0,
    ):
        self.iterations = iterations
        self.lr = lr
        self.lr_rotation = lr_rotation
        self.lr_scale = lr_scale
        self.rasterizer = rasterizer
        self.min_weight = min_weight
        self.depth_jitter = depth_jitter
        self.ray_thickness = max(ray_thickness, _MIN_THICKNESS)
        self.init_opacity = init_opacity
        self.optimize_rotation = optimize_rotation
        self.optimize_scale = optimize_scale
        self.max_log_scale_delta = max_log_scale_delta
        self.depth_prior_lambda = depth_prior_lambda
        if depth_anchor_mode not in _DEPTH_ANCHOR_MODES:
            raise ValueError(
                f"depth_anchor_mode must be one of {sorted(_DEPTH_ANCHOR_MODES)}, "
                f"got {depth_anchor_mode!r}"
            )
        if not math.isfinite(depth_anchor_beta) or depth_anchor_beta <= 0:
            raise ValueError("depth_anchor_beta must be finite and positive")
        if not math.isfinite(depth_confidence_threshold) or not (
            0 <= depth_confidence_threshold <= 1
        ):
            raise ValueError("depth_confidence_threshold must be finite and in [0, 1]")
        self.depth_anchor_mode = depth_anchor_mode
        self.depth_anchor_beta = depth_anchor_beta
        self.depth_confidence_threshold = depth_confidence_threshold
        if photometric_supervision_mode not in _PHOTOMETRIC_SUPERVISION_MODES:
            raise ValueError(
                "photometric_supervision_mode must be one of "
                f"{sorted(_PHOTOMETRIC_SUPERVISION_MODES)}, "
                f"got {photometric_supervision_mode!r}"
            )
        self.photometric_supervision_mode = photometric_supervision_mode
        if not math.isfinite(position_consistency_lambda) or position_consistency_lambda < 0:
            raise ValueError("position_consistency_lambda must be finite and non-negative")
        if not math.isfinite(position_consistency_beta) or position_consistency_beta <= 0:
            raise ValueError("position_consistency_beta must be finite and positive")
        self.position_consistency_lambda = position_consistency_lambda
        self.position_consistency_beta = position_consistency_beta
        if not math.isfinite(plane_consistency_lambda) or plane_consistency_lambda < 0:
            raise ValueError("plane_consistency_lambda must be finite and non-negative")
        if not math.isfinite(normal_consistency_lambda) or normal_consistency_lambda < 0:
            raise ValueError("normal_consistency_lambda must be finite and non-negative")
        if normal_consistency_lambda > 0 and not optimize_rotation:
            raise ValueError("positive normal_consistency_lambda requires optimize_rotation=True")
        self.plane_consistency_lambda = plane_consistency_lambda
        self.normal_consistency_lambda = normal_consistency_lambda
        self.scale_prior_lambda = scale_prior_lambda
        self.ssim_lambda = ssim_lambda
        self.merge = merge
        self.merge_voxel_frac = merge_voxel_frac
        self.sh_degree = sh_degree
        self.seed = seed
        self.history: list[float] = []
        self.anchor_history: list[float] = []
        self.position_history: list[float] = []
        self.plane_history: list[float] = []
        self.normal_history: list[float] = []
        self.anchor_weight_diagnostics: list[dict[str, object]] = []
        self.photometric_supervision_diagnostics: list[dict[str, object]] = []
        self.target_view_history: list[int] = []
        self.rendered_count_history: list[int] = []
        self.source_view_ids_before_merge: torch.Tensor | None = None
        self.source_xy_before_merge: torch.Tensor | None = None
        self.source_view_ranges_before_merge: list[tuple[int, int]] = []
        self.position_pairs_before_merge: torch.Tensor | None = None
        self.oriented_targets_before_merge: OrientedPointTargets | None = None
        self.oriented_axis_indices_before_merge: torch.Tensor | None = None
        self.initial_ray_fractions_before_merge: torch.Tensor | None = None
        self.final_ray_fractions_before_merge: torch.Tensor | None = None
        self.resolved_depth_anchor_scale = 1.0
        self.n_before_merge = 0

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        """Lift every view, optimize along the rays, and merge the result."""
        return self.lift_with_priors(gaussians2d, scene, priors=None)

    def lift_with_position_pairs(
        self,
        gaussians2d: list[Gaussians2D],
        scene: SceneData,
        position_pairs: torch.Tensor,
    ) -> Gaussians3D:
        """Lift with a fixed position-only cross-view graph and no depth priors."""
        return self.lift_with_priors(
            gaussians2d,
            scene,
            priors=None,
            position_pairs=position_pairs,
        )

    def lift_with_oriented_points(
        self,
        gaussians2d: list[Gaussians2D],
        scene: SceneData,
        oriented_targets: OrientedPointTargets,
    ) -> Gaussians3D:
        """Lift with fixed detached local-plane and normal-alignment targets."""
        return self.lift_with_priors(
            gaussians2d,
            scene,
            priors=None,
            oriented_targets=oriented_targets,
        )

    def lift_with_priors(
        self,
        gaussians2d: list[Gaussians2D],
        scene: SceneData,
        priors: list[AlignedDepthPrior] | None,
        merge_color_bin_size: float | None = None,
        position_pairs: torch.Tensor | None = None,
        oriented_targets: OrientedPointTargets | None = None,
    ) -> Gaussians3D:
        """Optimize bounded rays, optionally initialized and weighted by depth priors."""
        if self.position_consistency_lambda > 0 and position_pairs is None:
            raise ValueError("positive position_consistency_lambda requires position_pairs")
        if (
            self.plane_consistency_lambda > 0 or self.normal_consistency_lambda > 0
        ) and oriented_targets is None:
            raise ValueError("positive plane/normal consistency lambda requires oriented_targets")
        if priors is not None and len(priors) != len(gaussians2d):
            raise ValueError("one aligned depth prior is required per fitted view")
        if self.photometric_supervision_mode != "all" and len(gaussians2d) != scene.n_views:
            raise ValueError(
                "source-aware photometric supervision requires one fitted Gaussian set per "
                "scene camera"
            )
        center, extent = scene.center_and_extent()
        device = gaussians2d[0].xy.device
        center = center.to(device)
        gen = torch.Generator(device=device).manual_seed(self.seed)
        shuffle_gen = torch.Generator(device=device).manual_seed(self.seed + 1_000_003)

        origins, dirs, t_nears, t_fars, raw_t0 = [], [], [], [], []
        anchor_fractions, anchor_weights, anchor_validity = [], [], []
        source_view_ids = []
        source_xy = []
        self.anchor_weight_diagnostics = []
        base_parts: list[Gaussians3D] = []
        observation_weights = []
        half = 0.5 * extent
        for view_index, (g2d, camera) in enumerate(zip(gaussians2d, scene.cameras)):
            keep = g2d.weight > self.min_weight
            if scene.masks is not None:
                keep &= bilinear_sample(scene.masks[view_index].to(g2d.xy), g2d.xy) > 0.5
            g2d_v = Gaussians2D(
                xy=g2d.xy[keep], chol=g2d.chol[keep], color=g2d.color[keep], weight=g2d.weight[keep]
            )
            n = g2d_v.n
            if n == 0:
                continue
            o, d = camera.pixel_rays(g2d_v.xy)  # d has unit camera-space depth: t == depth
            near, far = _ray_box(o, d, center - half, center + half)
            intersects = far > near.clamp_min(0.05)
            if not bool(intersects.any()):
                continue
            g2d_v = Gaussians2D(
                xy=g2d_v.xy[intersects],
                chol=g2d_v.chol[intersects],
                color=g2d_v.color[intersects],
                weight=g2d_v.weight[intersects],
            )
            d = d[intersects]
            near = near[intersects].clamp_min(0.05)
            far = far[intersects]
            n = g2d_v.n
            origins.append(o.expand(n, 3))
            dirs.append(d)
            t_nears.append(near)
            t_fars.append(far)
            # Initialize at the optical-axis center depth, then jitter within the valid ray/AABB
            # interval. Unlike Euclidean camera distance this is the correct parameter for d.z=1.
            center_depth = camera.project(center[None])[1][0]
            initial_depth = center_depth.expand(n)
            confidence = torch.ones(n, device=device)
            valid_prior = torch.ones(n, dtype=torch.bool, device=device)
            sampled_confidence = torch.ones(n, device=device)
            if priors is not None:
                prior = priors[view_index]
                sampled = bilinear_sample(prior.depth.to(g2d_v.xy), g2d_v.xy)
                valid_prior = torch.isfinite(sampled) & (sampled > near) & (sampled < far)
                initial_depth = torch.where(valid_prior, sampled, initial_depth)
                if prior.confidence is not None:
                    sampled_confidence = bilinear_sample(prior.confidence.to(g2d_v.xy), g2d_v.xy)
                    sampled_confidence = torch.nan_to_num(
                        sampled_confidence, nan=0.0, posinf=0.0, neginf=0.0
                    ).clamp(0, 1)
                    confidence = torch.where(valid_prior, sampled_confidence, 0.1)
            anchor_weight, weight_diagnostics = _resolve_anchor_weights(
                valid_prior, sampled_confidence, self.depth_anchor_mode, shuffle_gen
            )
            if weight_diagnostics:
                weight_diagnostics["view_index"] = view_index
                self.anchor_weight_diagnostics.append(weight_diagnostics)
            anchor_fraction = ((initial_depth - near) / (far - near)).clamp(0.05, 0.95)
            jittered_fraction = (
                anchor_fraction
                + self.depth_jitter * (torch.rand(n, generator=gen, device=device) - 0.5)
            ).clamp(0.01, 0.99)
            raw_t0.append(torch.logit(jittered_fraction))
            anchor_fractions.append(anchor_fraction)
            anchor_weights.append(anchor_weight)
            anchor_validity.append(valid_prior)
            source_view_ids.append(torch.full((n,), view_index, dtype=torch.long, device=device))
            source_xy.append(g2d_v.xy)
            observation_weights.append((g2d_v.weight * confidence).clamp_min(1e-3))
            # Unit-depth lift fixes each gaussian's rotation and unit-depth scales; the
            # along-ray sigma is the footprint minor axis scaled by ray_thickness (the
            # "epsilon" knob, clamped away from degeneracy).
            s_min = eigvals_2x2(g2d_v.covariance())[:, 0].clamp_min(1e-8).sqrt()
            sigma_unit = self.ray_thickness * s_min / (0.5 * (camera.fx + camera.fy))
            base_parts.append(
                lift_view_at_depth(
                    camera,
                    g2d_v,
                    torch.ones(n, device=device),
                    sigma_unit,
                    sh_degree=self.sh_degree,
                    opacity=torch.full((n,), self.init_opacity, device=device),
                )
            )
        if not base_parts:
            raise ValueError("no gaussians above the weight threshold to lift")

        base = Gaussians3D.cat(base_parts)
        source_view_t = torch.cat(source_view_ids)
        self.source_view_ids_before_merge = source_view_t.detach().clone()
        self.source_xy_before_merge = torch.cat(source_xy).detach().clone()
        source_counts = torch.bincount(source_view_t, minlength=scene.n_views).tolist()
        cursor = 0
        self.source_view_ranges_before_merge = []
        for count in source_counts:
            self.source_view_ranges_before_merge.append((cursor, cursor + count))
            cursor += count
        origins_t = torch.cat(origins)
        dirs_t = torch.cat(dirs)
        near_t = torch.cat(t_nears)
        far_t = torch.cat(t_fars)
        anchor_fraction_t = torch.cat(anchor_fractions).detach()
        anchor_weight_t = torch.cat(anchor_weights).detach()
        anchor_valid_t = torch.cat(anchor_validity).detach()
        scale_reference = anchor_fraction_t * (1.0 - anchor_fraction_t)
        if bool(anchor_valid_t.any()):
            scale_reference = scale_reference[anchor_valid_t]
        # Match the local raw-parameter curvature of normalized Smooth L1 to legacy
        # raw-logit L2 at the median valid anchor. This keeps the ablation from becoming
        # an accidental comparison of regularization strengths.
        reference_jacobian = scale_reference.median().clamp_min(1e-4)
        self.resolved_depth_anchor_scale = float(
            2.0 * self.depth_anchor_beta / reference_jacobian.square()
        )

        # Optimized parameters. Lateral position and color/SH/opacity stay frozen.
        raw_t_init = torch.cat(raw_t0).detach()
        self.initial_ray_fractions_before_merge = torch.sigmoid(raw_t_init).detach().clone()
        raw_t = raw_t_init.clone().requires_grad_(True)
        quat_p = base.quats.detach().clone().requires_grad_(self.optimize_rotation)
        raw_scale = torch.zeros_like(base.log_scales).requires_grad_(self.optimize_scale)

        groups = [{"params": [raw_t], "lr": self.lr}]
        if self.optimize_rotation:
            groups.append({"params": [quat_p], "lr": self.lr_rotation})
        if self.optimize_scale:
            groups.append({"params": [raw_scale], "lr": self.lr_scale})
        opt = torch.optim.Adam(groups)

        def build() -> Gaussians3D:
            t = near_t + torch.sigmoid(raw_t) * (far_t - near_t)
            dscale = self.max_log_scale_delta * torch.tanh(raw_scale)
            return Gaussians3D(
                means=origins_t + t[:, None] * dirs_t,
                quats=torch.nn.functional.normalize(quat_p, dim=-1),
                # t-scaling keeps the projected footprint ~constant with depth; dscale is
                # the free per-axis correction the optimizer learns.
                log_scales=base.log_scales + t.log()[:, None] + dscale,
                opacity=base.opacity,
                sh=base.sh,
            )

        renderer = get_rasterizer(self.rasterizer, device=device)
        self.history = []
        self.anchor_history = []
        self.position_history = []
        self.plane_history = []
        self.normal_history = []
        self.position_pairs_before_merge = (
            None
            if position_pairs is None
            else _validate_position_pairs(position_pairs, source_view_t).detach().clone()
        )
        self.oriented_targets_before_merge = (
            None
            if oriented_targets is None
            else validate_oriented_point_targets(
                oriented_targets,
                n_retained=base.n,
                device=device,
                dtype=base.means.dtype,
            )
        )
        self.oriented_axis_indices_before_merge = (
            None
            if self.oriented_targets_before_merge is None
            else base.log_scales[self.oriented_targets_before_merge.indices]
            .argmin(dim=-1)
            .detach()
            .clone()
        )
        train_views = scene.training_views
        supervision_exclusions, supervision_diagnostics = _build_photometric_exclusions(
            source_view_t,
            train_views,
            self.photometric_supervision_mode,
            self.seed,
        )
        for item in supervision_diagnostics:
            excluded = supervision_exclusions[int(item["target_view"])]
            item["excluded_opacity_sum"] = float(base.opacity[excluded].double().sum())
            item["rendered_opacity_sum"] = float(base.opacity.double().sum()) - float(
                item["excluded_opacity_sum"]
            )
        self.photometric_supervision_diagnostics = supervision_diagnostics
        self.target_view_history = []
        self.rendered_count_history = []
        for _ in range(self.iterations):
            view_pos = int(torch.randint(0, len(train_views), (1,), generator=gen, device=device))
            v = train_views[view_pos]
            current = build()
            if self.photometric_supervision_mode == "all":
                rendered = current
            else:
                keep = torch.ones(current.n, dtype=torch.bool, device=device)
                keep[supervision_exclusions[v]] = False
                rendered = current.subset(keep)
            out = renderer.render(rendered, scene.cameras[v])
            target = scene.images[v]
            if scene.masks is not None:
                mask = scene.masks[v].to(target)[..., None]
                target = target * mask
                loss = ((out.color - target).abs() * (0.1 + 0.9 * mask)).mean()
            else:
                loss = (out.color - target).abs().mean()
            if self.ssim_lambda > 0:
                loss = loss + self.ssim_lambda * (1.0 - ssim(out.color, target))
            anchor_loss = bounded_ray_anchor_loss(
                raw_t,
                raw_t_init,
                anchor_fraction_t,
                anchor_weight_t,
                self.depth_anchor_mode,
                beta=self.depth_anchor_beta,
                confidence_threshold=self.depth_confidence_threshold,
                normalized_scale=self.resolved_depth_anchor_scale,
            )
            loss = loss + self.depth_prior_lambda * anchor_loss
            position_loss = None
            if self.position_pairs_before_merge is not None:
                position_loss = world_position_consistency_loss(
                    current.means,
                    self.position_pairs_before_merge,
                    extent,
                    beta=self.position_consistency_beta,
                )
                if self.position_consistency_lambda > 0:
                    loss = loss + self.position_consistency_lambda * position_loss
            plane_loss = None
            normal_loss = None
            if self.oriented_targets_before_merge is not None:
                plane_loss = local_plane_loss(
                    current.means,
                    self.oriented_targets_before_merge,
                    extent,
                )
                normal_loss = shortest_axis_normal_loss(
                    current.quats,
                    self.oriented_axis_indices_before_merge,
                    self.oriented_targets_before_merge,
                )
                if self.plane_consistency_lambda > 0:
                    loss = loss + self.plane_consistency_lambda * plane_loss
                if self.normal_consistency_lambda > 0:
                    loss = loss + self.normal_consistency_lambda * normal_loss
            if self.optimize_scale:
                loss = loss + self.scale_prior_lambda * raw_scale.square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.history.append(float(loss.detach()))
            self.anchor_history.append(float(anchor_loss.detach()))
            if position_loss is not None:
                self.position_history.append(float(position_loss.detach()))
            if plane_loss is not None:
                self.plane_history.append(float(plane_loss.detach()))
                self.normal_history.append(float(normal_loss.detach()))
            self.target_view_history.append(v)
            self.rendered_count_history.append(rendered.n)

        result = build().detach()
        self.final_ray_fractions_before_merge = torch.sigmoid(raw_t.detach()).clone()
        self.n_before_merge = result.n
        if self.merge:
            result = merge_by_voxel(
                result,
                self.merge_voxel_frac * extent,
                opacity_mode="mean",
                component_weights=torch.cat(observation_weights),
                color_bin_size=merge_color_bin_size,
            )
        return result


def _ray_box(
    origin: torch.Tensor, directions: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect one-origin camera rays with an axis-aligned scene volume."""
    safe = torch.where(directions.abs() < 1e-9, torch.full_like(directions, 1e-9), directions)
    ta = (lo[None] - origin[None]) / safe
    tb = (hi[None] - origin[None]) / safe
    return torch.minimum(ta, tb).amax(dim=-1), torch.maximum(ta, tb).amin(dim=-1)
