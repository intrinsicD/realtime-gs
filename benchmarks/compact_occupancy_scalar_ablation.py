#!/usr/bin/env python3
"""Footprint-sampled one-scalar occupancy ablation for compact Stage-2 lifting.

The experiment consumes the frozen seven-view exact masked ``ReconstructionInputs`` bundle and
samples masks only at deterministic points in each optimized 2D Gaussian footprint.  It compares
the current center scalar, a Gaussian sample mean, normalized log-sum-exp smooth maxima, and a
hard-maximum ceiling.  A fixed three-view tuning split selects one LSE temperature under a
precision guard before the four report-view metrics are queried.  Stage B then reruns the fixed
835-Gaussian component-center lift for center, mean, selected LSE, and an optional useful hard-max
ceiling.  Exact masked compact fields remain the only color teachers; source RGB is never decoded.

This is an exploratory development ablation, not held-out novel-view or default-changing evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib
import json
import math
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.base import bilinear_sample
from rtgs.lift.compact_carve import CompactCarveInitializer

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))
base = importlib.import_module("compact_masked_lift_screen")
ROOT = Path(__file__).resolve().parents[1]
ANCHOR_BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
PRIOR_SCREEN = ROOT / "runs/compact_masked_lift_screen_20260717"
PRIOR_CENTER_PLY = PRIOR_SCREEN / "arms/masked_compact_proxy_component_centers/gaussians_init.ply"
PRIOR_CENTER_PROXY_BUNDLE = PRIOR_SCREEN / "compact_occupancy_proxy_bundle"
PREREGISTRATION = ROOT / "benchmarks/results/20260717_compact_occupancy_scalar_PREREG.md"
IMPLEMENTATION_REVIEW = (
    ROOT / "benchmarks/results/20260717_compact_occupancy_scalar_IMPLEMENTATION_REVIEW.md"
)
DEFAULT_OUT = ROOT / "runs/compact_occupancy_scalar_ablation_20260717"

TUNING_VIEWS = ("C0001", "C0014", "C0026")
REPORT_VIEWS = ("C0008", "C0021", "C0031", "C0039")
FOOTPRINT_SAMPLES = 32
FOOTPRINT_SEED = 18017
OCCUPANCY_SEED = 18018
OCCUPANCY_SAMPLES_PER_VIEW = 16_384
LSE_BETAS = (2, 4, 8, 16)
VARIANT_NAMES = (
    "center",
    "mean",
    *(f"lse_beta_{beta}" for beta in LSE_BETAS),
    "hard_max",
)
PRECISION_GUARD_DROP = 0.01
HARD_MAX_RECALL_FLOOR = 0.02
HARD_MAX_IOU_FLOOR = 0.01
CPU_THREADS = 1
CENTER_REPLAY_MEANS_TOLERANCE = 1e-5
CENTER_REPLAY_COVARIANCE_TOLERANCE = 5e-3


def timestamp_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def pin_cpu_threads() -> dict[str, int]:
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(CPU_THREADS)
    except RuntimeError as error:
        if torch.get_num_interop_threads() != CPU_THREADS:
            raise RuntimeError("could not pin PyTorch inter-op threads") from error
    binding = {
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
    }
    if set(binding.values()) != {CPU_THREADS}:
        raise RuntimeError("PyTorch CPU thread pin did not take effect")
    return binding


def raw_mask_bindings(view_names: list[str]) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": (base.SCENE / "mask" / f"mask_{name}.png").relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(base.SCENE / "mask" / f"mask_{name}.png"),
        }
        for name in view_names
    }


def center_extent_binding(inputs: ReconstructionInputs, config: Any) -> dict[str, Any]:
    dtype = inputs.observations[0].dtype
    center, extent = base._center_and_extent(inputs, dtype)
    half = extent * config.bounds_scale
    lower = center - half
    upper = center + half
    return {
        "center": center.tolist(),
        "center_sha256": base.tensor_hash(center),
        "extent": extent,
        "bounds_scale": config.bounds_scale,
        "lower": lower.tolist(),
        "lower_sha256": base.tensor_hash(lower),
        "upper": upper.tolist(),
        "upper_sha256": base.tensor_hash(upper),
        "derivation": "frozen compact_carve._center_and_extent on bound inputs",
    }


def freeze_stage_b_center_extent(
    inputs: ReconstructionInputs,
    config: Any,
) -> tuple[ReconstructionInputs, dict[str, Any]]:
    dtype = inputs.observations[0].dtype
    center, extent = base._center_and_extent(inputs, dtype)
    frozen_center = center.detach().clone()
    frozen = ReconstructionInputs(
        observations=inputs.observations,
        cameras=inputs.cameras,
        view_names=inputs.view_names,
        points=inputs.points,
        point_visibility=inputs.point_visibility,
        bounds_hint=(frozen_center, extent),
        name=f"{inputs.name}_frozen_stage_b_bounds",
    )
    binding = center_extent_binding(frozen, config)
    binding["intervention"] = (
        "camera-only least-squares center evaluated exactly once, then injected as bounds_hint "
        "for every Stage-B arm and support diagnostic"
    )
    return frozen, binding


def source_binding() -> dict[str, Any]:
    inherited = base.source_binding()
    hashes = dict(inherited["files"])
    relative = Path("benchmarks/compact_occupancy_scalar_ablation.py")
    hashes[relative.as_posix()] = base.sha256_file(ROOT / relative)
    paths = [Path(path) for path in hashes]
    status = base.git_output(
        "status",
        "--porcelain=v1",
        "--",
        *(path.as_posix() for path in paths),
    )
    return {
        "git_revision": base.git_output("rev-parse", "HEAD"),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "files": hashes,
        "aggregate": base.canonical_hash(hashes),
    }


def footprint_standard_samples(dtype: torch.dtype) -> torch.Tensor:
    if FOOTPRINT_SAMPLES % 2:
        raise RuntimeError("antithetic footprint sample count must be even")
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=FOOTPRINT_SEED)
    unit = engine.draw(FOOTPRINT_SAMPLES // 2, dtype=dtype).clamp(1e-4, 1.0 - 1e-4)
    normal = torch.erfinv(2.0 * unit - 1.0) * math.sqrt(2.0)
    return torch.cat([normal, -normal], dim=0)


def footprint_mask_samples(
    field: GaussianObservationField,
    mask: torch.Tensor,
    standard: torch.Tensor,
) -> torch.Tensor:
    scales = field.effective_variances().sqrt()
    local = standard[None, :, :] * scales[:, None, :]
    cosine = torch.cos(field.rotations)[:, None]
    sine = torch.sin(field.rotations)[:, None]
    delta_x = cosine * local[:, :, 0] - sine * local[:, :, 1]
    delta_y = sine * local[:, :, 0] + cosine * local[:, :, 1]
    xy = field.means[:, None, :] + torch.stack([delta_x, delta_y], dim=-1)
    return bilinear_sample(mask, xy.reshape(-1, 2)).reshape(field.n, -1).clamp(0.0, 1.0)


def normalized_lse(samples: torch.Tensor, beta: float) -> torch.Tensor:
    if samples.ndim < 1 or samples.shape[-1] <= 0:
        raise ValueError("normalized LSE requires a non-empty sample dimension")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("normalized LSE beta must be finite and positive")
    return (torch.logsumexp(float(beta) * samples, dim=-1) - math.log(samples.shape[-1])) / float(
        beta
    )


def variant_scalars(
    field: GaussianObservationField,
    mask: torch.Tensor,
    standard: torch.Tensor,
) -> dict[str, torch.Tensor]:
    samples = footprint_mask_samples(field, mask, standard)
    result = {
        "center": bilinear_sample(mask, field.means).clamp(0.0, 1.0),
        "mean": samples.mean(dim=1),
        "hard_max": samples.max(dim=1).values,
    }
    for beta in LSE_BETAS:
        result[f"lse_beta_{beta}"] = normalized_lse(samples, float(beta))
    for name, scalar in result.items():
        if (
            not bool(torch.isfinite(scalar).all())
            or bool((scalar < 0.0).any())
            or bool((scalar > 1.0 + 1e-6).any())
        ):
            raise RuntimeError(f"invalid bounded occupancy scalar for {name}")
    return result


def make_proxy_field(
    *,
    view_name: str,
    anchor: GaussianObservationField,
    mask: torch.Tensor,
    scalar: torch.Tensor,
    variant: str,
    standard_sha256: str,
) -> GaussianObservationField:
    semantics = {
        "schema": "compact_occupancy_scalar_ablation.proxy.v1",
        "variant": variant,
        "footprint_samples": FOOTPRINT_SAMPLES,
        "footprint_seed": FOOTPRINT_SEED,
        "standard_samples_sha256": standard_sha256,
        "sampling_distribution": "antithetic scrambled Sobol transformed to N(0,I)",
        "scalar_role": "Gaussian occupancy amplitude; colors are unused ones",
    }
    source_digest = base.canonical_hash(
        {
            "view": view_name,
            "means": base.tensor_hash(anchor.means),
            "log_scales": base.tensor_hash(anchor.log_scales),
            "rotations": base.tensor_hash(anchor.rotations),
            "mask": base.tensor_hash(mask),
            "scalar": base.tensor_hash(scalar),
        }
    )
    return GaussianObservationField(
        width=anchor.width,
        height=anchor.height,
        means=anchor.means,
        log_scales=anchor.log_scales,
        rotations=anchor.rotations,
        colors=anchor.colors.new_ones((anchor.n, 3)),
        amplitudes=scalar,
        color_grads=None,
        filter_variance=anchor.filter_variance,
        blend_mode="normalized",
        epsilon=anchor.epsilon,
        sigma_cutoff=anchor.sigma_cutoff,
        support_fade_alpha=anchor.support_fade_alpha,
        aa_dilation=anchor.aa_dilation,
        view_id=view_name,
        fit_window=anchor.fit_window,
        n_init=anchor.n_init,
        provider="synthetic_fixture",
        producer_version=f"footprint-occupancy-{variant}-v1",
        producer_source_digest=source_digest,
        fit_config_digest=base.canonical_hash(semantics),
    )


def build_proxy_variants(
    anchor_inputs: ReconstructionInputs,
    masks: list[torch.Tensor],
) -> tuple[
    dict[str, list[GaussianObservationField]],
    list[dict[str, Any]],
    torch.Tensor,
]:
    standard = footprint_standard_samples(anchor_inputs.observations[0].dtype)
    standard_sha256 = base.tensor_hash(standard)
    fields: dict[str, list[GaussianObservationField]] = {name: [] for name in VARIANT_NAMES}
    records = []
    for view_name, anchor, mask in zip(
        anchor_inputs.view_names,
        anchor_inputs.observations,
        masks,
        strict=True,
    ):
        scalars = variant_scalars(anchor, mask, standard)
        item = {"view": view_name, "components": anchor.n, "variants": {}}
        for variant, scalar in scalars.items():
            fields[variant].append(
                make_proxy_field(
                    view_name=view_name,
                    anchor=anchor,
                    mask=mask,
                    scalar=scalar,
                    variant=variant,
                    standard_sha256=standard_sha256,
                )
            )
            item["variants"][variant] = {
                "scalar_sha256": base.tensor_hash(scalar),
            }
        records.append(item)
    return fields, records, standard


def center_proxy_replay_audit(
    center_fields: list[GaussianObservationField],
    prior_bundle: Path,
) -> dict[str, Any]:
    prior = ReconstructionInputs.load(prior_bundle, strict=True)
    if tuple(prior.view_names) != base.TRAIN_VIEWS or len(center_fields) != prior.n_views:
        raise RuntimeError("prior center proxy view contract differs from the frozen screen")
    records = []
    tensor_names = ("means", "log_scales", "rotations", "colors", "amplitudes")
    for name, current, reference in zip(
        prior.view_names,
        center_fields,
        prior.observations,
        strict=True,
    ):
        tensors = {
            tensor_name: torch.equal(
                getattr(current, tensor_name),
                getattr(reference, tensor_name),
            )
            for tensor_name in tensor_names
        }
        filter_variance_equal = (
            current.filter_variance is None and reference.filter_variance is None
        ) or (
            current.filter_variance is not None
            and reference.filter_variance is not None
            and torch.equal(current.filter_variance, reference.filter_variance)
        )
        semantics_equal = (
            current.width == reference.width
            and current.height == reference.height
            and current.fit_window == reference.fit_window
            and current.sigma_cutoff == reference.sigma_cutoff
            and current.support_fade_alpha == reference.support_fade_alpha
            and current.aa_dilation == reference.aa_dilation
        )
        records.append(
            {
                "view": name,
                "tensor_bit_exact": tensors,
                "filter_variance_bit_exact": filter_variance_equal,
                "query_semantics_equal": semantics_equal,
                "pass": all(tensors.values()) and filter_variance_equal and semantics_equal,
            }
        )
    passed = all(record["pass"] for record in records)
    if not passed:
        raise RuntimeError("new center proxy does not exactly replay prior proxy tensors")
    return {
        "status": "PASS",
        "scope": "proxy tensors/query semantics, not cross-run 3D floating-point output",
        "prior_bundle": base.bundle_binding(prior_bundle),
        "records": records,
    }


def save_proxy_bundles(
    output: Path,
    anchor_inputs: ReconstructionInputs,
    variant_fields: dict[str, list[GaussianObservationField]],
) -> dict[str, dict[str, Any]]:
    root = output / "proxy_bundles"
    root.mkdir()
    bindings = {}
    for variant, fields in variant_fields.items():
        inputs = ReconstructionInputs(
            observations=fields,
            cameras=anchor_inputs.cameras,
            view_names=anchor_inputs.view_names,
            points=anchor_inputs.points,
            point_visibility=anchor_inputs.point_visibility,
            bounds_hint=anchor_inputs.bounds_hint,
            name=f"compact_occupancy_{variant}_screen",
        )
        path = root / variant
        inputs.save(path)
        bindings[variant] = base.bundle_binding(path)
    return bindings


def build_single_variant_backends(
    anchor_inputs: ReconstructionInputs,
    anchor_indexes: list[GaussianObservationIndex],
    fields: list[GaussianObservationField],
    masks: list[torch.Tensor],
    config: Any,
) -> tuple[list[GaussianObservationIndex], list[base.CoverageOverrideBackend]]:
    indexes = [
        GaussianObservationIndex(
            field,
            tile_size=config.tile_size,
            max_entries=config.max_index_entries_per_view,
            max_candidates=config.max_candidates_per_tile,
        )
        for field in fields
    ]
    backends = base.make_backends(
        color_inputs=anchor_inputs,
        normalization_inputs=anchor_inputs,
        color_indexes=anchor_indexes,
        coverage_kind="compact_proxy",
        masks=masks,
        proxy_fields=fields,
        proxy_indexes=indexes,
    )
    return indexes, backends


def coordinates_for_views(
    inputs: ReconstructionInputs,
) -> dict[str, torch.Tensor]:
    coordinates = {}
    for view_index, (name, field) in enumerate(
        zip(inputs.view_names, inputs.observations, strict=True)
    ):
        generator = torch.Generator(device="cpu").manual_seed(OCCUPANCY_SEED + view_index)
        x = torch.randint(
            field.width,
            (OCCUPANCY_SAMPLES_PER_VIEW,),
            generator=generator,
        )
        y = torch.randint(
            field.height,
            (OCCUPANCY_SAMPLES_PER_VIEW,),
            generator=generator,
        )
        coordinates[name] = torch.stack([x, y], dim=-1).to(field.dtype) + 0.5
    return coordinates


def aggregate_records(
    records: list[dict[str, Any]],
    scores: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict[str, Any]:
    pooled = base._binary_metrics(torch.cat(scores), torch.cat(targets))
    return {
        "records": records,
        "pooled": pooled,
        "macro_mean_iou": sum(item["iou"] for item in records) / len(records),
        "macro_mean_auc": sum(item["auc"] for item in records) / len(records),
        "macro_mean_precision": sum(item["precision"] for item in records) / len(records),
        "macro_mean_recall": sum(item["recall"] for item in records) / len(records),
    }


def occupancy_audit_subset(
    inputs: ReconstructionInputs,
    masks: list[torch.Tensor],
    backends_by_variant: dict[str, list[base.CoverageOverrideBackend]],
    coordinates: dict[str, torch.Tensor],
    selected_views: tuple[str, ...],
) -> dict[str, Any]:
    view_to_index = {name: index for index, name in enumerate(inputs.view_names)}
    result = {}
    for variant, backends in backends_by_variant.items():
        records = []
        scores = []
        targets = []
        for name in selected_views:
            view_index = view_to_index[name]
            xy = coordinates[name]
            relative_density = backends[view_index].relative_density(xy, component_chunk=64)
            soft_coverage = 1.0 - torch.exp(-relative_density)
            target = bilinear_sample(masks[view_index], xy) > 0.5
            metrics = base._binary_metrics(soft_coverage, target)
            metrics.update(
                {
                    "view": name,
                    "xy_sha256": base.tensor_hash(xy),
                    "coverage_sha256": base.tensor_hash(soft_coverage),
                    "target_sha256": base.tensor_hash(target),
                }
            )
            records.append(metrics)
            scores.append(soft_coverage)
            targets.append(target)
        result[variant] = aggregate_records(records, scores, targets)
    return result


def audit_variants_sequentially(
    *,
    inputs: ReconstructionInputs,
    masks: list[torch.Tensor],
    anchor_indexes: list[GaussianObservationIndex],
    variant_fields: dict[str, list[GaussianObservationField]],
    coordinates: dict[str, torch.Tensor],
    selected_views: tuple[str, ...],
    config: Any,
    audit_contract: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = {}
    contracts = {}
    for variant, fields in variant_fields.items():
        proxy_indexes, backends = build_single_variant_backends(
            inputs,
            anchor_indexes,
            fields,
            masks,
            config,
        )
        if audit_contract:
            contracts[variant] = base.backend_contract_audit(
                inputs,
                anchor_indexes,
                {variant: backends},
            )[variant]
        metrics[variant] = occupancy_audit_subset(
            inputs,
            masks,
            {variant: backends},
            coordinates,
            selected_views,
        )[variant]
        del backends, proxy_indexes
        gc.collect()
    return metrics, contracts


def select_stage_b_variants(tuning: dict[str, Any]) -> dict[str, Any]:
    center_precision = tuning["center"]["pooled"]["precision"]
    precision_floor = center_precision - PRECISION_GUARD_DROP
    candidates = []
    rejected = []
    for beta in LSE_BETAS:
        name = f"lse_beta_{beta}"
        metrics = tuning[name]["pooled"]
        record = {
            "variant": name,
            "beta": beta,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "iou": metrics["iou"],
            "auc": metrics["auc"],
        }
        if metrics["precision"] >= precision_floor:
            candidates.append(record)
        else:
            rejected.append(record)
    if not candidates:
        raise RuntimeError("no normalized-LSE variant passed the frozen precision guard")
    selected = max(
        candidates,
        key=lambda item: (item["recall"], item["iou"], item["auc"], -item["beta"]),
    )
    hard = tuning["hard_max"]["pooled"]
    hard_max_useful = hard["precision"] >= precision_floor and (
        hard["recall"] >= selected["recall"] + HARD_MAX_RECALL_FLOOR
        or hard["iou"] >= selected["iou"] + HARD_MAX_IOU_FLOOR
    )
    stage_b = ["center", "mean", selected["variant"]]
    if hard_max_useful:
        stage_b.append("hard_max")
    return {
        "selection_schema": "compact_occupancy_scalar_ablation.selection.v1",
        "selection_timestamp_utc": timestamp_utc(),
        "tuning_views": list(TUNING_VIEWS),
        "report_metrics_not_passed_to_selector": list(REPORT_VIEWS),
        "center_pooled_precision": center_precision,
        "precision_guard_drop": PRECISION_GUARD_DROP,
        "precision_floor": precision_floor,
        "objective_order": [
            "maximum pooled recall",
            "maximum pooled IoU",
            "maximum pooled AUC",
            "lower beta",
        ],
        "eligible_lse_candidates": candidates,
        "rejected_lse_candidates": rejected,
        "selected_lse": selected,
        "hard_max_rule": {
            "precision_guard_required": True,
            "minimum_recall_gain": HARD_MAX_RECALL_FLOOR,
            "minimum_iou_gain": HARD_MAX_IOU_FLOOR,
            "included": hard_max_useful,
            "metrics": {
                "precision": hard["precision"],
                "recall": hard["recall"],
                "iou": hard["iou"],
                "auc": hard["auc"],
            },
        },
        "stage_b_variants": stage_b,
    }


def support_failure_record(
    error: Exception,
    inputs: ReconstructionInputs,
    backends: list[Any],
    config: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    support_failure = isinstance(error, ValueError) and (
        "fewer globally supported ray placements" in str(error)
        or "proposed fewer anchors than n_init_3d" in str(error)
    )
    diagnostics = None
    diagnostic_failure = None
    if support_failure:
        try:
            diagnostics = base.diagnose_candidate_support(inputs, backends, config)
        except Exception as diagnostic_error:
            diagnostic_failure = {
                "failure_type": type(diagnostic_error).__name__,
                "failure_message": str(diagnostic_error),
            }
    bounded_infeasible = (
        support_failure
        and diagnostics is not None
        and (
            diagnostics["eligible_candidate_count"] is None
            or diagnostics["eligible_candidate_count"] < config.n_init_3d
        )
    )
    return diagnostics, diagnostic_failure, bounded_infeasible


def run_stage_b_arm(
    *,
    variant: str,
    output: Path,
    inputs: ReconstructionInputs,
    masks: list[torch.Tensor],
    anchor_indexes: list[GaussianObservationIndex],
    backends: list[base.CoverageOverrideBackend],
    config: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    arm_dir = output / "stage_b" / variant
    arm_dir.mkdir()
    try:
        initialization = CompactCarveInitializer(config).initialize(inputs, backends=backends)
        initial_path = arm_dir / "gaussians_init.ply"
        final_path = arm_dir / "gaussians.ply"
        initialization.gaussians.save_ply(initial_path)
        initialization.gaussians.save_ply(final_path)
        if initial_path.read_bytes() != final_path.read_bytes():
            raise RuntimeError("unrefined Stage-B PLY copies differ")
        teacher_metrics = base.point_mse(
            initialization.gaussians,
            inputs,
            anchor_indexes,
            masks,
        )
        projection = base.foreground_projection_metrics(initialization, inputs, masks)
        unique_pairs = (
            torch.stack(
                [
                    initialization.lineage.source_view_indices,
                    initialization.lineage.source_component_indices,
                ],
                dim=1,
            )
            .unique(dim=0)
            .shape[0]
        )
        record = {
            "status": "PASS",
            "variant": variant,
            "config": dataclasses.asdict(config),
            "diagnostics": initialization.diagnostics,
            "selected_unique_source_component_pairs": int(unique_pairs),
            "selected_duplicate_source_component_count": int(
                initialization.n_init_3d - unique_pairs
            ),
            "score_quantiles": {
                str(quantile): float(torch.quantile(initialization.scores, quantile))
                for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
            },
            "foreground_projection": projection,
            "compact_teacher_point_metric_source": "exact_masked_teacher_anchor_bundle",
            "compact_teacher_point_metrics": teacher_metrics,
            "artifacts": {
                "gaussians_init": initial_path.relative_to(output).as_posix(),
                "gaussians_init_sha256": base.sha256_file(initial_path),
                "gaussians": final_path.relative_to(output).as_posix(),
                "gaussians_sha256": base.sha256_file(final_path),
            },
            "wall_seconds": time.perf_counter() - started,
        }
        if variant == "center":
            replay = Gaussians3D.load_ply(initial_path)
            reference = Gaussians3D.load_ply(PRIOR_CENTER_PLY)
            replay_record = {
                "scope": (
                    "cross-run diagnostic under a newly pinned one-thread environment; exact "
                    "proxy tensors/config are audited separately"
                ),
                "reference_path": PRIOR_CENTER_PLY.relative_to(ROOT).as_posix(),
                "reference_sha256": base.sha256_file(PRIOR_CENTER_PLY),
                "byte_exact": initial_path.read_bytes() == PRIOR_CENTER_PLY.read_bytes(),
                "means_max_abs": float((replay.means - reference.means).abs().max()),
                "covariance_max_abs": float(
                    (replay.covariance() - reference.covariance()).abs().max()
                ),
                "opacity_max_abs": float((replay.opacity - reference.opacity).abs().max()),
                "sh_max_abs": float((replay.sh - reference.sh).abs().max()),
                "means_tolerance": CENTER_REPLAY_MEANS_TOLERANCE,
                "covariance_tolerance": CENTER_REPLAY_COVARIANCE_TOLERANCE,
            }
            replay_record["within_declared_geometric_tolerance"] = (
                replay_record["means_max_abs"] <= CENTER_REPLAY_MEANS_TOLERANCE
                and replay_record["covariance_max_abs"] <= CENTER_REPLAY_COVARIANCE_TOLERANCE
            )
            replay_record["exact_cross_run_replay_claimed"] = False
            record["prior_center_replay"] = replay_record
            if not replay_record["within_declared_geometric_tolerance"]:
                record.update(
                    {
                        "status": "FAIL",
                        "failure_type": "PriorCenterReplayMismatch",
                        "failure_message": (
                            "center arm exceeded the predeclared cross-run geometric tolerance"
                        ),
                    }
                )
    except Exception as error:
        diagnostics, diagnostic_failure, bounded_infeasible = support_failure_record(
            error,
            inputs,
            backends,
            config,
        )
        record = {
            "status": "INFEASIBLE" if bounded_infeasible else "FAIL",
            "variant": variant,
            "config": dataclasses.asdict(config),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "failure_diagnostics": diagnostics,
            "diagnostic_failure": diagnostic_failure,
            "wall_seconds": time.perf_counter() - started,
        }
    base.write_json(arm_dir / "result.json", record)
    return record


def run(output: Path, anchor_bundle: Path) -> dict[str, Any]:
    thread_binding = pin_cpu_threads()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scalar ablation output: {output}")
    source_before = source_binding()
    exact_inputs = ReconstructionInputs.load(base.EXACT_BUNDLE, strict=True)
    anchor_inputs = ReconstructionInputs.load(anchor_bundle, strict=True)
    anchor_records = base.validate_anchor_inputs(anchor_inputs, exact_inputs)
    if set(TUNING_VIEWS).intersection(REPORT_VIEWS):
        raise RuntimeError("tuning and report view sets overlap")
    if set((*TUNING_VIEWS, *REPORT_VIEWS)) != set(anchor_inputs.view_names):
        raise RuntimeError("tuning/report split does not exactly cover the frozen seven views")
    anchor_binding_before = base.bundle_binding(anchor_bundle)
    exact_binding_before = base.bundle_binding(base.EXACT_BUNDLE)
    calibration_before = base.sha256_file(base.CALIBRATION)
    exact_plan_before = base.sha256_file(base.EXACT_PLAN)
    prior_result_before = base.sha256_file(PRIOR_SCREEN / "result.json")
    prior_center_before = base.sha256_file(PRIOR_CENTER_PLY)
    prior_proxy_before = base.bundle_binding(PRIOR_CENTER_PROXY_BUNDLE)
    preregistration_before = base.sha256_file(PREREGISTRATION)
    implementation_review_before = base.sha256_file(IMPLEMENTATION_REVIEW)
    raw_masks_before = raw_mask_bindings(anchor_inputs.view_names)
    base_config = dataclasses.replace(base.base_carve_config(), anchor_mode="component_centers")
    stage_b_inputs, center_extent_before = freeze_stage_b_center_extent(
        anchor_inputs,
        base_config,
    )
    standard = footprint_standard_samples(anchor_inputs.observations[0].dtype)
    output.mkdir(parents=True)
    (output / "stage_b").mkdir()
    plan = {
        "artifact_type": "compact_occupancy_scalar_ablation_plan_v1",
        "timestamp_utc": timestamp_utc(),
        "sealed_before_mask_decode": True,
        "decision_bearing": False,
        "scope": "exploratory real-data Stage-2 occupancy-scalar development ablation",
        "source_binding": source_before,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cpu_thread_binding": thread_binding,
        },
        "inputs": {
            "exact_masked_teacher_anchor_bundle": anchor_binding_before,
            "frozen_full_frame_bundle_for_calibration_validation_only": exact_binding_before,
            "calibration": {
                "path": base.CALIBRATION.relative_to(ROOT).as_posix(),
                "sha256": calibration_before,
            },
            "exact_compact_carve_plan": {
                "path": base.EXACT_PLAN.relative_to(ROOT).as_posix(),
                "sha256": exact_plan_before,
            },
            "prior_screen_result": {
                "path": (PRIOR_SCREEN / "result.json").relative_to(ROOT).as_posix(),
                "sha256": prior_result_before,
            },
            "prior_center_ply": {
                "path": PRIOR_CENTER_PLY.relative_to(ROOT).as_posix(),
                "sha256": prior_center_before,
            },
            "prior_center_proxy_bundle": prior_proxy_before,
            "protocol_artifacts": {
                "preregistration": {
                    "path": PREREGISTRATION.relative_to(ROOT).as_posix(),
                    "sha256": preregistration_before,
                },
                "implementation_review": {
                    "path": IMPLEMENTATION_REVIEW.relative_to(ROOT).as_posix(),
                    "sha256": implementation_review_before,
                },
            },
            "masks_not_decoded_at_seal": raw_masks_before,
            "source_rgb_policy": "forbidden/not decoded; compact masked field colors only",
        },
        "anchor_validation": anchor_records,
        "configuration": {
            "base_compact_carve": dataclasses.asdict(base_config),
            "frozen_center_extent": center_extent_before,
            "footprint_samples": FOOTPRINT_SAMPLES,
            "footprint_seed": FOOTPRINT_SEED,
            "standard_samples_sha256": base.tensor_hash(standard),
            "sampling": "16 scrambled Sobol N(0,I) samples plus their 16 antithetic pairs",
            "variants": list(VARIANT_NAMES),
            "normalized_lse_formula": "(logsumexp(beta*m_k)-log(K))/beta",
            "lse_betas": list(LSE_BETAS),
            "occupancy_seed": OCCUPANCY_SEED,
            "occupancy_samples_per_view": OCCUPANCY_SAMPLES_PER_VIEW,
            "coverage_threshold": 0.4,
            "tuning_views": list(TUNING_VIEWS),
            "selection_report_views": list(REPORT_VIEWS),
            "split_scope_caveat": (
                "report views are unseen only by scalar-temperature selection; their per-view "
                "fields are Stage-1 fitted and Stage B consumes all seven views"
            ),
            "precision_guard_drop": PRECISION_GUARD_DROP,
            "selection_objective": "max tune pooled recall, then IoU, AUC, lower beta",
            "hard_max_stage_b_rule": {
                "passes_precision_guard": True,
                "recall_gain_floor": HARD_MAX_RECALL_FLOOR,
                "iou_gain_floor": HARD_MAX_IOU_FLOOR,
                "combination": "recall OR IoU",
            },
            "stage_b_fixed_variants": ["center", "mean", "selected_lse"],
            "hard_max_role": "non-smooth ceiling only when frozen usefulness rule passes",
            "per_view_mass_normalization_caveat": (
                "coverage uses r=A*D/M independently per view, so multiplying every component "
                "scalar in one view by a common positive constant cancels; this ablation tests "
                "relative per-component occupancy distribution, not absolute scalar scale"
            ),
            "center_cross_run_replay": {
                "exact_replay_claimed": False,
                "means_tolerance": CENTER_REPLAY_MEANS_TOLERANCE,
                "covariance_tolerance": CENTER_REPLAY_COVARIANCE_TOLERANCE,
                "reason": "prior audit observed CPU reduction/near-tie nondeterminism",
                "exact_proxy_tensor_replay_required": True,
            },
            "stage_b_center_extent_intervention": (
                "all arms receive one ReconstructionInputs clone with the one-time center and "
                "extent injected as bounds_hint; no arm reruns camera-only torch.linalg.lstsq"
            ),
        },
    }
    plan["plan_sha256"] = base.canonical_hash(plan)
    base.write_json(output / "plan.json", plan)

    # Everything below this point is post-seal observed data.
    masks, mask_bindings = base.load_masks(exact_inputs)
    variant_fields, scalar_records, observed_standard = build_proxy_variants(
        stage_b_inputs,
        masks,
    )
    if not torch.equal(standard, observed_standard):
        raise RuntimeError("footprint sample construction changed after the outcome-free seal")
    center_proxy_audit = center_proxy_replay_audit(
        variant_fields["center"],
        PRIOR_CENTER_PROXY_BUNDLE,
    )
    proxy_bindings_before = save_proxy_bundles(output, stage_b_inputs, variant_fields)
    observed_metadata = {
        "artifact_type": "compact_occupancy_scalar_observed_metadata_v1",
        "plan_sha256": plan["plan_sha256"],
        "mask_tensor_bindings": mask_bindings,
        "scalar_tensor_bindings": scalar_records,
        "proxy_bundle_bindings": proxy_bindings_before,
        "center_proxy_replay_audit": center_proxy_audit,
    }
    observed_metadata["artifact_sha256"] = base.canonical_hash(observed_metadata)
    base.write_json(output / "observed_metadata.json", observed_metadata)

    anchor_indexes = [
        GaussianObservationIndex(
            field,
            tile_size=base_config.tile_size,
            max_entries=base_config.max_index_entries_per_view,
            max_candidates=base_config.max_candidates_per_tile,
        )
        for field in stage_b_inputs.observations
    ]
    coordinates = coordinates_for_views(stage_b_inputs)

    with base.deny_source_rgb_open() as denied:
        tuning, contract = audit_variants_sequentially(
            inputs=stage_b_inputs,
            masks=masks,
            anchor_indexes=anchor_indexes,
            variant_fields=variant_fields,
            coordinates=coordinates,
            selected_views=TUNING_VIEWS,
            config=base_config,
            audit_contract=True,
        )
        base.write_json(output / "backend_contract_audit.json", contract)
        tuning_artifact = {
            "artifact_type": "compact_occupancy_scalar_tuning_v1",
            "plan_sha256": plan["plan_sha256"],
            "views": list(TUNING_VIEWS),
            "metrics": tuning,
        }
        tuning_artifact["artifact_sha256"] = base.canonical_hash(tuning_artifact)
        base.write_json(output / "stage_a_tuning.json", tuning_artifact)
        selection = select_stage_b_variants(tuning)
        selection["plan_sha256"] = plan["plan_sha256"]
        selection["tuning_artifact_sha256"] = tuning_artifact["artifact_sha256"]
        selection["selection_sha256"] = base.canonical_hash(selection)
        base.write_json(output / "selection.json", selection)

        report, _ = audit_variants_sequentially(
            inputs=stage_b_inputs,
            masks=masks,
            anchor_indexes=anchor_indexes,
            variant_fields=variant_fields,
            coordinates=coordinates,
            selected_views=REPORT_VIEWS,
            config=base_config,
            audit_contract=False,
        )
        report_artifact = {
            "artifact_type": "compact_occupancy_scalar_selection_report_v1",
            "plan_sha256": plan["plan_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "views": list(REPORT_VIEWS),
            "selection_independent": True,
            "metrics": report,
        }
        report_artifact["artifact_sha256"] = base.canonical_hash(report_artifact)
        base.write_json(output / "stage_a_selection_report.json", report_artifact)

        stage_b = {}
        for variant in selection["stage_b_variants"]:
            proxy_indexes, backends = build_single_variant_backends(
                stage_b_inputs,
                anchor_indexes,
                variant_fields[variant],
                masks,
                base_config,
            )
            stage_b[variant] = run_stage_b_arm(
                variant=variant,
                output=output,
                inputs=stage_b_inputs,
                masks=masks,
                anchor_indexes=anchor_indexes,
                backends=backends,
                config=base_config,
            )
            del backends, proxy_indexes
            gc.collect()

    if denied["source_rgb_open_attempts"] != 0:
        raise RuntimeError("scalar ablation attempted forbidden source RGB access")
    if source_binding() != source_before:
        raise RuntimeError("bound scalar-ablation source changed during execution")
    if base.bundle_binding(anchor_bundle) != anchor_binding_before:
        raise RuntimeError("exact masked anchor bundle changed during execution")
    if base.bundle_binding(base.EXACT_BUNDLE) != exact_binding_before:
        raise RuntimeError("calibration-validation bundle changed during execution")
    if base.bundle_binding(PRIOR_CENTER_PROXY_BUNDLE) != prior_proxy_before:
        raise RuntimeError("prior center proxy bundle changed during execution")
    for variant, binding in proxy_bindings_before.items():
        if base.bundle_binding(output / "proxy_bundles" / variant) != binding:
            raise RuntimeError(f"derived proxy bundle changed during execution: {variant}")
    if base.sha256_file(base.CALIBRATION) != calibration_before:
        raise RuntimeError("mask calibration changed during execution")
    if base.sha256_file(base.EXACT_PLAN) != exact_plan_before:
        raise RuntimeError("compact-carve plan changed during execution")
    if base.sha256_file(PRIOR_SCREEN / "result.json") != prior_result_before:
        raise RuntimeError("prior Stage-2 result changed during execution")
    if base.sha256_file(PRIOR_CENTER_PLY) != prior_center_before:
        raise RuntimeError("prior center PLY changed during execution")
    if base.sha256_file(PREREGISTRATION) != preregistration_before:
        raise RuntimeError("preregistration changed during execution")
    if base.sha256_file(IMPLEMENTATION_REVIEW) != implementation_review_before:
        raise RuntimeError("implementation review changed during execution")
    center_extent_after = center_extent_binding(stage_b_inputs, base_config)
    if any(center_extent_before[key] != value for key, value in center_extent_after.items()):
        raise RuntimeError("frozen center/extent changed during execution")
    if pin_cpu_threads() != thread_binding:
        raise RuntimeError("PyTorch CPU thread binding changed during execution")
    for record in mask_bindings.values():
        if base.sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError("source mask changed during execution")

    center_passed = stage_b["center"]["status"] == "PASS"
    selected_name = selection["selected_lse"]["variant"]
    selected_passed = stage_b[selected_name]["status"] == "PASS"
    control_statuses_allowed = all(
        record["status"] in {"PASS", "INFEASIBLE"}
        for name, record in stage_b.items()
        if name not in {"center", selected_name}
    )
    status = "PASS" if center_passed and selected_passed and control_statuses_allowed else "FAIL"
    result = {
        "artifact_type": "compact_occupancy_scalar_ablation_result_v1",
        "timestamp_utc": timestamp_utc(),
        "status": status,
        "decision_bearing": False,
        "result_scope": (
            "exploratory seven-view masked-field occupancy-scalar screen and unrefined "
            "component-center initialization; not novel-view or refinement evidence"
        ),
        "plan_sha256": plan["plan_sha256"],
        "observed_metadata_sha256": observed_metadata["artifact_sha256"],
        "source_binding": source_before,
        "source_rgb_denial": denied,
        "backend_contract_audit": contract,
        "stage_a": {
            "tuning_artifact_sha256": tuning_artifact["artifact_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "selection_report_artifact_sha256": report_artifact["artifact_sha256"],
            "selected_lse": selection["selected_lse"],
            "hard_max_included": selection["hard_max_rule"]["included"],
        },
        "stage_b": stage_b,
        "completion_policy": {
            "center_replay_required": True,
            "selected_lse_pass_required": True,
            "mean_and_optional_hard_max_allow_bounded_infeasibility": True,
            "center_passed": center_passed,
            "selected_lse_passed": selected_passed,
            "control_statuses_allowed": control_statuses_allowed,
        },
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "viewer": {
            "status": "DEFERRED_TO_ROOT_EXACT_GSPLAT",
            "reason": "root agent will inspect selected Stage-B PLY after results audit",
        },
    }
    result["result_sha256"] = base.canonical_hash(result)
    base.write_json(output / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--anchor-bundle", type=Path, default=ANCHOR_BUNDLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.out.resolve(), args.anchor_bundle.resolve())
    summary = {
        "status": result["status"],
        "output": str(args.out.resolve()),
        "selected_lse": result["stage_a"]["selected_lse"],
        "hard_max_included": result["stage_a"]["hard_max_included"],
        "stage_b": {
            name: {
                "status": record["status"],
                "eligible": (
                    record.get("diagnostics") or record.get("failure_diagnostics") or {}
                ).get("eligible_candidate_count"),
                "background_all": record.get("foreground_projection", {}).get(
                    "background_in_all_views_fraction"
                ),
                "foreground_ge2": record.get("foreground_projection", {}).get(
                    "foreground_in_at_least_2_views_fraction"
                ),
                "foreground_psnr_db": record.get("compact_teacher_point_metrics", {})
                .get("pooled", {})
                .get("foreground", {})
                .get("psnr_db"),
            }
            for name, record in result["stage_b"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
