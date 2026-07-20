#!/usr/bin/env python3
"""Exploratory compact Stage-2 anchor/occupancy screen on calibrated full-resolution data.

The screen keeps the seven lossless full-frame StructSplat fields only as a frozen replay control.
Its four repaired-pipeline arms use a caller-supplied exact masked ``ReconstructionInputs`` bundle
for both source anchors and queried compact RGB/color.  Those arms compare density/mass-random
against component-center anchors, then replace only the coverage statistic with either dense
binary silhouettes (an RGB-free upper-bound control) or a compact Gaussian occupancy proxy
derived from the exact masked fields.  No source RGB file is decoded.

This is a reproducible exploratory mechanism screen, not a sealed/default-changing benchmark.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    ObservationQuery,
)
from rtgs.data.calibrated import _resize_image, _undistort
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.base import bilinear_sample
from rtgs.lift.compact_carve import (
    CompactCarveConfig,
    CompactCarveInitializer,
    _center_and_extent,
    _propose_anchors,
    _ray_box,
    score_world_points,
)
from rtgs.render.torch_points import TorchPointRasterizer

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
EXACT_BUNDLE = ROOT / "runs/compact_point_training_20260716/reconstruction_inputs"
EXACT_PLAN = ROOT / "runs/compact_point_training_20260716/CALIBRATED_PLAN.json"
EXACT_BASELINE_PLY = ROOT / "runs/compact_point_training_20260716/gaussians_init.ply"
DEFAULT_OUT = ROOT / "runs/compact_masked_lift_screen_20260717"
TRAIN_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
TRUE_MASK_TARGET_COVERAGE = 0.999
MASK_STRENGTH = -math.log1p(-TRUE_MASK_TARGET_COVERAGE)
OCCUPANCY_SAMPLES_PER_VIEW = 16_384
TEACHER_SAMPLES_PER_VIEW = 4_096
FOREGROUND_SAMPLES_PER_VIEW = 1_024
SEED = 75300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_binding() -> dict[str, Any]:
    paths = (
        Path("benchmarks/compact_masked_lift_screen.py"),
        Path("src/rtgs/core/gaussians2d.py"),
        Path("src/rtgs/core/gaussians3d.py"),
        Path("src/rtgs/core/camera.py"),
        Path("src/rtgs/core/observation2d.py"),
        Path("src/rtgs/core/sh.py"),
        Path("src/rtgs/data/calibrated.py"),
        Path("src/rtgs/data/reconstruction_inputs.py"),
        Path("src/rtgs/data/scene.py"),
        Path("src/rtgs/lift/base.py"),
        Path("src/rtgs/lift/compact_carve.py"),
        Path("src/rtgs/render/base.py"),
        Path("src/rtgs/render/point_base.py"),
        Path("src/rtgs/render/torch_ref.py"),
        Path("src/rtgs/render/torch_points.py"),
    )
    hashes = {path.as_posix(): sha256_file(ROOT / path) for path in paths}
    status = git_output(
        "status",
        "--porcelain=v1",
        "--",
        *(path.as_posix() for path in paths),
    )
    return {
        "git_revision": git_output("rev-parse", "HEAD"),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "files": hashes,
        "aggregate": canonical_hash(hashes),
    }


def camera_distortions() -> dict[str, list[float]]:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    result = {}
    for record in payload["cameras"]:
        camera_id = str(record["camera_id"]).upper()
        result[camera_id] = [
            float(value) for value in record["intrinsics"].get("distortion_coefficients", [])
        ]
    return result


def load_masks(inputs: ReconstructionInputs) -> tuple[list[torch.Tensor], dict[str, Any]]:
    distortions = camera_distortions()
    masks = []
    bindings = {}
    for name, camera in zip(inputs.view_names, inputs.cameras, strict=True):
        path = SCENE / "mask" / f"mask_{name}.png"
        mask = _undistort(
            _resize_image(path, camera.width, camera.height, mask=True),
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            distortions[name],
            mask=True,
        ).float()
        masks.append(mask)
        bindings[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "tensor_sha256": tensor_hash(mask),
            "foreground_fraction": float((mask > 0.5).float().mean()),
        }
    return masks, bindings


def _field_mass(field: GaussianObservationField) -> torch.Tensor:
    return (
        field.amplitudes * (2.0 * math.pi) * field.effective_variances().prod(dim=1).sqrt()
    ).sum()


def build_occupancy_proxies(
    anchor_inputs: ReconstructionInputs,
    masks: list[torch.Tensor],
) -> tuple[list[GaussianObservationField], list[dict[str, Any]]]:
    conversion = {
        "schema": "compact_masked_lift_screen.occupancy_proxy.v1",
        "source_semantics": "exact masked GaussianObservationField geometry",
        "occupancy_amplitude": "bilinear undistorted mask at component center",
        "occupancy_color": [1.0, 1.0, 1.0],
        "retained_geometry_and_support": True,
    }
    conversion_digest = canonical_hash(conversion)
    proxies = []
    records = []
    for name, anchor, mask in zip(
        anchor_inputs.view_names,
        anchor_inputs.observations,
        masks,
        strict=True,
    ):
        occupancy_amplitudes = bilinear_sample(mask, anchor.means).clamp(0.0, 1.0)
        source_digest = canonical_hash(
            {
                "view": name,
                "means": tensor_hash(anchor.means),
                "log_scales": tensor_hash(anchor.log_scales),
                "rotations": tensor_hash(anchor.rotations),
                "mask": tensor_hash(mask),
            }
        )
        proxy = GaussianObservationField(
            width=anchor.width,
            height=anchor.height,
            means=anchor.means,
            log_scales=anchor.log_scales,
            rotations=anchor.rotations,
            colors=anchor.colors.new_ones((anchor.n, 3)),
            amplitudes=occupancy_amplitudes,
            color_grads=None,
            filter_variance=anchor.filter_variance,
            blend_mode="normalized",
            epsilon=anchor.epsilon,
            sigma_cutoff=anchor.sigma_cutoff,
            support_fade_alpha=anchor.support_fade_alpha,
            aa_dilation=anchor.aa_dilation,
            view_id=name,
            fit_window=anchor.fit_window,
            n_init=anchor.n_init,
            provider="synthetic_fixture",
            producer_version="exact-masked-center-gated-occupancy-proxy-v1",
            producer_source_digest=source_digest,
            fit_config_digest=conversion_digest,
        )
        proxies.append(proxy)
        records.append(
            {
                "view": name,
                "fit_window": list(anchor.fit_window),
                "components": anchor.n,
                "source_provider": anchor.provider,
                "source_producer_version": anchor.producer_version,
                "source_semantic_hashes": {
                    "means": tensor_hash(anchor.means),
                    "log_scales": tensor_hash(anchor.log_scales),
                    "rotations": tensor_hash(anchor.rotations),
                    "amplitudes": tensor_hash(anchor.amplitudes),
                },
                "occupancy_positive_components": int((occupancy_amplitudes > 0.5).sum()),
                "occupancy_positive_component_fraction": float(
                    (occupancy_amplitudes > 0.5).float().mean()
                ),
                "occupancy_amplitude_sum": float(occupancy_amplitudes.sum()),
            }
        )
    return proxies, records


CoverageKind = Literal["exact_density", "true_mask", "compact_proxy"]


class CoverageOverrideBackend:
    """Return the bound compact teacher RGB while replacing only normalized coverage density."""

    def __init__(
        self,
        *,
        exact_field: GaussianObservationField,
        exact_backend: GaussianObservationIndex,
        normalization_field: GaussianObservationField,
        kind: CoverageKind,
        mask: torch.Tensor | None = None,
        proxy_field: GaussianObservationField | None = None,
        proxy_backend: GaussianObservationIndex | None = None,
    ):
        self.exact_field = exact_field
        self.exact_backend = exact_backend
        self.normalization_field = normalization_field
        self.kind = kind
        self.mask = mask
        self.proxy_field = proxy_field
        self.proxy_backend = proxy_backend
        self._normalization_mass = _field_mass(normalization_field)
        self._normalization_area = float(
            normalization_field.fit_window[2] * normalization_field.fit_window[3]
        )
        if kind == "true_mask" and mask is None:
            raise ValueError("true_mask coverage requires a mask")
        if kind == "compact_proxy" and (proxy_field is None or proxy_backend is None):
            raise ValueError("compact_proxy coverage requires a field and backend")

    def relative_density(self, xy: torch.Tensor, component_chunk: int = 4096) -> torch.Tensor:
        if self.kind == "true_mask":
            return MASK_STRENGTH * bilinear_sample(self.mask, xy).clamp(0.0, 1.0)
        if self.kind == "compact_proxy":
            query = self.proxy_backend.query(xy, component_chunk=component_chunk)
            proxy_area = float(self.proxy_field.fit_window[2] * self.proxy_field.fit_window[3])
            proxy_mass = _field_mass(self.proxy_field).clamp_min(
                torch.finfo(self.proxy_field.dtype).tiny
            )
            return proxy_area * query.weight_sum / proxy_mass
        query = self.exact_backend.query(xy, component_chunk=component_chunk)
        exact_area = float(self.exact_field.fit_window[2] * self.exact_field.fit_window[3])
        exact_mass = _field_mass(self.exact_field).clamp_min(
            torch.finfo(self.exact_field.dtype).tiny
        )
        return exact_area * query.weight_sum / exact_mass

    def query(self, xy: torch.Tensor, component_chunk: int = 4096) -> ObservationQuery:
        exact = self.exact_backend.query(xy, component_chunk=component_chunk)
        if self.kind == "exact_density":
            exact_area = float(self.exact_field.fit_window[2] * self.exact_field.fit_window[3])
            exact_mass = _field_mass(self.exact_field).clamp_min(
                torch.finfo(self.exact_field.dtype).tiny
            )
            relative_density = exact_area * exact.weight_sum / exact_mass
        else:
            relative_density = self.relative_density(xy, component_chunk=component_chunk)
        replacement = relative_density * self._normalization_mass / self._normalization_area
        return ObservationQuery(
            color=exact.color,
            numerator=exact.numerator,
            weight_sum=replacement,
            valid=exact.valid,
        )


def make_backends(
    *,
    color_inputs: ReconstructionInputs,
    normalization_inputs: ReconstructionInputs,
    color_indexes: list[GaussianObservationIndex],
    coverage_kind: CoverageKind,
    masks: list[torch.Tensor],
    proxy_fields: list[GaussianObservationField],
    proxy_indexes: list[GaussianObservationIndex],
) -> list[CoverageOverrideBackend]:
    return [
        CoverageOverrideBackend(
            exact_field=exact,
            exact_backend=exact_index,
            normalization_field=normalization,
            kind=coverage_kind,
            mask=mask,
            proxy_field=proxy,
            proxy_backend=proxy_index,
        )
        for exact, normalization, exact_index, mask, proxy, proxy_index in zip(
            color_inputs.observations,
            normalization_inputs.observations,
            color_indexes,
            masks,
            proxy_fields,
            proxy_indexes,
            strict=True,
        )
    ]


def backend_contract_audit(
    inputs: ReconstructionInputs,
    indexes: list[GaussianObservationIndex],
    backends_by_kind: dict[str, list[CoverageOverrideBackend]],
) -> dict[str, Any]:
    result = {}
    for kind, backends in backends_by_kind.items():
        records = []
        for name, field, index, backend in zip(
            inputs.view_names,
            inputs.observations,
            indexes,
            backends,
            strict=True,
        ):
            xy = field.means[: min(field.n, 128)]
            reference = index.query(xy, component_chunk=64)
            wrapped = backend.query(xy, component_chunk=64)
            record = {
                "view": name,
                "samples": xy.shape[0],
                "xy_sha256": tensor_hash(xy),
                "color_bit_exact": torch.equal(reference.color, wrapped.color),
                "numerator_bit_exact": torch.equal(reference.numerator, wrapped.numerator),
                "valid_bit_exact": torch.equal(reference.valid, wrapped.valid),
                "reference_color_sha256": tensor_hash(reference.color),
                "wrapped_color_sha256": tensor_hash(wrapped.color),
                "reference_numerator_sha256": tensor_hash(reference.numerator),
                "wrapped_numerator_sha256": tensor_hash(wrapped.numerator),
                "replacement_weight_sum_sha256": tensor_hash(wrapped.weight_sum),
            }
            record["preserves_all_noncoverage_outputs"] = all(
                record[key] for key in ("color_bit_exact", "numerator_bit_exact", "valid_bit_exact")
            )
            records.append(record)
        passed = all(record["preserves_all_noncoverage_outputs"] for record in records)
        result[kind] = {"status": "PASS" if passed else "FAIL", "records": records}
    if any(record["status"] != "PASS" for record in result.values()):
        raise RuntimeError(
            "coverage wrapper changed a compact teacher output other than weight_sum"
        )
    return result


def diagnose_candidate_support(
    inputs: ReconstructionInputs,
    backends: list[Any],
    config: CompactCarveConfig,
) -> dict[str, Any]:
    """Replay proposal/scoring far enough to retain eligibility after initializer failure."""
    dtype = inputs.observations[0].dtype
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    view_ids, _, xy, anchor_attempts, proposed_per_view = _propose_anchors(
        inputs,
        config,
        generator,
    )
    candidate_count = int(view_ids.numel())
    if candidate_count < config.n_init_3d:
        return {
            "candidate_count": candidate_count,
            "proposed_candidates_per_view": proposed_per_view,
            "eligible_candidate_count": None,
            "anchor_attempt_count": anchor_attempts,
            "failure_stage": "proposal_budget",
        }

    center, extent = _center_and_extent(inputs, dtype)
    half = extent * config.bounds_scale
    lo = center - half
    hi = center + half
    best_scores = torch.zeros(candidate_count, dtype=dtype)
    valid_rays = torch.zeros(candidate_count, dtype=torch.bool)
    ray_batch = max(1, config.query_batch_size // config.samples_per_ray)
    steps = (torch.arange(config.samples_per_ray, dtype=dtype) + 0.5) / config.samples_per_ray
    for start in range(0, candidate_count, ray_batch):
        end = min(start + ray_batch, candidate_count)
        local_views = view_ids[start:end]
        local_xy = xy[start:end]
        local_origins = torch.empty(end - start, 3, dtype=dtype)
        local_directions = torch.empty(end - start, 3, dtype=dtype)
        for view_index in local_views.unique(sorted=True).tolist():
            mask = local_views == view_index
            origin, direction = inputs.cameras[view_index].pixel_rays(local_xy[mask])
            local_origins[mask] = origin.to(dtype).expand(int(mask.sum()), -1)
            local_directions[mask] = direction.to(dtype)
        t0, t1 = _ray_box(local_origins, local_directions, lo, hi)
        t0 = t0.clamp_min(config.near)
        ray_valid = t1 > t0
        ts = t0[:, None] + (t1 - t0).clamp_min(0)[:, None] * steps[None, :]
        world = local_origins[:, None, :] + ts[:, :, None] * local_directions[:, None, :]
        scores = score_world_points(
            inputs,
            world.reshape(-1, 3),
            config,
            backends,
        ).score.reshape(end - start, -1)
        scores = scores * ray_valid[:, None]
        best_scores[start:end] = scores.max(dim=1).values
        valid_rays[start:end] = ray_valid
    eligible = valid_rays & (best_scores > config.min_score)
    return {
        "candidate_count": candidate_count,
        "proposed_candidates_per_view": proposed_per_view,
        "eligible_candidate_count": int(eligible.sum()),
        "anchor_attempt_count": anchor_attempts,
        "valid_ray_count": int(valid_rays.sum()),
        "failure_stage": "global_support",
        "best_score_quantiles": {
            str(quantile): float(torch.quantile(best_scores, quantile))
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
    }


def _average_tie_auc(score: torch.Tensor, target: torch.Tensor) -> float:
    score = score.double().cpu()
    target = target.bool().cpu()
    positives = int(target.sum())
    negatives = target.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(score, stable=True)
    sorted_score = score[order]
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    ends = counts.cumsum(0).double()
    starts = ends - counts.double() + 1.0
    average_ranks = 0.5 * (starts + ends)
    sorted_ranks = torch.repeat_interleave(average_ranks, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    rank_sum = ranks[target].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _binary_metrics(score: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    prediction = score >= 0.4
    target = target.bool()
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "sample_count": target.numel(),
        "foreground_fraction": float(target.float().mean()),
        "positive_fraction": float(prediction.float().mean()),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "specificity": ratio(tn, tn + fp),
        "auc": _average_tie_auc(score, target),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def coverage_audit(
    inputs: ReconstructionInputs,
    masks: list[torch.Tensor],
    backends_by_kind: dict[str, list[CoverageOverrideBackend]],
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    coordinates = []
    for field in inputs.observations:
        x = torch.randint(field.width, (OCCUPANCY_SAMPLES_PER_VIEW,), generator=generator)
        y = torch.randint(field.height, (OCCUPANCY_SAMPLES_PER_VIEW,), generator=generator)
        coordinates.append(torch.stack([x, y], dim=-1).to(field.dtype) + 0.5)
    result = {}
    for kind, backends in backends_by_kind.items():
        records = []
        pooled_score = []
        pooled_target = []
        for name, xy, mask, backend in zip(
            inputs.view_names,
            coordinates,
            masks,
            backends,
            strict=True,
        ):
            relative_density = backend.relative_density(xy, component_chunk=64)
            soft_coverage = 1.0 - torch.exp(-relative_density)
            target = bilinear_sample(mask, xy) > 0.5
            metrics = _binary_metrics(soft_coverage, target)
            metrics.update(
                {
                    "view": name,
                    "xy_sha256": tensor_hash(xy),
                    "coverage_sha256": tensor_hash(soft_coverage),
                    "target_sha256": tensor_hash(target),
                }
            )
            records.append(metrics)
            pooled_score.append(soft_coverage)
            pooled_target.append(target)
        pooled = _binary_metrics(torch.cat(pooled_score), torch.cat(pooled_target))
        result[kind] = {
            "records": records,
            "pooled": pooled,
            "macro_mean_iou": sum(item["iou"] for item in records) / len(records),
            "macro_mean_auc": sum(item["auc"] for item in records) / len(records),
        }
    return result


def uniform_points(field: GaussianObservationField, count: int, generator: torch.Generator):
    x = torch.randint(field.width, (count,), generator=generator)
    y = torch.randint(field.height, (count,), generator=generator)
    return torch.stack([x, y], dim=-1).to(field.dtype) + 0.5


def foreground_points(
    field: GaussianObservationField,
    mask: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    accepted = []
    remaining = count
    for _ in range(128):
        if remaining <= 0:
            break
        candidates = uniform_points(field, max(4096, remaining * 8), generator)
        active = bilinear_sample(mask, candidates) > 0.5
        selected = candidates[active][:remaining]
        if selected.numel():
            accepted.append(selected)
            remaining -= selected.shape[0]
    if remaining:
        raise RuntimeError("foreground sampler did not collect the bounded requested count")
    return torch.cat(accepted)


def point_mse(
    model: Gaussians3D,
    teacher_inputs: ReconstructionInputs,
    teacher_indexes: list[GaussianObservationIndex],
    masks: list[torch.Tensor],
) -> dict[str, Any]:
    renderer = TorchPointRasterizer(point_chunk=32, gaussian_chunk=64)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 2)
    records = []
    uniform_sse = 0.0
    uniform_count = 0
    foreground_sse = 0.0
    foreground_count = 0
    for name, field, camera, index, mask in zip(
        teacher_inputs.view_names,
        teacher_inputs.observations,
        teacher_inputs.cameras,
        teacher_indexes,
        masks,
        strict=True,
    ):
        strata = {
            "uniform": uniform_points(field, TEACHER_SAMPLES_PER_VIEW, generator),
            "foreground": foreground_points(
                field,
                mask,
                FOREGROUND_SAMPLES_PER_VIEW,
                generator,
            ),
        }
        item = {"view": name, "strata": {}}
        for stratum, xy in strata.items():
            teacher = index.query(xy, component_chunk=64).color
            prediction = renderer.render_points(
                model,
                camera,
                xy,
                background=torch.zeros(3),
                sh_degree=0,
            ).color
            difference = prediction.double() - teacher.double()
            sse = float(difference.square().sum())
            scalar_count = int(difference.numel())
            mse = sse / scalar_count
            item["strata"][stratum] = {
                "samples": xy.shape[0],
                "xy_sha256": tensor_hash(xy),
                "teacher_sha256": tensor_hash(teacher),
                "prediction_sha256": tensor_hash(prediction),
                "sse": sse,
                "scalar_count": scalar_count,
                "mse": mse,
                "psnr_db": -10.0 * math.log10(max(mse, 1e-30)),
            }
            if stratum == "uniform":
                uniform_sse += sse
                uniform_count += scalar_count
            else:
                foreground_sse += sse
                foreground_count += scalar_count
        records.append(item)
    uniform_mse = uniform_sse / uniform_count
    foreground_mse = foreground_sse / foreground_count
    return {
        "records": records,
        "pooled": {
            "uniform": {
                "sse": uniform_sse,
                "scalar_count": uniform_count,
                "mse": uniform_mse,
                "psnr_db": -10.0 * math.log10(max(uniform_mse, 1e-30)),
            },
            "foreground": {
                "sse": foreground_sse,
                "scalar_count": foreground_count,
                "mse": foreground_mse,
                "psnr_db": -10.0 * math.log10(max(foreground_mse, 1e-30)),
            },
        },
    }


def foreground_projection_metrics(
    result: Any,
    inputs: ReconstructionInputs,
    masks: list[torch.Tensor],
) -> dict[str, Any]:
    hits = []
    per_view = []
    for name, camera, mask in zip(inputs.view_names, inputs.cameras, masks, strict=True):
        uv, depth = camera.project(result.gaussians.means)
        inside = camera.in_image(uv) & (depth > 0.05)
        foreground = inside & (bilinear_sample(mask, uv) > 0.5)
        hits.append(foreground)
        per_view.append(
            {
                "view": name,
                "foreground_count": int(foreground.sum()),
                "foreground_fraction": float(foreground.float().mean()),
            }
        )
    view_count = torch.stack(hits).sum(dim=0)
    source_hits = []
    for view_index in range(inputs.n_views):
        selected = result.lineage.source_view_indices == view_index
        source_xy = result.lineage.source_xy[selected]
        source_hits.append(
            inputs.cameras[view_index].in_image(source_xy)
            & (bilinear_sample(masks[view_index], source_xy) > 0.5)
        )
    source_hits_tensor = torch.cat(source_hits)
    return {
        "per_view": per_view,
        "foreground_view_count_histogram": torch.bincount(
            view_count, minlength=inputs.n_views + 1
        ).tolist(),
        "background_in_all_views_fraction": float((view_count == 0).float().mean()),
        "foreground_in_at_least_2_views_fraction": float((view_count >= 2).float().mean()),
        "foreground_in_at_least_6_views_fraction": float((view_count >= 6).float().mean()),
        "source_lineage_foreground_fraction": float(source_hits_tensor.float().mean()),
        "source_lineage_foreground_count": int(source_hits_tensor.sum()),
    }


@contextmanager
def deny_source_rgb_open():
    from PIL import Image

    original = Image.open
    rgb_root = (SCENE / "rgb").resolve()
    counters = {"source_rgb_open_attempts": 0}

    def guarded(path, *args, **kwargs):
        try:
            resolved = Path(path).resolve()
            resolved.relative_to(rgb_root)
        except (TypeError, ValueError):
            return original(path, *args, **kwargs)
        counters["source_rgb_open_attempts"] += 1
        raise RuntimeError("source RGB decoding is forbidden during compact Stage-2 lift")

    Image.open = guarded
    try:
        yield counters
    finally:
        Image.open = original


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def bundle_binding(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = directory / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    files = {}
    for record in payload["views"]:
        path = directory / record["teacher"]
        computed = sha256_file(path)
        if computed != record["teacher_sha256"]:
            raise RuntimeError(f"teacher archive hash differs from manifest for {path}")
        files[record["view_id"]] = {
            "path": display_path(path),
            "sha256": computed,
            "declared_sha256": record["teacher_sha256"],
            "n_init_2d": record["n_init_2d"],
            "n_opt_2d": record["n_opt_2d"],
        }
    geometry = payload.get("geometry")
    geometry_binding = None
    if geometry is not None:
        geometry_path = directory / geometry["path"]
        geometry_computed = sha256_file(geometry_path)
        if geometry_computed != geometry["sha256"]:
            raise RuntimeError("geometry archive hash differs from manifest")
        geometry_binding = {
            "path": display_path(geometry_path),
            "sha256": geometry_computed,
            "declared_sha256": geometry["sha256"],
        }
    return {
        "directory": display_path(directory),
        "manifest_path": display_path(manifest),
        "manifest_sha256": sha256_file(manifest),
        "semantic_digest": payload["semantic_digest"],
        "calibration_digest": payload["calibration_digest"],
        "teachers": files,
        "geometry": geometry_binding,
    }


def validate_anchor_inputs(
    anchor_inputs: ReconstructionInputs,
    exact_inputs: ReconstructionInputs,
) -> list[dict[str, Any]]:
    if tuple(anchor_inputs.view_names) != TRAIN_VIEWS:
        raise RuntimeError("anchor bundle view order differs from the frozen seven-view screen")
    if anchor_inputs.n_views != exact_inputs.n_views:
        raise RuntimeError("anchor and exact teacher bundles contain different view counts")
    if (anchor_inputs.points is None) != (exact_inputs.points is None):
        raise RuntimeError("anchor and replay bundles contain different sparse geometry")
    if anchor_inputs.points is not None and not torch.equal(
        anchor_inputs.points, exact_inputs.points
    ):
        raise RuntimeError("anchor and replay bundles contain different sparse points")
    if (anchor_inputs.point_visibility is None) != (exact_inputs.point_visibility is None):
        raise RuntimeError("anchor and replay bundles contain different point visibility")
    if anchor_inputs.point_visibility is not None and any(
        not torch.equal(anchor, exact)
        for anchor, exact in zip(
            anchor_inputs.point_visibility,
            exact_inputs.point_visibility,
            strict=True,
        )
    ):
        raise RuntimeError("anchor and replay bundles contain different point visibility")
    if (anchor_inputs.bounds_hint is None) != (exact_inputs.bounds_hint is None):
        raise RuntimeError("anchor and replay bundles contain different bounds hints")
    if anchor_inputs.bounds_hint is not None:
        anchor_center, anchor_extent = anchor_inputs.bounds_hint
        exact_center, exact_extent = exact_inputs.bounds_hint
        if not torch.equal(anchor_center, exact_center) or anchor_extent != exact_extent:
            raise RuntimeError("anchor and replay bundles contain different bounds hints")
    records = []
    for name, anchor, exact, anchor_camera, exact_camera in zip(
        anchor_inputs.view_names,
        anchor_inputs.observations,
        exact_inputs.observations,
        anchor_inputs.cameras,
        exact_inputs.cameras,
        strict=True,
    ):
        scalar_names = ("fx", "fy", "cx", "cy")
        scalar_delta = max(
            abs(float(getattr(anchor_camera, key)) - float(getattr(exact_camera, key)))
            for key in scalar_names
        )
        matrix_delta = max(
            float((anchor_camera.R - exact_camera.R).abs().max()),
            float((anchor_camera.t - exact_camera.t).abs().max()),
        )
        if (
            anchor_camera.width != exact_camera.width
            or anchor_camera.height != exact_camera.height
            or scalar_delta > 1e-6
            or matrix_delta > 1e-6
        ):
            raise RuntimeError(f"anchor calibration differs from exact teacher for {name}")
        if anchor.width != exact.width or anchor.height != exact.height:
            raise RuntimeError(f"anchor canvas differs from exact teacher for {name}")
        records.append(
            {
                "view": name,
                "n_init_2d": anchor.n_init,
                "n_opt_2d": anchor.n,
                "fit_window": list(anchor.fit_window),
                "provider": anchor.provider,
                "producer_version": anchor.producer_version,
                "calibration_scalar_max_abs": scalar_delta,
                "calibration_matrix_max_abs": matrix_delta,
            }
        )
    return records


def base_carve_config() -> CompactCarveConfig:
    payload = json.loads(EXACT_PLAN.read_text(encoding="utf-8"))
    config = dict(payload["configuration"]["compact_carve"])
    config.setdefault("anchor_mode", "mass_random")
    config.setdefault("max_anchor_candidates", 1_000_000)
    return CompactCarveConfig(**config)


def arm_specs() -> list[dict[str, str]]:
    return [
        {
            "name": "exact_density_mass_random",
            "input": "exact_lossless_full_frame",
            "anchor_mode": "mass_random",
            "coverage": "exact_density",
        },
        {
            "name": "masked_density_mass_random",
            "input": "exact_masked_teacher_anchor_bundle",
            "anchor_mode": "mass_random",
            "coverage": "exact_density",
        },
        {
            "name": "masked_density_component_centers",
            "input": "exact_masked_teacher_anchor_bundle",
            "anchor_mode": "component_centers",
            "coverage": "exact_density",
        },
        {
            "name": "masked_true_mask_component_centers",
            "input": "exact_masked_teacher_anchor_bundle",
            "anchor_mode": "component_centers",
            "coverage": "true_mask",
        },
        {
            "name": "masked_compact_proxy_component_centers",
            "input": "exact_masked_teacher_anchor_bundle",
            "anchor_mode": "component_centers",
            "coverage": "compact_proxy",
        },
    ]


def run(output: Path, anchor_bundle: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing screen output: {output}")
    source_before = source_binding()
    exact_inputs = ReconstructionInputs.load(EXACT_BUNDLE, strict=True)
    if tuple(exact_inputs.view_names) != TRAIN_VIEWS:
        raise RuntimeError("exact bundle view order drifted from the frozen screen")
    anchor_inputs = ReconstructionInputs.load(anchor_bundle, strict=True)
    anchor_records = validate_anchor_inputs(anchor_inputs, exact_inputs)
    masks, mask_bindings = load_masks(exact_inputs)
    exact_bundle_before = bundle_binding(EXACT_BUNDLE)
    anchor_bundle_before = bundle_binding(anchor_bundle)
    calibration_before = sha256_file(CALIBRATION)
    exact_plan_before = sha256_file(EXACT_PLAN)
    exact_baseline_before = sha256_file(EXACT_BASELINE_PLY)
    proxy_fields, proxy_records = build_occupancy_proxies(anchor_inputs, masks)
    proxy_inputs = ReconstructionInputs(
        observations=proxy_fields,
        cameras=anchor_inputs.cameras,
        view_names=anchor_inputs.view_names,
        points=anchor_inputs.points,
        point_visibility=anchor_inputs.point_visibility,
        bounds_hint=anchor_inputs.bounds_hint,
        name="exact_masked_center_gated_occupancy_proxy_screen",
    )
    output.mkdir(parents=True)
    (output / "arms").mkdir()
    proxy_bundle = output / "compact_occupancy_proxy_bundle"
    proxy_inputs.save(proxy_bundle)
    proxy_bundle_before = bundle_binding(proxy_bundle)

    base_config = base_carve_config()
    plan = {
        "artifact_type": "compact_masked_lift_screen_plan_v1",
        "decision_bearing": False,
        "scope": "exploratory calibrated Stage-2 mechanism screen",
        "source_binding": source_before,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "inputs": {
            "frozen_full_frame_replay_bundle": exact_bundle_before,
            "exact_masked_teacher_anchor_bundle": anchor_bundle_before,
            "exact_plan": {
                "path": EXACT_PLAN.relative_to(ROOT).as_posix(),
                "sha256": exact_plan_before,
            },
            "exact_baseline_ply": {
                "path": EXACT_BASELINE_PLY.relative_to(ROOT).as_posix(),
                "sha256": exact_baseline_before,
            },
            "calibration": {
                "path": CALIBRATION.relative_to(ROOT).as_posix(),
                "sha256": calibration_before,
            },
            "masks": mask_bindings,
            "source_rgb_policy": "forbidden/not decoded during every lift arm",
        },
        "anchor_validation": {
            "scientific_label": (
                "strict exact ReconstructionInputs archive supplied by the caller; compact "
                "fields provide both source anchors and queried RGB/color for all four masked "
                "arms; the full-frame bundle is isolated to a non-comparable replay control"
            ),
            "records": anchor_records,
        },
        "occupancy_proxy": {
            "scientific_label": (
                "derived center-gated Gaussian occupancy proxy; exact anchor geometry and "
                "support with mask samples replacing component amplitudes"
            ),
            "records": proxy_records,
            "occupancy_bundle": "compact_occupancy_proxy_bundle",
            "occupancy_bundle_binding": proxy_bundle_before,
        },
        "configuration": {
            "base_compact_carve": dataclasses.asdict(base_config),
            "mask_strength": MASK_STRENGTH,
            "true_mask_target_coverage": TRUE_MASK_TARGET_COVERAGE,
            "true_mask_mapping": "r=-log(1-h_target), h=1-exp(-r)",
            "occupancy_samples_per_view": OCCUPANCY_SAMPLES_PER_VIEW,
            "teacher_samples_per_view": TEACHER_SAMPLES_PER_VIEW,
            "foreground_samples_per_view": FOREGROUND_SAMPLES_PER_VIEW,
            "seed": SEED,
            "arms": arm_specs(),
        },
    }
    plan["plan_sha256"] = canonical_hash(plan)
    write_json(output / "plan.json", plan)

    exact_indexes = [
        GaussianObservationIndex(
            field,
            tile_size=base_config.tile_size,
            max_entries=base_config.max_index_entries_per_view,
            max_candidates=base_config.max_candidates_per_tile,
        )
        for field in exact_inputs.observations
    ]
    anchor_indexes = [
        GaussianObservationIndex(
            field,
            tile_size=base_config.tile_size,
            max_entries=base_config.max_index_entries_per_view,
            max_candidates=base_config.max_candidates_per_tile,
        )
        for field in anchor_inputs.observations
    ]
    proxy_indexes = [
        GaussianObservationIndex(
            field,
            tile_size=base_config.tile_size,
            max_entries=base_config.max_index_entries_per_view,
            max_candidates=base_config.max_candidates_per_tile,
        )
        for field in proxy_fields
    ]
    anchor_backends = {
        kind: make_backends(
            color_inputs=anchor_inputs,
            normalization_inputs=anchor_inputs,
            color_indexes=anchor_indexes,
            coverage_kind=kind,
            masks=masks,
            proxy_fields=proxy_fields,
            proxy_indexes=proxy_indexes,
        )
        for kind in ("exact_density", "true_mask", "compact_proxy")
    }
    backend_contract = backend_contract_audit(
        anchor_inputs,
        anchor_indexes,
        {
            "true_mask": anchor_backends["true_mask"],
            "compact_proxy": anchor_backends["compact_proxy"],
        },
    )
    write_json(output / "backend_contract_audit.json", backend_contract)
    occupancy = coverage_audit(
        anchor_inputs,
        masks,
        {
            "exact_density": anchor_backends["exact_density"],
            "true_mask": anchor_backends["true_mask"],
            "compact_proxy": anchor_backends["compact_proxy"],
        },
    )
    write_json(output / "occupancy_audit.json", occupancy)

    arm_results = {}
    with deny_source_rgb_open() as denied:
        for spec in arm_specs():
            arm_start = time.perf_counter()
            arm_dir = output / "arms" / spec["name"]
            arm_dir.mkdir()
            active_inputs = (
                exact_inputs if spec["input"] == "exact_lossless_full_frame" else anchor_inputs
            )
            if active_inputs is exact_inputs:
                backends = exact_indexes
                metric_inputs = exact_inputs
                metric_indexes = exact_indexes
                compact_teacher_label = "frozen_full_frame_replay_bundle"
            else:
                backends = (
                    anchor_indexes
                    if spec["coverage"] == "exact_density"
                    else anchor_backends[spec["coverage"]]
                )
                metric_inputs = anchor_inputs
                metric_indexes = anchor_indexes
                compact_teacher_label = "exact_masked_teacher_anchor_bundle"
            config = dataclasses.replace(base_config, anchor_mode=spec["anchor_mode"])
            try:
                initialization = CompactCarveInitializer(config).initialize(
                    active_inputs,
                    backends=backends,
                )
                initial_path = arm_dir / "gaussians_init.ply"
                final_path = arm_dir / "gaussians.ply"
                initialization.gaussians.save_ply(initial_path)
                initialization.gaussians.save_ply(final_path)
                if initial_path.read_bytes() != final_path.read_bytes():
                    raise RuntimeError("unrefined screen PLY copies differ")
                teacher_metrics = point_mse(
                    initialization.gaussians,
                    metric_inputs,
                    metric_indexes,
                    masks,
                )
                projection = foreground_projection_metrics(
                    initialization,
                    metric_inputs,
                    masks,
                )
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
                arm_record = {
                    "status": "PASS",
                    "spec": spec,
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
                    "compact_teacher_point_metric_source": compact_teacher_label,
                    "compact_teacher_point_metrics": teacher_metrics,
                    "artifacts": {
                        "gaussians_init": initial_path.relative_to(output).as_posix(),
                        "gaussians_init_sha256": sha256_file(initial_path),
                        "gaussians": final_path.relative_to(output).as_posix(),
                        "gaussians_sha256": sha256_file(final_path),
                    },
                    "wall_seconds": time.perf_counter() - arm_start,
                }
                if spec["name"] == "exact_density_mass_random":
                    baseline = Gaussians3D.load_ply(EXACT_BASELINE_PLY)
                    replay = Gaussians3D.load_ply(initial_path)
                    replay_record = {
                        "reference_path": EXACT_BASELINE_PLY.relative_to(ROOT).as_posix(),
                        "reference_sha256": exact_baseline_before,
                        "byte_exact": initial_path.read_bytes() == EXACT_BASELINE_PLY.read_bytes(),
                        "means_max_abs": float((replay.means - baseline.means).abs().max()),
                        "covariance_max_abs": float(
                            (replay.covariance() - baseline.covariance()).abs().max()
                        ),
                        "opacity_max_abs": float((replay.opacity - baseline.opacity).abs().max()),
                        "sh_max_abs": float((replay.sh - baseline.sh).abs().max()),
                    }
                    replay_record["numeric_match_at_1e-6"] = all(
                        replay_record[key] <= 1e-6
                        for key in (
                            "means_max_abs",
                            "covariance_max_abs",
                            "opacity_max_abs",
                            "sh_max_abs",
                        )
                    )
                    arm_record["frozen_baseline_replay"] = replay_record
                    if not replay_record["numeric_match_at_1e-6"]:
                        arm_record.update(
                            {
                                "status": "FAIL",
                                "failure_type": "FrozenBaselineReplayMismatch",
                                "failure_message": (
                                    "mass-random replay differs numerically from the frozen "
                                    "exact compact initializer"
                                ),
                            }
                        )
            except Exception as error:
                failure_diagnostics = None
                diagnostic_failure = None
                support_failure = isinstance(error, ValueError) and (
                    "fewer globally supported ray placements" in str(error)
                    or "proposed fewer anchors than n_init_3d" in str(error)
                )
                if support_failure:
                    try:
                        failure_diagnostics = diagnose_candidate_support(
                            active_inputs,
                            backends,
                            config,
                        )
                    except Exception as diagnostic_error:
                        diagnostic_failure = {
                            "failure_type": type(diagnostic_error).__name__,
                            "failure_message": str(diagnostic_error),
                        }
                allowed_proxy_infeasibility = (
                    spec["coverage"] == "compact_proxy"
                    and support_failure
                    and failure_diagnostics is not None
                    and (
                        failure_diagnostics["eligible_candidate_count"] is None
                        or failure_diagnostics["eligible_candidate_count"] < config.n_init_3d
                    )
                )
                arm_record = {
                    "status": "INFEASIBLE" if allowed_proxy_infeasibility else "FAIL",
                    "spec": spec,
                    "config": dataclasses.asdict(config),
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "failure_diagnostics": failure_diagnostics,
                    "diagnostic_failure": diagnostic_failure,
                    "wall_seconds": time.perf_counter() - arm_start,
                }
            arm_results[spec["name"]] = arm_record
            write_json(arm_dir / "result.json", arm_record)

    if denied["source_rgb_open_attempts"] != 0:
        raise RuntimeError("a lift arm attempted forbidden source RGB access")
    source_after = source_binding()
    if source_after != source_before:
        raise RuntimeError("bound screen source changed during execution")
    if bundle_binding(EXACT_BUNDLE) != exact_bundle_before:
        raise RuntimeError("exact RGB teacher bundle changed during execution")
    if bundle_binding(anchor_bundle) != anchor_bundle_before:
        raise RuntimeError("exact masked anchor bundle changed during execution")
    if bundle_binding(proxy_bundle) != proxy_bundle_before:
        raise RuntimeError("derived compact occupancy proxy bundle changed during execution")
    if sha256_file(CALIBRATION) != calibration_before:
        raise RuntimeError("mask calibration changed during execution")
    if sha256_file(EXACT_PLAN) != exact_plan_before:
        raise RuntimeError("frozen compact-carve plan changed during execution")
    if sha256_file(EXACT_BASELINE_PLY) != exact_baseline_before:
        raise RuntimeError("frozen exact baseline PLY changed during execution")
    for record in mask_bindings.values():
        if sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError("source mask file changed during execution")
    mandatory_arms = {
        "exact_density_mass_random",
        "masked_density_mass_random",
        "masked_density_component_centers",
        "masked_true_mask_component_centers",
    }
    mandatory_passed = all(arm_results[name]["status"] == "PASS" for name in mandatory_arms)
    proxy_status = arm_results["masked_compact_proxy_component_centers"]["status"]
    screen_passed = mandatory_passed and proxy_status in {"PASS", "INFEASIBLE"}
    result = {
        "artifact_type": "compact_masked_lift_screen_result_v1",
        "status": "PASS" if screen_passed else "FAIL",
        "decision_bearing": False,
        "result_scope": (
            "exploratory full-resolution Stage-2 mechanism screen; the caller-supplied exact "
            "masked bundle provides anchors and queried compact RGB/color for all masked arms, "
            "while dense masks remain an upper-bound coverage control"
        ),
        "plan_sha256": plan["plan_sha256"],
        "source_binding": source_after,
        "source_rgb_denial": denied,
        "backend_contract_audit": backend_contract,
        "occupancy_audit": occupancy,
        "completion_policy": {
            "mandatory_arms": sorted(mandatory_arms),
            "mandatory_arms_passed": mandatory_passed,
            "proxy_arm_allows_bounded_infeasibility": True,
            "proxy_arm_status": proxy_status,
        },
        "arms": arm_results,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "viewer": {
            "status": "DEFERRED_TO_ROOT_EXACT_GSPLAT",
            "reason": "root agent will select an arm and perform exact gsplat/viewer handoff",
        },
    }
    result["result_sha256"] = canonical_hash(result)
    write_json(output / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--anchor-bundle",
        type=Path,
        required=True,
        help="strict exact seven-view masked ReconstructionInputs bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.out.resolve(), args.anchor_bundle.resolve())
    summary = {
        "status": result["status"],
        "output": str(args.out.resolve()),
        "occupancy": {
            name: {
                "macro_iou": item["macro_mean_iou"],
                "macro_auc": item["macro_mean_auc"],
            }
            for name, item in result["occupancy_audit"].items()
        },
        "arms": {
            name: {
                "status": item["status"],
                "eligible": (item.get("diagnostics") or item.get("failure_diagnostics") or {}).get(
                    "eligible_candidate_count"
                ),
                "foreground_ge2": item.get("foreground_projection", {}).get(
                    "foreground_in_at_least_2_views_fraction"
                ),
                "uniform_mse": item.get("compact_teacher_point_metrics", {})
                .get("pooled", {})
                .get("uniform", {})
                .get("mse"),
                "foreground_mse": item.get("compact_teacher_point_metrics", {})
                .get("pooled", {})
                .get("foreground", {})
                .get("mse"),
            }
            for name, item in result["arms"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
