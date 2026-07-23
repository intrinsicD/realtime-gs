#!/usr/bin/env python3
"""Masked native-anchor density partitions as a Beam Fusion covariance treatment.

The unchanged covariance-intersection (``ci``) initialization is compared with two
covariance-only treatments.  Both identify anchors solely through Beam Fusion's stored CSR
lineage and partition every view's source Gaussian density strictly through its packed mask:

``pou-area``
    Match each native contributor covariance's determinant to its fixed-anchor partition moment
    while retaining the contributor's original 2D anisotropy and orientation.

``pou-full``
    Use the complete fixed-anchor partition second moment, including its anisotropy and
    orientation.

All arms retain the same 800 3D means, contributor identities and implied depths, opacity, SH,
and colors.  They receive the same 1,000-step fixed-topology CPU refinement used by the earlier
Beam covariance screen.  This remains a downscale-32, all-fitted-view, single-scene development
diagnostic rather than held-out or default-change evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.lift.beam_partition import (
    MaskedPartitionConfig,
    refit_beam_covariances_from_masked_partitions,
)

try:
    from benchmarks.beam_convergence_dynamics import (
        DOWNSCALE,
        EVAL_VIEWS,
        ITERATIONS,
        N_INIT,
        SEED,
        SELECTED_VIEWS,
        build_scene,
        run_arm,
        save_preview,
        selected_inputs,
    )
    from benchmarks.beam_covariance_refit import (
        _beam_result,
        _covariance_diagnostics,
        _member_observations,
        _projection_jacobians,
        _same_non_covariance_fields,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/beam_partition_covariance.py``
    from beam_convergence_dynamics import (  # type: ignore[no-redef]
        DOWNSCALE,
        EVAL_VIEWS,
        ITERATIONS,
        N_INIT,
        SEED,
        SELECTED_VIEWS,
        build_scene,
        run_arm,
        save_preview,
        selected_inputs,
    )
    from beam_covariance_refit import (  # type: ignore[no-redef]
        _beam_result,
        _covariance_diagnostics,
        _member_observations,
        _projection_jacobians,
        _same_non_covariance_fields,
    )

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
DEFAULT_OUT = ROOT / "runs/beam_partition_covariance_20260723"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260723_beam_partition_covariance_PREREG.md"

ARMS = ("ci", "pou-area", "pou-full")
PARTITION_CONFIG = MaskedPartitionConfig(
    quadrature_order=5,
    assignment_chunk=8_192,
    min_partition_mass=1e-12,
    min_variance_px=1e-6,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_covariances_from_tables(
    beam,
    tables: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-view component covariance tables through the Beam CSR lineage."""
    count = beam.n_components
    n_views = len(tables)
    mask = torch.zeros(count, n_views, dtype=torch.bool)
    covariances = torch.zeros(count, n_views, 2, 2, dtype=torch.float64)
    offsets = beam.component_offsets.tolist()
    for component in range(count):
        start, stop = offsets[component], offsets[component + 1]
        views = beam.contributor_view_indices[start:stop]
        splats = beam.contributor_component_indices[start:stop]
        mask[component, views] = True
        for view, splat in zip(views.tolist(), splats.tolist(), strict=True):
            covariances[component, view] = tables[view][splat]
    return mask, covariances


def build_initializations(
    dataset: CompactDataset,
) -> tuple[dict[str, Gaussians3D], dict]:
    """Build CI, determinant-matched, and full partition-moment initializations."""
    beam, beam_receipt = _beam_result(dataset)
    inputs = selected_inputs(dataset)
    alphas = [dataset.views[index].alpha for index in SELECTED_VIEWS]
    fitted = refit_beam_covariances_from_masked_partitions(
        inputs,
        alphas,
        beam,
        # `_beam_result` freezes this exact configuration in its receipt.
        beam_config_from_receipt(dataset, beam_receipt),
        PARTITION_CONFIG,
    )
    inits = {
        "ci": beam.gaussians.detach(),
        "pou-area": fitted.area_matched,
        "pou-full": fitted.full_moment,
    }
    for name, initialization in inits.items():
        if not _same_non_covariance_fields(inits["ci"], initialization):
            raise RuntimeError(f"{name} changed a frozen non-covariance field")

    member_mask, native_members = _member_observations(beam, inputs)
    native_tables = []
    area_tables = []
    full_tables = []
    for field, partition in zip(inputs.observations, fitted.partitions, strict=True):
        from rtgs.lift.beam_fusion import _component_covariances_2d

        native = _component_covariances_2d(field)
        area = native.clone()
        full = native.clone()
        anchors = partition.anchor_component_indices
        area[anchors] = partition.area_matched_covariances2d
        full[anchors] = partition.covariances2d
        native_tables.append(native)
        area_tables.append(area)
        full_tables.append(full)
    area_mask, area_members = _member_covariances_from_tables(beam, area_tables)
    full_mask, full_members = _member_covariances_from_tables(beam, full_tables)
    if not torch.equal(member_mask, area_mask) or not torch.equal(member_mask, full_mask):
        raise RuntimeError("partition target expansion changed Beam contributor membership")

    jacobians = _projection_jacobians(inits["ci"].means, inputs.cameras)
    target_members = {
        "ci": native_members,
        "pou-area": area_members,
        "pou-full": full_members,
    }
    diagnostics = {
        "beam": beam_receipt,
        "partition": fitted.diagnostics,
        "n_observation_links": int(member_mask.sum()),
        "arms_against_native_contributor_covariances": {
            name: _covariance_diagnostics(
                initialization.covariance(),
                jacobians,
                member_mask,
                native_members,
            )
            for name, initialization in inits.items()
        },
        "arms_against_own_partition_targets": {
            name: _covariance_diagnostics(
                initialization.covariance(),
                jacobians,
                member_mask,
                target_members[name],
            )
            for name, initialization in inits.items()
        },
        "frozen_field_assertions": {
            name: {
                "same_count": initialization.n == inits["ci"].n,
                "means_bit_exact": torch.equal(initialization.means, inits["ci"].means),
                "opacity_bit_exact": torch.equal(initialization.opacity, inits["ci"].opacity),
                "sh_bit_exact": torch.equal(initialization.sh, inits["ci"].sh),
            }
            for name, initialization in inits.items()
        },
    }
    return inits, diagnostics


def beam_config_from_receipt(dataset: CompactDataset, receipt: dict):
    """Reconstruct the exact Beam config frozen by the shared placement helper."""
    from rtgs.lift.beam_fusion import BeamFusionConfig

    config = receipt["config"]
    return BeamFusionConfig(
        min_views=config["min_views"],
        transverse_gate_sigma=config["transverse_gate_sigma"],
        max_color_distance=config["max_color_distance"],
        color_sigma=config["color_sigma"],
        fold_in_gate_sigma=config["fold_in_gate_sigma"],
        nms_voxel_size=float(dataset.bounds_hint[1]) / 100.0,
        init_opacity=config["init_opacity"],
        source_chunk=config["source_chunk"],
        max_components=config["max_components"],
        seed_budget_multiplier=config["seed_budget_multiplier"],
    )


def _write_viewer_manifest(out: Path) -> Path:
    manifest = {
        "schema": "rtgs.viewer-comparison.v1",
        "methods": [
            {
                "name": name,
                "initial": f"../../runs/{out.name}/{name}/gaussians_init.ply",
                "final": f"../../runs/{out.name}/{name}/gaussians_final.ply",
            }
            for name in ARMS
        ],
    }
    path = ROOT / "benchmarks/results/20260723_beam_partition_covariance_VIEWER.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError(
            f"frozen protocol missing: {args.protocol}; write it before reading Janelle outcomes"
        )
    if args.iterations != ITERATIONS:
        print(
            f"[warning] iterations={args.iterations} is a smoke override; frozen value is "
            f"{ITERATIONS}",
            flush=True,
        )

    torch.manual_seed(SEED)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = CompactDataset.load(DATASET, device="cpu")
    scene = build_scene(dataset)
    initializations, covariance = build_initializations(dataset)
    reference_surface = initializations["ci"].means.detach().clone()
    records: dict[str, dict] = {}
    for arm in args.arms:
        records[arm] = run_arm(
            arm,
            initializations[arm],
            scene,
            reference_surface,
            False,
            out,
            args.iterations,
        )
        final = Gaussians3D.load_ply(out / arm / "gaussians_final.ply")
        save_preview(scene, initializations[arm], out / arm / "preview_init.png")
        save_preview(scene, final, out / arm / "preview_final.png")

    summary = {
        "schema": "rtgs.beam_partition_covariance.v1",
        "status": "complete" if set(args.arms) == set(ARMS) else "partial",
        "protocol": {
            "path": str(args.protocol.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "dataset": str(DATASET.relative_to(ROOT)),
        "selected_global_views": list(SELECTED_VIEWS),
        "evaluation_local_views": list(EVAL_VIEWS),
        "downscale": DOWNSCALE,
        "n_init": N_INIT,
        "iterations": args.iterations,
        "fixed_topology": True,
        "seed": SEED,
        "partition_config": {
            "quadrature_order": PARTITION_CONFIG.quadrature_order,
            "assignment_chunk": PARTITION_CONFIG.assignment_chunk,
            "min_partition_mass": PARTITION_CONFIG.min_partition_mass,
            "min_variance_px": PARTITION_CONFIG.min_variance_px,
        },
        "covariance": covariance,
        "arms": {
            name: {
                "init_metrics": record["init_metrics"],
                "final_metrics": {
                    key.removeprefix("metric_"): value
                    for key, value in record["curve"][-1].items()
                    if key.startswith("metric_")
                },
                "final_n": record["final_n"],
                "elapsed_seconds": record["elapsed_seconds"],
                "loss_first_20": record["loss_first_20"],
                "loss_last_20": record["loss_last_20"],
            }
            for name, record in records.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    viewer_manifest = _write_viewer_manifest(out)
    print(
        json.dumps(
            {
                name: {
                    "init_psnr_fg": arm["init_metrics"].get("psnr_fg"),
                    "init_alpha_iou": arm["init_metrics"].get("alpha_iou"),
                    "final_psnr_fg": arm["final_metrics"].get("psnr_fg"),
                    "final_alpha_iou": arm["final_metrics"].get("alpha_iou"),
                }
                for name, arm in summary["arms"].items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"[viewer] {viewer_manifest.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
