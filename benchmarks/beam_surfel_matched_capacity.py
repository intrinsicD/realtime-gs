#!/usr/bin/env python3
"""Matched-capacity falsification of the surfel-initialization count result.

The 20260724 screen compared arms under *unmatched* budgets: both ran with an 8,000 cap and
stopped growing at different points, so "better held-out quality with under half the primitives"
is equally consistent with "the control was allowed to overshoot".  This harness gives every arm
the same hard budget (3x the initialization size) and three trainer seeds, and is written to
*withdraw* the capacity claim if the control catches up.

Reuses the frozen scene, initialization, and evaluation code of ``beam_surfel_init`` unchanged;
only the density budget and the seed vary.  Same root as the screen it tests, so this confirms a
mechanism and cannot establish generalization.

Frozen protocol: ``benchmarks/results/20260724_beam_surfel_matched_capacity_PREREG.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.data.scene import SceneData
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

try:
    from benchmarks.beam_surfel_init import (
        DOWNSCALE,
        GRAD_THRESHOLD,
        N_INIT,
        build_initializations,
        build_scene,
        save_preview,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/beam_surfel_matched_capacity.py``
    from beam_surfel_init import (  # type: ignore[no-redef]
        DOWNSCALE,
        GRAD_THRESHOLD,
        N_INIT,
        build_initializations,
        build_scene,
        save_preview,
    )

ROOT = Path(__file__).resolve().parents[1]
FRAME = "frame_00009"
DEFAULT_OUT = ROOT / "runs/beam_surfel_matched_capacity_20260724"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260724_beam_surfel_matched_capacity_PREREG.md"

ARMS = ("ci", "surfel", "cover-iso-op")
SEEDS = (0, 1, 2)
ITERATIONS = 1_000
EVAL_EVERY = 25
MATCHED_BUDGET = 3 * N_INIT  # 2400, derived from the frozen initialization size
DECISION_MARGIN_DB = 0.15
OUTSIDE_ALPHA_GUARDRAIL = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_config(seed: int, iterations: int, budget: int) -> TrainConfig:
    return TrainConfig(
        iterations=iterations,
        rasterizer="torch",
        device="cpu",
        densify=True,
        density_strategy="classic",
        density=DensityConfig(
            start_iter=20,
            stop_iter=500,
            every=4,
            grad_threshold=GRAD_THRESHOLD,
            absgrad=False,
            prune_opacity=0.005,
            prune_scale_frac=0.1,
            max_gaussians=budget,
            opacity_reset_every=100,
            opacity_reset_value=0.011,
        ),
        eval_every=EVAL_EVERY,
        target_sh_degree=3,
        sh_degree_interval=33,
        use_masks=True,
        random_background=False,
        seed=seed,
    )


def run_cell(
    arm: str,
    seed: int,
    init: Gaussians3D,
    scene: SceneData,
    train_local: list[int],
    heldout_local: list[int],
    extrapolative_local: int,
    out: Path,
    iterations: int,
    budget: int,
) -> dict:
    cell_dir = out / f"seed{seed}" / arm
    cell_dir.mkdir(parents=True, exist_ok=True)
    renderer = get_rasterizer("torch", device=torch.device("cpu"))

    def metrics(model: Gaussians3D) -> dict:
        return {
            "train": Trainer.evaluate_metrics(scene, model, renderer, train_local),
            "heldout": Trainer.evaluate_metrics(scene, model, renderer, heldout_local),
            "heldout_extrapolative": Trainer.evaluate_metrics(
                scene, model, renderer, [extrapolative_local]
            ),
        }

    torch.manual_seed(seed)
    init_metrics = metrics(init)
    curve: list[dict] = []
    started = time.perf_counter()

    def checkpoint(snapshot: Gaussians3D, step: int) -> None:
        values = metrics(snapshot)
        curve.append(
            {
                "step": step,
                "n": snapshot.n,
                **{f"heldout_{k}": v for k, v in values["heldout"].items()},
                "train_psnr_fg": values["train"].get("psnr_fg"),
            }
        )
        print(
            f"[seed{seed}/{arm}] step {step} n={snapshot.n} "
            f"held_fg={values['heldout'].get('psnr_fg', float('nan')):.3f}dB "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    final, history = Trainer(train_config(seed, iterations, budget)).train(
        scene, init, checkpoint_callback=checkpoint
    )
    final.save_ply(cell_dir / "gaussians_final.ply")
    final_metrics = metrics(final)
    samples = [init_metrics["heldout"]["psnr_fg"]] + [
        row["heldout_psnr_fg"] for row in curve if "heldout_psnr_fg" in row
    ]
    record = {
        "arm": arm,
        "seed": seed,
        "budget": budget,
        "elapsed_seconds": time.perf_counter() - started,
        "init_metrics": init_metrics,
        "final_metrics": final_metrics,
        "final_n": final.n,
        "reached_budget": final.n >= budget,
        "heldout_psnr_auc": sum(samples) / len(samples),
        "curve": curve,
        "loss_last_20": history["loss"][-20:],
    }
    (cell_dir / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def decide(cells: dict[str, dict[int, dict]], seeds: list[int]) -> dict:
    """Score M1/M2/M3 exactly as preregistered, from the recorded cells alone."""
    per_seed = []
    for seed in seeds:
        control = cells["ci"][seed]
        treatment = cells["surfel"][seed]
        delta = (
            treatment["final_metrics"]["heldout"]["psnr_fg"]
            - control["final_metrics"]["heldout"]["psnr_fg"]
        )
        per_seed.append(
            {
                "seed": seed,
                "delta_psnr_fg_db": delta,
                "surfel_final_n": treatment["final_n"],
                "ci_final_n": control["final_n"],
                "treatment_not_larger": treatment["final_n"] <= control["final_n"],
                "m1_cell": delta >= DECISION_MARGIN_DB
                and treatment["final_n"] <= control["final_n"],
                "m2_cell": abs(delta) < DECISION_MARGIN_DB,
                "m3_cell": -delta >= DECISION_MARGIN_DB,
            }
        )
    n = len(seeds)
    majority = n // 2 + 1
    m1 = sum(c["m1_cell"] for c in per_seed) >= majority
    m2 = sum(c["m2_cell"] for c in per_seed) >= majority
    m3 = sum(c["m3_cell"] for c in per_seed) >= majority
    if m1:
        verdict = "M1_CAPACITY_ADVANTAGE_HOLDS"
    elif m2:
        verdict = "M2_CAPACITY_CLAIM_WITHDRAWN"
    elif m3:
        verdict = "M3_TREATMENT_WORSE_UNDER_MATCHED_BUDGET"
    else:
        verdict = "INCONCLUSIVE_NO_MAJORITY"
    guardrail = {
        f"{arm}/seed{seed}": cells[arm][seed]["final_metrics"]["heldout"]["alpha_outside"]
        for arm in cells
        for seed in cells[arm]
    }
    return {
        "margin_db": DECISION_MARGIN_DB,
        "per_seed": per_seed,
        "verdict": verdict,
        "guardrail_max_outside_alpha": max(guardrail.values()),
        "guardrail_passed": max(guardrail.values()) <= OUTSIDE_ALPHA_GUARDRAIL,
        "guardrail_by_cell": guardrail,
    }


def write_viewer_manifest(out: Path, arms: list[str], seeds: list[int]) -> Path:
    """Synchronized comparison over every (seed, arm) final model.

    This run varies only the trainer seed and the density budget, so each arm's initialization is
    the shared one from the screen under test; `initial` therefore points at that screen's saved
    init PLY and `final` at this run's matched-budget endpoint.
    """
    manifest = {
        "schema": "rtgs.viewer-comparison.v1",
        "methods": [
            {
                "name": f"seed{seed}:{arm}",
                "initial": f"../../runs/beam_surfel_init_20260724/fixed/{arm}/gaussians_init.ply",
                "final": f"../../runs/{out.name}/seed{seed}/{arm}/gaussians_final.ply",
            }
            for seed in seeds
            for arm in arms
        ],
    }
    path = ROOT / "benchmarks/results/20260724_beam_surfel_matched_capacity_VIEWER.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


def write_index(out: Path, summary: dict) -> Path:
    """Summary-bound results page with relative links to this run's own artifacts."""
    rows = []
    for arm, by_seed in summary["cells"].items():
        for seed, cell in sorted(by_seed.items()):
            held = cell["final_metrics"]["heldout"]
            rows.append(
                "<tr><td>{arm}</td><td>{seed}</td><td>{n}</td><td>{cap}</td><td>{p:.4f}</td>"
                "<td>{i:.4f}</td><td>{o:.4f}</td><td>{a:.4f}</td>"
                '<td><a href="seed{seed}/{arm}/preview_final.png">final</a></td></tr>'.format(
                    arm=arm,
                    seed=seed,
                    n=cell["final_n"],
                    cap="yes" if cell["reached_budget"] else "no",
                    p=held.get("psnr_fg", float("nan")),
                    i=held.get("alpha_iou", float("nan")),
                    o=held.get("alpha_outside", float("nan")),
                    a=cell["heldout_psnr_auc"],
                )
            )
    decision = summary["decision"]
    per_seed = "".join(
        "<li>seed {s}: surfel - ci = {d:+.4f} dB, N {a} vs {b}</li>".format(
            s=c["seed"], d=c["delta_psnr_fg_db"], a=c["surfel_final_n"], b=c["ci_final_n"]
        )
        for c in decision.get("per_seed", [])
    )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Matched-capacity falsification — {summary["frame"]}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 66rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.6rem; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
code {{ background: #f4f4f4; padding: 0 0.2rem; }}
</style>
<h1>Matched-capacity falsification of the surfel-initialization count result</h1>
<p>Every arm shares one hard budget of <strong>{summary["matched_budget"]}</strong> primitives
(3 x the frozen {summary["n_init"]}-Gaussian initialization) over seeds {summary["seeds"]} on
<code>{summary["frame"]}</code>. All metrics are on the held-out cameras, which entered neither
Beam Fusion nor refinement. Protocol:
<a href="../../benchmarks/results/20260724_beam_surfel_matched_capacity_PREREG.md"
>preregistration</a>.</p>
<p><strong>Verdict: {decision["verdict"]}</strong> at a {decision["margin_db"]} dB margin;
outside-alpha guardrail {"passed" if decision["guardrail_passed"] else "FAILED"}
(worst {decision["guardrail_max_outside_alpha"]:.5f}).</p>
<ul>{per_seed}</ul>
<table>
<tr><th>arm</th><th>seed</th><th>final N</th><th>hit cap</th><th>held-out PSNR</th>
<th>alpha IoU</th><th>alpha out</th><th>PSNR AUC</th><th>preview</th></tr>
{chr(10).join(rows)}
</table>
<p>Same root as the screen under test, so this confirms a mechanism and does not generalize.
One device, CPU reference rasterizer, downscale {summary["downscale"]}.</p>
"""
    path = out / "index.html"
    path.write_text(html)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--budget", type=int, default=MATCHED_BUDGET)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError(f"frozen protocol missing: {args.protocol}")
    for name, value, frozen in (
        ("iterations", args.iterations, ITERATIONS),
        ("budget", args.budget, MATCHED_BUDGET),
    ):
        if value != frozen:
            print(
                f"[warning] {name}={value} is a smoke override; frozen value is {frozen}",
                flush=True,
            )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = CompactDataset.load(
        ROOT / f"dataset/2025_03_07_stage_with_fabric/{FRAME}/gaussians2d", device="cpu"
    )
    scene, train_local, heldout_local, extrapolative_local = build_scene(dataset)
    # Beam Fusion is deterministic, so one placement is shared by every arm and seed.
    torch.manual_seed(0)
    initializations, diagnostics = build_initializations(dataset)

    cells: dict[str, dict[int, dict]] = {arm: {} for arm in args.arms}
    for seed in args.seeds:
        for arm in args.arms:
            cells[arm][seed] = run_cell(
                arm,
                seed,
                initializations[arm],
                scene,
                train_local,
                heldout_local,
                extrapolative_local,
                out,
                args.iterations,
                args.budget,
            )
            final = Gaussians3D.load_ply(out / f"seed{seed}" / arm / "gaussians_final.ply")
            save_preview(
                scene, final, out / f"seed{seed}" / arm / "preview_final.png", heldout_local[0]
            )

    decision = (
        decide(cells, list(args.seeds))
        if {"ci", "surfel"} <= set(args.arms)
        else {"verdict": "NOT_EVALUATED_PARTIAL_ARMS"}
    )
    summary = {
        "schema": "rtgs.beam_surfel_matched_capacity.v1",
        "status": (
            "complete"
            if set(args.arms) == set(ARMS) and set(args.seeds) == set(SEEDS)
            else "partial"
        ),
        "protocol": {
            "path": str(args.protocol.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "frame": FRAME,
        "same_root_as_screen_under_test": True,
        "downscale": DOWNSCALE,
        "n_init": N_INIT,
        "matched_budget": args.budget,
        "iterations": args.iterations,
        "seeds": list(args.seeds),
        "grad_threshold": GRAD_THRESHOLD,
        "initialization_diagnostics": diagnostics["arm_scale_summary"],
        "decision": decision,
        "cells": {
            arm: {
                str(seed): {
                    key: record[key]
                    for key in (
                        "init_metrics",
                        "final_metrics",
                        "final_n",
                        "reached_budget",
                        "heldout_psnr_auc",
                        "elapsed_seconds",
                        "curve",
                    )
                }
                for seed, record in by_seed.items()
            }
            for arm, by_seed in cells.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    index = write_index(out, summary)
    viewer = write_viewer_manifest(out, args.arms, list(args.seeds))
    print(json.dumps(decision, indent=2), flush=True)
    print(f"[index] {index}", flush=True)
    print(f"[viewer] {viewer}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
