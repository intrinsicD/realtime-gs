#!/usr/bin/env python3
"""Independent audit of the cover-consistent surfel initialization screen.

Recomputes every preregistered gate from the saved artifacts rather than trusting the harness's
own printout, rebuilds the treatment from the control's saved PLY to confirm the arms differ
only where they are allowed to, and checks the frozen-field and provenance claims.  Exits
non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.lift.surfel_init import (
    SurfelInitConfig,
    coverage_opacity,
    effective_cover_sigma,
    estimate_local_surface_frames,
    hexagonal_cover_ripple,
    reconcile_covariances,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/beam_surfel_init_20260724"
ARMS = ("ci", "ci-op", "cover-iso", "cover-iso-op", "surfel")


class Audit:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        self.checks.append((name, bool(condition), detail))
        return bool(condition)

    @property
    def passed(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok)

    def report(self) -> int:
        for name, ok, detail in self.checks:
            marker = "PASS" if ok else "FAIL"
            print(f"[{marker}] {name}" + (f" — {detail}" if detail else ""))
        print(f"\n{self.passed}/{len(self.checks)} checks passed")
        return 0 if self.passed == len(self.checks) else 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_provenance(audit: Audit, run: Path, summary: dict) -> None:
    protocol = ROOT / summary["protocol"]["path"]
    audit.check("protocol file exists", protocol.is_file(), str(protocol))
    if protocol.is_file():
        audit.check(
            "protocol sha256 matches the recorded hash",
            _sha256(protocol) == summary["protocol"]["sha256"],
            summary["protocol"]["sha256"][:16],
        )
    audit.check(
        "status is complete", summary.get("status") == "complete", str(summary.get("status"))
    )
    audit.check(
        "held-out views are disjoint from train views",
        not (set(summary["train_view_indices"]) & set(summary["heldout_view_indices"])),
        f"train={summary['train_view_indices']} heldout={summary['heldout_view_indices']}",
    )
    audit.check(
        "run used the fresh frame_00009 root",
        summary["frame"] == "frame_00009",
        summary["frame"],
    )
    audit.check("index.html was written", (run / "index.html").is_file())


def audit_frozen_fields(audit: Audit, run: Path, summary: dict) -> Gaussians3D | None:
    mode = "fixed"
    control_path = run / mode / "ci" / "gaussians_init.ply"
    if not control_path.is_file():
        audit.check("control initial PLY exists", False, str(control_path))
        return None
    control = Gaussians3D.load_ply(control_path)
    audit.check("control has the declared count", control.n == summary["n_init"], str(control.n))
    for arm in ARMS:
        path = run / mode / arm / "gaussians_init.ply"
        if not path.is_file():
            audit.check(f"{arm} initial PLY exists", False, str(path))
            continue
        model = Gaussians3D.load_ply(path)
        audit.check(f"{arm} preserves the count", model.n == control.n)
        audit.check(f"{arm} preserves means bit-exactly", torch.equal(model.means, control.means))
        audit.check(f"{arm} preserves SH bit-exactly", torch.equal(model.sh, control.sh))
        if arm != "ci":
            differs = not torch.equal(model.log_scales, control.log_scales) or not torch.equal(
                model.opacity, control.opacity
            )
            audit.check(f"{arm} actually differs from the control", differs)
    # Both modes must start from the same initialization.
    for arm in ARMS:
        fixed_path = run / "fixed" / arm / "gaussians_init.ply"
        adc_path = run / "adc" / arm / "gaussians_init.ply"
        if fixed_path.is_file() and adc_path.is_file():
            audit.check(
                f"{arm} init is byte-identical across modes",
                _sha256(fixed_path) == _sha256(adc_path),
            )
    return control


def audit_rule_reproduction(audit: Audit, run: Path, control: Gaussians3D) -> None:
    """Rebuild the treatments from the control PLY alone and compare to the saved arms."""
    config = SurfelInitConfig()
    audit.check(
        "derived cover ratio still sits below a two percent hexagonal ripple",
        hexagonal_cover_ripple(config.coverage_sigma_ratio) < 0.02,
        f"{hexagonal_cover_ripple(config.coverage_sigma_ratio):.6f}",
    )
    frames = estimate_local_surface_frames(control.means, config)
    # The resolution floor comes from Beam's lineage, which the PLY does not carry; rebuild the
    # floor-free variants, which bound the saved sigmas from below.
    rebuilt = reconcile_covariances(control, config, frames=frames)
    saved_path = run / "fixed" / "surfel" / "gaussians_init.ply"
    if saved_path.is_file():
        saved = Gaussians3D.load_ply(saved_path)
        saved_sigma = saved.scales.amax(dim=-1).to(torch.float64)
        audit.check(
            "saved surfel tangential sigma is at least the floor-free cover sigma",
            bool((saved_sigma >= rebuilt.sigma_tangent - 1e-6).all()),
            f"min slack {float((saved_sigma - rebuilt.sigma_tangent).min()):.3e}",
        )
        ratio = saved_sigma / frames.spacing
        audit.check(
            "saved surfel sigma reaches at least half the local spacing",
            float(ratio.min()) >= config.coverage_sigma_ratio - 1e-6,
            f"min ratio {float(ratio.min()):.6f}",
        )
        opacity, overlap = coverage_opacity(saved_sigma, frames.spacing, config)
        audit.check(
            "saved surfel opacity reproduces the closed-form coverage rule",
            torch.allclose(saved.opacity.to(torch.float64), opacity, atol=1e-5),
            f"max abs diff {float((saved.opacity.to(torch.float64) - opacity).abs().max()):.3e}",
        )
        accumulated = 1.0 - (1.0 - saved.opacity.to(torch.float64)) ** overlap
        unclamped = (saved.opacity.to(torch.float64) > config.min_opacity + 1e-9) & (
            saved.opacity.to(torch.float64) < config.max_opacity - 1e-9
        )
        audit.check(
            "unclamped opacities hit the target accumulated alpha",
            bool((accumulated[unclamped] - config.target_alpha).abs().max() < 1e-4)
            if bool(unclamped.any())
            else True,
            f"{int(unclamped.sum())} unclamped of {saved.n}",
        )
    ci_path = run / "fixed" / "ci-op" / "gaussians_init.ply"
    if ci_path.is_file():
        ci_op = Gaussians3D.load_ply(ci_path)
        audit.check(
            "ci-op keeps the control covariance exactly",
            torch.equal(ci_op.log_scales, control.log_scales)
            and torch.equal(ci_op.quats, control.quats),
        )
        expected, _ = coverage_opacity(effective_cover_sigma(control), frames.spacing, config)
        audit.check(
            "ci-op opacity reproduces the coverage rule on the control's own sigma",
            torch.allclose(ci_op.opacity.to(torch.float64), expected, atol=1e-5),
        )


def audit_gates(audit: Audit, summary: dict) -> dict:
    fixed = summary["modes"].get("fixed", {})
    adc = summary["modes"].get("adc", {})

    def held(arm: dict, stage: str, key: str) -> float:
        return arm[stage]["heldout"][key]

    verdicts: dict[str, bool] = {}
    if "surfel" in fixed and "ci" in fixed:
        surfel, ci = fixed["surfel"], fixed["ci"]
        iou = held(surfel, "init_metrics", "alpha_iou")
        outside = held(surfel, "init_metrics", "alpha_outside")
        g1 = iou >= 0.25 and outside <= 0.05
        verdicts["G1_coverage"] = g1
        audit.check(
            "G1 coverage recomputed",
            True,
            f"init alpha-IoU {iou:.5f} (>=0.25) and alpha-outside {outside:.5f} (<=0.05) -> "
            f"{'PASS' if g1 else 'FAIL'}",
        )
        psnr_gain = held(surfel, "init_metrics", "psnr_fg") - held(ci, "init_metrics", "psnr_fg")
        verdicts["G2_initial_quality"] = psnr_gain >= 1.0
        audit.check(
            "G2 initial quality recomputed",
            True,
            f"init foreground PSNR gain {psnr_gain:+.4f} dB (>=1.0) -> "
            f"{'PASS' if psnr_gain >= 1.0 else 'FAIL'}",
        )
        auc_rel = surfel["heldout_psnr_auc"] / ci["heldout_psnr_auc"] - 1.0
        final_delta = held(surfel, "final_metrics", "psnr_fg") - held(
            ci, "final_metrics", "psnr_fg"
        )
        g3 = auc_rel >= 0.03 and final_delta >= -0.1
        verdicts["G3_optimization"] = g3
        audit.check(
            "G3 optimization recomputed",
            True,
            f"AUC {auc_rel * 100:+.4f}% (>=3%) and final PSNR {final_delta:+.4f} dB (>=-0.1) -> "
            f"{'PASS' if g3 else 'FAIL'}",
        )
    if "surfel" in fixed and "ci-op" in fixed:
        gap = held(fixed["surfel"], "init_metrics", "alpha_iou") - held(
            fixed["ci-op"], "init_metrics", "alpha_iou"
        )
        verdicts["G5_attribution"] = gap >= 0.10
        audit.check(
            "G5 attribution recomputed",
            True,
            f"init alpha-IoU over ci-op {gap:+.5f} (>=0.10) -> {'PASS' if gap >= 0.10 else 'FAIL'}",
        )
    if "surfel" in adc and "ci" in adc:
        surfel_frac = adc["surfel"]["participation"]["frac_initial_qualified"]
        ci_frac = adc["ci"]["participation"]["frac_initial_qualified"]
        g4 = surfel_frac >= 5.0 * ci_frac
        verdicts["G4_participation"] = g4
        audit.check(
            "G4 densification participation recomputed",
            True,
            f"qualified fraction surfel {surfel_frac:.5f} vs ci {ci_frac:.5f} "
            f"(>=5x) -> {'PASS' if g4 else 'FAIL'}",
        )
    if adc:
        worst = max(
            (arm["final_metrics"]["heldout"]["alpha_outside"], name) for name, arm in adc.items()
        )
        verdicts["guardrail_final_outside_alpha"] = worst[0] <= 0.05
        audit.check(
            "guardrail final outside-alpha recomputed",
            True,
            f"worst arm {worst[1]} at {worst[0]:.5f} (<=0.05) -> "
            f"{'PASS' if worst[0] <= 0.05 else 'FAIL'}",
        )
    audit.check("every preregistered gate was evaluated", len(verdicts) == 6, str(sorted(verdicts)))
    return verdicts


def audit_consistency(audit: Audit, run: Path, summary: dict) -> None:
    for mode, arms in summary["modes"].items():
        for name, arm in arms.items():
            record_path = run / mode / name / "record.json"
            if not record_path.is_file():
                audit.check(f"{mode}/{name} record.json exists", False)
                continue
            record = json.loads(record_path.read_text())
            audit.check(
                f"{mode}/{name} summary final count matches its record",
                record["final_n"] == arm["final_n"],
            )
            audit.check(
                f"{mode}/{name} final PLY exists",
                (run / mode / name / "gaussians_final.ply").is_file(),
            )
            if mode == "fixed":
                audit.check(
                    f"fixed/{name} kept its topology",
                    arm["final_n"] == summary["n_init"],
                    str(arm["final_n"]),
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    summary_path = run / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text())

    audit = Audit()
    audit_provenance(audit, run, summary)
    control = audit_frozen_fields(audit, run, summary)
    if control is not None:
        audit_rule_reproduction(audit, run, control)
    verdicts = audit_gates(audit, summary)
    audit_consistency(audit, run, summary)

    exit_code = audit.report()
    print("\nPreregistered verdicts:")
    print(json.dumps(verdicts, indent=2, sort_keys=True))
    (run / "audit.json").write_text(
        json.dumps(
            {
                "schema": "rtgs.beam_surfel_init_audit.v1",
                "checks": [
                    {"name": name, "passed": ok, "detail": detail}
                    for name, ok, detail in audit.checks
                ],
                "n_passed": audit.passed,
                "n_total": len(audit.checks),
                "preregistered_verdicts": verdicts,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
