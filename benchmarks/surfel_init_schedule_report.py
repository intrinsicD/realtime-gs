#!/usr/bin/env python3
"""Render the ADR-XXXX / ADR-YYYY screen's ``index.html`` results page.

The page is summary-bound: every number a reader would want is on the page, and every link is
relative and resolves inside the run directory (``scripts/check_results_bundle.py`` enforces
both).  It states the evidence status before it states any number, because the fastest way to
turn a development screen into a false claim is to publish the table without the caveat.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STYLE = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 1080px; padding: 2rem 1.25rem 4rem; }
h1 { font-size: 1.6rem; margin-bottom: .2rem; }
h2 { font-size: 1.2rem; margin-top: 2.2rem; border-bottom: 1px solid currentColor;
     padding-bottom: .3rem; opacity: .95; }
h3 { font-size: 1rem; margin-top: 1.4rem; }
.sub { opacity: .7; margin-top: 0; }
.warn { border-left: 4px solid #c0870a; padding: .7rem 1rem; margin: 1.2rem 0;
        background: rgba(192,135,10,.10); border-radius: 0 6px 6px 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: .8rem 0 1.4rem; font-size: 14px; }
th, td { border: 1px solid rgba(128,128,128,.35); padding: .34rem .55rem; text-align: right; }
th:first-child, td:first-child { text-align: left; white-space: nowrap; }
thead th { background: rgba(128,128,128,.14); }
tbody tr:nth-child(even) { background: rgba(128,128,128,.06); }
code { font-size: .92em; background: rgba(128,128,128,.16); padding: .1em .35em;
       border-radius: 3px; }
pre { background: rgba(128,128,128,.12); padding: .8rem; border-radius: 6px; overflow-x: auto; }
.pos { color: #1a7f37; font-weight: 600; }
.neg { color: #b3261e; font-weight: 600; }
footer { margin-top: 3rem; opacity: .65; font-size: 13px; }
"""


def _median(values):
    return float(statistics.median(values)) if values else float("nan")


def _fmt(value, digits=3):
    if value is None:
        return "&mdash;"
    if isinstance(value, float) and value != value:
        return "&mdash;"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _signed(value, digits=2):
    if value is None or (isinstance(value, float) and value != value):
        return "&mdash;"
    css = "pos" if value > 0 else "neg" if value < 0 else ""
    return f'<span class="{css}">{value:+.{digits}f}</span>'


def _table(headers, rows) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return (
        '<div class="scroll"><table>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody>"
        "</table></div>"
    )


def _screen_a(summary: dict) -> str:
    parts = []
    for scene, block in summary["screen_a"].items():
        rows = []
        for arm, by_seed in block["records"].items():
            records = list(by_seed.values())
            if not records:
                continue
            contrast = block["contrast_vs_ci"].get(arm)
            rows.append(
                [
                    f"<code>{html.escape(arm)}</code>",
                    _fmt(_median([r["init_alpha_iou"] for r in records]), 3),
                    _fmt(_median([r["init_alpha_mean"] for r in records]), 3),
                    _fmt(_median([r["q_init"] for r in records]), 2),
                    _fmt(_median([r["q_final"] for r in records]), 2),
                    _fmt(_median([r["ssim_final"] for r in records]), 4),
                    _signed(contrast["delta_median"]) if contrast else "reference",
                    f"{contrast['material_wins']}/{contrast['seeds']}" if contrast else "&mdash;",
                    html.escape(contrast["verdict"]) if contrast else "&mdash;",
                    _fmt(_median([r["init_seconds"] for r in records]), 1),
                ]
            )
        parts.append(f"<h3>{html.escape(scene)}</h3>")
        parts.append(
            _table(
                [
                    "arm",
                    "init alpha IoU",
                    "init alpha mean",
                    "Q@0 (dB)",
                    "Q@final (dB)",
                    "SSIM",
                    "&Delta; vs ci (dB)",
                    "material wins",
                    "verdict",
                    "init s",
                ],
                rows,
            )
        )
    return "".join(parts)


def _screen_b(summary: dict) -> str:
    parts = []
    for scene, block in summary.get("screen_b", {}).items():
        parts.append(f"<h3>{html.escape(scene)}</h3>")
        rows = []
        for init_name, levels in block["records"].items():
            for level, by_seed in levels.items():
                records = list(by_seed.values())
                if not records:
                    continue
                rows.append(
                    [
                        f"<code>{html.escape(init_name)}</code>",
                        f"<code>{html.escape(level)}</code>",
                        _fmt(_median([r["q_final"] for r in records]), 2),
                        _fmt(_median([r["ssim_final"] for r in records]), 4),
                        _fmt(int(_median([r["n_final"] for r in records])), 0),
                        _fmt(_median([r["end_to_end_seconds"] for r in records]), 0),
                    ]
                )
        parts.append(
            _table(
                ["init", "growth / trust", "Q@final (dB)", "SSIM", "n final", "e2e s"],
                rows,
            )
        )
        analysis = block["analysis"]
        effect_rows = []
        levels = sorted({lv for e in analysis["effects"].values() for lv in e})
        for level in levels:
            effect_rows.append(
                [
                    f"<code>{html.escape(level)}</code>",
                    *[
                        _signed(analysis["effects"].get(init, {}).get(level))
                        for init in analysis["effects"]
                    ],
                    _signed(analysis["interaction_contrast_db"].get(level)),
                ]
            )
        parts.append("<h4>Schedule effect per init, and the interaction contrast</h4>")
        parts.append(
            "<p>Each cell is the median held-out gain over that init's own "
            "<code>none/notrust</code> baseline. The last column is the difference between the "
            "two inits' effects: that difference <em>is</em> the ADR-YYYY thesis, and a value "
            "inside the noise band refutes it on this scene.</p>"
        )
        parts.append(
            _table(
                ["growth / trust", *[html.escape(i) for i in analysis["effects"]], "interaction"],
                effect_rows,
            )
        )
    return "".join(parts)


def render(summary: dict, run: Path) -> Path:
    config = summary["config"]
    metrics_exists = (run / "metrics.json").is_file()
    ply_exists = (run / "gaussians.ply").is_file()
    previews = [
        name
        for name in (
            "reconstruction_contact_sheet.png",
            "reconstruction.gif",
            "novel_orbit.gif",
            "novel_elevation.gif",
        )
        if (run / name).is_file()
    ]
    preview_html = "".join(
        f'<figure><img src="{name}" alt="{name}" style="max-width:100%"/>'
        f"<figcaption><code>{name}</code></figcaption></figure>"
        for name in previews
    )
    links = ['<li><a href="summary.json">summary.json</a> &mdash; every cell, every seed</li>']
    if metrics_exists:
        links.append(
            '<li><a href="metrics.json">metrics.json</a> &mdash; representative-run metrics</li>'
        )
    if ply_exists:
        links.append(
            '<li><a href="gaussians.ply">gaussians.ply</a> and '
            '<a href="gaussians_init.ply">gaussians_init.ply</a> &mdash; viewer handoff</li>'
        )
    if (run / "training_history.json").is_file():
        links.append('<li><a href="training_history.json">training_history.json</a></li>')
    if (run / "viewer_smoke.json").is_file():
        links.append(
            '<li><a href="viewer_smoke.json">viewer_smoke.json</a> &mdash; page and viewer '
            "receipts</li>"
        )
    viewer_block = ""
    if (run / "comparison.json").is_file():
        methods = json.loads((run / "comparison.json").read_text())["methods"]
        viewer_block = f"""
<h2>See it</h2>
<p><a href="comparison.json">comparison.json</a> lists {len(methods)} named initial/final pairs
&mdash; every Screen-A arm and every Screen-B schedule level on the representative scene &mdash;
under one orbit camera. Treat the WebGL view as a diagnostic; the numbers above come from the
exact rasterizer.</p>
<pre>.venv/bin/rtgs view --comparison-manifest {html.escape(str(run.name))}/comparison.json \\
    --rasterizer gsplat --device cuda</pre>"""

    page = f"""<title>ADR-XXXX / ADR-YYYY init-parameter and schedule screen</title>
<style>{STYLE}</style>
<h1>Init parameters and the schedule that keeps them</h1>
<p class="sub">ADR-XXXX (surfel lift) and ADR-YYYY (init-preserving densification and trust
schedule), measured on the two alpha-bearing Stage frames.</p>

<div class="warn">
<strong>Development-only evidence.</strong> {html.escape(summary["status"])}.
Stage frames are outcome-exposed under §1.2 of the master protocol: earlier experiments in this
repository selected variants on them. Nothing on this page may enter <code>README.md</code>,
<code>docs/</code>, or <code>ara/logic/claims.md</code> as a method claim. These frames are used
because they are the only compact bundles here that carry packed alpha, and ADR-XXXX §3 solves
against exactly that alpha.
</div>

<h2>What was measured</h2>
<p>Held-out quality <code>Q</code> is the median per-view foreground-weighted PSNR against each
unseen view's <em>own compact 2D fit</em>, composited on black &mdash; not against source RGB.
It bounds agreement with Stage 1, not with the camera.</p>
<p><code>init alpha IoU</code> is ADR-XXXX acceptance criterion A2, evaluated
<strong>before</strong> any optimizer step: the intersection over union of the lifted state's
rendered alpha and the ground-truth foreground, at threshold 0.5, on the fitted views. A lift
whose means are right but whose covariance and opacity are wrong scores near zero here, which is
the failure this ADR exists to fix.</p>
<pre>{html.escape(json.dumps(config, indent=2))}</pre>

<h2>Screen A &mdash; the init parameters (fixed topology)</h2>
{_screen_a(summary)}

<h2>Screen B &mdash; the schedule, crossed with the init</h2>
{_screen_b(summary)}

{viewer_block}

<h2>Artifacts</h2>
<ul>{"".join(links)}</ul>
{"<h2>Previews</h2>" + preview_html if preview_html else ""}

<footer>
Generated from <code>summary.json</code> by
<code>benchmarks/surfel_init_schedule_report.py</code>.
Wall clock {_fmt(summary.get("wall_clock_seconds", 0.0) / 60.0, 1)} min on
{html.escape(str(summary["environment"].get("gpu", "cpu")))}.
</footer>
"""
    target = run / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/surfel_init_schedule_screen")
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    print(render(summary, run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
