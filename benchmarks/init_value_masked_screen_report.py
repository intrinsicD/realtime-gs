#!/usr/bin/env python3
"""Render the masked screen's ``summary.json`` into a self-contained ``index.html``.

Relative links only, no external assets, so the page works straight out of the run directory and
satisfies the results-page half of Hard Rule 7.  It reports what the run measured and, just as
importantly, what the run could not measure.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from benchmarks.init_value_dev_screen_report import (
    ARM_ORDER,
    ARM_STYLE,
    fmt,
    line_chart,
    mad,
    mean_curve,
    median,
    signed,
)

SUPERVISION_NOTE = {
    "full": "trained on each view's complete 2D fit (identical to the unmasked Karate screen)",
    "masked": "trained with the masked loss: target composited on black, weighted L1, "
    "outside-alpha penalty",
}


def cell_rows(cell: dict, supervision: str) -> list[dict]:
    rows = []
    records = cell["records"].get(supervision, {})
    contrasts = cell.get("contrasts", {}).get(supervision, {})
    for arm in ARM_ORDER:
        by_seed = records.get(arm)
        if not by_seed:
            continue
        q = [r["q_final"] for r in by_seed.values()]
        contrast = contrasts.get(arm) if isinstance(contrasts, dict) else None
        rows.append(
            {
                "arm": arm,
                "q0": median([r["q_init"] for r in by_seed.values()]),
                "q": median(q),
                "q_mad": mad(q),
                "crop": median([r["q_crop_final"] for r in by_seed.values()]),
                "ssim": median([r["ssim_final"] for r in by_seed.values()]),
                "seconds": median([r["end_to_end_seconds"] for r in by_seed.values()]),
                "delta": contrast["quality_delta_median"] if contrast else None,
                "wins": (
                    f"{contrast['quality_material_wins']}/{contrast['quality_seeds']}"
                    if contrast
                    else "—"
                ),
                "verdict": contrast["quality_verdict"] if contrast else "reference arm",
                "time_ratio": contrast["time_ratio_median"] if contrast else None,
                "time_verdict": contrast["time_verdict"] if contrast else "—",
            }
        )
    return rows


def arm_table(cell: dict, supervision: str) -> str:
    rows = cell_rows(cell, supervision)
    parts = [
        "<table><thead><tr><th>Arm</th><th>Q<sub>fg</sub>@0</th><th>Q<sub>fg</sub>@final</th>"
        "<th>MAD</th><th>ΔQ vs ci</th><th>Wins</th><th>crop PSNR</th><th>crop SSIM</th>"
        "<th>End-to-end s</th><th>Time to target</th><th>Verdict</th></tr></thead><tbody>"
    ]
    for row in rows:
        color, role = ARM_STYLE.get(row["arm"], ("#111827", ""))
        parts.append(
            f"<tr><td><span class='key'><i style='background:{color}'></i>"
            f"{html.escape(row['arm'])}</span><br><span class='muted'>{html.escape(role)}</span>"
            f"</td>"
            f"<td>{fmt(row['q0'], 2)}</td><td><strong>{fmt(row['q'])}</strong></td>"
            f"<td>{fmt(row['q_mad'])}</td>"
            f"<td>{signed(row['delta']) if row['delta'] is not None else '—'}</td>"
            f"<td>{html.escape(str(row['wins']))}</td>"
            f"<td>{fmt(row['crop'], 2)}</td><td>{fmt(row['ssim'], 4)}</td>"
            f"<td>{fmt(row['seconds'], 1)}</td>"
            f"<td>{fmt(row['time_ratio'], 2, '×') if row['time_ratio'] is not None else '—'}"
            f"<br><span class='muted'>{html.escape(str(row['time_verdict']))}</span></td>"
            f"<td>{html.escape(str(row['verdict']))}</td></tr>"
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def render(summary: dict, out: Path) -> Path:
    cells = summary["cells"]
    complete = {k: v for k, v in cells.items() if v.get("status") == "complete"}
    anchor = summary["anchor"]
    anchor_cells = {
        k: v
        for k, v in complete.items()
        if v["viewset"] == anchor["viewset"] and v["n_init"] == anchor["n_init"]
    }
    supervisions = summary["config"]["supervisions"]

    css = """
:root{--fg:#111827;--muted:#6b7280;--bg:#ffffff;--panel:#f8fafc;--line:#e5e7eb;--warn:#b45309;
--warnbg:#fffbeb}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 4rem;font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',
Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg)}
main{max-width:1120px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .35rem;letter-spacing:-.01em}
h2{font-size:1.2rem;margin:2.5rem 0 .75rem;padding-bottom:.35rem;
border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.75rem 0 .5rem}
p{margin:.6rem 0}
.sub{color:var(--muted);margin:0 0 1.25rem}
.banner{background:var(--warnbg);border:1px solid #fcd34d;border-left:4px solid var(--warn);
padding:.85rem 1rem;border-radius:6px;margin:1rem 0 1.5rem}
.banner strong{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.6rem 0 1rem}
th,td{text-align:right;padding:.42rem .6rem;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--muted);font-size:.82rem;text-transform:uppercase;
letter-spacing:.03em}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.8rem 1rem;
margin:.6rem 0}
.muted{color:var(--muted);font-size:.85rem}
.key{display:inline-flex;align-items:center;gap:.4rem;white-space:nowrap}
.key i{width:.7rem;height:.7rem;border-radius:2px;display:inline-block}
.grid{stroke:var(--line);stroke-width:1}
.axis{stroke:#9ca3af;stroke-width:1}
.tick{font-size:10px;fill:var(--muted)}
.axlabel{font-size:11px;fill:var(--muted)}
.chart{width:100%;height:auto}
figure{margin:.5rem 0}
.legend{font-size:.8rem;color:var(--muted);display:flex;gap:.9rem;flex-wrap:wrap;margin-top:.3rem}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}
code{background:var(--panel);padding:.1rem .3rem;border-radius:3px;font-size:.85em}
ul{margin:.4rem 0 .4rem 1.1rem;padding:0}
li{margin:.25rem 0}
"""

    parts = ["<main>"]
    cfg = summary["config"]
    env = summary["environment"]
    parts.append("<h1>Initialization Value Program — masked development screen</h1>")
    parts.append(
        f"<p class='sub'>{html.escape(cfg['topology'])} · {cfg['iterations']} updates · "
        f"seeds {', '.join(str(s) for s in cfg['seeds'])} · "
        f"{html.escape(str(env.get('gpu', 'cpu')))}</p>"
    )
    parts.append(
        "<div class='banner'><strong>DEVELOPMENT-ONLY — NOT CONFIRMATORY — OUTCOME-EXPOSED "
        "INPUTS.</strong><ul>"
        + "".join(f"<li>{html.escape(limit)}</li>" for limit in summary["evidence_limits"])
        + "</ul></div>"
    )

    parts.append("<h2>What was run</h2><table><tbody>")
    for key, value in [
        (
            "Protocol",
            f"{summary['protocol']['path']} <code>{summary['protocol']['sha256'][:16]}…</code>",
        ),
        (
            "Review",
            f"{summary['review']['path']} <code>{summary['review']['sha256'][:16]}…</code>",
        ),
        (
            "Unmasked companion",
            f"{summary['companion_run']['path']} "
            f"<code>{summary['companion_run']['sha256'][:16]}…</code>",
        ),
        ("Primary endpoint", cfg["primary_endpoint"]),
        ("Secondary", ", ".join(cfg["secondary"])),
        ("Supervision factor", " · ".join(f"{s}: {SUPERVISION_NOTE[s]}" for s in supervisions)),
        ("Anchor", f"{anchor['viewset']} train views, N={anchor['n_init']}"),
        ("Target rule", cfg["tau_rule"]),
        ("Rasterizer", f"{cfg['rasterizer']} · {cfg['device']}"),
        (
            "Environment",
            f"torch {env.get('torch')} · gsplat {env.get('gsplat')} · CUDA {env.get('cuda')}",
        ),
        (
            "Source",
            f"<code>{str(env.get('git_head'))[:12]}</code> "
            f"{'(dirty)' if env.get('git_dirty') else ''}",
        ),
        ("Wall clock", f"{summary.get('total_seconds', 0) / 60:.1f} min"),
    ]:
        parts.append(
            f"<tr><th>{html.escape(key)}</th><td style='text-align:left'>{value}</td></tr>"
        )
    parts.append("</tbody></table>")

    parts.append("<h2>Scenes, masks, and splits</h2>")
    parts.append(
        "<div class='panel'><p>Stage frames have no section 2.2 split. The rule below was fixed "
        "before the run and reads no metric; validation and report-only IDs are inherited from the "
        "frozen Karate protocol and removed from the pool before selection.</p></div>"
    )
    parts.append(
        "<table><thead><tr><th>Scene</th><th>Views</th><th>With alpha</th><th>Train (3)</th>"
        "<th>Train (+5 → 8)</th><th>Validation</th><th>Report-only</th><th>Foreground</th>"
        "</tr></thead><tbody>"
    )
    for name, spec in summary["scenes"].items():
        extra = [v for v in spec["train8"] if v not in spec["train3"]]
        fg = spec.get("foreground_fraction", {})
        fg_range = f"{100 * min(fg.values()):.1f}–{100 * max(fg.values()):.1f}%" if fg else "—"
        parts.append(
            f"<tr><td>{html.escape(name)}</td><td>{spec['n_views_total']}</td>"
            f"<td>{spec['n_views_with_alpha']}</td>"
            f"<td>{html.escape(','.join(spec['train3']))}</td>"
            f"<td>{html.escape(','.join(extra))}</td>"
            f"<td>{html.escape(','.join(spec['validation']))}</td>"
            f"<td>{html.escape(','.join(spec['report_only_sealed']))} (sealed)</td>"
            f"<td>{fg_range}</td></tr>"
        )
    parts.append("</tbody></table>")

    parts.append(
        f"<h2>Anchor result ({anchor['viewset'][1:]} train views, N={anchor['n_init']})</h2>"
    )
    parts.append(
        "<div class='panel'><p>Seed-median held-out foreground PSNR at the final checkpoint, "
        "paired per seed against the comparator. The material margin is 0.25 dB; “wins” counts "
        "seeds clearing it.</p></div>"
    )
    for key, cell in anchor_cells.items():
        parts.append(f"<h3>{html.escape(key)}</h3>")
        for supervision in supervisions:
            parts.append(
                f"<p class='muted'><strong>supervision = {html.escape(supervision)}</strong> — "
                f"{html.escape(SUPERVISION_NOTE[supervision])}</p>"
            )
            parts.append(arm_table(cell, supervision))
        series_parts = []
        for supervision in supervisions:
            series = [
                (arm, mean_curve(cell["records"][supervision][arm]))
                for arm in ARM_ORDER
                if arm in cell["records"].get(supervision, {})
            ]
            series_parts.append(
                f"<div><p class='muted'>supervision = {html.escape(supervision)}</p>"
                + line_chart(
                    series,
                    x_key="step",
                    y_key="psnr",
                    x_label="optimizer updates",
                    y_label="held-out foreground PSNR (dB)",
                )
                + "</div>"
            )
        parts.append(f"<div class='cols'>{''.join(series_parts)}</div>")

    if len(supervisions) > 1:
        parts.append("<h2>Does masked supervision change the answer?</h2>")
        parts.append(
            "<div class='panel'><p>Paired per (arm, seed): masked-supervision Q minus "
            "full-supervision Q, on the same masked endpoint and from a bit-identical "
            "initialization. Positive means training on the mask helped the masked score. The "
            "last column is the thing that matters for the protocol: whether the candidate's "
            "margin over the comparator survives the change.</p></div>"
        )
        parts.append(
            "<table><thead><tr><th>Cell</th><th>Arm</th><th>Δ from masked supervision</th>"
            "<th>MAD</th><th>ΔQ vs ci (full)</th><th>ΔQ vs ci (masked)</th></tr></thead><tbody>"
        )
        for key, cell in complete.items():
            sup = cell.get("supervision_contrast", {})
            for arm in ARM_ORDER:
                if arm not in sup:
                    continue
                cf = cell.get("contrasts", {}).get("full", {}).get(arm)
                cm = cell.get("contrasts", {}).get("masked", {}).get(arm)
                cf_d = cf.get("quality_delta_median") if cf else None
                cm_d = cm.get("quality_delta_median") if cm else None
                parts.append(
                    f"<tr><td>{html.escape(key)}</td>"
                    f"<td><span class='key'><i style='background:"
                    f"{ARM_STYLE.get(arm, ('#111827', ''))[0]}'></i>{html.escape(arm)}</span></td>"
                    f"<td>{signed(sup[arm]['delta_median'])}</td>"
                    f"<td>{fmt(sup[arm]['delta_mad'])}</td>"
                    f"<td>{signed(cf_d) if cf_d is not None else '—'}</td>"
                    f"<td>{signed(cm_d) if cm_d is not None else '—'}</td>"
                    f"</tr>"
                )
        parts.append("</tbody></table>")

    parts.append("<h2>Every cell</h2>")
    parts.append(
        "<div class='panel'><p>Descriptive by design: the anchor is the decision cell, and the "
        "other cells only show whether the direction survives a change of view count or primitive "
        "count.</p></div>"
    )
    parts.append(
        "<table><thead><tr><th>Cell</th><th>Sup.</th><th>Arm</th><th>Q<sub>fg</sub></th>"
        "<th>MAD</th><th>ΔQ vs ci</th><th>Wins</th><th>crop PSNR</th><th>crop SSIM</th>"
        "<th>End-to-end s</th><th>N</th></tr></thead><tbody>"
    )
    for key, cell in complete.items():
        for supervision in supervisions:
            for row in cell_rows(cell, supervision):
                parts.append(
                    f"<tr><td>{html.escape(key)}</td><td>{html.escape(supervision)}</td>"
                    f"<td>{html.escape(row['arm'])}</td><td>{fmt(row['q'])}</td>"
                    f"<td>{fmt(row['q_mad'])}</td>"
                    f"<td>{signed(row['delta']) if row['delta'] is not None else '—'}</td>"
                    f"<td>{html.escape(str(row['wins']))}</td><td>{fmt(row['crop'], 2)}</td>"
                    f"<td>{fmt(row['ssim'], 4)}</td><td>{fmt(row['seconds'], 1)}</td>"
                    f"<td>{cell['n_init']}</td></tr>"
                )
    parts.append("</tbody></table>")

    parts.append("<h2>What this screen does not establish</h2>")
    parts.append(
        "<div class='panel'><ul>"
        "<li>No confirmatory evidence of any kind. Stage frames are outcome-exposed pilot inputs; "
        "the frozen protocol requires three newly acquired scenes for a pipeline claim.</li>"
        "<li>No comparison against structure-from-motion. The COLMAP arm could not be built and "
        "section 3.3 forbids converting that absence into a win.</li>"
        "<li>No held-out RGB quality. The target is each view's own 2D fit restricted to the "
        "ground-truth mask, so these numbers bound agreement with Stage 1, not with the "
        "camera.</li>"
        "<li>No claim about dynamic densification. Topology is fixed and birth is disabled.</li>"
        "<li>Timing is one consumer GPU with no interleaving; treat seconds as indicative.</li>"
        "</ul></div>"
    )

    parts.append("<h2>Artifacts</h2><ul>")
    parts.append("<li><a href='summary.json'>summary.json</a> — every number on this page</li>")
    parts.append(
        "<li><code>cells/&lt;scene&gt;/&lt;viewset&gt;/N&lt;count&gt;/&lt;supervision&gt;/"
        "&lt;arm&gt;/&lt;seed&gt;/</code> — per-run <code>record.json</code>, "
        "<code>gaussians_init.ply</code>, <code>gaussians_final.ply</code></li>"
    )
    parts.append("</ul>")

    bundle = [
        ("gaussians.ply", "representative final gaussians (viewer handoff)"),
        ("gaussians_init.ply", "representative initial gaussians"),
        ("metrics.json", "representative per-view masked metrics"),
        ("training_history.json", "representative quality curve"),
        ("gaussians.config.json", "protocol, config, and environment of the representative run"),
        ("viewer_smoke.json", "results-page and viewer smoke receipt"),
    ]
    present = [(name, note) for name, note in bundle if (out / name).exists()]
    if present:
        parts.append("<h2>Representative bundle</h2>")
        parts.append(
            "<div class='panel'><p>One run promoted to the run root for viewer handoff, chosen by "
            "a fixed outcome-independent rule (lexicographically first anchor cell, masked "
            "supervision, candidate arm, lowest seed). It is a handoff and a preview, not a "
            "result.</p></div><ul>"
        )
        for name, note in present:
            parts.append(f"<li><a href='{name}'>{name}</a> — {note}</li>")
        parts.append("</ul>")

    previews = [p.name for p in sorted(out.glob("*.png")) + sorted(out.glob("*.gif"))]
    if previews:
        parts.append("<h2>Previews</h2><ul>")
        for name in previews:
            parts.append(f"<li><a href='{name}'>{name}</a></li>")
        parts.append("</ul>")

    parts.append("</main>")
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Initialization Value Program — masked development screen</title>"
        f"<style>{css}</style></head><body>{''.join(parts)}</body></html>"
    )
    path = out / "index.html"
    path.write_text(page)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/init_value_masked_screen"))
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    path = render(summary, run)
    print(f"[written] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
