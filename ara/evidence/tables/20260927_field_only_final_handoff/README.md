# RTGS-025 field-only distillation: final handoff receipts

Tracked copies of run-root receipts for `runs/20260926_field_only_distillation_stage_frame00008/` (the run root is local and git-ignored).

- `run_receipt.json`, `preflight.json`: single non-development run, exit 0.
- `launch_context.json`: the GPU was shared at launch with an unrelated process; timings are contended.
- `results_audit_authorization.json`, `results_audit_payload_manifest.json`: the user's explicit approval
  for the Fable 5.1 results audit to read 22 dome-derived preview PNGs, and the exact 126-item payload.
- `http_link_mirrors.json`: byte-identical evidence copies placed inside the run root so outward relative
  report links resolve under the run-root HTTP server (as in RTGS-016).
- `viewer_smoke.json`: headless Chrome 149 (ANGLE SwiftShader, WebGL2) report and viewer check. The user
  was away, so the Claude app browser pane could not be shown; the check used the RTGS-016
  `browser_check_source.py` with a re-created minimal CDP client (`cdp_client.py.txt`). The frozen report
  port 8765 was occupied by an unrelated local server, so the report was served on 8766; the viewer used
  the frozen argv on port 8879. A neutral favicon (copied from RTGS-016) removed the only failed resource.
  The Driver visually inspected the viewer screenshot (final model, 42863 splats, subject visible).
- After the viewer receipt the report was rendered again; `check-run` and
  `scripts/check_results_bundle.py` (with previews) both pass; 974 report targets return HTTP 200.
