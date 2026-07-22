# Full `frame_00008` compact-initializer suite — independent results audit

## Verdict

**Accept with narrowing.** The six prospective compact-compatible arms completed their frozen
schedule, the historical beam anchor remains hash-bound to its prior accepted audit, and a fresh
compact-only CUDA replay reproduced every reported selected-model metric exactly. The valid
conclusion is `NO_MATERIALLY_SUPERIOR_CONVERGED_INITIALIZER`. No production default, held-out
quality, generalization, causal initializer benefit, or portable timing claim is authorized.

The result JSON SHA-256 is
`f9e64398f141c53c61816c31ed246285ab9832015199dbb4f9a7f0dd2f436953`. The machine-readable audit
JSON SHA-256 is `296093fbf6ee4c5917b97b7a123ea27689c8f85ec9751908126a1cfd8cf45d24`.

## Claim table

| Claim | Kind and scope | Disposition | Independent evidence |
| --- | --- | --- | --- |
| Every repository initializer applicable to the compact-only bundle was compared. | Capability/inventory | **Confirm narrowly.** | Six prospective compact arms completed and beam is a disclosed historical anchor. Gradient, legacy carve, depth, hybrid, and classic SfM are explicitly inapplicable because required RGB/depth/points are absent. |
| Dense+merge has the best converged foreground PSNR. | Measured, one scene, fitted views | **Confirm.** | Raw and replayed value is 38.248049 dB, 0.360674 dB above beam. |
| Dense+merge is the materially superior converged initializer. | Comparative | **Retire.** | Its objective is 0.002554868 versus beam's better 0.002447185: 4.4003% worse, failing the second frozen gate. |
| Beam fusion is the converged winner. | Comparative | **Retire.** | Beam has the best objective and crop SSIM but is 0.360674 dB behind dense on foreground PSNR. |
| The suite identifies a production default. | Default/capability | **Retire.** | The Pareto front splits; all views were fit; there is one scene, one seed, native count confounding, and no held-out evidence. |
| Initializer choice caused the terminal ranking. | Causal | **Not tested.** | Adaptive density changed every topology, ending at 35,644–49,177 Gaussians. Random finished fourth. |
| Splat-SfM works mechanically on the real compact bundle. | Measured, initialization | **Confirm narrowly.** | It yielded 943 finite tracks under frozen gates and completed training; unmatched coverage remains very high. |
| Complete field lift works mechanically on the real compact bundle. | Measured, initialization | **Confirm with receipt caveat.** | It returned 127 finite Gaussians without fallback and completed training, but individual topology move receipts are absent. |
| Result quality generalizes to held-out/novel views or other scenes/seeds. | Generalization | **Retire.** | All 26 cameras entered placement, fit, selection, and stopping. `V` is a fitted-view subset. |
| Placement/training timings compare method speed. | Performance | **Retire.** | Sequential single executions ran on a contended, non-randomized machine without warm repeats. |
| A CPU viewer has zero training impact. | Performance | **Retire.** | The viewer did not run during this suite. CPU/RAM/I/O and browser WebGL remain nonzero without an on/off trial. |

## Chronology, isolation, and source binding

The protocol was frozen with the earlier beam result and top-K initialization metric disclosed;
the six downstream prospective outcomes were unopened. This makes the suite prospective
descriptive development evidence, not a fresh confirmatory beam selection.

Every prospective parent binds protocol SHA-256
`217a4fecceca161f4291e78e0e53b201be3e1560e33a875bd29a9fd54534aaf6`, harness SHA-256
`47fb0492c646766f88bc2e752870003ba4f8bd45f366880400d60b4183bc4e93`, revision
`d74c9a623cba8af4694e0112753927407c7fdab5`, and the same 61-file executed-source manifest. The
audit rehashed every snapshot for every arm and found no mismatch. The suite operator hash
`e398817f8b901c98be9177362962c13a6742ac43217d18dc73b04cf0ed9a4f0f` matches the protocol and
current preserved file; unlike the scientific parent source, it was not copied into each parent
snapshot. This is a preservation weakness but not a computation-path ambiguity because its
resume/status outputs and commands are retained.

All parents bind manifest SHA-256
`b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847`, calibration SHA-256
`51b8fc396fc8447f24e325e0a525f2e7d422388790dd9a293e1a81804b265091`, and identical hashes/sizes
for 26 bundles totaling 130,000 components. All 30 prospective phase directories share canonical
compact-target identity `388abaf82cb164413abe5f3b0375c587bad1445bb77c38bf7257618680d82f0d`.
No `evaluation_request.json` or `original_metrics.json` exists under the prospective suite, and
`suite_status.json` records `source_rgb_opened=false`.

## Independent recomputation

The audit reopened raw placement, initial-metric, history, model-selection, compact-metric, fit,
status, and prior beam JSON rather than copying reporter output. It independently recomputed:

- every per-view model-selection objective from its RGB/SSIM/alpha terms;
- every candidate's equal-view mean;
- earliest-within-`1e-6` checkpoint selection;
- all five-transition materiality decisions;
- the last-six Theil–Sen/median/per-view trend rule;
- each segment's joint status and every terminal plateau;
- initial and final foreground-PSNR rankings;
- the dense-versus-beam PSNR and objective gates; and
- the empty frozen practical-equivalence intersection and two-arm Pareto front.

All 10,012 fail-closed checks passed. The audit CPU-loaded **482 PLY artifacts** (4,630,567,983
bytes), including every prospective 1k checkpoint, every phase init/final, and beam init/selected
models. Counts matched receipts, all tensors were finite, opacity/quaternion invariants passed,
and the observed range 7–49,177 stayed below the 100,000 cap.

Finally, the audit independently rematerialized all 26 compact teachers from the checked-in
bundles and reevaluated the seven selected PLYs with CUDA gsplat, packed antialiased rendering.
The teacher identity matched exactly and every foreground/full/crop PSNR, crop SSIM, alpha IoU,
inside alpha, and outside alpha value available in the result replayed with **zero recorded
floating-point delta**. This replay did not load source RGB.

## Findings and corrections

1. **No single winner.** Dense leads the primary display metric, while beam leads the selected
   training objective and SSIM. Public prose must report both and the failed joint gate.
2. **Native counts confound the rank.** Initial cardinalities span 7–5,000 and density-stop counts
   span 35,644–49,177. The suite intentionally did not post-hoc trim methods; count and quality
   must stay in the same table.
3. **All views are fitted.** The configured `T/V/H` labels survive for compatibility, but fit mode
   is `all`; even `V` and `H` entered optimization. No validation, held-out, or novel-view language
   is permitted.
4. **Field topology receipts are incomplete.** Aggregate diagnostics say seven proposals and one
   acceptance, but the harness omitted the individual move receipts required by the protocol.
   Final PLY/count/quality remain valid; topology move-level utility is unaudited. A deterministic
   post-run replay would not be the missing execution-time receipt and was not substituted.
5. **Timings are diagnostic only.** Placement, optimizer, callback, and VRAM values are retained
   for accounting but cannot rank performance.
6. **Incompatible methods did not lose.** RGB/depth/SfM-dependent methods were not run because the
   compact-only dataset lacks their inputs. A separate, named cohort is required to compare them.

## Verification commands

```bash
.venv/bin/python benchmarks/summarize_compact_initializer_suite.py

CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  benchmarks/audit_compact_initializer_suite.py \
  --output /tmp/rtgs_compact_suite_audit_dryrun_20260721.json

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/audit_compact_initializer_suite.py \
  --replay-metrics
```

The audit also rehashed every source/input/result binding and recomputed the decision directly
from raw JSON. The expensive initializer/training suite was not rerun. No fresh RGB/depth/SfM
cohort, held-out split, additional scene/seed, matched-cardinality run, clean repeated performance
benchmark, or viewer on/off trial was performed.

Task-focused CPU verification passed all **199** collected compact-carve/gate/evaluation/suite,
recovery, splat-SfM, field, beam, and viewer tests. Ruff check, Ruff format check, docs-sync, and
`git diff --check` pass. The repository-wide non-slow gate is not green: 16 of 1,502 collected
tests fail, with 1,475 pass and 11 skip. None of the failures touches the suite, initializer,
trainer, metric, audit, or viewer paths changed here:

- six occupancy-factorial tests reject changed frozen system `libstdc++`/PyTorch source bindings;
- two responsibility-birth tests reject the same frozen `libstdc++` binding;
- one capacity-crossover test requires the absent historical `libstdc++.so.6.0.33`; and
- seven G2SR diagnostic tests require the absent
  `runs/compact_masked_bundle_640_20260717/reconstruction_inputs/manifest.json`.

These are fail-closed historical-environment/artifact checks, not evidence that the entire
repository passes. They were preserved rather than weakened to make this result green.

## Evidence needed to promote a claim

- A default change requires a preregistered multi-scene, multi-seed, train-only selection protocol
  with genuinely held-out cameras and matched or explicitly modeled capacity/budget effects.
- A causal initializer claim requires matched downstream topology/budget controls or a density-
  disabled attribution arm, not merely the terminal adaptive-density rank.
- A performance claim requires an idle named GPU, randomized arm order, warmup, repeats, and a
  frozen aggregation rule.
- Field topology utility requires execution-time serialization of every proposal/receipt before a
  fresh run; replay cannot repair this result retrospectively.
- A zero-overhead viewer claim requires controlled randomized viewer-on/off repeats. CPU mode can
  avoid a viewer-server CUDA allocation, but cannot eliminate CPU, RAM, storage, network, or
  browser display-GPU work.
