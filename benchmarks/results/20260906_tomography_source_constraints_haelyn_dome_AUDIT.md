All evidence is in hand. Writing the audit now.

# RTGS-016 independent results audit

## Verdict, provenance, and checks executed

**Verdict: accepted_with_limits.** The run completed, every producer number I could recompute reproduces exactly, and the prospective chronology holds. The limits are interpretive and substantial: the proxy stage is seed-invariant, the free arm's gains come from footprint drift and blur, no arm yields a recognizable reconstruction, and the prospectively selected soft tether is not a consistent improvement. Final report rendering, bundle, viewer, and browser validation have not happened and are listed below as remaining handoff work, not as failures.

**Reviewer and scope.** Claude-Code-Fable-5.1-reviewer, model claude-fable-5-1, effort max, no nested agents, no fallback model. Producer narratives were treated as untrusted. I wrote no files, ran no reconstruction, re-rendered nothing, and did not open raw dome photographs, masks, or full Gaussian archives. The 12 dome-derived previews were viewed under the user's explicit approval recorded in the authorization file.

**Run status.** The coordinator exit record shows exit code 0 and the run receipt shows status completed with no failure phase. No failure artifacts exist anywhere under the run root.

| Item | Value |
|---|---|
| Protocol SHA-256, recomputed via review-digest | 3b0b25fbb0721b138cfcb42ddfeddbc5dc7f0e883ab53dbea8d2c045931600e7 |
| Live source binding aggregate, 122 files | 3a3a7f0c1e1dc92c300c43e7a0ca53aa3b1e780fb2dcd8bfb39e93174d0c1430 |
| Source commit, dirty development lock | 2ebe52cc28a1b0e812a6c436d49225fac4ee943f |
| Lock source_diff_sha256, equal to archived development_source_state.bin | 1e2a84c11827d6b6642c21f63abc0b89a30b1deac612d90f2a8cce2e2f82f07e |
| Task file | 7fd81d81cf6ef346a0255d5fdc526a9827e1c643431f04911a2e1154119e125a |
| Approved protocol review artifact | 59023aeba8aee8240817af6af029e918f92a85171a408d0935e864897c18213c |
| Rejected V1 review, preserved | 5229175043fd0fce7053c45f64d11950621885cff758d58ee6f0853bec0d8ed2 |
| Compact data seal | 9085780d67dca0a8df1a8b7485153fc66fe22a527745935fd0338196b45dbf19 |
| External reference seal | a0b9173030b242badb52d22fa0240ee8e8afafbda2fb33b25eed428a838a573c |
| Source snapshot archive | 39863faf3359298ade883af9b87ab6a32f370191b6a3b0c762908f4d3ed12a9d |
| RESULT.json, byte-identical to run cell_results.json | cd23a1cf260d6fb206585569eb6eb765f5387d65fd3b75dc6a3217d4d52ab66e |
| RESULT.md | b98c6bc3dbb1c1fb79af0af351e125317f55b73e576d2043289ff4443492d4ad |
| Payload manifest, equal to authorization record | 43ef926b2c7fd83a02a17c28fc9570bfbdd995ba03b807439e18157a99a070a0 |
| Haelyn reference PLY, hashed opaquely | b9ba7059bfce2a015e719ee4a0de3ea7dad111d7f4ae33d1ad2435e477d0b8c2 |

**Chronology, all UTC on 2026-09-06 unless noted.** V1 rejection was saved at 23:46 on 09-05. The Driver response landed at 00:07. The approved review, the task transition to ready, and the preflight receipt recording a verify.sh exit code of 0 all landed at 00:24:50. The lock was written at 00:25:01, the coordinator launched at 00:25:22 with the frozen command and CPU-only environment, warmup finished by 00:27, and the first measured cell summary appeared at 00:43. The coordinator finished at 09:55:59 after 9.51 hours. The task file, review artifact, protocol digest, data seals, and source binding are unchanged since the lock.

**Commands actually executed.**

```
git status --short
git rev-parse HEAD
git diff --stat
git diff --check
git diff ara/logic/claims.md
sha256sum <task, reviews, seals, lock, snapshot, archive, producer artifacts, payload manifest, reference PLY, representative PLYs>
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260906_tomography_source_constraints_haelyn_dome.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260906_tomography_source_constraints_haelyn_dome.json
.venv/bin/python .scratch/rtgs016/claude_review_02/verify_inputs.py
.venv/bin/python -m pytest -q tests/test_tomography_person_protocol.py tests/test_experiment_contract.py tests/test_field_refit.py
.venv/bin/python -m pytest --co -q <same three files>
.venv/bin/python -c <read-only recomputation scripts, listed below>
```

Results: review-digest printed the protocol digest above; validate printed `experiment_contract: OK`; validate-data printed `experiment_data: OK`; the verify helper reported task validation passed, both seals passed, effective configuration matched, and the live aggregate above. The focused tests collected 53 and all passed with no failure output. The read-only Python scripts independently recomputed: all 252 saved model hashes; initial-state equality; all 36 guard records and view roles; 468 selected-input hashes against the 55-file seal; 36 effective-configuration digests from the recorded configs; 252 validation records and 216 stage markers; 36 proxy objective histories; held-out equal-view MSE from per-view values for 108 evaluations; 792 external PSNR values from MSE; all group medians, 36 paired rows, wall ratios, chart values, and top-level metrics; the archive membership and the lock diff digest; the run validator; and all 90 payload manifest items.

**Denied and not performed.** The permission mode denied `git -C`, `tar -tzf`, and two larger Python scripts; the plain git forms and smaller scripts ran and cover the same checks. I did not run verify.sh, the full CPU suite, the bundle checker, render or check-run, any viewer or browser, any reconstruction or re-render, or the Driver's 1,703-file historical preservation rehash. The raw MSE and SSIM values in the external evaluation come from the producer's evaluator and were checked only for internal consistency.

## Results and claim dispositions

**Per-condition medians across the three seeds.** Held-out teacher MSE is the equal-view mean over four Haelyn or three dome held-out views on full-canvas windows. External metrics are seed medians of per-seed view means at the final endpoint. Initial and proxy values are identical across seeds within an arm.

| Condition | Arm | Carriers | J init | J proxy | J final | Ext full PSNR dB | Ext foreground PSNR dB | Ext SSIM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| haelyn_masked | hard | 217 | 0.073674 | 0.068512 | 0.049790 | 12.962 | 7.219 | 0.5237 |
| haelyn_masked | soft | 217 | 0.073674 | 0.070119 | 0.047711 | 13.190 | 7.183 | 0.6741 |
| haelyn_masked | free | 217 | 0.073674 | 0.049442 | 0.015017 | 17.667 | 13.470 | 0.4277 |
| haelyn_masked | beam_reference | 256 | 0.074960 | 0.075051 | 0.073881 | 11.334 | 5.234 | 0.6777 |
| haelyn_unmasked | hard | 256 | 0.071021 | 0.065803 | 0.046741 | 13.114 | 7.386 | 0.5268 |
| haelyn_unmasked | soft | 256 | 0.071021 | 0.061927 | 0.023092 | 15.835 | 10.903 | 0.4649 |
| haelyn_unmasked | free | 256 | 0.071021 | 0.050510 | 0.014079 | 17.639 | 13.354 | 0.4216 |
| haelyn_unmasked | beam_reference | 256 | 0.073874 | 0.073912 | 0.073678 | 11.266 | 5.167 | 0.6370 |
| dome_masked | hard | 256 | 0.005855 | 0.005120 | 0.004375 | 3.642 | 12.588 | 0.0202 |
| dome_masked | soft | 256 | 0.005855 | 0.005535 | 0.004769 | 3.590 | 11.980 | 0.0103 |
| dome_masked | free | 256 | 0.005855 | 0.004437 | 0.002963 | 3.700 | 14.576 | 0.0348 |
| dome_masked | beam_reference | 256 | 0.005725 | 0.005832 | 0.005432 | 3.577 | 11.341 | 0.0075 |

External SSIM of the nearly empty initial model: 0.6963 masked Haelyn, 0.6979 maskless Haelyn, 0.0028 dome. Dome foreground PSNR of the initial model: 11.005 dB.

**Seed-level paired differences, treatment minus hard, held-out teacher MSE.** Final values are per seed 90601, 90602, 90603. Proxy values are one deterministic number per dataset because the proxy models are byte-identical across seeds.

| Condition | Pair | Final diff per seed | Signs | Final reduction % per seed | Median % | Proxy diff | Proxy reduction % |
|---|---|---|---|---|---:|---:|---:|
| haelyn_masked | soft-hard | -0.002374, -0.002600, -0.001820 | --- | 4.74, 5.22, 3.66 | 4.74 | +0.001607 | -2.35 |
| haelyn_masked | free-hard | -0.035067, -0.034814, -0.034439 | --- | 70.02, 69.92, 69.24 | 69.92 | -0.019070 | 27.83 |
| haelyn_unmasked | soft-hard | -0.023649, -0.023404, -0.023901 | --- | 50.60, 49.99, 51.17 | 50.60 | -0.003875 | 5.89 |
| haelyn_unmasked | free-hard | -0.032662, -0.032299, -0.032828 | --- | 69.88, 68.99, 70.29 | 69.88 | -0.015293 | 23.24 |
| dome_masked | soft-hard | +0.000394, +0.000416, +0.000402 | +++ | -9.00, -9.50, -9.31 | -9.31 | +0.000414 | -8.10 |
| dome_masked | free-hard | -0.001412, -0.001437, -0.001227 | --- | 32.27, 32.79, 28.40 | 32.27 | -0.000683 | 13.34 |

**External RGB paired signs at the final endpoint, treatment minus hard, per seed.**

| Condition | Pair | Full-canvas MSE | Foreground MSE | SSIM |
|---|---|---|---|---|
| haelyn_masked | soft-hard | --- | +-+ | +++ |
| haelyn_masked | free-hard | --- | --- | --- |
| haelyn_unmasked | soft-hard | --- | --- | --- |
| haelyn_unmasked | free-hard | --- | --- | --- |
| dome_masked | soft-hard | +++ | +++ | --- |
| dome_masked | free-hard | --- | --- | +++ |

**Resources, seed medians.** The primary clock runs from driver entry through the saved endpoint and includes imports, source validation, input loading, and validation observers. The secondary clock subtracts measured observer evaluation time only.

| Condition | Arm | Endpoint s | Observer-excluded s | Peak host bytes | Mean drift, whitened RMS | Accepted proxy steps of 100 |
|---|---|---:|---:|---:|---:|---:|
| haelyn_masked | hard | 862.89 | 716.75 | 5463490560 | 0.0 | 100 |
| haelyn_masked | soft | 999.08 | 838.77 | 5447041024 | 0.028107 | 98 |
| haelyn_masked | free | 1078.07 | 899.78 | 5480681472 | 1.071949 | 100 |
| haelyn_masked | beam_reference | 167.88 | 14.71 | 699994112 | n/a | n/a |
| haelyn_unmasked | hard | 1067.81 | 903.49 | 6546898944 | 0.0 | 100 |
| haelyn_unmasked | soft | 1048.26 | 891.28 | 6617714688 | 0.095357 | 99 |
| haelyn_unmasked | free | 992.66 | 840.46 | 6571016192 | 1.082009 | 100 |
| haelyn_unmasked | beam_reference | 158.41 | 13.17 | 698314752 | n/a | n/a |
| dome_masked | hard | 987.06 | 755.93 | 5913059328 | 0.0 | 100 |
| dome_masked | soft | 1006.08 | 773.95 | 5898469376 | 0.020291 | 97 |
| dome_masked | free | 999.55 | 764.80 | 5884190720 | 0.908577 | 69 |
| dome_masked | beam_reference | 237.25 | 13.84 | 697769984 | n/a | n/a |

Paired wall ratio medians versus hard, primary clock then observer-excluded: masked soft 1.1580 and 1.1713, masked free 1.2565 and 1.2626, maskless soft 0.9817 and 0.9865, maskless free 0.9296 and 0.9308, dome soft 1.0193 and 1.0238, dome free 1.0103 and 1.0117. Observer time is 15 to 23 percent of the field-arm endpoint and 91 to 94 percent of the Beam endpoint.

**Did source constraints help?** Relaxing the tether to zero lowered held-out teacher MSE relative to hard equality in every condition at both endpoints, in all nine final seed pairs and in all three deterministic proxy comparisons, and lowered external full-canvas and foreground RGB MSE in all nine final pairs. It did so by abandoning the source footprints: mean drift near one whitened standard deviation, maximum source mean deviation 10.29 px on masked Haelyn and 2.71 px on the dome, maximum source covariance deviation 2344 px squared on masked Haelyn and 88.7 on the dome. The previews show the free arm as a blurred low-frequency smear over the subject, hard and soft as sparse streaks or a few blobs, and Beam as nearly nothing. The prospectively selected soft tether at weight 1 helped only on Haelyn after native refinement, hurt masked Haelyn before refinement, and hurt the dome at both endpoints in all three seeds. No arm produced a recognizable reconstruction, so no arm consistently helps in the sense the research question intends. The free arm consistently lowers the MSE endpoints; the soft arm does not.

**Claim dispositions.**

| # | Claim | Disposition | Evidence and recomputed values |
|---|---|---|---|
| K1 | Run completed with all 36 measured cells, one warmup, and three report phases | confirm | coordinator_exit.json exit 0; run_receipt.json completed; 36 process receipts exit 0; warmup summary with empty held-out metrics and 2-iteration budgets; no failure.json |
| K2 | Protocol approved before init-run, digest and source unchanged, exact dirty source preserved | confirm | digests in the table above; review saved 00:24:50, lock 00:25:01; 149 archived files equal listed hashes; development_source_state.bin equals the lock diff digest; live binding equals frozen aggregate; only ara/trace/sessions/2026-09-06_001.yaml, outside the binding, differs from its archived copy |
| K3 | Reconstruction saw only calibration and compact teachers; roles disjoint; held-out archives opened only after the saved endpoint; alpha policy honored | confirm | 36 guard records: 0 image opens, 0 forbidden imports, 4 of 4 negative controls; optimizer 10 or 9 views, validation 2 or 3, held-out 4 or 3, disjoint; held-out opens only in the reporting phase; load_alpha true only for masked field arms with hard support gating, false for maskless and Beam; 468 input hashes equal seal entries; entry and exit bindings equal; full-canvas scalar counts 147456 and 166221 |
| K4 | Causal arms start from byte-identical initial states per dataset and seed | confirm, with F1 | hard, soft, free gaussians_init.ply hashes equal within every dataset and seed; they are also equal across the three seeds; Beam differs |
| K5 | Producer groups, paired rows, charts, and top-level metrics follow from raw summaries | confirm | 0 mismatches in 12 groups, 36 paired rows, 36 stage values, quality and resource charts; top-level values equal the masked soft group: MSE 0.04771091694764775, wall 999.0805752759916 s, peak 5447041024 bytes, drift 0.02810707546161558; representative PLYs equal the masked soft seed 90601 cell |
| K6 | Free footprints lower held-out teacher MSE versus hard before and after identical refinement | confirm as descriptive, narrowed | table above; proxy comparison is one deterministic observation per dataset; final spread reflects native-refinement RNG only; three seeds, no pooled statistic |
| K7 | Task hypothesis: a soft tether improves native held-out color relative to hard equality | retire as a general claim; narrow to Haelyn after refinement | masked final 4.74 percent median gain but proxy 2.35 percent loss; maskless gains at both endpoints; dome losses at both endpoints in all seeds |
| K8 | Task hypothesis: zero tether may drift | confirm | drift 1.071949, 1.082009, 0.908577 whitened RMS; source projection deviations above |
| K9 | External RGB evaluation supports the free-arm error reduction | narrow | full and foreground MSE fall in all nine pairs; Haelyn SSIM falls in all pairs; SSIM on black-background targets rewards empty predictions, the initial model scoring 0.696 above every refined model; dome foreground PSNR floor for an empty render is 11.005 dB because the subject wears dark clothing |
| K10 | Any detailed reconstruction, production default, physical geometry truth, or unseen-scene generalization | retire, not supported | previews of all four arms in all three conditions; claim_boundary; 217 or 256 carriers and fixed finite budgets |
| K11 | Effect of masks on reconstruction | unresolved | masked Haelyn has 217 carriers after 39 alpha-rejected sources, maskless 256, different initial geometry and projection dilation 0.16 versus 0.30; not isolable |
| K12 | Runtime or memory effect of source constraints | narrow to descriptive, no arm claim | wall ratio medians from 0.93 to 1.26 with inconsistent sign across captures; memory within 2 percent among field arms; contended local machine; primary clock includes imports and observers |
| K13 | Beam is a practical reference only | confirm as descriptive | distinct alpha-free initialization; final MSE within 0.002 of its own initial in all conditions; near-empty previews; excluded from causal claims |

## Findings, limitations, and remaining handoff

**Severity-ranked findings.**

1. **High, interpretive: the placement and proxy stages are seed-invariant.** Initial and proxy models are byte-identical across seeds 90601, 90602, 90603 within every arm, and their held-out values are identical to full precision. The compact-carve seed feeds only sampling paths that this configuration does not use, and the refit is deterministic. Every proxy-endpoint comparison is therefore one observation per dataset, and the final-endpoint spread reflects native-refinement randomness only. "Three paired seeds" overstates replication and must be stated this way wherever these results are cited.
2. **High, interpretive: the free-arm gain is footprint inflation, not reconstruction.** The free arm's MSE gains coincide with large footprint drift and covariance growth, and its previews are blurred smears. No arm renders a recognizable subject. SSIM is not a valid detail measure here because an empty prediction on a black background scores higher than any refined model; the higher SSIM of hard, soft, and Beam reflects emptier predictions.
3. **Medium: the prospectively chosen soft tether is not a consistent improvement.** It loses on the dome at both endpoints in all seeds and on masked Haelyn before refinement.
4. **Medium: the dome free arm accepted only 69 of 100 proxy steps.** Thirty-one rejected steps each halved the learning rate under the frozen rollback policy, so that cell is a fixed-budget comparison, not a converged optimization. Other field cells accepted 97 to 100 steps.
5. **Medium: timing evidence is descriptive only.** The primary clock includes imports, source hashing, input loading, and validation observers. Driver diagnostics ran concurrently with measured cells at 01:23 and 09:20, so contention is documented and unquantified. Arm timing ratios change sign across captures.
6. **Low: dome full-frame metrics are dominated by the unmodeled room.** All arms sit near 0.43 full-canvas MSE and 3.6 dB; only foreground metrics separate arms, and their floor is set by dark clothing.
7. **Low: chronology of mid-run diagnostics.** The Driver inspected completed hard-baseline outcomes while later cells were running. Source, effective configurations, and seals were re-verified at every worker entry and exit and at coordinator exit, the diagnosed summary bytes are unchanged, and no setting changed, so no retuning was possible. These diagnostics directories inside the run root are non-protocol, append-only additions that the final bundle manifest must inventory.
8. **Low: staged prose.** The second RTGS-016 entry in docs/EXPERIMENTS.md still says the audit was not dispatched, and staging observations O169 and O170 are accurate but must not be promoted beyond the narrowed dispositions above. A benign PyTorch warning during drift extraction affects no value.

**Limitations of this audit and of the evidence.** Haelyn is a downloaded captured Gaussian model rendered offline on black; it is not physical geometry truth, and the maskless condition does not test cluttered backgrounds. The dome capture was previously outcome-exposed. The study uses 217 or 256 carriers, 100 proxy iterations, and 120 native iterations, so it says nothing about full-capacity quality or why detail is poor. Converter timings were contended GPU loop timers excluding startup. External MSE and SSIM values were not re-rendered by me. Three seeds support sign description only, with the proxy-stage caveat above. The full verify gate and CPU suite on the final tree were not run in this session.

**Remaining handoff checks, to be performed by the Driver after this verdict is persisted.**

- Persist this Markdown and JSON verbatim as the canonical AUDIT.md and AUDIT.json under benchmarks/results, which clears the two evidence errors the run validator currently reports.
- Render the shared v2 report so index.html, README.md, and manifest.json exist, then rerun the contract validate and check-run gates; the run validator with index required currently lists exactly those three files plus the two AUDIT records.
- Run the bundle checker with the flag `--no-previews`, because the legacy run-preview filenames do not apply; the 36 per-cell target, prediction, error previews must still be inventoried and link-checked, and the diagnostics directories inside the run root must be inventoried or explicitly excluded by policy.
- Produce a real browser smoke of the served report and the `rtgs view` viewer receipt in viewer_smoke.json using the recorded serve and viewer commands.
- Append-only updates by the Driver: the task record, docs/EXPERIMENTS.md, and any ara claim rows, each bound to this audit and limited to the dispositions above, followed by verify.sh on the final tree.

```json
{
  "schema_version": 1,
  "task_id": "20260906_tomography_source_constraints_haelyn_dome",
  "reviewer": "Claude-Code-Fable-5.1-reviewer",
  "model": "claude-fable-5-1",
  "effort": "max",
  "verdict": "accepted_with_limits",
  "protocol_sha256": "3b0b25fbb0721b138cfcb42ddfeddbc5dc7f0e883ab53dbea8d2c045931600e7",
  "source_aggregate_sha256": "3a3a7f0c1e1dc92c300c43e7a0ca53aa3b1e780fb2dcd8bfb39e93174d0c1430",
  "run_status": "completed",
  "digests": {
    "source_commit": "2ebe52cc28a1b0e812a6c436d49225fac4ee943f",
    "source_diff_sha256": "1e2a84c11827d6b6642c21f63abc0b89a30b1deac612d90f2a8cce2e2f82f07e",
    "task_sha256": "7fd81d81cf6ef346a0255d5fdc526a9827e1c643431f04911a2e1154119e125a",
    "protocol_review_artifact_sha256": "59023aeba8aee8240817af6af029e918f92a85171a408d0935e864897c18213c",
    "protocol_review_v1_rejected_sha256": "5229175043fd0fce7053c45f64d11950621885cff758d58ee6f0853bec0d8ed2",
    "data_seal_sha256": "9085780d67dca0a8df1a8b7485153fc66fe22a527745935fd0338196b45dbf19",
    "external_seal_sha256": "a0b9173030b242badb52d22fa0240ee8e8afafbda2fb33b25eed428a838a573c",
    "source_snapshot_archive_sha256": "39863faf3359298ade883af9b87ab6a32f370191b6a3b0c762908f4d3ed12a9d",
    "result_json_sha256": "cd23a1cf260d6fb206585569eb6eb765f5387d65fd3b75dc6a3217d4d52ab66e",
    "result_md_sha256": "b98c6bc3dbb1c1fb79af0af351e125317f55b73e576d2043289ff4443492d4ad",
    "payload_manifest_sha256": "43ef926b2c7fd83a02a17c28fc9570bfbdd995ba03b807439e18157a99a070a0",
    "haelyn_reference_ply_sha256": "b9ba7059bfce2a015e719ee4a0de3ea7dad111d7f4ae33d1ad2435e477d0b8c2"
  },
  "checks": [
    {"id": "git_state", "command": "git status --short; git rev-parse HEAD; git diff --stat; git diff --check", "result": "HEAD 2ebe52cc; 24 modified tracked files, 1300 insertions, 356 deletions; whitespace clean; RTGS-016 files untracked"},
    {"id": "protocol_digest", "command": ".venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260906_tomography_source_constraints_haelyn_dome.json", "result": "3b0b25fbb0721b138cfcb42ddfeddbc5dc7f0e883ab53dbea8d2c045931600e7 equals task, lock, and approved review"},
    {"id": "contract_validate", "command": ".venv/bin/python scripts/experiment_contract.py validate", "result": "experiment_contract: OK"},
    {"id": "data_seal", "command": ".venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260906_tomography_source_constraints_haelyn_dome.json", "result": "experiment_data: OK"},
    {"id": "verify_inputs", "command": ".venv/bin/python .scratch/rtgs016/claude_review_02/verify_inputs.py", "result": "task validation passed; compact and external seals passed; effective configuration matched; live source aggregate 3a3a7f0c across 122 files"},
    {"id": "file_digests", "command": "sha256sum on task, reviews, seals, lock, snapshot, archive, producer artifacts, payload manifest, reference PLY, representative PLYs", "result": "all equal to lock, snapshot, payload manifest, and authorization values; RESULT.json byte-identical to cell_results.json"},
    {"id": "focused_tests", "command": ".venv/bin/python -m pytest -q tests/test_tomography_person_protocol.py tests/test_experiment_contract.py tests/test_field_refit.py", "result": "53 collected, all passed, no failure output"},
    {"id": "cell_recomputation", "command": ".venv/bin/python -c read-only parse of cell_results.json and 36 summaries", "result": "36 completed non-warmup cells; 12 group medians, 36 paired differences and wall ratios, 36 stage medians, quality and resource charts, and top-level metrics reproduce with 0 mismatches"},
    {"id": "model_hashes", "command": ".venv/bin/python -c sha256 of 252 saved PLY files", "result": "252 of 252 equal summary digests; run-root representative PLYs equal the masked soft seed 90601 cell"},
    {"id": "initial_state_equality", "command": ".venv/bin/python -c compare initial_sha256", "result": "hard, soft, free identical within every dataset and seed and across seeds; proxy models identical across seeds; Beam differs"},
    {"id": "input_guards", "command": ".venv/bin/python -c parse input_boundary records", "result": "36 of 36 passed; 0 image opens; 0 forbidden imports; 4 of 4 negative controls; heldout archives opened only after saved endpoint; roles disjoint; entry and exit bindings equal; load_alpha true only for masked field arms"},
    {"id": "sealed_inputs", "command": ".venv/bin/python -c compare selected_input_binding to data seal", "result": "468 hashes equal the 55 sealed entries"},
    {"id": "effective_configuration", "command": ".venv/bin/python -c rehash recorded configs", "result": "36 of 36 equal the frozen effective_configuration digests"},
    {"id": "histories", "command": ".venv/bin/python -c check records, markers, proxy and native histories", "result": "252 finite validation records; 216 ordered markers; 101-entry finite proxy objective and elapsed histories; native histories finite; teacher and proposal digests unchanged"},
    {"id": "heldout_metric_semantics", "command": ".venv/bin/python -c recompute J_pixel from per-view MSE", "result": "108 of 108 equal; scalar counts 147456 for Haelyn and 166221 for dome equal full canvas times three channels"},
    {"id": "external_rows", "command": ".venv/bin/python -c parse three external_evaluation.json files", "result": "144, 144, 108 rows; views equal frozen heldout sets; 792 PSNR values recomputed from MSE with 0 mismatches; 0 non-finite; 0 perfect matches"},
    {"id": "source_snapshot", "command": ".venv/bin/python -c hash archive members", "result": "archive digest equal; 149 members equal listed hashes and bytes; development_source_state.bin equals lock source_diff_sha256; 148 equal live tree; ara/trace/sessions/2026-09-06_001.yaml outside the binding differs"},
    {"id": "validate_run", "command": ".venv/bin/python -c experiment_contract.validate_run", "result": "require_index false: only AUDIT.md and AUDIT.json missing; require_index true: also index.html, README.md, manifest.json"},
    {"id": "payload_manifest", "command": ".venv/bin/python -c hash 90 manifest items", "result": "90 items, 36 PNG, 12 dome previews flagged, 0 mismatches"},
    {"id": "previews_inspected", "command": "Read tool on PNG previews", "result": "12 final previews for all arms and conditions plus masked Haelyn and dome initial and free proxy previews viewed"},
    {"id": "denied_commands", "command": "git -C, tar -tzf, two larger python -c scripts", "result": "denied by permission mode; plain git forms and smaller scripts covered the same checks"},
    {"id": "not_executed", "command": "verify.sh, full pytest suite, check_results_bundle.py, render, check-run, viewer, browser, reconstruction, re-render, 1703-file historical rehash", "result": "not run in this audit"}
  ],
  "results": {
    "final_teacher_mse_median": {
      "haelyn_masked": {"hard": 0.04978992777082769, "soft": 0.04771091694764775, "free": 0.015017458629441201, "beam_reference": 0.07388142158180906},
      "haelyn_unmasked": {"hard": 0.046741315851722476, "soft": 0.023092238193506184, "free": 0.014079458613907773, "beam_reference": 0.07367847337192612},
      "dome_masked": {"hard": 0.004375221165011506, "soft": 0.0047689812508749764, "free": 0.0029631503828112496, "beam_reference": 0.005432033229129747}
    },
    "proxy_teacher_mse_deterministic": {
      "haelyn_masked": {"hard": 0.068512, "soft": 0.070119, "free": 0.049442, "beam_reference": 0.075051},
      "haelyn_unmasked": {"hard": 0.065803, "soft": 0.061927, "free": 0.050510, "beam_reference": 0.073912},
      "dome_masked": {"hard": 0.005120, "soft": 0.005535, "free": 0.004437, "beam_reference": 0.005832}
    },
    "paired_final_difference_vs_hard": {
      "haelyn_masked": {"soft": [-0.002374, -0.0026, -0.00182], "free": [-0.035067, -0.034814, -0.034439]},
      "haelyn_unmasked": {"soft": [-0.023649, -0.023404, -0.023901], "free": [-0.032662, -0.032299, -0.032828]},
      "dome_masked": {"soft": [0.000394, 0.000416, 0.000402], "free": [-0.001412, -0.001437, -0.001227]}
    },
    "paired_final_reduction_percent_vs_hard": {
      "haelyn_masked": {"soft": [4.74, 5.22, 3.66], "free": [70.02, 69.92, 69.24]},
      "haelyn_unmasked": {"soft": [50.6, 49.99, 51.17], "free": [69.88, 68.99, 70.29]},
      "dome_masked": {"soft": [-9.0, -9.5, -9.31], "free": [32.27, 32.79, 28.4]}
    },
    "paired_proxy_difference_vs_hard_deterministic": {
      "haelyn_masked": {"soft": 0.001607, "free": -0.01907},
      "haelyn_unmasked": {"soft": -0.003875, "free": -0.015293},
      "dome_masked": {"soft": 0.000414, "free": -0.000683}
    },
    "external_final_medians": {
      "haelyn_masked": {"hard": {"full_psnr": 12.962, "fg_psnr": 7.219, "ssim": 0.5237}, "soft": {"full_psnr": 13.19, "fg_psnr": 7.183, "ssim": 0.6741}, "free": {"full_psnr": 17.667, "fg_psnr": 13.47, "ssim": 0.4277}, "beam_reference": {"full_psnr": 11.334, "fg_psnr": 5.234, "ssim": 0.6777}, "initial_ssim": 0.6963},
      "haelyn_unmasked": {"hard": {"full_psnr": 13.114, "fg_psnr": 7.386, "ssim": 0.5268}, "soft": {"full_psnr": 15.835, "fg_psnr": 10.903, "ssim": 0.4649}, "free": {"full_psnr": 17.639, "fg_psnr": 13.354, "ssim": 0.4216}, "beam_reference": {"full_psnr": 11.266, "fg_psnr": 5.167, "ssim": 0.637}, "initial_ssim": 0.6979},
      "dome_masked": {"hard": {"full_psnr": 3.642, "fg_psnr": 12.588, "ssim": 0.0202}, "soft": {"full_psnr": 3.59, "fg_psnr": 11.98, "ssim": 0.0103}, "free": {"full_psnr": 3.7, "fg_psnr": 14.576, "ssim": 0.0348}, "beam_reference": {"full_psnr": 3.577, "fg_psnr": 11.341, "ssim": 0.0075}, "initial_ssim": 0.0028, "initial_fg_psnr": 11.005}
    },
    "wall_ratio_median_vs_hard_primary": {"haelyn_masked": {"soft": 1.158, "free": 1.2565}, "haelyn_unmasked": {"soft": 0.9817, "free": 0.9296}, "dome_masked": {"soft": 1.0193, "free": 1.0103}},
    "wall_ratio_median_vs_hard_observer_excluded": {"haelyn_masked": {"soft": 1.1713, "free": 1.2626}, "haelyn_unmasked": {"soft": 0.9865, "free": 0.9308}, "dome_masked": {"soft": 1.0238, "free": 1.0117}},
    "source_mean_drift_median": {"haelyn_masked": {"hard": 0.0, "soft": 0.02810707546161558, "free": 1.071948541278719}, "haelyn_unmasked": {"hard": 0.0, "soft": 0.09535668916508082, "free": 1.0820094230541573}, "dome_masked": {"hard": 0.0, "soft": 0.02029119828180544, "free": 0.9085769100797927}},
    "accepted_proxy_steps": {"haelyn_masked": {"hard": 100, "soft": 98, "free": 100}, "haelyn_unmasked": {"hard": 100, "soft": 99, "free": 100}, "dome_masked": {"hard": 100, "soft": 97, "free": 69}},
    "carriers": {"haelyn_masked_field": 217, "haelyn_unmasked_field": 256, "dome_masked_field": 256, "beam_reference": 256, "alpha_rejected_sources_haelyn_masked": 39},
    "top_level_reference_condition": {"native_teacher_mse": 0.04771091694764775, "wall_seconds": 999.0805752759916, "peak_host_bytes": 5447041024, "source_mean_drift": 0.02810707546161558},
    "run_duration_hours": 9.51
  },
  "claim_dispositions": [
    {"id": "K1", "claim": "Run completed with all 36 measured cells, warmup, and three report phases", "disposition": "confirm", "evidence": "coordinator_exit.json exit 0; run_receipt.json completed; 36 process receipts exit 0; no failure.json"},
    {"id": "K2", "claim": "Protocol approved before init-run; digest and source unchanged; exact dirty source preserved", "disposition": "confirm", "evidence": "review saved 00:24:50 UTC, lock 00:25:01 UTC; archive 149 files verified; lock diff digest equals archived state; live binding equals frozen aggregate"},
    {"id": "K3", "claim": "Reconstruction consumed only calibration and compact teachers; roles disjoint; heldout opened after endpoint; alpha policy honored", "disposition": "confirm", "evidence": "36 guard records; 468 sealed input hashes; full-canvas scalar counts 147456 and 166221"},
    {"id": "K4", "claim": "Causal arms start from byte-identical initial states", "disposition": "confirm", "evidence": "identical gaussians_init.ply within dataset and seed, also across seeds; Beam differs; see finding F1"},
    {"id": "K5", "claim": "Producer groups, paired rows, charts, top-level metrics follow from raw summaries", "disposition": "confirm", "evidence": "0 mismatches; representative PLYs equal masked soft seed 90601"},
    {"id": "K6", "claim": "Free footprints lower held-out teacher MSE versus hard before and after identical refinement", "disposition": "narrow", "evidence": "all 9 final pairs negative; 3 deterministic proxy comparisons negative; proxy comparison is one observation per dataset; descriptive only"},
    {"id": "K7", "claim": "Task hypothesis: soft tether improves native held-out color relative to hard", "disposition": "retire", "evidence": "masked final 4.74 percent median gain but proxy 2.35 percent loss; maskless gains; dome losses at both endpoints in all seeds"},
    {"id": "K8", "claim": "Task hypothesis: zero tether may drift", "disposition": "confirm", "evidence": "drift 1.071949, 1.082009, 0.908577 whitened RMS; max source mean deviation 10.29 px masked Haelyn and 2.71 px dome; max covariance deviation 2344 and 88.7"},
    {"id": "K9", "claim": "External RGB evaluation supports the free-arm error reduction", "disposition": "narrow", "evidence": "full and foreground MSE fall in all 9 final pairs; Haelyn SSIM falls in all pairs; empty initial model SSIM 0.696 exceeds every refined model; dome foreground floor 11.005 dB"},
    {"id": "K10", "claim": "Detailed reconstruction, production default, physical truth, or generalization", "disposition": "retire", "evidence": "previews of all arms show streaks, blobs, or blur; no recognizable subject; 217 or 256 carriers and fixed budgets"},
    {"id": "K11", "claim": "Effect of masks on reconstruction", "disposition": "unresolved", "evidence": "217 versus 256 carriers, different initial geometry and dilation 0.16 versus 0.30; not isolable"},
    {"id": "K12", "claim": "Runtime or memory effect of source constraints", "disposition": "narrow", "evidence": "wall ratio medians 0.93 to 1.26 with inconsistent sign; memory within 2 percent; contended machine; clock includes imports and observers"},
    {"id": "K13", "claim": "Beam is a practical reference only", "disposition": "confirm", "evidence": "alpha-free distinct initialization; final MSE within 0.002 of its initial; near-empty previews; excluded from causal claims"}
  ],
  "findings": [
    {"id": "F1", "severity": "high", "text": "Placement and proxy refit are seed-invariant: initial and proxy models are byte-identical across the three seeds, so proxy-endpoint comparisons are single observations and final-endpoint spread reflects native-refinement randomness only."},
    {"id": "F2", "severity": "high", "text": "Free-arm MSE gains coincide with large footprint drift and covariance growth and appear as blurred smears; no arm renders a recognizable subject; SSIM on black-background targets rewards empty predictions and is not a detail measure here."},
    {"id": "F3", "severity": "medium", "text": "The prospectively chosen soft tether is not a consistent improvement: it loses on the dome at both endpoints in all seeds and on masked Haelyn before refinement."},
    {"id": "F4", "severity": "medium", "text": "The dome free arm accepted only 69 of 100 proxy steps with learning-rate halving on each rejection; it is a fixed-budget comparison, not a converged optimization."},
    {"id": "F5", "severity": "medium", "text": "Timing is descriptive only: the primary clock includes imports, hashing, loading, and observers; Driver diagnostics ran concurrently with measured cells at 01:23 and 09:20 UTC; arm ratios change sign across captures."},
    {"id": "F6", "severity": "low", "text": "Dome full-frame metrics are dominated by the unmodeled room at about 0.43 MSE and 3.6 dB for every arm; only foreground metrics separate arms and their floor is set by dark clothing."},
    {"id": "F7", "severity": "low", "text": "Mid-run Driver diagnostics accessed completed outcomes while later cells ran; source, configurations, and seals were re-verified at each worker and at coordinator exit and no setting changed; the diagnostics directories inside the run root are non-protocol additions the bundle manifest must inventory."},
    {"id": "F8", "severity": "low", "text": "docs/EXPERIMENTS.md second RTGS-016 entry is stale about audit dispatch; staging observations O169 and O170 must not be promoted beyond the narrowed dispositions; a benign PyTorch warning during drift extraction affects no value."}
  ],
  "limitations": [
    "Haelyn is a downloaded captured Gaussian model rendered offline on black, not physical geometry truth; maskless Haelyn does not test cluttered backgrounds",
    "The dome capture was previously outcome-exposed",
    "217 or 256 carriers, 100 proxy iterations, and 120 native iterations; no full-capacity quality statement and no identification of why detail is poor",
    "Converter timings were contended GPU loop timers excluding startup; no uncontended speedup",
    "External MSE and SSIM values were not re-rendered by this audit; only internal consistency was verified",
    "Three seeds support sign description only; proxy-stage comparisons are single deterministic observations",
    "verify.sh, the full CPU suite, the bundle checker, render, browser, and viewer checks were not run in this audit",
    "The Driver's 1,703-file historical preservation rehash was not independently recomputed"
  ],
  "remaining_handoff_checks": [
    "Persist this audit verbatim as benchmarks/results/20260906_tomography_source_constraints_haelyn_dome_AUDIT.md and _AUDIT.json to clear the two evidence errors reported by validate_run",
    "Render the shared v2 report to produce index.html, README.md, and manifest.json, then rerun experiment_contract validate and check-run",
    "Run scripts/check_results_bundle.py on the run root with --no-previews, inventorying and link-checking the 36 per-cell previews and the diagnostics directories",
    "Perform the real browser smoke of the served report and the rtgs view viewer receipt in viewer_smoke.json using the recorded serve_report and viewer commands",
    "Append-only Driver updates to the task record, docs/EXPERIMENTS.md, and ara claim rows bound to this audit, then verify.sh on the final tree"
  ]
}
```
