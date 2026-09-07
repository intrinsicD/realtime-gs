# BENCH019 publication/inventory/failure integration review

Reviewer: Codex-cross-repo-review
Self-reviewed: No
Verdict: accepted within reviewed source and metadata scope; final experiment approval pending
Outcome access: none. All execution in this review used synthetic temporary fixtures; no calibrated capture or protected run output was opened.

## Exact reviewed files

- src/rtgs/bench019_local_report.py SHA256 10c7d640e39e4a504162f27c704c05d9fd6f25b71ef7a2c267061d248de09968
- tests/test_bench019_local_report.py SHA256 7a447f05c4a3a40cb8cdb55c182f74f8104ac839f59569b23bdf8470da2be5bd
- scripts/experiments/20260907_bench019_local_stage_frames00008_00009.py SHA256 57c50de238c3836b688d11233f02c46f1a05b746af6fbf84cee2bea3bad7bd67
- tests/test_bench019_local_driver.py SHA256 fd4a705fcb528038c6635cc8af80642cffc70a5dd521069b3edf921123a187b0
- scripts/experiment_contract.py SHA256 15f129398aabb46a50d435c76fec27b4df6c75e6d08d2e7d78a5a96657b25959
- scripts/check_results_bundle.py SHA256 0f5e1293a4c6651d1fed80237bd1319f44ed2ca268d81cb27ce87d2e54f2fedf

The coordinator review delta is its bounded failure-publication call after the failed machine receipt; the rest was reviewed previously. Each checker change is one line in its v2 manifest inventory: exclude only run/manifest.json rather than every basename manifest.json.

## Correctness and simplicity

No blocking findings. Publication reuses the strict existing formal exporter/assembler and shared v2 renderer. It verifies task/cell identities, saved model binding, held-out view membership, per-view-to-cell arithmetic means, training-only sampled views and fixed count/horizon. It preserves raw receipts for independent audit, complete resolved configurations, model/preview descriptors, source-observation metrics, actual optimizer-clock histories and separately linked Stage-1 histories.

Primary quality groups use means across the three paired seeds; wall/memory groups use medians. Stage-1 acquisition is included exactly once per frame/family, and warmups/A-A do not enter primary aggregates. Paired development materiality uses both frame units, positive differences in every seed, the 0.25 dB mean floor and 0.02 alpha-IoU guard. No independent-capture confidence interval or broad surrogate claim is manufactured.

The 18 primary cells plus native A/A form the unchanged portable 19-row protocol set. The second contained replay remains in the 20-cell local result and resource/audit evidence. Camera-only bounds and camera geometry are checked equal across families; unsupported-anchor diagnostic flags are reported without retuning. Representative models/previews use the fixed native seed and do not perform selection.

The producer writes RESULT evidence only. Canonical AUDIT evidence remains the independent reviewer's responsibility; immutable page summary now directs acceptance to that evidence. Producer-time pending-audit status is retained as provenance rather than rewritten into an approval. Real browser/viewer smoke remains a later requirement; the synthetic renderer test does not pretend to satisfy it.

The nested-manifest correction restores consistency between the generated full file inventory and both validators. Nested consumed field manifests are included, omission is rejected, and tampering is detected. This does not weaken the historical v1 path or rewrite old artifacts.

Failure publication preserves the coordinator's failed run receipt and available partial sources, emits empty metric/chart sources only when task context exists, and labels absent environment information unprobed. Without task context it retains a truthful minimal failure record instead of fabricating a task or quantitative result. Failure cannot become a successful result bundle.

## Independent verification

From /home/alex/Documents/realtime-gs:

CUDA_VISIBLE_DEVICES= MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/home/alex/Documents/structsplat/src:/home/alex/Documents/realtime-gs/src /tmp/structsplat-rtgs-collab/venv/bin/python -m pytest -q tests/test_bench019_local_report.py tests/test_bench019_local_driver.py

Result: 32 passed (11 report, 21 coordinator). This includes strict formal 19-cell export, real shared renderer/v2 schemas, nested-manifest omission/tamper checks by both validators, source rejection and malformed-task failure glue. Only source-authority lookup is mocked in the synthetic renderer fixture; synthetic AUDIT fixture files exist exclusively in pytest temporary directories.

Additional independent probes: 100 random paired comparison matrices matched a NumPy mean/sample-standard-deviation/materiality oracle. A complete synthetic publication with enormous warmup and A/A time/memory values left all primary medians unchanged. Exactly six Stage-1 timing observations and six acquisition chart values remained. All passed. git diff --check also passed.

## StructSplat initial metadata

Reviewed and accepted metadata-only commit 6ff898e8682cc7d932d540b818b3f0d6a4a0e529, affecting only:

- tasks/BENCH-019-stage1-downstream-objective.md SHA256 edfce5b80208eaf10e15858bf65d8e91ca870f4a3270394a714252803bcd6b8f
- tasks/INDEX.md SHA256 32847b9d5e3a3df50e8715a46308cd0328e8312d11d4453dd0ffd10dc1a5cb1f
- tasks/SESSION-BRIEF.md SHA256 50a17f528a902693ee447f753c425e26d9b8f511e8c469afb54bca1766df82b4

The stale active Reviewed revision field was corrected to pending current RTGS-019 source/protocol review; dated older descriptions remain as history. The new notes identify the authorized two-phase local development scope and leave prospective review, execution outcomes, general surrogate and default gates open. Production source and old protocols/evidence are unchanged. The committed StructSplat tree was clean when inspected.

## Remaining gates

Final review must bind the complete frozen task, internal/external source inventories and exact clean source commits. Required full CPU/structural gates and retained real CUDA diagnostic evidence must pass before canonical protocol approval and init-run. Phase-two authoring/finalization remains an independent exact-envelope approval after fields are hashed, before downstream execution. Later result acceptance requires raw-results audit and both repositories' report/browser gates. No such later approval is supplied by this source review.
