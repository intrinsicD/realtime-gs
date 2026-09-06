# RTGS-016 prospective review packet

Driver preparation only. No independent verdict, review approval, run lock, real reconstruction
outcome, RESULT/AUDIT record, or browser-ready comparison exists yet.

- Protocol: `experiments/tasks/20260906_tomography_source_constraints_haelyn_dome.json`
- Current draft protocol SHA-256: `e8c2e3d7a45f2710675521d470f4aed8b6d43bb98f245f8eb34f6eea7dc86ed3`
- Source binding: `95884b5650ee5e8fb5eae524a33321c065e7bf823085f5e907c9760d3497b308`
  across 122 files.
- Data seals: `experiments/data/haelyn_dome_source_constraints.json` (55 compact/input files)
  and `experiments/data/haelyn_dome_source_constraints_external.json` (66 offline/report files).
- Source baseline: `2ebe52cc28a1b0e812a6c436d49225fac4ee943f`; uncommitted implementation.
  No commit, merge or push was authorized or performed. Any later run from this state must use
  the explicitly labelled development path and its dirty-source receipt.

The user selected a real captured female Gaussian reference before masked, maskless and
second-capture repetitions. Haelyn's selected person PLY is 58,614,278 bytes, 357,398 Gaussians,
and retains full degree-2 SH. Published splats map uniquely to original PLY rows under the
recorded similarity. Camera transforms preserve original SH directions. The 16-view orbit was
visually inspected: useful clothing detail and front/back coverage, incomplete lower edge and
some floating fragments. This is a controlled rendered reference derived from a real capture,
not physical density truth or another physical photograph capture.

Prepared inputs are 16 masked, 16 freshly fitted maskless, and 15 second-capture compact views.
The second capture is the existing stage/Janelle frame 00008, previously outcome-exposed
and therefore development data. Reconstructed models are capped at 256 initial carriers;
this first screen does not establish high-detail reconstruction at reference-model capacity.

Review must independently check the historical validation/execution split, source envelope
negative controls, input/camera seals, image-free worker boundaries, matched hard/soft/free
initial states, Beam's different placement, field/native semantics, fixed endpoint selection,
resource scopes and report-only mask use. Full-seal hashing is coordinator-only; workers may
hash selected train/validation archives before fitting but open heldout archives only after
the final model exists. A paused source-bound run cannot be resumed by silently replacing a
cell. The producer currently fails closed on existing cells and preserves failed artifacts.

The full CPU suite, focused worker/Beam/history/report checks, and temporary v2 aggregation
fixture pass. See `ara/evidence/tables/20260906_tomography_preflight/receipt.json`.
The final mandatory `./scripts/verify.sh` gate passed; its complete log is retained in the
preflight directory. Both ARA ledgers and the final task/documentation metadata also pass.
The 1,703-file historical preservation manifest proves zero changes to original task, driver,
review and run evidence; old results are not repaired or rewritten.

The prospective reviewer must consume source and preparation evidence only, before result
access. All task parameters remain draft. The explicit pending-review blocker was retained
after automatic approval review rejected clearing it. Reviewer-agent authorization is pending.
Resolve that human authorization and the blocker transition before obtaining the final
prospective review digest and canonical review artifact; do not reinterpret this packet as
permission to execute. Any reviewed protocol change needs a new exact digest.

After actual prospective approval, initialize the canonical development run and execute the
frozen command. Report the masked phase, then maskless, then the second capture without
retuning. Obtain a distinct results audit, render the shared v2 report, exercise a real browser
(WebGL2, nonblank scene, orbit), rerender with receipts, and pass check-run/check_results_bundle.
No reconstruction quality, convergence benefit, physical tomography validity or default
recommendation is established by this preflight.
