# realtime-gs research context

Use this as a routing map, then verify it against the checkout.

## Mission and seams

The repository asks whether compact fitted 2D Gaussian fields can provide a useful cold start for
3D Gaussian reconstruction. The principal seams are:

- image fitting (`rtgs.image2gs`);
- 2D-to-3D lifting (`rtgs.lift`);
- dense and sparse rendering (`rtgs.render`);
- depth backends (`rtgs.depth`);
- RGB-backed and compact-only optimization (`rtgs.optim`);
- strict reconstruction/evaluation input boundaries (`rtgs.data`);
- the compact carrier and legacy orchestration paths.

The CPU reference implementations are correctness anchors. CUDA/gsplat paths are separate evidence
classes and must retain parity/fallback checks.

## Current research distinctions

Do not collapse these questions:

- placement/topology versus initialization value;
- compact-only reconstruction versus RGB-backed refinement;
- fixed topology versus adaptive density;
- fitted-view conditioning versus held-out generalization;
- geometric covariance versus optical-thickness/coverage heuristics;
- mechanism validity versus material end-to-end utility;
- CPU correctness versus GPU performance;
- development/calibrated evidence versus claim-ready confirmation.

Read the active three-arm program and experiment registry before proposing a comparator. Existing
negative results and failed gates are part of the frontier; repeating them under a new name is not
novelty.

## Evidence path

A candidate normally moves through:

1. task framing and an explicit maturity target;
2. a cheap CPU/synthetic mechanism screen;
3. a frozen experiment task with train/held-out roles, controls, seeds, gates, resource scope, and
   exact command;
4. prospective protocol review bound by digest;
5. a calibrated local-data interaction with saved artifacts and viewer receipt;
6. independent result audit;
7. an ARA claim disposition.

The first useful output of ideation is usually the smallest falsifying experiment, not a new
backend abstraction.

## High-value anomaly sources

Search:

- refuted and superseded rows in `ara/logic/claims.md`;
- failed or withheld branches in `docs/EXPERIMENTS.md`;
- unresolved items in `docs/ROADMAP.md`;
- modality, gauge, covariance, coverage, topology, and optimizer confounds in the ADR/design
  notes;
- gaps between compact fitted-view evidence and fresh held-out scenes;
- places where a 2D representation does not identify a unique 3D cause;
- memory/throughput surfaces that remain unmeasured on controlled hardware.

Prefer ideas that explain several failures with one testable mechanism over ideas that add knobs.
