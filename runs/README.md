# Local run artifacts

New experiment output has exactly one top-level directory:

`runs/YYYYMMDD_<task_slug>_<data_slug>/`

Create it through:

```bash
python scripts/experiment_contract.py init-run experiments/tasks/<task_id>.json
```

Everything except this README is ignored. Do not create `_v2`, `_final`, `_failed`, timestamp, or
“latest” siblings. Repeats and failures belong under the task root (`cells/`, `attempts/`, or
`repeats/`) with machine-readable receipts. Official run roots are never overwritten.

For report-template v2, a completed root has this minimum shape:

```text
runs/<task_id>/
├── task.lock.json
├── metrics.json
├── training_history.json
├── gaussians.config.json
├── environment.json
├── run_receipt.json
├── input_boundary_receipt.json
├── resource_receipt.json
├── gaussians_init.ply
├── gaussians.ply
├── previews and smoke receipt
├── index.html        # generated
├── README.md         # generated
└── manifest.json     # generated last; checksums everything except itself
```

The exact preview names and external RESULT/AUDIT records are defined in `experiments/README.md`.
Run `experiment_contract.py render` after the last receipt is written, then use `check-run` and
`check_results_bundle.py`. Failed protected runs keep the same identity and may omit model/evidence
outputs, but must produce a generated failure report and can never pass the results-bearing gate.

Historical directories predate this contract and remain in place because result notes and source
seals cite them. They are legacy evidence, not naming examples.
