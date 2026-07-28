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

Historical directories predate this contract and remain in place because result notes and source
seals cite them. They are legacy evidence, not naming examples.
