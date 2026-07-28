# Benchmarks and historical evidence

`benchmarks/run.py` is the reusable benchmark suite. The many other top-level Python files are
historical experiment drivers whose paths and hashes are cited by append-only evidence. Do not
rename or regroup them as a cosmetic cleanup.

New result-bearing research starts under `experiments/tasks/`. Its task-specific driver is named exactly
`scripts/experiments/<task_id>.py`; it must not add another ad-hoc renderer or report template
here. Reusable performance cases may still be added to `benchmarks/run.py` and documented through
the benchmark workflow.

Machine/human evidence remains under `benchmarks/results/` and follows the task id. See that
directory's README.
