# Append-only result records

Historical records in this directory are immutable even when their old names are inconsistent.
They bind claims, source hashes, and audit chronology.

New task-first experiments use:

- `<task_id>_RESULT.json` and `<task_id>_RESULT.md`
- `<task_id>_AUDIT.json` and `<task_id>_AUDIT.md`
- additional sealed receipts only when the task requires them

The task itself replaces a separately mutable preregistration file. Its SHA-256 is copied to
`runs/<task_id>/task.lock.json` before execution. Never overwrite a result or audit; a materially
changed protocol receives a new task id.
