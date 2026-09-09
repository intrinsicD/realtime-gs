# RTGS-021 delivery evidence

The JSON receipts and logs preserve the completed experiment delivery checks.
`final_http_check.py.txt` is the exact executed one-off HTTP-check source, archived as
text rather than installed as repository tooling. It writes its receipt exclusively
and must not overwrite the existing `final_http_check.json`.

The September 9 pre-commit check found formatting errors in the source copy added
following the September 8 full verification. The file was renamed without changing
its bytes; the experiment driver, protocol, models, results and audits were unchanged.
The fresh pre-commit gate is recorded in `prepush_verify.log`.

The canonical HTML, previews, models and target caches remain under the ignored
`runs/20260908_field_teacher_information_stage_frame00008/` directory. They are not
included in a Git pull. Tracked research, protocols, numerical results and audits are
available from the repository at work; the earlier no-additional-images/masks transfer
instruction remains in force.
