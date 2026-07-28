#!/usr/bin/env bash
# Full verification: lint + format + tests + docs-sync.
# This is what CI runs and what every agent/human runs before committing.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
if [ ! -x "$PY" ]; then PY=python3; fi

echo "==> ruff check"
"$PY" -m ruff check .
echo "==> ruff format --check"
"$PY" -m ruff format --check .
echo "==> pytest (not slow)"
"$PY" -m pytest -q -m "not slow"
echo "==> docs sync"
"$PY" scripts/docs_sync.py
echo "==> ara claim ledger"
"$PY" scripts/check_ara.py
echo "==> script layout"
"$PY" scripts/check_script_layout.py
echo "==> experiment contracts"
"$PY" scripts/experiment_contract.py validate
echo "==> verify OK"
