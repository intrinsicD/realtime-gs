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
echo "==> verify OK"
