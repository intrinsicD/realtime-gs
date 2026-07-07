#!/bin/bash
# SessionStart hook for Claude Code on the web: create the venv and install the
# package (CPU torch) so tests/linters/benchmarks work immediately. Idempotent —
# skips installation when the environment is already functional.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

if ! { .venv/bin/python -c "import rtgs, pytest" && .venv/bin/python -m ruff --version; } >/dev/null 2>&1; then
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
fi

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi

.venv/bin/python -c "import torch, rtgs; print(f'rtgs env ready: torch {torch.__version__}, rtgs {rtgs.__version__}')"
