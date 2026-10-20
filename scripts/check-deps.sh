#!/usr/bin/env bash
# The dependency law (ARCHITECTURE.md §II) enforced by import-linter. The named contracts live in
# pyproject [tool.importlinter]. Output goes to .logs/deps.log; run-checks.sh tails it on failure.
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run --frozen lint-imports --no-cache
