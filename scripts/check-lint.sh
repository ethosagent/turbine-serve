#!/usr/bin/env bash
# ruff lint + format check. Scope (src, excludes) lives in pyproject [tool.ruff].
# Function-size rule lives there too: ruff PLR0915 (max 50 statements per function).
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run --frozen ruff check .
uv run --frozen ruff format --check .
