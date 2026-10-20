#!/usr/bin/env bash
# mypy --strict over turbine_serve (config in pyproject). Compile errors replace reading code.
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run --frozen mypy
