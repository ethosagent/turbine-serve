#!/usr/bin/env bash
# pytest, quiet. gpu-marked tests skip without CUDA; e2e tests skip without TURBINE_BASE_URL.
# On the Spark run the same file through scripts/spark.sh so the gpu tests execute.
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run --frozen pytest
