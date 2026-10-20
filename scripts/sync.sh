#!/usr/bin/env bash
# Push the working tree to the DGX Spark (host `dgx-spark` in ~/.ssh/config). Editing happens on
# the Mac; the Spark only runs. Excluded dirs are never deleted on the Spark side.
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
rsync -az --delete \
  --exclude .venv --exclude .git --exclude logs --exclude .logs \
  --exclude __pycache__ --exclude .pytest_cache --exclude .mypy_cache --exclude .ruff_cache \
  ./ dgx-spark:~/spark-engine/
echo "synced → dgx-spark:~/spark-engine"
