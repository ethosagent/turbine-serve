#!/usr/bin/env bash
# Run a command inside the turbine-serve-dev container on the Spark with the synced tree at /w.
#   scripts/spark.sh pytest                    # gpu tests run here
#   scripts/spark.sh --serve Qwen/Qwen3-4B --kv-gb 30   # server on host 127.0.0.1:8000
#   scripts/spark.sh --build                   # (re)build the dev image from the Dockerfile deps stage
# Requires: scripts/sync.sh first; qwen38-flash stopped (docs/runbooks/spark-session.md).
set -euo pipefail
IMAGE=turbine-serve-dev
BASE='docker run --rm --gpus all --ipc=host -v ~/spark-engine:/w -w /w -v ~/.cache/huggingface:/root/.cache/huggingface -e PYTHONPATH=/w'
case "${1:-}" in
  --build) ssh dgx-spark "cd ~/spark-engine && docker build --target deps -t $IMAGE ." ;;
  --serve) shift; ssh -t dgx-spark "$BASE -p 127.0.0.1:8000:8000 $IMAGE python -m spark_engine.main serve $* --host 0.0.0.0" ;;
  "") echo "usage: scripts/spark.sh [--build | --serve <model> [flags] | <cmd...>]"; exit 2 ;;
  *) ssh dgx-spark "$BASE $IMAGE $*" ;;
esac
