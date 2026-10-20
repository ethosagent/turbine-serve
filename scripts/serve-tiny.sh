#!/usr/bin/env bash
# Serve a random-weight tiny Qwen3 on CPU so the API runs end to end on the Mac.
# Usage: scripts/serve-tiny.sh [extra turbine serve flags]   (PORT=18000 to change the port)
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m turbine_serve.testing.tiny_model .tiny-qwen3
exec uv run turbine serve .tiny-qwen3 --device cpu --dtype float32 --kv-gb 0.01 \
  --port "${PORT:-8000}" "$@"
