# Runbook — a GPU session on the Spark

Edit on the Mac. The Spark only runs. Full procedure with checks: `.claude/skills/spark-session`.

1. **Free memory** (the owner's Qwen3.8 server holds ~98 GB):
   `ssh dgx-spark 'docker stop qwen38-flash && free -g | head -2'` → `available` ≥ 90 GB.
2. **Sync**: `make sync` (rsync to `dgx-spark:~/turbine-serve`, `.venv`/`.git`/logs excluded).
3. **Dev image** (first time, or after `pyproject.toml`/`uv.lock` change): `make spark-build`.
4. **Model cache** (first time): `scripts/spark.sh python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B', allow_patterns=['*.safetensors','*.json','*.txt'])"`.
5. **Run tests**: `scripts/spark.sh pytest -q` (whole suite, gpu tests included) or one file.
6. **Serve**: `make spark-serve` (foreground, Ctrl-C to stop). Reachable only on the Spark's
   loopback: `ssh dgx-spark curl -N localhost:8000/v1/chat/completions ...`.
7. **Logs**: `ssh dgx-spark 'jq -c "select(.request_id==\"req_…\")" ~/turbine-serve/logs/engine.jsonl'`.
8. **End of session**: `ssh dgx-spark docker start qwen38-flash`.

Gotchas: never `pip install torch` in the container; `--ipc=host` is required for torch's shared
memory; the HF cache mount is what makes the 8 GB download happen once.
