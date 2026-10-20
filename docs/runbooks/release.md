# Runbook — build and ship the image

1. Bump `__version__` in `spark_engine/__init__.py` (pyproject reads it dynamically). Commit.
2. `make sync` then on the Spark:
   `ssh dgx-spark 'cd ~/spark-engine && docker build -t ghcr.io/ethosagent/turbine-serve:<v> .'`
3. Smoke test exactly as users run it:
   ```
   docker run --rm --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     ghcr.io/ethosagent/turbine-serve:<v> Qwen/Qwen3-4B --kv-gb 30
   curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"hi"}],"stream":true}'
   ```
4. Push (owner only, needs `docker login ghcr.io`): `docker push ghcr.io/ethosagent/turbine-serve:<v>`.
5. pip: `uv build` then `uv publish` (owner only). The pip path needs an sm_121 torch on the
   host, so the container is the supported path; say so in the release notes.
