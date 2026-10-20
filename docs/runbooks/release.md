# Runbook — build and ship the image

1. Bump `__version__` in `turbine_serve/__init__.py` (pyproject reads it dynamically). Commit.
2. `make sync` then on the Spark:
   `ssh dgx-spark 'cd ~/turbine-serve && docker build -t miteshdocker123/turbine-serve:<v> .'`
3. Smoke test exactly as users run it:
   ```
   docker run --rm --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     miteshdocker123/turbine-serve:<v> Qwen/Qwen3-4B --kv-gb 30
   curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"hi"}],"stream":true}'
   ```
4. Push (owner only, needs `docker login`): `docker tag miteshdocker123/turbine-serve:<v> miteshdocker123/turbine-serve:latest`, then `docker push miteshdocker123/turbine-serve:<v>` and `docker push miteshdocker123/turbine-serve:latest`.
5. pip: `uv build` then `uv publish` (owner only). The pip path needs an sm_121 torch on the
   host, so the container is the supported path; say so in the release notes.
