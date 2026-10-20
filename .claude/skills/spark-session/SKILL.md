---
name: spark-session
description: >-
  Operate the DGX Spark for GPU work: free memory, sync the tree, build the dev container, run gpu
  tests, serve the model, read logs, restore the owner's other services. Use when a task says
  "on the Spark", "run the gpu tests", "parity test", "serve Qwen", "e2e", or touches
  models/, attention/, loader.py, model_runner.py, Dockerfile.
---

# spark-session

Rules that must never be missed:

- **The Spark is shared.** `qwen38-flash` (the owner's vLLM server, ~98 GB) must be stopped before
  our GPU work and started again at the end of the session. Never stop `rudder-embed`,
  `parakeet-asr`, `litellm`, `litellm-postgres`.
- **Edit on the Mac, run on the Spark.** Never edit files on the Spark; `scripts/sync.sh`
  overwrites them. Commits happen on the Mac.
- **Never `pip install torch` in the container.** NGC's build is the only one with sm_121.
- **Check memory with `free -g`,** not `nvidia-smi`; the 128 GB is unified.

## Procedure

1. **Reachability:** `ssh -o BatchMode=yes dgx-spark hostname` → `spark-50fe`.
2. **Free memory:** `ssh dgx-spark 'docker stop qwen38-flash; sleep 5; free -g | head -2'`.
   Expected `available` ≥ 90 GB. Our MVP needs ≈ 8 GB weights + `--kv-gb` + the HF reference
   model (another 8 GB) during parity.
3. **Sync:** `make sync`.
4. **Dev image** (first time or after `pyproject.toml` / `uv.lock` changed): `make spark-build`.
   Verify: `scripts/spark.sh python -c "import torch;print(torch.cuda.get_device_capability())"`
   → `(12, 1)`.
5. **Model** (first time): see `docs/runbooks/spark-session.md` step 4; ≈ 8 GB download into
   the mounted HF cache. `Qwen/Qwen3-4B` is the checkpoint; `unsloth/Qwen3-4B-Instruct-2507`
   already in the cache is NOT it.
6. **Tests:** `scripts/spark.sh pytest tests/test_<x>.py -q -m gpu` for one file;
   `scripts/spark.sh pytest -q` for everything. Quiet output; pipe through `tail -40` if long.
7. **Serve:** `make spark-serve` (foreground). In a second terminal the e2e tests run on the
   Spark host: `ssh dgx-spark 'cd ~/spark-engine && TURBINE_BASE_URL=http://127.0.0.1:8000/v1 ~/.venvs/e2e/bin/pytest tests/test_api_e2e.py -q'`
   (host venv once: `python3 -m venv ~/.venvs/e2e && ~/.venvs/e2e/bin/pip install pytest openai`).
8. **Logs:** `logs/engine.jsonl` lives in `~/spark-engine/logs` on the Spark.
   `ssh dgx-spark "jq -c 'select(.level==\"error\")' ~/spark-engine/logs/engine.jsonl | tail -5"`.
9. **End:** `ssh dgx-spark docker start qwen38-flash`. Say so in the hand-off.

## Reference

- Host: `dgx-spark` in `~/.ssh/config` (192.168.1.49, user miteshsharma). Docker 29, GB10,
  driver 580, container `nvcr.io/nvidia/pytorch:26.07-py3` (torch 2.13 NVIDIA build, CUDA 13.3).
- `scripts/spark.sh` runs `docker run --rm --gpus all --ipc=host -v ~/spark-engine:/w -w /w
  -v ~/.cache/huggingface:/root/.cache/huggingface -e PYTHONPATH=/w turbine-serve-dev <cmd>`.
- The server inside the container binds `0.0.0.0`; `-p 127.0.0.1:8000:8000` keeps it local to
  the Spark. From the Mac, reach it only through `ssh dgx-spark curl ...`.
