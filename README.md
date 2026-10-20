# turbine-serve

A minimal LLM inference server for Qwen3-4B on NVIDIA DGX Spark, with an OpenAI-compatible API.
Built from scratch to learn how vLLM, SGLang, TensorRT-LLM and TensorFold work.

## Run

Docker (main way):

```bash
docker run --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ethosagent/turbine-serve:0.1.0 \
  Qwen/Qwen3-4B --kv-gb 30
```

`--gpus all` gives the container the GPU. `--ipc=host` gives PyTorch enough shared memory.
`-p 127.0.0.1:8000:8000` makes it reachable only from your own machine. The `-v` mount reuses
your Hugging Face cache, so weights are downloaded once and never baked into the image.

pip (second way):

```bash
pip install turbine-serve
turbine serve Qwen/Qwen3-4B --kv-gb 30
```

The pip path needs a torch build with sm_121 (GB10) support on the host; stock PyPI torch lacks
it, so on the Spark serve from the container.

Then any OpenAI client works:

```bash
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Port 8000 matches vLLM, so Open WebUI or anything already pointed at vLLM switches over by
changing nothing but the server.

## Baseline (MVP, SDPA, no CUDA graphs)

2026-10-20 · NVIDIA DGX Spark (GB10, driver 580.173.02) · `ghcr.io/ethosagent/turbine-serve:0.1.0`
· Qwen/Qwen3-4B bf16 · `--kv-gb 30` · prompt "Count from 1 to 100, separated by commas.",
temperature 0, 128 tokens per request, 3 requests per stream:

```bash
python bench.py --base-url http://127.0.0.1:8000/v1 --concurrency 1 8 --max-tokens 128
```

| concurrency | requests | total tok/s | per-stream tok/s | mean TTFT ms |
|---:|---:|---:|---:|---:|
| 1 | 3 | 19.8 | 19.8 | 53 |
| 8 | 24 | 144.0 | 18.0 | 127 |

## Develop

```bash
make install        # uv sync + git hooks
make check          # every gate, failures only
make sync           # push the tree to the DGX Spark
scripts/spark.sh pytest -q   # run the suite (GPU tests included) in the NGC container
```

`scripts/serve-tiny.sh` serves a random tiny Qwen3 on CPU for end-to-end runs on the Mac.

Agent map: `CLAUDE.md`. Constitution: `ARCHITECTURE.md`. Tour: `docs/orientation.md`.
