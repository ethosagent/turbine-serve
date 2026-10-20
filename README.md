# turbine-serve

A minimal LLM inference server for Qwen3-4B on NVIDIA DGX Spark, with an OpenAI-compatible API.

[![docker](https://img.shields.io/docker/v/miteshdocker123/turbine-serve?label=docker&sort=semver)](https://hub.docker.com/r/miteshdocker123/turbine-serve)
[![pypi](https://img.shields.io/pypi/v/turbine-serve)](https://pypi.org/project/turbine-serve/)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
![hardware](https://img.shields.io/badge/hardware-DGX%20Spark%20(GB10)-76b900)
![api](https://img.shields.io/badge/API-OpenAI--compatible-black)

turbine-serve is built from scratch to learn how vLLM, SGLang, TensorRT-LLM and TensorFold work.
Each stage of a request lives in one small file you can read in a sitting, and every speed-up
plugs in behind an interface. It is a learning engine, not a vLLM replacement: correctness
first, speed after.

## Quickstart

Docker (main way):

```bash
docker run --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  miteshdocker123/turbine-serve:0.1.0 \
  Qwen/Qwen3-4B --kv-gb 30
```

`--gpus all` gives the container the GPU. `--ipc=host` gives PyTorch enough shared memory.
`-p 127.0.0.1:8000:8000` makes it reachable only from your own machine. The `-v` mount reuses
your Hugging Face cache, so weights are downloaded once and never baked into the image.

The image is arm64 only (built for the DGX Spark) and about 11 GB to pull, almost all of it the
NVIDIA PyTorch base (CUDA, cuDNN, NCCL, PyTorch). Model weights are not included.

pip (second way):

```bash
pip install turbine-serve
turbine serve Qwen/Qwen3-4B --kv-gb 30
```

The pip path is unverified on the DGX Spark (GB10 needs a torch build with sm_121 kernels); on
the Spark, serve from the container.

No GPU? `scripts/serve-tiny.sh` serves a random-weight tiny Qwen3 on CPU, so the whole API runs
on a laptop (the output is gibberish, the plumbing is real).

Then send a request:

```bash
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Port 8000 matches vLLM, so Open WebUI or anything already pointed at vLLM switches over by
changing nothing but the server.

## API

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | `{"status":"ok"}` |
| `GET` | `/v1/models` | the one served model |
| `POST` | `/v1/chat/completions` | a chat completion, or an SSE stream with `"stream": true` |

Request fields: `messages` (roles `system`, `user`, `assistant`), `stream`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, and `chat_template_kwargs` (`{"enable_thinking": true}`
turns on Qwen3 thinking; it is off by default). `model` is accepted and ignored.

Response (non-streaming):

```json
{
  "id": "req_9f2c41ab",
  "object": "chat.completion",
  "created": 1792483200,
  "model": "Qwen/Qwen3-4B",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"},
               "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
}
```

A stream sends one `chat.completion.chunk` per token, then a chunk carrying `finish_reason`,
then `data: [DONE]`.

| Status | When | `error.type` |
|---|---|---|
| `200` | success | |
| `400` | prompt longer than `--max-model-len`, or KV cache can never fit it | `invalid_request_error` |
| `422` | body fails schema validation (empty `messages`, `top_p` > 1, ...) | FastAPI detail |
| `500` | the engine failed the request before any output | `server_error` |
| `503` | request queue full (`--max-queue`) | `server_overloaded` |

Errors use the OpenAI shape: `{"error": {"message": "...", "type": "..."}}`.

Any OpenAI SDK works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Explain paged attention in one line."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Features

Built:

- [x] OpenAI-compatible chat completions, streaming (SSE) and non-streaming
- [x] Continuous batching: FCFS admission with KV reservation, up to `--max-num-seqs` sequences
- [x] Paged KV cache with a fixed `--kv-gb` budget (sized for the Spark's unified memory)
- [x] Hand-written Qwen3 (QK-Norm, RoPE, SwiGLU), checked token-for-token against `transformers`
- [x] SDPA attention backend
- [x] Sampling: greedy, temperature, top-p, top-k, seed
- [x] Back-pressure (`503` when the queue is full) and client-disconnect cleanup
- [x] One JSON log line per event, every line tagged with its `request_id`

## How it works

One file per stage, in the order a request moves through them:

```text
HTTP ─▶ api/server.py ─▶ tokenizer.py ─▶ engine_client.py ─▶ engine.py (loop)
                                                               │
               ┌───────────────────────────────────────────────┘
               ▼
          scheduler.py ──▶ kv_cache.py ──▶ model_runner.py ──▶ models/qwen3.py
               ▲                                                 │  attention/sdpa.py
               │                                                 ▼
               └──────────────── sampler.py ◀────────────── logits
                                     │
  SSE ◀── api/server.py ◀── detokenizer.py ◀── token ids
```

| Block | Job |
|---|---|
| `api/server.py` | OpenAI schemas, routes, SSE, maps engine errors to status codes |
| `tokenizer.py` | applies the Qwen3 chat template, text to token ids |
| `engine_client.py` | async bridge from the HTTP side to the engine thread, bounded queue |
| `engine.py` | the step loop; the only file that wires blocks together |
| `scheduler.py` | WAITING/RUNNING queues, picks the batch for each step, decides when a sequence ends |
| `kv_cache.py` | paged K/V pools per layer and a free-block list |
| `model_runner.py` | packs the batch into flat tensors and runs the forward pass |
| `models/qwen3.py` | the Qwen3 decoder, written from scratch |
| `attention/sdpa.py` | attention over the paged cache via PyTorch SDPA |
| `sampler.py` | turns logits into the next token |
| `detokenizer.py` | incremental token ids to text, safe across multi-byte characters |

Blocks talk only through `types.py` and swap behind the Protocols in `interfaces.py`;
import-linter enforces the layering. Function-by-function walkthrough of one request, with
every error path and a debugging guide: [`docs/how-it-works/`](docs/how-it-works/README.md).

## Benchmarks

Measured on an NVIDIA DGX Spark (GB10) with `turbine-serve:0.1.0`, Qwen/Qwen3-4B bf16,
`--kv-gb 30`, SDPA attention, no CUDA graphs. Same prompt for every request, temperature 0,
128 output tokens, 3 requests per stream:

| concurrency | requests | total tok/s | per-stream tok/s | mean TTFT ms |
|---:|---:|---:|---:|---:|
| 1 | 3 | 19.8 | 19.8 | 53 |
| 8 | 24 | 144.0 | 18.0 | 127 |

Batching 8 streams gives 7.3× the throughput, and each stream keeps about 90% of its speed.

Reproduce against any OpenAI-compatible server:

```bash
python bench.py --base-url http://127.0.0.1:8000/v1 --concurrency 1 8 --max-tokens 128
```

## Development

```bash
make install                 # uv sync + git hooks
make check                   # every gate (lint, mypy strict, import-linter, sizes, tests), failures only
uv run pytest tests/test_scheduler.py -q   # one file
make sync                    # push the tree to the DGX Spark
scripts/spark.sh pytest -q   # the same suite on the Spark, GPU tests included, in the NGC container
```

Everything is built and tested on the Mac first, against a fake runner and a tiny random-weight
Qwen3. The Spark then runs the same tests with the real weights.

- Design rules and the dependency law: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Block-by-block spec: [`docs/SPEC.md`](docs/SPEC.md)
- Working on the Spark: [`docs/runbooks/spark-session.md`](docs/runbooks/spark-session.md)
- Agent map for Claude Code and other coding agents: [`CLAUDE.md`](CLAUDE.md)
