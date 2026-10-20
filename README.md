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

Then any OpenAI client works:

```bash
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Port 8000 matches vLLM, so Open WebUI or anything already pointed at vLLM switches over by
changing nothing but the server.

## Develop

```bash
make install        # uv sync + git hooks
make check          # every gate, failures only
make sync           # push the tree to the DGX Spark
scripts/spark.sh pytest -q   # run the suite (GPU tests included) in the NGC container
```

Agent map: `CLAUDE.md`. Constitution: `ARCHITECTURE.md`. Tour: `docs/orientation.md`.
