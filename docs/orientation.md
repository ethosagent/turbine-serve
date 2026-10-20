# Orientation — repo tour

One pass over where things live and how a request flows. Laws are in `ARCHITECTURE.md`.

## Layout

```
turbine-serve/
  pyproject.toml      uv project: deps, ruff, mypy, pytest, import-linter contracts
  Makefile            thin wrappers: make check / test / sync / spark-serve
  scripts/            check-*.sh gates, run-checks.sh, sync.sh, spark.sh, hooks/, handoff.sh
  turbine_serve/       the package (one block per file; see below); testing/ holds the tiny checkpoint writer
  tests/              pytest; fakes.py holds FakeRunner for GPU-free engine tests
  docs/               SPEC.md (block spec), this file, conventions, numbers, runbooks/
  .claude/            settings.json (hooks, deny list) and skills/
  Dockerfile          NGC PyTorch base + uv-exported deps; `deps` stage = dev image
```

## How a request flows

1. `api/server.py` validates the OpenAI-style body, makes a `request_id`, logs `request.received`.
2. `tokenizer.py` applies Qwen3's chat template (thinking off unless asked) and encodes; a prompt
   over `max_model_len` raises `PromptTooLong` → HTTP 400.
3. `engine_client.py` wraps the ids in a `Sequence`, hands it to the engine thread, and returns an
   async iterator of `TokenOutput`. A client disconnect aborts the sequence.
4. `engine.py` runs the loop on its thread: `scheduler.schedule()` → `model_runner.execute()` →
   `sampler.sample()` → `scheduler.update()` → emit. One `step.done` log line per iteration.
5. `scheduler.py` gives every RUNNING sequence one decode token and admits WAITING ones while
   `max_num_seqs` and KV allow, reserving `prompt_len + max_tokens` blocks up front (no preemption).
6. `kv_cache.py` owns the paged K/V pools (`[num_blocks, 16, 8, 128]` per layer) and the free list.
7. `model_runner.py` flattens the batch into `input_ids`, `positions`, `slot_mapping`,
   `logits_indices` and calls the model under `inference_mode`.
8. `models/qwen3.py` runs 36 layers; `attention/sdpa.py` gathers each sequence's blocks and calls
   torch SDPA with `enable_gqa=True`.
9. `sampler.py` turns logits into one token per sequence, seeded by `(seed, position)`.
10. Back in the API, `detokenizer.py` turns ids into text deltas, holding back partial UTF-8, and
    the server streams SSE chunks ending with `data: [DONE]`.

Follow one request end to end: `jq -c 'select(.request_id=="req_…")' logs/engine.jsonl`.

## Milestones

| | Blocks | Done when |
|---|---|---|
| M1 | config, types, interfaces, log | a block emits a valid JSON line |
| M2 | loader, ops, qwen3, sdpa, runner | greedy 64 tokens match `transformers` token-for-token |
| M3 | kv_cache, scheduler, sampler, engine | `engine.generate_batch()` matches M2 for 1 and 4 sequences |
| M4 | tokenizer, detokenizer, engine_client, server, main | the openai SDK streams a reply; `jq` shows the full trail |

Every block is built and tested on the Mac first: the control
plane against `tests/fakes.py::FakeRunner`, the data plane against a tiny random-weight Qwen3
checkpoint from `turbine_serve/testing/tiny_model.py` (real tensor names, real tokenizer, HF
reference model for fp32 parity on CPU). The Spark then runs the
same tests with the real 8 GB model in bf16 on the GB10, then end to end, image, benchmark.

## Two machines

Everything runs on the Mac first (`scripts/serve-tiny.sh` serves the tiny model on CPU). Edit on the Mac. `scripts/sync.sh` rsyncs to `dgx-spark:~/turbine-serve`; `scripts/spark.sh`
runs a command inside the `turbine-serve-dev` container with the tree mounted at `/w`.
Runbook: `docs/runbooks/spark-session.md`.
