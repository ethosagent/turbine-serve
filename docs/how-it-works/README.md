# How it works

Follow one chat request from the HTTP socket, through the engine, and back as streamed text.
Every step names the file and function that does it, so you can open the code next to the page.

Read in order:

| # | Page | Covers |
|---|---|---|
| 1 | [01-request-in.md](01-request-in.md) | HTTP body → token ids → a `Sequence` waiting in the scheduler |
| 2 | [02-engine-step.md](02-engine-step.md) | one engine step: schedule → flatten → forward → attention → sample → update |
| 3 | [03-response-out.md](03-response-out.md) | a sampled token id → engine thread → asyncio queue → text → SSE chunk |
| 4 | [04-errors-and-debugging.md](04-errors-and-debugging.md) | every failure path, the log trail, where to put a breakpoint |

## The whole trip on one screen

```text
 client                    event-loop thread (uvicorn)                  engine thread
 ──────                    ───────────────────────────                  ─────────────
 POST /v1/chat/completions
   │
   ├─▶ api/server.py      chat_completions() → handle()
   │     tokenizer.py     HFTokenizer.encode_chat()        text → ids
   │     engine_client.py EngineClient.generate()          ids → Sequence
   │                        └─ engine.add_request() ───────────▶ scheduler.add()  (WAITING)
   │                                                              │
   │                                                     Engine.run_forever() loop:
   │                                                       step()
   │                                                         scheduler.schedule()   admit + batch
   │                                                         runner.execute()       flat tensors
   │                                                           qwen3.forward()      36 layers
   │                                                             sdpa.forward()     paged attention
   │                                                         sampler.sample()       logits → id
   │                                                         scheduler.update()     EOS / length?
   │                                                         _emit() → on_output()
   │                        ┌─ asyncio.Queue ◀── call_soon_threadsafe ─┘
   │     engine_client.py generate() yields TokenOutput
   │     api/server.py    _stream(): detokenizer.step() → SSE chunk
   ◀─────────────────────  data: {...chat.completion.chunk...}   ... data: [DONE]
```

Two threads, one lock:

- **Event-loop thread** runs FastAPI: parsing, tokenizing, detokenizing, writing SSE. It never
  runs the model.
- **Engine thread** (started by `EngineClient.start()` from the app's lifespan) runs
  `Engine.run_forever()`. It is the only thread that touches the GPU.
- `Engine._lock` guards every scheduler call. `add_request` takes it on the event-loop thread;
  `step` takes it on the engine thread. Forward and sampling run outside the lock.

## Component map

| Stage | File | Start reading at | Log events (`component`) | Tests |
|---|---|---|---|---|
| CLI + startup | `main.py` | `cli()` → `serve()` | `config.resolved`, `server.started` | `test_config.py` |
| Wiring | `engine.py` | `build_engine()` | `kv.initialized`, `model.loaded`, `engine.built` | `test_engine_model.py` |
| HTTP + SSE | `api/server.py` | `create_app()` → `chat_completions()` | `request.received`, `request.completed`, `client.disconnected` | `test_api.py`, `test_api_e2e.py` |
| Chat template | `tokenizer.py` | `HFTokenizer.encode_chat()` | `prompt.encoded`, `prompt.too_long` | `test_tokenizer.py` |
| Thread bridge | `engine_client.py` | `EngineClient.generate()`, `_on_output()` | `seq.submitted`, `seq.aborted` | `test_engine_client.py` |
| Step loop | `engine.py` | `Engine.run_forever()` → `step()` | `engine.started`, `step.done`*, `step.failed`, `seq.completed` | `test_engine.py` |
| Batching policy | `scheduler.py` | `FCFSScheduler.schedule()`, `update()` | `seq.admitted`, `seq.waiting`*, `seq.finished`, `seq.failed` | `test_scheduler.py` |
| KV memory | `kv_cache.py` | `PagedKVCache.allocate()`, `free()` | `kv.initialized`, `blocks.allocated`*, `blocks.freed`* | `test_kv_cache.py` |
| Flatten batch | `model_runner.py` | `GPUModelRunner.prepare()`, `execute()` | `batch.prepared`*, `forward.failed` | `test_parity.py`, `test_engine_model.py` |
| Transformer | `models/qwen3.py` | `Qwen3Model.forward()` → `_layer()` → `_attention()` | `model.loaded` | `test_parity.py`, `test_bf16_report.py` |
| Math kernels | `layers/ops.py` | `rms_norm`, `apply_rope`, `silu_and_mul` | — | `test_ops.py` |
| Attention | `attention/sdpa.py` | `SDPAAttention.forward()` → `_attend_one()` | `attention.metadata`* | `test_sdpa.py` |
| Next token | `sampler.py` | `TorchSampler.sample()` | `sample.done`* | `test_sampler.py` |
| Ids → text | `detokenizer.py` | `IncrementalDetokenizer.step()` | — | `test_tokenizer.py` |
| Weights | `loader.py` | `resolve_model_path()`, `iter_weights()` | — | `test_loader.py` |
| Contracts | `types.py`, `interfaces.py` | `Sequence`, `ScheduledSeq`, `ForwardBatch`, `TokenOutput` | — | `test_types.py`, `test_interfaces.py` |

`*` = debug level; start the server with `--log-level debug` to see it. Paths are under
`turbine_serve/`, tests under `tests/`. The `component` field of a log line is always the file
name, so a line tells you which file wrote it.

## The four objects that travel

| Object (`types.py`) | Made by | Carried to | What it holds |
|---|---|---|---|
| `Sequence` | `EngineClient.generate()` | scheduler, runner, sampler | prompt ids, output ids, status, `num_computed`, `block_table` |
| `SchedulerOutput` | `FCFSScheduler.schedule()` | runner, attention, `update()` | `step_id` + a `ScheduledSeq` per sequence (`num_new_tokens`) |
| `ForwardBatch` | `GPUModelRunner.prepare()` | `Qwen3Model.forward()` | flat `[T]` tensors: `input_ids`, `positions`, `slot_mapping`, `logits_indices` |
| `TokenOutput` | `FCFSScheduler.update()` / `fail()` | engine client → API | one token id, `finished`, `finish_reason` |

Rules for the layers between them are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md); the
per-block spec is [`SPEC.md`](../SPEC.md).
