# spark-engine — Architectural Constitution

> The laws of this codebase's structure: axioms, layers, dependency direction, extension
> boundaries. Short, restrictive, slow to change. Amend deliberately (§VI), never in a drive-by.

**Belongs here:** invariants that, if broken, make this a different project.
**Does not belong here:** file inventories, versions, gotchas, how-tos. Those live in
`CLAUDE.md` (the agent map), `docs/`, and `.claude/skills/`.

---

## I. Axioms

1. **Correctness over speed.** The MVP is proven by token-for-token parity with
   `transformers` on greedy decoding. Every speed-up must keep that parity test green.
2. **One block, one file.** A block is one responsibility, one file, one log `component`
   equal to its file name. No block reaches into another's internals.
3. **Contracts are data, seams are Protocols.** Blocks exchange only the dataclasses in
   `types.py` and depend on each other only through the `typing.Protocol`s in
   `interfaces.py`. New fields always get defaults so older blocks keep working.
4. **Batched shapes from day one.** Every forward pass takes flat `[T]` token tensors and a
   paged KV cache, even with one user. Concurrency is a config number, not a code path.
5. **Every block logs its decision.** One JSON object per line with `component`, `event`,
   and the ids (`request_id`, `seq_id`, `step_id`) needed to follow one request with `jq`.
6. **Explicit budgets.** KV memory is a stated number of gigabytes (D9), never a fraction of
   "free" memory; unified memory on the Spark makes the latter unreliable.
7. **Extension = add a file + register + config flag.** A new attention backend, model, or
   KV manager is a new file implementing an existing Protocol, registered by name. Nothing
   above it changes.

---

## II. Layer model and the dependency law

```
  client (curl / openai SDK) ──HTTP/SSE──► api/server.py          [1]
                                              │ engine_client.py   [3]   asyncio ↔ thread
                                              ▼
                                           engine.py              [4]   composition root + step loop
            ┌──────────────┬─────────────────┼──────────────┬──────────────┐
       scheduler.py   kv_cache.py    model_runner.py    sampler.py    tokenizer.py → detokenizer.py
          [5]            [6]              [7]              [10]          [2]            [11]
                                            │
                                   models/qwen3.py [8] → layers/ops.py
                                   attention/sdpa.py [9]
                                   loader.py
            ────────────────────────── kernel ──────────────────────────
                 config.py   interfaces.py   types.py   log.py
```

Numbers are the order a request passes through. Arrows point from importer to imported.

**Allowed (compile-time):** everything may import the kernel. `engine.py` is the only module
that imports blocks and wires them (the composition root). `engine_client.py` imports
`engine.py`. `api/` imports `engine_client.py` and `tokenizer.py` only. `main.py` imports
everything. `models/*` may import `layers/`. `tokenizer.py` may import `detokenizer.py`
(a factory).

**Forbidden (each is a named import-linter contract in `pyproject.toml`; CI fails on it):**

| Rule id | Forbids |
|---|---|
| `layer-order` | Any upward import, and any block importing a sibling block (scheduler ↔ kv_cache, model_runner ↔ models, …). Blocks see each other only as Protocols. |
| `front-door-no-blocks` | `api/` or `engine_client.py` importing any block. Only `engine.py` composes blocks. |
| `api-only-via-engine-client` | `api/` importing `engine.py`. The API's only door is `engine_client.py`. |
| `control-plane-no-torch` | `torch` or `transformers` inside `scheduler.py`, `api/`, `engine_client.py`. The control plane is pure Python and testable on a laptop. |

**Runtime:** the API runs on the asyncio loop in the main thread; the engine loop runs on one
background thread (D1). They share a process and talk only through queues owned by
`engine_client.py`. Moving the engine to its own process replaces that one file.

---

## III. Module map

| Module | Owns | Public surface (Protocol in `interfaces.py`) |
|---|---|---|
| `types.py` | Dataclasses: `SamplingParams`, `Sequence`, `ScheduledSeq`, `SchedulerOutput`, `AttentionMetadata`, `ForwardBatch`, `TokenOutput` | the data itself |
| `interfaces.py` | Protocols for every swappable block | `Tokenizer`, `Detokenizer`, `KVCacheManager`, `Scheduler`, `AttentionBackend`, `Model`, `ModelRunner`, `Sampler` |
| `log.py` | structlog JSON config, request context | `configure_logging`, `get_logger(component)` |
| `config.py` | `EngineConfig` and its defaults | fields only |
| `api/server.py` | OpenAI schemas, routes, SSE, HTTP error mapping | `create_app` |
| `tokenizer.py` | chat template, encode, prompt-length check | `Tokenizer` |
| `detokenizer.py` | incremental decode, UTF-8 hold-back | `Detokenizer` |
| `engine_client.py` | async iterator over engine outputs, abort on disconnect | `EngineClient` |
| `engine.py` | builds blocks from config; `step()`; per-step failure isolation | `Engine`, `build_engine` |
| `scheduler.py` | WAITING/RUNNING, admission, finish rules | `Scheduler` |
| `kv_cache.py` | paged K/V pools, free list, block tables | `KVCacheManager` |
| `model_runner.py` | `SchedulerOutput` → `ForwardBatch` → logits | `ModelRunner` |
| `models/` | model definitions by HF `model_type`; registry | `Model` |
| `attention/` | attention backends by name; registry | `AttentionBackend` |
| `sampler.py` | greedy / temperature / top-k / top-p, seeded | `Sampler` |
| `loader.py` | safetensors → GPU tensors | `iter_weights` |
| `layers/ops.py` | `rms_norm`, `rope`, `silu_and_mul` | functions |

---

## IV. Seams

- **HTTP ↔ engine:** `api/server.py` sees `EngineClient.generate()` yielding `TokenOutput`.
  Engine errors reach HTTP as typed exceptions (`PromptTooLong` → 400, `QueueFull` → 503).
- **Scheduler ↔ memory:** the scheduler reserves KV through `KVCacheManager.can_allocate` /
  `allocate` and never touches tensors. A prefix-caching manager is a second implementation.
- **Model ↔ attention ↔ cache:** the model writes K/V at `ForwardBatch.slot_mapping` and calls
  `AttentionBackend.forward(q, k_cache, v_cache, meta)`. A kernel library swap is a new backend file.
- **Decision ↔ compute:** `device` and `dtype` are config, never code. The control plane is
  tested with a fake runner; the data plane is tested on CPU in fp32 against a tiny checkpoint
  (`testing/tiny_model.py`) with `transformers` as the reference. The Spark only changes the
  model path, device and dtype.

---

## V. Fixed decisions

The MVP's scope decisions. They are not re-litigated in code review; changing one is agreed
with the owner first.

| ID | Decision |
|---|---|
| D1 | One process: FastAPI on the asyncio loop, engine loop on one background thread, bridged only by `engine_client.py`. |
| D2 | API: `POST /v1/chat/completions` (stream and non-stream), `GET /v1/models`, `GET /health`. |
| D3 | Batched data path from day one (flat `[T]` tensors, paged KV); `max_num_seqs = 8`. |
| D4 | Admission reserves KV for `prompt_len + max_tokens`, so no preemption. |
| D5 | `max_model_len = 8192`; the whole prompt is prefilled in one step. |
| D6 | Sampling: temperature, top_p, top_k, max_tokens, seed; EOS 151645 and 151643; defaults 0.7 / 0.8 / 20. |
| D7 | Thinking off by default; `chat_template_kwargs.enable_thinking` turns it on; think text inline. |
| D8 | Attention: torch SDPA per sequence over the paged cache, `enable_gqa=True`. |
| D9 | Explicit KV budget `--kv-gb` (default 30), block size 16. |
| D10 | NGC PyTorch container on the Spark; bind 127.0.0.1; no auth. |

Parity gate: M2 is gated on fp32-on-cuda parity with `transformers`; bf16 is reported, not gated.

---

## VI. AI-first commitments and amendment procedure

1. `CLAUDE.md` and `AGENTS.md` are byte-identical twins, same commit (gate: `check-twins`).
2. Every rule in §II is enforced by a tool (`check-deps`, `check-components`), not by memory.
3. Logs are machine-readable JSON with stable event names; the CLI is non-interactive.
4. Skills and docs ship in the same commit as the change; every gotcha lands in a skill.
5. **Amending this document:** write down the axiom or rule to change and why,
   add or update the import-linter contract in the same commit, then edit here.
