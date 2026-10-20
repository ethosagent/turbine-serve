# 4. Errors and debugging

## Every failure path

| What goes wrong | Raised in | Caught in | Client gets | Log events |
|---|---|---|---|---|
| Body fails the schema | FastAPI, against `ChatCompletionRequest` | FastAPI | **422** (FastAPI's own `detail` shape) | none from our code |
| Prompt over `--max-model-len` | `HFTokenizer.encode_chat()` → `PromptTooLong` | `handle()` `except ValueError` | **400** `invalid_request_error` | `prompt.too_long`, `request.completed` |
| `prompt + max_tokens` can never fit in KV | `FCFSScheduler.add()` → `ValueError` | `handle()` `except ValueError` | **400** `invalid_request_error` | `request.completed` |
| Too many requests waiting | `EngineClient.generate()` → `QueueFull` | `handle()` `except QueueFull` | **503** `server_overloaded` | `request.completed` |
| Forward / sampler throws, before any output | `GPUModelRunner.execute()` or `TorchSampler.sample()` | `Engine._execute()` → `scheduler.fail()` | **500** `server_error` | `forward.failed`, `step.failed`, `seq.failed` |
| Same, mid-stream | same | same | 200 already sent; final chunk `finish_reason: "error"`, then `[DONE]` | same |
| Same, non-streaming | same | same | **500** `server_error` | same |
| Client disconnects | Starlette cancels `_stream()` | `_stream()` `except CancelledError` | nothing (gone) | `client.disconnected`, `seq.aborted`, `seq.finished reason=abort`, `request.completed status=499` |
| `on_output` consumer throws | `EngineClient._on_output()` | `Engine.step()` | depends | `emit.failed`; the loop keeps running |

Why each one lands where it does:

- **Before the first token** every error can still become a proper status, because `handle()`
  awaits the first output (`await anext(outs)`) before choosing a response.
- **A step failure fails only that step's sequences.** `_execute()` calls `scheduler.fail()` for
  each, which frees their blocks and returns a `TokenOutput` with `token_id=-1` and
  `finish_reason="error"`. Other requests waiting in the queue are not affected.
- **A disconnect frees the GPU work.** Cancelling `_stream()` closes the `generate()` generator; its
  `finally` sees `submitted and not finished` and calls `engine.abort(seq_id)`. The abort is queued,
  applied by `_drain_aborts()` at the start of the next step, and `scheduler._finish(seq, "abort")`
  returns the blocks.
- **A KV-full request waits, it does not fail.** Admission checks happen every step; a sequence that
  could fit later stays in `_waiting` with `seq.waiting reason=kv_full` (debug).

## Following a request

```bash
# start with debug lines on (per-step timings, blocks, batch shapes)
turbine serve Qwen/Qwen3-4B --kv-gb 30 --log-level debug

# the whole life of one request (request_id is in the response "id")
jq -c 'select(.request_id=="req_9f2c41ab")' logs/engine.jsonl

# which step admitted it, then every step line for that step
jq -c 'select(.event=="seq.admitted" and .request_id=="req_9f2c41ab") | .step_id' logs/engine.jsonl
jq -c 'select(.step_id==42)' logs/engine.jsonl

# anything that went wrong
jq -c 'select(.level=="error" or .level=="warning")' logs/engine.jsonl

# slow steps
jq -c 'select(.event=="step.done" and .forward_ms > 200)' logs/engine.jsonl
```

Logs go to stdout and are appended to `--log-file` (default `logs/engine.jsonl`). In Docker the
file is inside the container; use `docker logs` or mount `logs/`.

## Symptom → where to look

| Symptom | First check | Then open |
|---|---|---|
| Garbage or repeated text | parity test: `uv run pytest tests/test_parity.py -q` | `models/qwen3.py` `_attention()` (QK-Norm before RoPE), `layers/ops.py` (fp32 RMSNorm/RoPE) |
| Never stops, always hits `max_tokens` | `seq.finished` shows `length`, not `stop` | `HFTokenizer.eos_token_ids`, `FCFSScheduler.update()` |
| Correct first token, wrong after | decode path | `GPUModelRunner.prepare()` slot math, `sdpa._attend_one()` gather and trim |
| Request stuck, no tokens | `seq.waiting` reasons (debug) | `FCFSScheduler.schedule()`, `--max-num-seqs`, `--kv-gb` |
| 503 under load | `seq.submitted` vs `seq.admitted` gaps | `--max-queue`, `--max-num-seqs` |
| High TTFT | `seq.completed ttft_ms`, `step.done schedule_ms` | prompts waiting behind long prefills (no chunked prefill yet) |
| Low tok/s | `step.done forward_ms` vs `sample_ms` | `sdpa.py` Python loop per sequence, no CUDA graphs |
| Emoji or CJK broken in the stream | detokenizer test | `IncrementalDetokenizer.step()` U+FFFD hold-back |
| Text out of order between requests | — | `EngineClient._on_output()` queue lookup by `seq_id` |

The full debugging playbook is `.claude/skills/engine-debug/SKILL.md`.

## Where to put a breakpoint

| To see | Break in | Look at |
|---|---|---|
| the exact prompt the model sees | `HFTokenizer.encode_chat()` | `text` after `apply_chat_template` |
| who is in this step's batch | `Engine.step()` after `schedule()` | `sched.seqs` (`num_new_tokens`: >1 prefill, 1 decode) |
| the flat tensors | `GPUModelRunner.prepare()` before `return` | `input_ids`, `positions`, `slots`, `logits_indices` |
| one layer's attention | `Qwen3Model._attention()` | `q`, `k` shapes, `batch.slot_mapping` |
| what was sampled and why | `TorchSampler.sample()` | `row.topk(5)` for each sequence |
| why a sequence ended | `FCFSScheduler.update()` | `token`, `reason` |
| text deltas | `IncrementalDetokenizer.step()` | `prefix_text`, `new_text` |

The engine runs on its own thread: in VS Code or `debugpy` that just works; with `pdb`, put
`breakpoint()` in the engine-side code and run with one request at a time.

## Reproduce without a GPU

Every page of this walkthrough runs on a laptop:

```bash
scripts/serve-tiny.sh                      # random-weight tiny Qwen3 on CPU, real tokenizer
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":8}'
```

The text is gibberish (random weights) but every function, log line and error path above is the
real one. Engine logic without any model: `tests/fakes.py::FakeRunner`, used by `test_engine.py`.
