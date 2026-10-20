# 1. Request in: HTTP body → `Sequence` in the scheduler

Everything on this page runs on the event-loop thread, inside one call to
`chat_completions()`. Nothing touches the GPU yet.

Running example: a user sends `"hi"` with `stream: true` and no `max_tokens`.

## Step 1. The route — `api/server.py`

`chat_completions(req)` is the `POST /v1/chat/completions` handler inside `create_app()`.

1. FastAPI validates the body against `ChatCompletionRequest` before the handler runs. A bad body
   (empty `messages`, `top_p` > 1, unknown role) is rejected here with **422**; no log line from
   our code.
2. It makes a request id: `"req_" + secrets.token_hex(4)`, e.g. `req_9f2c41ab`.
3. `bind_request(request_id)` (`log.py`) puts the id in structlog's contextvars, so every log line
   written on this thread for this request carries `request_id`.
4. Logs `request.received` with `stream` and `max_tokens`, then calls `handle()`.

## Step 2. Chat template + encode — `tokenizer.py`

`handle()` reads `chat_template_kwargs.enable_thinking` (default `False`) and calls
`HFTokenizer.encode_chat(messages, enable_thinking=...)`:

1. `apply_chat_template(..., add_generation_prompt=True)` renders Qwen3's template: the messages
   wrapped in `<|im_start|>user ... <|im_end|>` plus the opening of the assistant turn.
2. `hf.encode(text, add_special_tokens=False)` gives the prompt ids. The template adds more ids
   than the message itself; the example uses a round 10.
3. More than `--max-model-len` ids (default 8192) raises `PromptTooLong`, a `ValueError` → **400**.
4. Logs `prompt.encoded` with `num_prompt_tokens` and `ms`.

## Step 3. Sampling params — `api/server.py`

`_sampling_params()` builds a frozen `SamplingParams` (`types.py`):

| Field | From the request | Default |
|---|---|---|
| `max_tokens` | `max_tokens`, then clamped to `max_model_len - prompt_len` | 512 |
| `temperature` | `temperature` (0 = greedy) | 0.7 |
| `top_p` / `top_k` | `top_p` / `top_k` | 0.8 / 20 |
| `seed` | `seed` | none |

The defaults are Qwen3's recommended non-thinking settings.

## Step 4. Into the bridge — `engine_client.py`

`client.generate(prompt_ids, params, request_id)` is an async generator. `handle()` calls
`await anext(outs)` right away, which runs everything up to the first token. That is why queue
and capacity errors surface **before** any HTTP status is sent.

Inside `EngineClient.generate()`:

1. **Back-pressure.** If `engine.num_waiting >= --max-queue` (default 64), raise `QueueFull` → **503**.
2. **Make the `Sequence`.** `seq_id` is `s0`, `s1`, ... from a counter. Status starts `WAITING`.
3. **Register the output queue first.** `self._queues[seq_id] = asyncio.Queue()` happens before
   the engine sees the sequence, so the first token can never arrive with nowhere to go.
4. `engine.add_request(seq)` (next step), then log `seq.submitted`.
5. Wait on `await out_q.get()` and yield each `TokenOutput` until one has `finished=True`.

## Step 5. Admission check — `engine.py` → `scheduler.py`

`Engine.add_request(seq)` runs on the **caller's** thread (the event loop), on purpose: a
`ValueError` from the scheduler must reach the HTTP handler.

1. Stamps `seq.arrival_ts` (used later for TTFT).
2. Under `Engine._lock`: `FCFSScheduler.add(seq)`.
   - It checks the request could **ever** fit: `blocks_needed(prompt_len + max_tokens) >
     kv.num_blocks` raises `ValueError("prompt + max_tokens exceeds KV capacity")` → **400**.
   - Otherwise it appends the sequence to the `_waiting` deque and to `_by_id`.
3. Appends to `Engine._added` (backs `num_waiting` for the back-pressure check).
4. `self._wake.set()` wakes the engine thread if it was idle.

Note what `add` does **not** do: it reserves no KV blocks. Blocks are taken at admission, in the
next `schedule()` (page 2).

## State at the end of page 1

```text
Sequence s0  request_id=req_9f2c41ab  status=WAITING
             prompt_ids=[~10 ids]  output_ids=[]  num_computed=0  block_table=[]
scheduler._waiting = [s0]        EngineClient._queues = {"s0": <empty asyncio.Queue>}
handler is parked on: await anext(outs)  →  await out_q.get()
```

Log lines so far, all tagged `request_id=req_9f2c41ab`:

```text
api            request.received
tokenizer      prompt.encoded
engine_client  seq.submitted
```

Next: [02-engine-step.md](02-engine-step.md) — the engine thread picks `s0` up.
