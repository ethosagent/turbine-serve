# 3. Response out: token id → SSE chunk

Page 2 ended with a list of `TokenOutput`s on the engine thread. This page carries one of them
across the thread boundary and turns it into bytes on the socket.

## Step 1. Emit — `engine.py`

`Engine._emit(sched, outs)`, still on the engine thread:

1. Stamps `seq.first_token_ts` the first time a sequence gets a real token.
2. For finished sequences, `_log_finished()` writes `seq.completed` with `ttft_ms`
   (arrival → first token) and `tpot_ms` (average time per output token after the first).
3. Calls `self.on_output(outs)`, which `EngineClient.start()` set to `EngineClient._on_output`.

An exception in step 3 is logged as `emit.failed` and swallowed: a bug in the consumer must
not stop the loop that every other request depends on.

## Step 2. Cross the thread boundary — `engine_client.py`

`EngineClient._on_output(outs)` runs on the engine thread but must not touch asyncio objects
directly. For each output:

1. Look up `self._queues[seq_id]`. Missing means the request was already cancelled: drop it.
2. `loop.call_soon_threadsafe(out_q.put_nowait, out)` schedules the put on the event loop.
   This is the only crossing from the engine thread to the event-loop thread.

On the event loop, `generate()` wakes from `await out_q.get()` and yields the `TokenOutput`.
When it yields one with `finished=True`, it leaves the loop; its `finally` removes the queue.

## Step 3. Ids to text — `detokenizer.py`

`handle()` gave each request its own `IncrementalDetokenizer` via `tokenizer.new_detokenizer()`.
`_Request.text(out)` calls `detok.step(token_id)`:

1. Append the id, decode a short window twice: `ids[prefix:read]` and `ids[prefix:]`.
2. The new text is whatever the second decode adds beyond the first.
3. If it ends in `�` (U+FFFD) it is half of a multi-byte character (an emoji, CJK). Return `""`
   and wait for the next token to complete it.

So one token does **not** always mean one text delta. Decoding only the tail window keeps every
step the same cost, however long the reply gets (vLLM uses the same prefix/read offset scheme).

## Step 4a. Streaming — `api/server.py` `_stream()`

With `stream: true`, `handle()` returns `StreamingResponse(_stream(outs, first, ctx))`. HTTP 200
and the headers go out now; from here on the status code can no longer change.

```text
for each TokenOutput:
    text = ctx.text(out)
    if text: send  data: {"object":"chat.completion.chunk", "choices":[{"delta":{...}}]}
             (the first chunk's delta also carries "role":"assistant")
after the last one:
    send  data: {... "delta":{}, "finish_reason":"stop" | "length" | "error"}
    send  data: [DONE]
    log   request.completed status=200
```

What the client sees for `"hi"`:

```text
data: {"id":"req_9f2c41ab","object":"chat.completion.chunk","created":1792483200,"model":"Qwen/Qwen3-4B","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"req_9f2c41ab","object":"chat.completion.chunk","created":1792483200,"model":"Qwen/Qwen3-4B","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"req_9f2c41ab","object":"chat.completion.chunk","created":1792483200,"model":"Qwen/Qwen3-4B","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

`_stream()` calls `bind_request()` again because Starlette may run the response body in a task
that did not inherit the handler's contextvars; without it the last log lines lose `request_id`.

## Step 4b. Non-streaming — `api/server.py` `_complete()`

With `stream: false`, `_complete()` drains the whole generator first, then:

- last output `finish_reason == "error"` → **500** `server_error`;
- otherwise joins all text deltas into one `message.content` and fills `usage`
  (`prompt_tokens`, `completion_tokens`, `total_tokens`) → **200**.

## Step 5. Cleanup

When the last output is consumed:

| Where | What is released |
|---|---|
| `scheduler._finish()` (page 2, already) | KV blocks back to the free list; sequence removed from `_running` and `_by_id` |
| `EngineClient.generate()` `finally` | the per-request `asyncio.Queue` |
| `chat_completions()` / `_stream()` `finally` | `clear_request()` drops `request_id` from the log context |

## The complete log trail of one request

`jq -c 'select(.request_id=="req_9f2c41ab")' logs/engine.jsonl` gives, in order:

```text
api            request.received      stream, max_tokens
tokenizer      prompt.encoded        num_prompt_tokens, ms
engine_client  seq.submitted         seq_id, prompt_tokens
scheduler      seq.admitted          step_id, blocks_reserved, waiting, running
scheduler      seq.finished          finish_reason, completion_tokens
engine         seq.completed         ttft_ms, tpot_ms
api            request.completed     status, duration_ms
```

The debug-level step lines (`batch.prepared`, `step.done`, ...) are per step, not per request, so
they carry `step_id` instead of `request_id`. Join them through the `step_id` in `seq.admitted`.

Next: [04-errors-and-debugging.md](04-errors-and-debugging.md).
