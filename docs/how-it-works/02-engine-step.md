# 2. One engine step: schedule → forward → sample → update

Everything on this page runs on the engine thread. One call to `Engine.step()` produces exactly
one new token for every sequence in the batch.

## The loop — `engine.py`

`Engine.run_forever()`:

```text
while not stopped:
    busy = scheduler.has_work() or aborts pending      # under _lock
    if busy: step()
    else:    _wake.wait(0.05 s)                        # add_request / abort sets _wake
```

`Engine.step()` does five things in order:

| # | Call | Under `_lock`? | Page section |
|---|---|---|---|
| 1 | `_drain_aborts()` | yes | applies `abort()` calls queued by disconnected clients |
| 2 | `scheduler.schedule()` | yes | [A. Schedule](#a-schedule--schedulerpy-kv_cachepy) |
| 3 | `_execute()` → `runner.execute()` | no | [B](#b-flatten--model_runnerpy), [C](#c-forward--modelsqwen3py), [D](#d-attention--attentionsdpapy) |
| 4 | `_execute()` → `sampler.sample()`, then `scheduler.update()` | `update` only | [E](#e-sample--samplerpy), [F](#f-update--schedulerpy) |
| 5 | `_emit()` | no | page 3 |

If the scheduler returns no sequences, `step()` returns at once.

## A. Schedule — `scheduler.py`, `kv_cache.py`

`FCFSScheduler.schedule()` builds the batch for this step:

1. `step_id += 1`.
2. Every **RUNNING** sequence gets `ScheduledSeq(seq, 1)`: one decode token.
3. Then, strictly first come first served, it admits from the head of `_waiting` while both hold:
   - `num_running < --max-num-seqs` (default 8), else logs `seq.waiting reason=max_seqs`;
   - `kv.can_allocate(prompt_len + max_tokens)`, else logs `seq.waiting reason=kv_full`.

   It stops at the first sequence that does not fit (no skipping ahead).
4. Admitting a sequence: `kv.allocate(seq, prompt_len + max_tokens)` pops that many blocks off the
   free list into `seq.block_table`, status becomes `RUNNING`, and it gets
   `ScheduledSeq(seq, prompt_len)`: a **prefill**. Logs `seq.admitted` with `blocks_reserved`.

Our example: prompt 10 + `max_tokens` 512 = 522 tokens = `ceil(522 / 16)` = **33 blocks**,
reserved now, for the whole life of the sequence. There is no preemption, so a running sequence
can never run out of KV.

`PagedKVCache` (`kv_cache.py`) is two tensors, K and V, each
`[num_layers, num_blocks, block_size, num_kv_heads, head_dim]` = `[36, NB, 16, 8, 128]` for Qwen3-4B.
`NB` comes from `--kv-gb` in `compute_num_blocks()`. The free list is a stack; the lowest block
id comes out first.

## B. Flatten — `model_runner.py`

`GPUModelRunner.execute(sched)` → `prepare(sched)` turns the batch into one flat stream of `T`
tokens. There is no `[batch, seq_len]` padding anywhere.

For each scheduled sequence:

- tokens = the whole prompt if `num_computed == 0` (prefill), else just `output_ids[-1:]` (decode);
- positions = `num_computed ... num_computed + len(tokens) - 1`;
- slot for each position = `block_table[pos // 16] * 16 + pos % 16`: where its K/V row goes;
- `logits_indices` gets the index of the sequence's **last** token in the flat stream.

Worked example: `s0` prefilling 10 tokens, batched with `s1` decoding at position 40 with
`block_table = [33, 34, 35, ...]`:

```text
input_ids      [p0 p1 p2 ... p9 | t40]                  T = 11
positions      [ 0  1  2 ...  9 | 40 ]
slot_mapping   [ 0  1  2 ...  9 | 35*16 + 8 = 568 ]      block 40//16=2 → id 35, offset 8
logits_indices [9, 10]                                   one row per sequence, B = 2
```

`SDPAAttention.build_metadata()` adds per-sequence `q_start`, `q_len`, `ctx_len` and the block
tables. Logs `batch.prepared` and `attention.metadata` (debug).

## C. Forward — `models/qwen3.py`

`Qwen3Model.forward(batch, kv, attn)` under `torch.inference_mode()`. Weights are plain tensors in
a dict keyed by checkpoint name; there is no `nn.Module`.

```text
h = embed[input_ids]                          [T, C]
cos, sin = rope_cos_sin(positions)            fp32
36 × _layer():
    x = rms_norm(h);  h += _attention(x)
    x = rms_norm(h);  h += down(silu(gate(x)) * up(x))
h = rms_norm(h)
```

Inside `_attention()`:

1. Project to `q [T, 32, 128]`, `k` and `v` `[T, 8, 128]` (grouped-query attention: 4 query heads
   share one KV head).
2. **QK-Norm**: `rms_norm` per head on `q` and `k`, then `apply_rope`. Order matters; forgetting
   it is the classic garbage-output bug.
3. **Write this step's K/V into the paged cache**: `k_cache.view(-1, 8, 128)[slot_mapping] = k`
   (same for `v`). This is the only place the cache is written.
4. `attn.forward(q, k_cache, v_cache, batch.attn)` reads it back (section D).

Then `compute_logits(hidden, logits_indices)` multiplies only the B last-token rows by the LM head:
logits `[B, vocab]` in fp32. On cuda, `torch.cuda.synchronize()` runs so `forward_ms` is real.

## D. Attention — `attention/sdpa.py`

`SDPAAttention.forward()` loops over sequences and calls `_attend_one()` for each:

1. Take the first `ceil(ctx_len / 16)` ids from the block table.
2. Gather those blocks from the cache and trim to `ctx_len` rows → `k, v [ctx, 8, 128]`.
3. `F.scaled_dot_product_attention(q, k, v, is_causal=q_len > 1, enable_gqa=True)`.
   - Prefill: `q_len == ctx_len`, causal mask on.
   - Decode: `q_len == 1`, attends to every cached token, no mask needed.
   - Anything else trips an assert (chunked prefill is not built yet).

A Python loop per sequence is the simple MVP choice; FlashInfer would replace this whole file
behind the same `AttentionBackend` Protocol.

## E. Sample — `sampler.py`

`TorchSampler.sample(logits, seqs)` returns one token id per row:

- `temperature == 0` → `argmax` (greedy, deterministic).
- otherwise `_filter()`: divide by temperature → keep top-k → keep the smallest set whose
  probability passes top-p → `softmax` → `multinomial`.
- With a `seed`, the RNG is seeded by `draw_seed(seed, seq.num_tokens)`, so the same request
  gives the same text regardless of what else is in the batch.

Back in `Engine._execute()`, a token count that differs from the sequence count raises before
anything is mutated.

## F. Update — `scheduler.py`

`FCFSScheduler.update(sched, tokens)`, under the lock, for each sequence still `RUNNING`:

1. `output_ids.append(token)`; `num_computed += num_new_tokens` (10 after our prefill, +1 per decode).
2. Finish check: token in `{151645, 151643}` (`<|im_end|>`, `<|endoftext|>`) → `"stop"`;
   `len(output_ids) >= max_tokens` → `"length"`.
3. On finish, `_finish()` removes it from `_running`, **frees its blocks**, sets `FINISHED`, logs
   `seq.finished`.
4. Returns a `TokenOutput(request_id, seq_id, token, finished, finish_reason)` per sequence.

Note the off-by-one that makes decoding work: the token sampled in this step has **no** K/V in
the cache yet. Next step feeds it as `output_ids[-1:]` at position `num_computed`, and its K/V is
written then.

`_execute()` logs `step.done` (debug) with `num_seqs`, `num_tokens`, `schedule_ms`, `forward_ms`
and `sample_ms`: the first place to look when something is slow.

Next: [03-response-out.md](03-response-out.md) — getting the token back to the client.
