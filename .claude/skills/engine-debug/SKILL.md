---
name: engine-debug
description: >-
  Debug playbook for spark-engine: garbage or repeated output, parity mismatch against
  transformers, wrong shapes, CUDA OOM on the Spark, a request that hangs, log forensics with jq.
  Use when output is wrong, a gpu test fails, or "it worked in HF but not here". Gotchas are
  appended here in the same commit that found them.
---

# engine-debug

Rules that must never be missed:

- **Reproduce with the smallest thing.** Greedy (`temperature=0`), one sequence, `max_tokens=8`,
  a fixed prompt. Only then widen.
- **Compare against `transformers`, not against memory.** `tests/test_parity.py` has the
  teacher-forced check that reports the first mismatching position; run it before guessing.
- **Read the log trail, not the stack dump first.**
  `jq -c 'select(.request_id=="req_…")' logs/engine.jsonl` shows which block decided what.
- **Do not loosen a gate to make it pass.** A parity tolerance change needs the owner.

## Symptom → first checks

| Symptom | Check, in order |
|---|---|
| Fluent-looking garbage, wrong language, repeats | QK-Norm applied to q and k *before* RoPE? RMSNorm in fp32 with HF's cast order (`weight * x.to(dtype)`)? RoPE rotate-half, theta 1e6, positions from `num_computed`? |
| Never stops | both EOS ids `151645` and `151643` in `eos_token_ids`; `max_tokens` clamped |
| First tokens right, drifts later | teacher-forced parity: find the first bad position; if it is a near-tie in bf16, check cast order in ops before anything else |
| Right for 1 seq, wrong for 4 | `slot_mapping` built from each seq's own `block_table`; `logits_indices` = last token of each seq in the flat batch; SDPA `is_causal` only when `q_len == ctx_len` |
| Prefill right, decode wrong | decode input is `output_ids[-1]`, position `num_computed`; attention gathers `ctx_len = num_computed + 1` tokens |
| `KeyError` on load | tied head: `lm_head.weight` absent → alias to `model.embed_tokens.weight`; names in `docs/qwen3-and-spark.md` |
| CUDA OOM at start | `qwen38-flash` still running (`free -g`); `--kv-gb` too big; parity test loads two models (use `kv_gb=4`) |
| Request hangs, no tokens | scheduler `seq.waiting` reason (`kv_full` / `max_seqs`); impossible request should have been a 400 at `add` |
| SSE client sees nothing until the end | chunks flushed per token? detokenizer returning `""` (partial UTF-8 is expected for a few tokens, not forever) |
| Step fails every time | `.logs/tests.log` or `step.failed` line carries `exc`; the loop keeps running, so look at the first failure |

## Log queries

```bash
jq -c 'select(.level=="error")' logs/engine.jsonl | tail -5
jq -r 'select(.event=="step.done") | [.step_id,.num_seqs,.num_tokens,.forward_ms] | @tsv' logs/engine.jsonl | tail -20
jq -c 'select(.event=="seq.completed") | {seq_id,finish_reason,ttft_ms,tpot_ms}' logs/engine.jsonl
```

## Gotchas (append below, newest last, one line each with the date)

- 2026-10-20: `apply_chat_template(tokenize=True)` changed return type across transformers majors;
  use `tokenize=False` then `encode(add_special_tokens=False)`.
- 2026-10-20: ruff formats fenced Python in `.md` files; keep `extend-exclude` for docs/ and plan/.
