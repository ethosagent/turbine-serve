# Conventions — behavioural rules and spark-engine specifics

`CLAUDE.md` carries the condensed version; this file is the authority when it is ambiguous.

## Behavioural rules

1. **Think before coding.** State assumptions. If two readings exist, present them. If a simpler
   approach exists, say so. If something is unclear, stop and ask.
2. **Simplicity first.** Minimum code that solves the problem. No speculative features, no
   abstractions for single-use code, no configurability nobody asked for. 200 lines that could
   be 50 get rewritten.
3. **Surgical changes.** Touch only what the request requires. Do not reformat or "improve"
   adjacent code. Remove what your change made unused. Every changed line traces to the request.
4. **Goal-driven execution.** Define the test first, then make it pass. "Fix the bug" means a
   reproducing test, then the fix. `make check` before declaring done.
5. **Surface conflicts, don't average them.** If two patterns in the repo contradict, pick one,
   say why, flag the other. Code that half-satisfies both rules is the worst outcome.
6. **Ask before adding to code you don't understand.** If you cannot say why a block is shaped the
   way it is, read `docs/SPEC.md` for that block, then ask.
7. **Follow the constitution.** `ARCHITECTURE.md` before moving modules or crossing layers.

## spark-engine specifics

- **One block, one file, one component.** The log `component` is the file name; the gate
  `scripts/check-components.sh` enforces it. Events are `noun.verb` (`seq.admitted`,
  `step.done`, `forward.failed`). Lifecycle at info, per-step at debug.
- **fp32 math, bf16 storage.** RMSNorm, RoPE and SwiGLU compute in fp32 and cast back. Weights
  and KV stay bf16.
- **Shapes in comments.** Every tensor-producing line names its shape: `# [T, 32, 128]`.
- **Numbers come from config.** Head counts, layer counts, vocab, rope theta are read from
  `AutoConfig`; literal numbers appear only in tests.
- **Fakes over mocks, tiny over fake where the real code can run.** `tests/fakes.py::FakeRunner`
  returns deterministic logits (next token = last token + 1) for control-plane tests. Data-plane
  tests run the real code on CPU against `spark_engine/testing/tiny_model.py` (2-layer random
  Qwen3 with the real tokenizer) and compare with `transformers`. Nothing is patched.
- **Device and dtype are config.** No `.cuda()` in blocks; `device`/`dtype` flow from
  `EngineConfig`. `torch.cuda.synchronize()` only when the device is cuda.
- **Timing is measured after the GPU sync** so `forward_ms` is real.
- **Errors fail sequences, not the loop.** A step exception finishes the affected sequences with
  `finish_reason="error"` and the engine keeps running.
- **Function ≤ 50 statements (ruff PLR0915), file ≤ 300 lines soft / 800 hard, docs ≤ 200.**
- **Commits:** one concern each, imperative subject (`feat: paged KV cache manager`), ending with
  the attribution line the session provides.
