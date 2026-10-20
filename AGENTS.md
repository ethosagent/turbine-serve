# spark-engine — agent map

A minimal LLM inference server (distribution `turbine-serve`, package `spark_engine`) that serves
Qwen3-4B on one NVIDIA DGX Spark through an OpenAI-compatible API, built from scratch to learn how
vLLM, SGLang, TensorRT-LLM and TensorFold work. Correctness over speed; speed-ups plug in later.

- `spark_engine/` — one block per file in request order: `api/server.py` → `tokenizer.py` →
  `engine_client.py` → `engine.py` → `scheduler.py` · `kv_cache.py` · `model_runner.py` →
  `models/qwen3.py` · `attention/sdpa.py` → `sampler.py` → `detokenizer.py`. Kernel:
  `types.py` (contracts), `interfaces.py` (Protocols), `config.py`, `log.py`.
- `tests/` — pytest; `gpu`-marked tests skip without CUDA, `e2e` without `TURBINE_BASE_URL`.
- `plan/` — tracked: Plan A `2026-10-20-build-on-mac.md` (everything, on the Mac), Plan B
  `2026-10-20-verify-on-spark.md` (real model on the GB10), `decisions/` (ADRs), `reference/`
  (visual references, never read). The hand-off between sessions.
- `docs/` — `SPEC.md` (block spec; read the block you touch), orientation, conventions, numbers, runbooks.

## Golden path

- `make check` runs every `scripts/check-*.sh` through `scripts/run-checks.sh`: lint (ruff) ·
  typecheck (mypy strict) · deps (import-linter) · components · sizes · twins · tests. Output is
  failures only; full logs in `.logs/`. The pre-push hook and CI run the SAME scripts. Never claim
  it passes from memory; re-run.
- `uv run pytest tests/test_x.py -q` for one file. `make fmt` to format.
- GPU work: `make sync` then `scripts/spark.sh pytest -q` (runs inside the NGC container on the
  Spark); `make spark-serve` to serve. Procedure: `.claude/skills/spark-session/SKILL.md`.
- Editing happens on the Mac only; the Spark runs what `scripts/sync.sh` pushed.

## Dependency law (ARCHITECTURE.md §II; enforced by `check-deps`)

Allowed: everything imports the kernel; `engine.py` alone imports and wires blocks;
`api/` → `engine_client.py` + `tokenizer.py`; `models/` → `layers/`; `tokenizer` → `detokenizer`.

FORBIDDEN (named import-linter contracts in `pyproject.toml`):
- A block importing a sibling block, or anything importing upward — `layer-order`.
- `api/` or `engine_client.py` importing any block — `front-door-no-blocks`.
- `api/` importing `engine.py` — `api-only-via-engine-client`.
- `torch` / `transformers` in `scheduler.py`, `api/`, `engine_client.py` — `control-plane-no-torch`.

## Non-negotiables

1. **D1–D10 are fixed** (`plan/decisions/0001-mvp-scope.md`). Ask before changing any.
2. **Contracts only in `types.py`, seams only in `interfaces.py`.** New fields get defaults.
3. **Flat `[T]` tensors; never pad to `[batch, len]`.** Paged KV from day one.
4. **Log component == file name.** `get_logger("scheduler")` lives only in `scheduler.py`
   (gate: `check-components`). One JSON line per event; lifecycle at info, per-step at debug.
5. **fp32-on-cuda parity is the gate.** M2 passes only when Qwen3-4B in fp32 on cuda matches
   `transformers` fp32: argmax at every position, greedy 64 tokens. bf16 is reported, not
   gated (ADR 0004).
6. **Twin docs.** `CLAUDE.md` == `AGENTS.md`, same commit (gate: `check-twins`).
7. **Done ships docs.** Tests for the change, skills/docs updated, `make check` green, one commit.
   Workflow: `.claude/skills/milestone-workflow/SKILL.md`.

## Where new code goes

| Change | Where |
|---|---|
| New attention backend / model / KV manager / sampler | new file implementing the Protocol + registry entry + config flag. Recipe: `.claude/skills/add-block/SKILL.md` |
| New sampling or request field | `types.SamplingParams` (default!) → `api/server.py` schema → `sampler.py` |
| New endpoint | route in `api/server.py`; nothing below changes |
| New config knob | `config.EngineConfig` field with default; `main.py` gets the flag for free |
| New log event | in the block that decides it; name `noun.verb` (`seq.admitted`) |
| Scheduling policy (chunked prefill, priorities) | `scheduler.py` only |

## Testing

- Everything is tested on the Mac first. Control plane: `tests/fakes.py::FakeRunner`. Data plane
  (loader, model, attention, runner, parity): the tiny checkpoint from
  `spark_engine/testing/tiny_model.py` on CPU in fp32, with `transformers` as the reference.
  `scripts/serve-tiny.sh` serves it so the API runs end to end on the Mac.
- The Spark runs the SAME tests with `TURBINE_REAL_MODEL=1` (real weights, cuda; fp32 gated,
  bf16 reported). A failure seen on the Spark is reproduced on the Mac first if it is structural, then fixed here.
- Bug fix ⇒ a test that reproduces it first. A `types.py` change needs a test in every consumer.
- `tests/test_interfaces.py` checks every concrete class against its Protocol; add each new pair.

## What not to do

- Never forget QK-Norm (`q_norm`/`k_norm` before RoPE), fp32 math in RMSNorm/RoPE, EOS
  `151645` and `151643`. These are the three ways a hand-written Qwen3 produces garbage.
- Never pass `is_causal=True` to SDPA when `q_len != ctx_len`; MVP prefill always has them equal.
- Never size KV as a fraction of free memory on the Spark (unified memory). Use `--kv-gb`.
- Never `pip install torch` inside the NGC container; its build is the only one for sm_121.
- Never run GPU tests while `qwen38-flash` holds the Spark's memory (`docker stop` it first).
- Never read `plan/reference/*.html` (large visual references) or `uv.lock`; both are denied.
- Never import vLLM or SGLang code. Libraries are the ones in `pyproject.toml`.
- Never edit ARCHITECTURE.md casually, or `CLAUDE.md` without `AGENTS.md`.
- Never claim `make check` passes without re-running it.

## Pointers

| Need | Where |
|---|---|
| Axioms, layer model, dependency law, module map | `ARCHITECTURE.md` |
| Repo tour, how a request flows, milestones | `docs/orientation.md` |
| Behavioural rules in full | `docs/conventions.md` |
| D1–D10 and later ADRs | `plan/decisions/README.md` |
| Qwen3-4B numbers and the Spark's numbers | `docs/qwen3-and-spark.md` |
| Running on the Spark, memory, containers | `docs/runbooks/spark-session.md` |
| Building and shipping the image | `docs/runbooks/release.md` |
| Workflow: plan task → tests → code → check → commit | `.claude/skills/milestone-workflow/` |
| Debugging garbage output, parity drift, OOM | `.claude/skills/engine-debug/` |
| Ending a session | `.claude/skills/handoff/` |

# Compact instructions

Keep: the active `plan/*.md` path and current task number, files changed this session, the last
`make check` result, and any open decision waiting on the owner. Drop tool output and file dumps.
