# spark-engine MVP spec

See CLAUDE.md for scope, decisions and rules. This file details each block.

## Contracts

```python
# types.py — the only things blocks pass to each other
@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    seed: int | None = None

class SeqStatus(Enum):
    WAITING = "waiting"; RUNNING = "running"; FINISHED = "finished"

@dataclass
class Sequence:
    seq_id: str
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    num_computed: int = 0                 # tokens whose KV is in cache
    block_table: list[int] = field(default_factory=list)
    finish_reason: str | None = None      # "stop" | "length" | "abort" | "error"
    arrival_ts: float = 0.0
    first_token_ts: float | None = None

@dataclass(frozen=True)
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int                   # prompt length on prefill, 1 on decode
    num_lookahead: int = 0                # reserved for speculative decoding

@dataclass(frozen=True)
class SchedulerOutput:
    step_id: int
    seqs: list[ScheduledSeq]

@dataclass(frozen=True)
class AttentionMetadata:                  # backend-specific subclasses allowed
    q_start: Tensor; q_len: Tensor; ctx_len: Tensor; block_tables: list[list[int]]

@dataclass(frozen=True)
class ForwardBatch:
    input_ids: Tensor                     # [T]
    positions: Tensor                     # [T]
    slot_mapping: Tensor                  # [T]  block_id * block_size + offset
    logits_indices: Tensor                # [B]  last token of each seq
    attn: AttentionMetadata

@dataclass(frozen=True)
class TokenOutput:
    request_id: str
    seq_id: str
    token_id: int
    finished: bool
    finish_reason: str | None = None
```

```python
# interfaces.py — every seam you can swap
class Tokenizer(Protocol):
    eos_token_ids: tuple[int, ...]
    def encode_chat(self, messages: list[dict], **template_kwargs) -> list[int]: ...
    def new_detokenizer(self) -> "Detokenizer": ...

class Detokenizer(Protocol):
    def step(self, token_id: int) -> str: ...

class KVCacheManager(Protocol):
    block_size: int
    num_free_blocks: int
    def can_allocate(self, num_tokens: int) -> bool: ...
    def allocate(self, seq: Sequence, num_tokens: int) -> None: ...
    def free(self, seq: Sequence) -> None: ...
    def layer_cache(self, layer: int) -> tuple[Tensor, Tensor]: ...

class Scheduler(Protocol):
    def add(self, seq: Sequence) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
    def update(self, out: SchedulerOutput, tokens: list[int]) -> list[TokenOutput]: ...
    def abort(self, seq_id: str) -> None: ...
    def has_work(self) -> bool: ...

class AttentionBackend(Protocol):
    def build_metadata(self, sched: SchedulerOutput, block_size: int) -> AttentionMetadata: ...
    def forward(self, q: Tensor, k_cache: Tensor, v_cache: Tensor,
                meta: AttentionMetadata) -> Tensor: ...

class Model(Protocol):
    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> None: ...
    def forward(self, batch: ForwardBatch, kv: KVCacheManager,
                attn: AttentionBackend) -> Tensor: ...
    def compute_logits(self, hidden: Tensor, indices: Tensor) -> Tensor: ...

class ModelRunner(Protocol):
    def execute(self, sched: SchedulerOutput) -> Tensor: ...

class Sampler(Protocol):
    def sample(self, logits: Tensor, seqs: list[Sequence]) -> list[int]: ...
```

## Blocks (request order)

### 1. API server — `api/server.py` (~170 lines)

Libraries: fastapi, uvicorn, pydantic

- Defines OpenAI request/response schemas with pydantic and validates input.
- Generates `request_id`, binds it to the logging context.
- Calls the tokenizer, submits to the engine client, streams SSE chunks or returns one JSON body.
- Maps engine errors to HTTP status codes (400 prompt too long, 503 queue full, 500 engine
  error before any output; a mid-stream failure ends the stream with `finish_reason: "error"`).
- Passes only `enable_thinking` from `chat_template_kwargs` to the tokenizer; other keys are ignored.

Interface:
```
POST /v1/chat/completions   ChatCompletionRequest -> SSE | ChatCompletionResponse
GET  /v1/models
GET  /health
```

Logs: request.received · request.rejected · request.completed (status, duration_ms) · client.disconnected

Extend later: New endpoints (/v1/completions, /v1/responses) are new routes here; nothing below changes.

### 2. Tokenizer — `tokenizer.py` (~50 lines)

Libraries: transformers (AutoTokenizer)

- Applies Qwen3's Jinja chat template, including the thinking toggle.
- Encodes to token IDs; checks prompt length against max_model_len.
- Exposes EOS IDs (151645, 151643) for the stop check.

Interface:
```
class Tokenizer(Protocol):
    eos_token_ids: tuple[int, ...]
    def encode_chat(self, messages, **template_kwargs) -> list[int]: ...
    def new_detokenizer(self) -> Detokenizer: ...
```

Logs: prompt.encoded (num_prompt_tokens, ms) · prompt.too_long

Extend later: Another model family only needs its own template, which transformers already reads from the checkpoint.

### 3. Engine client — `engine_client.py` (~90 lines)

Libraries: asyncio, threading (stdlib)

- Bridges the async API and the engine thread.
- `submit()` puts a Sequence on a thread-safe queue and returns an async iterator.
- Engine thread pushes TokenOutputs back with `loop.call_soon_threadsafe`.
- Handles client disconnect by asking the engine to abort the sequence.

Interface:
```
class EngineClient:
    async def generate(self, prompt_ids, params, request_id) -> AsyncIterator[TokenOutput]
    def abort(self, request_id) -> None
```

Logs: seq.submitted · seq.aborted

Extend later: Moving the engine to its own process (vLLM style) replaces this file with a ZeroMQ version; the API doesn't notice.

### 4. Engine loop — `engine.py` (~130 lines)

Libraries: stdlib

- Builds all blocks from config via registries.
- Loop: drain new requests → `schedule()` → `runner.execute()` → `sampler.sample()` → `scheduler.update()` → emit tokens.
- Times each phase and writes one `step.done` line.
- Catches exceptions per step, fails only the affected sequences, keeps running.

Interface:
```
class Engine:
    def add_request(self, seq: Sequence) -> None
    def abort(self, seq_id: str) -> None
    def step(self) -> list[TokenOutput]
    def run_forever(self) -> None
```

Logs: engine.started · step.done (debug: num_seqs, num_tokens, *_ms) · seq.completed (ttft_ms, tpot_ms) · step.failed

Extend later: Overlap scheduling (SGLang) splits step() into launch and collect here.

### 5. Scheduler — `scheduler.py` (~110 lines)

Libraries: stdlib (collections.deque)

- Holds WAITING and RUNNING queues.
- Each step: one decode token for every running seq; admit waiting seqs while `max_num_seqs` and KV allow (FCFS).
- Admission reserves KV for prompt + max_tokens (D4), so no preemption.
- `update()` appends sampled tokens, checks EOS / max_tokens, frees finished seqs.

Interface:
```
class Scheduler(Protocol):
    def add(self, seq: Sequence) -> None
    def schedule(self) -> SchedulerOutput
    def update(self, out: SchedulerOutput, tokens: list[int]) -> list[TokenOutput]
    def abort(self, seq_id: str) -> None
    def has_work(self) -> bool
```

Logs: seq.admitted (blocks_reserved, waiting, running) · seq.waiting (reason: kv_full | max_seqs) · seq.finished

Extend later: Chunked prefill, priorities and preemption are new policies in this one file.

### 6. KV cache manager — `kv_cache.py` (~100 lines)

Libraries: torch

- Allocates K and V pools on the GPU: per layer `[num_blocks, 16, 8, 128]` bf16.
- Keeps a free-block list on the CPU; gives each sequence a block table.
- Frees blocks when a sequence finishes or is aborted.
- Gives the model runner the pool tensors for each layer.

Interface:
```
class KVCacheManager(Protocol):
    block_size: int
    num_free_blocks: int
    def can_allocate(self, num_tokens: int) -> bool
    def allocate(self, seq: Sequence, num_tokens: int) -> None
    def free(self, seq: Sequence) -> None
    def layer_cache(self, layer: int) -> tuple[Tensor, Tensor]
```

Logs: kv.initialized (num_blocks, gb) · blocks.allocated · blocks.freed (free_blocks)

Extend later: A prefix-caching manager (vLLM / SGLang idea) is a second class implementing the same Protocol.

### 7. Model runner — `model_runner.py` (~110 lines)

Libraries: torch

- Turns SchedulerOutput into a ForwardBatch: flat input_ids, positions, slot_mapping, logits_indices.
- Asks the attention backend to build its metadata.
- Calls `model.forward()` then `model.compute_logits()` on the last token of each sequence.
- Runs under `torch.inference_mode()`; syncs once per step.

Interface:
```
class ModelRunner(Protocol):
    def execute(self, sched: SchedulerOutput) -> Tensor   # logits [B, vocab]
```

Logs: forward.failed (step_id, seq_ids, error) · debug: batch.prepared (num_tokens)

Extend later: CUDA graphs (vLLM / TRT-LLM / TensorFold) are captured and replayed here.

### 8. Qwen3 model — `models/qwen3.py` (~170 lines)

Libraries: torch

- 36 decoder layers: RMSNorm → Q/K/V → QK-Norm → RoPE → write KV → attention → O → residual → RMSNorm → SwiGLU → residual.
- Tied embedding / LM head.
- Owns weights as plain tensors; `load_weights()` maps checkpoint names.
- Registered as `"qwen3"` in `models/__init__.py`.

Interface:
```
class Model(Protocol):
    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> None
    def forward(self, batch: ForwardBatch, kv: KVCacheManager,
                attn: AttentionBackend) -> Tensor              # hidden [T, 2560]
    def compute_logits(self, hidden: Tensor, idx: Tensor) -> Tensor
```

Logs: model.loaded (params, gb, ms)

Extend later: Add models/llama.py and register it. Shared ops live in layers/ops.py (~70 lines: rms_norm, rope, swiglu).

### 9. Attention backend — `attention/sdpa.py` (~90 lines)

Libraries: torch (scaled_dot_product_attention)

- Builds per-sequence metadata: context length, query range, block table.
- For each sequence: gathers its K/V blocks, runs SDPA with causal mask and `enable_gqa=True`.
- Writes output into the flat `[T, 32, 128]` tensor.

Interface:
```
class AttentionBackend(Protocol):
    def build_metadata(self, sched: SchedulerOutput, block_size: int) -> AttentionMetadata
    def forward(self, q: Tensor, k_cache: Tensor, v_cache: Tensor,
                meta: AttentionMetadata) -> Tensor
```

Logs: debug only: attention.metadata (num_prefill, num_decode)

Extend later: FlashInfer = add attention/flashinfer.py, register, pass --attention-backend flashinfer.

### 10. Sampler — `sampler.py` (~70 lines)

Libraries: torch

- Greedy when temperature = 0, else temperature → top-k → top-p → draw.
- Seeds each draw from (seed, position), TensorFold style, so outputs are reproducible.
- Returns one token ID per sequence.

Interface:
```
class Sampler(Protocol):
    def sample(self, logits: Tensor, seqs: list[Sequence]) -> list[int]
```

Logs: debug only: sample.done (batch, ms)

Extend later: Swap in FlashInfer's sampling kernels or add penalties as another Sampler.

### 11. Detokenizer — `detokenizer.py` (~40 lines)

Libraries: transformers

- Incremental decode: turns new IDs into the new text since last send.
- Holds back incomplete UTF-8 (e.g. half a Hindi character).
- Skips special tokens.

Interface:
```
class Detokenizer(Protocol):
    def step(self, token_id: int) -> str        # text delta, may be ""
```

Logs: none per token; errors only

Extend later: Stop strings and reasoning-content parsing hook in here.

### Weight loader — `loader.py` (~60 lines)

Libraries: safetensors, huggingface_hub

- Startup only. Downloads Qwen/Qwen3-4B if missing.
- Streams (name, tensor) pairs from .safetensors shards in bf16 to the GPU.
- Hands them to `model.load_weights()`.

Interface:
```
def iter_weights(model_path: str, device="cuda") -> Iterator[tuple[str, Tensor]]
```

Logs: weights.downloading · weights.loaded (num_tensors, gb, ms)

Extend later: FP8 quantization (TRT-LLM) is a transform applied to this iterator.

### Config + entry — `config.py · main.py` (~90 lines)

Libraries: pydantic, argparse

- One `EngineConfig`: model path, kv_gb, block_size, max_num_seqs, max_model_len, attention_backend, log_level, host, port.
- main.py parses CLI, sets up logging, builds the engine, starts uvicorn.

Interface:
```
python -m spark_engine.main --model Qwen/Qwen3-4B --kv-gb 30 \
    --max-num-seqs 8 --max-model-len 8192 --port 8000 --log-level info
```

Logs: config.resolved (all values, once at startup)

Extend later: New features arrive as new config fields with defaults.

### Contracts — `types.py · interfaces.py` (~200 lines)

Libraries: dataclasses, typing

- types.py: SamplingParams, Sequence, SchedulerOutput, ForwardBatch, AttentionMetadata, TokenOutput.
- interfaces.py: the Protocols for every swappable block.
- Full code is in the Contracts section below.

Interface:
```
from spark_engine.interfaces import Scheduler, KVCacheManager, ModelRunner, \
    Model, AttentionBackend, Sampler, Tokenizer, Detokenizer
```

Logs: none

Extend later: New fields always get defaults so existing blocks keep working.

### JSON logging — `log.py` (~60 lines)

Libraries: structlog

- Configures structlog: ISO timestamps, level, JSON renderer, stdout + rotating file.
- `get_logger(component)` returns a logger pre-bound with its block name.
- contextvars carry request_id on the API side.

Interface:
```
log = get_logger("scheduler")
log.info("seq.admitted", request_id=r, seq_id=s, blocks_reserved=83)
```

Logs: (writes everyone else's)

Extend later: Add OpenTelemetry traces later by adding a processor here.

## Later extensions (not MVP)

| Feature | From | Files |
|---|---|---|
| FlashInfer attention | vLLM, SGLang | add attention/flashinfer.py |
| Prefix cache | vLLM, SGLang | add kv_cache_prefix.py |
| CUDA graphs | vLLM, TRT-LLM, TensorFold | model_runner.py |
| Chunked prefill | vLLM | scheduler.py |
| FP8 weights | TRT-LLM | loader.py, layers/ops.py |
| Speculative decoding | TensorFold | add proposer.py; uses ScheduledSeq.num_lookahead |
| Overlap scheduling | SGLang | engine.py |
| Another model | all | add models/<name>.py |
