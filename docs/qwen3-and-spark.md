# Qwen3-4B and DGX Spark — the numbers

From `Qwen/Qwen3-4B/config.json` and the Spark's hardware. Read `AutoConfig` in code; these are
for reasoning and tests.

## Qwen3-4B

| | |
|---|---|
| layers | 36 |
| hidden | 2560 |
| MLP intermediate | 9728 |
| Q heads / KV heads / head_dim | 32 / 8 / 128 (GQA group 4: query head h uses KV head h // 4) |
| q_proj / k_proj / v_proj / o_proj | 2560→4096 / 2560→1024 / 2560→1024 / 4096→2560, no biases |
| vocab | 151936 |
| rope_theta | 1e6, rotate-half convention |
| rms_norm_eps | 1e-6 |
| tie_word_embeddings | true (`lm_head.weight` may be absent from the shards) |
| EOS | 151645 `<|im_end|>`, 151643 `<|endoftext|>` |
| weights | ≈ 8.0 GB bf16, 398 tensors (399 if lm_head is stored) |
| KV per token | 36 × 2 × 8 × 128 × 2 B = 144 KiB; 2.25 MiB per 16-token block |
| QK-Norm | per-head RMSNorm on q and k before RoPE (`self_attn.q_norm`, `k_norm`, shape [128]) |

Checkpoint tensor names: `model.embed_tokens.weight`, `model.layers.{i}.input_layernorm.weight`,
`model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `model.layers.{i}.self_attn.{q,k}_norm.weight`,
`model.layers.{i}.post_attention_layernorm.weight`, `model.layers.{i}.mlp.{gate,up,down}_proj.weight`,
`model.norm.weight`.

## DGX Spark (host `dgx-spark`, 192.168.1.49)

| | |
|---|---|
| GPU | GB10, compute capability 12.1 (sm_121), arm64 host |
| memory | 128 GB unified (CPU + GPU share it); `free -g` is the truth, not `nvidia-smi` |
| bandwidth | ≈ 273 GB/s → single-stream ceiling ≈ 273 / 8 ≈ 34 tok/s for bf16 Qwen3-4B |
| container | `nvcr.io/nvidia/pytorch:26.07-py3`: torch 2.13 NVIDIA build, CUDA 13.3 |
| KV budget | `--kv-gb 30` → 13653 blocks ≈ 218k tokens |
| other tenants | `qwen38-flash` (vLLM, ~98 GB) must be stopped before our tests; `rudder-embed`, `parakeet-asr`, `litellm` are small and stay |

Expect the MVP well below the ceiling: Python overhead and ~500 kernel launches per token
dominate. Measure with `bench.py` before any speed-up.
