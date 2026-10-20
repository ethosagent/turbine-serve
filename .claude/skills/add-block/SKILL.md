---
name: add-block
description: >-
  Recipe for adding a swappable block to turbine-serve: a new attention backend (FlashInfer), a
  new model family, a prefix-caching KV manager, another sampler, or a new scheduling policy.
  Use when asked to "add a backend", "support another model", "plug in X", or extend a Protocol.
---

# add-block

Rules that must never be missed:

- **Extension = new file + registry entry + config flag.** Nothing above the seam changes. If it
  does, the seam is wrong; stop and propose an `ARCHITECTURE.md` amendment.
- **The new file implements an existing Protocol from `interfaces.py` verbatim.** Need a new
  method? Add it to the Protocol with every existing implementation updated in the same commit.
- **The new block may import only the kernel** (`types`, `interfaces`, `log`, `config`) and, for
  models, `layers/`. `make deps` tells you if you crossed a line.
- **Parity stays green.** A new attention backend or model must pass `tests/test_parity.py`
  when selected; a new KV manager must pass `tests/test_engine_gpu.py`.

## Procedure

1. **Pick the seam** and the registry:

   | Block | Protocol | Registry | Config flag |
   |---|---|---|---|
   | attention backend | `AttentionBackend` | `attention/__init__.py` | `--attention-backend <name>` |
   | model family | `Model` | `models/__init__.py` keyed by HF `model_type` | automatic via `AutoConfig` |
   | KV manager | `KVCacheManager` | wired in `engine.build_engine` | `--kv-manager <name>` (add) |
   | sampler | `Sampler` | wired in `engine.build_engine` | `--sampler <name>` (add) |
   | scheduling policy | `Scheduler` | wired in `engine.build_engine` | `--scheduler <name>` (add) |

2. **Write the file** `turbine_serve/<package>/<name>.py` (or `turbine_serve/<name>.py` for
   top-level blocks). `get_logger("<name>")`. Shape comments on every tensor line.
3. **Register** by name. Unknown names raise `ValueError` listing the known ones.
4. **Config.** Add the field to `EngineConfig` with the old behaviour as default; `main.py`
   derives the flag.
5. **Tests.** A unit test file `tests/test_<name>.py` (mark `gpu` if it needs CUDA) plus the
   existing parity/engine test parametrised over the new name.
6. **Docs.** One row in `ARCHITECTURE.md` §III if it is a new module; a line in
   `docs/orientation.md` only if the request flow changed; README flag if user-facing.
7. `make check`, then on the Spark `scripts/spark.sh pytest -q`.

## Metadata subclasses

An attention backend that needs more than the base `AttentionMetadata` subclasses it in its own
file (`SDPAMetadata` is the model). The model runner only ever calls `attn.build_metadata` and
passes the result back to `attn.forward`; it never reads backend fields.
