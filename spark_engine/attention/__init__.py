"""Attention backend registry: --attention-backend name -> class."""

from spark_engine.attention.sdpa import SDPAAttention
from spark_engine.interfaces import AttentionBackend

ATTENTION_BACKENDS: dict[str, type[AttentionBackend]] = {"sdpa": SDPAAttention}


def get_attention_backend(name: str) -> type[AttentionBackend]:
    try:
        return ATTENTION_BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown attention backend {name!r}; known: {sorted(ATTENTION_BACKENDS)}"
        ) from None
