"""Attention backend registry: --attention-backend name -> class."""

from turbine_serve.attention.sdpa import SDPAAttention
from turbine_serve.interfaces import AttentionBackend

ATTENTION_BACKENDS: dict[str, type[AttentionBackend]] = {"sdpa": SDPAAttention}


def get_attention_backend(name: str) -> type[AttentionBackend]:
    try:
        return ATTENTION_BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown attention backend {name!r}; known: {sorted(ATTENTION_BACKENDS)}"
        ) from None
