"""rms_norm, rope, silu_and_mul: fp32 math, cast back to bf16."""

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF order: fp32 variance and rsqrt, cast to x.dtype, then multiply by the weight."""
    x32 = x.float()  # [..., C]
    var = x32.pow(2).mean(-1, keepdim=True)  # [..., 1]
    normed = x32 * torch.rsqrt(var + eps)  # [..., C]
    return weight * normed.to(x.dtype)  # [..., C]


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 cos and sin tables for flat positions; the frequencies are duplicated across halves."""
    exponent = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim  # [D/2]
    inv_freq = 1.0 / (theta**exponent)  # [D/2]
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
    return emb.cos(), emb.sin()  # [T, D] each


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)  # [..., D]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on x [T, H, D]; cos and sin [T, D] broadcast over heads."""
    x32 = x.float()  # [T, H, D]
    out = x32 * cos[:, None, :] + _rotate_half(x32) * sin[:, None, :]  # [T, H, D]
    return out.to(x.dtype)  # [T, H, D]


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation in fp32, cast back to the gate's dtype."""
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)  # [..., I]
