"""Layer ops tests: RMSNorm, RoPE and SwiGLU against naive fp32 formulas, relative-position
property, and dtype round trip."""

import pytest
import torch
from spark_engine.layers.ops import apply_rope, rms_norm, rope_cos_sin, silu_and_mul

T, H, D = 6, 3, 8
EPS = 1e-6
THETA = 1_000_000.0


def _tol(dtype: torch.dtype) -> float:
    return 1e-6 if dtype == torch.float32 else 2e-2


def _rand(*shape: int, dtype: torch.dtype = torch.float32, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(dtype)


def _ref_cos_sin(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = torch.tensor([THETA ** (-2.0 * i / D) for i in range(D // 2)])  # [D/2]
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
    return emb.cos(), emb.sin()


def _ref_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)  # [..., D]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_reference(dtype: torch.dtype) -> None:
    x = _rand(T, 16, dtype=dtype)  # [T, 16]
    weight = _rand(16, dtype=dtype, seed=1)  # [16]
    x32 = x.float()  # [T, 16]
    ref = weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)).to(dtype)  # [T, 16]
    out = rms_norm(x, weight, EPS)  # [T, 16]
    assert out.dtype == dtype
    assert torch.allclose(out.float(), ref.float(), atol=_tol(dtype))


def test_rope_cos_sin_matches_reference() -> None:
    positions = torch.tensor([0, 1, 7, 33, 64, 100])  # [T]
    cos, sin = rope_cos_sin(positions, D, THETA)  # [T, D] each
    ref_cos, ref_sin = _ref_cos_sin(positions)  # [T, D] each
    assert cos.dtype == torch.float32 and cos.shape == (T, D)
    assert torch.allclose(cos, ref_cos, atol=1e-6)
    assert torch.allclose(sin, ref_sin, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_position_zero_is_identity(dtype: torch.dtype) -> None:
    x = _rand(T, H, D, dtype=dtype)  # [T, H, D]
    cos, sin = rope_cos_sin(torch.zeros(T, dtype=torch.long), D, THETA)  # [T, D] each
    assert torch.equal(apply_rope(x, cos, sin), x)


def test_rope_is_relative() -> None:
    q = _rand(1, 1, D, seed=2)  # [1, 1, D]
    k = _rand(1, 1, D, seed=3)  # [1, 1, D]

    def dot(q_pos: int, k_pos: int) -> torch.Tensor:
        cq, sq = rope_cos_sin(torch.tensor([q_pos]), D, THETA)  # [1, D] each
        ck, sk = rope_cos_sin(torch.tensor([k_pos]), D, THETA)  # [1, D] each
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum()  # []

    assert torch.allclose(dot(5, 3), dot(12, 10), atol=1e-5)
    assert not torch.allclose(dot(5, 3), dot(5, 4), atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_rope_matches_reference(dtype: torch.dtype) -> None:
    x = _rand(T, H, D, dtype=dtype)  # [T, H, D]
    positions = torch.tensor([3, 4, 5, 6, 7, 8])  # [T]
    cos, sin = _ref_cos_sin(positions)  # [T, D] each
    x32 = x.float()  # [T, H, D]
    ref = x32 * cos[:, None, :] + _ref_rotate_half(x32) * sin[:, None, :]  # [T, H, D]
    out = apply_rope(x, cos, sin)  # [T, H, D]
    assert out.dtype == dtype
    assert torch.allclose(out.float(), ref.to(dtype).float(), atol=_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_silu_and_mul_matches_reference(dtype: torch.dtype) -> None:
    gate = _rand(T, 16, dtype=dtype)  # [T, 16]
    up = _rand(T, 16, dtype=dtype, seed=1)  # [T, 16]
    g32, u32 = gate.float(), up.float()  # [T, 16] each
    ref = (g32 * torch.sigmoid(g32) * u32).to(dtype)  # [T, 16]
    out = silu_and_mul(gate, up)  # [T, 16]
    assert out.dtype == dtype
    assert torch.allclose(out.float(), ref.float(), atol=_tol(dtype))


def test_bf16_inputs_return_bf16() -> None:
    x = _rand(T, H, D, dtype=torch.bfloat16)  # [T, H, D]
    cos, sin = rope_cos_sin(torch.arange(T), D, THETA)  # [T, D] each
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32
    assert rms_norm(x, _rand(D, dtype=torch.bfloat16), EPS).dtype == torch.bfloat16
    assert apply_rope(x, cos, sin).dtype == torch.bfloat16
    assert silu_and_mul(x, x).dtype == torch.bfloat16
