"""Shared pytest setup: hardware/server skips and the tiny/real model cases.

`model_case` is parametrized over "tiny" (2-layer random Qwen3, cpu fp32, runs everywhere) and
"real" (Qwen/Qwen3-4B, cuda fp32; needs TURBINE_REAL_MODEL=1 and a CUDA device, so it skips on
the Mac). The real case is fp32 because M2 is gated on fp32-on-cuda parity (ARCHITECTURE.md §V); bf16 is
reported by tests/test_bf16_report.py. Engine fixtures are module-scoped; the real ones free GPU
memory before they build.

torch and turbine_serve are imported inside the hooks/fixtures that need them, so the e2e tests
load with only pytest + openai installed (the Spark host venv has no torch).
"""
# ruff: noqa: PLC0415  (function-level imports are the point: see the docstring)

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from turbine_serve.engine import Engine
    from turbine_serve.tokenizer import HFTokenizer

    from tests.model_cases import ModelCase


def _has_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_cuda = any("gpu" in item.keywords for item in items) and _has_cuda()
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    skip_e2e = pytest.mark.skip(reason="set TURBINE_BASE_URL to a running server")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "e2e" in item.keywords and not os.environ.get("TURBINE_BASE_URL"):
            item.add_marker(skip_e2e)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-layer random Qwen3 checkpoint with the real tokenizer, built once per session."""
    from turbine_serve.testing.tiny_model import make_tiny_qwen3

    return make_tiny_qwen3(tmp_path_factory.mktemp("tiny-qwen3"))


@pytest.fixture(scope="session")
def real_model_dir() -> str:
    """The real Qwen3-4B snapshot; skips unless TURBINE_REAL_MODEL=1 on a CUDA machine."""
    from turbine_serve.loader import resolve_model_path

    from tests.model_cases import REAL_MODEL

    if os.environ.get("TURBINE_REAL_MODEL") != "1":
        pytest.skip("set TURBINE_REAL_MODEL=1 to run the real Qwen3-4B cases")
    if not _has_cuda():
        pytest.skip("the real Qwen3-4B cases need a CUDA device")
    return resolve_model_path(REAL_MODEL)


@pytest.fixture(scope="module")
def tiny_engine(tiny_model_dir: Path) -> tuple[Engine, HFTokenizer]:
    from turbine_serve.config import EngineConfig
    from turbine_serve.engine import build_engine

    config = EngineConfig(model=str(tiny_model_dir), device="cpu", dtype="float32", kv_gb=0.01)
    return build_engine(config)


@pytest.fixture(scope="module", params=["tiny", pytest.param("real", marks=pytest.mark.gpu)])
def model_case(request: pytest.FixtureRequest) -> ModelCase:
    from tests.model_cases import ModelCase

    if request.param == "tiny":
        tiny_dir = request.getfixturevalue("tiny_model_dir")
        return ModelCase("tiny", str(tiny_dir), "cpu", "float32", 0.01, 16)
    real_dir = request.getfixturevalue("real_model_dir")
    # fp32: the gated parity case (ARCHITECTURE.md §V). ~16 GB weights + kv 4 GB; HF fp32 adds ~16 GB.
    return ModelCase("real", real_dir, "cuda", "float32", 4, 64)


@pytest.fixture(scope="module")
def model_engine(
    model_case: ModelCase, request: pytest.FixtureRequest
) -> Iterator[tuple[Engine, HFTokenizer]]:
    """The engine for the current case: the shared tiny engine, or a fresh real fp32 engine."""
    from turbine_serve.engine import build_engine

    from tests.model_cases import free_gpu_memory

    if model_case.name == "tiny":
        yield request.getfixturevalue("tiny_engine")
        return
    free_gpu_memory()  # the previous module's real engine/HF model are finalized by now
    engine = build_engine(model_case.config)
    yield engine
    del engine
    free_gpu_memory()
