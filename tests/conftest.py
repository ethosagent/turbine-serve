"""Shared pytest setup: markers that skip when the hardware or server is absent."""

import os
from pathlib import Path

import pytest
import torch
from spark_engine.testing.tiny_model import make_tiny_qwen3


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_cuda = torch.cuda.is_available()
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
    return make_tiny_qwen3(tmp_path_factory.mktemp("tiny-qwen3"))
