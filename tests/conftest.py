"""Shared pytest setup: markers that skip when the hardware or server is absent."""

import os

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_cuda = torch.cuda.is_available()
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    skip_e2e = pytest.mark.skip(reason="set TURBINE_BASE_URL to a running server")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "e2e" in item.keywords and not os.environ.get("TURBINE_BASE_URL"):
            item.add_marker(skip_e2e)
