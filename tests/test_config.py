"""EngineConfig defaults and the `turbine serve` argument parsing."""

import argparse
import json
from pathlib import Path

import pytest
import torch
from pydantic import BaseModel
from spark_engine.config import EngineConfig
from spark_engine.main import _add_field_flags, config_from_args


def test_defaults_match_decisions() -> None:
    cfg = EngineConfig(model="x")
    assert cfg.kv_gb == 30
    assert cfg.block_size == 16
    assert cfg.max_num_seqs == 8
    assert cfg.max_model_len == 8192
    assert cfg.max_queue == 64
    assert cfg.attention_backend == "sdpa"
    assert cfg.device == "cuda"
    assert cfg.dtype == "bfloat16"
    assert cfg.torch_dtype is torch.bfloat16
    assert cfg.log_level == "info"
    assert cfg.log_file == "logs/engine.jsonl"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000


def test_flags_override(tmp_path: Path) -> None:
    cfg = config_from_args(
        [
            "serve", "x", "--kv-gb", "0.01", "--device", "cpu", "--dtype", "float32",
            "--host", "0.0.0.0", "--port", "9000", "--log-file", str(tmp_path / "e.jsonl"),
        ]
    )  # fmt: skip
    assert cfg.model == "x"
    assert cfg.kv_gb == 0.01
    assert cfg.device == "cpu"
    assert cfg.dtype == "float32"
    assert cfg.torch_dtype is torch.float32
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9000


def test_missing_model_is_usage_error() -> None:
    with pytest.raises(SystemExit):
        config_from_args(["serve"])


def test_bad_dtype_rejected() -> None:
    with pytest.raises(ValueError):
        EngineConfig(model="x", dtype="int4")
    with pytest.raises(SystemExit):
        config_from_args(["serve", "x", "--dtype", "int4"])


def test_config_resolved_is_logged(tmp_path: Path) -> None:
    log_file = tmp_path / "engine.jsonl"
    config_from_args(["serve", "x", "--device", "cpu", "--log-file", str(log_file)])
    events = [json.loads(line) for line in log_file.read_text().splitlines()]
    resolved = [e for e in events if e["event"] == "config.resolved"]
    assert len(resolved) == 1
    assert resolved[0]["component"] == "config"
    assert resolved[0]["model"] == "x"
    assert resolved[0]["device"] == "cpu"
    assert resolved[0]["port"] == 8000


def test_unsupported_field_type_raises_at_build_time() -> None:
    class Bad(BaseModel):
        model: str = "x"
        verbose: bool = False

    with pytest.raises(TypeError, match="verbose"):
        _add_field_flags(argparse.ArgumentParser(), Bad)
