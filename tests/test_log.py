"""JSON logging: one object per line, level filtering, request context."""

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
import structlog
from spark_engine.log import bind_request, clear_request, configure_logging, get_logger


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    clear_request()
    yield
    clear_request()
    structlog.reset_defaults()


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_log_line_is_json(tmp_path: Path) -> None:
    path = tmp_path / "engine.jsonl"
    configure_logging("info", str(path))
    get_logger("scheduler").info("seq.admitted", request_id="r1", seq_id="s1", blocks_reserved=83)
    line = _lines(path)[-1]
    assert line["component"] == "scheduler" and line["event"] == "seq.admitted"
    assert line["level"] == "info" and line["request_id"] == "r1" and line["blocks_reserved"] == 83
    datetime.fromisoformat(str(line["ts"]))


def test_debug_filtered_at_info(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "engine.jsonl"
    configure_logging("info", str(path))
    log = get_logger("scheduler")
    log.debug("step.planned", n=1)
    log.info("seq.finished")
    assert [line["event"] for line in _lines(path)] == ["seq.finished"]


def test_debug_written_at_debug(tmp_path: Path) -> None:
    path = tmp_path / "engine.jsonl"
    configure_logging("debug", str(path))
    get_logger("scheduler").debug("step.planned", n=1)
    assert _lines(path)[-1]["level"] == "debug"


def test_request_context_is_merged(tmp_path: Path) -> None:
    path = tmp_path / "engine.jsonl"
    configure_logging("info", str(path))
    bind_request("r9")
    get_logger("engine").info("seq.started")
    clear_request()
    get_logger("engine").info("seq.after")
    first, second = _lines(path)
    assert first["request_id"] == "r9"
    assert "request_id" not in second
