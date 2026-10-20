"""OpenAI-compatible API over the real EngineClient + FakeRunner engine (and once over the tiny model)."""

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from spark_engine.api.server import _Request, _stream, create_app
from spark_engine.config import EngineConfig
from spark_engine.engine import build_engine
from spark_engine.engine_client import EngineClient
from spark_engine.interfaces import Detokenizer, EngineLike, Tokenizer
from spark_engine.log import configure_logging
from spark_engine.tokenizer import PromptTooLong
from spark_engine.types import SamplingParams

from tests.fakes import FakeRunner, make_engine

MAX_LEN = 64
GREEDY = {"temperature": 0}


class FakeDetokenizer:
    def step(self, token_id: int) -> str:
        return "" if token_id == 7 else f"<{token_id}>"


class FakeTokenizer:
    """[1, 2] for any chat; content "long:N" encodes to N ones (PromptTooLong past max_model_len)."""

    eos_token_ids: tuple[int, ...] = (7,)

    def __init__(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.template_kwargs: list[dict[str, Any]] = []  # one entry per encode_chat call

    def encode_chat(self, messages: list[dict[str, Any]], **template_kwargs: Any) -> list[int]:
        self.template_kwargs.append(template_kwargs)
        content = str(messages[-1]["content"])
        if not content.startswith("long:"):
            return [1, 2]
        n = int(content.removeprefix("long:"))
        if n > self.max_model_len:
            raise PromptTooLong(n, self.max_model_len)
        return [1] * n

    def new_detokenizer(self) -> Detokenizer:
        return FakeDetokenizer()


@asynccontextmanager
async def serve(
    engine: EngineLike | None = None,
    tokenizer: Tokenizer | None = None,
    max_model_len: int = MAX_LEN,
    max_queue: int = 4,
) -> AsyncIterator[httpx.AsyncClient]:
    client = EngineClient(engine or make_engine()[0], max_queue)
    app = create_app(client, tokenizer or FakeTokenizer(max_model_len), "fake-model", max_model_len)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        yield http


@pytest.fixture
async def api() -> AsyncIterator[httpx.AsyncClient]:
    async with serve() as http:
        yield http


def chat(content: str = "hi", **extra: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": content}], **extra}


def sse_events(body: str) -> list[str]:
    assert body.endswith("\n\n")
    events = body.split("\n\n")[:-1]
    assert all(e.startswith("data: ") for e in events)
    return [e.removeprefix("data: ") for e in events]


async def test_health_and_models(api: httpx.AsyncClient) -> None:
    health = await api.get("/health")
    assert health.status_code == 200 and health.json() == {"status": "ok"}
    models = await api.get("/v1/models")
    assert models.status_code == 200
    assert models.json() == {
        "object": "list",
        "data": [{"id": "fake-model", "object": "model", "owned_by": "turbine-serve"}],
    }


async def test_non_stream_response(api: httpx.AsyncClient) -> None:
    resp = await api.post("/v1/chat/completions", json=chat(**GREEDY))
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion" and body["model"] == "fake-model"
    assert re.fullmatch(r"req_[0-9a-f]{8}", body["id"])
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "<3><4><5><6>"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}


async def test_stream_sse_format(api: httpx.AsyncClient) -> None:
    resp = await api.post("/v1/chat/completions", json=chat(stream=True, **GREEDY))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert deltas[0]["role"] == "assistant"
    assert [d.get("content") for d in deltas[:-1]] == ["<3>", "<4>", "<5>", "<6>"]
    assert deltas[-1] == {} and chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all("finish_reason" in c["choices"][0] for c in chunks)  # OpenAI always sends it
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    assert all(None not in d.values() for d in deltas)  # delta stays free of null fields


async def test_max_tokens_length_finish(api: httpx.AsyncClient) -> None:
    resp = await api.post("/v1/chat/completions", json=chat(max_tokens=2, **GREEDY))
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "<3><4>"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 2


async def test_prompt_too_long_is_400(api: httpx.AsyncClient) -> None:
    resp = await api.post("/v1/chat/completions", json=chat(f"long:{MAX_LEN + 1}"))
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "invalid_request_error" and "max_model_len" in error["message"]


async def test_max_tokens_is_clamped(api: httpx.AsyncClient) -> None:
    resp = await api.post(
        "/v1/chat/completions", json=chat(f"long:{MAX_LEN - 3}", max_tokens=1000, **GREEDY)
    )
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "<2><3><4>"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 3


async def test_impossible_kv_request_is_400() -> None:
    engine, _ = make_engine(num_blocks=4)  # 64 KV slots, far below max_model_len
    async with serve(engine, max_model_len=4096) as http:
        resp = await http.post("/v1/chat/completions", json=chat("long:200", stream=True))
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


async def test_queue_full_is_503() -> None:
    async with serve(max_queue=0) as http:
        resp = await http.post("/v1/chat/completions", json=chat(stream=True))
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "server_overloaded"


async def test_empty_messages_is_422(api: httpx.AsyncClient) -> None:
    assert (await api.post("/v1/chat/completions", json={"messages": []})).status_code == 422
    for bad in ({"temperature": -0.1}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": -1}):
        assert (await api.post("/v1/chat/completions", json=chat(**bad))).status_code == 422
    assert (await api.post("/v1/chat/completions", json=chat(max_tokens=0))).status_code == 422


async def test_temperature_zero_is_deterministic(api: httpx.AsyncClient) -> None:
    bodies = [
        (await api.post("/v1/chat/completions", json=chat(**GREEDY))).json() for _ in range(3)
    ]
    contents = {b["choices"][0]["message"]["content"] for b in bodies}
    assert contents == {"<3><4><5><6>"}


async def test_request_id_in_logs(api: httpx.AsyncClient, tmp_path: Path) -> None:
    log_file = tmp_path / "api.jsonl"
    configure_logging("info", str(log_file))
    try:
        resp = await api.post("/v1/chat/completions", json=chat(max_tokens=3, **GREEDY))
    finally:
        configure_logging("info", None)
    request_id = resp.json()["id"]
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    api_lines = {line["event"]: line for line in lines if line["component"] == "api"}
    received, completed = api_lines["request.received"], api_lines["request.completed"]
    assert received["request_id"] == request_id and received["stream"] is False
    assert received["max_tokens"] == 3
    assert completed["request_id"] == request_id and completed["status"] == 200
    assert completed["duration_ms"] >= 0
    submitted = next(line for line in lines if line["event"] == "seq.submitted")
    assert submitted["request_id"] == request_id


async def test_chat_template_kwargs_are_not_splatted() -> None:
    tokenizer = FakeTokenizer(MAX_LEN)
    async with serve(tokenizer=tokenizer) as http:
        bad = chat(max_tokens=2, chat_template_kwargs={"messages": 1})
        resp = await http.post("/v1/chat/completions", json=bad)
        assert resp.status_code == 200
        thinking = chat(max_tokens=2, chat_template_kwargs={"enable_thinking": True})
        resp = await http.post("/v1/chat/completions", json=thinking)
        assert resp.status_code == 200
    assert tokenizer.template_kwargs == [{"enable_thinking": False}, {"enable_thinking": True}]


@pytest.mark.parametrize("stream", [False, True])
async def test_engine_error_before_output_is_500(stream: bool, tmp_path: Path) -> None:
    log_file = tmp_path / "api.jsonl"
    engine = make_engine(runner=FakeRunner(fail_on_step=1))[0]
    configure_logging("info", str(log_file))
    try:
        async with serve(engine=engine) as http:
            resp = await http.post("/v1/chat/completions", json=chat(stream=stream, **GREEDY))
    finally:
        configure_logging("info", None)
    assert resp.status_code == 500
    assert resp.json() == {"error": {"message": "engine error", "type": "server_error"}}
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    completed = [line for line in lines if line["event"] == "request.completed"]
    assert len(completed) == 1 and completed[0]["status"] == 500


async def test_stream_disconnect_logs_499(tmp_path: Path) -> None:
    log_file = tmp_path / "api.jsonl"
    client = EngineClient(make_engine()[0], 4)
    client.start()
    configure_logging("info", str(log_file))
    try:
        outs = client.generate([1, 2], SamplingParams(temperature=0), "req_gone")
        first = await anext(outs)
        ctx = _Request("req_gone", "fake-model", 2, FakeDetokenizer())
        body = _stream(outs, first, ctx)
        assert (await anext(body)).startswith("data: ")
        await body.aclose()  # what Starlette does when the client goes away mid-stream
    finally:
        configure_logging("info", None)
        await client.stop()
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    api_events = {line["event"]: line for line in lines if line["component"] == "api"}
    assert api_events["client.disconnected"]["request_id"] == "req_gone"
    completed = api_events["request.completed"]
    assert completed["request_id"] == "req_gone" and completed["status"] == 499
    assert completed["duration_ms"] >= 0


async def test_tiny_model_streams_text(tiny_model_dir: Path) -> None:
    config = EngineConfig(model=str(tiny_model_dir), device="cpu", dtype="float32", kv_gb=0.01)
    engine, tokenizer = build_engine(config)
    async with serve(engine, tokenizer, max_model_len=config.max_model_len) as http:
        resp = await http.post(
            "/v1/chat/completions", json=chat("Say hello.", stream=True, max_tokens=8, **GREEDY)
        )
    assert resp.status_code == 200
    events = sse_events(resp.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.strip()
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}
