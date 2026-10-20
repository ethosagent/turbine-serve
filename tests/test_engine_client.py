"""EngineClient: async streaming over a FakeRunner engine on a real engine thread."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from spark_engine.engine_client import EngineClient, QueueFull
from spark_engine.types import SamplingParams, TokenOutput

from tests.fakes import make_engine

GREEDY = SamplingParams(max_tokens=4, temperature=0.0)


@pytest.fixture
async def client_and_kv() -> AsyncIterator[tuple[EngineClient, object]]:
    engine, kv = make_engine()
    client = EngineClient(engine, max_queue=2)
    client.start()
    yield client, kv
    await client.stop()


async def _collect(client: EngineClient, prompt: list[int], rid: str) -> list[TokenOutput]:
    return [out async for out in client.generate(prompt, GREEDY, rid)]


async def test_streams_tokens_until_finished(client_and_kv: tuple[EngineClient, object]) -> None:
    client, _ = client_and_kv
    outs = await _collect(client, [1, 2], "r1")
    assert [o.token_id for o in outs] == [3, 4, 5, 6]
    assert outs[-1].finished and outs[-1].finish_reason == "length"
    assert all(o.request_id == "r1" for o in outs)


async def test_two_concurrent_requests_do_not_mix(
    client_and_kv: tuple[EngineClient, object],
) -> None:
    client, _ = client_and_kv
    a, b = await asyncio.gather(_collect(client, [1], "a"), _collect(client, [9], "b"))
    assert {o.request_id for o in a} == {"a"} and {o.request_id for o in b} == {"b"}
    assert [o.token_id for o in a] == [2, 3, 4, 5]
    assert [o.token_id for o in b] == [10, 11, 12, 13]


async def test_queue_full_raises() -> None:
    engine, _ = make_engine()
    client = EngineClient(engine, max_queue=0)  # num_waiting (0) >= 0: always full
    client.start()
    try:
        with pytest.raises(QueueFull):
            await _collect(client, [1], "r")
    finally:
        await client.stop()


async def test_cancel_frees_kv() -> None:
    engine, kv = make_engine(num_blocks=4096, eos=())  # no EOS: only the cancel can end it
    client = EngineClient(engine, max_queue=2)
    client.start()
    params = SamplingParams(max_tokens=50000, temperature=0.0)
    got = asyncio.Event()

    async def consume() -> None:
        async for _ in client.generate([1, 2], params, "r"):
            got.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(got.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        for _ in range(100):
            if kv.num_free_blocks == kv.num_blocks:
                return
            await asyncio.sleep(0.02)
        pytest.fail("KV blocks not freed after cancel")
    finally:
        await client.stop()


async def test_impossible_request_raises_value_error(
    client_and_kv: tuple[EngineClient, object],
) -> None:
    client, _ = client_and_kv
    with pytest.raises(ValueError):
        await _collect(client, list(range(5000)), "big")


async def test_failed_add_request_logs_no_abort(
    client_and_kv: tuple[EngineClient, object], capsys: pytest.CaptureFixture[str]
) -> None:
    client, _ = client_and_kv
    with pytest.raises(ValueError):
        await _collect(client, list(range(5000)), "big")
    assert "seq.aborted" not in capsys.readouterr().out
    assert client._queues == {}
