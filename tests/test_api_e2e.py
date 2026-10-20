"""End to end over HTTP with the openai SDK against a running server (TURBINE_BASE_URL).

On the Mac the server is `scripts/serve-tiny.sh` (random weights), so these tests assert
shapes, not content; Plan B runs them against the real model on the Spark.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import openai
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

pytestmark = pytest.mark.e2e

MESSAGES: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Say hello in one word."}]


@pytest.fixture(scope="module")
def client() -> OpenAI:
    return OpenAI(base_url=os.environ["TURBINE_BASE_URL"], api_key="none", max_retries=0)


@pytest.fixture(scope="module")
def model_id(client: OpenAI) -> str:
    return client.models.list().data[0].id


def test_models_lists_model(client: OpenAI) -> None:
    models = client.models.list().data
    assert len(models) == 1 and models[0].id


def test_stream_reply(client: OpenAI, model_id: str) -> None:
    stream = client.chat.completions.create(
        model=model_id, messages=MESSAGES, stream=True, max_tokens=16, temperature=0
    )
    chunks = list(stream)
    assert chunks, "no chunks streamed"
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text, "streamed no text"
    assert all(c.choices[0].finish_reason is None for c in chunks[:-1])
    assert chunks[-1].choices[0].finish_reason in {"stop", "length"}


def test_non_stream_usage_and_length(client: OpenAI, model_id: str) -> None:
    resp = client.chat.completions.create(
        model=model_id, messages=MESSAGES, max_tokens=5, temperature=0
    )
    assert resp.choices[0].finish_reason == "length"
    assert resp.usage is not None
    assert resp.usage.completion_tokens == 5
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + 5


def test_thinking_inline(client: OpenAI, model_id: str) -> None:
    # 200 only on the tiny model; Plan B also asserts "</think>" in content on the real model
    raw = client.chat.completions.with_raw_response.create(
        model=model_id,
        messages=MESSAGES,
        max_tokens=8,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    assert raw.status_code == 200
    assert raw.parse().choices[0].finish_reason in {"stop", "length"}


def test_prompt_too_long_400(client: OpenAI, model_id: str) -> None:
    huge = "hello " * 50_000
    with pytest.raises(openai.BadRequestError) as exc:
        client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": huge}], max_tokens=1
        )
    assert exc.value.status_code == 400


def test_eight_parallel_requests(client: OpenAI, model_id: str) -> None:
    def one(i: int) -> tuple[str | None, int]:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": f"Count to {i}."}],
            max_tokens=8,
            temperature=0,
        )
        assert resp.usage is not None
        return resp.choices[0].finish_reason, resp.usage.completion_tokens

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(8)))
    assert len(results) == 8
    assert all(reason in {"stop", "length"} and 1 <= n <= 8 for reason, n in results)
