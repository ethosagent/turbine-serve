"""Block 1: pydantic schemas, routes, SSE streaming; maps engine errors to HTTP status codes."""

import asyncio
import json
import secrets
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from spark_engine.engine_client import EngineClient, QueueFull
from spark_engine.interfaces import Detokenizer, Tokenizer
from spark_engine.log import bind_request, clear_request, get_logger
from spark_engine.types import SamplingParams, TokenOutput

log = get_logger("api")

_DEFAULTS = SamplingParams()
_SSE_DONE = "data: [DONE]\n\n"


# ---------------------------------------------------------------- schemas (OpenAI field names)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=_DEFAULTS.temperature, ge=0)
    top_p: float = Field(default=_DEFAULTS.top_p, gt=0, le=1)
    top_k: int = Field(default=_DEFAULTS.top_k, ge=0)
    seed: int | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]


# ---------------------------------------------------------------- per-request state and helpers


@dataclass
class _Request:
    request_id: str
    model: str
    prompt_tokens: int
    detok: Detokenizer
    start: float = field(default_factory=time.perf_counter)
    created: int = field(default_factory=lambda: int(time.time()))
    role_sent: bool = False

    def text(self, out: TokenOutput) -> str:
        """Error outputs carry token_id=-1 and never reach the detokenizer."""
        return "" if out.finish_reason == "error" else self.detok.step(out.token_id)

    def chunk(self, delta: Delta, finish_reason: str | None = None) -> str:
        choice = ChunkChoice(delta=delta, finish_reason=finish_reason)
        body = ChatCompletionChunk(
            id=self.request_id, created=self.created, model=self.model, choices=[choice]
        )
        data = body.model_dump(exclude_none=True)
        data["choices"][0]["finish_reason"] = finish_reason  # OpenAI always sends the key
        return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"

    def token_event(self, out: TokenOutput) -> str | None:
        """One SSE event per non-empty text delta; the first one carries the assistant role."""
        text = self.text(out)
        if not text:
            return None
        role: Literal["assistant"] | None = None if self.role_sent else "assistant"
        self.role_sent = True
        return self.chunk(Delta(role=role, content=text))

    def completed(self, status: int) -> None:
        duration_ms = round((time.perf_counter() - self.start) * 1000, 3)
        log.info("request.completed", status=status, duration_ms=duration_ms)


def _error(status: int, message: str, error_type: str, start: float) -> JSONResponse:
    duration_ms = round((time.perf_counter() - start) * 1000, 3)
    log.info("request.completed", status=status, duration_ms=duration_ms, error=error_type)
    content = {"error": {"message": message, "type": error_type}}
    return JSONResponse(status_code=status, content=content)


def _sampling_params(
    req: ChatCompletionRequest, prompt_len: int, max_model_len: int
) -> SamplingParams:
    requested = req.max_tokens if req.max_tokens is not None else _DEFAULTS.max_tokens
    return SamplingParams(
        max_tokens=min(requested, max(1, max_model_len - prompt_len)),
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
    )


async def _stream(
    outs: AsyncGenerator[TokenOutput, None], first: TokenOutput, ctx: _Request
) -> AsyncIterator[str]:
    """SSE body: text deltas, a final chunk with finish_reason, then [DONE]."""
    bind_request(ctx.request_id)  # the body may run in a task that did not inherit the binding
    try:
        async with aclosing(outs):
            last = first
            if event := ctx.token_event(first):
                yield event
            async for last in outs:
                if event := ctx.token_event(last):
                    yield event
        yield ctx.chunk(Delta(), finish_reason=last.finish_reason)
        yield _SSE_DONE
        ctx.completed(200)
    except (asyncio.CancelledError, GeneratorExit):
        log.info("client.disconnected")
        ctx.completed(499)
        raise
    finally:
        clear_request()


async def _complete(
    outs: AsyncGenerator[TokenOutput, None], first: TokenOutput, ctx: _Request
) -> JSONResponse:
    async with aclosing(outs):
        all_outs = [first] + [out async for out in outs]
    content = "".join(ctx.text(out) for out in all_outs)
    completion_tokens = sum(1 for out in all_outs if out.finish_reason != "error")
    body = ChatCompletionResponse(
        id=ctx.request_id,
        created=ctx.created,
        model=ctx.model,
        choices=[
            Choice(
                message=ResponseMessage(content=content), finish_reason=all_outs[-1].finish_reason
            )
        ],
        usage=Usage(
            prompt_tokens=ctx.prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=ctx.prompt_tokens + completion_tokens,
        ),
    )
    ctx.completed(200)
    return JSONResponse(content=body.model_dump())


# ---------------------------------------------------------------- app


def create_app(
    client: EngineClient, tokenizer: Tokenizer, model_id: str, max_model_len: int
) -> FastAPI:
    """Build the app; the engine thread starts in the lifespan, never at import time."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        client.start()
        try:
            yield
        finally:
            await client.stop()

    app = FastAPI(title="turbine-serve", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        model = {"id": model_id, "object": "model", "owned_by": "turbine-serve"}
        return {"object": "list", "data": [model]}

    async def handle(req: ChatCompletionRequest, request_id: str, start: float) -> Response:
        messages = [m.model_dump() for m in req.messages]
        try:
            prompt_ids = tokenizer.encode_chat(messages, **req.chat_template_kwargs)
            params = _sampling_params(req, len(prompt_ids), max_model_len)
            outs = client.generate(prompt_ids, params, request_id)
            first = await anext(outs)  # QueueFull / impossible-KV ValueError surface here
        except QueueFull as exc:
            return _error(503, str(exc), "server_overloaded", start)
        except ValueError as exc:  # includes PromptTooLong
            return _error(400, str(exc), "invalid_request_error", start)
        ctx = _Request(request_id, model_id, len(prompt_ids), tokenizer.new_detokenizer(), start)
        if req.stream:
            return StreamingResponse(_stream(outs, first, ctx), media_type="text/event-stream")
        return await _complete(outs, first, ctx)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        request_id = "req_" + secrets.token_hex(4)
        bind_request(request_id)
        start = time.perf_counter()
        log.info("request.received", stream=req.stream, max_tokens=req.max_tokens)
        try:
            return await handle(req, request_id, start)
        finally:
            clear_request()

    return app
