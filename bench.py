"""Baseline throughput/latency benchmark against any OpenAI-compatible server (streaming).

    python bench.py --base-url http://127.0.0.1:8000/v1 --concurrency 1 8 --max-tokens 128

For each concurrency C, C workers each send --rounds requests back to back (C * rounds total).
Tokens are counted as streamed content chunks (the server emits one chunk per decoded token).
total tok/s = all tokens / wall time; per-stream tok/s = mean of tokens / request duration;
TTFT = time from request start to the first content chunk. Needs only the openai package.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

PROMPT = "Count from 1 to 100, separated by commas."


@dataclass
class Result:
    tokens: int
    ttft_s: float
    duration_s: float


async def one_request(client: AsyncOpenAI, model: str, max_tokens: int) -> Result:
    start = time.perf_counter()
    first: float | None = None
    tokens = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.perf_counter()
            tokens += 1
    end = time.perf_counter()
    return Result(tokens, (first or end) - start, end - start)


async def run_level(
    client: AsyncOpenAI, model: str, concurrency: int, rounds: int, max_tokens: int
) -> tuple[list[Result], float]:
    async def worker() -> list[Result]:
        return [await one_request(client, model, max_tokens) for _ in range(rounds)]

    start = time.perf_counter()
    per_worker = await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - start
    return [r for rs in per_worker for r in rs], wall


async def main(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="unused", timeout=600)
    model = args.model or (await client.models.list()).data[0].id
    await one_request(client, model, 8)  # warm-up, not measured
    print(f"model {model}  max_tokens {args.max_tokens}  rounds {args.rounds}")
    print("| concurrency | requests | total tok/s | per-stream tok/s | mean TTFT ms |")
    print("|---:|---:|---:|---:|---:|")
    for c in args.concurrency:
        results, wall = await run_level(client, model, c, args.rounds, args.max_tokens)
        total = sum(r.tokens for r in results) / wall
        per_stream = statistics.mean(r.tokens / r.duration_s for r in results)
        ttft_ms = statistics.mean(r.ttft_s for r in results) * 1000
        print(f"| {c} | {len(results)} | {total:.1f} | {per_stream:.1f} | {ttft_ms:.0f} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None, help="default: first id from /v1/models")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    p.add_argument("--rounds", type=int, default=3, help="requests per worker")
    p.add_argument("--max-tokens", type=int, default=128)
    asyncio.run(main(p.parse_args()))
