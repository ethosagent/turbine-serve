"""Contract tests for spark_engine.types."""

import dataclasses

import pytest
from spark_engine.types import SamplingParams, SeqStatus, Sequence, TokenOutput


def test_sampling_defaults() -> None:
    p = SamplingParams()
    assert (p.max_tokens, p.temperature, p.top_p, p.top_k, p.seed) == (512, 0.7, 0.8, 20, None)


def test_sequence_defaults_and_num_tokens() -> None:
    seq = Sequence(seq_id="s1", request_id="r1", prompt_ids=[1, 2, 3], params=SamplingParams())
    assert seq.status is SeqStatus.WAITING
    assert seq.num_computed == 0
    assert seq.block_table == []
    assert seq.output_ids == []
    assert seq.finish_reason is None
    assert seq.num_tokens == 3
    seq.output_ids.extend([7, 8])
    assert seq.num_tokens == 5


def test_sequence_default_lists_are_not_shared() -> None:
    a = Sequence("a", "r", [1], SamplingParams())
    b = Sequence("b", "r", [1], SamplingParams())
    a.block_table.append(4)
    assert b.block_table == []


def test_token_output_is_frozen() -> None:
    out = TokenOutput(request_id="r", seq_id="s", token_id=1, finished=False)
    assert out.finish_reason is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.token_id = 2  # type: ignore[misc]
