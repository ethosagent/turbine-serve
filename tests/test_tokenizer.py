"""Tokenizer and incremental detokenizer against the real Qwen3 tokenizer (cached after first run)."""

import pytest
from turbine_serve.detokenizer import IncrementalDetokenizer
from turbine_serve.tokenizer import HFTokenizer, PromptTooLong

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(scope="module")
def tok() -> HFTokenizer:
    return HFTokenizer("Qwen/Qwen3-4B", 8192)


def _stream(tok: HFTokenizer, ids: list[int]) -> str:
    detok = tok.new_detokenizer()
    return "".join(detok.step(i) for i in ids)


def test_eos_ids(tok: HFTokenizer) -> None:
    assert tok.eos_token_ids == (151645, 151643)


def test_encode_chat_thinking_off_appends_empty_think(tok: HFTokenizer) -> None:
    text = tok.hf.decode(tok.encode_chat(MESSAGES))
    assert text.endswith("<think>\n\n</think>\n\n")


def test_encode_chat_thinking_on_does_not(tok: HFTokenizer) -> None:
    text = tok.hf.decode(tok.encode_chat(MESSAGES, enable_thinking=True))
    assert text.endswith("<|im_start|>assistant\n")
    assert "</think>" not in text


def test_prompt_too_long_raises(tok: HFTokenizer) -> None:
    small = HFTokenizer("Qwen/Qwen3-4B", 5)
    with pytest.raises(PromptTooLong) as exc:
        small.encode_chat(MESSAGES)
    assert exc.value.max_model_len == 5
    assert exc.value.num_tokens > 5
    assert str(exc.value.num_tokens) in str(exc.value)


def test_detokenizer_roundtrip_hindi(tok: HFTokenizer) -> None:
    text = "नमस्ते दुनिया"
    ids = tok.hf.encode(text, add_special_tokens=False)
    assert _stream(tok, ids) == text


def test_detokenizer_roundtrip_emoji_and_ascii(tok: HFTokenizer) -> None:
    text = "Hello 👋🏽 world, ok 🎉!"
    ids = tok.hf.encode(text, add_special_tokens=False)
    detok = tok.new_detokenizer()
    pieces = [detok.step(i) for i in ids]
    assert "".join(pieces) == text
    assert all("�" not in p for p in pieces)


def test_detokenizer_skips_eos(tok: HFTokenizer) -> None:
    ids = tok.hf.encode("done", add_special_tokens=False)
    assert _stream(tok, [*ids, 151645]) == "done"


def test_detokenizer_keeps_think_tags(tok: HFTokenizer) -> None:
    text = "<think>hi</think>"
    ids = tok.hf.encode(text, add_special_tokens=False)
    assert _stream(tok, ids) == text


def test_detokenizer_type(tok: HFTokenizer) -> None:
    assert isinstance(tok.new_detokenizer(), IncrementalDetokenizer)
