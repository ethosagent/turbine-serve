"""Block 2: Qwen3 chat template + encode; raises PromptTooLong past max_model_len."""

import time
from typing import Any

from transformers import AutoTokenizer

from turbine_serve.detokenizer import IncrementalDetokenizer
from turbine_serve.interfaces import Detokenizer
from turbine_serve.log import get_logger

log = get_logger("tokenizer")


class PromptTooLong(ValueError):
    def __init__(self, num_tokens: int, max_model_len: int) -> None:
        super().__init__(f"prompt has {num_tokens} tokens, exceeding max_model_len {max_model_len}")
        self.num_tokens = num_tokens
        self.max_model_len = max_model_len


class HFTokenizer:
    """Qwen3 tokenizer; model_path is a Hub id or a local directory."""

    eos_token_ids: tuple[int, ...] = (151645, 151643)  # <|im_end|>, <|endoftext|>

    def __init__(self, model_path: str, max_model_len: int) -> None:
        self.hf = AutoTokenizer.from_pretrained(model_path)
        self.max_model_len = max_model_len

    def encode_chat(self, messages: list[dict[str, Any]], **template_kwargs: Any) -> list[int]:
        start = time.perf_counter()
        text = self.hf.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=template_kwargs.get("enable_thinking", False),
        )
        ids: list[int] = self.hf.encode(text, add_special_tokens=False)
        if len(ids) > self.max_model_len:
            log.warning("prompt.too_long", num_tokens=len(ids), max_model_len=self.max_model_len)
            raise PromptTooLong(len(ids), self.max_model_len)
        log.info(
            "prompt.encoded",
            num_prompt_tokens=len(ids),
            ms=round((time.perf_counter() - start) * 1000, 3),
        )
        return ids

    def new_detokenizer(self) -> Detokenizer:
        return IncrementalDetokenizer(self.hf)
