"""Block 11: incremental decode that holds back partial UTF-8."""

from typing import Any

from turbine_serve.log import get_logger

log = get_logger("detokenizer")

_REPLACEMENT = "�"


class IncrementalDetokenizer:
    """Streams text one token at a time (vLLM's prefix/read offset algorithm).

    Decoding only the tail window keeps each step O(window), not O(sequence). Text ending in
    U+FFFD is an incomplete multi-byte character: it is held back until a later token completes it.
    """

    def __init__(self, hf: Any) -> None:
        self._hf = hf
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def _decode(self, ids: list[int]) -> str:
        text: str = self._hf.decode(ids, skip_special_tokens=True)
        return text

    def step(self, token_id: int) -> str:
        """Append one token id; return the newly completed text ('' if nothing is ready)."""
        self.ids.append(token_id)
        prefix_text = self._decode(self.ids[self.prefix_offset : self.read_offset])
        new_text = self._decode(self.ids[self.prefix_offset :])
        if len(new_text) <= len(prefix_text) or new_text.endswith(_REPLACEMENT):
            return ""
        delta = new_text[len(prefix_text) :]
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.ids)
        return delta
