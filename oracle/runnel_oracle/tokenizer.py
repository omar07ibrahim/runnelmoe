"""Fixed tokenizer used solely by the deterministic tiny fixture."""

from __future__ import annotations

from collections.abc import Iterable

from .spec import FixtureSpec


class TinyTokenizer:
    def __init__(self, spec: FixtureSpec) -> None:
        tokens = tuple(spec.tokenizer["tokens"])
        self._tokens = tokens
        self._encode = {token: token_id for token_id, token in enumerate(tokens)}
        self.eos_token_id = int(spec.tokenizer["eos_token_id"])
        self.bos_token_id = int(spec.tokenizer["bos_token_id"])

    def encode(self, text: str) -> list[int]:
        """Encode lowercase fixture text, always prepending BOS."""

        encoded = [self.bos_token_id]
        for position, character in enumerate(text):
            token_id = self._encode.get(character)
            if token_id is None or token_id < 2:
                raise ValueError(f"unsupported character at position {position}: {character!r}")
            encoded.append(token_id)
        return encoded

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        pieces: list[str] = []
        for position, token_id in enumerate(token_ids):
            if not isinstance(token_id, int) or not 0 <= token_id < len(self._tokens):
                raise ValueError(f"invalid token ID at position {position}: {token_id!r}")
            if token_id == self.eos_token_id:
                if not skip_special:
                    pieces.append(self._tokens[token_id])
                break
            if skip_special and token_id == self.bos_token_id:
                continue
            pieces.append(self._tokens[token_id])
        return "".join(pieces)
