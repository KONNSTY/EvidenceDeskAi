"""Deterministische Test-Doubles: keine Netzwerkaufrufe, reproduzierbare Ergebnisse."""

import hashlib
import json
import math
import re

from evidencedesk.config import Settings
from evidencedesk.llm import LLMResult

_ID_PATTERN = re.compile(r"<<<UNTRUSTED_DOCUMENT id=([0-9a-fA-F-]+)")
_TOKEN_PATTERN = re.compile(r"\w+")

DEFAULT_SYSTEM_PROMPT = "system"


def deterministic_embedding(text: str, dim: int = 1536) -> list[float]:
    """Bag-of-Words-Hashing: gleiche Texte ergeben gleiche Vektoren."""
    vector = [0.0] * dim
    for token in _TOKEN_PATTERN.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "score_threshold": 0.0,
        "top_k": 3,
        "chunk_size": 200,
        "chunk_overlap": 20,
        "openai_api_key": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class FakeLLM:
    """LLM-Ersatz, der standardmaessig die erste Kontext-id zitiert."""

    def __init__(self, response_text: str | None = None) -> None:
        self._response_text = response_text
        self.calls: list[tuple[str, str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [deterministic_embedding(text) for text in texts]

    def complete(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        text = self._response_text if self._response_text is not None else self._cite_first(user)
        return LLMResult(text=text, prompt_tokens=120, completion_tokens=18)

    @staticmethod
    def _cite_first(user: str) -> str:
        match = _ID_PATTERN.search(user)
        if match is None:
            return json.dumps({"answer": None, "citations": [], "insufficient_evidence": True})
        return json.dumps(
            {
                "answer": "Antwort aus dem Kontext.",
                "citations": [match.group(1)],
                "insufficient_evidence": False,
            }
        )
