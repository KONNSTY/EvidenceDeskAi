from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from evidencedesk.config import Settings


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    """Schmale Naht zum LLM-Anbieter.

    Produktion nutzt :class:`OpenAIClient`; Tests setzen einen deterministischen Fake ein,
    damit die Suite offline und reproduzierbar laeuft.
    """

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def complete(self, system: str, user: str) -> LLMResult: ...


class OpenAIClient:
    """Direkte Anbindung an die OpenAI-API (Chat + Embeddings).

    Der eigentliche Client wird erst beim ersten Aufruf erzeugt. Eine fehlende Konfiguration
    faellt dadurch erst dort auf, wo sie relevant ist - nicht schon beim Aufloesen der
    Request-Dependencies.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: OpenAI | None = None
        self._chat_model = settings.chat_model
        self._embedding_model = settings.embedding_model

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self._settings.openai_api_key:
                raise RuntimeError("ED_OPENAI_API_KEY ist nicht gesetzt")
            self._client = OpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self._embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def complete(self, system: str, user: str) -> LLMResult:
        response = self.client.chat.completions.create(
            model=self._chat_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = response.usage
        return LLMResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
