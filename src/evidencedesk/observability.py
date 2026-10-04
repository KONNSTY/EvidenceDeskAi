import json
import logging
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from evidencedesk.config import Settings

logger = logging.getLogger("evidencedesk")


class JsonFormatter(logging.Formatter):
    """Formatiert Logrecords als eine JSON-Zeile, damit sie maschinell auswertbar sind."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def log_event(message: str, **fields: Any) -> None:
    logger.info(message, extra={"fields": fields})


@dataclass(frozen=True)
class Usage:
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


def estimate_cost(prompt_tokens: int, completion_tokens: int, settings: Settings) -> float:
    """Schaetzt die USD-Kosten anhand der konfigurierten Token-Preise."""
    cost = (
        prompt_tokens / 1_000_000 * settings.price_prompt_per_mtok
        + completion_tokens / 1_000_000 * settings.price_completion_per_mtok
    )
    return round(cost, 6)


class Stopwatch:
    """Kontextmanager zur Latenzmessung in Millisekunden."""

    def __init__(self) -> None:
        self._started_at = 0.0
        self.elapsed_ms = 0

    def __enter__(self) -> "Stopwatch":
        self._started_at = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._started_at) * 1000)
