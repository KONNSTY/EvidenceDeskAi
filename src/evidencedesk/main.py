from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from evidencedesk import __version__
from evidencedesk.api import documents, health, query
from evidencedesk.config import get_settings
from evidencedesk.deps import get_database
from evidencedesk.observability import configure_logging


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    get_database().dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="EvidenceDesk AI",
        version=__version__,
        description="Support-Copilot mit pruefbaren Quellen und Abstention bei fehlender Evidenz.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    return app


app = create_app()
