from fastapi import APIRouter
from sqlalchemy import text

from evidencedesk.deps import SessionDep
from evidencedesk.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=HealthResponse)
def readyz(session: SessionDep) -> HealthResponse:
    session.execute(text("SELECT 1"))
    return HealthResponse(status="ok")
