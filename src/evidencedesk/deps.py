from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from evidencedesk.config import Settings, get_settings
from evidencedesk.db import Database, build_database
from evidencedesk.llm import LLMClient, OpenAIClient
from evidencedesk.models import Tenant
from evidencedesk.retrieval import PgVectorRetriever, Retriever


@lru_cache
def get_database() -> Database:
    return build_database(get_settings())


def get_session(database: Annotated[Database, Depends(get_database)]) -> Iterator[Session]:
    with database.session() as session:
        yield session


@lru_cache
def get_llm() -> LLMClient:
    return OpenAIClient(get_settings())


@lru_cache
def get_retriever() -> Retriever:
    return PgVectorRetriever()


def get_tenant(
    session: Annotated[Session, Depends(get_session)],
    x_tenant_id: Annotated[UUID, Header(alias="X-Tenant-Id")],
) -> UUID:
    """Loest den Mandanten aus dem Header auf und lehnt unbekannte Mandanten ab."""
    if session.get(Tenant, x_tenant_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unbekannter Mandant")
    return x_tenant_id


SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
TenantDep = Annotated[UUID, Depends(get_tenant)]
