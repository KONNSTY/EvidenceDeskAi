"""Integrationstests gegen eine echte Postgres-Instanz mit pgvector.

Werden uebersprungen, wenn ``TEST_DATABASE_URL`` nicht gesetzt ist.
"""

import os
from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from evidencedesk.config import Settings, get_settings
from evidencedesk.db import Database
from evidencedesk.deps import get_database, get_llm, get_retriever
from evidencedesk.main import app
from evidencedesk.models import Base, Chunk, Document, Tenant
from evidencedesk.retrieval import PgVectorRetriever
from tests.fakes import FakeLLM, deterministic_embedding

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(DATABASE_URL is None, reason="TEST_DATABASE_URL ist nicht gesetzt"),
]

QUESTION = "wie lange ist die rueckgabefrist"
CONTENT = "Die Rueckgabefrist fuer Artikel betraegt 30 Tage."


@pytest.fixture(scope="module")
def engine():
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine) -> Iterator[Session]:
    with Session(engine) as session:
        yield session


@pytest.fixture
def api_client(engine) -> Iterator[TestClient]:
    assert DATABASE_URL is not None
    database = Database(DATABASE_URL)
    app.dependency_overrides[get_database] = lambda: database
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    app.dependency_overrides[get_retriever] = lambda: PgVectorRetriever()
    app.dependency_overrides[get_settings] = lambda: Settings(
        score_threshold=-1.0, top_k=5, chunk_size=400, chunk_overlap=50
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    database.dispose()


def _create_tenant(session: Session, name: str) -> UUID:
    tenant_id = uuid4()
    session.add(Tenant(id=tenant_id, name=name))
    session.commit()
    return tenant_id


def _insert_chunk(session: Session, tenant_id: UUID, title: str, content: str) -> Chunk:
    document = Document(
        tenant_id=tenant_id, title=title, content_hash=uuid4().hex + uuid4().hex[:32]
    )
    session.add(document)
    session.flush()
    chunk = Chunk(
        tenant_id=tenant_id,
        document_id=document.id,
        ordinal=0,
        content=content,
        token_count=len(content) // 4,
        embedding=deterministic_embedding(QUESTION),
    )
    session.add(chunk)
    session.commit()
    return chunk


def test_pgvector_retriever_enforces_tenant_isolation(session: Session) -> None:
    tenant_a = _create_tenant(session, "Mandant A")
    tenant_b = _create_tenant(session, "Mandant B")
    chunk_a = _insert_chunk(session, tenant_a, "A-Handbuch", "Inhalt von Mandant A")
    _insert_chunk(session, tenant_b, "B-Handbuch", "Inhalt von Mandant B")

    results = PgVectorRetriever().search(
        session,
        tenant_id=tenant_a,
        embedding=deterministic_embedding(QUESTION),
        top_k=10,
        score_threshold=0.0,
    )

    assert [chunk.chunk_id for chunk in results] == [chunk_a.id]
    assert results[0].score > 0.99


def test_query_returns_verified_citations_end_to_end(api_client: TestClient) -> None:
    tenant_id = _create_tenant_from_client(api_client)

    ingest = api_client.post(
        "/v1/documents",
        headers={"X-Tenant-Id": str(tenant_id)},
        json={"title": "Rueckgabe-Richtlinie", "content": CONTENT},
    )
    assert ingest.status_code == 201
    assert ingest.json()["chunk_count"] >= 1

    response = api_client.post(
        "/v1/query",
        headers={"X-Tenant-Id": str(tenant_id)},
        json={"question": QUESTION},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is False
    assert body["citations"], "Es wird mindestens ein Zitat erwartet"
    assert body["citations"][0]["quote"] in CONTENT
    assert body["citations"][0]["score"] > 0.0
    assert body["usage"]["latency_ms"] >= 0


def test_ingest_is_idempotent_for_identical_content(api_client: TestClient) -> None:
    tenant_id = _create_tenant_from_client(api_client)
    payload = {"title": "Versandzeiten", "content": CONTENT}

    first = api_client.post("/v1/documents", headers={"X-Tenant-Id": str(tenant_id)}, json=payload)
    second = api_client.post("/v1/documents", headers={"X-Tenant-Id": str(tenant_id)}, json=payload)

    assert first.status_code == 201
    assert first.json()["duplicate"] is False
    assert second.status_code == 201
    assert second.json()["duplicate"] is True
    assert second.json()["document_id"] == first.json()["document_id"]


def test_tenant_without_documents_abstains(api_client: TestClient) -> None:
    tenant_id = _create_tenant_from_client(api_client)

    response = api_client.post(
        "/v1/query",
        headers={"X-Tenant-Id": str(tenant_id)},
        json={"question": QUESTION},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["reason"] == "no_chunks_above_threshold"


def test_unknown_tenant_is_rejected(api_client: TestClient) -> None:
    response = api_client.post(
        "/v1/query",
        headers={"X-Tenant-Id": str(uuid4())},
        json={"question": QUESTION},
    )

    assert response.status_code == 404


def test_empty_question_is_rejected(api_client: TestClient) -> None:
    tenant_id = _create_tenant_from_client(api_client)

    response = api_client.post(
        "/v1/query",
        headers={"X-Tenant-Id": str(tenant_id)},
        json={"question": ""},
    )

    assert response.status_code == 422


def test_readyz_reports_database_connectivity(api_client: TestClient) -> None:
    response = api_client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def _create_tenant_from_client(api_client: TestClient) -> UUID:
    """Legt einen Mandanten direkt in der Datenbank an, die die App benutzt."""
    assert DATABASE_URL is not None
    database = Database(DATABASE_URL)
    with database.session() as session:
        tenant_id = uuid4()
        session.add(Tenant(id=tenant_id, name=f"Mandant {tenant_id}"))
        session.commit()
    database.dispose()
    return tenant_id
