import math
from dataclasses import dataclass, replace
from typing import Protocol
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from evidencedesk.models import Chunk, Document


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    title: str
    content: str
    score: float


class Retriever(Protocol):
    def search(
        self,
        session: Session | None,
        *,
        tenant_id: UUID,
        embedding: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievedChunk]: ...


class PgVectorRetriever:
    """Vektorsuche in Postgres via pgvector, hart auf den Mandanten gefiltert."""

    def search(
        self,
        session: Session | None,
        *,
        tenant_id: UUID,
        embedding: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievedChunk]:
        if session is None:
            raise ValueError("PgVectorRetriever benoetigt eine Datenbank-Session")

        distance = Chunk.embedding.cosine_distance(embedding)
        statement: Select[tuple[UUID, UUID, str, str, float]] = (
            select(
                Chunk.id,
                Chunk.document_id,
                Document.title,
                Chunk.content,
                distance.label("distance"),
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.tenant_id == tenant_id)
            .where(distance <= 1 - score_threshold)
            .order_by(distance)
            .limit(top_k)
        )
        rows = session.execute(statement).all()
        return [
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                title=row.title,
                content=row.content,
                score=1 - float(row.distance),
            )
            for row in rows
        ]


class InMemoryRetriever:
    """Retriever fuer Unit-Tests: gleiche Semantik ohne Datenbank."""

    def __init__(self) -> None:
        self._items: list[tuple[UUID, list[float], RetrievedChunk]] = []

    def add(self, *, tenant_id: UUID, embedding: list[float], chunk: RetrievedChunk) -> None:
        self._items.append((tenant_id, embedding, chunk))

    def search(
        self,
        session: Session | None,
        *,
        tenant_id: UUID,
        embedding: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievedChunk]:
        scored: list[RetrievedChunk] = []
        for item_tenant, item_embedding, chunk in self._items:
            if item_tenant != tenant_id:
                continue
            score = _cosine_similarity(embedding, item_embedding)
            if score >= score_threshold:
                scored.append(replace(chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
