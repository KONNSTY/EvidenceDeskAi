import hashlib
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evidencedesk.chunking import chunk_text
from evidencedesk.config import Settings
from evidencedesk.llm import LLMClient
from evidencedesk.models import Chunk, Document


class EmptyDocumentError(ValueError):
    """Wird ausgeloest, wenn ein Dokument keinen verwertbaren Text enthaelt."""


@dataclass(frozen=True)
class IngestResult:
    document_id: UUID
    chunk_count: int
    duplicate: bool


def ingest_document(
    session: Session,
    llm: LLMClient,
    settings: Settings,
    *,
    tenant_id: UUID,
    title: str,
    source: str | None,
    content: str,
) -> IngestResult:
    """Chunked, embedded und speichert ein Dokument mandantengetrennt."""
    content_hash = hashlib.sha256(" ".join(content.split()).encode("utf-8")).hexdigest()
    existing = session.scalar(
        select(Document).where(
            Document.tenant_id == tenant_id,
            Document.content_hash == content_hash,
        )
    )
    if existing is not None:
        count = (
            session.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.document_id == existing.id)
            )
            or 0
        )
        return IngestResult(existing.id, count, True)

    chunks = chunk_text(content, size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise EmptyDocumentError("Dokument enthaelt keinen verwertbaren Text")

    embeddings = llm.embed([chunk.content for chunk in chunks])

    document = Document(tenant_id=tenant_id, title=title, source=source, content_hash=content_hash)
    session.add(document)
    session.flush()
    session.add_all(
        [
            Chunk(
                tenant_id=tenant_id,
                document_id=document.id,
                ordinal=chunk.ordinal,
                content=chunk.content,
                token_count=chunk.token_count,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
    )
    session.commit()
    return IngestResult(document.id, len(chunks), False)
