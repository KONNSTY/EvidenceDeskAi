from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from evidencedesk.deps import LLMDep, SessionDep, SettingsDep, TenantDep
from evidencedesk.ingest import EmptyDocumentError, ingest_document
from evidencedesk.models import Document
from evidencedesk.observability import log_event
from evidencedesk.schemas import DocumentCreate, DocumentCreated, DocumentSummary

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("", response_model=DocumentCreated, status_code=status.HTTP_201_CREATED)
def create_document(
    payload: DocumentCreate,
    tenant_id: TenantDep,
    session: SessionDep,
    llm: LLMDep,
    settings: SettingsDep,
) -> DocumentCreated:
    try:
        result = ingest_document(
            session,
            llm,
            settings,
            tenant_id=tenant_id,
            title=payload.title,
            source=payload.source,
            content=payload.content,
        )
    except EmptyDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    log_event(
        "document.ingested",
        tenant_id=str(tenant_id),
        document_id=str(result.document_id),
        chunk_count=result.chunk_count,
        duplicate=result.duplicate,
    )
    return DocumentCreated(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        duplicate=result.duplicate,
    )


@router.get("", response_model=list[DocumentSummary])
def list_documents(tenant_id: TenantDep, session: SessionDep) -> list[DocumentSummary]:
    documents = session.scalars(
        select(Document).where(Document.tenant_id == tenant_id).order_by(Document.created_at.desc())
    ).all()
    return [
        DocumentSummary(
            document_id=document.id,
            title=document.title,
            source=document.source,
            created_at=document.created_at,
        )
        for document in documents
    ]
