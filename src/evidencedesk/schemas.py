from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    source: str | None = Field(default=None, max_length=500)
    content: str = Field(min_length=1)


class DocumentCreated(BaseModel):
    document_id: UUID
    chunk_count: int
    duplicate: bool


class DocumentSummary(BaseModel):
    document_id: UUID
    title: str
    source: str | None
    created_at: datetime


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    chunk_id: UUID
    document_id: UUID
    title: str
    score: float
    quote: str


class UsageOut(BaseModel):
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class QueryResponse(BaseModel):
    answer: str | None
    citations: list[CitationOut]
    insufficient_evidence: bool
    reason: str | None = None
    usage: UsageOut
