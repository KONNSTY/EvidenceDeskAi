from fastapi import APIRouter

from evidencedesk.answering import AnswerResult, answer_question
from evidencedesk.deps import LLMDep, RetrieverDep, SessionDep, SettingsDep, TenantDep
from evidencedesk.models import QueryLog
from evidencedesk.observability import Stopwatch, estimate_cost, log_event
from evidencedesk.schemas import CitationOut, QueryRequest, QueryResponse, UsageOut

router = APIRouter(prefix="/v1/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def create_query(
    payload: QueryRequest,
    tenant_id: TenantDep,
    session: SessionDep,
    llm: LLMDep,
    retriever: RetrieverDep,
    settings: SettingsDep,
) -> QueryResponse:
    with Stopwatch() as stopwatch:
        result = answer_question(
            llm=llm,
            retriever=retriever,
            session=session,
            settings=settings,
            tenant_id=tenant_id,
            question=payload.question,
            top_k=payload.top_k or settings.top_k,
        )

    cost = estimate_cost(result.prompt_tokens, result.completion_tokens, settings)
    _write_query_log(session, tenant_id, payload.question, result, stopwatch.elapsed_ms, cost)
    log_event(
        "query.completed",
        tenant_id=str(tenant_id),
        insufficient_evidence=result.insufficient_evidence,
        reason=result.reason,
        latency_ms=stopwatch.elapsed_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=cost,
        flagged_patterns=list(result.flagged_patterns),
    )

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                title=citation.title,
                score=citation.score,
                quote=citation.quote,
            )
            for citation in result.citations
        ],
        insufficient_evidence=result.insufficient_evidence,
        reason=result.reason,
        usage=UsageOut(
            latency_ms=stopwatch.elapsed_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=cost,
        ),
    )


def _write_query_log(
    session: SessionDep,
    tenant_id: TenantDep,
    question: str,
    result: AnswerResult,
    latency_ms: int,
    cost: float,
) -> None:
    session.add(
        QueryLog(
            tenant_id=tenant_id,
            question=question,
            answer=result.answer,
            citations=[
                {"chunk_id": str(citation.chunk_id), "document_id": str(citation.document_id)}
                for citation in result.citations
            ],
            insufficient_evidence=result.insufficient_evidence,
            latency_ms=latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=cost,
        )
    )
    session.commit()
