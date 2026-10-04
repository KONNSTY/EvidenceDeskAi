import json
from uuid import UUID, uuid4

from evidencedesk.answering import (
    INSUFFICIENT_REASON,
    INVALID_OUTPUT_REASON,
    NO_EVIDENCE_REASON,
    answer_question,
)
from evidencedesk.retrieval import InMemoryRetriever, RetrievedChunk
from tests.fakes import FakeLLM, deterministic_embedding, make_settings

TENANT = uuid4()
QUESTION = "wie lange ist die rueckgabefrist"
CONTENT = "Kunden koennen Artikel innerhalb von 30 Tagen zurueckgeben."


def _retriever() -> tuple[InMemoryRetriever, RetrievedChunk]:
    retriever = InMemoryRetriever()
    chunk = RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title="Rueckgabe-Richtlinie",
        content=CONTENT,
        score=0.0,
    )
    retriever.add(
        tenant_id=TENANT,
        embedding=deterministic_embedding(QUESTION),
        chunk=chunk,
    )
    return retriever, chunk


def _ask(retriever: InMemoryRetriever, llm: FakeLLM, **settings_overrides: object):
    return answer_question(
        llm=llm,
        retriever=retriever,
        session=None,
        settings=make_settings(**settings_overrides),
        tenant_id=TENANT,
        question=QUESTION,
        top_k=3,
    )


def test_answer_is_returned_with_verified_citation() -> None:
    retriever, chunk = _retriever()

    result = _ask(retriever, FakeLLM())

    assert result.insufficient_evidence is False
    assert result.answer == "Antwort aus dem Kontext."
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == chunk.chunk_id
    assert result.citations[0].document_id == chunk.document_id
    assert result.citations[0].title == "Rueckgabe-Richtlinie"
    assert result.citations[0].quote in CONTENT


def test_hallucinated_citation_id_is_discarded() -> None:
    retriever, _ = _retriever()
    llm = FakeLLM(
        json.dumps(
            {
                "answer": "Erfundene Antwort.",
                "citations": [str(uuid4())],
                "insufficient_evidence": False,
            }
        )
    )

    result = _ask(retriever, llm)

    assert result.insufficient_evidence is True
    assert result.citations == ()
    assert result.answer is None
    assert result.reason == INSUFFICIENT_REASON


def test_answer_without_citations_is_treated_as_abstention() -> None:
    retriever, _ = _retriever()
    payload = {"answer": "Antwort ohne Beleg.", "citations": [], "insufficient_evidence": False}
    llm = FakeLLM(json.dumps(payload))

    result = _ask(retriever, llm)

    assert result.insufficient_evidence is True
    assert result.reason == INSUFFICIENT_REASON


def test_model_can_abstain_explicitly() -> None:
    retriever, _ = _retriever()
    llm = FakeLLM(json.dumps({"answer": None, "citations": [], "insufficient_evidence": True}))

    result = _ask(retriever, llm)

    assert result.insufficient_evidence is True
    assert result.reason == INSUFFICIENT_REASON
    assert len(llm.calls) == 1


def test_malformed_model_output_is_treated_as_abstention() -> None:
    retriever, _ = _retriever()
    llm = FakeLLM("das ist kein json")

    result = _ask(retriever, llm)

    assert result.insufficient_evidence is True
    assert result.reason == INVALID_OUTPUT_REASON


def test_no_context_means_no_llm_call() -> None:
    retriever, _ = _retriever()
    llm = FakeLLM()

    result = _ask(retriever, llm, score_threshold=1.5)

    assert result.insufficient_evidence is True
    assert result.reason == NO_EVIDENCE_REASON
    assert result.prompt_tokens == 0
    assert llm.calls == []


def test_context_budget_limits_prompt_size() -> None:
    retriever = InMemoryRetriever()
    embedding = deterministic_embedding(QUESTION)
    for index in range(3):
        retriever.add(
            tenant_id=TENANT,
            embedding=embedding,
            chunk=RetrievedChunk(
                chunk_id=UUID(int=index + 1),
                document_id=uuid4(),
                title=f"Dokument {index}",
                content="x" * 400,
                score=0.0,
            ),
        )
    llm = FakeLLM()

    result = _ask(retriever, llm, max_context_chars=500)

    assert len(llm.calls) == 1
    assert "x" * 400 in llm.calls[0][1]
    assert llm.calls[0][1].count("UNTRUSTED_DOCUMENT id=") == 1
    assert result.insufficient_evidence is False
