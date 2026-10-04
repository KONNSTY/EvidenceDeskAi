from uuid import uuid4

from evidencedesk.retrieval import InMemoryRetriever, RetrievedChunk
from tests.fakes import deterministic_embedding

TENANT_A = uuid4()
TENANT_B = uuid4()
EMBEDDING = deterministic_embedding("handbuch rueckgabe frist")


def _chunk(title: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title=title,
        content=f"Inhalt fuer {title}",
        score=0.0,
    )


def test_search_only_returns_chunks_of_the_requested_tenant() -> None:
    retriever = InMemoryRetriever()
    chunk_a = _chunk("Mandant A")
    chunk_b = _chunk("Mandant B")
    retriever.add(tenant_id=TENANT_A, embedding=EMBEDDING, chunk=chunk_a)
    retriever.add(tenant_id=TENANT_B, embedding=EMBEDDING, chunk=chunk_b)

    results = retriever.search(
        None,
        tenant_id=TENANT_A,
        embedding=EMBEDDING,
        top_k=10,
        score_threshold=0.0,
    )

    assert [chunk.chunk_id for chunk in results] == [chunk_a.chunk_id]


def test_search_for_unknown_tenant_returns_nothing() -> None:
    retriever = InMemoryRetriever()
    retriever.add(tenant_id=TENANT_A, embedding=EMBEDDING, chunk=_chunk("Mandant A"))

    results = retriever.search(
        None,
        tenant_id=uuid4(),
        embedding=EMBEDDING,
        top_k=10,
        score_threshold=0.0,
    )

    assert results == []


def test_search_respects_top_k_and_threshold() -> None:
    retriever = InMemoryRetriever()
    for index in range(5):
        retriever.add(
            tenant_id=TENANT_A,
            embedding=EMBEDDING,
            chunk=RetrievedChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                title=f"Dokument {index}",
                content="Inhalt",
                score=0.0,
            ),
        )

    results = retriever.search(
        None,
        tenant_id=TENANT_A,
        embedding=EMBEDDING,
        top_k=2,
        score_threshold=0.0,
    )

    assert len(results) == 2
    assert results[0].score >= results[1].score
