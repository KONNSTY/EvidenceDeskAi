# EvidenceDesk AI

Ein Support-Copilot für interne Wissensbasen. Er beantwortet Fragen **ausschliesslich** anhand
ingestierter Dokumente, liefert **prüfbare Zitate** und **enthält sich** (`insufficient_evidence`),
wenn die Evidenz nicht reicht.

Der Fokus liegt auf dem Engineering-Nachweis: Zitat-Validierung gegen halluzinierte Quellen,
Mandantentrennung, Prompt-Injection-Behandlung sowie Messung von Latenz und Kosten pro Anfrage.

## Kernversprechen

- **Keine Antwort ohne Quelle.** Jedes Zitat wird gegen die tatsächlich abgerufenen Chunks validiert;
  erfundene Quellen-IDs werden verworfen.
- **Abstention statt Raten.** Unterhalb des Ähnlichkeits-Schwellwerts wird gar kein LLM aufgerufen.
- **Mandantentrennung.** Jeder Retrieval-Query filtert hart auf die Mandanten-ID.
- **Messbar.** Latenz, Token-Nutzung und Kostenschätzung werden pro Anfrage protokolliert.

## Architektur

```
HTTP (X-Tenant-Id)
      │
      ▼
 FastAPI Routen  ──►  ingest.py        Chunking ──► Embeddings ──► Postgres + pgvector
      │                                (security.wrap_untrusted beim Prompt-Aufbau)
      ▼
 answering.py  ──►  retrieval.py  (Top-k, tenant-gefiltert, score >= threshold)
      │                    │
      │                    └── keine Treffer ──► insufficient_evidence (kein LLM-Call)
      ▼
   LLM (OpenAI)  ──►  JSON  ──►  Zitat-Validierung ──► Antwort + Citations + usage
```

**Stack:** Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.0, PostgreSQL + pgvector, Alembic, OpenAI.

### Warum pgvector statt eines dedizierten Vektor-Stores?

Eine Infrastruktur, Mandantentrennung als gewöhnlicher SQL-Filter (`WHERE tenant_id = ...`),
HNSW-Index und transaktionale Konsistenz zwischen Dokument- und Vektor-Daten. Für Mandantenfähigkeit
ist das der direkteste Weg – ein separater Vektor-Store hätte eine zweite Quelle der Wahrheit eingeführt.

## Setup

```bash
# 1. Abhängigkeiten (uv legt .venv mit Python 3.12 an)
make setup

# 2. Datenbank starten
make up

# 3. Schema anlegen
make migrate

# 4. Konfiguration
cp .env.example .env        # ED_OPENAI_API_KEY setzen

# 5. Server starten
make dev
```

Alle `make`-Ziele setzen `PYTHONPATH=src`, damit die Imports unabhängig von der
Editable-Installation des Pakets funktionieren.

## API

| Methode | Pfad            | Beschreibung                                              |
|---------|-----------------|-----------------------------------------------------------|
| GET     | `/healthz`      | Liveness                                                  |
| GET     | `/readyz`       | Readiness (DB-Ping)                                       |
| POST    | `/v1/documents` | Dokument ingestieren (chunken, embedden, speichern)      |
| GET     | `/v1/documents` | Dokumente des Mandanten                                   |
| POST    | `/v1/query`     | Frage beantworten – mit Zitaten oder mit Abstention       |

Alle `/v1`-Routen erwarten den Header `X-Tenant-Id` (UUID).

```bash
TENANT=11111111-1111-1111-1111-111111111111

curl -s -X POST localhost:8000/v1/documents \
  -H "X-Tenant-Id: $TENANT" -H 'Content-Type: application/json' \
  -d '{"title":"Rückgabe-Richtlinie","content":"Kunden können Artikel innerhalb von 30 Tagen zurückgeben. ..."}'

curl -s -X POST localhost:8000/v1/query \
  -H "X-Tenant-Id: $TENANT" -H 'Content-Type: application/json' \
  -d '{"question":"Wie lange ist die Rückgabefrist?"}'
```

Antwort mit Evidenz:

```json
{
  "answer": "Artikel können innerhalb von 30 Tagen zurückgegeben werden.",
  "citations": [
    {"chunk_id": "…", "document_id": "…", "title": "Rückgabe-Richtlinie",
     "score": 0.82, "quote": "Kunden können Artikel innerhalb von 30 Tagen zurückgeben."}
  ],
  "insufficient_evidence": false,
  "usage": {"latency_ms": 640, "prompt_tokens": 812, "completion_tokens": 24, "cost_usd": 0.000136}
}
```

Antwort ohne Evidenz – es wird kein LLM aufgerufen:

```json
{"answer": null, "citations": [], "insufficient_evidence": true,
 "reason": "no_chunks_above_threshold",
 "usage": {"latency_ms": 12, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}}
```

## Tests

```bash
make test                 # Unit-Tests: laufen ohne Netzwerk und ohne API-Key
make test-integration     # benötigt Postgres (make up); nutzt sonst TEST_DATABASE_URL
make check                # lint + typecheck + tests
```

Unit-Tests nutzen einen deterministischen Fake-LLM und einen In-Memory-Retriever. Integrationstests
prüfen die Mandantentrennung auf echter SQL-Ebene und werden übersprungen, wenn `TEST_DATABASE_URL`
nicht gesetzt ist.

## Design-Entscheidungen

- **Zitate stammen aus den Chunks, nicht aus dem Modell.** Das Modell gibt nur Chunk-IDs zurück;
  Zitattext und Dokumenttitel werden serverseitig aus der Datenbank angehängt. Damit ist ein Zitat
  nicht fälschbar, selbst wenn das Modell halluziniert.
- **Abstention vor dem LLM-Call.** Spart Kosten und verhindert erzwungene Antworten.
- **Retrieved Text ist untrusted.** Er wird eingerahmt und als Daten deklariert; verdächtige
  Injection-Muster werden erkannt und protokolliert.
- **LLM hinter einer schmalen Naht.** Produktion nutzt das OpenAI-SDK direkt; Tests setzen einen
  deterministischen Fake ein, damit die Suite offline und reproduzierbar läuft.

## Status

Erster lauffähiger Vertical Slice. Noch nicht enthalten: Eval-Harness mit Golden-Set
(Citation-Precision, Abstention-Accuracy), echte Authentifizierung, Re-Ranking, Streaming, Frontend.
