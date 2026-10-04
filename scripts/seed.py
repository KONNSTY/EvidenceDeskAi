"""Legt einen Demo-Mandanten mit Beispieldokumenten an.

Voraussetzung: laufende Datenbank (``docker compose up -d db``), Migration angewendet
und ``ED_OPENAI_API_KEY`` gesetzt.
"""

from uuid import UUID

from evidencedesk.config import get_settings
from evidencedesk.db import build_database
from evidencedesk.ingest import ingest_document
from evidencedesk.llm import OpenAIClient
from evidencedesk.models import Tenant

DEMO_TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")

DEMO_DOCUMENTS: list[tuple[str, str]] = [
    (
        "Rueckgabe-Richtlinie",
        "Kunden koennen Artikel innerhalb von 30 Tagen nach Erhalt zurueckgeben. "
        "Die Ware muss unbenutzt sein und in der Originalverpackung vorliegen. "
        "Erstattungen erfolgen auf das urspruengliche Zahlungsmittel innerhalb von "
        "fuenf Werktagen nach Eingang der Retoure.",
    ),
    (
        "Versandzeiten",
        "Standardversand innerhalb Deutschlands dauert zwei bis vier Werktage. "
        "Expressversand wird am naechsten Werktag zugestellt, sofern die Bestellung "
        "vor 14 Uhr aufgegeben wurde. Bestellungen am Wochenende werden am folgenden "
        "Montag bearbeitet.",
    ),
]


def main() -> None:
    settings = get_settings()
    database = build_database(settings)
    llm = OpenAIClient(settings)

    with database.session() as session:
        tenant = session.get(Tenant, DEMO_TENANT_ID)
        if tenant is None:
            tenant = Tenant(id=DEMO_TENANT_ID, name="Demo Mandant")
            session.add(tenant)
            session.commit()
            print(f"Mandant angelegt: {DEMO_TENANT_ID}")
        else:
            print(f"Mandant existiert bereits: {DEMO_TENANT_ID}")

        for title, content in DEMO_DOCUMENTS:
            result = ingest_document(
                session,
                llm,
                settings,
                tenant_id=DEMO_TENANT_ID,
                title=title,
                source=None,
                content=content,
            )
            status = "uebersprungen (Duplikat)" if result.duplicate else "angelegt"
            print(f"{title}: {status}, {result.chunk_count} Chunks")

    database.dispose()


if __name__ == "__main__":
    main()
