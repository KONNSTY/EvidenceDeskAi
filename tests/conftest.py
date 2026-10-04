from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from evidencedesk.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """TestClient der echten App; die hier geprueften Faelle brauchen keine Datenbank."""
    with TestClient(app) as test_client:
        yield test_client
