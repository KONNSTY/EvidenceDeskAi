from fastapi.testclient import TestClient


def test_healthz_is_ok(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_without_tenant_header_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "Wie lange ist die Frist?"})

    assert response.status_code == 422


def test_query_with_malformed_tenant_header_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/query",
        headers={"X-Tenant-Id": "keine-uuid"},
        json={"question": "Wie lange ist die Frist?"},
    )

    assert response.status_code == 422
