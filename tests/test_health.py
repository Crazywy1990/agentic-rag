from fastapi.testclient import TestClient

from rag.main import create_app


def test_healthz() -> None:
    client = TestClient(create_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics_exposed() -> None:
    client = TestClient(create_app())
    assert client.get("/metrics").status_code == 200
