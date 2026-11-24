"""Tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    # The `with` block runs the startup (lifespan) so the model gets loaded
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid(client):
    response = client.post("/predict", json={"features": [0.1] * 10})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)


def test_predict_wrong_length(client):
    response = client.post("/predict", json={"features": [0.1] * 3})
    assert response.status_code == 422


def test_predict_wrong_type(client):
    response = client.post("/predict", json={"features": ["a"] * 10})
    assert response.status_code == 422