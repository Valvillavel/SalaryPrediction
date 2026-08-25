import pytest
from fastapi.testclient import TestClient
from src.api.main import app


VALID_PAYLOAD = {
    "age": 39, "workclass": "Private", "fnlwgt": 77516,
    "education": "Bachelors", "education-num": 13,
    "marital-status": "Never-married", "occupation": "Adm-clerical",
    "relationship": "Not-in-family", "race": "White", "sex": "Male",
    "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40,
    "native-country": "United-States",
}


@pytest.fixture
def client():
    with TestClient(app) as c:  # dispara lifespan (carga modelo/preprocesador)
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.slow  # depende de que existan los .pkl generados por `dvc repro`
def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["salary_prediction"] in {">50K", "<=50K"}
    assert 0.0 <= body["probability_above_50k"] <= 1.0


def test_predict_rejects_invalid_payload(client):
    bad_payload = {**VALID_PAYLOAD, "age": -5}  # viola ge=17
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422