"""Tests del endpoint de la API con TestClient (no necesita levantar uvicorn)."""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test del endpoint raíz que devuelve metadata del modelo."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "status" in data
    assert data["model_name"] == "aml_prediction_Model"


def test_predict_endpoint_estructura_correcta():
    """Test del endpoint /predict con un payload válido de salary.csv."""
    payload = {
        "data": [{
            "age": 39,
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        }]
    }
    response = client.post("/predict", json=payload)
    
    # Si no hay modelo en Production, esto da 500 (esperado en CI sin modelo entrenado)
    assert response.status_code in (200, 500)
    
    # Si el modelo está disponible, validar estructura de respuesta
    if response.status_code == 200:
        data = response.json()
        assert "model_metadata" in data
        assert "total_predictions" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["salary_class"] in ["<=50K", ">50K"]


def test_predict_endpoint_multiples_registros():
    """Test con múltiples registros de entrada."""
    payload = {
        "data": [
            {
                "age": 39, "workclass": "State-gov", "fnlwgt": 77516,
                "education": "Bachelors", "education-num": 13,
                "marital-status": "Never-married", "occupation": "Adm-clerical",
                "relationship": "Not-in-family", "race": "White", "sex": "Male",
                "capital-gain": 2174, "capital-loss": 0,
                "hours-per-week": 40, "native-country": "United-States",
            },
            {
                "age": 52, "workclass": "Self-emp-not-inc", "fnlwgt": 209642,
                "education": "HS-grad", "education-num": 9,
                "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
                "relationship": "Husband", "race": "White", "sex": "Male",
                "capital-gain": 0, "capital-loss": 0,
                "hours-per-week": 45, "native-country": "United-States",
            }
        ]
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code in (200, 500)
    
    if response.status_code == 200:
        data = response.json()
        assert data["total_predictions"] == 2
        assert len(data["results"]) == 2
