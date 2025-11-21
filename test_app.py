from fastapi.testclient import TestClient
from app import app  # Importa tu aplicación FastAPI
import pytest

# Crea una instancia del cliente de pruebas
client = TestClient(app)

# =========================
# Prueba del endpoint "/health"
# =========================
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "model" in data
    assert isinstance(data["n_expected"], int)

# =========================
# Prueba del endpoint "/schema"
# =========================
def test_schema():
    response = client.get("/schema")
    assert response.status_code == 200
    data = response.json()
    assert "expected_feature_columns" in data
    assert isinstance(data["expected_feature_columns"], list)

# =========================
# Prueba del endpoint "/predict"
# =========================
def test_predict():
    payload = {
        "features": {
            "Date": "2025-11-20T10:30",
            "z1_S1(degC)": 28.6,
            "z1_S1(RH%)": 64.8,
            "z1_S1(lux)": 120.0,
            "z2_S1(degC)": 28.7,
            "z2_S1(RH%)": 66.2,
            "z2_S1(lux)": 110.0,
            "z4_S1(degC)": 29.0,
            "z4_S1(RH%)": 65.4,
            "z4_S1(lux)": 95.0,
            "z5_S1(degC)": 28.9,
            "z5_S1(RH%)": 66.0,
            "z5_S1(lux)": 100.0
        }
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)

# =========================
# Prueba de error en "/predict" (Faltan columnas)
# =========================
def test_predict_missing_columns():
    payload = {
        "features": {
            "z1_S1(degC)": 28.6,
            "z1_S1(RH%)": 64.8
        }
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "missing" in data
    assert isinstance(data["missing"], list)
    assert "Date" in data["missing"]
