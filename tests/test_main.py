
import math
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import app.main as main


# Run the application lifespan once for this test module.
@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as test_client:
        yield test_client

#Automated API tests

def test_home(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Real Estate Price Prediction API is running."
    }

#test the health endpoint
def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True
    }



def test_successful_prediction(client):
    payload = {
        "area_type": "Built-up Area",
        "location": "Whitefield",
        "bedrooms": 2,
        "is_rk": False,
        "total_sqft": 1200,
        "bath": 2,
        "balcony": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_price_lakhs" in data
    assert isinstance(data["predicted_price_lakhs"], (int, float))
    assert math.isfinite(data["predicted_price_lakhs"])
    assert data["predicted_price_lakhs"] > 0


def test_prediction_missing_required_field(client):
    payload = {
        "area_type": "Built-up Area",
        # location is intentionally missing
        "bedrooms": 2,
        "is_rk": False,
        "total_sqft": 1200,
        "bath": 2,
        "balcony": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_invalid_area_type(client):
    payload = {
        "area_type": "Apartment Area",
        "location": "Whitefield",
        "bedrooms": 2,
        "is_rk": False,
        "total_sqft": 1200,
        "bath": 2,
        "balcony": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_negative_total_sqft(client):
    payload = {
        "area_type": "Built-up Area",
        "location": "Whitefield",
        "bedrooms": 2,
        "is_rk": False,
        "total_sqft": -500,
        "bath": 2,
        "balcony": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_rejects_extra_fields(client):
    payload = {
        "area_type": "Built-up Area",
        "location": "Whitefield",
        "bedrooms": 2,
        "is_rk": False,
        "total_sqft": 1200,
        "bath": 2,
        "balcony": 1,
        "city": "Bengaluru"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_unsupported_location(client):
    payload = {
        "area_type": "Built-up Area",
        "location": "Unknown Location",
        "bedrooms": 2,
        "is_rk": False,
        "total_sqft": 1200,
        "bath": 2,
        "balcony": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "Unsupported location" in response.json()["detail"]


def test_prediction_internal_error(client, monkeypatch):
    # Simulate an unexpected failure inside the model.
    monkeypatch.setattr(
        main.model_service.model,
        "predict",
        Mock(side_effect=RuntimeError("Simulated model failure"))
    )

    payload = {
        "area_type": "Built-up Area",
        "location": "Whitefield",
        "bedrooms": 2,
        "is_rk": False,
        "total_sqft": 1200,
        "bath": 2,
        "balcony": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Prediction failed due to an internal server error."
    }

def test_health_model_unavailable(client, monkeypatch):
    # Simulate a model that is unavailable.
    monkeypatch.setattr(main.model_service, "model", None)

    response = client.get("/health")

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Prediction service is unavailable."
    }