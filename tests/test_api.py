"""Integration tests for the FastAPI app (api/app.py) via TestClient."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers lifespan -> warms the models
        yield c


def test_root_alive(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.text.strip('"') == "alive"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == {"sale": True, "rent": True}


def test_schema_contract(client):
    body = client.get("/schema").json()
    assert set(body["markets"]) == {"sale", "rent"}
    assert "property_type" in body["categorical"]
    assert "livable_surface" in body["numeric"]


@pytest.mark.parametrize("market", ["sale", "rent"])
def test_predict(client, market):
    payload = {
        "livable_surface": 100, "bedrooms": 2, "property_type": "flat",
        "province": "Brussels", "epc": "B",
    }
    r = client.post("/predict", params={"market": market}, json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] > 0
    assert body["market"] == market
    assert body["interval"]["low"] <= body["prediction"] <= body["interval"]["high"]


def test_predict_batch(client):
    payload = {
        "market": "rent",
        "properties": [
            {"livable_surface": 60, "property_type": "flat", "province": "Brussels"},
            {"livable_surface": 200, "property_type": "villa", "province": "Walloon Brabant"},
        ],
    }
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    # The bigger villa should rent for more than the small flat.
    assert body["predictions"][1]["prediction"] > body["predictions"][0]["prediction"]


def test_empty_batch_rejected(client):
    r = client.post("/predict/batch", json={"market": "sale", "properties": []})
    assert r.status_code == 422
