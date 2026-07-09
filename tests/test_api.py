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


# --------------------------------------------------------------------------- #
# Neighbourhood-aware routes
# --------------------------------------------------------------------------- #
_PROP = {"livable_surface": 85, "bedrooms": 2, "property_type": "flat",
         "province": "Brussels", "epc": "C", "latitude": 50.846, "longitude": 4.352}


def test_schema_exposes_priciness_feature(client):
    numeric = client.get("/schema").json()["numeric"]
    assert "neighbourhood_price_index" in numeric


def test_explain_route(client):
    r = client.post("/explain", params={"market": "sale", "top": 6}, json=_PROP)
    assert r.status_code == 200
    body = r.json()
    assert body["contributions"] and len(body["contributions"]) <= 6
    assert {"feature", "value_eur", "direction"} <= set(body["contributions"][0])


def test_similar_route(client):
    r = client.post("/similar", params={"market": "sale"}, json=_PROP)
    assert r.status_code == 200
    body = r.json()
    assert 1 <= body["count"] <= 5
    assert all(c["price"] > 0 for c in body["comparables"])


def test_priciness_point_and_tiles(client):
    p = client.get("/priciness", params={"lat": 50.846, "lon": 4.352, "market": "sale"}).json()
    assert 0 <= p["percentile"] <= 100 and p["price_per_sqm"] > 0
    t = client.get("/priciness/tiles", params={"market": "sale"}).json()
    assert t["count"] > 50 and t["national_price_per_sqm"] > 0


def test_invest_route(client):
    r = client.post("/invest", json={"purchase_price": 300000, "monthly_rent": 1200,
                                     "province": "Brussels", "region": "Brussels",
                                     "ptype": "apartment", "scenario": "hist"})
    assert r.status_code == 200
    body = r.json()
    assert body["gross_yield_pct"] > 0 and body["series"]
    assert "10" in body["milestones"] or 10 in body["milestones"]


def test_geocode_suggest_offline(client, monkeypatch):
    monkeypatch.setenv("IMMO_GEOCODE_OFFLINE", "1")
    r = client.get("/geocode/suggest", params={"q": "Veldstraat Gent"})
    assert r.status_code == 200
    assert r.json()["suggestions"] == []
