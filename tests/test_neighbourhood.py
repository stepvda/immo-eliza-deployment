"""
Unit tests for the neighbourhood-aware extensions: priciness surface, SHAP
explanations, comparables, ROI engine, geocoding (offline) and the scraper's
pure helpers. These exercise the local engine directly (no network, no API).
"""
import os
import sys

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "api")))

from features import FEATURE_COLUMNS, NUMERIC_FEATURES  # noqa: E402

BRUSSELS = {"livable_surface": 85, "bedrooms": 2, "bathrooms": 1, "property_type": "flat",
            "province": "Brussels", "epc": "C", "building_state": "Normal",
            "latitude": 50.846, "longitude": 4.352}


# --------------------------------------------------------------------------- #
# Feature contract
# --------------------------------------------------------------------------- #
def test_contract_has_priciness_feature():
    assert "neighbourhood_price_index" in NUMERIC_FEATURES
    assert "neighbourhood_price_index" in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 30


# --------------------------------------------------------------------------- #
# Priciness surface
# --------------------------------------------------------------------------- #
def test_priciness_lookup_shape_and_ordering():
    from geo import priciness
    s = pricey = priciness.load("sale")
    r = s.lookup(50.846, 4.352)                       # Brussels centre
    for key in ("price_per_sqm", "percentile", "confidence", "granularity"):
        assert key in r
    assert 0 <= float(r["percentile"]) <= 100
    assert r["price_per_sqm"] > 0
    # A pricey central address should out-rank a cheap rural one.
    rural = pricey.lookup(50.0, 5.7)
    assert float(r["percentile"]) >= float(rural["percentile"])


def test_priciness_tiles_exist():
    from geo import priciness
    tiles = priciness.load("sale").tiles
    assert len(tiles) > 50
    assert all({"cell", "lat", "lon", "percentile"} <= set(t) for t in tiles[:5])


# --------------------------------------------------------------------------- #
# SHAP explanations
# --------------------------------------------------------------------------- #
def test_explain_reconciles_and_includes_priciness():
    import explain
    out = explain.explain_one(BRUSSELS, market="sale")
    labels = {c["feature"] for c in out["contributions"]}
    assert "neighbourhood_price_index" in labels
    total = out["base_value"] + sum(c["value_eur"] for c in out["contributions"])
    # base + contributions reconciles to the reported prediction.
    assert out["prediction"] == pytest.approx(max(0.0, total), rel=0.02, abs=1.0)


# --------------------------------------------------------------------------- #
# Comparables
# --------------------------------------------------------------------------- #
def test_similar_returns_nearby_comparables():
    import predict, similar
    pred = predict.predict_one(BRUSSELS, "sale")["prediction"]
    comps = similar.similar_properties(BRUSSELS, market="sale", prediction=pred, k=5)
    assert 1 <= len(comps) <= 5
    assert all(c["price"] > 0 for c in comps)
    # The nearest comparable is genuinely near the Brussels query point.
    assert min(c["distance_km"] for c in comps) < 30


# --------------------------------------------------------------------------- #
# Invest ROI engine
# --------------------------------------------------------------------------- #
def test_roi_series_milestones_and_breakeven():
    from invest import roi
    out = roi.compute_roi(300_000, 1_200, province="Brussels", region="Brussels",
                          ptype="apartment", scenario="hist")
    assert out["gross_yield_pct"] > 0
    assert len(out["series"]) >= 20
    for h in (5, 10, 15, 20):
        m = out["milestones"].get(h) or out["milestones"].get(str(h))
        assert m and "roi_total_pct" in m
    # Total ROI (rent + appreciation) always dominates rent-only at every horizon.
    for row in out["series"]:
        assert row["roi_total_pct"] >= row["roi_rent_only_pct"] - 1e-6


def test_roi_rejects_bad_scenario():
    from invest import roi
    with pytest.raises(Exception):
        roi.compute_roi(300_000, 1_200, scenario="not-a-scenario")


# --------------------------------------------------------------------------- #
# Geocoding — offline determinism
# --------------------------------------------------------------------------- #
def test_geocode_offline_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("IMMO_GEOCODE_OFFLINE", "1")
    from geo import geocode
    assert geocode.suggest("Veldstraat Gent") == []
    assert geocode.resolve("Veldstraat", city="Gent") is None
    # The local postcode->province crosswalk still works fully offline.
    prov = geocode.PROVINCE_BY_POSTCODE(9000)
    assert prov and prov.get("province") == "East Flanders"


# --------------------------------------------------------------------------- #
# Scraper pure helpers
# --------------------------------------------------------------------------- #
def test_scraper_dedup_collapses_cross_site_duplicate():
    from scraper.dedup import dedupe
    from scraper.schema import empty_record

    def rec(source, url, **kw):
        r = empty_record(); r.update(source=source, url=url, market="sale", **kw); return r

    records = [
        rec("immoweb", "u1", postal_code="1000", street="Kerkstraat", house_number="1",
            livable_surface=85.0, bedrooms=2, price=300000.0,
            latitude=50.85, longitude=4.35, epc="B"),
        rec("realo", "u2", postal_code="1000", street="Kerkstraat", house_number="1",
            livable_surface=84.0, bedrooms=2, price=301000.0,
            latitude=50.8501, longitude=4.3501),
    ]
    kept, report = dedupe(records)
    assert len(kept) == 1
    assert report["duplicates_removed"] == 1


def test_scraper_normalize_vocabularies():
    from scraper import normalize
    assert normalize.to_binary_flag("yes") == 1
    assert normalize.to_binary_flag("no") == 0
    # A missing amenity is "unknown" (None), not asserted-absent — the model's
    # imputer decides how to fill it downstream.
    assert normalize.to_binary_flag(None) is None
    assert normalize.parse_surface("85,5 m²") == pytest.approx(85.5)
