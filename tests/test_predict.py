"""Unit tests for the prediction engine (api/predict.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import predict as engine  # noqa: E402
from features import FEATURE_COLUMNS, MARKETS  # noqa: E402

SAMPLE = {
    "livable_surface": 120, "bedrooms": 3, "bathrooms": 1,
    "property_type": "house", "province": "Antwerp", "epc": "C",
    "building_state": "Normal", "garden": 1,
}


@pytest.mark.parametrize("market", MARKETS)
def test_predict_one_returns_positive_price(market):
    result = engine.predict_one(SAMPLE, market=market)
    assert result["market"] == market
    assert result["prediction"] > 0
    assert result["currency"] == "EUR"
    assert result["interval"]["low"] <= result["prediction"] <= result["interval"]["high"]
    assert result["unit"] == ("per month" if market == "rent" else "total")


def test_sale_price_larger_than_rent():
    sale = engine.predict_one(SAMPLE, market="sale")["prediction"]
    rent = engine.predict_one(SAMPLE, market="rent")["prediction"]
    # A sale price should dwarf a monthly rent for the same property.
    assert sale > rent * 50


def test_batch_matches_single():
    batch = engine.predict([SAMPLE, SAMPLE], market="sale")
    single = engine.predict_one(SAMPLE, market="sale")
    assert len(batch) == 2
    assert batch[0]["prediction"] == pytest.approx(single["prediction"])


def test_preprocess_produces_full_feature_matrix():
    X = engine.preprocess([{"province": "Brussels"}])
    assert list(X.columns) == FEATURE_COLUMNS
    # Region + coordinates auto-filled from the province centroid.
    assert X.loc[0, "region"] == "Brussels"
    assert X.loc[0, "latitude"] == pytest.approx(50.845, abs=0.01)


def test_bigger_house_costs_more():
    small = engine.predict_one({**SAMPLE, "livable_surface": 60}, market="sale")["prediction"]
    big = engine.predict_one({**SAMPLE, "livable_surface": 300}, market="sale")["prediction"]
    assert big > small


def test_unknown_market_raises():
    with pytest.raises(engine.MarketError):
        engine.predict_one(SAMPLE, market="lease")
