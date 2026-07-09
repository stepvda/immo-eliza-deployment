"""
predict.py
==========

Prediction engine for the Immo Eliza deployment.

Loads the best model from last week's training project — the **tuned XGBoost**
pipeline — once per market (``sale`` and ``rent``) and exposes a plain
``predict()`` function (not a CLI) that turns a single raw property dict, or a
list of them, into a price estimate.

Because each artifact is a full scikit-learn ``Pipeline`` (shared preprocessor +
``XGBRegressor``), scoring a new property is literally ``pipeline.predict(X)`` —
the exact cleaning/imputation/one-hot/scaling learned at training time is
reapplied automatically. This module only has to:

1. build a dataframe with the 29 expected feature columns (missing ones filled
   with sensible defaults, extra ones ignored),
2. auto-fill geography (region + lat/long from province) when omitted,
3. call the pipeline, and
4. attach a rough +/- confidence band derived from the model's held-out MAE.

The models were trained with scikit-learn 1.8.0 / xgboost 3.3.0; the pinned
``requirements.txt`` reproduces that environment so the pickles load cleanly.
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any, Iterable

import joblib
import pandas as pd

from features import (
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    MARKETS,
    PROVINCE_CENTROIDS,
    default_property,
    region_for_province,
)

# Repo root on the path so we can import the shared ``geo`` package (priciness
# surface) without depending on the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --------------------------------------------------------------------------- #
# Paths & model registry
# --------------------------------------------------------------------------- #
API_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("IMMO_MODELS_DIR", os.path.join(API_DIR, "models"))

# We ship only the best overall model (tuned XGBoost) for each market — small
# (<3 MB each), fast, and the top performer on the held-out test set.
MODEL_FILES: dict[str, str] = {
    "sale": "pricing_xgboost_sale.joblib",
    "rent": "pricing_xgboost_rent.joblib",
}

# Head-line metrics of the shipped models on the held-out test set (from
# immo-eliza-ml/models/evaluation_results.csv). Used to build a plausible
# confidence band around each point estimate and to surface model quality
# through the API. MAE = typical euro miss; R2 = share of variance explained.
# Head-line tuned-XGBoost metrics on the held-out test set, for the 30-feature
# pipeline that now includes the neighbourhood-priciness feature (from
# ml/models/evaluation_results.csv). Accuracy is on par with the pre-priciness
# model; the feature's payoff is exact-address pricing, the heatmap and SHAP.
MODEL_METRICS: dict[str, dict[str, float]] = {
    "sale": {"r2": 0.8108, "mae": 81854.9, "rmse": 185185.8},
    "rent": {"r2": 0.6215, "mae": 255.49,  "rmse": 663.66},
}

ALGORITHM = "XGBoost (tuned)"


class MarketError(ValueError):
    """Raised when an unknown market is requested."""


@functools.lru_cache(maxsize=None)
def load_model(market: str):
    """Load and cache the fitted pipeline for a market ('sale' or 'rent')."""
    if market not in MARKETS:
        raise MarketError(
            f"Unknown market {market!r}. Expected one of {list(MARKETS)}."
        )
    path = os.path.join(MODELS_DIR, MODEL_FILES[market])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found: {path}. "
            "Ensure api/models/*.joblib are present."
        )
    return joblib.load(path)


def warm_up() -> None:
    """Eagerly load every market's model (called at API start-up)."""
    for market in MARKETS:
        load_model(market)


# --------------------------------------------------------------------------- #
# Preprocessing helpers (light — the heavy lifting is inside the pipeline)
# --------------------------------------------------------------------------- #
def _fill_geo(record: dict[str, Any]) -> dict[str, Any]:
    """Fill region + lat/long from the province centroid when they are missing."""
    province = record.get("province")
    centroid = PROVINCE_CENTROIDS.get(province)
    if centroid:
        record.setdefault("region", centroid["region"])
        if record.get("latitude") in (None, ""):
            record["latitude"] = centroid["lat"]
        if record.get("longitude") in (None, ""):
            record["longitude"] = centroid["lon"]
    # Final safety net: keep region consistent with province if still unset.
    if record.get("region") in (None, "") and province:
        record["region"] = region_for_province(province)
    return record


@functools.lru_cache(maxsize=None)
def _load_surface(market: str):
    """Load the priciness surface for a market (None if artifacts are absent)."""
    try:
        from geo import priciness
        return priciness.load(market)
    except Exception:  # noqa: BLE001 - surface is optional; fall back to defaults
        return None


def _fill_priciness(record: dict[str, Any], market: str, user_coords: bool) -> float:
    """Neighbourhood priciness percentile for a record.

    * exact address (user-supplied lat/lon) -> read the surface at that point,
    * province only (lat/lon are province-centroid fills) -> the province's
      median percentile (a fairer, coarser value than the exact centroid pixel),
    * no surface / no province -> the record's existing value (defaults to 50).
    """
    surface = _load_surface(market)
    current = record.get("neighbourhood_price_index", 50.0)
    if surface is None:
        return float(current if current is not None else 50.0)
    lat, lon = record.get("latitude"), record.get("longitude")
    if user_coords and lat is not None and lon is not None:
        return float(surface.lookup(lat, lon)["percentile"])
    pm = surface.prov_median.get(record.get("province"))
    if pm is not None:
        return float(surface.percentile_of(pm))
    return float(current if current is not None else 50.0)


def preprocess(records: Iterable[dict[str, Any]], market: str = "sale") -> pd.DataFrame:
    """Turn raw property dicts into the feature dataframe the pipeline expects.

    * missing features are filled from :func:`features.default_property`,
    * geography (region/lat/long) is auto-completed from the province,
    * ``neighbourhood_price_index`` is read off the priciness surface from the
      exact address (or the province median when only a province is given),
    * amenity flags are coerced to clean 0/1 integers,
    * columns are returned in the exact order the pipeline was fitted on.

    Everything else (median imputation, one-hot encoding, standardisation) is
    handled *inside* the pickled pipeline, so this stays deliberately thin.
    """
    defaults = default_property()
    rows: list[dict[str, Any]] = []
    for raw in records:
        user_coords = (raw.get("latitude") not in (None, "")
                       and raw.get("longitude") not in (None, ""))
        user_index = raw.get("neighbourhood_price_index") not in (None, "")
        record = {**defaults, **{k: v for k, v in raw.items() if v is not None}}
        record = _fill_geo(record)
        for flag in BINARY_FEATURES:
            record[flag] = int(bool(record.get(flag, 0)))
        if not user_index:
            record["neighbourhood_price_index"] = _fill_priciness(record, market, user_coords)
        rows.append({col: record.get(col) for col in FEATURE_COLUMNS})
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _band(prediction: float, mae: float) -> dict[str, float]:
    """A simple, honest +/- band: the model's typical miss (MAE), floored at 0."""
    low = max(0.0, prediction - mae)
    high = prediction + mae
    return {"low": round(low, 2), "high": round(high, 2)}


def predict_one(features: dict[str, Any], market: str = "sale") -> dict[str, Any]:
    """Predict the price of a **single** property.

    Parameters
    ----------
    features : dict
        Raw property attributes. Any subset of the 29 model features is accepted;
        missing ones fall back to sensible defaults. ``province`` is enough to
        auto-fill region and coordinates.
    market : {'sale', 'rent'}
        Which model to use. ``sale`` returns a purchase price, ``rent`` a monthly
        rent — both in euros.

    Returns
    -------
    dict with keys ``prediction`` (float, EUR), ``market``, ``currency``,
    ``interval`` (low/high band), ``unit`` and model ``metrics``.
    """
    model = load_model(market)  # raises MarketError / FileNotFoundError
    X = preprocess([features], market)
    value = float(model.predict(X)[0])
    value = max(0.0, round(value, 2))
    metrics = MODEL_METRICS[market]
    return {
        "prediction": value,
        "market": market,
        "currency": "EUR",
        "unit": "per month" if market == "rent" else "total",
        "interval": _band(value, metrics["mae"]),
        "model": ALGORITHM,
        "metrics": metrics,
    }


def predict(
    features: dict[str, Any] | list[dict[str, Any]],
    market: str = "sale",
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict for one property (dict) or many (list of dicts).

    A thin dispatcher over :func:`predict_one`. Batches share a single vectorised
    ``pipeline.predict`` call for speed.
    """
    if isinstance(features, dict):
        return predict_one(features, market)

    model = load_model(market)
    X = preprocess(features, market)
    values = [max(0.0, round(float(v), 2)) for v in model.predict(X)]
    metrics = MODEL_METRICS[market]
    return [
        {
            "prediction": v,
            "market": market,
            "currency": "EUR",
            "unit": "per month" if market == "rent" else "total",
            "interval": _band(v, metrics["mae"]),
            "model": ALGORITHM,
            "metrics": metrics,
        }
        for v in values
    ]


if __name__ == "__main__":
    # Tiny smoke test so `python api/predict.py` shows it works end-to-end.
    demo = {
        "livable_surface": 85, "bedrooms": 2, "bathrooms": 1,
        "property_type": "flat", "province": "Brussels", "epc": "C",
        "building_state": "Normal", "terrace": 1, "elevator": 1,
    }
    for mkt in MARKETS:
        out = predict_one(demo, market=mkt)
        band = out["interval"]
        print(
            f"[{mkt:>4}] EUR {out['prediction']:>12,.0f} {out['unit']:<9} "
            f"(± band {band['low']:,.0f}–{band['high']:,.0f})  "
            f"model={out['model']}  test-R²={out['metrics']['r2']}"
        )
