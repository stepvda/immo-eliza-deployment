"""
app.py
======

FastAPI backend for the Immo Eliza price-prediction service.

Routes
------
* ``GET  /``            -> liveness probe, returns ``"alive"``.
* ``GET  /health``      -> richer health payload (models loaded, versions).
* ``GET  /schema``      -> the full input contract (categories, ranges, defaults)
                           so any frontend can build a form from one call.
* ``GET  /metrics``     -> held-out test metrics of the shipped models.
* ``POST /predict``     -> price a single property (query/body ``market``).
* ``POST /predict/batch`` -> price many properties in one call.

The service loads the **tuned XGBoost** pipeline for each market once at
start-up (the best model from the training project) and reuses it for every
request. FastAPI autogenerates interactive docs at ``/docs`` and ``/redoc``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ConfigDict

import predict as engine
from features import (
    BINARY_META,
    CATEGORICAL_META,
    CATEGORY_OPTIONS,
    MARKETS,
    NUMERIC_META,
    PROVINCE_CENTROIDS,
)

API_VERSION = "1.0.0"
START_TIME = time.time()


# --------------------------------------------------------------------------- #
# Pydantic models (request / response schemas -> power the /docs page)
# --------------------------------------------------------------------------- #
class PropertyFeatures(BaseModel):
    """Raw features of a single property.

    Only ``livable_surface``, ``property_type`` and ``province`` really matter to
    get a sensible estimate — everything else has a default. Region and
    coordinates are auto-derived from the province when omitted.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "livable_surface": 85,
                "bedrooms": 2,
                "bathrooms": 1,
                "property_type": "flat",
                "province": "Brussels",
                "epc": "C",
                "building_state": "Normal",
                "kitchen_equipment": "Fully equipped",
                "heating_type": "Gas",
                "terrace": 1,
                "elevator": 1,
                "has_parking": 1,
            }
        },
    )

    # Numerics — all optional, all defaulted.
    livable_surface: Optional[float] = Field(None, ge=5, le=5000, description="Habitable floor area (m²).")
    bedrooms: Optional[int] = Field(None, ge=0, le=20, description="Number of bedrooms (0 = studio).")
    bathrooms: Optional[int] = Field(None, ge=0, le=15, description="Number of bathrooms.")
    toilets: Optional[int] = Field(None, ge=0, le=15, description="Number of separate toilets.")
    build_year: Optional[int] = Field(None, ge=1750, le=2031, description="Year of construction.")
    facades: Optional[int] = Field(None, ge=1, le=4, description="Number of facades.")
    number_of_floors: Optional[int] = Field(None, ge=1, le=20, description="Number of floors.")
    primary_energy_consumption: Optional[float] = Field(None, ge=0, le=2000, description="Primary energy use (kWh/m²/yr).")
    land_surface: Optional[float] = Field(None, ge=0, le=100000, description="Plot area (m²).")
    latitude: Optional[float] = Field(None, ge=49.0, le=52.0, description="Latitude (auto-filled from province).")
    longitude: Optional[float] = Field(None, ge=2.0, le=7.0, description="Longitude (auto-filled from province).")
    nearest_city_distance_km: Optional[float] = Field(None, ge=0, le=100, description="Distance to nearest city (km).")

    # Categoricals.
    property_type: Optional[str] = Field(None, description="Kind of dwelling.")
    province: Optional[str] = Field(None, description="Belgian province.")
    region: Optional[str] = Field(None, description="Belgian region (auto-filled).")
    epc: Optional[str] = Field(None, description="EPC energy label (A++ … G).")
    building_state: Optional[str] = Field(None, description="Overall building condition.")
    kitchen_equipment: Optional[str] = Field(None, description="Kitchen fit-out level.")
    heating_type: Optional[str] = Field(None, description="Primary heating system.")

    # Amenity flags (accept bool or 0/1).
    new_construction: Optional[bool] = Field(None, description="Brand-new build.")
    furnished: Optional[bool] = Field(None, description="Sold/let furnished.")
    terrace: Optional[bool] = Field(None, description="Has a terrace.")
    garden: Optional[bool] = Field(None, description="Has a garden.")
    swimming_pool: Optional[bool] = Field(None, description="Has a pool.")
    elevator: Optional[bool] = Field(None, description="Building has a lift.")
    cellar: Optional[bool] = Field(None, description="Has a cellar.")
    solar_panels: Optional[bool] = Field(None, description="Has solar panels.")
    air_conditioning: Optional[bool] = Field(None, description="Has air conditioning.")
    has_parking: Optional[bool] = Field(None, description="Has parking / garage.")


class Interval(BaseModel):
    low: float = Field(..., description="Lower bound (prediction − MAE).")
    high: float = Field(..., description="Upper bound (prediction + MAE).")


class Metrics(BaseModel):
    r2: float
    mae: float
    rmse: float


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Estimated price in EUR.")
    market: str
    currency: str = "EUR"
    unit: str = Field(..., description="'total' for sale, 'per month' for rent.")
    interval: Interval
    model: str
    metrics: Metrics
    status_code: int = 200


class BatchRequest(BaseModel):
    market: Literal["sale", "rent"] = "sale"
    properties: list[PropertyFeatures]


# --------------------------------------------------------------------------- #
# App + lifespan (warm the models at start-up)
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.warm_up()  # load both pipelines once before serving traffic
    yield


app = FastAPI(
    title="Immo Eliza — Price Prediction API",
    description=(
        "Predict **sale prices** and **monthly rents** for Belgian residential "
        "properties using a tuned XGBoost model. Interactive docs below; try "
        "`POST /predict` with `?market=sale` or `?market=rent`."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    contact={"name": "Immo Eliza", "url": "https://github.com/"},
    license_info={"name": "MIT"},
)

# Allow the Streamlit frontend (any origin in this learning project) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=PlainTextResponse, tags=["health"])
def root() -> str:
    """Liveness probe — returns the literal string ``alive``."""
    return "alive"


@app.get("/health", tags=["health"])
def health() -> dict[str, Any]:
    """Detailed health: uptime, which markets are loaded, library versions."""
    import sklearn
    import xgboost

    loaded = {}
    for market in MARKETS:
        try:
            engine.load_model(market)
            loaded[market] = True
        except Exception:  # pragma: no cover - defensive
            loaded[market] = False
    return {
        "status": "ok" if all(loaded.values()) else "degraded",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "api_version": API_VERSION,
        "models_loaded": loaded,
        "model": engine.ALGORITHM,
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }


@app.get("/schema", tags=["meta"])
def schema() -> dict[str, Any]:
    """Full input contract: category options, numeric ranges, defaults, geo.

    A frontend can build its entire form from this single call — the Streamlit
    app does exactly that, so the UI never drifts from the model's contract.
    """
    return {
        "markets": list(MARKETS),
        "numeric": NUMERIC_META,
        "categorical": {k: {**CATEGORICAL_META[k], "options": v} for k, v in CATEGORY_OPTIONS.items()},
        "binary": BINARY_META,
        "province_centroids": PROVINCE_CENTROIDS,
    }


@app.get("/metrics", tags=["meta"])
def metrics() -> dict[str, Any]:
    """Held-out test-set metrics of the shipped (tuned XGBoost) models."""
    return {"model": engine.ALGORITHM, "markets": engine.MODEL_METRICS}


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict_route(
    features: PropertyFeatures,
    market: Literal["sale", "rent"] = Query("sale", description="'sale' or 'rent'."),
) -> PredictionResponse:
    """Predict the price of a **single** property.

    Send the property as JSON in the body and choose the model with the
    ``market`` query parameter. Returns the estimate plus a ±MAE band and the
    model's test metrics.
    """
    try:
        result = engine.predict_one(features.model_dump(exclude_none=True), market=market)
    except engine.MarketError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected model error
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(**result, status_code=200)


@app.post("/predict/batch", tags=["predict"])
def predict_batch_route(request: BatchRequest) -> dict[str, Any]:
    """Predict prices for **many** properties in a single call."""
    if not request.properties:
        raise HTTPException(status_code=422, detail="`properties` must not be empty.")
    try:
        payload = [p.model_dump(exclude_none=True) for p in request.properties]
        results = engine.predict(payload, market=request.market)
    except engine.MarketError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return {"market": request.market, "count": len(results), "predictions": results}


@app.exception_handler(404)
async def not_found(request, exc):  # pragma: no cover - cosmetic
    return JSONResponse(
        status_code=404,
        content={"detail": "Not found. See /docs for available routes.", "status_code": 404},
    )


if __name__ == "__main__":
    import uvicorn

    import os

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8010)), reload=True)
