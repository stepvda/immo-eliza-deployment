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

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ConfigDict

# Repo root on the path so the API can import the shared cross-cutting packages
# (geo/ priciness+geocode, invest/ ROI) that live above api/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import predict as engine
import explain as explainer
import similar as comparables
from features import (
    BINARY_META,
    CATEGORICAL_META,
    CATEGORY_OPTIONS,
    MARKETS,
    NUMERIC_META,
    PROVINCE_CENTROIDS,
)

API_VERSION = "2.0.0"
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


class ResolveRequest(BaseModel):
    """Resolve a (partial) address to structured components + coordinates."""
    street: str = Field(..., description="Street name (optionally with house number).")
    city: Optional[str] = Field(None, description="City / municipality to disambiguate.")
    house_number: Optional[str] = Field(None, description="House number, if known.")


class InvestRequest(BaseModel):
    """Inputs for the ROI projection (Invest tab)."""
    purchase_price: float = Field(..., gt=0, description="Purchase price (EUR).")
    monthly_rent: float = Field(..., ge=0, description="Expected monthly rent (EUR).")
    refnis: Optional[int] = Field(None, description="Municipality NIS code (best signal).")
    province: Optional[str] = Field(None, description="Province (fallback for yield/growth).")
    region: Optional[str] = Field(None, description="Region (used for acquisition-cost rate).")
    ptype: Literal["house", "apartment"] = "house"
    scenario: Literal["hist", "cons", "base", "opt"] = "hist"
    horizons: list[int] = Field(default_factory=lambda: [5, 10, 15, 20])
    include_costs: bool = False


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
    """Held-out test-set metrics, training sizes and comparables-pool sizes."""
    pools, totals = {}, {}
    for market in MARKETS:
        try:
            pools[market] = comparables.pool_size(market)
            totals[market] = comparables.total_size(market)
        except Exception:  # noqa: BLE001 - counts are optional metadata
            pools[market] = totals.get(market)
    return {
        "model": engine.ALGORITHM,
        "markets": engine.MODEL_METRICS,
        "input_data_counts": totals,
        "train_counts": engine.TRAIN_COUNTS,
        "pool_sizes": pools,
    }


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


# --------------------------------------------------------------------------- #
# Explainability — why did the property get this price? (SHAP)
# --------------------------------------------------------------------------- #
@app.post("/explain", tags=["explain"])
def explain_route(
    features: PropertyFeatures,
    market: Literal["sale", "rent"] = Query("sale"),
    top: Optional[int] = Query(None, description="Keep only the N biggest drivers."),
) -> dict[str, Any]:
    """Per-feature € contributions to the price (SHAP), mapped to entered fields."""
    try:
        return explainer.explain_one(features.model_dump(exclude_none=True), market=market, top=top)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Explain failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Comparables — five similar nearby properties
# --------------------------------------------------------------------------- #
@app.post("/similar", tags=["explain"])
def similar_route(
    features: PropertyFeatures,
    market: Literal["sale", "rent"] = Query("sale"),
    k: int = Query(5, ge=1, le=25),
    prediction: Optional[float] = Query(None, description="Predicted price to bracket around."),
) -> dict[str, Any]:
    """Return up to ``k`` real comparable listings near the described property."""
    feats = features.model_dump(exclude_none=True)
    pred = prediction
    if pred is None:
        try:
            pred = engine.predict_one(feats, market=market)["prediction"]
        except Exception:  # noqa: BLE001 - comparables still work without it
            pred = None
    comps = comparables.similar_properties(feats, market=market, prediction=pred, k=k)
    return {"market": market, "count": len(comps), "prediction": pred, "comparables": comps}


# --------------------------------------------------------------------------- #
# Priciness surface — point lookup + heatmap tiles
# --------------------------------------------------------------------------- #
def _surface(market: str):
    from geo import priciness
    return priciness.load(market)


def _native(v):
    """Coerce a numpy scalar to a plain Python scalar (JSON-safe)."""
    return v.item() if hasattr(v, "item") else v


@app.get("/priciness", tags=["geo"])
def priciness_point(
    lat: float = Query(..., ge=49.0, le=52.0),
    lon: float = Query(..., ge=2.0, le=7.0),
    market: Literal["sale", "rent"] = Query("sale"),
) -> dict[str, Any]:
    """Neighbourhood priciness (€/m², percentile, confidence, granularity) at a point."""
    try:
        r = _surface(market).lookup(lat, lon)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"market": market, "lat": lat, "lon": lon,
            **{k: _native(v) for k, v in r.items()}}


@app.get("/priciness/tiles", tags=["geo"])
def priciness_tiles(market: Literal["sale", "rent"] = Query("sale")) -> dict[str, Any]:
    """Heatmap tiles (H3 hexagons) of €/m² for the whole country."""
    try:
        s = _surface(market)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"market": market, "national_price_per_sqm": round(s.national_median, 1),
            "count": len(s.tiles), "tiles": s.tiles}


# --------------------------------------------------------------------------- #
# Geocoding / address autocomplete (Geopunt + Photon)
# --------------------------------------------------------------------------- #
@app.get("/geocode/suggest", tags=["geo"])
def geocode_suggest(
    q: str = Query(..., min_length=2, description="Partial street/address text."),
    limit: int = Query(6, ge=1, le=15),
) -> dict[str, Any]:
    """Address autocomplete suggestions (Geopunt for FL/BXL, Photon nationwide)."""
    from geo import geocode
    return {"query": q, "suggestions": geocode.suggest(q, limit=limit)}


@app.post("/geocode/resolve", tags=["geo"])
def geocode_resolve(req: ResolveRequest) -> dict[str, Any]:
    """Resolve an address to structured components + coordinates (+ priciness)."""
    from geo import geocode
    resolved = geocode.resolve(req.street, city=req.city, house_number=req.house_number)
    if not resolved:
        raise HTTPException(status_code=404, detail="Address could not be resolved.")
    # Enrich with priciness for both markets when coordinates are available.
    lat, lon = resolved.get("latitude"), resolved.get("longitude")
    if lat is not None and lon is not None:
        pricey = {}
        for market in MARKETS:
            try:
                pricey[market] = {k: _native(v) for k, v in _surface(market).lookup(lat, lon).items()}
            except Exception:  # noqa: BLE001 - priciness is optional enrichment
                pass
        resolved["priciness"] = pricey
    return resolved


# --------------------------------------------------------------------------- #
# Invest — ROI projection (rental yield + capital appreciation)
# --------------------------------------------------------------------------- #
@app.post("/invest", tags=["invest"])
def invest_route(req: InvestRequest) -> dict[str, Any]:
    """Project ROI (rent-only and rent+appreciation) with break-even + milestones."""
    from invest import roi
    try:
        return roi.compute_roi(
            purchase_price=req.purchase_price,
            monthly_rent=req.monthly_rent,
            refnis=req.refnis,
            province=req.province,
            region=req.region,
            ptype=req.ptype,
            horizons=tuple(req.horizons),
            scenario=req.scenario,
            include_costs=req.include_costs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"ROI computation failed: {exc}") from exc


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
