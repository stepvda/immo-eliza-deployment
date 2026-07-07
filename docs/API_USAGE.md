# API usage guide

Full reference for the Immo Eliza prediction API. Interactive docs are always
available at `/docs` (Swagger UI) and `/redoc` on a running server.

## Routes

### `GET /`
Liveness probe. Returns the plain string `alive`. Used by Render's health check
and the container `HEALTHCHECK`.

### `GET /health`
```json
{
  "status": "ok",
  "uptime_seconds": 12.3,
  "api_version": "1.0.0",
  "models_loaded": { "sale": true, "rent": true },
  "model": "XGBoost (tuned)",
  "versions": { "scikit-learn": "1.8.0", "xgboost": "3.3.0" }
}
```

### `GET /schema`
Returns the complete input contract — every categorical option, numeric range,
default value, and province centroid. The Streamlit frontend builds its form
from this, so the UI can never drift from what the model expects.

### `GET /metrics`
Held-out test-set metrics for both shipped models (R², MAE, RMSE).

### `POST /predict?market=sale|rent`
Body: a `PropertyFeatures` object (all fields optional). Returns one prediction.

### `POST /predict/batch`
Body: `{ "market": "sale"|"rent", "properties": [ {…}, {…} ] }`. Returns a list
of predictions in the same order.

## The 29 features

| Group | Fields |
|---|---|
| **Numeric** (12) | `livable_surface`, `bedrooms`, `bathrooms`, `toilets`, `build_year`, `facades`, `number_of_floors`, `primary_energy_consumption`, `land_surface`, `latitude`, `longitude`, `nearest_city_distance_km` |
| **Categorical** (7) | `property_type`, `province`, `region`, `epc`, `building_state`, `kitchen_equipment`, `heating_type` |
| **Binary** (10) | `new_construction`, `furnished`, `terrace`, `garden`, `swimming_pool`, `elevator`, `cellar`, `solar_panels`, `air_conditioning`, `has_parking` |

### Allowed categorical values

- **property_type**: `house`, `flat`, `villa`, `flatStudio`, `duplex`, `penthouse`, `groundFloor`, `mansion`, `masterHouse`, `bungalow`, `loft`, `triplex`, `cottage`, `chalet`, `studentFlat`
- **province**: `Antwerp`, `Brussels`, `East Flanders`, `Flemish Brabant`, `Hainaut`, `Limburg`, `Liège`, `Luxembourg`, `Namur`, `Walloon Brabant`, `West Flanders`
- **region**: `Brussels`, `Flanders`, `Wallonia` *(auto-filled from province if omitted)*
- **epc**: `A++`, `A+`, `A`, `B`, `C`, `D`, `E`, `F`, `G`
- **building_state**: `New`, `Excellent`, `Fully renovated`, `Normal`, `To renovate`, `To be renovated`, `To restore`, `Under construction`, `To demolish`
- **kitchen_equipment**: `Super equipped`, `Fully equipped`, `Partially equipped`, `Not equipped`
- **heating_type**: `Gas`, `Electricity`, `Fuel oil`, `Wood`, `Solar energy`, `Hot air`, `Coal`

> Unknown categories don't crash prediction — the one-hot encoder was fitted
> with `handle_unknown="ignore"` — but sticking to these values gives the most
> reliable estimates.

## Python client example

```python
import requests

BASE = "http://localhost:8010"

resp = requests.post(
    f"{BASE}/predict",
    params={"market": "rent"},
    json={
        "livable_surface": 90,
        "bedrooms": 2,
        "property_type": "flat",
        "province": "Antwerp",
        "epc": "C",
        "terrace": True,
        "has_parking": True,
    },
    timeout=15,
)
resp.raise_for_status()
print(resp.json()["prediction"], "EUR/month")
```
