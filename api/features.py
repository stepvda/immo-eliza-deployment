"""
features.py
===========

Single source of truth for the **feature contract** of the Immo Eliza models.

The trained pipelines (``pricing_xgboost_<market>.joblib``) are self-contained
scikit-learn pipelines: ``preprocessor -> XGBRegressor``. The preprocessor was
fitted on exactly the 29 raw feature columns listed below, so at prediction time
we only need to hand it a dataframe with those columns — imputation, one-hot
encoding and standardisation are reapplied automatically inside the pipeline.

This module mirrors, verbatim, the feature definition from the training project
(``immo-eliza-ml/src/preprocessing.py``) and additionally records the *domain*
of every field (allowed categories, sane numeric bounds, human labels, units and
sensible defaults) so both the FastAPI backend and the Streamlit frontend can be
generated from one place and never drift out of sync with the model.

Nothing here re-implements the preprocessing — that lives inside the pickled
pipeline. This file only describes the *inputs* the pipeline expects.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# The 29-feature contract (identical to immo-eliza-ml/src/preprocessing.py)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES: list[str] = [
    "livable_surface",
    "bedrooms",
    "bathrooms",
    "toilets",
    "build_year",
    "facades",
    "number_of_floors",
    "primary_energy_consumption",
    "land_surface",
    "latitude",
    "longitude",
    "nearest_city_distance_km",
    "neighbourhood_price_index",
]

CATEGORICAL_FEATURES: list[str] = [
    "property_type",
    "province",
    "region",
    "epc",
    "building_state",
    "kitchen_equipment",
    "heating_type",
]

BINARY_FEATURES: list[str] = [
    "new_construction",
    "furnished",
    "terrace",
    "garden",
    "swimming_pool",
    "elevator",
    "cellar",
    "solar_panels",
    "air_conditioning",
    "has_parking",
]

# Order matters: the pipeline was fit on this column order.
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

MARKETS: tuple[str, ...] = ("sale", "rent")

# --------------------------------------------------------------------------- #
# Allowed category values (extracted from the cleaned training data)
# --------------------------------------------------------------------------- #
# The one-hot encoder was fitted with ``handle_unknown="ignore"``, so an unseen
# category will not crash prediction — it simply encodes as "all zeros" for that
# feature. Still, we constrain the API to the values the model actually learned
# from, which gives the most reliable estimates and self-documenting validation.
PROPERTY_TYPES: list[str] = [
    "house", "flat", "villa", "flatStudio", "duplex", "penthouse",
    "groundFloor", "mansion", "masterHouse", "bungalow", "loft", "triplex",
    "cottage", "chalet", "studentFlat",
]

PROVINCES: list[str] = [
    "Antwerp", "Brussels", "East Flanders", "Flemish Brabant", "Hainaut",
    "Limburg", "Liège", "Luxembourg", "Namur", "Walloon Brabant",
    "West Flanders",
]

REGIONS: list[str] = ["Brussels", "Flanders", "Wallonia"]

# EPC energy-performance certificate labels, best -> worst.
EPC_LABELS: list[str] = ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"]

BUILDING_STATES: list[str] = [
    "New", "Excellent", "Fully renovated", "Normal", "To renovate",
    "To be renovated", "To restore", "Under construction", "To demolish",
]

KITCHEN_EQUIPMENT: list[str] = [
    "Super equipped", "Fully equipped", "Partially equipped", "Not equipped",
]

HEATING_TYPES: list[str] = [
    "Gas", "Electricity", "Fuel oil", "Wood", "Solar energy", "Hot air", "Coal",
]

CATEGORY_OPTIONS: dict[str, list[str]] = {
    "property_type": PROPERTY_TYPES,
    "province": PROVINCES,
    "region": REGIONS,
    "epc": EPC_LABELS,
    "building_state": BUILDING_STATES,
    "kitchen_equipment": KITCHEN_EQUIPMENT,
    "heating_type": HEATING_TYPES,
}

# --------------------------------------------------------------------------- #
# Geography helpers — province centroids (used to auto-fill lat/long/region)
# --------------------------------------------------------------------------- #
# Approximate provincial centroids (from the training data) so a user only has
# to pick a province and the model still gets plausible coordinates. Users can
# override lat/long directly for a precise location.
PROVINCE_CENTROIDS: dict[str, dict[str, Any]] = {
    "Brussels":        {"lat": 50.845, "lon": 4.357, "region": "Brussels"},
    "Antwerp":         {"lat": 51.210, "lon": 4.410, "region": "Flanders"},
    "East Flanders":   {"lat": 51.020, "lon": 3.720, "region": "Flanders"},
    "West Flanders":   {"lat": 51.100, "lon": 3.150, "region": "Flanders"},
    "Flemish Brabant": {"lat": 50.880, "lon": 4.700, "region": "Flanders"},
    "Limburg":         {"lat": 50.930, "lon": 5.340, "region": "Flanders"},
    "Walloon Brabant": {"lat": 50.690, "lon": 4.450, "region": "Wallonia"},
    "Hainaut":         {"lat": 50.410, "lon": 4.010, "region": "Wallonia"},
    "Liège":           {"lat": 50.630, "lon": 5.570, "region": "Wallonia"},
    "Luxembourg":      {"lat": 49.910, "lon": 5.515, "region": "Wallonia"},
    "Namur":           {"lat": 50.465, "lon": 4.870, "region": "Wallonia"},
}


def region_for_province(province: str) -> str:
    """Return the Belgian region a province belongs to ('Flanders' default)."""
    return PROVINCE_CENTROIDS.get(province, {}).get("region", "Flanders")


# --------------------------------------------------------------------------- #
# Field metadata — used to auto-build the API schema and the Streamlit form
# --------------------------------------------------------------------------- #
# Each numeric field: (min, max, default, step, label, unit, help).
NUMERIC_META: dict[str, dict[str, Any]] = {
    "livable_surface":            {"min": 10,   "max": 1000, "default": 120,  "step": 5,   "label": "Livable surface",           "unit": "m²",  "help": "Net habitable floor area."},
    "bedrooms":                   {"min": 0,    "max": 15,   "default": 2,    "step": 1,   "label": "Bedrooms",                  "unit": "",    "help": "Number of bedrooms (0 = studio)."},
    "bathrooms":                  {"min": 0,    "max": 10,   "default": 1,    "step": 1,   "label": "Bathrooms",                 "unit": "",    "help": "Number of bathrooms."},
    "toilets":                    {"min": 0,    "max": 10,   "default": 1,    "step": 1,   "label": "Toilets",                   "unit": "",    "help": "Number of separate toilets."},
    "build_year":                 {"min": 1750, "max": 2031, "default": 1995, "step": 1,   "label": "Build year",                "unit": "",    "help": "Year of construction."},
    "facades":                    {"min": 1,    "max": 4,    "default": 2,    "step": 1,   "label": "Facades",                   "unit": "",    "help": "Number of building facades (2 = terraced, 4 = detached)."},
    "number_of_floors":           {"min": 1,    "max": 10,   "default": 2,    "step": 1,   "label": "Floors",                    "unit": "",    "help": "Number of floors."},
    "primary_energy_consumption": {"min": 0,    "max": 1500, "default": 250,  "step": 10,  "label": "Primary energy use",        "unit": "kWh/m²/yr", "help": "Primary energy consumption (lower = greener)."},
    "land_surface":               {"min": 0,    "max": 10000,"default": 0,    "step": 10,  "label": "Land / plot surface",       "unit": "m²",  "help": "Total plot area (0 for most apartments)."},
    "latitude":                   {"min": 49.4, "max": 51.6, "default": 50.85,"step": 0.001,"label": "Latitude",                 "unit": "°",   "help": "Auto-filled from province; override for a precise spot."},
    "longitude":                  {"min": 2.5,  "max": 6.5,  "default": 4.35, "step": 0.001,"label": "Longitude",                "unit": "°",   "help": "Auto-filled from province; override for a precise spot."},
    "nearest_city_distance_km":   {"min": 0,    "max": 60,   "default": 3,    "step": 1,   "label": "Distance to nearest city",  "unit": "km",  "help": "Distance to the nearest major city centre."},
    "neighbourhood_price_index":  {"min": 0,    "max": 100,  "default": 50,   "step": 1,   "label": "Neighbourhood priciness",   "unit": "pct", "help": "€/m² percentile of the exact location (0=cheapest, 100=priciest). Auto-filled from the address; defaults to 50 (national median) when only a province is given."},
}

# Each binary field: (default, label, icon, help).
BINARY_META: dict[str, dict[str, Any]] = {
    "new_construction": {"default": 0, "label": "New construction", "icon": "🏗️", "help": "Brand-new build."},
    "furnished":        {"default": 0, "label": "Furnished",        "icon": "🛋️", "help": "Sold/let furnished."},
    "terrace":          {"default": 0, "label": "Terrace",          "icon": "☀️", "help": "Has a terrace."},
    "garden":           {"default": 0, "label": "Garden",           "icon": "🌳", "help": "Has a garden."},
    "swimming_pool":    {"default": 0, "label": "Swimming pool",    "icon": "🏊", "help": "Has a pool."},
    "elevator":         {"default": 0, "label": "Elevator",         "icon": "🛗", "help": "Building has a lift."},
    "cellar":           {"default": 0, "label": "Cellar",           "icon": "🗄️", "help": "Has a cellar/basement."},
    "solar_panels":     {"default": 0, "label": "Solar panels",     "icon": "🔆", "help": "Has solar panels."},
    "air_conditioning": {"default": 0, "label": "Air conditioning", "icon": "❄️", "help": "Has air conditioning."},
    "has_parking":      {"default": 0, "label": "Parking",          "icon": "🅿️", "help": "Has a parking space/garage."},
}

CATEGORICAL_META: dict[str, dict[str, Any]] = {
    "property_type":     {"label": "Property type",     "help": "Kind of dwelling.",              "default": "house"},
    "province":          {"label": "Province",          "help": "Belgian province.",              "default": "Brussels"},
    "region":            {"label": "Region",            "help": "Auto-filled from province.",     "default": "Brussels"},
    "epc":               {"label": "EPC energy label",  "help": "Energy-performance certificate.","default": "C"},
    "building_state":    {"label": "Building condition", "help": "Overall state of the building.", "default": "Normal"},
    "kitchen_equipment": {"label": "Kitchen",           "help": "Kitchen fit-out level.",         "default": "Fully equipped"},
    "heating_type":      {"label": "Heating",           "help": "Primary heating system.",        "default": "Gas"},
}


def default_property() -> dict[str, Any]:
    """A fully-populated, plausible property dict (all 29 features present)."""
    values: dict[str, Any] = {}
    for name, meta in NUMERIC_META.items():
        values[name] = meta["default"]
    for name, meta in CATEGORICAL_META.items():
        values[name] = meta["default"]
    for name, meta in BINARY_META.items():
        values[name] = meta["default"]
    return values
