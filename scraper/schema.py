"""
scraper/schema.py
=================

The **canonical listing schema** — the single, source-agnostic shape every
collected property is normalised into, whether it comes from a live site adapter
(:mod:`scraper.sites`) or from seeding the existing cleaned dataset
(:mod:`scraper.seed`).

It is a *superset* of the 29-column model feature contract (``api/features.py``)
plus geography (address → refnis/lat/lon), provenance (source/url/scraped_at) and
a couple of derived fields (``price_per_sqm``). Downstream consumers:

* :mod:`scraper.dedup`      — cross-site de-duplication keys,
* :mod:`geo.priciness`      — the spatial price surface,
* ``api/similar.py``        — the comparables index,
* ``ml/src/preprocessing``  — training data (after projecting to the 29 features).

Canonical files live at ``data/listings/<market>.parquet`` (one per market), which
is what the priciness surface and comparables read. Raw per-source captures land
under ``data/listings/raw/<source>/<market>/`` before being merged.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LISTINGS_DIR = REPO_ROOT / "data" / "listings"
RAW_DIR = LISTINGS_DIR / "raw"

MARKETS = ("sale", "rent")


def market_path(market: str) -> Path:
    """Canonical merged parquet for a market (what priciness/comparables read)."""
    return LISTINGS_DIR / f"{market}.parquet"


# Provenance / identity ------------------------------------------------------
PROVENANCE = ["listing_id", "source", "url", "market", "scraped_at"]

# Address & geography --------------------------------------------------------
GEO = [
    "street", "house_number", "postal_code", "locality", "municipality",
    "refnis", "province", "region", "latitude", "longitude",
    "nearest_city", "nearest_city_distance_km",
]

# The 12 numeric model features (minus geo, which lives in GEO) --------------
NUMERIC = [
    "livable_surface", "land_surface", "bedrooms", "bathrooms", "toilets",
    "build_year", "facades", "number_of_floors", "primary_energy_consumption",
]

# Low-cardinality categoricals ----------------------------------------------
CATEGORICAL = [
    "property_type", "category", "epc", "building_state",
    "kitchen_equipment", "heating_type",
]

# Amenity 0/1 flags ----------------------------------------------------------
BINARY = [
    "new_construction", "furnished", "terrace", "garden", "swimming_pool",
    "elevator", "cellar", "solar_panels", "air_conditioning", "has_parking",
]

# Target & derived -----------------------------------------------------------
PRICE = ["price", "price_per_sqm"]

CANONICAL_COLUMNS: list[str] = PROVENANCE + GEO + NUMERIC + CATEGORICAL + BINARY + PRICE

# ``category`` is the coarse house/apartment split used by the ROI/priciness
# aggregations; ``property_type`` is the fine model category.
CATEGORY_VALUES = ("house", "apartment")


def empty_record() -> dict:
    """A canonical record with every column present and ``None``-valued."""
    return {col: None for col in CANONICAL_COLUMNS}


# --------------------------------------------------------------------------- #
# Dtype canonicalisation                                                       #
# --------------------------------------------------------------------------- #
# Text-ish columns must stay strings across every source, otherwise a scraped
# ``postal_code="1000"`` (str) collides with a seeded ``1000.0`` (float) and the
# concatenated column becomes an un-writable mixed ``object`` (pyarrow errors).
_STRING_COLS = [
    "listing_id", "source", "url", "market", "scraped_at",
    "street", "house_number", "postal_code", "locality", "municipality",
    "province", "region", "nearest_city",
    "property_type", "category", "epc", "building_state",
    "kitchen_equipment", "heating_type",
]
_INT_COLS = ["refnis", *BINARY]
_FLOAT_COLS = [
    "latitude", "longitude", "nearest_city_distance_km",
    *[c for c in NUMERIC], "price", "price_per_sqm",
]


def _clean_str(v) -> str | None:
    """Normalise a scalar to a clean string (int-valued floats lose the ``.0``)."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer():
            return str(int(v))
    s = str(v).strip()
    return s or None


def canonicalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a listings frame to the canonical columns + stable Arrow dtypes.

    Applied by every writer (seed, raw append, merge) so parquet files from
    different sources concatenate and de-duplicate cleanly.
    """
    df = df.reindex(columns=CANONICAL_COLUMNS)
    for c in _STRING_COLS:
        df[c] = df[c].map(_clean_str).astype("object")
    for c in _INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in _FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
