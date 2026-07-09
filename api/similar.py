"""
api/similar.py
==============

"Five similar properties in the area" — the comparables that illustrate and
support a price estimate (and become the blue pins around the red query pin on
the map).

Given the property the user described + the model's predicted price, we search
the real listings store (``data/listings/<market>.parquet``, seeded from the
cleaned dataset and grown by the scraper) for the closest genuine comparables:

* **same market** and, where possible, the **same property category** (house /
  apartment) and type,
* **geographically near** the query address (haversine),
* **similar in size, bedrooms and price** — ranked by a small blended distance
  that also rewards listings whose price is close to the predicted one, so the
  five shown genuinely bracket the estimate.

No heavy persisted index is needed (≈13k/5k rows) — a cached in-memory frame is
loaded once per market and filtered per request.
"""
from __future__ import annotations

import functools
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from features import PROVINCE_CENTROIDS  # noqa: E402

LISTINGS_DIR = os.path.join(_REPO_ROOT, "data", "listings")
EARTH_RADIUS_KM = 6371.0

# Columns surfaced back to the caller / UI for each comparable.
OUT_COLS = [
    "listing_id", "source", "url", "price", "price_per_sqm", "livable_surface",
    "bedrooms", "property_type", "category", "locality", "municipality",
    "province", "latitude", "longitude", "epc", "building_state",
]


def pool_size(market: str) -> int:
    """Number of usable listings the comparables for ``market`` are drawn from
    (those with a price and coordinates)."""
    return int(len(_listings(market)))


@functools.lru_cache(maxsize=None)
def _total_size_at(market: str, _mtime_key: float) -> int:
    path = _market_parquet(market)
    if not os.path.exists(path):
        return 0
    return int(len(pd.read_parquet(path, columns=["price"])))


def total_size(market: str) -> int:
    """Total properties in the input data store for ``market`` (all rows)."""
    return _total_size_at(market, _mtime(market))


def _market_parquet(market: str) -> str:
    return os.path.join(LISTINGS_DIR, f"{market}.parquet")


def _mtime(market: str) -> float:
    """Modification time of a market's parquet (0 if absent). Used as a cache key
    so the in-memory listings invalidate automatically when the file changes —
    e.g. after a redeploy ships a newer dataset, so the app never serves a stale,
    url-less pool."""
    path = _market_parquet(market)
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@functools.lru_cache(maxsize=None)
def _listings_at(market: str, _mtime_key: float) -> pd.DataFrame:
    path = _market_parquet(market)
    if not os.path.exists(path):
        return pd.DataFrame(columns=OUT_COLS)
    df = pd.read_parquet(path)
    for c in ("price", "livable_surface", "bedrooms", "latitude", "longitude",
              "price_per_sqm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["price", "latitude", "longitude"]).reset_index(drop=True)


def _listings(market: str) -> pd.DataFrame:
    return _listings_at(market, _mtime(market))


def _haversine_km(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    latr, lonr = np.radians(lat), np.radians(lon)
    d = (np.sin((latr - lat0r) / 2) ** 2
         + np.cos(lat0r) * np.cos(latr) * np.sin((lonr - lon0r) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(d))


def _query_latlon(features: dict[str, Any]) -> tuple[float | None, float | None]:
    lat, lon = features.get("latitude"), features.get("longitude")
    if lat in (None, "") or lon in (None, ""):
        centroid = PROVINCE_CENTROIDS.get(features.get("province"), {})
        lat, lon = centroid.get("lat"), centroid.get("lon")
    return (float(lat) if lat not in (None, "") else None,
            float(lon) if lon not in (None, "") else None)


def similar_properties(
    features: dict[str, Any],
    market: str = "sale",
    prediction: float | None = None,
    k: int = 5,
    radius_km: float = 25.0,
) -> list[dict[str, Any]]:
    """Return up to ``k`` real comparable listings near the queried property."""
    df = _listings(market)
    if df.empty:
        return []

    lat0, lon0 = _query_latlon(features)
    if lat0 is None or lon0 is None:
        return []

    df = df.copy()
    # The same physical property can appear as both a seed and a scraped row
    # (they share a listing id / URL but differed on the store's blocking key), so
    # collapse by URL first — otherwise "5 similar" could show one property twice.
    # URL-less rows (if any) are kept as-is, never collapsed together.
    if "url" in df.columns:
        with_url = df[df["url"].notna()].drop_duplicates(subset="url")
        without_url = df[df["url"].isna()]
        df = pd.concat([with_url, without_url], ignore_index=True) if len(without_url) else with_url
    df["distance_km"] = _haversine_km(lat0, lon0, df["latitude"].to_numpy(),
                                      df["longitude"].to_numpy())

    # Progressive geographic filter: prefer nearby, but widen until we have a
    # healthy candidate pool so dense cities stay local and rural areas still fill.
    pool = df[df["distance_km"] <= radius_km]
    if len(pool) < k * 4:
        pool = df.nsmallest(max(k * 8, 40), "distance_km")

    # Soft-match category/type where the query specifies them.
    q_cat = features.get("category")
    q_type = features.get("property_type")
    if q_type and "property_type" in pool.columns:
        typed = pool[pool["property_type"] == q_type]
        pool = typed if len(typed) >= k else pool
    elif q_cat and "category" in pool.columns:
        catted = pool[pool["category"] == q_cat]
        pool = catted if len(catted) >= k else pool

    if pool.empty:
        return []

    # Blended similarity distance (lower = more similar). Each component is scaled
    # so no single axis dominates.
    surf0 = float(features.get("livable_surface") or pool["livable_surface"].median() or 100)
    beds0 = float(features.get("bedrooms") or pool["bedrooms"].median() or 2)
    surf = pool["livable_surface"].fillna(surf0).to_numpy()
    beds = pool["bedrooms"].fillna(beds0).to_numpy()

    d_geo = pool["distance_km"].to_numpy() / max(radius_km, 1.0)
    d_surf = np.abs(surf - surf0) / max(surf0, 1.0)
    d_beds = np.abs(beds - beds0) / 3.0
    d_price = np.zeros(len(pool))
    if prediction:
        d_price = np.abs(pool["price"].to_numpy() - prediction) / max(prediction, 1.0)

    pool = pool.assign(_score=0.9 * d_geo + 1.0 * d_surf + 0.5 * d_beds + 1.2 * d_price)
    top = pool.nsmallest(k, "_score")

    out = []
    for _, r in top.iterrows():
        rec = {c: (None if pd.isna(r.get(c)) else r.get(c)) for c in OUT_COLS}
        rec["distance_km"] = round(float(r["distance_km"]), 2)
        # JSON-friendly scalar types
        for c in ("price", "price_per_sqm", "livable_surface", "latitude", "longitude"):
            if rec.get(c) is not None:
                rec[c] = round(float(rec[c]), 4 if c in ("latitude", "longitude") else 1)
        if rec.get("bedrooms") is not None:
            rec["bedrooms"] = int(rec["bedrooms"])
        if rec.get("listing_id") is not None:
            rec["listing_id"] = str(rec["listing_id"])
        # Final safety net: coerce any lingering numpy scalars to native types.
        for c, v in list(rec.items()):
            if hasattr(v, "item"):
                rec[c] = v.item()
        out.append(rec)
    return out
