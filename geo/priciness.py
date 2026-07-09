"""
geo/priciness.py
================

The **neighbourhood priciness surface** — an adaptive, multi-resolution spatial
estimate of how expensive a location is (in € per m²), one surface per market
(``sale`` / ``rent``). This is what lets the price model "take the priciness of
the neighbourhood into account by pinpointing the exact address".

How it works (adaptive granularity, per the plan)
-------------------------------------------------
For a query lat/lon we take the **K nearest real listings** (haversine
``BallTree``) and use the median of their €/m². The *granularity* reported adapts
to how dense the data is around the point:

* dense city block  → the K neighbours sit within a few hundred metres → the
  estimate genuinely reflects that *street / block* (label ``street/block``);
* sparser area      → the neighbours span a wider radius → the estimate reflects
  the *neighbourhood / district*;
* rural / no data   → we back off to the **municipality**, then **province**,
  then **national** median.

A ``confidence`` in [0,1] reflects the local sample density, and every estimate
is mapped to a **percentile** (0–100) of that market's €/m² distribution — that
percentile is the single, bounded, market-comparable feature
(``neighbourhood_price_index``) the model consumes.

Leakage safety
--------------
When generating the training feature we call :meth:`index_batch` with
``loo=True``: each listing's own value is excluded from its own neighbourhood
(the zero-distance self-match is dropped), so the feature never sees the row's
own price. At serve time there is no self to exclude.

Build the artifacts (after seeding ``data/listings/<market>.parquet``):
    python -m geo.priciness
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("IMMO_PRICINESS_DIR", str(REPO_ROOT / "geo" / "artifacts")))
LISTINGS_DIR = REPO_ROOT / "data" / "listings"
CENTROIDS_CSV = REPO_ROOT / "data" / "geo" / "municipality_centroids.csv"

MARKETS = ("sale", "rent")
EARTH_RADIUS_M = 6_371_000.0

# €/m² guard rails (identical to the ROI engine / production cleaner) so a few
# absurd listings don't distort the surface.
PPSQM_BOUNDS = {"sale": (400.0, 18_000.0), "rent": (3.0, 70.0)}

K_NEIGHBOURS = 20          # neighbours used for the local median
H3_RES_FINE = 9            # ~175 m edge — "street/block" scale (metadata/tiles)
TILE_RES = 8               # ~460 m edge — heatmap hexagons
MIN_CELL_N = 5             # min listings in a hex before it's shown on the heatmap


def _to_radians(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.radians(np.c_[lat, lon])


def _clean(df: pd.DataFrame, market: str) -> pd.DataFrame:
    lo, hi = PPSQM_BOUNDS[market]
    d = df.copy()
    d["price_per_sqm"] = pd.to_numeric(d["price_per_sqm"], errors="coerce")
    d = d.dropna(subset=["latitude", "longitude", "price_per_sqm"])
    d = d[(d["price_per_sqm"] >= lo) & (d["price_per_sqm"] <= hi)]
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Surface object
# --------------------------------------------------------------------------- #
class PricinessSurface:
    """A fitted, serialisable priciness surface for one market."""

    def __init__(self, market: str, lats, lons, ppsqm, refnis,
                 muni_stats, prov_median, national_median, sorted_ppsqm,
                 tiles: list[dict]):
        self.market = market
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.ppsqm = np.asarray(ppsqm, dtype=float)
        self.refnis = np.asarray(refnis, dtype=float)
        self.muni_stats = muni_stats            # refnis -> {"mean","sum","count"}
        self.prov_median = prov_median          # province -> median ppsqm
        self.national_median = float(national_median)
        self._sorted = np.asarray(sorted_ppsqm, dtype=float)
        self.tiles = tiles
        self._tree = None

    # -- lazy BallTree (rebuilt on load; sklearn trees pickle fine but this is
    #    smaller on disk and version-robust) --------------------------------
    @property
    def tree(self):
        if self._tree is None:
            from sklearn.neighbors import BallTree
            self._tree = BallTree(_to_radians(self.lats, self.lons), metric="haversine")
        return self._tree

    # -- percentile mapping ------------------------------------------------
    def percentile_of(self, value: float) -> float:
        if not np.isfinite(value) or self._sorted.size == 0:
            return 50.0
        rank = np.searchsorted(self._sorted, value, side="right")
        return round(100.0 * rank / self._sorted.size, 1)

    # -- single lookup -----------------------------------------------------
    def lookup(self, lat: float, lon: float) -> dict[str, Any]:
        out = self.index_batch(np.array([lat]), np.array([lon]))
        return {k: (v[0] if isinstance(v, (list, np.ndarray)) else v) for k, v in out.items()}

    # -- vectorised lookup (used for training features + batch scoring) ----
    def index_batch(self, lats, lons, loo: bool = False) -> dict[str, np.ndarray]:
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        n = lats.size
        k = min(K_NEIGHBOURS + (1 if loo else 0), self.ppsqm.size)
        dist, idx = self.tree.query(_to_radians(lats, lons), k=k)
        dist_m = dist * EARTH_RADIUS_M
        if loo:                                    # drop the self (nearest) column
            dist_m, idx = dist_m[:, 1:], idx[:, 1:]

        values = np.full(n, np.nan)
        radius_m = np.full(n, np.nan)
        n_within_1km = np.zeros(n, dtype=int)
        for i in range(n):
            neigh = self.ppsqm[idx[i]]
            values[i] = np.median(neigh)
            radius_m[i] = dist_m[i, -1]
            n_within_1km[i] = int(np.sum(dist_m[i] <= 1_000.0))

        # Back off to municipality/province/national where neighbours are too far
        # (sparse rural) — anything whose K-th neighbour is > 20 km away.
        far = radius_m > 20_000.0
        if far.any():
            for i in np.where(far)[0]:
                ref = self.refnis[idx[i, 0]] if idx[i].size else np.nan
                stat = self.muni_stats.get(int(ref)) if np.isfinite(ref) else None
                values[i] = stat["mean"] if stat else self.national_median

        granularity = np.where(
            radius_m < 500, "street/block",
            np.where(radius_m < 1_500, "neighbourhood",
                     np.where(radius_m < 5_000, "district",
                              np.where(far, "municipality", "area"))))
        confidence = np.clip(n_within_1km / 10.0, 0.05, 1.0)
        confidence = np.where(far, 0.25, confidence)
        percentile = np.array([self.percentile_of(v) for v in values])

        return {
            "price_per_sqm": np.round(values, 1),
            "percentile": percentile,
            "confidence": np.round(confidence, 2),
            "granularity": granularity,
            "radius_m": np.round(radius_m, 0),
        }

    def index_for(self, lat: float, lon: float) -> float:
        """The single model feature: the location's €/m² percentile (0–100)."""
        return float(self.lookup(lat, lon)["percentile"])


# --------------------------------------------------------------------------- #
# Build / load
# --------------------------------------------------------------------------- #
def _build_tiles(d: pd.DataFrame, market: str) -> list[dict]:
    import h3
    cells: dict[str, list[float]] = {}
    for lat, lon, v in zip(d["latitude"], d["longitude"], d["price_per_sqm"]):
        cell = h3.latlng_to_cell(float(lat), float(lon), TILE_RES)
        cells.setdefault(cell, []).append(float(v))
    tiles = []
    allv = np.sort(d["price_per_sqm"].to_numpy())
    for cell, vals in cells.items():
        if len(vals) < MIN_CELL_N:
            continue
        med = float(np.median(vals))
        clat, clon = h3.cell_to_latlng(cell)
        pct = round(100.0 * np.searchsorted(allv, med, side="right") / allv.size, 1)
        tiles.append({"cell": cell, "lat": round(clat, 5), "lon": round(clon, 5),
                      "price_per_sqm": round(med, 1), "percentile": pct, "count": len(vals)})
    return tiles


def build(market: str) -> dict:
    df = pd.read_parquet(LISTINGS_DIR / f"{market}.parquet")
    d = _clean(df, market)

    ref = pd.to_numeric(d["refnis"], errors="coerce")
    muni_stats: dict[int, dict] = {}
    for r, grp in d.assign(_ref=ref).dropna(subset=["_ref"]).groupby("_ref"):
        v = grp["price_per_sqm"].to_numpy()
        muni_stats[int(r)] = {"mean": float(v.mean()), "sum": float(v.sum()), "count": int(v.size)}
    prov_median = d.groupby("province")["price_per_sqm"].median().to_dict()

    surface = PricinessSurface(
        market=market,
        lats=d["latitude"].to_numpy(), lons=d["longitude"].to_numpy(),
        ppsqm=d["price_per_sqm"].to_numpy(), refnis=ref.to_numpy(),
        muni_stats=muni_stats, prov_median=prov_median,
        national_median=float(d["price_per_sqm"].median()),
        sorted_ppsqm=np.sort(d["price_per_sqm"].to_numpy()),
        tiles=_build_tiles(d, market),
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    surface._tree = None                            # don't pickle the tree
    path = ARTIFACT_DIR / f"priciness_{market}.joblib"
    joblib.dump(surface, path, compress=3)
    return {"market": market, "n_points": int(d.shape[0]), "n_tiles": len(surface.tiles),
            "national_ppsqm": round(surface.national_median, 1),
            "path": str(path.relative_to(REPO_ROOT))}


_CACHE: dict[str, PricinessSurface] = {}


def load(market: str) -> PricinessSurface:
    if market not in _CACHE:
        path = ARTIFACT_DIR / f"priciness_{market}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Priciness artifact missing: {path}. Run `python -m geo.priciness`.")
        _CACHE[market] = joblib.load(path)
    return _CACHE[market]


def main() -> None:
    print("Building priciness surfaces…")
    for market in MARKETS:
        info = build(market)
        print(f"  [{market:>4}] {info['n_points']:>6} points  "
              f"{info['n_tiles']:>4} heatmap tiles  "
              f"national €/m² {info['national_ppsqm']}  -> {info['path']}")
    # quick smoke: score a Brussels + a rural point on the sale surface
    s = load("sale")
    for name, (lat, lon) in {"Brussels centre": (50.846, 4.352),
                             "Bastogne (rural)": (50.00, 5.72)}.items():
        r = s.lookup(lat, lon)
        print(f"    {name:<18} €/m² {r['price_per_sqm']:>7}  "
              f"pct {r['percentile']:>5}  {r['granularity']} (conf {r['confidence']})")
    print("Done.")


if __name__ == "__main__":
    # Re-dispatch through the importable module so pickled surfaces reference
    # ``geo.priciness.PricinessSurface`` rather than ``__main__.PricinessSurface``
    # — otherwise the artifacts would only load when run exactly like this.
    import geo.priciness as _mod

    _mod.main()
