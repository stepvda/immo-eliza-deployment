"""
geo/build_reference.py
======================

One-off (idempotent) builder for the compact geographic reference artifacts the
app ships with. It distils the bulky external source ``nis_postal.csv`` (~31 MB,
one polygon *per postal code*) down to two small, version-controllable files
keyed by Belgian municipality (``refnis``):

* ``data/geo/municipality_centroids.csv`` — refnis → lat/lon + name/province/region
  (municipality centroid = mean of its postal-code centroids). Used to place map
  markers and to give an address-free province query a plausible coordinate.
* ``data/geo/municipalities.geojson`` — refnis → **simplified**, dissolved polygon
  (all postal polygons of a municipality unioned + Douglas-Peucker simplified) with
  ``{refnis, municipality, province, region}`` properties. Backs the choropleth
  layer of the priciness heatmap.

Source columns (``nis_postal.csv`` is ``;``-delimited):
    refnis_code ; postal_code ; nom_commune ; gemeentenaam ; centroid ; geom
    centroid = "lat, lon"  (decimal degrees, lat first)
    geom     = GeoJSON geometry string, coordinates in [lon, lat] order

Run once (or whenever the source changes):
    python -m geo.build_reference
    # or point at a non-default source:
    IMMO_NIS_POSTAL=/path/to/nis_postal.csv python -m geo.build_reference
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
GEO_DIR = DATA_DIR / "geo"
XWALK_CSV = DATA_DIR / "invest" / "nis_postal_crosswalk.csv"

# The heavy source lives in the sibling analysis repo by default; override with
# IMMO_NIS_POSTAL. It is intentionally *not* copied into this repo (31 MB).
DEFAULT_NIS_POSTAL = (
    REPO_ROOT.parent
    / "immo-eliza-teamname-analysis"
    / "data"
    / "external_raw"
    / "nis_postal.csv"
)
NIS_POSTAL_CSV = Path(os.environ.get("IMMO_NIS_POSTAL", str(DEFAULT_NIS_POSTAL)))

CENTROIDS_OUT = GEO_DIR / "municipality_centroids.csv"
GEOJSON_OUT = GEO_DIR / "municipalities.geojson"

# Douglas-Peucker tolerance in degrees (~0.0015° ≈ 120 m) — keeps the GeoJSON
# small (a few hundred KB) while preserving recognisable municipality shapes.
SIMPLIFY_TOLERANCE = 0.0012


def _muni_attributes() -> pd.DataFrame:
    """refnis → (municipality, province, region) from the crosswalk (one row/refnis)."""
    xw = pd.read_csv(XWALK_CSV)
    xw = xw.drop_duplicates("refnis")
    xw["municipality"] = xw["municipality_nl"].fillna(xw["municipality_fr"])
    return xw.set_index("refnis")[["municipality", "province", "region"]]


def build_centroids() -> pd.DataFrame:
    """Mean postal-code centroid per municipality, joined with names/province/region."""
    cg = pd.read_csv(NIS_POSTAL_CSV, sep=";", usecols=["refnis_code", "centroid"])
    latlon = cg["centroid"].str.split(",", expand=True)
    cg["lat"] = pd.to_numeric(latlon[0], errors="coerce")
    cg["lon"] = pd.to_numeric(latlon[1], errors="coerce")
    cg = cg.dropna(subset=["lat", "lon"])
    cent = (
        cg.groupby("refnis_code")[["lat", "lon"]]
        .mean()
        .rename_axis("refnis")
        .round(5)
    )
    out = cent.join(_muni_attributes(), how="left").reset_index()
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CENTROIDS_OUT, index=False)
    print(f"  wrote {CENTROIDS_OUT.relative_to(REPO_ROOT)}  ({len(out)} municipalities)")
    return out


def build_geojson() -> int:
    """Dissolve postal-code polygons to municipalities, simplify, write a FeatureCollection."""
    try:
        from shapely.geometry import mapping, shape
        from shapely.ops import unary_union
    except ImportError:  # pragma: no cover
        print("  ! shapely not installed — skipping municipality polygons "
              "(choropleth will fall back to centroid bubbles).")
        return 0

    attrs = _muni_attributes()
    df = pd.read_csv(NIS_POSTAL_CSV, sep=";", usecols=["refnis_code", "geom"])
    df = df.dropna(subset=["geom"])

    features = []
    for refnis, grp in df.groupby("refnis_code"):
        geoms = []
        for raw in grp["geom"]:
            try:
                geoms.append(shape(json.loads(raw)))
            except Exception:
                continue
        if not geoms:
            continue
        merged = unary_union(geoms).simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        if merged.is_empty:
            continue
        meta = attrs.loc[refnis] if refnis in attrs.index else None
        features.append({
            "type": "Feature",
            "properties": {
                "refnis": int(refnis),
                "municipality": None if meta is None else str(meta["municipality"]),
                "province": None if meta is None else str(meta["province"]),
                "region": None if meta is None else str(meta["region"]),
            },
            "geometry": mapping(merged),
        })

    fc = {"type": "FeatureCollection", "features": features}
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    GEOJSON_OUT.write_text(json.dumps(fc, separators=(",", ":")), encoding="utf-8")
    size_kb = GEOJSON_OUT.stat().st_size / 1024
    print(f"  wrote {GEOJSON_OUT.relative_to(REPO_ROOT)}  "
          f"({len(features)} municipalities, {size_kb:,.0f} KB)")
    return len(features)


def main() -> None:
    print("Building compact geo reference artifacts…")
    if not NIS_POSTAL_CSV.exists():
        raise SystemExit(
            f"Source not found: {NIS_POSTAL_CSV}\n"
            "Set IMMO_NIS_POSTAL to the nis_postal.csv path."
        )
    build_centroids()
    build_geojson()
    print("Done.")


if __name__ == "__main__":
    main()
