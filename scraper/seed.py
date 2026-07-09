"""
scraper/seed.py
===============

Seed the canonical listings store from the project's existing cleaned datasets
(``ml/data/in/cleaned_{sale,rent}_properties.csv`` — the ~13.7k sale / ~5k rent
real Belgian listings the model was trained on). This gives every downstream
consumer (priciness surface, comparables, retraining) a populated
``data/listings/<market>.parquet`` **immediately**, before the live scraper
(:mod:`scraper.run`) has crawled anything.

As the real crawl accumulates fresh listings, ``scraper.store.merge_market``
folds them into the same canonical file and de-duplicates across sources; the
seed rows carry ``source="seed"`` so they are always distinguishable.

Run:
    python -m scraper.seed
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scraper.schema import CANONICAL_COLUMNS, LISTINGS_DIR, canonicalize_frame, market_path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEANED = {
    "sale": REPO_ROOT / "ml" / "data" / "in" / "cleaned_sale_properties.csv",
    "rent": REPO_ROOT / "ml" / "data" / "in" / "cleaned_rent_properties.csv",
}
XWALK = REPO_ROOT / "data" / "invest" / "nis_postal_crosswalk.csv"

# cleaned-CSV column -> canonical column (identical names omitted).
RENAME = {"property_id": "listing_id"}


def _crosswalk_maps():
    xw = pd.read_csv(XWALK)
    xw = xw.sort_values("refnis").drop_duplicates("postal_code")
    muni = xw["municipality_nl"].fillna(xw["municipality_fr"])
    pc = xw["postal_code"]
    return (
        dict(zip(pc, xw["refnis"])),
        dict(zip(pc, muni)),
        dict(zip(pc, xw["province"])),
        dict(zip(pc, xw["region"])),
    )


def seed_market(market: str) -> dict:
    df = pd.read_csv(CLEANED[market], low_memory=False)
    df = df.rename(columns=RENAME)

    df["source"] = "seed"
    df["url"] = None
    df["market"] = market
    df["scraped_at"] = None
    df["house_number"] = None

    # Fill refnis / municipality / province / region from the postal crosswalk
    # (authoritative), falling back to whatever the cleaned CSV already carries.
    pc = pd.to_numeric(df.get("postal_code"), errors="coerce")
    ref2, muni2, prov2, reg2 = _crosswalk_maps()
    df["refnis"] = pc.map(ref2)
    df["municipality"] = pc.map(muni2).fillna(df.get("locality"))
    df["province"] = df.get("province").where(df.get("province").notna(), pc.map(prov2))
    df["region"] = df.get("region").where(df.get("region").notna(), pc.map(reg2))

    # Derived price/m².
    surf = pd.to_numeric(df.get("livable_surface"), errors="coerce")
    price = pd.to_numeric(df.get("price"), errors="coerce")
    df["price_per_sqm"] = (price / surf).where(surf > 0)

    # Project onto the canonical schema (add missing cols) with stable dtypes.
    out = canonicalize_frame(df)

    LISTINGS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(market_path(market), index=False)
    return {
        "market": market,
        "rows": len(out),
        "with_refnis": int(out["refnis"].notna().sum()),
        "with_latlon": int(out["latitude"].notna().sum() & out["longitude"].notna().sum())
        if len(out) else 0,
        "path": str(market_path(market).relative_to(REPO_ROOT)),
    }


def main() -> None:
    print("Seeding canonical listings from the cleaned datasets…")
    for market in ("sale", "rent"):
        info = seed_market(market)
        print(f"  [{market:>4}] {info['rows']:>6} rows  "
              f"(refnis: {info['with_refnis']}, lat/lon: {info['with_latlon']})  "
              f"-> {info['path']}")
    print("Done.")


if __name__ == "__main__":
    main()
