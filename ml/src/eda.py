"""
eda.py
======

Exploratory data analysis + an **honest spatial-generalisation check** for the
enlarged, priciness-enriched dataset.

Two things beyond the head-line R² the training pipeline already reports:

1. **EDA summary** — rows/missingness, price and price/m² by province, and how
   strongly the new ``neighbourhood_price_index`` feature correlates with price.
2. **Spatial cross-validation** — random k-fold CV leaks location (nearby
   properties land in both train and test folds), so it *flatters* a model that
   leans on geography. We compare random-fold CV with **GroupKFold grouped by a
   coarse lat/lon grid** (whole neighbourhoods held out at once). The gap is the
   honest cost of generalising to *unseen* areas — and a check that the priciness
   feature is not just memorising locations.
3. **Feature importance** — XGBoost gain per raw feature, so we can see where the
   priciness feature ranks.

Run:
    cd ml && python src/eda.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from sklearn.model_selection import GroupKFold, cross_val_score  # noqa: E402

from create_models import build_model  # noqa: E402
from preprocessing import load_split, split_xy  # noqa: E402

MARKETS = ("sale", "rent")


def _spatial_groups(X: pd.DataFrame, precision: int = 1) -> np.ndarray:
    """Coarse lat/lon grid cell id per row (≈11 km at precision=1) for GroupKFold."""
    lat = X["latitude"].round(precision)
    lon = X["longitude"].round(precision)
    return pd.factorize(lat.astype(str) + "_" + lon.astype(str))[0]


def eda_summary(market: str) -> None:
    train = load_split(market, "train")
    test = load_split(market, "test")
    print(f"\n{'='*70}\n[{market.upper()}]  train={len(train):,}  test={len(test):,}")
    miss = train.isna().mean().sort_values(ascending=False)
    top_miss = miss[miss > 0].head(5)
    print("  missingness (top):", {k: f"{v:.1%}" for k, v in top_miss.items()} or "none")

    npi = train["neighbourhood_price_index"]
    print(f"  neighbourhood_price_index: range {npi.min():.0f}-{npi.max():.0f}  "
          f"corr(price)={npi.corr(train['price']):.3f}")

    prov = (train.assign(ppsqm=train["price"] / train["livable_surface"])
            .groupby("province")["ppsqm"].median().sort_values(ascending=False))
    unit = "€/m²/mo" if market == "rent" else "€/m²"
    print(f"  median {unit} by province (top 3 / bottom 3):")
    for name, v in list(prov.items())[:3] + list(prov.items())[-3:]:
        print(f"      {name:<18} {v:8.1f}")


def spatial_cv(market: str) -> None:
    X, y = split_xy(load_split(market, "train"))
    model = build_model("xgboost")

    random_r2 = cross_val_score(model, X, y, cv=4, scoring="r2", n_jobs=-1)
    groups = _spatial_groups(X)
    n_groups = len(np.unique(groups))
    spatial_r2 = cross_val_score(model, X, y, groups=groups,
                                 cv=GroupKFold(n_splits=4), scoring="r2", n_jobs=-1)
    print(f"\n[{market.upper()}] cross-validated R²  ({n_groups} spatial cells)")
    print(f"  random 4-fold CV   : {random_r2.mean():.3f} ± {random_r2.std():.3f}")
    print(f"  spatial GroupKFold : {spatial_r2.mean():.3f} ± {spatial_r2.std():.3f}"
          f"   (Δ {random_r2.mean() - spatial_r2.mean():+.3f} — cost of unseen areas)")


def feature_importance(market: str, top: int = 8) -> None:
    X, y = split_xy(load_split(market, "train"))
    model = build_model("xgboost").fit(X, y)
    pre = model.named_steps["preprocessor"]
    names = list(pre.get_feature_names_out())
    gains = model.named_steps["model"].feature_importances_

    # Aggregate one-hot columns back to their source feature.
    agg: dict[str, float] = {}
    for name, g in zip(names, gains):
        base = name.split("__", 1)[-1]
        for cat in ("property_type", "building_state", "kitchen_equipment",
                    "heating_type", "province", "region", "epc"):
            if base.startswith(cat):
                base = cat
                break
        agg[base] = agg.get(base, 0.0) + float(g)
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top]
    print(f"\n[{market.upper()}] top XGBoost feature importances (gain, aggregated):")
    for name, g in ranked:
        star = "  ⬅ priciness" if name == "neighbourhood_price_index" else ""
        print(f"      {name:<28} {g:6.3f}{star}")


def main() -> None:
    print("=" * 70)
    print("EDA + SPATIAL CROSS-VALIDATION")
    print("=" * 70)
    for market in MARKETS:
        eda_summary(market)
    for market in MARKETS:
        spatial_cv(market)
    for market in MARKETS:
        feature_importance(market)
    print("\nDone.")


if __name__ == "__main__":
    main()
