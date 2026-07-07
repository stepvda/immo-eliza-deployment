"""
predict.py
==========

Load the saved models and predict prices for the **dummy** properties in
``data/dummy`` (10 for-sale + 10 to-rent listings that the models have never
seen, and which deliberately contain no price).

Because every model is a self-contained pipeline (preprocessor + estimator),
prediction is simply ``pipeline.predict(raw_features)`` -- the exact same
cleaning/imputation/encoding/scaling learned at training time is reapplied
automatically. The dummy CSVs only need to provide the raw feature columns.

For each market the script prints, side by side:

* the estimate from every **untuned** (default) model,
* the estimate from every **tuned** model, and
* a head-line estimate from the best overall model (tuned XGBoost) next to the
  full property description.

Run ``python src/train_models.py`` (and ``tune_models.py``) first so the model
files exist.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd

from create_models import ALGORITHMS, model_filename
from preprocessing import FEATURE_COLUMNS, PROJECT_ROOT

DUMMY_PATHS = {
    "sale": os.path.join(PROJECT_ROOT, "data", "dummy", "dummy_sale_properties.csv"),
    "rent": os.path.join(PROJECT_ROOT, "data", "dummy", "dummy_rent_properties.csv"),
}
STAGE_DIRS = {
    "trained": os.path.join(PROJECT_ROOT, "models", "1_trained"),
    "tuned": os.path.join(PROJECT_ROOT, "models", "2_tuned"),
}
HEADLINE_ALGO = "xgboost"  # best overall model -> used for the head-line estimate
ALGO_LABELS = {
    "linear_regression": "Linear",
    "decision_tree": "Tree",
    "random_forest": "Forest",
    "xgboost": "XGBoost",
}


def load_models(stage: str, market: str) -> dict:
    """Return {algorithm: fitted_pipeline} for one stage/market (missing skipped)."""
    models = {}
    for algorithm in ALGORITHMS:
        path = os.path.join(STAGE_DIRS[stage], model_filename(algorithm, market))
        if os.path.exists(path):
            models[algorithm] = joblib.load(path)
    return models


def predict_stage(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Predict with every model in ``models``; return DataFrame (algos as cols)."""
    return pd.DataFrame({algo: m.predict(X) for algo, m in models.items()})


def _money(value: float, market: str) -> str:
    suffix = "/mo" if market == "rent" else ""
    return f"EUR {value:>10,.0f}{suffix}"


def _short(text: str, width: int = 42) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _print_table(title: str, descriptions, preds: pd.DataFrame, market: str) -> None:
    print(f"\n  {title}")
    cols = [c for c in ALGORITHMS if c in preds.columns]
    header = f"  {'#':>2}  {'property':<44}" + "".join(
        f"{ALGO_LABELS[c]:>12}" for c in cols
    ) + f"{'avg':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, desc in enumerate(descriptions):
        row = preds.iloc[i]
        avg = row[cols].mean()
        cells = "".join(f"{row[c]:>12,.0f}" for c in cols)
        print(f"  {i + 1:>2}  {_short(desc):<44}{cells}{avg:>12,.0f}")


def run_market(market: str) -> None:
    df = pd.read_csv(DUMMY_PATHS[market])
    descriptions = df["description"].tolist()
    X = df[FEATURE_COLUMNS]

    print("\n" + "=" * 96)
    print(f"  {market.upper()} PROPERTIES  -- estimated "
          f"{'monthly rent' if market == 'rent' else 'sale price'} "
          f"(EUR), {len(df)} dummy listings")
    print("=" * 96)

    trained = load_models("trained", market)
    tuned = load_models("tuned", market)

    trained_preds = predict_stage(trained, X)
    _print_table("UNTUNED (default) model estimates:", descriptions,
                 trained_preds, market)

    if tuned:
        tuned_preds = predict_stage(tuned, X)
        _print_table("TUNED model estimates:", descriptions, tuned_preds, market)
    else:
        tuned_preds = None
        print("\n  (no tuned models found -- run src/tune_models.py)")

    # Head-line estimate from the best model (tuned XGBoost, falling back to
    # trained XGBoost) shown next to the full description.
    best = None
    if tuned_preds is not None and HEADLINE_ALGO in tuned_preds:
        best, best_label = tuned_preds[HEADLINE_ALGO], "tuned XGBoost"
    elif HEADLINE_ALGO in trained_preds:
        best, best_label = trained_preds[HEADLINE_ALGO], "trained XGBoost"

    if best is not None:
        print(f"\n  HEAD-LINE ESTIMATE  ({best_label}, best overall model):")
        print("  " + "-" * 70)
        for i, desc in enumerate(descriptions):
            print(f"  {i + 1:>2}  {desc:<52} {_money(best.iloc[i], market)}")


def main() -> None:
    print("=" * 96)
    print("PREDICT  -- price estimates for the dummy properties (unseen by the models)")
    print("=" * 96)
    for market in ("sale", "rent"):
        run_market(market)
    print()


if __name__ == "__main__":
    main()
