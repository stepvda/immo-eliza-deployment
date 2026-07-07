"""
evaluate.py
===========

Evaluate the saved models on the held-out **test** data and report, for every
model, a single head-line accuracy number plus supporting error metrics.

For a regression problem there is no "accuracy" in the classification sense, so
the head-line number is **R^2** (the coefficient of determination -- the share
of price variance the model explains, where 1.0 is perfect). Two error metrics
in euros are reported alongside it:

* **MAE**  -- mean absolute error (typical miss, in euros)
* **RMSE** -- root mean squared error (penalises big misses more)

The script evaluates two stages and prints them side by side so the effect of
hyper-parameter tuning is immediately visible:

* ``trained``  -> ``models/1_trained``   (default parameters)
* ``tuned``    -> ``models/2_tuned``      (best parameters from ``tune_models.py``)

The ``tuned`` column is simply skipped for any model that has not been tuned
yet, so the same script works before and after ``tune_models.py`` has run.

A per-model **train R^2** is also shown so over-fitting is easy to spot: a large
gap between train R^2 and test R^2 means the model memorised the training data.

Results are printed as a table and saved to ``models/evaluation_results.csv``.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from create_models import ALGORITHMS, MARKETS, model_filename
from preprocessing import PROJECT_ROOT, load_split, split_xy

STAGE_DIRS = {
    "trained": os.path.join(PROJECT_ROOT, "models", "1_trained"),
    "tuned": os.path.join(PROJECT_ROOT, "models", "2_tuned"),
}
RESULTS_PATH = os.path.join(PROJECT_ROOT, "models", "evaluation_results.csv")


def score_model(pipeline, X, y) -> dict:
    """Return R^2, MAE and RMSE of a fitted pipeline on (X, y)."""
    preds = pipeline.predict(X)
    return {
        "r2": r2_score(y, preds),
        "mae": mean_absolute_error(y, preds),
        "rmse": root_mean_squared_error(y, preds),
    }


def evaluate_all() -> pd.DataFrame:
    """Evaluate every available model on the test set; return a tidy DataFrame."""
    rows = []
    for market in MARKETS:
        X_train, y_train = split_xy(load_split(market, "train"))
        X_test, y_test = split_xy(load_split(market, "test"))

        for algorithm in ALGORITHMS:
            fname = model_filename(algorithm, market)
            row = {"market": market, "algorithm": algorithm}
            for stage, directory in STAGE_DIRS.items():
                path = os.path.join(directory, fname)
                if not os.path.exists(path):
                    continue
                pipeline = joblib.load(path)
                test_metrics = score_model(pipeline, X_test, y_test)
                train_r2 = r2_score(y_train, pipeline.predict(X_train))
                row[f"{stage}_test_r2"] = test_metrics["r2"]
                row[f"{stage}_test_mae"] = test_metrics["mae"]
                row[f"{stage}_test_rmse"] = test_metrics["rmse"]
                row[f"{stage}_train_r2"] = train_r2
            rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value, kind):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "   --   "
    if kind == "r2":
        return f"{value:7.3f}"
    return f"{value:>9,.0f}"


def print_report(results: pd.DataFrame) -> None:
    has_tuned = any(c.startswith("tuned_") for c in results.columns)
    for market in MARKETS:
        sub = results[results["market"] == market]
        print("\n" + "=" * 78)
        print(f"  {market.upper()} MODELS  -- evaluated on the held-out test set")
        print("=" * 78)
        header = f"  {'algorithm':<18}{'test R2':>9}{'train R2':>10}" \
                 f"{'MAE (EUR)':>12}{'RMSE (EUR)':>12}"
        if has_tuned:
            header += f"{'tuned R2':>10}{'tuned MAE':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        # rank by best available test R2 (tuned if present else trained)
        sub = sub.copy()
        sub["_rank"] = sub.apply(
            lambda r: r.get("tuned_test_r2") if pd.notna(r.get("tuned_test_r2"))
            else r.get("trained_test_r2"),
            axis=1,
        )
        for _, r in sub.sort_values("_rank", ascending=False).iterrows():
            line = (
                f"  {r['algorithm']:<18}"
                f"{_fmt(r.get('trained_test_r2'), 'r2')}"
                f"{_fmt(r.get('trained_train_r2'), 'r2')}"
                f"{_fmt(r.get('trained_test_mae'), 'eur')}"
                f"{_fmt(r.get('trained_test_rmse'), 'eur')}"
            )
            if has_tuned:
                line += (
                    f"{_fmt(r.get('tuned_test_r2'), 'r2')}"
                    f"{_fmt(r.get('tuned_test_mae'), 'eur')}"
                )
            print(line)


def main() -> None:
    print("=" * 78)
    print("EVALUATE  -- scoring saved models on the test data")
    print("=" * 78)
    results = evaluate_all()
    print_report(results)
    results.drop(columns=[c for c in results.columns if c == "_rank"],
                 errors="ignore").to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved full metrics table to "
          f"{os.path.relpath(RESULTS_PATH, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
