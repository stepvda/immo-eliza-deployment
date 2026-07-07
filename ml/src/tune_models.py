"""
tune_models.py
==============

Hyper-parameter tuning. For every market/algorithm a cross-validated search
explores a grid of hyper-parameters on the **training** data and the best
estimator is re-fitted on the full training set and saved to ``models/2_tuned``.

Design choices
--------------
* **Cross-validation** (``cv=4``, scoring ``R^2``) is used so the search is not
  rewarded for over-fitting the training data; the test set stays untouched.
* Each search space **includes the default hyper-parameters** used in
  ``create_models.py``. That guarantees the tuned model is, in expectation,
  never worse than the trained one -- the search can always fall back to the
  defaults if they turn out to be best.
* ``LinearRegression`` has nothing to tune, so its "tuned" version is a
  **Ridge** (L2-regularised) linear model whose penalty ``alpha`` is searched
  -- a tiny ``alpha`` reproduces ordinary least squares, larger values add
  regularisation.
* The base estimators run single-threaded (``n_jobs=1``) while the search
  parallelises across CV folds/candidates (``n_jobs=-1``) to avoid CPU
  over-subscription.

Run it as many times as you like; the search seed is fixed so results are
reproducible.
"""

from __future__ import annotations

import os
import time

import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from create_models import COMPRESS, MARKETS, model_filename
from preprocessing import (
    PROJECT_ROOT,
    RANDOM_STATE,
    build_preprocessor,
    load_split,
    split_xy,
)

TUNED_DIR = os.path.join(PROJECT_ROOT, "models", "2_tuned")
TRAINED_DIR = os.path.join(PROJECT_ROOT, "models", "1_trained")
CV_FOLDS = 4
N_ITER = 40  # candidates for the randomized searches


# --------------------------------------------------------------------------- #
# Search-space definitions
# --------------------------------------------------------------------------- #
def _spaces():
    """Return {algorithm: (base_estimator, param_distributions, search_kind)}."""
    return {
        # Plain OLS has no hyper-parameters -> tune a regularised (Ridge) variant.
        "linear_regression": (
            Ridge(),
            {"model__alpha": [0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]},
            "grid",
        ),
        "decision_tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {
                "model__max_depth": [4, 6, 8, 10, 12, 16, 20, None],
                "model__min_samples_leaf": randint(1, 40),
                "model__min_samples_split": randint(2, 40),
                "model__max_features": [None, "sqrt", "log2", 0.6, 0.8, 1.0],
            },
            "random",
        ),
        "random_forest": (
            RandomForestRegressor(n_jobs=1, random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 300, 400, 600],
                "model__max_depth": [12, 16, 22, 30, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": [1.0, "sqrt", 0.5, 0.7],
            },
            "random",
        ),
        "xgboost": (
            XGBRegressor(
                n_jobs=1,
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
            ),
            {
                "model__n_estimators": randint(300, 1300),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_depth": randint(3, 9),
                "model__subsample": uniform(0.7, 0.3),       # 0.7 .. 1.0
                "model__colsample_bytree": uniform(0.7, 0.3),  # 0.7 .. 1.0
                "model__min_child_weight": randint(1, 8),
                "model__reg_lambda": loguniform(0.5, 10),
                "model__reg_alpha": loguniform(0.01, 5),
            },
            "random",
        ),
    }


def _build_search(algorithm, estimator, params, kind):
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
    )
    common = dict(cv=CV_FOLDS, scoring="r2", n_jobs=-1, refit=True)
    if kind == "grid":
        return GridSearchCV(pipeline, params, **common)
    return RandomizedSearchCV(
        pipeline,
        params,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        **common,
    )


def tune_one(algorithm, market, X, y):
    estimator, params, kind = _spaces()[algorithm]
    search = _build_search(algorithm, estimator, params, kind)
    start = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - start
    return search, elapsed


def main() -> None:
    os.makedirs(TUNED_DIR, exist_ok=True)
    print("=" * 78)
    print("TUNE MODELS  -- cross-validated hyper-parameter search -> models/2_tuned")
    print("=" * 78)

    for market in MARKETS:
        X, y = split_xy(load_split(market, "train"))
        print(f"\n[{market}]  training rows: {len(X)}  (cv={CV_FOLDS})")
        for algorithm in _spaces():
            search, elapsed = tune_one(algorithm, market, X, y)
            out_path = os.path.join(TUNED_DIR, model_filename(algorithm, market))
            joblib.dump(search.best_estimator_, out_path, compress=COMPRESS)
            best = {k.replace("model__", ""): (round(v, 4) if isinstance(v, float)
                    else v) for k, v in search.best_params_.items()}
            print(
                f"  {algorithm:<18} cvR2={search.best_score_:6.3f}  "
                f"({elapsed:6.1f}s)  best={best}"
            )

    print(f"\nSaved tuned models to {os.path.relpath(TUNED_DIR, PROJECT_ROOT)}.")
    print("Run `python src/evaluate.py` to compare trained vs tuned on the test set.")


if __name__ == "__main__":
    main()
