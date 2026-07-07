"""
create_models.py
================

Build the *untrained* model pipelines and save them to ``models/0_empty``.

Every model is a scikit-learn :class:`~sklearn.pipeline.Pipeline` that chains
the shared, reusable preprocessor (imputation + one-hot encoding +
standardisation, see :func:`preprocessing.build_preprocessor`) with a
regression estimator::

    raw cleaned features  ->  preprocessor  ->  estimator  ->  price

Because the preprocessor is part of the pipeline, every model consumes the
*exact same* feature set and the same transformation logic -- the only thing
that differs between models is the final estimator.

Four algorithm families are built, spanning the bias/variance spectrum
requested by the brief (a linear baseline, a single tree, a bagging ensemble
and a boosting ensemble):

    * ``linear_regression`` -- ordinary least squares (baseline)
    * ``decision_tree``     -- a single regression tree
    * ``random_forest``     -- bagged trees
    * ``xgboost``           -- gradient-boosted trees

Each algorithm is instantiated once per market (``sale`` and ``rent``), giving
``4 x 2 = 8`` model files named ``pricing_<algorithm>_<market>.joblib``.
"""

from __future__ import annotations

import os

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from preprocessing import PROJECT_ROOT, RANDOM_STATE, build_preprocessor

EMPTY_DIR = os.path.join(PROJECT_ROOT, "models", "0_empty")

# joblib compression level used when saving models. Keeps the (otherwise large)
# random-forest files well under GitHub's 100 MB limit so they can be published.
COMPRESS = 3

MARKETS = ("sale", "rent")

# Factory functions -> a fresh, unfitted estimator each call.
MODEL_BUILDERS = {
    "linear_regression": lambda: LinearRegression(),
    "decision_tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300,
        max_depth=22,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "xgboost": lambda: XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
    ),
}

ALGORITHMS = tuple(MODEL_BUILDERS.keys())


def model_filename(algorithm: str, market: str) -> str:
    """Canonical file name for a model, e.g. ``pricing_xgboost_sale.joblib``."""
    return f"pricing_{algorithm}_{market}.joblib"


def build_model(algorithm: str) -> Pipeline:
    """Create one untrained pipeline = shared preprocessor + estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", MODEL_BUILDERS[algorithm]()),
        ]
    )


def main() -> None:
    os.makedirs(EMPTY_DIR, exist_ok=True)
    print("=" * 70)
    print("CREATE MODELS  -- building untrained pipelines -> models/0_empty")
    print("=" * 70)
    for market in MARKETS:
        for algorithm in ALGORITHMS:
            pipeline = build_model(algorithm)
            path = os.path.join(EMPTY_DIR, model_filename(algorithm, market))
            joblib.dump(pipeline, path, compress=COMPRESS)
            print(f"  [{market:>4}] {algorithm:<18} -> {os.path.basename(path)}")
    n = len(MARKETS) * len(ALGORITHMS)
    print(f"\nSaved {n} untrained models to "
          f"{os.path.relpath(EMPTY_DIR, PROJECT_ROOT)}.")


if __name__ == "__main__":
    main()
