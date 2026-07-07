"""
train_models.py
===============

Train every untrained pipeline from ``models/0_empty`` on the cleaned training
data and save the fitted models to ``models/1_trained``.

For each market the matching ``<market>_train.csv`` produced by
``preprocessing.py`` is loaded, split into features ``X`` and target ``y``, and
each model pipeline is fitted end-to-end (the preprocessor learns its
imputation/encoding/scaling statistics from the training fold only, then the
estimator is fitted on the transformed features).

A quick in-sample R^2 is printed per model purely as a sanity check -- the
honest, out-of-sample evaluation is done separately in ``evaluate.py``.
"""

from __future__ import annotations

import os
import time

import joblib

from create_models import ALGORITHMS, COMPRESS, MARKETS, model_filename
from preprocessing import PROJECT_ROOT, load_split, split_xy

EMPTY_DIR = os.path.join(PROJECT_ROOT, "models", "0_empty")
TRAINED_DIR = os.path.join(PROJECT_ROOT, "models", "1_trained")


def train_one(algorithm: str, market: str, X, y):
    """Load the untrained pipeline, fit it, return (fitted_pipeline, seconds)."""
    untrained_path = os.path.join(EMPTY_DIR, model_filename(algorithm, market))
    pipeline = joblib.load(untrained_path)
    start = time.perf_counter()
    pipeline.fit(X, y)
    elapsed = time.perf_counter() - start
    return pipeline, elapsed


def main() -> None:
    os.makedirs(TRAINED_DIR, exist_ok=True)
    print("=" * 70)
    print("TRAIN MODELS  -- fitting on training data -> models/1_trained")
    print("=" * 70)

    for market in MARKETS:
        train_df = load_split(market, "train")
        X, y = split_xy(train_df)
        print(f"\n[{market}]  training rows: {len(X)}")
        for algorithm in ALGORITHMS:
            pipeline, elapsed = train_one(algorithm, market, X, y)
            in_sample_r2 = pipeline.score(X, y)
            out_path = os.path.join(TRAINED_DIR, model_filename(algorithm, market))
            joblib.dump(pipeline, out_path, compress=COMPRESS)
            print(
                f"  {algorithm:<18} fit in {elapsed:6.2f}s  "
                f"in-sample R2={in_sample_r2:6.3f}  -> {os.path.basename(out_path)}"
            )

    n = len(MARKETS) * len(ALGORITHMS)
    print(f"\nSaved {n} trained models to "
          f"{os.path.relpath(TRAINED_DIR, PROJECT_ROOT)}.")


if __name__ == "__main__":
    main()
