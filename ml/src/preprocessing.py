"""
preprocessing.py
=================

Reusable data-preparation pipeline for the Immo Eliza price-prediction project.

This module is the single source of truth for:

1. Which raw columns become model features (``NUMERIC_FEATURES``,
   ``CATEGORICAL_FEATURES``, ``BINARY_FEATURES``).  The **same** feature set is
   used by every model and for both the *sale* and *rent* markets, so the dummy
   data and the prediction step can rely on it too.
2. How the raw scraped CSVs are *cleaned* (``clean_data``).
3. How the cleaned features are *transformed* into a numeric matrix that ML
   models can consume (``build_preprocessor``) -- imputation, one-hot encoding
   and standardisation, bundled in a single reusable scikit-learn transformer.

Running the module as a script (``python src/preprocessing.py``) cleans both
input datasets, performs a train/test split and writes the cleaned splits to
``data/training`` and ``data/test``.

The transformer returned by ``build_preprocessor`` is intentionally *not*
fitted here: it is embedded inside every model pipeline (see
``create_models.py``) so that imputation/encoding/scaling are learned only from
the training fold and applied identically at prediction time -- exactly the
"reusable pipeline" the project requires.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATHS = {
    "sale": os.path.join(PROJECT_ROOT, "data", "in", "cleaned_sale_properties.csv"),
    "rent": os.path.join(PROJECT_ROOT, "data", "in", "cleaned_rent_properties.csv"),
}
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "training")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")

TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------- #
# Feature definition  (identical for the sale and rent models)
# --------------------------------------------------------------------------- #
# Continuous / count features -> median imputation + standardisation.
NUMERIC_FEATURES = [
    "livable_surface",
    "bedrooms",
    "bathrooms",
    "toilets",
    "build_year",
    "facades",
    "number_of_floors",
    "primary_energy_consumption",
    "land_surface",
    "latitude",
    "longitude",
    "nearest_city_distance_km",
]

# Low-cardinality text features -> "missing" category + one-hot encoding.
CATEGORICAL_FEATURES = [
    "property_type",
    "province",
    "region",
    "epc",
    "building_state",
    "kitchen_equipment",
    "heating_type",
]

# Amenity flags. A missing value almost always means "not present / not
# advertised", so these are imputed with 0 and passed straight through.
BINARY_FEATURES = [
    "new_construction",
    "furnished",
    "terrace",
    "garden",
    "swimming_pool",
    "elevator",
    "cellar",
    "solar_panels",
    "air_conditioning",
    "has_parking",
]

# Order matters only for readability of the saved CSVs.
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Per-market sanity bounds for the target (euros). Anything outside is treated
# as a data-entry error and dropped; the 1st/99th percentiles additionally trim
# the extreme tail so the models are not dominated by a handful of outliers.
TARGET_BOUNDS = {
    "sale": {"floor": 25_000, "ceil": 5_000_000},
    "rent": {"floor": 200, "ceil": 12_000},
}


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _to_binary(series: pd.Series) -> pd.Series:
    """Coerce an amenity column to a clean 0/1 integer flag (NaN -> 0)."""
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(int)
    )


def clean_data(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Clean a raw properties dataframe and return only model columns + target.

    Steps
    -----
    * drop exact duplicates and duplicate ``property_id`` rows,
    * drop rows with a missing / implausible target price,
    * trim target outliers (domain bounds + 1st/99th percentiles),
    * null-out impossible feature values (0 m2 surfaces, negative energy
      scores, out-of-range build years, absurd room counts),
    * coerce amenity flags to clean 0/1 integers,
    * keep only the agreed feature columns plus the target.

    Imputation / encoding / scaling are deliberately **not** done here -- they
    live in the reusable preprocessor so they are fitted on the training fold
    only.
    """
    df = df.copy()

    # --- de-duplicate -----------------------------------------------------
    df = df.drop_duplicates()
    if "property_id" in df.columns:
        df = df.drop_duplicates(subset="property_id")

    # --- target -----------------------------------------------------------
    # Only CONSTANT domain bounds are applied here (they are hard-coded, not
    # data-derived, so they are leakage-free on every split). The percentile
    # outlier trim is applied later to the TRAINING split only -- see
    # `trim_target_outliers` / `process_market` -- so the held-out test set is
    # never filtered using statistics it should not have seen.
    bounds = TARGET_BOUNDS[market]
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()]
    df = df[(df[TARGET] >= bounds["floor"]) & (df[TARGET] <= bounds["ceil"])]

    # --- impossible feature values -> NaN (imputer will handle them) ------
    for col in ["livable_surface", "land_surface"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            df[col] = vals.where(vals > 0)

    if "primary_energy_consumption" in df.columns:
        pec = pd.to_numeric(df["primary_energy_consumption"], errors="coerce")
        df["primary_energy_consumption"] = pec.where((pec >= 0) & (pec <= 1500))

    if "build_year" in df.columns:
        by = pd.to_numeric(df["build_year"], errors="coerce")
        df["build_year"] = by.where((by >= 1750) & (by <= 2031))

    for col in ["bedrooms", "bathrooms", "toilets"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            df[col] = vals.where(vals <= 15)

    # --- amenity flags ----------------------------------------------------
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = _to_binary(df[col])

    # --- keep model columns + target -------------------------------------
    keep = [c for c in FEATURE_COLUMNS if c in df.columns] + [TARGET]
    df = df[keep].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Transformer (imputation + encoding + scaling)
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Return an **unfitted** ColumnTransformer for the agreed feature set.

    * numeric     -> median imputation + standardisation
    * categorical -> constant "missing" imputation + one-hot encoding
                     (``handle_unknown='ignore'`` so unseen dummy-data
                     categories don't break prediction)
    * binary      -> already 0/1 in ``clean_data``; passed through (with a
                     safety 0-imputer for any value that slips through).
    """
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary_pipe = Pipeline(
        steps=[("impute", SimpleImputer(strategy="constant", fill_value=0))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
        ],
        remainder="drop",
    )


# --------------------------------------------------------------------------- #
# Split + save
# --------------------------------------------------------------------------- #
def split_xy(df: pd.DataFrame):
    """Split a cleaned dataframe into feature matrix X and target vector y."""
    X = df[[c for c in FEATURE_COLUMNS if c in df.columns]]
    y = df[TARGET]
    return X, y


def trim_target_outliers(df: pd.DataFrame, low_q: float = 0.01,
                         high_q: float = 0.99) -> pd.DataFrame:
    """Drop the extreme price tails using percentiles of *this* dataframe.

    Applied to the TRAINING split only, so a handful of extreme-priced rows do
    not dominate the fit. The test split is deliberately left untrimmed (beyond
    the constant domain bounds) for an honest, leakage-free held-out evaluation.
    """
    low, high = df[TARGET].quantile([low_q, high_q])
    return df[(df[TARGET] >= low) & (df[TARGET] <= high)].reset_index(drop=True)


def process_market(market: str) -> dict:
    """Clean one market, split train/test, save the cleaned splits to disk."""
    raw = pd.read_csv(RAW_PATHS[market])
    cleaned = clean_data(raw, market)

    train_df, test_df = train_test_split(
        cleaned, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Outlier trim is fitted on the training split only (no test leakage).
    train_df = trim_target_outliers(train_df)
    test_df = test_df.reset_index(drop=True)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    train_path = os.path.join(TRAIN_DIR, f"{market}_train.csv")
    test_path = os.path.join(TEST_DIR, f"{market}_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "market": market,
        "raw_rows": len(raw),
        "clean_rows": len(cleaned),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_path": train_path,
        "test_path": test_path,
    }


def load_split(market: str, split: str) -> pd.DataFrame:
    """Load a previously saved cleaned split ('train' or 'test')."""
    directory = TRAIN_DIR if split == "train" else TEST_DIR
    return pd.read_csv(os.path.join(directory, f"{market}_{split}.csv"))


def main() -> None:
    print("=" * 70)
    print("PREPROCESSING  -- cleaning, splitting and saving the datasets")
    print("=" * 70)
    print(
        f"Feature set ({len(FEATURE_COLUMNS)} columns): "
        f"{len(NUMERIC_FEATURES)} numeric + "
        f"{len(CATEGORICAL_FEATURES)} categorical + "
        f"{len(BINARY_FEATURES)} binary\n"
    )
    for market in ("sale", "rent"):
        info = process_market(market)
        kept = 100 * info["clean_rows"] / info["raw_rows"]
        print(
            f"[{market:>4}] raw={info['raw_rows']:>6}  "
            f"clean={info['clean_rows']:>6} ({kept:4.1f}% kept)  "
            f"train={info['train_rows']:>6}  test={info['test_rows']:>6}"
        )
        print(f"        -> {os.path.relpath(info['train_path'], PROJECT_ROOT)}")
        print(f"        -> {os.path.relpath(info['test_path'], PROJECT_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
