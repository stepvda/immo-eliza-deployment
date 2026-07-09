"""
api/explain.py
==============

Explains *why* a property got its price — the per-feature contribution barchart.

Uses **SHAP** (``TreeExplainer``) on the gradient-boosted model inside the fitted
pipeline. Because the pipeline one-hot-encodes and scales the raw inputs, a single
entered field (e.g. ``province``) becomes many transformed columns; we map SHAP
values on the transformed columns **back to the original 29+ features** and sum
them, so the barchart speaks the user's language ("Province: +€40k", "EPC: −€8k")
rather than the model's ("province_Brussels: +0.3").

The signed values are in the model's target units (€ for sale, €/month for rent):
positive pushes the price up, negative pulls it down; they sum (with the base
value) to the prediction.
"""
from __future__ import annotations

import functools
import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from features import (  # noqa: E402
    BINARY_META,
    CATEGORICAL_FEATURES,
    CATEGORICAL_META,
    NUMERIC_META,
)
from predict import load_model, preprocess  # noqa: E402

# Human labels for every raw feature (numeric + categorical + binary).
_LABELS: dict[str, str] = {
    **{k: v["label"] for k, v in NUMERIC_META.items()},
    **{k: v["label"] for k, v in CATEGORICAL_META.items()},
    **{k: v["label"] for k, v in BINARY_META.items()},
}


def _origin_feature(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to its source raw feature.

    ``num__livable_surface`` -> ``livable_surface``
    ``bin__terrace``         -> ``terrace``
    ``cat__province_Brussels``-> ``province``   (longest categorical prefix match)
    """
    for prefix in ("num__", "bin__"):
        if transformed_name.startswith(prefix):
            return transformed_name[len(prefix):]
    if transformed_name.startswith("cat__"):
        rest = transformed_name[len("cat__"):]
        # Original categorical names contain underscores, so match the longest
        # known feature that ``rest`` starts with ("property_type_house" -> ...).
        candidates = [c for c in CATEGORICAL_FEATURES if rest.startswith(c + "_") or rest == c]
        if candidates:
            return max(candidates, key=len)
        return rest.rsplit("_", 1)[0]
    return transformed_name


@functools.lru_cache(maxsize=None)
def _explainer(market: str):
    import shap

    pipeline = load_model(market)
    model = pipeline.named_steps["model"]
    return shap.TreeExplainer(model)


def explain_one(features: dict[str, Any], market: str = "sale",
                top: int | None = None) -> dict[str, Any]:
    """Return the SHAP feature-contribution breakdown for one property."""
    pipeline = load_model(market)
    pre = pipeline.named_steps["preprocessor"]
    X = preprocess([features], market)
    Xt = pre.transform(X)
    names = list(pre.get_feature_names_out())

    explainer = _explainer(market)
    shap_vals = explainer.shap_values(Xt)
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]
    base = float(np.ravel(explainer.expected_value)[0])

    # Aggregate transformed-column SHAP back to the original raw feature.
    agg: dict[str, float] = {}
    for name, val in zip(names, shap_vals):
        agg[_origin_feature(name)] = agg.get(_origin_feature(name), 0.0) + float(val)

    contributions = [
        {
            "feature": feat,
            "label": _LABELS.get(feat, feat),
            "value_eur": round(v, 2),
            "direction": "up" if v >= 0 else "down",
            "input": features.get(feat),
        }
        for feat, v in agg.items()
    ]
    contributions.sort(key=lambda c: abs(c["value_eur"]), reverse=True)
    if top:
        contributions = contributions[:top]

    prediction = base + float(np.sum(shap_vals))
    return {
        "market": market,
        "base_value": round(base, 2),
        "prediction": round(max(0.0, prediction), 2),
        "currency": "EUR",
        "unit": "per month" if market == "rent" else "total",
        "contributions": contributions,
    }
