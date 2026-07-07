"""
smoke_test_api.py
=================

Fire a handful of realistic requests at a running Immo Eliza API and pretty-print
the responses. Handy after deploying to Render.

Usage
-----
    python scripts/smoke_test_api.py                       # hits http://localhost:8010
    python scripts/smoke_test_api.py https://your.onrender.com
"""

from __future__ import annotations

import sys

import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8010").rstrip("/")

CASES = [
    ("sale", "Brussels 2-bed apartment", {
        "livable_surface": 85, "bedrooms": 2, "bathrooms": 1, "property_type": "flat",
        "province": "Brussels", "epc": "C", "building_state": "Normal",
        "terrace": 1, "elevator": 1, "has_parking": 1,
    }),
    ("sale", "Walloon Brabant villa + pool", {
        "livable_surface": 280, "bedrooms": 4, "bathrooms": 2, "property_type": "villa",
        "province": "Walloon Brabant", "epc": "B", "building_state": "Excellent",
        "land_surface": 1200, "garden": 1, "swimming_pool": 1, "has_parking": 1,
    }),
    ("rent", "Antwerp 2-bed apartment", {
        "livable_surface": 90, "bedrooms": 2, "property_type": "flat",
        "province": "Antwerp", "epc": "C", "terrace": 1, "has_parking": 1,
    }),
    ("rent", "Knokke furnished coastal flat", {
        "livable_surface": 80, "bedrooms": 2, "property_type": "flat",
        "province": "West Flanders", "epc": "B", "furnished": 1, "terrace": 1,
    }),
]


def main() -> None:
    print(f"→ Target: {BASE}\n")

    r = requests.get(f"{BASE}/", timeout=10)
    print(f"GET /         -> {r.status_code} {r.text}")
    h = requests.get(f"{BASE}/health", timeout=10).json()
    print(f"GET /health   -> {h['status']}  models={h['models_loaded']}  "
          f"(sklearn {h['versions']['scikit-learn']}, xgboost {h['versions']['xgboost']})\n")

    for market, label, body in CASES:
        resp = requests.post(f"{BASE}/predict", params={"market": market}, json=body, timeout=20)
        resp.raise_for_status()
        d = resp.json()
        suffix = " /mo" if market == "rent" else ""
        band = d["interval"]
        print(f"[{market:>4}] {label:<34} €{d['prediction']:>12,.0f}{suffix}"
              f"   (band €{band['low']:,.0f}–€{band['high']:,.0f})")

    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    main()
