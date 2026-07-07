"""
Immo Eliza — Streamlit web app
==============================

A friendly front-end for non-technical users to price Belgian properties with
the tuned-XGBoost models from the training project.

Two markets, two tabs:
  🏠 **Buy**  — estimate a property's *sale price*.
  🔑 **Rent** — estimate a property's *monthly rent*.

The app talks to the FastAPI backend (``/predict``) when one is configured
(``IMMO_API_URL`` / sidebar), and otherwise **falls back to loading the model
locally** so it still works standalone on Streamlit Community Cloud. Either way
the prediction is identical — same pickled pipeline.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd
import requests
import streamlit as st

# Make the shared feature contract importable whether we run from the repo root
# or from inside streamlit/. We reuse api/features.py as the single source of
# truth for options, ranges, defaults and geography.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for path in (os.path.join(REPO_ROOT, "api"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from features import (  # noqa: E402  (import after sys.path tweak)
    BINARY_META,
    CATEGORY_OPTIONS,
    NUMERIC_META,
    PROVINCE_CENTROIDS,
    default_property,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
API_URL_DEFAULT = os.environ.get("IMMO_API_URL", "").rstrip("/")

MARKET_CFG = {
    "sale": {
        "tab": "🏠  Buy — sale price",
        "verb": "sale price",
        "unit": "",
        "cta": "💰  Estimate sale price",
        "accent": "#2563eb",       # blue
        "gradient": "linear-gradient(135deg,#2563eb 0%,#1e40af 100%)",
        "example": "Brussels 2-bed apartment near the EU quarter",
    },
    "rent": {
        "tab": "🔑  Rent — monthly rent",
        "verb": "monthly rent",
        "unit": "/month",
        "cta": "🔑  Estimate monthly rent",
        "accent": "#059669",       # green
        "gradient": "linear-gradient(135deg,#10b981 0%,#047857 100%)",
        "example": "Antwerp 2-bed apartment with parking",
    },
}

MODEL_METRICS = {
    "sale": {"r2": 0.8123, "mae": 81840.0, "rmse": 184419.0},
    "rent": {"r2": 0.6269, "mae": 253.72, "rmse": 658.84},
}

st.set_page_config(
    page_title="Immo Eliza — Property Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(135deg,#1e3a8a 0%,#2563eb 45%,#059669 100%);
        border-radius: 18px; padding: 2rem 2.4rem; color: white;
        margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(37,99,235,.25);
    }
    .hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing:-.5px;}
    .hero p  {margin: .4rem 0 0; opacity: .92; font-size: 1.02rem;}
    .result-card {
        border-radius: 16px; padding: 1.6rem 1.8rem; color: white;
        box-shadow: 0 12px 34px rgba(0,0,0,.18); margin-top: .5rem;
    }
    .result-card .big {font-size: 2.7rem; font-weight: 800; line-height: 1.05;}
    .result-card .lbl {opacity: .9; font-size: .95rem; text-transform: uppercase; letter-spacing: 1px;}
    .result-card .band {opacity: .92; margin-top: .5rem; font-size: 1rem;}
    .pill {display:inline-block; background:rgba(255,255,255,.18); border-radius:999px;
           padding:.15rem .7rem; font-size:.8rem; margin-right:.35rem; margin-top:.4rem;}
    .metric-chip {background:#f1f5f9; border-radius:12px; padding:.8rem 1rem; text-align:center;}
    .metric-chip .v {font-size:1.4rem; font-weight:700; color:#0f172a;}
    .metric-chip .k {font-size:.78rem; color:#64748b; text-transform:uppercase; letter-spacing:.5px;}
    .footer {color:#94a3b8; font-size:.85rem; text-align:center; margin-top:2.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Prediction — API first, local model as fallback
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _local_engine():
    """Import the local prediction engine (loads the joblib models once)."""
    import predict as engine  # from api/ on sys.path

    engine.warm_up()
    return engine


def get_prediction(features: dict[str, Any], market: str, api_url: str) -> dict[str, Any]:
    """Return a prediction dict, preferring the API, falling back to local model."""
    if api_url:
        try:
            resp = requests.post(
                f"{api_url}/predict",
                params={"market": market},
                json=features,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            data["_source"] = "API"
            return data
        except Exception as exc:  # noqa: BLE001 - show the user and fall back
            st.warning(f"API unreachable ({exc}). Falling back to the local model.")
    result = _local_engine().predict_one(features, market=market)
    result["_source"] = "local model"
    return result


def api_status(api_url: str) -> tuple[bool, str]:
    if not api_url:
        return False, "not configured"
    try:
        r = requests.get(f"{api_url}/", timeout=5)
        return (r.text.strip('"') == "alive"), f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:60]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️  Settings")
    api_url = st.text_input(
        "Prediction API URL",
        value=API_URL_DEFAULT,
        placeholder="https://immo-eliza-api.onrender.com",
        help="Leave blank to run fully offline with the bundled model.",
    ).rstrip("/")

    ok, detail = api_status(api_url)
    if not api_url:
        st.info("🖥️  Running with the **local bundled model** (no API needed).")
    elif ok:
        st.success(f"🟢  API online — {detail}")
    else:
        st.warning(f"🟡  API offline ({detail}); using local model.")

    st.markdown("---")
    st.markdown(
        "### 🧠  The model\n"
        "**Tuned XGBoost** — the best of four algorithms benchmarked in the "
        "[training project](https://github.com/). Gradient-boosted trees over a "
        "29-feature pipeline (imputation → one-hot → scaling), trained separately "
        "for the **sale** and **rent** markets."
    )
    st.markdown("---")
    st.caption(
        "Immo Eliza · educational project · predictions are model estimates, "
        "not professional valuations."
    )


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <h1>🏡 Immo Eliza — Property Price Predictor</h1>
      <p>Instant AI estimates for Belgian real estate. Choose <b>Buy</b> or
         <b>Rent</b>, describe the property, and get a price with a confidence band —
         powered by a tuned XGBoost model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# The form (shared by both tabs, parametrised by market)
# --------------------------------------------------------------------------- #
def money(value: float, market: str) -> str:
    suffix = " /mo" if market == "rent" else ""
    return f"€{value:,.0f}{suffix}"


def property_form(market: str) -> None:
    cfg = MARKET_CFG[market]
    defaults = default_property()
    key = lambda name: f"{market}_{name}"  # noqa: E731 - unique widget keys per tab

    st.markdown(f"#### Describe the property to estimate its **{cfg['verb']}**")

    with st.form(key=f"form_{market}"):
        # --- Location -----------------------------------------------------
        st.markdown("##### 📍 Location")
        c1, c2, c3 = st.columns(3)
        province = c1.selectbox(
            "Province", CATEGORY_OPTIONS["province"],
            index=CATEGORY_OPTIONS["province"].index("Brussels"), key=key("province"),
        )
        centroid = PROVINCE_CENTROIDS.get(province, {})
        c2.text_input("Region (auto)", value=centroid.get("region", "—"),
                      disabled=True, key=key("region_display"))
        nearest = c3.number_input(
            "Distance to city (km)", min_value=0, max_value=60,
            value=int(NUMERIC_META["nearest_city_distance_km"]["default"]),
            key=key("nearest"),
        )

        # --- Core attributes ---------------------------------------------
        st.markdown("##### 🏠 Property")
        c1, c2, c3, c4 = st.columns(4)
        property_type = c1.selectbox(
            "Type", CATEGORY_OPTIONS["property_type"], key=key("property_type"),
        )
        livable = c2.number_input(
            "Livable surface (m²)", min_value=10, max_value=1000,
            value=int(defaults["livable_surface"]), step=5, key=key("livable"),
        )
        land = c3.number_input(
            "Land surface (m²)", min_value=0, max_value=10000,
            value=0, step=10, key=key("land"),
            help="0 for most apartments.",
        )
        build_year = c4.number_input(
            "Build year", min_value=1750, max_value=2031,
            value=int(defaults["build_year"]), key=key("build_year"),
        )

        c1, c2, c3, c4 = st.columns(4)
        bedrooms = c1.number_input("Bedrooms", 0, 15, int(defaults["bedrooms"]), key=key("bed"))
        bathrooms = c2.number_input("Bathrooms", 0, 10, int(defaults["bathrooms"]), key=key("bath"))
        toilets = c3.number_input("Toilets", 0, 10, int(defaults["toilets"]), key=key("wc"))
        facades = c4.number_input("Facades", 1, 4, int(defaults["facades"]), key=key("fac"),
                                  help="2 = terraced, 4 = detached.")

        c1, c2, c3, c4 = st.columns(4)
        floors = c1.number_input("Floors", 1, 10, int(defaults["number_of_floors"]), key=key("flr"))
        epc = c2.selectbox("EPC label", CATEGORY_OPTIONS["epc"],
                           index=CATEGORY_OPTIONS["epc"].index("C"), key=key("epc"))
        pec = c3.number_input(
            "Energy use (kWh/m²/yr)", 0, 1500,
            int(defaults["primary_energy_consumption"]), step=10, key=key("pec"),
        )
        building_state = c4.selectbox("Condition", CATEGORY_OPTIONS["building_state"],
                                      index=CATEGORY_OPTIONS["building_state"].index("Normal"),
                                      key=key("state"))

        c1, c2 = st.columns(2)
        kitchen = c1.selectbox("Kitchen", CATEGORY_OPTIONS["kitchen_equipment"],
                               index=CATEGORY_OPTIONS["kitchen_equipment"].index("Fully equipped"),
                               key=key("kitchen"))
        heating = c2.selectbox("Heating", CATEGORY_OPTIONS["heating_type"], key=key("heating"))

        # --- Amenities ----------------------------------------------------
        st.markdown("##### ✨ Amenities")
        amenity_vals: dict[str, int] = {}
        cols = st.columns(5)
        for i, (name, meta) in enumerate(BINARY_META.items()):
            with cols[i % 5]:
                amenity_vals[name] = int(
                    st.checkbox(f"{meta['icon']} {meta['label']}", value=False, key=key(name))
                )

        submitted = st.form_submit_button(cfg["cta"], use_container_width=True, type="primary")

    if not submitted:
        st.caption(f"👆 Fill in the details and press **{cfg['cta']}**. "
                   f"Example: *{cfg['example']}*.")
        return

    # --- Assemble the feature dict -------------------------------------
    features: dict[str, Any] = {
        "livable_surface": livable, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "toilets": toilets, "build_year": build_year, "facades": facades,
        "number_of_floors": floors, "primary_energy_consumption": pec,
        "land_surface": land, "nearest_city_distance_km": nearest,
        "property_type": property_type, "province": province,
        "epc": epc, "building_state": building_state,
        "kitchen_equipment": kitchen, "heating_type": heating,
        **amenity_vals,
    }

    with st.spinner("Scoring the property with XGBoost…"):
        result = get_prediction(features, market, api_url)

    _render_result(result, market, features, province, centroid)


def _render_result(result: dict[str, Any], market: str, features: dict[str, Any],
                   province: str, centroid: dict) -> None:
    cfg = MARKET_CFG[market]
    pred = result["prediction"]
    band = result.get("interval", {})
    metrics = result.get("metrics", MODEL_METRICS[market])

    left, right = st.columns([3, 2])
    with left:
        band_txt = ""
        if band:
            band_txt = (
                f"<div class='band'>Likely range: "
                f"<b>{money(band['low'], market)}</b> – "
                f"<b>{money(band['high'], market)}</b></div>"
            )
        st.markdown(
            f"""
            <div class="result-card" style="background:{cfg['gradient']};">
              <div class="lbl">Estimated {cfg['verb']}</div>
              <div class="big">{money(pred, market)}</div>
              {band_txt}
              <span class="pill">🧠 {result.get('model', 'XGBoost (tuned)')}</span>
              <span class="pill">⚡ via {result.get('_source', 'model')}</span>
              <span class="pill">📊 test R² {metrics['r2']:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"The band is ± the model's typical error (MAE ≈ "
            f"{money(metrics['mae'], market)}) on unseen homes — a realistic "
            "margin, not a guarantee."
        )

    with right:
        st.markdown("###### Model quality (held-out test set)")
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metric-chip'><div class='v'>{metrics['r2']*100:.0f}%</div>"
                    f"<div class='k'>variance explained (R²)</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-chip'><div class='v'>{money(metrics['mae'], market)}</div>"
                    f"<div class='k'>typical error (MAE)</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-chip'><div class='v'>{money(pred, market)}</div>"
                    f"<div class='k'>this estimate</div></div>", unsafe_allow_html=True)

    # --- Map of the location ---------------------------------------------
    lat = features.get("latitude") or centroid.get("lat")
    lon = features.get("longitude") or centroid.get("lon")
    if lat and lon:
        st.markdown("###### 📍 Location used for the estimate")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=9, size=200)

    with st.expander("🔎 Feature vector sent to the model"):
        st.json(features)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_sale, tab_rent, tab_about = st.tabs(
    [MARKET_CFG["sale"]["tab"], MARKET_CFG["rent"]["tab"], "📈  About & model"]
)

with tab_sale:
    property_form("sale")

with tab_rent:
    property_form("rent")

with tab_about:
    st.markdown("### 📈 How it works")
    st.markdown(
        "This app is the front-end of a two-part system:\n\n"
        "1. **FastAPI backend** (`/predict`) serving a tuned XGBoost model, deployed on Render via Docker.\n"
        "2. **This Streamlit app**, deployed on Streamlit Community Cloud, which calls that API "
        "(and falls back to the bundled model when offline).\n\n"
        "Both markets share the **same 29-feature pipeline** and preprocessing; only the "
        "training data differs (sale prices vs monthly rents)."
    )
    st.markdown("### 🏆 Model leaderboard (held-out test set)")
    board = pd.DataFrame(
        {
            "Market": ["Sale", "Sale", "Sale", "Sale", "Rent", "Rent", "Rent", "Rent"],
            "Model": ["XGBoost ⭐", "Random Forest", "Decision Tree", "Linear (baseline)",
                      "XGBoost ⭐", "Random Forest", "Decision Tree", "Linear (baseline)"],
            "Test R²": [0.812, 0.772, 0.707, 0.642, 0.627, 0.586, 0.518, 0.532],
            "MAE (€)": ["81,840", "90,111", "116,293", "139,201",
                        "254", "269", "334", "353"],
        }
    )
    st.dataframe(board, use_container_width=True, hide_index=True)
    st.caption("The **tuned XGBoost** model (⭐) is the one shipped in this app for both markets.")

st.markdown(
    "<div class='footer'>Built for the Immo Eliza deployment project · "
    "FastAPI + Docker + Render · Streamlit Community Cloud · "
    "model: tuned XGBoost</div>",
    unsafe_allow_html=True,
)
