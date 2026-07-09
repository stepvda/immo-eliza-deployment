"""
Immo Eliza — Streamlit web app (v2)
===================================

A neighbourhood-aware property pricer for Belgium.

What's new vs v1
----------------
* **Shared property form** rendered once, above the tabs — so switching between
  **Buy** and **Rent** keeps everything you typed and lets you price the *same*
  property both ways (and feed it straight into **Invest**).
* **Address autocomplete** (Geopunt for Flanders/Brussels, OSM Photon nationwide)
  → the exact location drives a **neighbourhood-priciness** feature, so the model
  prices by pinpointing the address, not just the province.
* **Why this price?** — a SHAP contribution barchart of every entered feature.
* **Comparables** — five real nearby listings at similar prices (blue map pins),
  around the queried address (red pin), over an adaptive **priciness heatmap**.
* **Invest** tab — ROI from rent alone and rent + projected capital appreciation,
  with break-even timing and returns at 5 / 10 / 15 / 20 years.

The rich features run against the bundled local engine (so the app is fully
self-contained on Streamlit Community Cloud). ``/predict`` can optionally be
routed through the FastAPI backend via the sidebar; everything else is local.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import altair as alt
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# --- make the shared engine + cross-cutting packages importable -------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for path in (os.path.join(REPO_ROOT, "api"), REPO_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from features import (  # noqa: E402
    BINARY_META,
    CATEGORY_OPTIONS,
    NUMERIC_META,
    PROVINCE_CENTROIDS,
    default_property,
)

# --------------------------------------------------------------------------- #
# Palette (from the data-viz reference instance — validated)
# --------------------------------------------------------------------------- #
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
UP, DOWN, NEUTRAL = "#e34948", "#2a78d6", "#c3c2b7"          # diverging (SHAP)
SERIES = ["#2a78d6", "#1baf7a"]                              # categorical (ROI)
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

API_URL_DEFAULT = os.environ.get("IMMO_API_URL", "").rstrip("/")

MARKET_CFG = {
    "sale": {"tab": "🏠  Buy — sale price", "verb": "sale price", "unit": "",
             "cta": "💰  Estimate sale price", "gradient": "linear-gradient(135deg,#2563eb 0%,#1e40af 100%)"},
    "rent": {"tab": "🔑  Rent — monthly rent", "verb": "monthly rent", "unit": "/month",
             "cta": "🔑  Estimate monthly rent", "gradient": "linear-gradient(135deg,#10b981 0%,#047857 100%)"},
}

st.set_page_config(page_title="Immo Eliza — Neighbourhood-Aware Pricing",
                   page_icon="🏡", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1240px;}
    .hero {background: linear-gradient(135deg,#1e3a8a 0%,#2563eb 45%,#059669 100%);
        border-radius: 18px; padding: 1.6rem 2rem; color: white; margin-bottom: 1.1rem;
        box-shadow: 0 10px 30px rgba(37,99,235,.25);}
    .hero h1 {margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing:-.5px;}
    .hero p {margin: .35rem 0 0; opacity: .93; font-size: 1rem;}
    .result-card {border-radius: 16px; padding: 1.4rem 1.6rem; color: white;
        box-shadow: 0 12px 34px rgba(0,0,0,.18);}
    .result-card .big {font-size: 2.5rem; font-weight: 800; line-height: 1.05;}
    .result-card .lbl {opacity: .9; font-size: .9rem; text-transform: uppercase; letter-spacing: 1px;}
    .result-card .band {opacity: .92; margin-top: .4rem; font-size: .98rem;}
    .pill {display:inline-block; background:rgba(255,255,255,.18); border-radius:999px;
        padding:.15rem .7rem; font-size:.78rem; margin-right:.3rem; margin-top:.4rem;}
    .metric-chip {background:#f1f5f9; border-radius:12px; padding:.7rem 1rem; text-align:center;}
    .metric-chip .v {font-size:1.3rem; font-weight:700; color:#0f172a;}
    .metric-chip .k {font-size:.72rem; color:#64748b; text-transform:uppercase; letter-spacing:.5px;}
    .addr-badge {background:#ecfdf5; border:1px solid #a7f3d0; border-radius:10px; padding:.5rem .8rem;
        font-size:.9rem; color:#065f46;}
    .footer {color:#94a3b8; font-size:.82rem; text-align:center; margin-top:2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Local engine (cached) + optional API for /predict
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _engine():
    import predict as engine
    engine.warm_up()
    return engine


@st.cache_resource(show_spinner=False)
def _explainer():
    import explain
    return explain


@st.cache_resource(show_spinner=False)
def _comparables():
    import similar
    return similar


@st.cache_resource(show_spinner=False)
def _priciness():
    from geo import priciness
    return priciness


@st.cache_resource(show_spinner=False)
def _roi():
    from invest import roi
    return roi


@st.cache_resource(show_spinner=False)
def _geocode():
    from geo import geocode
    return geocode


def _fkey(features: dict[str, Any]) -> tuple:
    return tuple(sorted((k, (round(v, 5) if isinstance(v, float) else v)) for k, v in features.items()))


@st.cache_data(show_spinner=False)
def cached_predict(fkey: tuple, market: str, api_url: str) -> dict:
    features = dict(fkey)
    if api_url:
        try:
            resp = requests.post(f"{api_url}/predict", params={"market": market},
                                 json=features, timeout=12)
            resp.raise_for_status()
            data = resp.json(); data["_source"] = "API"
            return data
        except Exception:  # noqa: BLE001 - fall back to the bundled model
            pass
    out = _engine().predict_one(features, market=market)
    out["_source"] = "local model"
    return out


@st.cache_data(show_spinner=False)
def cached_explain(fkey: tuple, market: str) -> dict:
    return _explainer().explain_one(dict(fkey), market=market, top=10)


@st.cache_data(show_spinner=False)
def cached_similar(fkey: tuple, market: str, prediction: float) -> list[dict]:
    return _comparables().similar_properties(dict(fkey), market=market, prediction=prediction, k=5)


@st.cache_data(show_spinner=False)
def cached_tiles(market: str) -> list[dict]:
    try:
        return _priciness().load(market).tiles
    except Exception:  # noqa: BLE001
        return []


@st.cache_data(show_spinner=False)
def cached_roi(purchase: float, rent: float, refnis, province, region, ptype, scenario, include_costs) -> dict:
    return _roi().compute_roi(purchase, rent, refnis=refnis, province=province, region=region,
                              ptype=ptype, scenario=scenario, horizons=(5, 10, 15, 20),
                              include_costs=include_costs)


def geocode_search(query: str) -> list[tuple[str, str]]:
    """streamlit-searchbox callback: label + JSON-encoded suggestion payload."""
    if not query or len(query) < 3:
        return []
    try:
        out = []
        for s in _geocode().suggest(query, limit=7):
            out.append((s.get("label", ""), json.dumps(s)))
        return out
    except Exception:  # noqa: BLE001
        return []


@st.cache_data(show_spinner=False)
def dataset_stats() -> dict:
    """Per-market training size + comparables-pool size, for display in the UI.

    Defensive by design: it must **never raise**. This panel is optional, and on
    Streamlit Cloud a hot-reload can hand back a stale ``predict`` module that
    predates ``TRAIN_COUNTS`` — that must not take down the whole app, so every
    lookup is guarded and missing values fall back to ``None``.
    """
    out = {m: {"total": None, "train": None, "pool": None} for m in ("sale", "rent")}
    try:
        train_counts = getattr(_engine(), "TRAIN_COUNTS", {}) or {}
    except Exception:  # noqa: BLE001
        train_counts = {}
    for m in ("sale", "rent"):
        out[m]["train"] = train_counts.get(m)
        try:
            comp = _comparables()
            out[m]["total"] = comp.total_size(m)
            out[m]["pool"] = comp.pool_size(m)
        except Exception:  # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def money(value: float, market: str) -> str:
    suffix = " /mo" if market == "rent" else ""
    return f"€{value:,.0f}{suffix}"


def _pct_to_rgb(pct: float, alpha: int = 150) -> list[int]:
    """Map a 0–100 priciness percentile to the sequential blue ramp."""
    pct = max(0.0, min(100.0, float(pct)))
    pos = pct / 100.0 * (len(SEQ_BLUE) - 1)
    lo = int(pos); hi = min(lo + 1, len(SEQ_BLUE) - 1); frac = pos - lo
    c0 = SEQ_BLUE[lo].lstrip("#"); c1 = SEQ_BLUE[hi].lstrip("#")
    rgb = [round(int(c0[i:i+2], 16) * (1 - frac) + int(c1[i:i+2], 16) * frac) for i in (0, 2, 4)]
    return rgb + [alpha]


def effective_latlon(state: dict) -> tuple[float, float]:
    if state.get("resolved") and state["resolved"].get("latitude"):
        r = state["resolved"]
        return float(r["latitude"]), float(r["longitude"])
    c = PROVINCE_CENTROIDS.get(state.get("province", "Brussels"), {"lat": 50.85, "lon": 4.35})
    return c["lat"], c["lon"]


def build_features(state: dict) -> dict[str, Any]:
    """Assemble the model feature dict from the shared input state."""
    feats = {k: state[k] for k in (
        "livable_surface", "land_surface", "build_year", "bedrooms", "bathrooms",
        "toilets", "facades", "number_of_floors", "primary_energy_consumption",
        "nearest_city_distance_km", "property_type", "province", "epc",
        "building_state", "kitchen_equipment", "heating_type") if k in state}
    feats.update({name: int(state.get(name, 0)) for name in BINARY_META})
    lat, lon = effective_latlon(state)
    if state.get("resolved") and state["resolved"].get("latitude"):
        feats["latitude"], feats["longitude"] = lat, lon   # exact address → surface lookup
    return feats


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️  Settings")
    api_url = st.text_input("Prediction API URL", value=API_URL_DEFAULT,
                            placeholder="https://immo-eliza-api.onrender.com",
                            help="Optional — routes /predict through the API. "
                                 "All other features run on the bundled engine.").rstrip("/")
    if not api_url:
        st.info("🖥️  Running fully local (bundled model + priciness + ROI).")
    st.markdown("---")
    st.markdown(
        "### 🧠  The model\n**Tuned XGBoost** over a 30-feature pipeline that now "
        "includes a **neighbourhood-priciness** feature — the €/m² percentile of the "
        "exact address, from an adaptive spatial surface built on real listings.")
    _stats = dataset_stats()
    _s, _r = _stats["sale"], _stats["rent"]
    _lines = []
    if _s["total"] and _r["total"]:
        _lines.append(f"- **📥 Input data (total):** {_s['total']:,} sale + {_r['total']:,} rent properties")
    if _s["train"] and _r["train"]:
        _lines.append(f"- **🧠 Model trained on:** {_s['train']:,} sale + {_r['train']:,} rent "
                      f"(80% training split)")
    if _s["pool"] and _r["pool"]:
        _lines.append(f"- **🏘️ Comparables pool:** {_s['pool']:,} sale + {_r['pool']:,} rent "
                      f"(usable for similar-property search)")
    if _lines:
        st.markdown("### 📊  Data")
        st.markdown("\n".join(_lines))
    st.caption("Immo Eliza · educational project · estimates, not valuations.")


st.markdown(
    """
    <div class="hero">
      <h1>🏡 Immo Eliza — Neighbourhood-Aware Property Pricing</h1>
      <p>Describe a property <b>once</b>, pin its <b>exact address</b>, and price it for
         <b>Buy</b> and <b>Rent</b> — with the drivers behind the price, five real
         comparables, a priciness heatmap, and an <b>investment ROI</b> projection.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Shared input panel (rendered once → values persist across all tabs)
# --------------------------------------------------------------------------- #
def shared_inputs() -> dict[str, Any]:
    d = default_property()
    st.session_state.setdefault("resolved", None)

    st.markdown("#### 📍 Location")
    ca, cb = st.columns([3, 2])
    with ca:
        picked = None
        try:
            from streamlit_searchbox import st_searchbox
            picked = st_searchbox(geocode_search, key="addr_search",
                                  placeholder="Start typing a street… (e.g. Veldstraat Gent)",
                                  label="Address (optional — enables exact-location pricing)")
        except Exception:
            st.text_input("Address search unavailable — use province below",
                          disabled=True, key="addr_fallback")

        if picked:
            try:
                sug = json.loads(picked) if isinstance(picked, str) else picked
                resolved = _geocode().resolve(sug.get("street") or sug.get("label", ""),
                                              city=sug.get("city"),
                                              house_number=sug.get("house_number"))
                st.session_state["resolved"] = resolved or {**sug,
                    "latitude": None, "longitude": None}
            except Exception:
                st.session_state["resolved"] = None

    resolved = st.session_state.get("resolved")
    default_prov = (resolved or {}).get("province") or "Brussels"
    if default_prov not in CATEGORY_OPTIONS["province"]:
        default_prov = "Brussels"
    with cb:
        province = st.selectbox("Province (default / fallback)", CATEGORY_OPTIONS["province"],
                                index=CATEGORY_OPTIONS["province"].index(default_prov),
                                key="province")
        centroid = PROVINCE_CENTROIDS.get(province, {})
        st.caption(f"Region: **{centroid.get('region', '—')}**")

    if resolved and resolved.get("latitude"):
        hn = resolved.get("house_number") or ""
        st.markdown(
            f"<div class='addr-badge'>📌 Using exact address: "
            f"<b>{resolved.get('street','')} {hn}, {resolved.get('postcode','')} "
            f"{resolved.get('municipality') or resolved.get('city','')}</b> "
            f"({resolved.get('latitude'):.4f}, {resolved.get('longitude'):.4f} · "
            f"{resolved.get('source','')})</div>", unsafe_allow_html=True)

    st.markdown("#### 🏠 Property")
    c1, c2, c3, c4 = st.columns(4)
    property_type = c1.selectbox("Type", CATEGORY_OPTIONS["property_type"], key="property_type")
    livable_surface = c2.number_input("Livable surface (m²)", 10, 1000,
                                      int(d["livable_surface"]), 5, key="livable_surface")
    land_surface = c3.number_input("Land surface (m²)", 0, 10000, 0, 10, key="land_surface")
    build_year = c4.number_input("Build year", 1750, 2031, int(d["build_year"]), key="build_year")

    c1, c2, c3, c4 = st.columns(4)
    bedrooms = c1.number_input("Bedrooms", 0, 15, int(d["bedrooms"]), key="bedrooms")
    bathrooms = c2.number_input("Bathrooms", 0, 10, int(d["bathrooms"]), key="bathrooms")
    toilets = c3.number_input("Toilets", 0, 10, int(d["toilets"]), key="toilets")
    facades = c4.number_input("Facades", 1, 4, int(d["facades"]), key="facades",
                              help="2 = terraced, 4 = detached.")

    c1, c2, c3, c4 = st.columns(4)
    number_of_floors = c1.number_input("Floors", 1, 10, int(d["number_of_floors"]), key="number_of_floors")
    epc = c2.selectbox("EPC label", CATEGORY_OPTIONS["epc"],
                       index=CATEGORY_OPTIONS["epc"].index("C"), key="epc")
    primary_energy_consumption = c3.number_input("Energy use (kWh/m²/yr)", 0, 1500,
                                                 int(d["primary_energy_consumption"]), 10,
                                                 key="primary_energy_consumption")
    nearest_city_distance_km = c4.number_input("Distance to city (km)", 0, 60, 3,
                                               key="nearest_city_distance_km")

    c1, c2 = st.columns(2)
    kitchen_equipment = c1.selectbox("Kitchen", CATEGORY_OPTIONS["kitchen_equipment"],
                                     index=CATEGORY_OPTIONS["kitchen_equipment"].index("Fully equipped"),
                                     key="kitchen_equipment")
    heating_type = c2.selectbox("Heating", CATEGORY_OPTIONS["heating_type"], key="heating_type")

    st.markdown("#### ✨ Amenities")
    cols = st.columns(5)
    for i, (name, meta) in enumerate(BINARY_META.items()):
        cols[i % 5].checkbox(f"{meta['icon']} {meta['label']}", key=name)

    return {k: st.session_state[k] for k in (
        "province", "property_type", "livable_surface", "land_surface", "build_year",
        "bedrooms", "bathrooms", "toilets", "facades", "number_of_floors", "epc",
        "primary_energy_consumption", "nearest_city_distance_km", "kitchen_equipment",
        "heating_type", *BINARY_META)} | {"resolved": st.session_state.get("resolved")}


state = shared_inputs()
st.markdown("---")


# --------------------------------------------------------------------------- #
# Rendering helpers (result card, SHAP chart, comparables, heatmap)
# --------------------------------------------------------------------------- #
def render_result_card(result: dict, market: str) -> None:
    cfg = MARKET_CFG[market]
    pred = result["prediction"]
    band = result.get("interval", {})
    metrics = result.get("metrics", {})
    left, right = st.columns([3, 2])
    with left:
        band_txt = (f"<div class='band'>Likely range: <b>{money(band['low'], market)}</b> – "
                    f"<b>{money(band['high'], market)}</b></div>") if band else ""
        st.markdown(
            f"""<div class="result-card" style="background:{cfg['gradient']};">
              <div class="lbl">Estimated {cfg['verb']}</div>
              <div class="big">{money(pred, market)}</div>
              {band_txt}
              <span class="pill">🧠 {result.get('model','XGBoost (tuned)')}</span>
              <span class="pill">⚡ via {result.get('_source','model')}</span>
              <span class="pill">📊 test R² {metrics.get('r2',0):.2f}</span>
            </div>""", unsafe_allow_html=True)
    with right:
        st.markdown("###### Model quality (held-out test)")
        m1, m2 = st.columns(2)
        m1.markdown(f"<div class='metric-chip'><div class='v'>{metrics.get('r2',0)*100:.0f}%</div>"
                    f"<div class='k'>variance explained</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-chip'><div class='v'>{money(metrics.get('mae',0), market)}</div>"
                    f"<div class='k'>typical error (MAE)</div></div>", unsafe_allow_html=True)


def render_shap(features: dict, market: str) -> None:
    st.markdown("###### 🔎 What drives this price?")
    try:
        ex = cached_explain(_fkey(features), market)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Explanation unavailable ({exc}).")
        return
    rows = [c for c in ex["contributions"] if abs(c["value_eur"]) > 0][:9]
    if not rows:
        st.caption("No notable drivers.")
        return
    df = pd.DataFrame(rows)
    df["input"] = df["input"].map(lambda v: "" if v is None else str(v))   # keep column single-typed
    df["Effect"] = df["direction"].map({"up": "raises price", "down": "lowers price"})
    chart = (alt.Chart(df).mark_bar(cornerRadius=3, height=16)
             .encode(
                 x=alt.X("value_eur:Q", title=f"contribution ({'€/mo' if market=='rent' else '€'})",
                         axis=alt.Axis(grid=True, gridColor=GRID)),
                 y=alt.Y("label:N", sort="-x", title=None),
                 color=alt.Color("Effect:N",
                                 scale=alt.Scale(domain=["raises price", "lowers price"],
                                                 range=[UP, DOWN]),
                                 legend=alt.Legend(orient="bottom", title=None)),
                 tooltip=[alt.Tooltip("label:N", title="feature"),
                          alt.Tooltip("input:N", title="your value"),
                          alt.Tooltip("value_eur:Q", title="€ effect", format=",.0f")])
             .properties(height=max(180, 26 * len(df))))
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Base value {money(ex['base_value'], market)} + drivers = "
               f"{money(ex['prediction'], market)}. Red raises the price, blue lowers it.")


def render_map(features: dict, market: str, comps: list[dict]) -> None:
    st.markdown("###### 🗺️ Neighbourhood priciness — red = this property, blue = comparables")
    lat, lon = features.get("latitude"), features.get("longitude")
    if lat is None or lon is None:
        c = PROVINCE_CENTROIDS.get(features.get("province"), {"lat": 50.85, "lon": 4.35})
        lat, lon = c["lat"], c["lon"]

    tiles = cached_tiles(market)
    layers = []
    if tiles:
        tdf = pd.DataFrame(tiles)
        tdf["fill"] = tdf["percentile"].map(lambda p: _pct_to_rgb(p, 130))
        layers.append(pdk.Layer("H3HexagonLayer", data=tdf, get_hexagon="cell",
                                get_fill_color="fill", pickable=True, opacity=0.55,
                                extruded=False, stroked=False))
    if comps:
        cdf = pd.DataFrame(comps)
        cdf = cdf.dropna(subset=["latitude", "longitude"])
        cdf["price_txt"] = cdf["price"].map(lambda v: f"€{v:,.0f}")
        layers.append(pdk.Layer("ScatterplotLayer", data=cdf,
                                get_position="[longitude, latitude]",
                                get_fill_color=[42, 120, 214], get_radius=90,
                                radius_min_pixels=7, radius_max_pixels=16,
                                stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1,
                                pickable=True))
    layers.append(pdk.Layer("ScatterplotLayer",
                            data=pd.DataFrame([{"longitude": lon, "latitude": lat}]),
                            get_position="[longitude, latitude]",
                            get_fill_color=[227, 73, 72], get_radius=130,
                            radius_min_pixels=9, radius_max_pixels=20,
                            stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2))

    tooltip = {"html": "<b>{price_txt}</b><br/>{livable_surface} m² · {bedrooms} bd<br/>"
                       "{locality} · {distance_km} km",
               "style": {"backgroundColor": "#0b0b0b", "color": "white", "fontSize": "12px"}}
    st.pydeck_chart(pdk.Deck(
        map_style="light", layers=layers,
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=11.5, pitch=0),
        tooltip=tooltip), use_container_width=True)


def render_comparables(comps: list[dict], market: str) -> None:
    pool = dataset_stats().get(market, {}).get("pool")
    pool_txt = f" (from {pool:,} {market} listings)" if pool else ""
    st.markdown(f"###### 🏘️ Five similar properties nearby{pool_txt}")
    if not comps:
        st.caption("No comparable listings found in range.")
        return
    rows = []
    for c in comps:
        rows.append({
            "Price": money(c["price"], market),
            "€/m²": f"{c['price_per_sqm']:,.0f}" if c.get("price_per_sqm") else "—",
            "Surface": f"{c['livable_surface']:.0f} m²" if c.get("livable_surface") else "—",
            "Beds": str(c.get("bedrooms")) if c.get("bedrooms") is not None else "—",
            "Type": c.get("property_type") or "—",
            "Where": c.get("locality") or c.get("municipality") or "—",
            "EPC": c.get("epc") or "—",
            "Distance": f"{c.get('distance_km','—')} km",
            "Listing": c.get("url") or None,   # clickable link to the source site
        })
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
        column_config={
            "Listing": st.column_config.LinkColumn(
                "Listing", display_text="Open ↗",
                help="Open the property on the source immo site (Immoweb / Immovlan)."),
        },
    )


# --------------------------------------------------------------------------- #
# Market tab (Buy / Rent) — shares the inputs above
# --------------------------------------------------------------------------- #
def market_tab(market: str) -> dict | None:
    cfg = MARKET_CFG[market]
    features = build_features(state)
    with st.spinner(f"Pricing with XGBoost… ({cfg['verb']})"):
        result = cached_predict(_fkey(features), market, api_url)
    render_result_card(result, market)
    st.markdown("")
    left, right = st.columns([1, 1])
    comps = cached_similar(_fkey(features), market, result["prediction"])
    with left:
        render_shap(features, market)
    with right:
        render_comparables(comps, market)
    render_map({**features, **{k: v for k, v in zip(("latitude", "longitude"), effective_latlon(state))}},
               market, comps)
    return result


# --------------------------------------------------------------------------- #
# Invest tab
# --------------------------------------------------------------------------- #
def invest_tab(sale_result: dict | None, rent_result: dict | None) -> None:
    st.markdown("#### 📈 Investment ROI — rent alone vs rent + capital appreciation")
    resolved = state.get("resolved") or {}
    refnis = resolved.get("refnis")
    region = resolved.get("region") or PROVINCE_CENTROIDS.get(state["province"], {}).get("region")
    ptype = "apartment" if state["property_type"] in ("flat", "flatStudio", "penthouse",
                                                       "duplex", "triplex", "loft",
                                                       "groundFloor", "studentFlat") else "house"

    c1, c2, c3, c4 = st.columns(4)
    default_purchase = int(sale_result["prediction"]) if sale_result else 300000
    default_rent = int(rent_result["prediction"]) if rent_result else 1200
    purchase = c1.number_input("Purchase price (€)", 25000, 5_000_000, default_purchase, 5000,
                               help="Pre-filled from the Buy estimate for this property.")
    rent = c2.number_input("Monthly rent (€)", 100, 12000, default_rent, 25,
                           help="Pre-filled from the Rent estimate for this property.")
    scenario = c3.selectbox("Appreciation scenario",
                            ["hist", "cons", "base", "opt"], index=0,
                            format_func=lambda s: {"hist": "Historical trend (municipality)",
                                                   "cons": "Conservative +2%/yr",
                                                   "base": "Base +3%/yr",
                                                   "opt": "Optimistic +4.3%/yr"}[s])
    include_costs = c4.toggle("Include acquisition costs", value=False,
                              help="Adds registration/notary (~12.5% FL / 13% WA) + a net-rent haircut.")

    try:
        roi = cached_roi(float(purchase), float(rent), refnis, state["province"], region,
                         ptype, scenario, include_costs)
    except Exception as exc:  # noqa: BLE001
        st.error(f"ROI computation failed: {exc}")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gross rental yield", f"{roi['gross_yield_pct']:.1f}%")
    k2.metric("Appreciation", f"{roi['appreciation_pct_per_yr']*100:.1f}%/yr"
              if roi['appreciation_pct_per_yr'] < 1 else f"{roi['appreciation_pct_per_yr']:.1f}%/yr")
    be = roi.get("breakeven_year_total")
    k3.metric("Break-even (rent+growth)", f"{be:.1f} yr" if be else ">20 yr")
    ber = roi.get("breakeven_year_rent_only")
    k4.metric("Break-even (rent only)", f"{ber:.1f} yr" if ber else ">20 yr")

    # ROI-over-time chart (rent-only vs total), with milestone markers.
    series = pd.DataFrame(roi["series"])
    long = series.melt(id_vars="year", value_vars=["roi_rent_only_pct", "roi_total_pct"],
                       var_name="kind", value_name="roi")
    long["kind"] = long["kind"].map({"roi_rent_only_pct": "Rent only",
                                     "roi_total_pct": "Rent + appreciation"})
    base = alt.Chart(long).encode(
        x=alt.X("year:Q", title="years held", axis=alt.Axis(grid=False)),
        y=alt.Y("roi:Q", title="cumulative ROI (%)", axis=alt.Axis(grid=True, gridColor=GRID)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Rent only", "Rent + appreciation"],
                                                  range=SERIES),
                        legend=alt.Legend(orient="bottom", title=None)))
    line = base.mark_line(strokeWidth=2.5)
    rule = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(
        strokeDash=[4, 4], color=MUTED).encode(y="y:Q")
    layers = [rule, line]
    if be:
        layers.append(alt.Chart(pd.DataFrame({"x": [be]})).mark_rule(color=NEUTRAL).encode(x="x:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=340), use_container_width=True)

    ms = roi["milestones"]
    cols = st.columns(4)
    for col, h in zip(cols, (5, 10, 15, 20)):
        m = ms.get(str(h)) or ms.get(h) or {}
        col.markdown(f"<div class='metric-chip'><div class='v'>{m.get('roi_total_pct',0):.0f}%</div>"
                     f"<div class='k'>{h}Y total ROI · profit €{m.get('profit_eur',0):,.0f}</div></div>",
                     unsafe_allow_html=True)
    st.caption(" · ".join(roi.get("assumptions", {}).values())
               if isinstance(roi.get("assumptions"), dict) else "")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_sale, tab_rent, tab_invest, tab_about = st.tabs(
    [MARKET_CFG["sale"]["tab"], MARKET_CFG["rent"]["tab"], "📈  Invest", "ℹ️  About & model"])

with tab_sale:
    sale_result = market_tab("sale")
with tab_rent:
    rent_result = market_tab("rent")
with tab_invest:
    invest_tab(sale_result, rent_result)

with tab_about:
    st.markdown("### 📈 How it works")
    st.markdown(
        "- **Shared inputs** above the tabs → the same property is priced for Buy and Rent and fed into Invest.\n"
        "- **Neighbourhood priciness**: the exact address is looked up on an adaptive spatial surface "
        "(street/block → postcode → municipality) built from real listings; its €/m² percentile is a model feature.\n"
        "- **Explainability** via SHAP; **comparables** are real nearby listings; the **heatmap** shows priciness "
        "with a red pin for your property and blue pins for the comparables.\n"
        "- **Invest** combines gross rental yield with historical/projected capital appreciation "
        "(Statbel + ING/KBC scenarios) into a cumulative-ROI projection with break-even timing.")
    try:
        board = pd.read_csv(os.path.join(REPO_ROOT, "data", "reference", "evaluation_results.csv"))
        st.markdown("### 🏆 Model leaderboard (held-out test set)")
        st.dataframe(board, use_container_width=True, hide_index=True)
    except Exception:
        pass

st.markdown(
    "<div class='footer'>Immo Eliza · neighbourhood-aware pricing · FastAPI + XGBoost + "
    "priciness surface · address autocomplete (Geopunt/Photon) · SHAP · ROI</div>",
    unsafe_allow_html=True,
)
