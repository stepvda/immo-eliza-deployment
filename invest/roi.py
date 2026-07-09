#!/usr/bin/env python3
"""Investment ROI engine — combine gross rental yield with expected capital
appreciation into a per-property total-ROI projection.

This is the headless numeric engine ported from the validated ``totalroi``
investor dashboard (the HTML/Plotly presentation layer is intentionally NOT
ported).  It exposes one public entry point, :func:`compute_roi`, which the
FastAPI ``/invest`` route and the Streamlit *Invest* tab call, plus a handful of
cached data loaders.

Total ROI over a horizon ``H`` (cumulative %, on the purchase price), matching
the reference ``roiOf``::

    rent-only ROI(H) = gross_yield% * H
    total     ROI(H) = gross_yield% * H + ((1 + g)^H - 1) * 100

where ``g`` is the annual capital-appreciation rate — either each
municipality's own historical trend (2015->2025 CAGR from the Statbel price
series, clipped 0-6 %/yr) or a uniform national scenario.

Data inputs (all under ``data/``, resolved relative to this file):
    data/invest/property_prices_historical_by_municipality.csv
    data/invest/property_prices_projected_by_municipality.csv
    data/invest/nis_postal_crosswalk.csv
    data/listings/sale.parquet     (optional — yields degrade gracefully if absent)
    data/listings/rent.parquet     (optional)

Dependencies: pandas, numpy (no plotting).  Python 3.13.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_INVEST = REPO_ROOT / "data" / "invest"
LISTINGS_DIR = REPO_ROOT / "data" / "listings"

HIST_CSV = DATA_INVEST / "property_prices_historical_by_municipality.csv"
PROJ_CSV = DATA_INVEST / "property_prices_projected_by_municipality.csv"
XWALK_CSV = DATA_INVEST / "nis_postal_crosswalk.csv"

# --------------------------------------------------------------------------- #
# Cleaning thresholds — identical to the production model's guard rails
# --------------------------------------------------------------------------- #
SALE_PRICE_MIN, SALE_PRICE_MAX = 25_000, 15_000_000
RENT_PRICE_MIN, RENT_PRICE_MAX = 200, 25_000
SURFACE_MIN, SURFACE_MAX = 9, 3_000
SALE_PPSQM_MIN, SALE_PPSQM_MAX = 400, 18_000
RENT_PPSQM_MIN, RENT_PPSQM_MAX = 3, 70

# --------------------------------------------------------------------------- #
# Analysis config
# --------------------------------------------------------------------------- #
MIN_SALE_N = 5            # min sale listings of a type in a municipality to trust it
MIN_RENT_N = 5            # min rent comparables before falling back to province/region
CAGR_LO, CAGR_HI = 0.0, 0.06     # clip forward appreciation to a realistic 0-6 %/yr
YIELD_CAP = 9.0           # clip gross yield to drop rent-fallback artifacts (rural tails)
SCEN = {"cons": 0.02, "base": 0.03, "opt": 0.043}   # uniform annual appreciation scenarios
BASE_G = 0.03             # last-resort appreciation when no historical trend exists

# The last year covered by the published-forecast tables (data/invest/projected).
# ROI beyond this is a trend extrapolation, not a forecast.
FORECAST_LAST_YEAR = 2030
BASE_YEAR = 2025

# --------------------------------------------------------------------------- #
# Acquisition-cost & holding-cost assumptions (only applied when include_costs)
# --------------------------------------------------------------------------- #
ACQ_COST_WALLONIA = 0.13    # Wallonia registration duties (~12.5-13 %)
ACQ_COST_FL_BXL = 0.125     # Flanders / Brussels registration duties (~12.5 %)
HOLDING_COST_FACTOR = 0.15  # annual allowance for maintenance / vacancy / management

SCEN_LABEL = {
    "hist": "Historical trend (per municipality 2015->2025 CAGR)",
    "cons": "Conservative +2%/yr",
    "base": "Base +3%/yr",
    "opt": "Optimistic +4.3%/yr",
}


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def historical() -> pd.DataFrame:
    """The Statbel historical municipal median-price series (2010-2025)."""
    return pd.read_csv(HIST_CSV)


@lru_cache(maxsize=1)
def projected() -> pd.DataFrame:
    """The forward price projections (2026-2030) per municipality/scenario."""
    return pd.read_csv(PROJ_CSV)


@lru_cache(maxsize=1)
def crosswalk() -> pd.DataFrame:
    """The postal-code <-> NIS (refnis) crosswalk with region/province."""
    return pd.read_csv(XWALK_CSV)


@lru_cache(maxsize=1)
def _geo() -> dict:
    """Geography lookups derived from the crosswalk (keyed by refnis / postal)."""
    xw = crosswalk()
    pc2ref = (xw.sort_values("refnis").drop_duplicates("postal_code")
              .set_index("postal_code")["refnis"].astype(int).to_dict())
    g = xw.drop_duplicates("refnis").set_index("refnis")
    return {
        "pc2ref": pc2ref,
        "ref2region": g["region"].to_dict(),
        "ref2province": g["province"].to_dict(),
        "ref2name": g["municipality_fr"].to_dict(),
    }


def postal_to_refnis(postal_code: int | str) -> int | None:
    """Map a Belgian postal code to its NIS (refnis) municipality code."""
    try:
        pc = int(postal_code)
    except (TypeError, ValueError):
        return None
    ref = _geo()["pc2ref"].get(pc)
    return int(ref) if ref is not None else None


# --------------------------------------------------------------------------- #
# Capital appreciation — historical CAGR (ported verbatim from the reference)
# --------------------------------------------------------------------------- #
def cagr_by_municipality(hist: pd.DataFrame, htype: str) -> dict[int, float]:
    """Annualised price growth ending 2025, starting from the earliest year >=2015
    (else earliest available), for one Statbel property type."""
    d = hist[hist.property_type == htype].dropna(subset=["median_price_eur"])
    out: dict[int, float] = {}
    for ref, g in d.groupby("refnis"):
        g = g.sort_values("year")
        end = g[g.year == 2025]
        if end.empty:
            continue
        p_end = float(end.median_price_eur.iloc[0])
        cand = g[g.year >= 2015]
        srow = (cand if not cand.empty else g).iloc[0]
        y0, p0, span = int(srow.year), float(srow.median_price_eur), 2025 - int(srow.year)
        if span >= 3 and p0 > 0:
            out[int(ref)] = float(np.clip((p_end / p0) ** (1 / span) - 1, CAGR_LO, CAGR_HI))
    return out


@lru_cache(maxsize=1)
def cagr_tables() -> dict:
    """Per-municipality apartment/house CAGR dicts + their regional medians.

    Returns a dict with keys ``cagr_apt``, ``cagr_house`` (refnis -> CAGR) and
    ``reg_cagr_apt``, ``reg_cagr_house`` (region -> median CAGR), used as the
    fallback chain by :func:`cagr_for`.
    """
    hist = historical()
    ref2region = _geo()["ref2region"]
    cagr_apt = cagr_by_municipality(hist, "apartments")
    cagr_house = cagr_by_municipality(hist, "houses_all")

    def _reg_median(cagr: dict[int, float]) -> dict[str, float]:
        if not cagr:
            return {}
        s = pd.Series(cagr).rename_axis("refnis").reset_index(name="c")
        s["region"] = s["refnis"].map(ref2region)
        return s.dropna(subset=["region"]).groupby("region")["c"].median().to_dict()

    return {
        "cagr_apt": cagr_apt,
        "cagr_house": cagr_house,
        "reg_cagr_apt": _reg_median(cagr_apt),
        "reg_cagr_house": _reg_median(cagr_house),
    }


def cagr_for(refnis: int | None, ptype: str, region: str | None) -> float:
    """Expected annual appreciation for a (municipality, type), with the
    municipality -> region -> BASE_G fallback chain, clipped to 0-6 %/yr.

    Note: 0.0 is treated as "missing" here (matching the reference's ``||``
    fallback semantics) so a clipped-to-zero municipality trend falls through to
    the regional median rather than pinning appreciation at 0 %.
    """
    t = cagr_tables()
    if ptype == "apartment":
        v = (t["cagr_apt"].get(refnis) or t["cagr_house"].get(refnis)
             or t["reg_cagr_apt"].get(region) or t["reg_cagr_house"].get(region)
             or BASE_G)
    else:
        v = t["cagr_house"].get(refnis) or t["reg_cagr_house"].get(region) or BASE_G
    return float(np.clip(v, CAGR_LO, CAGR_HI))


# --------------------------------------------------------------------------- #
# Listings-derived yields (optional — degrade gracefully if parquet is absent)
# --------------------------------------------------------------------------- #
def _load_listings(kind: str) -> pd.DataFrame:
    """Load and guard-rail a canonical listings parquet (``sale`` or ``rent``).

    Raises FileNotFoundError if the parquet is absent so callers can degrade to
    the historical/projected CSVs only.
    """
    path = LISTINGS_DIR / f"{kind}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    for c in ("price", "livable_surface"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["price", "livable_surface", "category"])
    df = df[df["category"].isin(["house", "apartment"])]
    df = df[df["livable_surface"].between(SURFACE_MIN, SURFACE_MAX)]
    df["ppsqm"] = df["price"] / df["livable_surface"]
    if kind == "sale":
        df = df[df["price"].between(SALE_PRICE_MIN, SALE_PRICE_MAX)]
        df = df[df["ppsqm"].between(SALE_PPSQM_MIN, SALE_PPSQM_MAX)]
    else:
        df = df[df["price"].between(RENT_PRICE_MIN, RENT_PRICE_MAX)]
        df = df[df["ppsqm"].between(RENT_PPSQM_MIN, RENT_PPSQM_MAX)]

    geo = _geo()
    # Resolve refnis (fall back to postal-code mapping) then derive region /
    # province from the crosswalk so their keys match compute_roi's lookups.
    if "refnis" not in df.columns or df["refnis"].isna().all():
        df["refnis"] = pd.to_numeric(df.get("postal_code"), errors="coerce").map(geo["pc2ref"])
    else:
        df["refnis"] = pd.to_numeric(df["refnis"], errors="coerce")
        miss = df["refnis"].isna()
        if miss.any():
            df.loc[miss, "refnis"] = (
                pd.to_numeric(df.loc[miss, "postal_code"], errors="coerce").map(geo["pc2ref"]))
    df = df.dropna(subset=["refnis"])
    df["refnis"] = df["refnis"].astype(int)
    df["ptype"] = df["category"]
    df["region"] = df["refnis"].map(geo["ref2region"])
    df["province"] = df["refnis"].map(geo["ref2province"])
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def _yield_tables() -> dict | None:
    """Median sale/rent EUR/m2 aggregations from the listings, or ``None`` when
    the parquet files are absent (in which case yields fall back to the passed
    monthly_rent/purchase_price only)."""
    try:
        sale = _load_listings("sale")
        rent = _load_listings("rent")
    except FileNotFoundError:
        return None
    if sale.empty or rent.empty:
        return None

    sg = sale.groupby(["refnis", "ptype"]).agg(
        n_sale=("ppsqm", "size"), sale_ppsqm=("ppsqm", "median"), price=("price", "median"))
    rm = rent.groupby(["refnis", "ptype"]).agg(
        n_rent=("ppsqm", "size"), rent_ppsqm=("ppsqm", "median"))
    rent_prov = rent.groupby(["province", "ptype"])["ppsqm"].median()
    rent_reg = rent.groupby(["region", "ptype"])["ppsqm"].median()

    return {
        "n_sale": sg["n_sale"].to_dict(),
        "sale_ppsqm": sg["sale_ppsqm"].to_dict(),
        "sale_price": sg["price"].to_dict(),
        "rent_muni": {k: (float(r.rent_ppsqm), int(r.n_rent)) for k, r in rm.iterrows()},
        "rent_prov": rent_prov.to_dict(),
        "rent_reg": rent_reg.to_dict(),
    }


def municipality_yield(refnis: int | None, ptype: str,
                       province: str | None, region: str | None) -> tuple[float, str] | None:
    """Gross rental yield (%) from the listings for a (municipality, type), using
    the muni -> province -> region rent fallback and YIELD_CAP.  Returns
    ``(gross_yield_pct, basis)`` or ``None`` when not derivable."""
    yt = _yield_tables()
    if yt is None or refnis is None:
        return None
    key = (int(refnis), ptype)
    n_sale = yt["n_sale"].get(key, 0)
    sale_ppsqm = yt["sale_ppsqm"].get(key)
    if not n_sale or n_sale < MIN_SALE_N or not sale_ppsqm or sale_ppsqm <= 0:
        return None

    rm = yt["rent_muni"].get(key)
    if rm is not None and rm[1] >= MIN_RENT_N:
        rent_ppsqm, basis = rm[0], "muni"
    elif (province, ptype) in yt["rent_prov"]:
        rent_ppsqm, basis = float(yt["rent_prov"][(province, ptype)]), "prov"
    elif (region, ptype) in yt["rent_reg"]:
        rent_ppsqm, basis = float(yt["rent_reg"][(region, ptype)]), "region"
    else:
        return None

    gy = min(rent_ppsqm * 12 / float(sale_ppsqm) * 100, YIELD_CAP)
    return (float(gy), basis)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm_ptype(ptype: str) -> str:
    p = str(ptype or "").strip().lower()
    if p.startswith("apart") or p.startswith("flat") or p == "appartement":
        return "apartment"
    return "house"


def _acq_rate(region: str | None) -> float:
    """Acquisition-cost rate (registration duties) by region."""
    if region and "wallon" in region.lower():
        return ACQ_COST_WALLONIA
    return ACQ_COST_FL_BXL  # Flanders / Brussels / unknown default


def _crossing_year(threshold: float, roi_at, max_year: int) -> float | None:
    """First (fractional) year at which ``roi_at(year)`` reaches ``threshold``,
    linearly interpolated between integer years; ``None`` if never reached
    within ``max_year``."""
    prev = roi_at(0)
    if prev >= threshold:
        return 0.0
    for y in range(1, max_year + 1):
        cur = roi_at(y)
        if cur >= threshold:
            frac = (threshold - prev) / (cur - prev) if cur != prev else 0.0
            return round((y - 1) + frac, 2)
        prev = cur
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_roi(
    purchase_price: float,
    monthly_rent: float,
    *, refnis: int | None = None, province: str | None = None,
    region: str | None = None, ptype: str = "house",
    horizons: tuple[int, ...] = (5, 10, 15, 20),
    scenario: str = "hist",            # 'hist' | 'cons' | 'base' | 'opt'
    include_costs: bool = False,       # layer acquisition costs for a real payback
) -> dict:
    """Project total ROI for a single property.

    ``purchase_price`` and ``monthly_rent`` describe the property being priced.
    ``refnis`` / ``province`` / ``region`` locate it so the appreciation trend
    (``g``) and, as a fallback, the rental yield can be looked up.  See the
    module docstring for the ROI formulas.  Returns the dict documented in the
    project spec (gross_yield_pct, appreciation_pct_per_yr, series, milestones,
    breakeven years, assumptions, ...).
    """
    if scenario not in ("hist", *SCEN):
        raise ValueError(f"scenario must be 'hist' or one of {tuple(SCEN)}, got {scenario!r}")
    if not horizons:
        raise ValueError("horizons must be a non-empty tuple")

    purchase_price = float(purchase_price)
    monthly_rent = float(monthly_rent or 0.0)
    ptype = _norm_ptype(ptype)
    refnis = int(refnis) if refnis is not None else None
    horizons = tuple(sorted({int(h) for h in horizons}))
    max_h = max(horizons)

    # Resolve geography from refnis where not explicitly given.
    geo = _geo()
    if refnis is not None:
        if region is None:
            region = geo["ref2region"].get(refnis)
        if province is None:
            province = geo["ref2province"].get(refnis)

    # --- appreciation rate g ---------------------------------------------- #
    if scenario == "hist":
        g = cagr_for(refnis, ptype, region)
        g_source = ("municipality historical CAGR" if (refnis is not None
                    and (cagr_tables()["cagr_apt"].get(refnis)
                         or cagr_tables()["cagr_house"].get(refnis)))
                    else "regional-median historical CAGR / base fallback")
    else:
        g = SCEN[scenario]
        g_source = f"uniform scenario ({SCEN_LABEL[scenario]})"

    # --- gross rental yield ----------------------------------------------- #
    # Prefer the passed-in monthly_rent / purchase_price (the property the user
    # is actually pricing); fall back to the municipality listings tables.
    gross_yield = None
    yield_source = None
    if monthly_rent > 0 and purchase_price > 0:
        gross_yield = min(monthly_rent * 12 / purchase_price * 100, YIELD_CAP)
        yield_source = "implied from passed monthly_rent / purchase_price"
    if gross_yield is None:
        my = municipality_yield(refnis, ptype, province, region)
        if my is not None:
            gross_yield, basis = my
            yield_source = f"municipality listings ({basis} rent basis)"
    if gross_yield is None:
        gross_yield = 0.0
        yield_source = "unavailable (no rent input and no listings)"

    # --- costs ------------------------------------------------------------ #
    acq_rate = _acq_rate(region) if include_costs else 0.0
    net_factor = (1.0 - HOLDING_COST_FACTOR) if include_costs else 1.0
    outlay = purchase_price * (1.0 + acq_rate)          # cash to recover
    threshold_pct = (1.0 + acq_rate) * 100.0            # outlay as % of purchase price

    # Yield actually used in the ROI series (net of holding costs when asked).
    yield_used = gross_yield * net_factor
    annual_gross_rent = (monthly_rent * 12) if monthly_rent > 0 else gross_yield / 100 * purchase_price
    annual_net_rent = annual_gross_rent * net_factor

    # --- series (year 0 .. max horizon) ----------------------------------- #
    def roi_rent_at(y: float) -> float:
        return yield_used * y

    def roi_total_at(y: float) -> float:
        return yield_used * y + ((1 + g) ** y - 1) * 100

    series = []
    for y in range(0, max_h + 1):
        growth = (1 + g) ** y
        value = purchase_price * growth
        rent_cum = annual_net_rent * y
        series.append({
            "year": y,
            "roi_rent_only_pct": round(roi_rent_at(y), 2),
            "roi_total_pct": round(roi_total_at(y), 2),
            "cumulative_value_eur": round(value, 2),
            "cumulative_rent_eur": round(rent_cum, 2),
            "net_worth_eur": round(value + rent_cum, 2),
        })

    # --- milestones per requested horizon --------------------------------- #
    milestones = {}
    for h in horizons:
        growth = (1 + g) ** h
        value = purchase_price * growth
        rent_cum = annual_net_rent * h
        profit = value + rent_cum - outlay
        milestones[h] = {
            "roi_rent_only_pct": round(roi_rent_at(h), 2),
            "roi_total_pct": round(roi_total_at(h), 2),
            "profit_eur": round(profit, 2),
        }

    # --- breakeven -------------------------------------------------------- #
    breakeven_rent = _crossing_year(threshold_pct, roi_rent_at, max_h)
    breakeven_total = _crossing_year(threshold_pct, roi_total_at, max_h)

    # --- assumptions (human-readable) ------------------------------------- #
    outlay_desc = (f"purchase + ~{acq_rate * 100:.1f}% acquisition costs (EUR {outlay:,.0f})"
                   if include_costs else f"100% of purchase price (EUR {outlay:,.0f})")
    assumptions = {
        "scenario": SCEN_LABEL[scenario],
        "appreciation": f"Capital appreciation g = {g * 100:.2f}%/yr ({g_source}), "
                        f"compounded as (1+g)^H over the whole horizon.",
        "gross_yield": (
            f"Gross rental yield {gross_yield:.2f}% ({yield_source}); "
            + ("net of a "
               f"{HOLDING_COST_FACTOR * 100:.0f}% annual allowance for maintenance/vacancy/management "
               "in the ROI series." if include_costs
               else "GROSS — before costs, taxes and vacancy (matches the team dashboard).")),
        "costs": (f"Acquisition costs of ~{acq_rate * 100:.1f}% "
                  f"({'Wallonia' if acq_rate == ACQ_COST_WALLONIA else 'Flanders/Brussels'} "
                  "registration duties) added to the outlay."
                  if include_costs
                  else "Costs OFF (default): gross yield, outlay = purchase price only."),
        "breakeven_basis": f"Breakeven = first year the cumulative return recovers {outlay_desc}; "
                           "interpolated to a fractional year, None if not reached within the horizon.",
        "extrapolation": (
            f"Appreciation is a single annual rate held constant. Published forecasts extend only to "
            f"{FORECAST_LAST_YEAR}; from a {BASE_YEAR} base, horizons beyond ~{FORECAST_LAST_YEAR - BASE_YEAR} "
            f"years (post-{FORECAST_LAST_YEAR}) are a trend extrapolation, not a forecast."),
        "formula": "ROI(H) = gross_yield% * H + ((1+g)^H - 1) * 100  (cumulative %, on purchase price).",
    }

    return {
        "gross_yield_pct": round(gross_yield, 2),
        "appreciation_pct_per_yr": round(g * 100, 2),
        "scenario": scenario,
        "annual_net_rent_eur": round(annual_net_rent, 2),
        "series": series,
        "milestones": milestones,
        "breakeven_year_rent_only": breakeven_rent,
        "breakeven_year_total": breakeven_total,
        "assumptions": assumptions,
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    pp, mr = 300_000.0, 1_200.0
    print("Immo Eliza -- Investment ROI engine self-test")
    print(f"Property: EUR {pp:,.0f} Brussels apartment @ EUR {mr:,.0f}/mo "
          f"(implied gross yield {mr * 12 / pp * 100:.2f}%)\n")

    for scen in ("hist", "cons", "base", "opt"):
        res = compute_roi(pp, mr, region="Brussels", ptype="apartment", scenario=scen)
        print(f"=== scenario={scen} | gross yield {res['gross_yield_pct']:.2f}% "
              f"| appreciation {res['appreciation_pct_per_yr']:.2f}%/yr "
              f"| annual net rent EUR {res['annual_net_rent_eur']:,.0f} ===")
        print(f"  {'yr':>3}  {'rentROI%':>9}  {'totalROI%':>10}  "
              f"{'value EUR':>13}  {'rentCum EUR':>13}  {'netWorth EUR':>14}")
        for row in res["series"]:
            print(f"  {row['year']:>3}  {row['roi_rent_only_pct']:>9.2f}  "
                  f"{row['roi_total_pct']:>10.2f}  {row['cumulative_value_eur']:>13,.0f}  "
                  f"{row['cumulative_rent_eur']:>13,.0f}  {row['net_worth_eur']:>14,.0f}")
        print("  milestones:")
        for h, m in res["milestones"].items():
            print(f"    {h:>2}Y  rent {m['roi_rent_only_pct']:>7.2f}%  "
                  f"total {m['roi_total_pct']:>7.2f}%  profit EUR {m['profit_eur']:>12,.0f}")
        print(f"  breakeven  rent-only: {res['breakeven_year_rent_only']}  "
              f"total: {res['breakeven_year_total']}\n")

    # Demonstrate the include_costs "real payback" variant.
    res = compute_roi(pp, mr, region="Brussels", ptype="apartment",
                      scenario="base", include_costs=True)
    print("=== include_costs=True (scenario=base) ===")
    print(f"  gross yield {res['gross_yield_pct']:.2f}%  "
          f"annual net rent EUR {res['annual_net_rent_eur']:,.0f}")
    print(f"  breakeven  rent-only: {res['breakeven_year_rent_only']}  "
          f"total: {res['breakeven_year_total']}")
    for k, v in res["assumptions"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    _selftest()
