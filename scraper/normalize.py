"""
scraper/normalize.py
====================

Pure, offline helpers that coerce *site-specific* vocabularies into the
**canonical vocabularies** the ML model and downstream aggregations already
use (see ``ml/data/in/cleaned_sale_properties.csv`` for the ground-truth value
sets). None of these functions touch the network, so they are trivially
unit-testable.

Canonical value sets (mirrored from the cleaned dataset):

* ``property_type`` — flat, house, villa, penthouse, flatStudio, groundFloor,
  duplex, masterHouse, bungalow, chalet, loft, cottage, triplex, mansion,
  studentFlat.
* ``category``      — house / apartment (coarse split derived from type).
* ``epc``           — A++, A+, A, B, C, D, E, F, G.
* ``building_state``— A / B / C / D (ordinal condition code, A = best).
* ``kitchen_equipment`` — Super equipped / Fully equipped / Partially equipped /
  Not equipped.
* ``heating_type``  — Gas / Fuel oil / Hot air / Electricity / Wood /
  Solar energy / Coal.

:func:`finalize` is the one composite entry point: it fills ``category`` from
``property_type``, computes ``price_per_sqm``, and enriches
``refnis``/``province``/``region``/``municipality`` from the postal-code →
refnis crosswalk. Address → lat/lon geocoding is delegated to
``geo.geocode.resolve`` *if that module exists*, otherwise skipped.
"""
from __future__ import annotations

import csv
import functools
import re
import unicodedata
from typing import Optional

from scraper.schema import REPO_ROOT

CROSSWALK_PATH = REPO_ROOT / "data" / "invest" / "nis_postal_crosswalk.csv"


# --------------------------------------------------------------------------- #
# small scalar coercers                                                       #
# --------------------------------------------------------------------------- #
def _key(raw) -> str:
    """Lower-case, strip accents, keep only ``a-z0-9`` for lookup matching."""
    s = unicodedata.normalize("NFKD", str(raw))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def to_int(raw) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        return int(round(float(raw)))
    except (TypeError, ValueError):
        m = re.search(r"-?\d+", str(raw))
        return int(m.group()) if m else None


def to_float(raw) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(\.\d+)?", str(raw))
        return float(m.group()) if m else None


def to_binary_flag(val) -> Optional[int]:
    """Coerce a truthy/present indicator (bool, count, yes/no, ...) to 0/1."""
    if val is None:
        return None
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return 0 if float(val) == 0 else 1
    s = str(val).strip().lower()
    if s in {"true", "yes", "y", "1", "on", "present", "available", "oui", "ja"}:
        return 1
    if s in {"false", "no", "n", "0", "off", "absent", "none", "non", "nee", ""}:
        return 0
    return None


def parse_price(raw) -> Optional[float]:
    """Parse a price. Assumes Belgian formatting ('.' thousands, ',' decimals).

    Examples: ``"€ 849.000"`` -> 849000.0, ``849000`` -> 849000.0.
    Non-positive / unparseable -> ``None``.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
        return v if v > 0 else None
    s = str(raw).split(",")[0]  # drop any decimal part after the comma
    digits = re.sub(r"[^\d]", "", s)
    if not digits:
        return None
    v = float(digits)
    return v if v > 0 else None


def parse_surface(raw) -> Optional[float]:
    """Parse a surface in m² ('.' thousands, ',' decimals). ``"85,5 m²"`` -> 85.5."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
        return v if v > 0 else None
    s = str(raw).lower()
    for unit in ("m²", "m2", "sqm", "sq m", "m ²"):
        s = s.replace(unit, "")
    s = s.strip().replace(" ", "").replace(".", "").replace(",", ".")
    m = re.search(r"\d+(\.\d+)?", s)
    if not m:
        return None
    v = float(m.group())
    return v if v > 0 else None


# --------------------------------------------------------------------------- #
# categorical mappers (site vocab -> canonical vocab)                         #
# --------------------------------------------------------------------------- #
_PROPERTY_TYPE = {
    # flats / apartments
    "apartment": "flat", "flat": "flat", "appartement": "flat", "appartment": "flat",
    "penthouse": "penthouse",
    "studio": "flatStudio", "flatstudio": "flatStudio",
    "groundfloor": "groundFloor", "rezdechaussee": "groundFloor", "gelijkvloers": "groundFloor",
    "duplex": "duplex", "triplex": "triplex",
    "loft": "loft",
    "kot": "studentFlat", "studentflat": "studentFlat", "student": "studentFlat",
    "studentroom": "studentFlat", "koten": "studentFlat",
    # houses
    "house": "house", "maison": "house", "woning": "house", "huis": "house",
    "villa": "villa",
    "bungalow": "bungalow", "chalet": "chalet", "cottage": "cottage",
    "mansion": "mansion", "manoir": "mansion",
    "manorhouse": "masterHouse", "masterhouse": "masterHouse",
    "maisondemaitre": "masterHouse", "herenhuis": "masterHouse", "mansionhouse": "masterHouse",
}

_APARTMENT_TYPES = {
    "flat", "penthouse", "flatStudio", "groundFloor", "duplex", "triplex",
    "loft", "studentFlat",
}
_HOUSE_TYPES = {
    "house", "villa", "masterHouse", "bungalow", "chalet", "cottage", "mansion",
}

_BUILDING_STATE = {
    # canonical A (best) .. D (worst)
    "asnew": "A", "new": "A", "excellent": "A", "justrenovated": "A",
    "renovated": "A", "brandnew": "A", "nieuwbouw": "A",
    "good": "B", "verygood": "B", "bon": "B", "goed": "B",
    "tobedoneup": "C", "torefresh": "C", "torenew": "C", "torefurbish": "C",
    "torenovate": "D", "torestore": "D", "torebuild": "D", "toberestored": "D",
    "a": "A", "b": "B", "c": "C", "d": "D",
}

_KITCHEN = {
    "hyperequipped": "Super equipped", "usahyperequipped": "Super equipped",
    "superequipped": "Super equipped",
    "installed": "Fully equipped", "usainstalled": "Fully equipped",
    "fullyequipped": "Fully equipped", "equipped": "Fully equipped",
    "semiequipped": "Partially equipped", "usasemiequipped": "Partially equipped",
    "partiallyequipped": "Partially equipped",
    "notinstalled": "Not equipped", "notequipped": "Not equipped",
    "usauninstalled": "Not equipped", "none": "Not equipped", "no": "Not equipped",
}

_HEATING = {
    "gas": "Gas", "naturalgas": "Gas", "gaz": "Gas",
    "fueloil": "Fuel oil", "oil": "Fuel oil", "mazout": "Fuel oil", "stookolie": "Fuel oil",
    "electric": "Electricity", "electricity": "Electricity", "electrique": "Electricity",
    "wood": "Wood", "pellet": "Wood", "pellets": "Wood", "bois": "Wood",
    "solar": "Solar energy", "solarenergy": "Solar energy",
    "carbon": "Coal", "coal": "Coal", "charbon": "Coal",
    "hotair": "Hot air", "airheating": "Hot air", "aircirculation": "Hot air",
}


def normalize_property_type(raw) -> Optional[str]:
    if raw is None:
        return None
    return _PROPERTY_TYPE.get(_key(raw))


def normalize_epc(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper().replace(" ", "")
    valid = {"A++", "A+", "A", "B", "C", "D", "E", "F", "G"}
    if s in valid:
        return s
    m = re.match(r"^(A\+\+|A\+|[A-G])(?![A-Z])", s)
    return m.group(1) if m else None


def normalize_building_state(raw) -> Optional[str]:
    if raw is None:
        return None
    return _BUILDING_STATE.get(_key(raw))


def normalize_kitchen(raw) -> Optional[str]:
    if raw is None:
        return None
    return _KITCHEN.get(_key(raw))


def normalize_heating(raw) -> Optional[str]:
    if raw is None:
        return None
    return _HEATING.get(_key(raw))


def category_from_property_type(pt: Optional[str]) -> Optional[str]:
    if pt in _APARTMENT_TYPES:
        return "apartment"
    if pt in _HOUSE_TYPES:
        return "house"
    return None


# --------------------------------------------------------------------------- #
# geography enrichment                                                         #
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _crosswalk() -> dict[str, dict]:
    """postal_code -> {refnis, municipality, province, region} (first match)."""
    table: dict[str, dict] = {}
    if not CROSSWALK_PATH.exists():
        return table
    with open(CROSSWALK_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pc = (row.get("postal_code") or "").strip()
            if not pc or pc in table:
                continue
            table[pc] = {
                "refnis": (row.get("refnis") or "").strip() or None,
                "municipality": (row.get("municipality_fr")
                                 or row.get("municipality_nl") or "").strip() or None,
                "province": (row.get("province") or "").strip() or None,
                "region": (row.get("region") or "").strip() or None,
            }
    return table


def _postal_key(pc) -> Optional[str]:
    if pc is None or pc == "":
        return None
    if isinstance(pc, float):
        pc = int(pc)
    m = re.search(r"\d{4}", str(pc))
    return m.group() if m else str(pc).strip() or None


def _try_geocode(record: dict):
    """Best-effort address -> (lat, lon) via ``geo.geocode.resolve`` if present.

    ``geo.geocode`` is being built in parallel; degrade silently if absent.
    """
    try:
        from geo.geocode import resolve  # type: ignore
    except Exception:
        return None
    try:
        parts = [record.get(k) for k in ("street", "house_number", "postal_code", "locality")]
        address = ", ".join(str(p) for p in parts if p)
        if not address:
            return None
        res = resolve(address)
        if not res:
            return None
        lat = res.get("latitude") if isinstance(res, dict) else getattr(res, "latitude", None)
        lon = res.get("longitude") if isinstance(res, dict) else getattr(res, "longitude", None)
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    except Exception:
        return None
    return None


def finalize(record: dict) -> dict:
    """Fill derived + geography fields on an in-place canonical ``record``.

    * ``category``      from ``property_type`` when missing,
    * ``price_per_sqm`` from ``price`` / ``livable_surface``,
    * ``refnis``/``province``/``region``/``municipality`` from the crosswalk,
    * ``latitude``/``longitude`` via geocoding when both are missing.
    """
    if not record.get("category") and record.get("property_type"):
        record["category"] = category_from_property_type(record["property_type"])

    price = record.get("price")
    surface = record.get("livable_surface")
    if price and surface and surface > 0:
        record["price_per_sqm"] = round(float(price) / float(surface), 2)

    pk = _postal_key(record.get("postal_code"))
    if pk:
        record["postal_code"] = pk
        info = _crosswalk().get(pk)
        if info:
            if not record.get("refnis"):
                record["refnis"] = info["refnis"]
            if not record.get("province"):
                record["province"] = info["province"]
            if not record.get("region"):
                record["region"] = info["region"]
            if not record.get("municipality"):
                record["municipality"] = info["municipality"]

    if record.get("latitude") in (None, "") or record.get("longitude") in (None, ""):
        latlon = _try_geocode(record)
        if latlon:
            record["latitude"], record["longitude"] = latlon

    return record


# --------------------------------------------------------------------------- #
# Offline self-test                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from scraper.schema import empty_record

    print("== scraper.normalize self-test (offline) ==")

    cases_type = {"APARTMENT": "flat", "HOUSE": "house", "VILLA": "villa",
                  "FLAT_STUDIO": "flatStudio", "penthouse": "penthouse",
                  "manor_house": "masterHouse", "kot": "studentFlat"}
    for raw, exp in cases_type.items():
        got = normalize_property_type(raw)
        assert got == exp, f"property_type {raw!r} -> {got!r} != {exp!r}"
    print("normalize_property_type ->", cases_type)

    assert normalize_epc("a+") == "A+"
    assert normalize_epc(" B ") == "B"
    assert normalize_epc("A") == "A" and normalize_epc("g") == "G"
    print("normalize_epc('a+') ->", normalize_epc("a+"))

    assert normalize_building_state("AS_NEW") == "A"
    assert normalize_building_state("TO_RENOVATE") == "D"
    assert normalize_building_state("good") == "B"
    print("normalize_building_state('TO_RENOVATE') ->", normalize_building_state("TO_RENOVATE"))

    assert normalize_kitchen("HYPER_EQUIPPED") == "Super equipped"
    assert normalize_kitchen("SEMI_EQUIPPED") == "Partially equipped"
    print("normalize_kitchen('HYPER_EQUIPPED') ->", normalize_kitchen("HYPER_EQUIPPED"))

    assert normalize_heating("FUELOIL") == "Fuel oil"
    assert normalize_heating("gas") == "Gas"
    print("normalize_heating('FUELOIL') ->", normalize_heating("FUELOIL"))

    assert to_binary_flag(True) == 1 and to_binary_flag("no") == 0 and to_binary_flag(2) == 1
    assert parse_price("€ 849.000") == 849000.0
    assert parse_surface("85,5 m²") == 85.5
    print("parse_price('€ 849.000') ->", parse_price("€ 849.000"))
    print("parse_surface('85,5 m²') ->", parse_surface("85,5 m²"))

    rec = empty_record()
    rec.update(property_type="flat", price=849000.0, livable_surface=85.0, postal_code="1000")
    finalize(rec)
    assert rec["category"] == "apartment"
    assert rec["price_per_sqm"] == round(849000 / 85, 2)
    print("finalize -> category=%s price_per_sqm=%s refnis=%s province=%s region=%s"
          % (rec["category"], rec["price_per_sqm"], rec["refnis"], rec["province"], rec["region"]))
    assert rec["refnis"] == "21004" and rec["region"] == "Brussels" and rec["province"], \
        "crosswalk enrichment"  # postal 1000 -> refnis 21004, province 'Brussels-Capital'

    print("OK")
