"""
geo/geocode.py
==============

Belgian address autocomplete + resolution for the Streamlit property form and
the FastAPI proxy. Uses **free / keyless** providers only:

* **PRIMARY — Geopunt Geolocation API** (Flanders + Brussels, keyless, v4):
    - Suggestion (autocomplete):
        ``https://loc.geopunt.be/v4/Suggestion?q=<text>&c=<count>``
        -> ``{"SuggestionResult": ["Kerkstraat 1, 9000 Gent", ...]}``
    - Location (geocode a chosen string):
        ``https://loc.geopunt.be/v4/Location?q=<text>&c=1``
        -> ``{"LocationResult": [{"Municipality","Zipcode","Thoroughfarename",
             "Housenumber","Location":{"Lat_WGS84","Lon_WGS84"},
             "FormattedAddress", ...}]}``
* **FALLBACK — OSM Photon** (nationwide incl. Wallonia, keyless):
    ``https://photon.komoot.io/api/?q=<text>&limit=<n>&lang=en&lat=50.6&lon=4.7&bbox=2.5,49.4,6.5,51.6``
    -> GeoJSON FeatureCollection, features filtered to ``countrycode == "BE"``.

Design rules (the sandbox / CI may have **no network**):

* Every public function degrades gracefully to ``[]`` / ``None`` on *any*
  failure (timeout, DNS, non-200, bad JSON). Nothing here ever raises or hangs.
* Short timeouts (~6s), one retry, and an on-disk JSON cache under
  ``data/geo/cache/`` keyed by ``sha256(request_url)``.
* ``IMMO_GEOCODE_OFFLINE=1`` forces offline mode (all lookups return empty),
  used by the self-test / CI so behaviour is deterministic without a network.

Province / region names are kept consistent with the rest of the app
(``api/features.py``) by deriving them from ``data/invest/nis_postal_crosswalk.csv``
and normalising the two spellings that differ there
(``Brussels-Capital`` -> ``Brussels``, ``Liege`` -> ``Liège``).

Public API (called by the API + Streamlit):
    suggest(query, limit=6) -> list[dict]
    resolve(street, city=None, house_number=None) -> dict | None
    valid_house_numbers(street, city) -> list[str]
    PROVINCE_BY_POSTCODE(postcode) -> dict | None
    available() -> bool

Run the self-test:
    python -m geo.geocode
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

try:  # httpx is a hard dependency, but never let an import hiccup crash callers
    import httpx
except Exception:  # pragma: no cover - defensive
    httpx = None  # type: ignore[assignment]

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "geo" / "cache"
XWALK_CSV = DATA_DIR / "invest" / "nis_postal_crosswalk.csv"

GEOPUNT_SUGGESTION = "https://loc.geopunt.be/v4/Suggestion"
GEOPUNT_LOCATION = "https://loc.geopunt.be/v4/Location"
PHOTON_URL = "https://photon.komoot.io/api/"

HTTP_TIMEOUT = 6.0  # seconds
HTTP_RETRIES = 1    # one retry -> two attempts total
_HEADERS = {
    "User-Agent": "immo-eliza-geocode/1.0 (+https://github.com/immo-eliza)",
    "Accept": "application/json",
}

# Belgium bounding box for Photon bias/bounds: minlon,minlat,maxlon,maxlat
_PHOTON_BBOX = "2.5,49.4,6.5,51.6"
_PHOTON_LAT, _PHOTON_LON = 50.6, 4.7

# Crosswalk province spellings -> canonical app spellings (api/features.py).
_PROVINCE_NORMALIZE = {
    "Brussels-Capital": "Brussels",
    "Brussels Capital": "Brussels",
    "Liege": "Liège",
}

__all__ = [
    "suggest",
    "resolve",
    "valid_house_numbers",
    "PROVINCE_BY_POSTCODE",
    "available",
]


# --------------------------------------------------------------------------- #
# Offline flag / availability
# --------------------------------------------------------------------------- #
def _offline() -> bool:
    """True when IMMO_GEOCODE_OFFLINE is set to anything truthy."""
    val = os.environ.get("IMMO_GEOCODE_OFFLINE", "").strip().lower()
    return val not in ("", "0", "false", "no", "off")


def available() -> bool:
    """Cheap check: are network lookups possible at all?

    Returns ``False`` when ``IMMO_GEOCODE_OFFLINE`` is set or ``httpx`` is
    missing. Does *not* touch the network (so it is safe to call in a tight UI
    loop); a ``True`` result only means a lookup *may* succeed.
    """
    return httpx is not None and not _offline()


# --------------------------------------------------------------------------- #
# On-disk JSON cache (key = sha256 of the request URL)
# --------------------------------------------------------------------------- #
def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _cache_get(url: str) -> Any:
    try:
        path = _cache_path(url)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _cache_set(url: str, data: Any) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(url).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# HTTP with retry + cache + graceful degradation
# --------------------------------------------------------------------------- #
def _build_url(base: str, params: dict[str, Any]) -> str:
    clean = {k: v for k, v in params.items() if v is not None}
    return f"{base}?{urlencode(clean)}"


def _http_get_json(url: str) -> Any:
    """GET ``url`` and return parsed JSON, or ``None`` on any failure.

    Never raises, never hangs. Honours the offline flag and the on-disk cache.
    """
    if _offline() or httpx is None:
        return None

    cached = _cache_get(url)
    if cached is not None:
        return cached

    for _ in range(HTTP_RETRIES + 1):
        try:
            resp = httpx.get(
                url,
                timeout=HTTP_TIMEOUT,
                headers=_HEADERS,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                data = resp.json()
                _cache_set(url, data)
                return data
        except Exception:
            # timeout / DNS / connection / bad JSON -> try again, then give up
            continue
    return None


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #
def _first(mapping: Any, *keys: str) -> Any:
    """First non-empty value among ``keys`` in a dict (case-tolerant)."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    # case-insensitive second pass
    lower = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        v = lower.get(key.lower())
        if v not in (None, ""):
            return v
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_pc(postcode: Any) -> Optional[str]:
    """Normalise a Belgian postcode to a 4-digit string, or None."""
    if postcode in (None, ""):
        return None
    m = re.search(r"\d{4}", str(postcode))
    return m.group(0) if m else None


def _clean_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    s = str(value).strip()
    return s or None


def _split_street_hn(text: str) -> tuple[Optional[str], Optional[str]]:
    """Split "Veldstraat 1" -> ("Veldstraat", "1"). House number = trailing
    token that begins with a digit (e.g. "1", "16A", "12-14")."""
    text = (text or "").strip()
    if not text:
        return None, None
    toks = text.split()
    if len(toks) >= 2 and re.match(r"^\d", toks[-1]):
        return " ".join(toks[:-1]) or None, toks[-1]
    return text, None


def _hn_sort_key(hn: str) -> tuple[int, str]:
    m = re.match(r"\s*(\d+)", hn)
    return (int(m.group(1)) if m else 10**9, hn)


# --------------------------------------------------------------------------- #
# Crosswalk (postcode / municipality -> province / region)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_crosswalk() -> tuple[dict[str, dict], dict[str, dict]]:
    """Load ``nis_postal_crosswalk.csv`` once.

    Returns ``(by_postcode, by_municipality)`` where each value is
    ``{"municipality","province","region","postcode"}`` with province spellings
    normalised to the app's canonical names. ``by_municipality`` is keyed by the
    lower-cased NL *and* FR municipality names.
    """
    by_pc: dict[str, dict] = {}
    by_muni: dict[str, dict] = {}
    try:
        df = pd.read_csv(XWALK_CSV, dtype=str)
    except Exception:
        return by_pc, by_muni

    for _, row in df.iterrows():
        province = _clean_str(row.get("province"))
        province = _PROVINCE_NORMALIZE.get(province, province)
        region = _clean_str(row.get("region"))
        muni_nl = _clean_str(row.get("municipality_nl"))
        muni_fr = _clean_str(row.get("municipality_fr"))
        municipality = muni_nl or muni_fr
        pc = _norm_pc(row.get("postal_code"))

        rec = {
            "municipality": municipality,
            "province": province,
            "region": region,
            "postcode": pc,
        }
        if pc and pc not in by_pc:
            by_pc[pc] = rec
        for name in (muni_nl, muni_fr):
            if name:
                by_muni.setdefault(name.strip().lower(), rec)
    return by_pc, by_muni


def PROVINCE_BY_POSTCODE(postcode: Any) -> Optional[dict]:
    """Resolve a Belgian postcode to ``{"municipality","province","region",
    "postcode"}`` (canonical province/region names), or ``None`` if unknown.

    Local crosswalk lookup — works offline.
    """
    pc = _norm_pc(postcode)
    if not pc:
        return None
    by_pc, _ = _load_crosswalk()
    rec = by_pc.get(pc)
    return dict(rec) if rec else None


def _enrich(postcode: Any, municipality: Optional[str]) -> Optional[dict]:
    """Best crosswalk match by postcode first, then by municipality name."""
    by_pc, by_muni = _load_crosswalk()
    pc = _norm_pc(postcode)
    if pc and pc in by_pc:
        return by_pc[pc]
    if municipality:
        rec = by_muni.get(municipality.strip().lower())
        if rec:
            return rec
    return None


# --------------------------------------------------------------------------- #
# Geopunt suggestion-string parsing
# --------------------------------------------------------------------------- #
def _parse_geopunt_suggestion(text: str) -> dict:
    """Parse "Kerkstraat 1, 9000 Gent" into structured parts (best effort)."""
    text = (text or "").strip()
    street = house_number = postcode = city = None

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 2:
        tail = parts[-1]
        toks = tail.split()
        if toks and re.fullmatch(r"\d{4}", toks[0]):
            postcode = toks[0]
            city = " ".join(toks[1:]) or None
        else:
            city = tail or None
        street, house_number = _split_street_hn(parts[0])
    elif parts:
        only = parts[0]
        toks = only.split()
        if toks and re.fullmatch(r"\d{4}", toks[0]):
            postcode = toks[0]
            city = " ".join(toks[1:]) or None
        else:
            street, house_number = _split_street_hn(only)

    return {
        "label": text,
        "street": street,
        "house_number": house_number,
        "postcode": postcode,
        "city": city,
        "source": "geopunt",
    }


# --------------------------------------------------------------------------- #
# Provider: Geopunt
# --------------------------------------------------------------------------- #
def _geopunt_suggest(query: str, limit: int) -> list[dict]:
    url = _build_url(GEOPUNT_SUGGESTION, {"q": query, "c": limit})
    data = _http_get_json(url)
    if not isinstance(data, dict):
        return []
    results = _first(data, "SuggestionResult", "suggestionResult", "Suggestions")
    if not isinstance(results, list):
        return []
    out: list[dict] = []
    for item in results:
        if isinstance(item, str):
            out.append(_parse_geopunt_suggestion(item))
        elif isinstance(item, dict):
            label = _first(item, "FormattedAddress", "formattedAddress", "value", "label")
            if label:
                out.append(_parse_geopunt_suggestion(str(label)))
        if len(out) >= limit:
            break
    return out


def _geopunt_location(query: str, count: int = 1) -> list[dict]:
    """Return the raw LocationResult entries for a query, structured."""
    url = _build_url(GEOPUNT_LOCATION, {"q": query, "c": count})
    data = _http_get_json(url)
    if not isinstance(data, dict):
        return []
    results = _first(data, "LocationResult", "locationResult", "Locations", "results")
    if not isinstance(results, list):
        return []

    out: list[dict] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        loc = r.get("Location") or r.get("location") or {}
        lat = _to_float(_first(loc, "Lat_WGS84", "lat", "Lat", "latitude"))
        lon = _to_float(_first(loc, "Lon_WGS84", "lon", "Lon", "longitude"))
        if lat is None:
            lat = _to_float(_first(r, "Lat_WGS84", "lat", "Lat", "latitude"))
        if lon is None:
            lon = _to_float(_first(r, "Lon_WGS84", "lon", "Lon", "longitude"))
        out.append({
            "street": _clean_str(_first(r, "Thoroughfarename", "Street",
                                        "Streetname", "thoroughfarename")),
            "house_number": _clean_str(_first(r, "Housenumber", "HouseNumber",
                                              "housenumber")),
            "postcode": _norm_pc(_first(r, "Zipcode", "PostalCode", "Postcode",
                                        "zipcode")),
            "city": _clean_str(_first(r, "Municipality", "Gemeente",
                                      "municipality")),
            "municipality": _clean_str(_first(r, "Municipality", "Gemeente",
                                              "municipality")),
            "formatted": _clean_str(_first(r, "FormattedAddress",
                                           "formattedAddress")),
            "latitude": lat,
            "longitude": lon,
            "source": "geopunt",
        })
    return out


# --------------------------------------------------------------------------- #
# Provider: Photon (OSM, nationwide fallback)
# --------------------------------------------------------------------------- #
def _photon_features(query: str, limit: int) -> list[dict]:
    url = _build_url(PHOTON_URL, {
        "q": query,
        "limit": limit,
        "lang": "en",
        "lat": _PHOTON_LAT,
        "lon": _PHOTON_LON,
        "bbox": _PHOTON_BBOX,
    })
    data = _http_get_json(url)
    if not isinstance(data, dict):
        return []
    features = data.get("features")
    if not isinstance(features, list):
        return []

    out: list[dict] = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        cc = _first(props, "countrycode", "countryCode")
        if cc and str(cc).upper() != "BE":  # keep BE + unknowns (bbox-bounded)
            continue
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        lon = lat = None
        if isinstance(coords, list) and len(coords) >= 2:
            lon = _to_float(coords[0])
            lat = _to_float(coords[1])
        street = _clean_str(_first(props, "street", "name"))
        out.append({
            "street": street,
            "house_number": _clean_str(_first(props, "housenumber")),
            "postcode": _norm_pc(_first(props, "postcode")),
            "city": _clean_str(_first(props, "city", "town", "village",
                                      "county")),
            "municipality": _clean_str(_first(props, "city", "town", "village")),
            "state": _clean_str(_first(props, "state")),
            "latitude": lat,
            "longitude": lon,
            "source": "photon",
        })
    return out


def _photon_label(rec: dict) -> str:
    left = " ".join(x for x in (rec.get("street"), rec.get("house_number")) if x)
    right = " ".join(x for x in (rec.get("postcode"), rec.get("city")) if x)
    label = ", ".join(x for x in (left, right) if x)
    return label or (rec.get("city") or "")


# --------------------------------------------------------------------------- #
# PUBLIC API
# --------------------------------------------------------------------------- #
def suggest(query: str, limit: int = 6) -> list[dict]:
    """Autocomplete a partial Belgian address.

    Returns a list of
    ``{"label","street","house_number","postcode","city","source"}`` dicts.
    Geopunt (Flanders + Brussels) is tried first; if it yields nothing (e.g. a
    Wallonia query) it falls back to Photon. Returns ``[]`` on any failure or
    when offline.
    """
    query = (query or "").strip()
    if not query or not available():
        return []
    limit = max(1, int(limit or 6))

    results = _geopunt_suggest(query, limit)
    if results:
        return results[:limit]

    out: list[dict] = []
    for rec in _photon_features(query, limit):
        out.append({
            "label": _photon_label(rec),
            "street": rec.get("street"),
            "house_number": rec.get("house_number"),
            "postcode": rec.get("postcode"),
            "city": rec.get("city"),
            "source": "photon",
        })
    return out[:limit]


def resolve(
    street: str,
    city: Optional[str] = None,
    house_number: Optional[str] = None,
) -> Optional[dict]:
    """Geocode a chosen address to a full structured record.

    Returns
    ``{"street","house_number","postcode","city","municipality","province",
       "region","latitude","longitude","source","valid_house_numbers"}``
    or ``None`` when nothing resolves / offline. Province & region are derived
    from the postcode via ``nis_postal_crosswalk.csv`` (falling back to the
    municipality name when the provider gives no postcode).
    """
    street = (street or "").strip()
    if not street or not available():
        return None

    # Fold a house number that was passed inside the street string.
    if house_number is None:
        street_only, hn_in_street = _split_street_hn(street)
        if hn_in_street:
            street, house_number = street_only or street, hn_in_street

    query_parts = [street]
    if house_number:
        query_parts.append(str(house_number))
    query = " ".join(query_parts)
    if city:
        query = f"{query}, {city}"

    rec: Optional[dict] = None
    # Geopunt first (FL + BXL), then Photon (nationwide incl. Wallonia).
    geopunt = _geopunt_location(query, count=1)
    if geopunt:
        rec = geopunt[0]
    else:
        photon = _photon_features(query, limit=1)
        if photon:
            rec = photon[0]
    if rec is None:
        return None

    postcode = rec.get("postcode")
    provider_muni = rec.get("municipality") or rec.get("city")

    enriched = _enrich(postcode, provider_muni)
    if enriched:
        municipality = enriched.get("municipality") or provider_muni
        province = enriched.get("province")
        region = enriched.get("region")
        if not postcode:
            postcode = enriched.get("postcode")
    else:
        municipality = provider_muni
        province = None
        region = None

    resolved_street = rec.get("street") or street
    resolved_city = rec.get("city") or municipality

    return {
        "street": resolved_street,
        "house_number": rec.get("house_number") or (
            str(house_number) if house_number else None
        ),
        "postcode": postcode,
        "city": resolved_city,
        "municipality": municipality,
        "province": province,
        "region": region,
        "latitude": rec.get("latitude"),
        "longitude": rec.get("longitude"),
        "source": rec.get("source"),
        "valid_house_numbers": valid_house_numbers(resolved_street,
                                                   resolved_city or (city or "")),
    }


def valid_house_numbers(street: str, city: str) -> list[str]:
    """Best-effort distinct house numbers that exist for ``street`` in ``city``.

    Queries Geopunt Suggestion for "<street>, <city>" and collects the house
    numbers it returns. Returns ``[]`` when unavailable (offline, or a
    Wallonia/Photon-only street where existence can't be guaranteed) — the UI
    then allows free house-number entry.
    """
    street = (street or "").strip()
    city = (city or "").strip()
    if not street or not available():
        return []

    query = f"{street}, {city}" if city else street
    # Ask for a generous count so we see a range of numbers on the street.
    suggestions = _geopunt_suggest(query, limit=25)
    if not suggestions:
        return []

    street_lc = street.lower()
    city_lc = city.lower()
    numbers: set[str] = set()
    for s in suggestions:
        hn = s.get("house_number")
        if not hn:
            continue
        s_street = (s.get("street") or "").lower()
        s_city = (s.get("city") or "").lower()
        street_ok = (
            not s_street
            or s_street.startswith(street_lc)
            or street_lc.startswith(s_street)
        )
        city_ok = not city_lc or not s_city or city_lc in s_city or s_city in city_lc
        if street_ok and city_ok:
            numbers.add(str(hn))

    return sorted(numbers, key=_hn_sort_key)


# --------------------------------------------------------------------------- #
# Self-test (python -m geo.geocode)
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    print("geo.geocode self-test")
    print("=" * 60)

    # (a) Forced-offline: everything must degrade to []/None without error.
    os.environ["IMMO_GEOCODE_OFFLINE"] = "1"
    assert available() is False, "available() must be False when offline"
    assert suggest("Veldstraat, Gent") == [], "offline suggest() must be []"
    assert resolve("Rue de la Loi", "Brussels", "16") is None, \
        "offline resolve() must be None"
    assert valid_house_numbers("Veldstraat", "Gent") == [], \
        "offline valid_house_numbers() must be []"
    # local crosswalk lookups still work offline:
    bxl = PROVINCE_BY_POSTCODE("1000")
    print(f"(a) offline OK — suggest/resolve/valid_house_numbers empty; "
          f"PROVINCE_BY_POSTCODE('1000') = {bxl}")
    assert isinstance(bxl, dict) and bxl.get("region") == "Brussels", \
        "crosswalk lookup should resolve 1000 -> Brussels"
    gent = PROVINCE_BY_POSTCODE("9000")
    print(f"    PROVINCE_BY_POSTCODE('9000') = {gent}")

    # (b) Live network (if reachable). Never fatal — must exit 0 even offline.
    os.environ.pop("IMMO_GEOCODE_OFFLINE", None)
    print("-" * 60)
    print(f"(b) network available()? {available()}")
    live = False
    try:
        for q in ("Veldstraat, Gent", "Rue de la Loi, Brussels"):
            sg = suggest(q, limit=3)
            print(f"    suggest({q!r}) -> {len(sg)} result(s)")
            for row in sg[:3]:
                print(f"        {row}")
            if sg:
                live = True

        r1 = resolve("Rue de la Loi", "Brussels", "16")
        print(f"    resolve('Rue de la Loi', 'Brussels', '16') -> {r1}")
        r2 = resolve("Veldstraat", "Gent")
        print(f"    resolve('Veldstraat', 'Gent') -> {r2}")
        if r1 or r2:
            live = True
    except Exception as exc:  # pragma: no cover - must never be fatal
        print(f"    (network path raised, ignored) {type(exc).__name__}: {exc}")

    print("-" * 60)
    print(f"live network worked: {'YES' if live else 'NO (offline/blocked)'}")
    print("self-test PASSED (exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
