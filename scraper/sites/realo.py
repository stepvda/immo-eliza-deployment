"""
scraper/sites/realo.py
======================

Concrete :class:`~scraper.base.SiteAdapter` for **Realo** (realo.be).

How Realo exposes data
----------------------
Realo renders listings from a JSON estate object (delivered via its internal
JSON/GraphQL API and also embedded in the detail page). Compared with Immoweb it
is a little less field-rich, which is reflected in a lower ``COMPLETENESS``
profile — so when the *same* property appears on both portals, the de-dup layer
prefers the Immoweb copy on ties.

:meth:`RealoAdapter.parse` accepts a decoded ``dict`` (see :data:`REALO_FIXTURE`)
or a raw HTML/JSON string. As with Immoweb, the live endpoints and markup drift
and are anti-bot protected — the field mapping is realistic but the search URLs
and URL-extraction regex should be re-confirmed against an authorised capture.
"""
from __future__ import annotations

import json
import re
from typing import Iterator, Optional

from scraper.base import SiteAdapter
from scraper.normalize import (
    normalize_building_state,
    normalize_epc,
    normalize_heating,
    normalize_kitchen,
    normalize_property_type,
    parse_price,
    parse_surface,
    to_binary_flag,
    to_float,
    to_int,
)
from scraper.schema import empty_record

# Realo uses lower-case city/region slugs; iterating regions x pages gives broad
# coverage. Widen MAX_PAGES_PER_REGION under cron toward the 30k/100k targets.
REGIONS = ("flanders", "brussels", "wallonia")
MAX_PAGES_PER_REGION = 5  # validation-sample cap


def _estate_from_html(text: str) -> Optional[dict]:
    """Extract a Realo estate JSON object from a detail-page HTML string."""
    # Common shapes: an inline `window.__ESTATE__ = {...}` or a JSON-LD/estate
    # script. Try a couple of markers, then fall back to raw JSON.
    for marker in ('"estate":', "window.__ESTATE__", '"@type":"Residence"'):
        i = text.find(marker)
        if i < 0:
            continue
        j = text.find("{", i if marker.startswith("window") else i + len(marker) - 1)
        if j < 0:
            continue
        depth, in_str, esc = 0, False, False
        for k in range(j, len(text)):
            c = text[k]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[j:k + 1])
                        except Exception:
                            break
    try:
        return json.loads(text)
    except Exception:
        return None


class RealoAdapter(SiteAdapter):
    name = "realo"
    base_url = "https://www.realo.be"

    # Realo is slightly less complete than Immoweb -> lower weights so Immoweb
    # wins de-dup ties for the same property.
    COMPLETENESS = {
        "property_type": 0.95, "livable_surface": 0.9, "land_surface": 0.5,
        "bedrooms": 0.95, "bathrooms": 0.8, "toilets": 0.4, "build_year": 0.6,
        "facades": 0.5, "number_of_floors": 0.3, "primary_energy_consumption": 0.7,
        "epc": 0.8, "building_state": 0.5, "kitchen_equipment": 0.5,
        "heating_type": 0.5, "price": 1.0, "latitude": 0.9, "longitude": 0.9,
        "street": 0.75, "postal_code": 1.0, "terrace": 0.8, "garden": 0.8,
        "elevator": 0.7, "cellar": 0.5, "swimming_pool": 0.7, "has_parking": 0.7,
    }

    _TXN = {"sale": "for-sale", "rent": "to-rent"}

    def iter_search_urls(self, market: str) -> Iterator[str]:
        """Paginated Realo search URLs across regions.

        Real endpoint shape (subject to change)::

            https://www.realo.be/en/search/for-sale/flanders?page=1
        """
        txn = self._TXN.get(market, "for-sale")
        for region in REGIONS:
            for page in range(1, MAX_PAGES_PER_REGION + 1):
                yield f"{self.base_url}/en/search/{txn}/{region}?page={page}"

    def extract_listing_urls(self, payload, base_url: str) -> list[str]:
        """Pull Realo detail URLs from a search payload.

        Realo detail paths look like ``/en/apartment-for-sale/brussels/1000/987654``
        (a slug ending in a numeric id). Returns a de-duplicated absolute list.
        """
        text = payload if isinstance(payload, str) else json.dumps(payload)
        urls: list[str] = []
        seen = set()
        for m in re.finditer(r'/en/[a-z0-9\-]+/[a-z0-9\-]+/\d{4}/\d{4,}', text):
            href = m.group(0)
            if href not in seen:
                seen.add(href)
                urls.append(self.absolutise(href))
        return urls

    def parse(self, payload, url: str, market: str) -> Optional[dict]:
        """Map one Realo estate object (dict or HTML/JSON str) to canonical."""
        data = payload if isinstance(payload, dict) else _estate_from_html(payload)
        if not data or not isinstance(data, dict):
            return None

        geo = data.get("geo") or data.get("location") or {}
        energy = data.get("energy") or {}
        features = data.get("features") or {}

        rec = empty_record()

        rec["listing_id"] = str(data.get("id")) if data.get("id") is not None else None
        rec["source"] = self.name
        rec["url"] = url
        rec["market"] = market

        rec["street"] = geo.get("street")
        rec["house_number"] = geo.get("number")
        rec["postal_code"] = geo.get("zip") or geo.get("postalCode")
        rec["locality"] = geo.get("city") or geo.get("locality")
        rec["municipality"] = geo.get("city") or geo.get("locality")
        rec["latitude"] = to_float(geo.get("latitude"))
        rec["longitude"] = to_float(geo.get("longitude"))

        rec["property_type"] = normalize_property_type(data.get("subtype") or data.get("type"))
        rec["livable_surface"] = parse_surface(data.get("surface") or data.get("livableSurface"))
        rec["land_surface"] = parse_surface(data.get("landSurface"))
        rec["bedrooms"] = to_int(data.get("bedrooms"))
        rec["bathrooms"] = to_int(data.get("bathrooms"))
        rec["toilets"] = to_int(data.get("toilets"))
        rec["build_year"] = to_int(data.get("buildYear") or data.get("constructionYear"))
        rec["facades"] = to_int(data.get("facades"))
        rec["number_of_floors"] = to_int(data.get("floors"))
        rec["primary_energy_consumption"] = to_float(energy.get("consumption"))

        rec["building_state"] = normalize_building_state(data.get("condition"))
        rec["kitchen_equipment"] = normalize_kitchen(data.get("kitchen"))
        rec["heating_type"] = normalize_heating(energy.get("heating"))
        rec["epc"] = normalize_epc(energy.get("epc") or data.get("epc"))

        rec["price"] = parse_price(data.get("price"))

        rec["new_construction"] = to_binary_flag(features.get("newConstruction"))
        rec["furnished"] = to_binary_flag(features.get("furnished"))
        rec["terrace"] = to_binary_flag(features.get("terrace"))
        rec["garden"] = to_binary_flag(features.get("garden"))
        rec["swimming_pool"] = to_binary_flag(features.get("pool") or features.get("swimmingPool"))
        rec["elevator"] = to_binary_flag(features.get("elevator") or features.get("lift"))
        rec["cellar"] = to_binary_flag(features.get("cellar") or features.get("basement"))
        rec["solar_panels"] = to_binary_flag(features.get("solarPanels"))
        rec["air_conditioning"] = to_binary_flag(features.get("airConditioning"))
        rec["has_parking"] = to_binary_flag(features.get("parking"))

        return rec


# --------------------------------------------------------------------------- #
# Offline fixture — a realistic Realo estate JSON object                       #
# --------------------------------------------------------------------------- #
REALO_FIXTURE = {
    "id": 987654,
    "url": "/en/apartment-for-sale/brussels/1000/987654",
    "transactionType": "SALE",
    "type": "APARTMENT",
    "subtype": "APARTMENT",
    "price": 329000,
    "geo": {
        "street": "Rue Neuve",
        "number": "5",
        "zip": "1000",
        "city": "Brussels",
        "latitude": 50.8512,
        "longitude": 4.3543,
    },
    "bedrooms": 2,
    "bathrooms": 1,
    "toilets": 1,
    "surface": 78,
    "landSurface": None,
    "buildYear": 2010,
    "facades": 2,
    "floors": None,
    "condition": "GOOD",
    "kitchen": "INSTALLED",
    "energy": {"epc": "C", "consumption": 180, "heating": "GAS"},
    "features": {
        "terrace": True, "garden": False, "pool": False, "elevator": True,
        "cellar": False, "solarPanels": False, "airConditioning": False,
        "parking": True, "furnished": False, "newConstruction": False,
    },
}

REALO_FIXTURE_HTML = (
    '<html><head><script type="application/json">'
    '{"estate":' + json.dumps(REALO_FIXTURE) + "}"
    "</script></head><body>...</body></html>"
)


if __name__ == "__main__":
    from scraper.normalize import finalize

    print("== scraper.sites.realo self-test (offline) ==")
    adapter = RealoAdapter()
    url = "https://www.realo.be/en/apartment-for-sale/brussels/1000/987654"

    rec = adapter.parse(REALO_FIXTURE, url, "sale")
    assert rec is not None
    finalize(rec)

    rec_html = adapter.parse(REALO_FIXTURE_HTML, url, "sale")
    finalize(rec_html)
    assert rec_html["price"] == rec["price"] == 329000.0
    assert rec["property_type"] == "flat" and rec["category"] == "apartment"
    assert rec["epc"] == "C" and rec["kitchen_equipment"] == "Fully equipped"
    assert rec["heating_type"] == "Gas" and rec["building_state"] == "B"
    assert rec["elevator"] == 1 and rec["terrace"] == 1 and rec["garden"] == 0
    assert rec["price_per_sqm"] == round(329000 / 78, 2)
    assert rec["province"] and rec["refnis"]  # crosswalk: 1000 -> Brussels-Capital / 21004

    search_urls = list(adapter.iter_search_urls("sale"))
    print("search urls:", len(search_urls), "e.g.", search_urls[0])
    found = adapter.extract_listing_urls(
        '<a href="/en/apartment-for-sale/brussels/1000/987654">x</a>', adapter.base_url
    )
    print("extract_listing_urls ->", found)

    print("\ncanonical record:")
    for k, v in rec.items():
        if v is not None:
            print(f"  {k:28s} = {v!r}")
    print("OK")
