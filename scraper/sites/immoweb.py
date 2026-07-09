"""
scraper/sites/immoweb.py
========================

Concrete :class:`~scraper.base.SiteAdapter` for **Immoweb** (immoweb.be), the
largest Belgian portal.

How Immoweb exposes data
------------------------
Each listing detail page ships a big JSON blob describing the classified. In
practice it appears either as an inline ``window.classified = { ... };``
assignment or inside the Next.js ``<script id="__NEXT_DATA__">`` payload
(``props.pageProps.classified``). :meth:`ImmowebAdapter.parse` accepts *either*
a raw HTML string (from which it extracts and json-decodes that blob) *or* an
already-decoded ``dict`` — the latter makes it unit-testable offline against the
:data:`IMMOWEB_FIXTURE` below.

.. warning::
   The exact live endpoints, the search-result markup, and the field paths
   drift over time and Immoweb fronts everything with anti-bot protection. The
   mapping here is realistic and faithful to the classified JSON shape, but
   treat the search URLs / URL-extraction regexes as *starting points* to
   re-confirm against a live (authorised) capture before a production crawl.
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

# Immoweb's search-results JSON API is paginated (~30 listings/page). Its
# province filter is not honoured by this endpoint, so we paginate pages across
# the two property groups (house / apartment) and let orderBy=newest advance the
# frontier. Widen MAX_PAGES under cron for the 100k target.
PROPERTY_GROUPS = ("house", "apartment")
MAX_PAGES = 120  # a pass stops at --max or the run's time box, whichever first


def _extract_json_object(text: str, marker: str) -> Optional[dict]:
    """Extract the first balanced ``{...}`` JSON object appearing after ``marker``.

    Brace matching is string/escape aware, so ``}`` inside string values does
    not terminate the object prematurely.
    """
    i = text.find(marker)
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    in_str = False
    esc = False
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
                        return None
    return None


def _classified_from_html(text: str) -> Optional[dict]:
    """Pull the classified JSON out of a detail-page HTML string."""
    obj = _extract_json_object(text, "window.classified")
    if obj is not None:
        return obj
    nxt = _extract_json_object(text, "__NEXT_DATA__")
    if nxt is not None:
        try:
            return nxt["props"]["pageProps"]["classified"]
        except Exception:
            return None
    # Maybe the payload is already raw JSON.
    try:
        return json.loads(text)
    except Exception:
        return None


class ImmowebAdapter(SiteAdapter):
    name = "immoweb"
    base_url = "https://www.immoweb.be"

    # Which canonical fields Immoweb reliably provides (0..1). Used to
    # prioritise the source and to break de-dup ties. Immoweb is field-rich.
    COMPLETENESS = {
        "property_type": 1.0, "livable_surface": 0.95, "land_surface": 0.6,
        "bedrooms": 0.98, "bathrooms": 0.9, "toilets": 0.7, "build_year": 0.7,
        "facades": 0.8, "number_of_floors": 0.4, "primary_energy_consumption": 0.85,
        "epc": 0.9, "building_state": 0.7, "kitchen_equipment": 0.8,
        "heating_type": 0.7, "price": 1.0, "latitude": 0.85, "longitude": 0.85,
        "street": 0.8, "postal_code": 1.0, "terrace": 0.9, "garden": 0.9,
        "elevator": 0.8, "cellar": 0.7, "swimming_pool": 0.9, "has_parking": 0.7,
    }

    _TXN = {"sale": "for-sale", "rent": "for-rent"}

    def iter_search_urls(self, market: str) -> Iterator[str]:
        """Paginated **search-results JSON API** URLs (confirmed live endpoint).

        Endpoint shape::

            https://www.immoweb.be/en/search-results/house/for-sale
                ?countries=BE&page=1&orderBy=newest

        Returns a JSON body ``{"results": [{"id": ...}, ...]}``. Pages are
        interleaved across the two property groups so a short (time-boxed) pass
        still samples both houses and apartments.
        """
        txn = self._TXN.get(market, "for-sale")
        for page in range(1, MAX_PAGES + 1):
            for group in PROPERTY_GROUPS:
                yield (
                    f"{self.base_url}/en/search-results/{group}/{txn}"
                    f"?countries=BE&page={page}&orderBy=newest"
                )

    def extract_listing_urls(self, payload, base_url: str) -> list[str]:
        """Pull classified detail URLs out of a search-results payload.

        Preferred path: decode the JSON and take ``results[].id`` (precise —
        avoids grabbing unrelated media/customer ids). Falls back to regex over
        an HTML search page. Returns absolutised ``/en/classified/<id>`` URLs.
        """
        urls: list[str] = []
        seen: set[str] = set()

        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
        except Exception:
            data = None
        results = data.get("results") if isinstance(data, dict) else None
        if results:
            for item in results:
                lid = item.get("id") if isinstance(item, dict) else None
                if lid is not None and str(lid) not in seen:
                    seen.add(str(lid))
                    urls.append(f"{self.base_url}/en/classified/{lid}")
            return urls

        # Fallback: an HTML search page.
        text = payload if isinstance(payload, str) else json.dumps(payload)
        for m in re.finditer(r'/en/classified/[a-z0-9\-/]+/\d{6,}', text):
            href = m.group(0)
            if href not in seen:
                seen.add(href)
                urls.append(self.absolutise(href))
        if not urls:
            for m in re.finditer(r'"id"\s*:\s*(\d{6,})', text):
                if m.group(1) not in seen:
                    seen.add(m.group(1))
                    urls.append(f"{self.base_url}/en/classified/{m.group(1)}")
        return urls

    def parse(self, payload, url: str, market: str) -> Optional[dict]:
        """Map one Immoweb classified (HTML str or decoded dict) to canonical."""
        data = payload if isinstance(payload, dict) else _classified_from_html(payload)
        if not data or not isinstance(data, dict):
            return None

        prop = data.get("property") or {}
        txn = data.get("transaction") or {}
        loc = prop.get("location") or {}
        building = prop.get("building") or {}
        kitchen = prop.get("kitchen") or {}
        energy = prop.get("energy") or {}
        certs = txn.get("certificates") or {}
        sale = txn.get("sale") or {}
        rental = txn.get("rental") or {}
        flags = data.get("flags") or {}

        rec = empty_record()

        # provenance / identity (scraped_at is stamped by the runner)
        rec["listing_id"] = str(data.get("id")) if data.get("id") is not None else None
        rec["source"] = self.name
        rec["url"] = url
        rec["market"] = market

        # geography (province/region get authoritatively re-derived in finalize)
        rec["street"] = loc.get("street")
        rec["house_number"] = loc.get("number")
        rec["postal_code"] = loc.get("postalCode")
        rec["locality"] = loc.get("locality")
        rec["municipality"] = loc.get("locality")
        rec["province"] = loc.get("province")
        rec["region"] = loc.get("region")
        rec["latitude"] = to_float(loc.get("latitude"))
        rec["longitude"] = to_float(loc.get("longitude"))

        # numerics
        rec["property_type"] = normalize_property_type(prop.get("subtype") or prop.get("type"))
        rec["livable_surface"] = parse_surface(prop.get("netHabitableSurface"))
        rec["land_surface"] = parse_surface((prop.get("land") or {}).get("surface"))
        rec["bedrooms"] = to_int(prop.get("bedroomCount"))
        rec["bathrooms"] = to_int(prop.get("bathroomCount"))
        rec["toilets"] = to_int(prop.get("toiletCount"))
        rec["build_year"] = to_int(building.get("constructionYear"))
        rec["facades"] = to_int(building.get("facadeCount"))
        rec["number_of_floors"] = to_int(building.get("floorCount"))
        rec["primary_energy_consumption"] = to_float(certs.get("primaryEnergyConsumptionPerSqm"))

        # categoricals
        rec["building_state"] = normalize_building_state(building.get("condition"))
        rec["kitchen_equipment"] = normalize_kitchen(kitchen.get("type"))
        rec["heating_type"] = normalize_heating(energy.get("heatingType"))
        rec["epc"] = normalize_epc(certs.get("epcScore"))

        # price (sale price or monthly rent)
        rec["price"] = parse_price(sale.get("price") or rental.get("monthlyRentalPrice"))

        # amenity flags
        parking = (prop.get("parkingCountIndoor") or 0) + (prop.get("parkingCountOutdoor") or 0)
        rec["new_construction"] = to_binary_flag(sale.get("isNewlyBuilt") or flags.get("isNewlyBuilt"))
        rec["furnished"] = to_binary_flag(prop.get("isFurnished"))
        rec["terrace"] = to_binary_flag(prop.get("hasTerrace"))
        rec["garden"] = to_binary_flag(prop.get("hasGarden"))
        rec["swimming_pool"] = to_binary_flag(prop.get("hasSwimmingPool"))
        rec["elevator"] = to_binary_flag(prop.get("hasLift"))
        rec["cellar"] = to_binary_flag(prop.get("hasBasement"))
        rec["solar_panels"] = to_binary_flag(prop.get("hasSolarPanels"))
        rec["air_conditioning"] = to_binary_flag(prop.get("hasAirConditioning"))
        rec["has_parking"] = to_binary_flag(parking)

        return rec


# --------------------------------------------------------------------------- #
# Offline fixture — a realistic Immoweb classified JSON blob                   #
# --------------------------------------------------------------------------- #
IMMOWEB_FIXTURE = {
    "id": 12345678,
    "property": {
        "type": "APARTMENT",
        "subtype": "APARTMENT",
        "bedroomCount": 2,
        "bathroomCount": 2,
        "toiletCount": 2,
        "netHabitableSurface": 85,
        "land": {"surface": None},
        "building": {
            "condition": "GOOD",
            "constructionYear": 2005,
            "facadeCount": 2,
            "floorCount": 4,
        },
        "kitchen": {"type": "HYPER_EQUIPPED"},
        "energy": {"heatingType": "GAS"},
        "location": {
            "street": "Kreupelenstraat",
            "number": "12",
            "postalCode": "1000",
            "locality": "Brussels",
            "province": "Brussels",
            "region": "Brussels",
            "latitude": 50.8506724,
            "longitude": 4.3562902,
        },
        "hasLift": True,
        "hasBasement": True,
        "hasGarden": False,
        "hasTerrace": True,
        "hasSwimmingPool": False,
        "hasAirConditioning": False,
        "hasSolarPanels": False,
        "isFurnished": False,
        "parkingCountIndoor": 1,
        "parkingCountOutdoor": 0,
    },
    "transaction": {
        "type": "FOR_SALE",
        "sale": {"price": 849000, "isNewlyBuilt": False},
        "certificates": {"epcScore": "B", "primaryEnergyConsumptionPerSqm": 150},
    },
    "flags": {"isNewlyBuilt": False},
}

# The same blob as it would appear embedded in a detail page, to exercise the
# HTML-extraction path (string/escape aware brace matching).
IMMOWEB_FIXTURE_HTML = (
    "<!doctype html><html><head><script>"
    "window.classified = " + json.dumps(IMMOWEB_FIXTURE) + ";"
    "</script></head><body>...</body></html>"
)


if __name__ == "__main__":
    from scraper.normalize import finalize

    print("== scraper.sites.immoweb self-test (offline) ==")
    adapter = ImmowebAdapter()
    url = "https://www.immoweb.be/en/classified/apartment/for-sale/brussels/1000/12345678"

    # dict path
    rec = adapter.parse(IMMOWEB_FIXTURE, url, "sale")
    assert rec is not None
    finalize(rec)

    # HTML-extraction path must yield the identical mapping
    rec_html = adapter.parse(IMMOWEB_FIXTURE_HTML, url, "sale")
    finalize(rec_html)
    assert rec_html["price"] == rec["price"] == 849000.0
    assert rec["property_type"] == "flat" and rec["category"] == "apartment"
    assert rec["epc"] == "B" and rec["kitchen_equipment"] == "Super equipped"
    assert rec["heating_type"] == "Gas" and rec["building_state"] == "B"
    assert rec["elevator"] == 1 and rec["has_parking"] == 1 and rec["garden"] == 0
    assert rec["price_per_sqm"] == round(849000 / 85, 2)
    assert rec["province"] == "Brussels" and rec["refnis"]

    # a couple of search + listing-url examples
    search_urls = list(adapter.iter_search_urls("sale"))
    print("search urls:", len(search_urls), "e.g.", search_urls[0])
    found = adapter.extract_listing_urls(
        '<a href="/en/classified/apartment/for-sale/brussels/1000/12345678">x</a>', adapter.base_url
    )
    print("extract_listing_urls ->", found)

    print("\ncanonical record:")
    for k, v in rec.items():
        if v is not None:
            print(f"  {k:28s} = {v!r}")
    print("OK")
