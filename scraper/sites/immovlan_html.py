"""
scraper/sites/immovlan_html.py
==============================

**Offline** extractor for Immovlan detail pages that were captured to disk
(``immo-eliza-scraping/data/html/*.html``). Immovlan hard-rate-limits live
crawling, but a folder of already-saved detail pages is a first-class source: we
parse each file's embedded **schema.org JSON-LD** blocks into a canonical record
— no network, no ToS concern (the pages are already captured).

Each page carries several ``application/ld+json`` blocks (the ``+`` is HTML-
encoded as ``&#x2B;`` in the markup, hence the tolerant matcher):

* an ``Apartment`` / ``House`` / ``Residence`` product block — property type,
  ``floorSize`` (m²), ``numberOfRooms`` (bedrooms), ``numberOfBathroomsTotal``,
  and an ``offers`` block with ``price`` / ``priceCurrency``;
* a ``GeoCoordinates`` block — ``latitude`` / ``longitude`` / ``postalCode``;
* a ``PostalAddress`` block — ``addressLocality`` / ``postalCode``.

We also recover a few extras from the raw HTML where cheap (EPC value, the
for-sale/for-rent market, the canonical URL). Everything else is left null for
the downstream imputer.

Usage::

    # parse the whole captured folder into the canonical store (source=immovlan)
    python -m scraper.sites.immovlan_html --html-dir /path/to/data/html --limit 0
    # quick self-test on a few files
    python -m scraper.sites.immovlan_html --html-dir /path/to/data/html --sample 3
"""
from __future__ import annotations

import argparse
import glob
import html as htmllib
import json
import os
import re
from typing import Iterator, Optional

from scraper.normalize import finalize, normalize_property_type, parse_price, parse_surface, to_int
from scraper.schema import empty_record

SOURCE = "immovlan"

# schema.org @type -> our coarse property_type intent
_TYPE_MAP = {
    "Apartment": "apartment", "House": "house", "SingleFamilyResidence": "house",
    "Residence": "house", "Product": None,
}


def _iter_ld_json(html: str) -> Iterator[dict]:
    """Yield each decoded JSON-LD object (tolerant of the &#x2B; entity + nesting)."""
    # Match <script ... ld ... json ...> BODY </script> without caring how "+"
    # is encoded, then brace-match the first balanced object in the body.
    for m in re.finditer(r'<script[^>]*ld[^>]*json[^>]*>(.*?)</script>', html, re.S | re.I):
        body = htmllib.unescape(m.group(1)).strip()
        obj = _first_json(body)
        if isinstance(obj, dict):
            yield obj
        elif isinstance(obj, list):
            for o in obj:
                if isinstance(o, dict):
                    yield o


def _first_json(text: str):
    """Decode the first balanced {...} or [...] JSON value in ``text``."""
    start = None
    for k, ch in enumerate(text):
        if ch in "{[":
            start = k
            break
    if start is None:
        return None
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = esc = False
    for k in range(start, len(text)):
        c = text[k]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:k + 1])
                except Exception:
                    return None
    return None


def _any_price(blocks: list[dict]):
    """First plausible numeric price across all JSON-LD blocks (offers or direct)."""
    for obj in blocks:
        offers = obj.get("offers")
        if isinstance(offers, list) and offers:
            offers = offers[0]
        if isinstance(offers, dict) and offers.get("price"):
            return offers["price"]
        if obj.get("price"):
            return obj["price"]
    return None


def _canonical_url(html: str, source_id: str) -> str:
    m = re.search(r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"', html, re.I) \
        or re.search(r'<meta[^>]+property="og:url"[^>]+content="([^"]+)"', html, re.I)
    return m.group(1) if m else f"https://immovlan.be/en/detail/{source_id}"


def _url_meta(url: str) -> dict:
    """Parse ``/detail/<type>/<for-sale|for-rent>/<postal>/<city>/<id>`` (the
    canonical URL is the authoritative market/type/location signal)."""
    m = re.search(r"/detail/([a-z\-]+)/(for-sale|for-rent)/(\d{4})/([^/]+)/", url, re.I)
    if not m:
        return {}
    return {
        "type_slug": m.group(1),
        "market": "rent" if m.group(2).lower() == "for-rent" else "sale",
        "postal_code": m.group(3),
        "city": m.group(4).replace("-", " ").title(),
    }


def parse_html(html: str, source_id: str) -> Optional[dict]:
    """Parse one Immovlan detail-page HTML string into a canonical record."""
    blocks = list(_iter_ld_json(html))
    prop = geo = addr = None
    for obj in blocks:
        t = obj.get("@type")
        if t in _TYPE_MAP and prop is None:
            prop = obj
        elif t == "GeoCoordinates" and geo is None:
            geo = obj
        elif t == "PostalAddress" and addr is None:
            addr = obj
    if prop is None and geo is None:
        return None
    prop = prop or {}
    geo = geo or {}
    addr = addr or {}

    url = _canonical_url(html, source_id)
    meta = _url_meta(url)
    price = _any_price(blocks)

    rec = empty_record()
    rec["source"] = SOURCE
    rec["listing_id"] = source_id
    rec["url"] = url
    rec["market"] = meta.get("market") or ("rent" if (price and float(price) < 8000) else "sale")

    # Property type: prefer the URL slug (villa/house/apartment/…), fall back to
    # the schema.org @type.
    rec["property_type"] = (normalize_property_type(meta.get("type_slug"))
                            or normalize_property_type(prop.get("@type")))
    rec["price"] = parse_price(price)
    fs = prop.get("floorSize")
    rec["livable_surface"] = parse_surface(fs.get("value") if isinstance(fs, dict) else fs)
    rec["bedrooms"] = to_int(prop.get("numberOfRooms") or prop.get("numberOfBedrooms"))
    rec["bathrooms"] = to_int(prop.get("numberOfBathroomsTotal"))

    rec["postal_code"] = addr.get("postalCode") or geo.get("postalCode") or meta.get("postal_code")
    rec["locality"] = addr.get("addressLocality") or meta.get("city")
    rec["municipality"] = addr.get("addressLocality") or meta.get("city")
    rec["latitude"] = geo.get("latitude")
    rec["longitude"] = geo.get("longitude")

    # cheap extras from the raw HTML
    epc = re.search(r'EPC[^0-9A-G]{0,20}([A-G])\b', html)
    if epc:
        from scraper.normalize import normalize_epc
        rec["epc"] = normalize_epc(epc.group(1))

    return rec


def parse_file(path: str) -> Optional[dict]:
    source_id = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_html(fh.read(), source_id)


def iter_records(html_dir: str, limit: int = 0) -> Iterator[dict]:
    files = sorted(glob.glob(os.path.join(html_dir, "*.html")))
    if limit:
        files = files[:limit]
    for i, path in enumerate(files):
        rec = parse_file(path)
        if rec and rec.get("price") and rec.get("latitude"):
            finalize(rec)
            yield rec


def ingest(html_dir: str, limit: int = 0, batch_size: int = 2000) -> dict:
    """Parse the captured folder and append canonical records to the raw store."""
    from datetime import datetime, timezone

    from scraper import store

    stamp = datetime.now(timezone.utc).isoformat()
    counts = {"sale": 0, "rent": 0}
    batches = {"sale": [], "rent": []}
    seen_files = 0
    for rec in iter_records(html_dir, limit):
        seen_files += 1
        rec["scraped_at"] = stamp
        m = rec["market"] if rec["market"] in ("sale", "rent") else "sale"
        batches[m].append(rec)
        counts[m] += 1
        if len(batches[m]) >= batch_size:
            store.append_raw(SOURCE, m, batches[m]); batches[m] = []
    for m, batch in batches.items():
        if batch:
            store.append_raw(SOURCE, m, batch)
    return {"parsed": seen_files, **counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract Immovlan captured HTML into the canonical store.")
    ap.add_argument("--html-dir", required=True, help="folder of *.html detail pages")
    ap.add_argument("--limit", type=int, default=0, help="max files (0 = all)")
    ap.add_argument("--sample", type=int, default=0, help="just print N parsed records (no writes)")
    args = ap.parse_args(argv)

    if args.sample:
        for path in sorted(glob.glob(os.path.join(args.html_dir, "*.html")))[:args.sample]:
            rec = parse_file(path)
            if not rec:
                print(f"{os.path.basename(path)}: NO DATA"); continue
            finalize(rec)
            keep = {k: rec[k] for k in ("market", "price", "livable_surface", "bedrooms",
                    "bathrooms", "property_type", "category", "postal_code", "locality",
                    "province", "latitude", "price_per_sqm", "refnis", "epc")}
            print(f"{os.path.basename(path)}: {keep}")
        return 0

    rep = ingest(args.html_dir, args.limit)
    print(f"immovlan HTML ingest: parsed {rep['parsed']} usable "
          f"(sale {rep['sale']}, rent {rep['rent']}) -> raw store")
    print("Run `python -m scraper.run --market sale --merge-only` and "
          "`--market rent --merge-only` to fold them into the canonical parquet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
