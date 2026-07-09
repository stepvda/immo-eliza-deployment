"""
scraper/dedup.py
================

Cross-site de-duplication. The same physical property is routinely listed on
several portals (Immoweb, Realo, Zimmo, ...), sometimes by different agencies,
with slightly different prices and free-text addresses. We collapse those into a
single canonical record.

Strategy (pure / offline-testable):

1. **Blocking key** — cheap exact-ish key that co-locates candidate duplicates::

       (postal_code, normalized_street + house_number, round(livable_surface/5), bedrooms)

2. **Within-block clustering** — inside a block, two records are treated as the
   same property only if their prices are within a band (~5%) *and*, when both
   carry coordinates, they sit within ~40 m of each other. This guards against
   two genuinely different flats that happen to share a block.

3. **Survivor selection** — keep the record with the most non-null canonical
   fields (:func:`scraper.base.count_filled`); ties are broken by the source's
   ``COMPLETENESS`` weight (a more reliable source wins).

:func:`dedupe` returns ``(kept, report)`` where ``report`` carries per-source
input/kept counts and the number of duplicates removed.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Optional

from scraper.base import count_filled

# tolerances -----------------------------------------------------------------
GEO_TOLERANCE_M = 40.0     # merge only if coordinates agree within this radius
PRICE_BAND = 0.05          # ...and prices within +/- 5%
SURFACE_BUCKET = 5.0       # livable_surface bucket width (m²) for the block key


def _norm_street(street) -> str:
    if not street:
        return ""
    s = unicodedata.normalize("NFKD", str(street))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def blocking_key(record: dict) -> tuple:
    """Coarse key that groups candidate duplicates together."""
    surface = record.get("livable_surface")
    surf_bucket = round(float(surface) / SURFACE_BUCKET) if surface else None
    street = _norm_street(record.get("street")) + str(record.get("house_number") or "")
    return (
        str(record.get("postal_code") or ""),
        street,
        surf_bucket,
        record.get("bedrooms"),
    )


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0  # metres
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _same_property(a: dict, b: dict) -> bool:
    """Finer test used *inside* a block: price band + (optional) geo proximity."""
    pa, pb = a.get("price"), b.get("price")
    if pa and pb and abs(pa - pb) > PRICE_BAND * max(pa, pb):
        return False
    la, lo = a.get("latitude"), a.get("longitude")
    lb, lob = b.get("latitude"), b.get("longitude")
    if None not in (la, lo, lb, lob):
        if _haversine_m(float(la), float(lo), float(lb), float(lob)) > GEO_TOLERANCE_M:
            return False
    return True


def _cluster(group: list[dict]) -> list[list[dict]]:
    """Greedy single-link clustering of a block into same-property clusters."""
    clusters: list[list[dict]] = []
    for rec in group:
        placed = False
        for cluster in clusters:
            if any(_same_property(rec, other) for other in cluster):
                cluster.append(rec)
                placed = True
                break
        if not placed:
            clusters.append([rec])
    return clusters


def _default_source_priority() -> dict[str, float]:
    """Per-source tie-break weight = sum of that adapter's COMPLETENESS profile."""
    try:
        from scraper.sites import ADAPTERS
    except Exception:
        return {}
    weights = {}
    for name, cls in ADAPTERS.items():
        try:
            weights[name] = float(sum(cls.COMPLETENESS.values()))
        except Exception:
            weights[name] = 0.0
    return weights


def _survivor(cluster: list[dict], source_priority: dict[str, float]) -> dict:
    """The richest record in the cluster, with its **null fields coalesced** from
    the duplicates (higher-priority sources first). Collapsing duplicates should
    not throw away information the winner happens to lack — most importantly a
    ``url`` back to the listing, but also any other field only a sibling filled.
    """
    best = max(
        cluster,
        key=lambda r: (count_filled(r), source_priority.get(r.get("source"), 0.0)),
    )
    if len(cluster) == 1:
        return best
    merged = dict(best)
    others = sorted(
        (r for r in cluster if r is not best),
        key=lambda r: source_priority.get(r.get("source"), 0.0), reverse=True,
    )
    for rec in others:
        for key, value in rec.items():
            if merged.get(key) in (None, "") and value not in (None, ""):
                merged[key] = value
    return merged


def dedupe(records: list[dict], source_priority: Optional[dict[str, float]] = None):
    """De-duplicate ``records`` across sources.

    Returns ``(kept, report)``. ``report`` is a dict with total input/kept,
    duplicates removed, and per-source input/kept counters.
    """
    if source_priority is None:
        source_priority = _default_source_priority()

    blocks: dict[tuple, list[dict]] = {}
    for rec in records:
        blocks.setdefault(blocking_key(rec), []).append(rec)

    kept: list[dict] = []
    removed = 0
    for group in blocks.values():
        for cluster in _cluster(group):
            kept.append(_survivor(cluster, source_priority))
            removed += len(cluster) - 1

    per_source_input = Counter(r.get("source") for r in records)
    per_source_kept = Counter(r.get("source") for r in kept)
    report = {
        "input": len(records),
        "kept": len(kept),
        "duplicates_removed": removed,
        "per_source_input": dict(per_source_input),
        "per_source_kept": dict(per_source_kept),
        "source_priority": source_priority,
    }
    return kept, report


# --------------------------------------------------------------------------- #
# Offline self-test                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from scraper.schema import empty_record

    print("== scraper.dedup self-test (offline) ==")

    def mk(source, **kw):
        r = empty_record()
        r["source"] = source
        r.update(kw)
        return r

    # A: same flat on two portals (near-identical geo + price) -> 1 survivor.
    #    immoweb record is richer (more filled fields) -> it should win.
    a_immoweb = mk(
        "immoweb", street="Kreupelenstraat", house_number="12", postal_code="1000",
        livable_surface=85.0, bedrooms=2, price=849000.0,
        latitude=50.8506724, longitude=4.3562902,
        property_type="flat", epc="B", bathrooms=2, heating_type="Gas",
    )
    a_realo = mk(
        "realo", street="Kreupelenstraat", house_number="12", postal_code="1000",
        livable_surface=84.0, bedrooms=2, price=855000.0,  # ~4m / 0.7% away -> same
        latitude=50.8506900, longitude=4.3563100, property_type="flat",
    )
    # B: genuinely different, distinct block -> survives on its own.
    b_unique = mk(
        "realo", street="Rue Neuve", house_number="5", postal_code="1000",
        livable_surface=78.0, bedrooms=2, price=329000.0,
        latitude=50.8512, longitude=4.3543, property_type="flat",
    )

    kept, report = dedupe([a_immoweb, a_realo, b_unique])
    print("report:", report)
    assert report["input"] == 3
    assert report["duplicates_removed"] == 1, report
    assert report["kept"] == 2, report
    # richer immoweb copy of property A survived
    survivor_a = [r for r in kept if r["street"] == "Kreupelenstraat"]
    assert survivor_a and survivor_a[0]["source"] == "immoweb", "richest record must win"
    print("kept sources:", sorted(r["source"] for r in kept))

    # Same block but far apart / different price -> NOT merged.
    c1 = mk("immoweb", street="Grand Place", house_number="1", postal_code="1000",
            livable_surface=100.0, bedrooms=3, price=500000.0,
            latitude=50.8467, longitude=4.3525)
    c2 = mk("realo", street="Grand Place", house_number="1", postal_code="1000",
            livable_surface=100.0, bedrooms=3, price=1200000.0,  # price band fails
            latitude=50.8467, longitude=4.3525)
    kept2, report2 = dedupe([c1, c2])
    assert report2["duplicates_removed"] == 0, report2
    print("distinct-price same-block -> kept both:", report2["kept"])

    print("OK")
