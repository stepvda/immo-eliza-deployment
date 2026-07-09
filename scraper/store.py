"""
scraper/store.py
================

The **resumable, partitioned storage layer**.

Two independent pieces of state let a crawl run for *days* over cron and pick up
exactly where it left off:

* **``seen_urls`` (SQLite)** at ``data/listings/state.sqlite`` — every listing
  URL we have already fetched. :func:`already_seen` / :func:`mark_seen` make the
  crawl idempotent: re-running never re-downloads a listing.
* **Raw partitioned Parquet** under
  ``data/listings/raw/<source>/<market>/<batch>.parquet`` — append-only capture
  from each adapter, one file per batch (:func:`append_raw`).

:func:`merge_market` reads *all* raw parquet for a market, de-duplicates across
sources (:func:`scraper.dedup.dedupe`), and writes the single canonical
``data/listings/<market>.parquet`` that the priciness surface and comparables
read. It is idempotent — re-running reproduces the same output.

Module-level path globals (``RAW_DIR``, ``LISTINGS_DIR``, ``STATE_DB``) can be
reassigned before calling any function to redirect all I/O — used by the
offline self-test to write into a temp dir instead of the real data tree.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from scraper.dedup import dedupe
from scraper.schema import CANONICAL_COLUMNS, canonicalize_frame
from scraper.schema import LISTINGS_DIR as _LISTINGS_DIR
from scraper.schema import RAW_DIR as _RAW_DIR

# Rebindable path globals (see module docstring). Point these elsewhere to
# redirect all storage I/O without touching the functions.
LISTINGS_DIR = _LISTINGS_DIR
RAW_DIR = _RAW_DIR
STATE_DB = _LISTINGS_DIR / "state.sqlite"


def _market_path(market: str) -> Path:
    """Canonical merged parquet for ``market`` (honours the module globals)."""
    return LISTINGS_DIR / f"{market}.parquet"


# --------------------------------------------------------------------------- #
# seen-URL state (SQLite)                                                      #
# --------------------------------------------------------------------------- #
def _conn() -> sqlite3.Connection:
    LISTINGS_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(STATE_DB)
    con.execute(
        "CREATE TABLE IF NOT EXISTS seen_urls ("
        "url TEXT PRIMARY KEY, source TEXT, market TEXT, ts TEXT)"
    )
    return con


def already_seen(url: str) -> bool:
    con = _conn()
    try:
        cur = con.execute("SELECT 1 FROM seen_urls WHERE url = ?", (url,))
        return cur.fetchone() is not None
    finally:
        con.close()


def mark_seen(url: str, source: Optional[str] = None, market: Optional[str] = None) -> None:
    con = _conn()
    try:
        con.execute(
            "INSERT OR IGNORE INTO seen_urls(url, source, market, ts) VALUES (?, ?, ?, ?)",
            (url, source, market, datetime.now(timezone.utc).isoformat()),
        )
        con.commit()
    finally:
        con.close()


def seen_count() -> int:
    con = _conn()
    try:
        return int(con.execute("SELECT COUNT(*) FROM seen_urls").fetchone()[0])
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# raw partitioned parquet                                                      #
# --------------------------------------------------------------------------- #
def _project(record: dict) -> dict:
    """Ensure a record has exactly the canonical columns (missing -> None)."""
    return {col: record.get(col) for col in CANONICAL_COLUMNS}


def _batch_id(records: list[dict]) -> str:
    """A batch filename: UTC timestamp + short content hash (collision-safe)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    urls = "".join(str(r.get("url") or "") for r in records)
    digest = hashlib.sha256(urls.encode("utf-8")).hexdigest()[:8]
    return f"{stamp}-{digest}"


def append_raw(source: str, market: str, records: list[dict]) -> Optional[Path]:
    """Append a batch of canonical records to the raw store for a source/market.

    Writes ``data/listings/raw/<source>/<market>/<batch>.parquet``. Returns the
    written path (or ``None`` when ``records`` is empty).
    """
    if not records:
        return None
    df = canonicalize_frame(pd.DataFrame([_project(r) for r in records],
                                         columns=CANONICAL_COLUMNS))
    out_dir = RAW_DIR / source / market
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_batch_id(records)}.parquet"
    df.to_parquet(path, index=False)
    return path


def _raw_files(market: str) -> list[Path]:
    if not RAW_DIR.exists():
        return []
    return sorted(RAW_DIR.glob(f"*/{market}/*.parquet"))


def merge_market(market: str, include_existing: bool = True) -> dict:
    """De-duplicate every raw batch for ``market`` into the canonical parquet.

    Idempotent and **additive**: reads all ``raw/<source>/<market>/*.parquet``
    *plus* the existing canonical file (the seed and any prior merge, unless
    ``include_existing=False``), runs the cross-site de-dup, and rewrites
    ``data/listings/<market>.parquet``. Folding in the existing file means a
    fresh crawl grows the store rather than replacing the seeded baseline. No-ops
    cleanly (no output file touched) when there is neither raw nor existing data.
    """
    files = _raw_files(market)
    existing_path = _market_path(market)
    has_existing = include_existing and existing_path.exists()
    if not files and not has_existing:
        return {
            "market": market, "raw_files": 0, "input": 0, "kept": 0,
            "duplicates_removed": 0, "output": None,
            "note": "no raw data — nothing to merge",
        }
    if not files:
        # Only the seed/existing file — nothing new to fold in.
        return {
            "market": market, "raw_files": 0, "input": 0, "kept": 0,
            "duplicates_removed": 0, "output": str(existing_path),
            "note": "no raw data — existing canonical left as-is",
        }

    frames = [pd.read_parquet(f) for f in files]
    if has_existing:
        frames.append(pd.read_parquet(existing_path))
    df = pd.concat(frames, ignore_index=True)
    # NaN -> None so downstream (dedup / null-counting) sees real absences.
    records = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in df.to_dict("records")
    ]

    kept, report = dedupe(records)
    out_df = canonicalize_frame(pd.DataFrame(kept, columns=CANONICAL_COLUMNS))
    LISTINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _market_path(market)
    out_df.to_parquet(out_path, index=False)

    report.update(market=market, raw_files=len(files), output=str(out_path))
    return report


# --------------------------------------------------------------------------- #
# Offline self-test                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    from scraper.schema import empty_record

    print("== scraper.store self-test (offline, temp dir) ==")

    tmp = Path(tempfile.mkdtemp(prefix="immo-store-"))
    # Redirect all I/O into the temp dir (no touching real data/).
    LISTINGS_DIR = tmp / "listings"
    RAW_DIR = LISTINGS_DIR / "raw"
    STATE_DB = LISTINGS_DIR / "state.sqlite"

    # seen-url state
    u = "https://www.immoweb.be/en/classified/1"
    assert not already_seen(u)
    mark_seen(u, "immoweb", "sale")
    mark_seen(u, "immoweb", "sale")  # idempotent
    assert already_seen(u) and seen_count() == 1
    print("seen_urls -> mark/already_seen ok, count =", seen_count())

    def mk(source, url, **kw):
        r = empty_record()
        r.update(source=source, url=url, market="sale", **kw)
        return r

    # two sources, one overlapping property
    immoweb_batch = [
        mk("immoweb", "https://www.immoweb.be/en/classified/1", street="Kreupelenstraat",
           house_number="12", postal_code="1000", livable_surface=85.0, bedrooms=2,
           price=849000.0, latitude=50.8506724, longitude=4.3562902,
           property_type="flat", epc="B"),
        mk("immoweb", "https://www.immoweb.be/en/classified/2", street="Rue Haute",
           house_number="7", postal_code="1000", livable_surface=120.0, bedrooms=3,
           price=650000.0, property_type="house"),
    ]
    realo_batch = [
        mk("realo", "https://www.realo.be/en/1", street="Kreupelenstraat",
           house_number="12", postal_code="1000", livable_surface=84.0, bedrooms=2,
           price=852000.0, latitude=50.8506900, longitude=4.3563100, property_type="flat"),
    ]
    p1 = append_raw("immoweb", "sale", immoweb_batch)
    p2 = append_raw("realo", "sale", realo_batch)
    print("wrote raw parquet:", p1.name, "|", p2.name)

    rep = merge_market("sale")
    print("merge report:", {k: rep[k] for k in ("raw_files", "input", "kept", "duplicates_removed")})
    assert rep["raw_files"] == 2 and rep["input"] == 3
    assert rep["duplicates_removed"] == 1 and rep["kept"] == 2, rep

    merged = pd.read_parquet(_market_path("sale"))
    assert len(merged) == 2
    print("canonical parquet rows:", len(merged), "cols:", len(merged.columns))

    # idempotent re-merge
    rep2 = merge_market("sale")
    assert rep2["kept"] == 2
    print("re-merge idempotent -> kept =", rep2["kept"])

    # empty-market no-op
    rep3 = merge_market("rent")
    assert rep3["raw_files"] == 0 and rep3["output"] is None
    print("empty market -> no-op:", rep3["note"])

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("OK")
