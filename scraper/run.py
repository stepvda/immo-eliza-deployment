"""
scraper/run.py
==============

Orchestrator CLI that ties the framework together for one polite crawl pass.

Usage
-----
::

    python -m scraper.run --sites immoweb,realo --market sale --max 200
    python -m scraper.run --market sale --merge-only        # just re-merge raw
    python -m scraper.run --sites immoweb --market rent --max 50 --no-robots

Flow (per adapter)
------------------
1. iterate the adapter's search URLs,
2. fetch each search page (politely, cached, robots-respecting),
3. extract listing detail URLs, skip any already in ``seen_urls``,
4. fetch + :meth:`parse` each new listing into a canonical record,
5. :func:`scraper.normalize.finalize` it (category / price_per_sqm / geography),
6. append the batch to the raw partitioned store and mark URLs seen,
7. finally :func:`scraper.store.merge_market` de-dups all raw into the canonical
   ``data/listings/<market>.parquet``.

Because major Belgian portals sit behind Cloudflare/Datadome and forbid scraping
in their ToS — and in-session networking is typically blocked — a real fetch
here will usually return nothing. That is by design: the engine **degrades
gracefully**, logs the condition, and the merge step still runs. A small ``--max``
run is a *validation sample*, never the 100k/30k production target (see the
banner the CLI prints and ``scraper/README.md`` for the cron scale-out).
"""
from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timezone

from scraper import store
from scraper.base import Fetcher
from scraper.normalize import finalize
from scraper.schema import MARKETS
from scraper.sites import ADAPTERS

# Production targets — a small CLI run only *samples* toward these.
TARGET_SALE = 100_000
TARGET_RENT = 30_000

# Circuit breaker: if this many search pages in a row are unreachable *before*
# anything parses, assume the site is blocked (Cloudflare/Datadome/no network)
# and stop crawling it quickly instead of grinding every page through retries.
UNREACHABLE_CIRCUIT_BREAKER = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _crawl_site(fetcher: Fetcher, adapter, market: str, max_listings: int,
                      deadline: float | None = None) -> dict:
    """Crawl a single site up to ``max_listings`` new listings. Returns stats.

    Stops gracefully (writing the batch + letting the caller merge) when either
    ``max_listings`` is reached or the optional monotonic ``deadline`` passes.
    """
    stats = {
        "site": adapter.name, "search_pages": 0, "search_unreachable": 0,
        "listings_found": 0, "already_seen": 0, "fetched": 0, "parsed": 0,
        "written": 0,
    }
    batch: list[dict] = []
    consecutive_unreachable = 0

    for search_url in adapter.iter_search_urls(market):
        if stats["parsed"] >= max_listings:
            break
        if deadline and time.monotonic() > deadline:
            stats["time_limited"] = True
            break
        stats["search_pages"] += 1
        page = await fetcher.get(search_url)
        if page is None:
            stats["search_unreachable"] += 1
            consecutive_unreachable += 1
            # Bail out fast if the site is clearly unreachable and we have no
            # data yet (blocked / anti-bot / no network) rather than retrying
            # every remaining search page.
            if consecutive_unreachable >= UNREACHABLE_CIRCUIT_BREAKER and stats["parsed"] == 0:
                stats["circuit_broken"] = True
                break
            continue
        consecutive_unreachable = 0

        for listing_url in adapter.extract_listing_urls(page.text, adapter.base_url):
            if stats["parsed"] >= max_listings:
                break
            if deadline and time.monotonic() > deadline:
                stats["time_limited"] = True
                break
            stats["listings_found"] += 1
            if store.already_seen(listing_url):
                stats["already_seen"] += 1
                continue
            detail = await fetcher.get(listing_url)
            if detail is None:
                continue
            stats["fetched"] += 1
            record = adapter.parse(detail.text, listing_url, market)
            if not record:
                continue
            record["scraped_at"] = _now_iso()
            finalize(record)
            batch.append(record)
            store.mark_seen(listing_url, adapter.name, market)
            stats["parsed"] += 1

    if batch:
        path = store.append_raw(adapter.name, market, batch)
        stats["written"] = len(batch)
        stats["raw_path"] = str(path)
    return stats


async def _run(sites: list[str], market: str, max_listings: int,
               respect_robots: bool, rpm: int, time_limit: int = 0) -> list[dict]:
    fetcher = Fetcher(requests_per_minute=rpm, respect_robots=respect_robots)
    deadline = time.monotonic() + time_limit if time_limit else None
    all_stats = []
    try:
        for name in sites:
            adapter_cls = ADAPTERS[name]
            print(f"\n--- crawling {name} ({market}, max {max_listings}"
                  f"{f', {time_limit}s limit' if time_limit else ''}) ---")
            stats = await _crawl_site(fetcher, adapter_cls(), market, max_listings, deadline)
            all_stats.append(stats)
            if stats["written"] == 0 and stats["search_unreachable"] == stats["search_pages"]:
                print(f"[{name}] all {stats['search_pages']} search pages unreachable "
                      f"(network blocked / anti-bot / robots) — degraded gracefully, no data.")
            else:
                print(f"[{name}] parsed {stats['parsed']} new listings "
                      f"(found {stats['listings_found']}, already-seen {stats['already_seen']}).")
    finally:
        await fetcher.aclose()
    return all_stats


def _print_scale_banner(market: str) -> None:
    target = TARGET_SALE if market == "sale" else TARGET_RENT
    print("\n" + "=" * 72)
    print("  VALIDATION SAMPLE — this is NOT the production target.")
    print(f"  Target for '{market}': {target:,} deduplicated listings across many")
    print("  sites with dense geographic coverage, gathered over DAYS via cron.")
    print("  Scale out by:")
    print("    * registering more adapters in scraper/sites/__init__.py,")
    print("    * widening MAX_PAGES_PER_* in each adapter (province/region x pages),")
    print("    * running hourly cron passes with a small --max and low --rpm so the")
    print("      seen_urls DB advances the frontier a little each run, e.g.:")
    print("        0 * * * * cd <repo> && .venv/bin/python -m scraper.run \\")
    print("                    --sites immoweb,realo --market sale --max 500 --rpm 15")
    print("  See scraper/README.md for the full cron recipe, proxy + legal guidance.")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scraper.run",
        description="Polite, resumable, multi-site Belgian real-estate crawl (one pass).",
    )
    parser.add_argument("--sites", default="immoweb,realo",
                        help="comma-separated adapter names (default: immoweb,realo)")
    parser.add_argument("--market", default="sale", choices=list(MARKETS),
                        help="market to crawl (default: sale)")
    parser.add_argument("--max", type=int, default=200, dest="max_listings",
                        help="max NEW listings per site this pass (default: 200)")
    parser.add_argument("--rpm", type=int, default=20,
                        help="per-domain requests/minute (politeness; default: 20)")
    parser.add_argument("--time-limit", type=int, default=0, dest="time_limit",
                        help="stop crawling after N seconds (0 = no limit); still merges")
    parser.add_argument("--no-robots", action="store_true",
                        help="do NOT enforce robots.txt (discouraged; you own the risk)")
    parser.add_argument("--merge-only", action="store_true",
                        help="skip fetching; just re-merge existing raw parquet")
    args = parser.parse_args(argv)

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    unknown = [s for s in sites if s not in ADAPTERS]
    if unknown:
        parser.error(f"unknown site(s): {unknown}. Known: {sorted(ADAPTERS)}")

    if not args.merge_only:
        stats = asyncio.run(_run(sites, args.market, args.max_listings,
                                 respect_robots=not args.no_robots, rpm=args.rpm,
                                 time_limit=args.time_limit))
        total_written = sum(s["written"] for s in stats)
        print(f"\ncrawl pass complete: {total_written} raw records written across "
              f"{len(sites)} site(s); seen_urls now holds {store.seen_count()} URLs.")

    print(f"\nmerging market '{args.market}' ...")
    report = store.merge_market(args.market)
    if report["raw_files"] == 0:
        print(f"  {report['note']} — canonical {args.market}.parquet not written.")
    else:
        print(f"  raw files:          {report['raw_files']}")
        print(f"  records in:         {report['input']}")
        print(f"  duplicates removed: {report['duplicates_removed']}")
        print(f"  kept (canonical):   {report['kept']}")
        print(f"  per-source kept:    {report.get('per_source_kept')}")
        print(f"  output:             {report['output']}")

    _print_scale_banner(args.market)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
