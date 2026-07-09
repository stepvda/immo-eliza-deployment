# `scraper/` — polite, resumable, multi-site Belgian real-estate scraper

A production-grade **framework** ("machine") for collecting Belgian property
listings from multiple portals, normalising every source into one canonical
schema, de-duplicating across sites, and storing the result in a resumable,
partitioned Parquet store. It is engineered to run in **small polite passes over
days via cron** toward a target of **100k+ for-sale and 30k+ to-rent
deduplicated listings** with dense geographic coverage.

> **It is intentionally not run at full scale here.** The major sites sit behind
> Cloudflare / Datadome anti-bot protection and their Terms of Service forbid
> scraping, and in-session networking is typically blocked. Every network call
> therefore **degrades gracefully** (returns nothing, logs the condition) while
> all parsing, normalising, de-dup, storage and orchestration logic remains
> fully working and offline-testable.

---

## Read the legal note first ⚠️

Scraping most Belgian real-estate portals (Immoweb, Realo, Zimmo, …) **violates
their Terms of Service**, and automated access can breach the anti-bot measures
they deploy. Nothing in this repository is permission to scrape any site.

Before running any live crawl you **must**:

- **Get authorization.** Prefer an **official API / data partnership / licensed
  feed** (many portals and Statbel / the notaries' federation offer bulk or API
  access). This is the correct path to 100k+ listings for production.
- **Respect `robots.txt`.** Enforced by default (`respect_robots=True`); the
  `--no-robots` flag exists only for sites *you own or are authorised to crawl*.
- **Rate-limit aggressively.** Default is ~20 requests/minute **per domain**;
  go lower for production. Identify yourself honestly in the User-Agent with a
  contact address (see `scraper/base.py :: CONTACT`).
- **Respect personal data law (GDPR).** Listing data can contain personal
  information; store and process it lawfully, and honour takedown requests.

The framework is built so the *engineering* is production-ready while the
*operation* stays lawful and authorised.

---

## Architecture

```
scraper/
  schema.py         canonical listing schema (pre-existing) — the one true shape
  base.py           SiteAdapter ABC + async Fetcher (rate-limit, cache, robots, retry)
  normalize.py      pure site-vocab -> canonical-vocab coercers + finalize()
  dedup.py          cross-site blocking-key de-duplication -> (kept, report)
  store.py          seen_urls SQLite state + raw partitioned Parquet + merge_market()
  run.py            orchestrator CLI (argparse)
  sites/
    __init__.py     ADAPTERS registry  {"immoweb": ..., "realo": ...}
    immoweb.py      Immoweb adapter (+ offline fixture + self-test)
    realo.py        Realo adapter   (+ offline fixture + self-test)
```

Data flows **site payload → adapter.parse() → canonical record →
normalize.finalize() → raw parquet → merge_market() (dedupe) → canonical
parquet**:

```
data/listings/
  cache/                         on-disk HTTP cache, key = sha256(url)
  state.sqlite                   seen_urls frontier (resumability)
  raw/<source>/<market>/*.parquet   append-only per-batch captures
  <market>.parquet               canonical, de-duplicated output (sale / rent)
```

Everything except live fetching is **importable and self-testable offline**.
Each module has a `__main__` self-test:

```bash
.venv/bin/python -m scraper.base
.venv/bin/python -m scraper.normalize
.venv/bin/python -m scraper.dedup
.venv/bin/python -m scraper.store
.venv/bin/python -m scraper.sites.immoweb
.venv/bin/python -m scraper.sites.realo
.venv/bin/python -m scraper.run --merge-only --market sale   # no-ops cleanly with no raw data
```

---

## The canonical schema

Every scraped listing is normalised into a dict with the exact keys from
`scraper.schema.CANONICAL_COLUMNS` (built via `empty_record()`), grouped as:

- **Provenance** — `listing_id, source, url, market, scraped_at`
- **Geography** — `street, house_number, postal_code, locality, municipality,
  refnis, province, region, latitude, longitude, nearest_city,
  nearest_city_distance_km`
- **Numeric** — `livable_surface, land_surface, bedrooms, bathrooms, toilets,
  build_year, facades, number_of_floors, primary_energy_consumption`
- **Categorical** — `property_type, category, epc, building_state,
  kitchen_equipment, heating_type`
- **Binary flags (0/1)** — `new_construction, furnished, terrace, garden,
  swimming_pool, elevator, cellar, solar_panels, air_conditioning, has_parking`
- **Price** — `price, price_per_sqm`

Canonical value vocabularies (mirrored from
`ml/data/in/cleaned_sale_properties.csv`) are enforced by `scraper/normalize.py`:

| Field             | Canonical values |
|-------------------|------------------|
| `property_type`   | flat, house, villa, penthouse, flatStudio, groundFloor, duplex, masterHouse, bungalow, chalet, loft, cottage, triplex, mansion, studentFlat |
| `category`        | house, apartment (derived from `property_type`) |
| `epc`             | A++, A+, A, B, C, D, E, F, G |
| `building_state`  | A, B, C, D (ordinal condition code, A = best) |
| `kitchen_equipment` | Super equipped, Fully equipped, Partially equipped, Not equipped |
| `heating_type`    | Gas, Fuel oil, Hot air, Electricity, Wood, Solar energy, Coal |

`normalize.finalize(record)` then fills `category` from `property_type`, computes
`price_per_sqm`, and enriches `refnis / province / region / municipality` from
`data/invest/nis_postal_crosswalk.csv`. Address → lat/lon geocoding is delegated
to `geo.geocode.resolve` **if that module exists** (it is being built in
parallel), and skipped gracefully otherwise.

---

## Adding a new site adapter

1. Create `scraper/sites/<name>.py` with a `SiteAdapter` subclass:

   ```python
   from scraper.base import SiteAdapter
   from scraper.schema import empty_record
   from scraper.normalize import (normalize_property_type, parse_price, ...)

   class MySiteAdapter(SiteAdapter):
       name = "mysite"
       base_url = "https://www.mysite.be"
       COMPLETENESS = {"price": 1.0, "livable_surface": 0.9, ...}  # 0..1 per field

       def iter_search_urls(self, market):        # yield paginated search URLs
           ...

       def extract_listing_urls(self, payload, base_url):  # detail URLs from a search page
           ...

       def parse(self, payload, url, market):     # payload (str HTML/JSON or dict) -> canonical
           rec = empty_record()
           rec["source"] = self.name
           rec["url"] = url
           rec["market"] = market
           rec["price"] = parse_price(...)
           ...
           return rec
   ```

2. **Register it** in `scraper/sites/__init__.py`:

   ```python
   from scraper.sites.mysite import MySiteAdapter
   ADAPTERS = {"immoweb": ImmowebAdapter, "realo": RealoAdapter, "mysite": MySiteAdapter}
   ```

3. Add an **offline fixture** (a saved payload) plus a `__main__` self-test that
   parses it and prints the canonical record — so `parse()` is testable without
   the network, exactly like `immoweb.py` / `realo.py`.

### The `COMPLETENESS` profile

Each adapter declares `COMPLETENESS: dict[str, float]` — which canonical fields
the source reliably provides, 0..1. It is used to (a) prioritise richer sources
and (b) **break de-dup ties**: when the same physical property appears on two
portals with equal field counts, the one whose adapter has the higher summed
`COMPLETENESS` wins.

---

## De-duplication

`scraper.dedup.dedupe(records) -> (kept, report)`:

- **Blocking key**: `(postal_code, normalized street + house_number,
  round(livable_surface/5), bedrooms)`.
- **Within a block**, records collapse only when prices agree within ~5% **and**
  (when both have coordinates) they sit within ~40 m — guarding against distinct
  properties sharing a block.
- **Survivor**: most non-null canonical fields, ties broken by the adapter's
  `COMPLETENESS` weight.
- **Report**: per-source input/kept counts and duplicates removed.

`merge_market(market)` applies this across *all* raw parquet for a market and is
**idempotent**.

---

## Running

```bash
# One validation pass (small!). Prints a clear "VALIDATION SAMPLE" banner.
.venv/bin/python -m scraper.run --sites immoweb,realo --market sale --max 200

# Re-merge existing raw captures only (no network):
.venv/bin/python -m scraper.run --market sale --merge-only

# Rent market, one site, custom politeness:
.venv/bin/python -m scraper.run --sites realo --market rent --max 100 --rpm 12
```

Flags: `--sites`, `--market {sale,rent}`, `--max` (new listings/site this pass),
`--rpm` (per-domain requests/minute), `--no-robots` (only for sites you own),
`--merge-only`.

---

## Scaling to 100k / 30k over days (cron)

You cannot — and must not — grab 100k listings in one blast. Instead let the
**`seen_urls` frontier advance a little on every polite pass**:

1. **Breadth** — register more adapters (immoweb, realo, zimmo, immovlan,
   logic-immo, …) so coverage is national and cross-checked.
2. **Depth** — raise `MAX_PAGES_PER_PROVINCE` / `MAX_PAGES_PER_REGION` in each
   adapter so more of each province/region is enumerated.
3. **Cadence** — run frequent small passes; each skips already-seen URLs, so the
   union grows monotonically toward the target while staying polite.

Rough budget: at ~15 req/min a single pass of `--max 500` costs well under an
hour; a handful of adapters running hourly reaches 100k sale + 30k rent in a few
days without hammering any one domain.

### Example crontab

```cron
# m h  dom mon dow   command   (cd into the repo; use the venv python)
# Sale — hourly, staggered per site, low rate, robots enforced.
0  *  * * *  cd /path/to/immo-eliza-deployment && .venv/bin/python -m scraper.run --sites immoweb --market sale --max 500 --rpm 15 >> data/listings/cron.log 2>&1
20 *  * * *  cd /path/to/immo-eliza-deployment && .venv/bin/python -m scraper.run --sites realo   --market sale --max 500 --rpm 15 >> data/listings/cron.log 2>&1
# Rent — every 2 hours.
40 */2 * * * cd /path/to/immo-eliza-deployment && .venv/bin/python -m scraper.run --sites immoweb,realo --market rent --max 300 --rpm 12 >> data/listings/cron.log 2>&1
# Nightly consolidation (merge only; cheap, idempotent).
30 3  * * *  cd /path/to/immo-eliza-deployment && .venv/bin/python -m scraper.run --market sale --merge-only >> data/listings/cron.log 2>&1
35 3  * * *  cd /path/to/immo-eliza-deployment && .venv/bin/python -m scraper.run --market rent --merge-only >> data/listings/cron.log 2>&1
```

Because `seen_urls` and the raw parquet partitions persist, cron passes are
**resumable**: a crashed or throttled run loses nothing and the next pass
continues the frontier.

---

## Proxy guidance

For an **authorised** large crawl, a single IP will be rate-limited or blocked.
Options, in order of preference:

1. **Use the official API / licensed feed** — no proxies needed, no ToS risk.
2. If authorised to crawl the HTML: route through a **reputable rotating
   residential/ISP proxy pool** with **per-domain session stickiness**, and keep
   the *effective* per-domain rate low (politeness beats proxy count). Wire the
   pool into `Fetcher._get_client()` via `httpx`'s `proxies=`/`transport=`.
3. Add jittered scheduling (already deterministic here via
   `Fetcher.backoff_delay`), realistic headers, and a real contact UA.
4. **Never** use proxies to defeat anti-bot systems or ToS restrictions — that
   is exactly the line this framework is designed *not* to cross without
   authorisation.

Datadome / Cloudflare Turnstile challenges are a deliberate "do not scrape"
signal. The correct response is to seek a data partnership, not to escalate.
