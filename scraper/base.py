"""
scraper/base.py
===============

Foundations for the polite, resumable, multi-site scraping framework:

* :class:`SiteAdapter` — the abstract contract every site adapter implements.
  An adapter knows how to enumerate *search* URLs for a market, how to pull the
  individual *listing* URLs out of a search page, and how to :meth:`parse` a
  single listing payload into the **canonical schema** (:mod:`scraper.schema`).
* :class:`Fetcher` — a shared **async HTTP engine** that is deliberately
  *polite* and *robust*: per-domain rate limiting, exponential backoff with
  deterministic (non-random) jitter, a realistic User-Agent, an on-disk HTTP
  cache keyed by ``sha256(url)``, ``robots.txt`` fetching + enforcement, and a
  hard per-request timeout so it can *never hang*.

Design rules that make the framework testable **without any network**:

* Everything that touches the network lives inside :meth:`Fetcher.get`.
* Adapters' :meth:`parse` / URL builders are *pure functions* — give them a
  saved fixture and they produce a canonical record deterministically.
* If ``httpx`` is missing or a host is unreachable / blocked (Cloudflare,
  Datadome, robots ``Disallow``), :meth:`Fetcher.get` returns ``None`` and the
  caller degrades gracefully rather than crashing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
from urllib import robotparser
from urllib.parse import urljoin, urlparse

try:  # httpx is the only network dependency; keep import optional for offline tests
    import httpx
except Exception:  # pragma: no cover - exercised only when httpx is absent
    httpx = None  # type: ignore

from scraper.schema import CANONICAL_COLUMNS, LISTINGS_DIR

# --------------------------------------------------------------------------- #
# Configuration constants (polite defaults)                                   #
# --------------------------------------------------------------------------- #
CACHE_DIR = LISTINGS_DIR / "cache"

#: Polite default. Real sites want <= ~1 request every few seconds per domain.
#: 20/min == one request every 3s. Lower it further for production crawls.
REQUESTS_PER_MINUTE = 20

#: Hard per-request timeout. The engine must never hang, so this is short.
REQUEST_TIMEOUT_S = 15.0

#: Retries are capped; total attempts == MAX_RETRIES + 1.
MAX_RETRIES = 4

#: Base of the exponential backoff (seconds): base * 2**attempt + jitter.
BACKOFF_BASE_S = 1.0

#: A realistic desktop User-Agent plus a research contact string so site
#: operators can reach out. Override per deployment.
CONTACT = "immo-eliza-research (contact: research@example.org)"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 "
    f"{CONTACT}"
)

# HTTP status codes that are pointless to retry (hard client errors / blocks).
_NON_RETRYABLE = frozenset({400, 401, 403, 404, 405, 410, 451})


@dataclass
class Response:
    """A minimal, cache-serialisable HTTP response."""

    url: str
    status_code: int
    text: str
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return json.loads(self.text)


# --------------------------------------------------------------------------- #
# The async fetch engine                                                      #
# --------------------------------------------------------------------------- #
class Fetcher:
    """Polite async HTTP client shared by every adapter.

    Parameters
    ----------
    requests_per_minute:
        Per-domain rate cap. The engine serialises requests and sleeps to keep
        at least ``60 / rpm`` seconds between hits *to the same domain*.
    respect_robots:
        When ``True`` (default) every URL is checked against the host's
        ``robots.txt`` and disallowed URLs yield ``None``.
    cache_dir:
        On-disk HTTP cache directory (default :data:`CACHE_DIR`). Cache key is
        ``sha256(url)`` so repeated runs are cheap and resumable.
    timeout / max_retries / backoff_base:
        Robustness knobs — short timeout, capped retries, exponential backoff.
    user_agent:
        Sent on every request (including the ``robots.txt`` fetch).
    """

    def __init__(
        self,
        requests_per_minute: int = REQUESTS_PER_MINUTE,
        *,
        respect_robots: bool = True,
        cache_dir: Optional[Path] = None,
        timeout: float = REQUEST_TIMEOUT_S,
        max_retries: int = MAX_RETRIES,
        backoff_base: float = BACKOFF_BASE_S,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.respect_robots = respect_robots
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.user_agent = user_agent

        self._client = None  # lazily created httpx.AsyncClient
        self._lock = asyncio.Lock()  # serialises throttle bookkeeping
        self._last_hit: dict[str, float] = {}  # domain -> monotonic ts
        self._robots: dict[str, robotparser.RobotFileParser] = {}

    # -- cache ------------------------------------------------------------- #
    def cache_path(self, url: str) -> Path:
        """Deterministic on-disk location for a URL's cached response."""
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, url: str) -> Optional[Response]:
        path = self.cache_path(url)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Response(
                url=data["url"],
                status_code=int(data["status_code"]),
                text=data["text"],
                from_cache=True,
            )
        except Exception:
            return None  # corrupt cache entry -> treat as miss

    def _write_cache(self, resp: Response) -> None:
        path = self.cache_path(resp.url)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"url": resp.url, "status_code": resp.status_code, "text": resp.text}
        path.write_text(json.dumps(payload), encoding="utf-8")

    # -- backoff / throttle ------------------------------------------------ #
    def backoff_delay(self, attempt: int, url: str) -> float:
        """Exponential backoff with *deterministic* jitter.

        The runtime forbids ``random``-style calls, so the jitter is derived
        from ``sha256(url:attempt)`` — still spreads retries across time but is
        fully reproducible in tests.
        """
        base = self.backoff_base * (2 ** attempt)
        digest = hashlib.sha256(f"{url}:{attempt}".encode("utf-8")).hexdigest()
        jitter = (int(digest[:8], 16) % 1000) / 1000.0  # 0.0 .. 1.0 s
        return base + jitter

    async def _throttle(self, domain: str) -> None:
        async with self._lock:
            min_interval = 60.0 / self.requests_per_minute
            now = time.monotonic()
            wait = min_interval - (now - self._last_hit.get(domain, 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_hit[domain] = time.monotonic()

    # -- low-level single request ----------------------------------------- #
    def _get_client(self):
        if self._client is None:
            if httpx is None:
                raise RuntimeError("httpx is not installed; live fetching disabled")
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
        return self._client

    async def _raw_get(self, url: str) -> Optional[Response]:
        """One network attempt. Returns ``None`` on any transport error."""
        if httpx is None:
            return None
        try:
            client = self._get_client()
            r = await client.get(url)
            return Response(url=str(r.url), status_code=r.status_code, text=r.text)
        except Exception:
            return None

    # -- robots.txt -------------------------------------------------------- #
    async def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._robots.get(root)
        if parser is None:
            parser = robotparser.RobotFileParser()
            robots_url = urljoin(root, "/robots.txt")
            robots = await self._raw_get(robots_url)
            if robots is not None and robots.ok and robots.text.strip():
                try:
                    parser.parse(robots.text.splitlines())
                except Exception:
                    parser.parse([])  # unparseable -> permissive
            else:
                # No robots.txt (or unreachable) -> the RFC default is "allow".
                parser.parse([])
            self._robots[root] = parser
        try:
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return True

    # -- public API -------------------------------------------------------- #
    async def get(self, url: str, *, use_cache: bool = True) -> Optional[Response]:
        """Fetch ``url`` politely.

        Returns a :class:`Response` on success, or ``None`` when the URL is
        cached-miss + unreachable, blocked by ``robots.txt``, or exhausted its
        retry budget. Never raises for network conditions and never hangs.
        """
        if use_cache:
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        if not await self._allowed(url):
            return None

        domain = urlparse(url).netloc
        for attempt in range(self.max_retries + 1):
            await self._throttle(domain)
            resp = await self._raw_get(url)
            if resp is not None and resp.ok:
                if use_cache:
                    self._write_cache(resp)
                return resp
            if resp is not None and resp.status_code in _NON_RETRYABLE:
                return None  # blocked / gone — retrying will not help
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff_delay(attempt, url))
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None


# --------------------------------------------------------------------------- #
# The adapter contract                                                        #
# --------------------------------------------------------------------------- #
class SiteAdapter(ABC):
    """Abstract base every concrete site adapter subclasses.

    Subclasses set three class attributes and implement two methods:

    * ``name`` — short source id (also the raw-parquet partition name).
    * ``base_url`` — site root, used to absolutise relative listing URLs.
    * ``COMPLETENESS`` — a ``{canonical_field: 0..1}`` profile describing which
      fields this source *reliably* provides. It is used to (a) prioritise
      sources and (b) break de-duplication ties when two sources describe the
      same property. Sum it via :meth:`completeness_score`.

    Two abstract methods, plus one overridable helper:

    * :meth:`iter_search_urls` — yields paginated search/listing-index URLs.
    * :meth:`extract_listing_urls` — pulls detail URLs out of a search payload.
    * :meth:`parse` — maps one listing payload to a canonical record.
    """

    name: str = "base"
    base_url: str = ""
    COMPLETENESS: dict[str, float] = {}

    @abstractmethod
    def iter_search_urls(self, market: str) -> Iterator[str]:
        """Yield search-result page URLs for ``market`` ('sale' or 'rent')."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, payload, url: str, market: str) -> Optional[dict]:
        """Map one listing ``payload`` (str HTML/JSON or dict) to a canonical
        record built from :func:`scraper.schema.empty_record`. Returns ``None``
        if the payload is not a parseable listing."""
        raise NotImplementedError

    def extract_listing_urls(self, payload, base_url: str) -> list[str]:
        """Extract individual listing detail URLs from a search payload.

        Default is empty; adapters that crawl search pages override this.
        """
        return []

    # -- helpers ----------------------------------------------------------- #
    def completeness_score(self) -> float:
        """Sum of the COMPLETENESS profile — a scalar source-priority weight."""
        return float(sum(self.COMPLETENESS.values()))

    def absolutise(self, href: str) -> str:
        """Turn a possibly-relative href into an absolute URL under base_url."""
        return urljoin(self.base_url, href)


def count_filled(record: dict) -> int:
    """Number of non-null canonical fields in ``record`` — a completeness proxy."""
    return sum(1 for col in CANONICAL_COLUMNS if record.get(col) not in (None, ""))


# --------------------------------------------------------------------------- #
# Offline self-test                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    print("== scraper.base self-test (offline) ==")

    with tempfile.TemporaryDirectory() as tmp:
        f = Fetcher(requests_per_minute=20, cache_dir=Path(tmp))

        url = "https://example.org/en/classified/12345678"
        # cache path is deterministic sha256(url)
        cp = f.cache_path(url)
        assert cp.name.endswith(".json") and len(cp.stem) == 64
        print("cache_path ->", cp.name)

        # backoff is deterministic + monotonically growing in the base term
        d0 = f.backoff_delay(0, url)
        d1 = f.backoff_delay(1, url)
        d2 = f.backoff_delay(2, url)
        assert f.backoff_delay(0, url) == d0, "backoff must be deterministic"
        assert d0 < d1 < d2, "backoff must grow with attempt"
        print(f"backoff attempts 0..2 -> {d0:.3f}s {d1:.3f}s {d2:.3f}s (deterministic)")

        # cache round-trip
        resp = Response(url=url, status_code=200, text='{"id": 1}')
        f._write_cache(resp)
        back = f._read_cache(url)
        assert back is not None and back.from_cache and back.json()["id"] == 1
        print("cache round-trip -> ok (from_cache=%s)" % back.from_cache)

    # count_filled on an empty vs partial record
    from scraper.schema import empty_record

    rec = empty_record()
    assert count_filled(rec) == 0
    rec["price"] = 100.0
    rec["postal_code"] = "1000"
    assert count_filled(rec) == 2
    print("count_filled empty/partial -> 0 / 2")

    print("OK")
