"""
scraper.sites
=============

Registry of concrete :class:`~scraper.base.SiteAdapter` implementations.

Add a new portal by dropping a ``scraper/sites/<name>.py`` module that defines a
``SiteAdapter`` subclass, then register it in :data:`ADAPTERS` below. The
orchestrator (:mod:`scraper.run`) and the de-dup tie-breaker
(:mod:`scraper.dedup`) discover sources through this dict only.
"""
from __future__ import annotations

from scraper.sites.immoweb import ImmowebAdapter
from scraper.sites.realo import RealoAdapter

#: source id -> adapter class. The single source of truth for "which sites".
ADAPTERS = {
    "immoweb": ImmowebAdapter,
    "realo": RealoAdapter,
}

__all__ = ["ADAPTERS", "ImmowebAdapter", "RealoAdapter"]
