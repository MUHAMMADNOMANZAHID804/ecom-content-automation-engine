"""
tools/scrapper_mcp_gateway.py
--------------------------------
Gateway layer in front of tools/scrapper_mcp.py: adds an in-memory TTL cache
(avoid re-scraping the same ASIN within a run) and wraps every call through
scripts/circuit_breaker.py so a scraping outage doesn't take down the whole
pipeline — Phase 1 will fail cleanly instead of hanging.

Swap the underlying ScraperMCP for a licensed data provider client here
without touching core/manager.py or core/subagents.py.
"""

import time
import logging
from typing import Dict, List, Optional

from tools.scrapper_mcp import ScraperMCP
from scripts.circuit_breaker import CircuitBreaker

logger = logging.getLogger("scrapper_mcp_gateway")

CACHE_TTL_S = 600  # 10 minutes


class ScraperGateway:
    def __init__(self, scraper: Optional[ScraperMCP] = None):
        self.scraper = scraper or ScraperMCP()
        self.breaker = CircuitBreaker(failure_threshold=3, reset_timeout_s=60)
        self._cache: Dict[str, Dict] = {}

    def _cache_get(self, key: str):
        entry = self._cache.get(key)
        if not entry:
            return None
        value, ts = entry
        if time.time() - ts > CACHE_TTL_S:
            self._cache.pop(key, None)
            return None
        return value

    def fetch_product(self, asin: str) -> Dict:
        cache_key = f"product::{asin}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info("Cache hit for product %s", asin)
            return cached
        result = self.breaker.call(lambda: self.scraper.fetch_product(asin))
        self._cache[cache_key] = (result, time.time())
        return result

    def fetch_reviews(self, asin: str, max_pages: int = 3) -> List[str]:
        cache_key = f"reviews::{asin}::{max_pages}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info("Cache hit for reviews %s", asin)
            return cached
        result = self.breaker.call(lambda: self.scraper.fetch_reviews(asin, max_pages))
        self._cache[cache_key] = (result, time.time())
        return result