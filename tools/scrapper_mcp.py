"""
tools/scrapper_mcp.py
------------------------
MCP-style tool that fetches raw Amazon product + review data for a given
ASIN. Consumed by core/subagents.ASINResolverAgent in Phase 1.

Kept deliberately dumb (no LLM calls here) — this tool's job is ONLY to fetch
and lightly pre-clean HTML. Interpretation/structuring is the sub-agent's job.
Respect Amazon's robots.txt / ToS in production; consider an official API or
licensed data provider (e.g. Jungle Scout, Keepa) instead of raw scraping.
"""

import time
import logging
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("scrapper_mcp")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AntiGravityBot/2.0; +internal-use-only)"
}
REQUEST_TIMEOUT_S = 15
RATE_LIMIT_DELAY_S = 1.5


class ScraperMCP:
    def __init__(self, session: requests.Session = None):
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def fetch_product(self, asin: str) -> Dict:
        url = f"https://www.amazon.com/dp/{asin}"
        html = self._get(url)
        soup = BeautifulSoup(html, "html.parser") if html else None

        title_el = soup.select_one("#productTitle") if soup else None
        bullets_el = soup.select("#feature-bullets li span") if soup else []
        price_el = soup.select_one(".a-price .a-offscreen") if soup else None
        rating_el = soup.select_one("span[data-hook='rating-out-of-text']") if soup else None

        return {
            "asin": asin,
            "title": title_el.get_text(strip=True) if title_el else None,
            "bullet_points": [b.get_text(strip=True) for b in bullets_el],
            "price": price_el.get_text(strip=True) if price_el else None,
            "rating": rating_el.get_text(strip=True) if rating_el else None,
            "raw_html_len": len(html) if html else 0,
        }

    def fetch_reviews(self, asin: str, max_pages: int = 3) -> List[str]:
        reviews: List[str] = []
        for page in range(1, max_pages + 1):
            url = (
                f"https://www.amazon.com/product-reviews/{asin}"
                f"?pageNumber={page}&sortBy=recent"
            )
            html = self._get(url)
            if not html:
                break
            soup = BeautifulSoup(html, "html.parser")
            review_nodes = soup.select("span[data-hook='review-body']")
            page_reviews = [n.get_text(strip=True) for n in review_nodes]
            if not page_reviews:
                break
            reviews.extend(page_reviews)
            time.sleep(RATE_LIMIT_DELAY_S)
        return reviews

    def _get(self, url: str) -> str:
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning("Scrape request failed for %s: %s", url, e)
            return ""