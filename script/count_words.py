"""
scripts/count_words.py
-------------------------
Character counting used by core/manager.py Phase 7 to enforce the exact
TITLE (185-200), BULLET (300-350 x5), DESCRIPTION (1700-1800) ranges.

Deliberately simple and deterministic — character counting must never be
delegated to an LLM (models miscount reliably).
"""

import re


def count_chars(text: str) -> int:
    """Counts visible characters, collapsing internal whitespace runs to a
    single space (matches how Amazon/Etsy/Shopify render whitespace)."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return len(normalized)


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))