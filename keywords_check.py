"""
scripts/keywords_check.py
----------------------------
Deterministic (non-LLM) keyword-stuffing detector, used in core/manager.py
Phase 7 alongside the LLM-based audit/auto-fix. Kept as pure code per the
"Shift Intelligence Left" principle — stuffing detection doesn't need an LLM.
"""

import re
from collections import Counter
from typing import List

MAX_REPEAT_PER_100_WORDS = 4  # a single keyword repeated more than this = stuffing
MIN_WORD_LEN = 4


def check_keyword_stuffing(listing_text: str) -> List[str]:
    violations: List[str] = []
    words = re.findall(r"[A-Za-z]{%d,}" % MIN_WORD_LEN, listing_text.lower())
    if not words:
        return violations

    counts = Counter(words)
    total = len(words)
    for word, count in counts.items():
        rate_per_100 = (count / total) * 100
        if rate_per_100 > MAX_REPEAT_PER_100_WORDS and count >= 5:
            violations.append(
                f"Possible keyword stuffing: '{word}' appears {count} times "
                f"({rate_per_100:.1f} per 100 words)"
            )
    return violations