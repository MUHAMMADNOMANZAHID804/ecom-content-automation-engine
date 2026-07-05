"""
scripts/pdf_text_utils.py
----------------------------
FPDF's core (built-in) fonts only support the Latin-1 character set. Real
data WILL contain characters outside it — star symbols, curly quotes from
LLM output, em-dashes, emoji in product titles, Euro signs, accented brand
names, etc. Rather than fixing one crash at a time, every PDF-writing method
in modules/competitor_analyzer.py and modules/review_analyzer.py should route
ALL text through sanitize_pdf_text() before handing it to FPDF.
"""

import re
import unicodedata

# Explicit, human-readable replacements for characters that are common in
# this project's data (ratings, LLM-generated copy, EU currency) but fall
# outside Latin-1.
_CHAR_REPLACEMENTS = {
    "\u2605": "*",      # ★ filled star (star ratings)
    "\u2606": "*",      # ☆ outline star
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\u201c": '"', "\u201d": '"',   # curly double quotes
    # FIX: every dash/hyphen variant, not just en/em dash. The '?' corruption
    # bug ("Ultra?Durable", "tear?resistant") was caused by models emitting
    # U+2011 NON-BREAKING HYPHEN (common in generated copy to keep compound
    # words from wrapping) — it wasn't in this map, so it fell through to the
    # '?' fallback at the bottom of sanitize_pdf_text().
    "\u2010": "-",   # hyphen
    "\u2011": "-",   # non-breaking hyphen
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2015": "-",   # horizontal bar
    "\u2026": "...",    # ellipsis
    "\u2022": "-",      # bullet point
    "\u00ae": "(R)", "\u2122": "(TM)", "\u00a9": "(C)",
    "\u20ac": "EUR",    # € (not in Latin-1, unlike £ and $)
    "\u2192": "->", "\u2190": "<-",  # arrows sometimes used by LLM output
    "\u2705": "[OK]", "\u274c": "[X]",  # emoji checkmarks occasionally emitted
}


def sanitize_pdf_text(text) -> str:
    """
    Makes any string safe to pass into FPDF's multi_cell/cell.
    1. Applies explicit replacements for common characters (keeps meaning).
    2. Decomposes remaining accented Latin characters (e.g. e -> e) so
       international brand names degrade gracefully instead of vanishing.
    3. Drops/replaces anything still outside Latin-1 as a final safety net
       (e.g. stray emoji) so this can NEVER raise UnicodeEncodeError again.
    """
    if text is None:
        return ""
    text = str(text)

    for bad, good in _CHAR_REPLACEMENTS.items():
        text = text.replace(bad, good)

    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        pass

    # Decompose accented characters (é -> e + combining accent), then drop
    # the combining marks and any other remaining non-Latin-1 codepoints.
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))

    try:
        stripped.encode("latin-1")
        return stripped
    except UnicodeEncodeError:
        # Final safety net: replace anything still unencodable with '?'
        # rather than crash. This should be rare after the steps above.
        return stripped.encode("latin-1", errors="replace").decode("latin-1")


def sanitize_pdf_lines(lines) -> list:
    """Convenience helper for sanitizing a list of strings at once."""
    return [sanitize_pdf_text(line) for line in lines]