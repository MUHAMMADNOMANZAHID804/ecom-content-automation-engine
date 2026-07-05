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

from fpdf import FPDF
from fpdf.errors import FPDFException

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


# FIX: "Not enough horizontal space to render a single character" is a known
# fpdf2 line-break bug (fpdf2 GitHub issue #1250): its word-wrap engine treats
# any run of non-whitespace characters as a single unbreakable "word". If that
# run is wider than the available cell/page width — a raw URL, a run-on
# ASIN/SKU, a mangled title with no spaces, a long hyphen-less compound word —
# fpdf2 crashes instead of just overflowing. Truncating text length doesn't
# reliably prevent this (a 45-char unbroken token can already be wide enough
# at 8-10pt fonts), so instead we guarantee a break point exists in any long
# unbroken run by inserting a plain space every _MAX_UNBROKEN_RUN characters.
# This is applied to ALL PDF text (every method in both analyzer modules
# already routes through sanitize_pdf_text()), so it fixes the crash at its
# single common choke point without touching any PDF layout/table code.
_MAX_UNBROKEN_RUN = 25
_LONG_RUN_RE = re.compile(r"\S{%d,}" % (_MAX_UNBROKEN_RUN + 1))


def _break_long_runs(text: str, chunk: int = _MAX_UNBROKEN_RUN) -> str:
    def _splitter(match: "re.Match") -> str:
        run = match.group(0)
        return " ".join(run[i:i + chunk] for i in range(0, len(run), chunk))

    return _LONG_RUN_RE.sub(_splitter, text)


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
        return _break_long_runs(text)
    except UnicodeEncodeError:
        pass

    # Decompose accented characters (é -> e + combining accent), then drop
    # the combining marks and any other remaining non-Latin-1 codepoints.
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))

    try:
        stripped.encode("latin-1")
        return _break_long_runs(stripped)
    except UnicodeEncodeError:
        # Final safety net: replace anything still unencodable with '?'
        # rather than crash. This should be rare after the steps above.
        stripped = stripped.encode("latin-1", errors="replace").decode("latin-1")
        return _break_long_runs(stripped)


def sanitize_pdf_lines(lines) -> list:
    """Convenience helper for sanitizing a list of strings at once."""
    return [sanitize_pdf_text(line) for line in lines]


class SafeFPDF(FPDF):
    """
    Drop-in replacement for fpdf.FPDF — use `SafeFPDF()` instead of `FPDF()`
    and nothing else needs to change. Structurally cannot raise fpdf2's
    "Not enough horizontal space to render a single character" (a known
    line-break engine bug — fpdf2 GitHub issue #1250 — triggered by any
    unbroken run of characters that's wider than the available cell width,
    e.g. a URL, a long SKU, a mangled title). sanitize_pdf_text() already
    prevents most cases by pre-breaking long runs, but font size and column
    width vary across this report (16pt down to 8pt, 14mm up to full page),
    so this is a backstop that guarantees multi_cell() can NEVER crash the
    PDF, no matter what text or width it receives.

    On failure, it retries at smaller font sizes, and if that's still not
    enough, falls back to manually laying the text out one character at a
    time using plain cell() calls — which has no internal word-wrap step
    and therefore cannot trigger this bug. Original font is always restored.
    """

    def multi_cell(self, w=0, h=None, txt="", *args, **kwargs):
        try:
            return super().multi_cell(w, h, txt, *args, **kwargs)
        except FPDFException:
            pass

        family, style, size = self.font_family, self.font_style, self.font_size_pt

        # Attempt 1: shrink the font a few points and retry the normal path —
        # cheapest fix, and keeps the real multi_cell layout/wrap behavior.
        for smaller in range(int(size) - 1, 5, -1):
            try:
                self.set_font(family, style, smaller)
                result = super().multi_cell(w, h, txt, *args, **kwargs)
                self.set_font(family, style, size)
                return result
            except FPDFException:
                continue

        # Attempt 2: guaranteed-safe manual character-by-character layout.
        # No word-wrap algorithm involved at all, so it cannot hit this bug.
        self.set_font(family, style, size)
        line_h = h or (size / 2.0)
        usable_width = w if w and w > 0 else (self.w - self.r_margin - self.l_margin)
        start_x = self.x
        cur_width = 0.0
        for ch in str(txt):
            if ch == "\n":
                self.ln(line_h)
                self.set_x(start_x)
                cur_width = 0.0
                continue
            ch_w = self.get_string_width(ch) or 0.1
            if cur_width > 0 and cur_width + ch_w > usable_width:
                self.ln(line_h)
                self.set_x(start_x)
                cur_width = 0.0
            self.cell(ch_w, line_h, ch)
            cur_width += ch_w
        self.ln(line_h)
        self.set_font(family, style, size)
