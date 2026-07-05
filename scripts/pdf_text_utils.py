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


# ---------------------------------------------------------------------------
# SafeFPDF — hardens every cell()/multi_cell() call against
# "Not enough horizontal space to render a single character", a real fpdf2
# issue (see https://github.com/py-pdf/fpdf2/issues/1250) triggered by the
# library's internal line-break algorithm under certain text/width/font
# conditions that vary by fpdf2 version. This showed up identically in BOTH
# modules/competitor_analyzer.py (a bordered table) AND
# modules/review_analyzer.py (plain multi_cell() calls, no table at all) —
# meaning it isn't specific to one layout, it's a version-dependent fpdf2
# behavior that can fire on any cell/multi_cell call given the wrong
# conditions (empty/whitespace-only text is the most common trigger).
#
# Rather than chase the exact trigger further, this makes EVERY cell and
# multi_cell call self-healing:
#   1. Guarantees the text is never empty/whitespace-only (the single most
#      common trigger) — falls back to a single space.
#   2. If it STILL raises, retries once with a smaller font size.
#   3. If that still fails, writes a plain "-" placeholder instead.
#   4. If even that fails, skips the cell silently rather than crashing the
#      entire PDF (and therefore the entire competitor/review/listing tool).
# ---------------------------------------------------------------------------
try:
    from fpdf import FPDF as _BaseFPDF
except ImportError:  # pragma: no cover — fpdf2 is a hard requirement anyway
    _BaseFPDF = object

import logging
_logger = logging.getLogger("pdf_text_utils")


class SafeFPDF(_BaseFPDF):
    """Drop-in replacement for fpdf.FPDF — use exactly like FPDF(), just
    import it from here instead: `from scripts.pdf_text_utils import
    sanitize_pdf_text, SafeFPDF`."""

    def cell(self, *args, **kwargs):
        args = self._ensure_nonempty_text(args)
        try:
            return super().cell(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return self._safe_fallback("cell", e, args, kwargs)

    def multi_cell(self, *args, **kwargs):
        args = self._ensure_nonempty_text(args)
        try:
            return super().multi_cell(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return self._safe_fallback("multi_cell", e, args, kwargs)

    @staticmethod
    def _ensure_nonempty_text(args):
        """Every call site in this project passes (w, h, text, ...)
        positionally — text is always the 3rd positional argument."""
        args = list(args)
        if len(args) >= 3:
            val = args[2]
            if val is None or not str(val).strip():
                args[2] = " "
        return tuple(args)

    def _safe_fallback(self, method_name: str, error: Exception, args, kwargs):
        _logger.warning("SafeFPDF.%s failed (%s) — retrying with a smaller font.",
                         method_name, error)
        fn = getattr(_BaseFPDF, method_name)
        try:
            original_size = self.font_size_pt
            self.set_font_size(max(6, original_size - 2))
            result = fn(self, *args, **kwargs)
            self.set_font_size(original_size)
            return result
        except Exception as e2:  # noqa: BLE001
            _logger.warning("SafeFPDF.%s retry also failed (%s) — writing a placeholder.",
                             method_name, e2)
            try:
                safe_args = list(args)
                if len(safe_args) >= 3:
                    safe_args[2] = "-"
                return fn(self, *safe_args, **kwargs)
            except Exception as e3:  # noqa: BLE001
                _logger.error("SafeFPDF.%s placeholder also failed (%s) — skipping this cell.",
                              method_name, e3)
                try:
                    self.ln(6)
                except Exception:  # noqa: BLE001
                    pass
                return None
