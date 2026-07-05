"""
modules/competitor_analyzer.py
-------------------------------
Amazon-only tool. Input: Jungle Scout CSV export.
Process: rank by sales volume, extract titles + keywords, AND surface
star-rating signals (which competitors customers actually like, and what
their keywords/features are — these often overlap with what to emulate).
Output: structured gap-analysis data + a downloadable PDF report.

FIX (this version): real Jungle Scout exports use column names like
'Product Name' and 'Monthly Units Sold', not the generic 'Title' /
'Average Monthly Sales' this module originally assumed. Column matching is
now fuzzy (case/punctuation-insensitive + partial-match fallback) so it
survives export-format drift instead of hard-failing.
"""

import os
import re
import uuid
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
from fpdf import FPDF

from scripts.pdf_text_utils import sanitize_pdf_text

logger = logging.getLogger("competitor_analyzer")

REPORTS_DIR = os.getenv("REPORTS_DIR", "generated_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

STOPWORDS = {
    "the", "and", "for", "with", "of", "a", "an", "to", "in", "on", "your",
    "our", "is", "are", "pack", "set", "new", "by", "-", "&",
}

# Minimum number of reviews a competitor needs before its star rating is
# treated as a meaningful signal (a 5.0 rating on 2 reviews is noise).
MIN_REVIEWS_FOR_RATING_SIGNAL = 15


class CompetitorAnalyzer:
    """
    Standalone-runnable module (per spec: 'Users should be able to run tools
    independently'). Also called from core/manager.py Phase 2.
    """

    # Ordered by preference. First match wins. Matched case-insensitively,
    # ignoring punctuation/whitespace differences (see _find_column).
    SALES_COLUMN_CANDIDATES = [
        "Monthly Units Sold", "Average Monthly Sales", "Avg. Monthly Sales",
        "Monthly Sales", "Units Sold", "Monthly Revenue",
    ]
    TITLE_COLUMN_CANDIDATES = [
        "Product Name", "Title", "Product Title", "Listing Title",
    ]
    RATING_COLUMN_CANDIDATES = ["Star Rating", "Rating", "Avg Rating"]
    REVIEWS_COLUMN_CANDIDATES = ["Reviews", "Review Count", "Number of Reviews"]
    BRAND_COLUMN_CANDIDATES = ["Brand"]
    PRICE_COLUMN_CANDIDATES = ["Price"]
    ASIN_COLUMN_CANDIDATES = ["ASIN"]

    # ------------------------------------------------------------------
    # Column resolution (fuzzy — survives Jungle Scout export changes)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(name).lower())

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        norm_map = {self._normalize(c): c for c in df.columns}

        # 1. Exact match (case/punctuation-insensitive)
        for cand in candidates:
            key = self._normalize(cand)
            if key in norm_map:
                return norm_map[key]

        # 2. Partial/contains match as a fallback (e.g. "Monthly Units Sold (Est.)")
        for cand in candidates:
            key = self._normalize(cand)
            for norm_col, original_col in norm_map.items():
                if key in norm_col or norm_col in key:
                    return original_col

        return None

    def _load_and_map_columns(self, csv_path: str) -> tuple:
        df = pd.read_csv(csv_path)

        sales_col = self._find_column(df, self.SALES_COLUMN_CANDIDATES)
        title_col = self._find_column(df, self.TITLE_COLUMN_CANDIDATES)
        rating_col = self._find_column(df, self.RATING_COLUMN_CANDIDATES)
        reviews_col = self._find_column(df, self.REVIEWS_COLUMN_CANDIDATES)
        brand_col = self._find_column(df, self.BRAND_COLUMN_CANDIDATES)
        price_col = self._find_column(df, self.PRICE_COLUMN_CANDIDATES)
        asin_col = self._find_column(df, self.ASIN_COLUMN_CANDIDATES)

        if not sales_col or not title_col:
            raise ValueError(
                "Could not find a sales-volume column or a title/product-name "
                f"column in this CSV. Found columns: {list(df.columns)}. "
                "If your export uses different header names, add them to "
                "SALES_COLUMN_CANDIDATES / TITLE_COLUMN_CANDIDATES in "
                "modules/competitor_analyzer.py."
            )

        return df, sales_col, title_col, rating_col, reviews_col, brand_col, price_col, asin_col

    def _safe_float(self, value: Any) -> Optional[float]:
        """Strips currency symbols, thousands separators, and whitespace
        before parsing — handles '10,779', '£1,448', '4.4', etc. safely."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        cleaned = re.sub(r"[^0-9.\-]", "", str(value))
        if cleaned in ("", "-", "."):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _row_to_dict(self, row, title_col, sales_col, rating_col, reviews_col,
                      brand_col, price_col, asin_col) -> Dict[str, Any]:
        sales_val = self._safe_float(row.get(sales_col))
        rating_val = self._safe_float(row.get(rating_col)) if rating_col else None
        reviews_val = self._safe_float(row.get(reviews_col)) if reviews_col else None
        return {
            "title": str(row[title_col]),
            "avg_monthly_sales": sales_val if sales_val is not None else 0.0,
            "star_rating": rating_val,
            "review_count": int(reviews_val) if reviews_val is not None else None,
            "brand": str(row[brand_col]) if brand_col and pd.notna(row.get(brand_col)) else "",
            "price": str(row[price_col]) if price_col and pd.notna(row.get(price_col)) else "",
            "asin": str(row[asin_col]) if asin_col and pd.notna(row.get(asin_col)) else "",
        }

    # ------------------------------------------------------------------
    # Ranking #1 — by sales volume (drives the main gap report)
    # ------------------------------------------------------------------
    def get_top5_by_avg_monthly_sales(self, csv_path: str) -> List[Dict[str, Any]]:
        (df, sales_col, title_col, rating_col, reviews_col,
         brand_col, price_col, asin_col) = self._load_and_map_columns(csv_path)

        df = df.copy()
        df["_sales_numeric"] = df[sales_col].apply(self._safe_float).fillna(0.0)
        top5 = df.sort_values(by="_sales_numeric", ascending=False).head(5)

        return [
            self._row_to_dict(row, title_col, sales_col, rating_col, reviews_col,
                               brand_col, price_col, asin_col)
            for _, row in top5.iterrows()
        ]

    # ------------------------------------------------------------------
    # Ranking #2 — by star rating (customer-satisfaction signal you asked for)
    # Filters out low-review-count outliers so a 5.0 on 3 reviews doesn't win.
    # ------------------------------------------------------------------
    def get_top5_by_star_rating(self, csv_path: str) -> List[Dict[str, Any]]:
        (df, sales_col, title_col, rating_col, reviews_col,
         brand_col, price_col, asin_col) = self._load_and_map_columns(csv_path)

        if not rating_col:
            logger.warning("No star-rating column found in this CSV; skipping rating ranking.")
            return []

        df = df.copy()
        df["_rating_numeric"] = df[rating_col].apply(self._safe_float)
        df = df[df["_rating_numeric"].notna()]

        if reviews_col:
            df["_reviews_numeric"] = df[reviews_col].apply(self._safe_float).fillna(0.0)
            df = df[df["_reviews_numeric"] >= MIN_REVIEWS_FOR_RATING_SIGNAL]

        if df.empty:
            return []

        top5 = df.sort_values(by="_rating_numeric", ascending=False).head(5)
        return [
            self._row_to_dict(row, title_col, sales_col, rating_col, reviews_col,
                               brand_col, price_col, asin_col)
            for _, row in top5.iterrows()
        ]

    # ------------------------------------------------------------------
    # Keyword extraction (deterministic, no LLM — same as before)
    # ------------------------------------------------------------------
    def extract_keywords(self, titles: List[str], top_n: int = 20) -> List[str]:
        words: List[str] = []
        for title in titles:
            tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", title.lower())
            words.extend(t for t in tokens if t not in STOPWORDS)
        counts = Counter(words)
        return [w for w, _ in counts.most_common(top_n)]

    # ------------------------------------------------------------------
    # NEW — Data Compaction Protocol
    # Fixes: "Request too large ... Limit 8000, Requested 8552" (413/rate_limit
    # errors from Groq). Root cause: the old code sent full competitor dicts
    # (title, brand, price, asin, star_rating, review_count...) as JSON for
    # both top5 AND top_rated (10 rows total) into the gap-analysis prompt.
    # This protocol strips that down to ONLY what the LLM actually needs to
    # judge — ASIN, Price, Monthly Sales, Title — in a compact, whitespace-
    # minimized, non-JSON format, and automatically drops to the Top 3
    # competitors if even that would exceed a safe token budget.
    # ------------------------------------------------------------------

    # Conservative by design: Groq's TPM cap covers INPUT + OUTPUT combined,
    # and includes the system prompt too. Targeting well under the raw
    # 7,000/8,000 numbers leaves headroom for those, not just this one blob.
    COMPACT_TARGET_TOKENS = 3000    # preferred ceiling for the competitor blob alone
    COMPACT_HARD_CAP_TOKENS = 7000  # matches your spec; step-down triggers here

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap, dependency-free estimate (~4 chars/token for English) —
        no tiktoken/model-specific tokenizer needed for a safety-margin check."""
        return max(1, len(text) // 4)

    def _compact_row(self, row: Dict[str, Any]) -> Dict[str, str]:
        """
        Column Filtering: EXACTLY the 4 columns needed for gap analysis —
        Title, Monthly Units Sold, Star Rating, Price. ASIN is dropped from
        the LLM-facing payload entirely (it's an identifier, not analytical
        signal) — it still lives in the full row dict for the UI/PDF, which
        never touches the LLM prompt at all.
        """
        rating = row.get("star_rating")
        return {
            "Title": (row.get("title", "") or "")[:100],  # defensive cap per title
            "Sales": f"{row.get('avg_monthly_sales', 0):.0f}",
            "Rating": f"{rating:.1f}" if rating else "-",
            "Price": row.get("price", "") or "-",
        }

    def format_compact(self, rows: List[Dict[str, Any]]) -> str:
        """Compact Formatting: pipe-delimited, minimal whitespace, no JSON
        braces/quotes, no conversational filler — the cheapest possible
        token representation of the same information."""
        if not rows:
            return ""
        lines = ["Title|Sales|Rating|Price"]
        for row in rows:
            c = self._compact_row(row)
            lines.append(f"{c['Title']}|{c['Sales']}|{c['Rating']}|{c['Price']}")
        return "\n".join(lines)

    def build_compact_payload(self, top5: List[Dict[str, Any]],
                               top_rated: List[Dict[str, Any]],
                               features: str,
                               target_tokens: int = COMPACT_TARGET_TOKENS,
                               hard_cap_tokens: int = COMPACT_HARD_CAP_TOKENS
                               ) -> Dict[str, Any]:
        """
        Constraint Enforcement: builds the smallest payload that still gives
        the gap-analysis agent useful signal, stepping down in this order
        until it fits:
          1. Top 5 + Top Rated block (full detail)
          2. Top 5 only (drop Top Rated — least essential for gap analysis)
          3. Top 3 only (your specified fallback, triggered at hard_cap_tokens)

        Returns {"compact_text": str, "estimated_tokens": int,
                 "competitor_count": int, "stepped_down": bool}
        """
        def _build(n: int, include_rated: bool) -> str:
            parts = [f"FEATURES: {features[:300]}", "", "TOP COMPETITORS (Title|Sales|Rating|Price):",
                     self.format_compact(top5[:n])]
            if include_rated and top_rated:
                parts += ["", "TOP RATED (customer satisfaction signal):",
                          self.format_compact(top_rated[:n])]
            return "\n".join(parts)

        # Attempt 1: full 5 + rated
        text = _build(5, True)
        tokens = self.estimate_tokens(text)
        if tokens <= target_tokens:
            return {"compact_text": text, "estimated_tokens": tokens,
                    "competitor_count": min(5, len(top5)), "stepped_down": False}

        # Attempt 2: 5 competitors, drop the rated block
        text = _build(5, False)
        tokens = self.estimate_tokens(text)
        if tokens <= hard_cap_tokens:
            return {"compact_text": text, "estimated_tokens": tokens,
                    "competitor_count": min(5, len(top5)), "stepped_down": False}

        # Attempt 3: your specified fallback — Top 3 only, no rated block
        text = _build(3, False)
        tokens = self.estimate_tokens(text)
        return {"compact_text": text, "estimated_tokens": tokens,
                "competitor_count": min(3, len(top5)), "stepped_down": True}

    # ------------------------------------------------------------------
    # PDF report — now includes a "Top Rated Competitors" section
    # ------------------------------------------------------------------
    def _draw_competitor_table(self, pdf: FPDF, rows: List[Dict[str, Any]]) -> None:
        """Real table with fixed columns: ASIN, Brand, Price, Monthly Units
        Sold, Star Rating, Keywords — matching a proper market-matrix layout
        instead of a flat bullet list."""
        col_widths = [22, 22, 16, 24, 16, 90]  # mm, sums to ~190 (A4 usable width)
        headers = ["ASIN", "Brand", "Price", "Monthly Sold", "Rating", "Title Keywords"]

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(230, 230, 230)
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, h, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for c in rows:
            title_kws = ", ".join(self.extract_keywords([c.get("title", "")], top_n=4))
            rating_str = f"{c['star_rating']:.1f}" if c.get("star_rating") else "N/A"
            cells = [
                sanitize_pdf_text(c.get("asin", ""))[:12],
                sanitize_pdf_text(c.get("brand", ""))[:12],
                sanitize_pdf_text(c.get("price", ""))[:8],
                f"{c.get('avg_monthly_sales', 0):.0f}",
                rating_str,
                sanitize_pdf_text(title_kws)[:55],
            ]
            for w, val in zip(col_widths, cells):
                pdf.cell(w, 7, val, border=1)
            pdf.ln()

    def build_pdf_report(self, gap_report: Dict[str, Any], brand: str = "") -> str:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Competitor Analysis Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, sanitize_pdf_text(f"Brand: {brand or 'N/A'}"), ln=True)
        pdf.ln(4)

        # --- Phase 1: Top 5 Competitors Market Matrix (real table, not text) ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Phase 1: Top 5 Competitors Market Matrix (by Monthly Units Sold)", ln=True)
        top5 = gap_report.get("top5", [])
        if top5:
            self._draw_competitor_table(pdf, top5)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, "No competitor rows found in this export.")
        pdf.ln(4)

        top_rated = gap_report.get("top_rated", [])
        if top_rated:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Top Rated Competitors (Customer Satisfaction Signal)", ln=True)
            self._draw_competitor_table(pdf, top_rated)
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(
                0, 5,
                sanitize_pdf_text(
                    f"Minimum {MIN_REVIEWS_FOR_RATING_SIGNAL} reviews required to "
                    "appear here, filtering out low-review-count outliers."
                )
            )
            pdf.ln(4)

        # --- Phase 2: Actionable Sourcing & Copywriting Solutions ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Phase 2: Actionable Sourcing & Copywriting Solutions", ln=True)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "How to Beat These Competitors:", ln=True)
        pdf.set_font("Helvetica", "", 10)

        pdf.multi_cell(0, 6, sanitize_pdf_text(
            "Sourcing & Design Upgrade: identify high-volume, lower-rated "
            "competitors and out-build them on the specific gaps below."
        ))
        for g in gap_report.get("positioning_gaps", []):
            pdf.multi_cell(0, 6, sanitize_pdf_text(f"  - {g}"))
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Copywriting & Keyword Traffic Strategy:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, "Use these high-converting keywords, pulled directly "
                             "from top-selling and top-rated competitor titles:")
        for kw in gap_report.get("competitor_keywords", [])[:15]:
            pdf.multi_cell(0, 6, sanitize_pdf_text(f"  - \"{kw}\""))
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Missing Keywords (not yet in your listing):", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text(", ".join(gap_report.get("missing_keywords", [])) or "None found"))
        pdf.ln(1)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Missing Features (competitors have, you don't):", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text(", ".join(gap_report.get("missing_features", [])) or "None found"))
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text(gap_report.get("summary", "")))

        filename = f"competitor_report_{uuid.uuid4().hex[:8]}.pdf"
        path = os.path.join(REPORTS_DIR, filename)
        pdf.output(path)
        logger.info("Competitor PDF report written to %s", path)
        return path