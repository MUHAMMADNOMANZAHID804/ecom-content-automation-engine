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
# FPDF imported for type-reference only — SafeFPDF (below) is what we actually instantiate
from fpdf import FPDF

try:
    from scripts.pdf_text_utils import sanitize_pdf_text, SafeFPDF
except ImportError:  # pragma: no cover
    # FIX: this exact ImportError has happened repeatedly from partial
    # deploys (this file updated to import SafeFPDF before
    # scripts/pdf_text_utils.py was pushed with the SafeFPDF class defined).
    # An ImportError here kills the ENTIRE app at startup, not just PDF
    # generation — so if pdf_text_utils.py is somehow still the old version,
    # fall back to plain FPDF (losing the extra crash-resilience, but the
    # app stays up) instead of crashing on import.
    from scripts.pdf_text_utils import sanitize_pdf_text
    from fpdf import FPDF as SafeFPDF

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

    # NEW — additional Jungle Scout signals worth capturing (per your CSV):
    # BSR (Best Seller Rank — lower is more competitive, independent of raw
    # unit count), Monthly Revenue ($, different signal than unit volume —
    # a lower-volume/higher-price competitor can out-earn a high-volume one),
    # Fulfillment (FBA vs FBM — affects how hard a competitor is to beat on
    # shipping/Prime eligibility), Category, and Seller Count (more sellers
    # on one listing = more price competition on that exact product).
    BSR_COLUMN_CANDIDATES = ["BSR", "Best Sellers Rank", "Best Seller Rank"]
    REVENUE_COLUMN_CANDIDATES = ["Monthly Revenue", "Net Revenue"]
    FULFILLMENT_COLUMN_CANDIDATES = ["Fulfillment"]
    CATEGORY_COLUMN_CANDIDATES = ["Category"]
    SELLERS_COLUMN_CANDIDATES = ["No. of Sellers", "Number of Sellers", "Sellers"]

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

    # Recognizable header-cell keywords used to auto-detect which row is the
    # real header, when it isn't row 0 (some Jungle Scout / sheet exports
    # have a title row, blank row, or metadata row before the real headers).
    HEADER_DETECTION_KEYWORDS = [
        "asin", "title", "product name", "price", "sales", "units sold",
        "rating", "brand", "reviews",
    ]

    def _detect_header_row(self, csv_path: str, max_scan_rows: int = 10) -> int:
        """Scans the first `max_scan_rows` raw rows (no header assumed) and
        returns the index of the first one that looks like a real header —
        i.e. contains at least 2 recognizable column-name keywords. Returns
        0 (assume first row) if nothing better is found.

        Uses Python's csv module rather than pandas here deliberately:
        pandas' C parser infers the column count from the FIRST row and
        raises a ParserError on any later row with a different count —
        which is exactly the shape of the files that need this detection
        (a 1-column title/metadata row followed by a wider real header).
        """
        import csv as _csv
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = _csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= max_scan_rows:
                        break
                    values = [str(v).strip().lower() for v in row if v and str(v).strip()]
                    matches = sum(1 for v in values for kw in self.HEADER_DETECTION_KEYWORDS if kw in v)
                    if matches >= 2:
                        return i
        except Exception as e:  # noqa: BLE001
            logger.warning("Header-row detection failed, assuming row 0: %s", e)
        return 0

    def _load_and_map_columns(self, csv_path: str) -> tuple:
        df = pd.read_csv(csv_path)

        sales_col = self._find_column(df, self.SALES_COLUMN_CANDIDATES)
        title_col = self._find_column(df, self.TITLE_COLUMN_CANDIDATES)

        if not sales_col or not title_col:
            # FIX: some exports have a title/metadata/blank row before the
            # real header row, so pandas reads everything as 'Unnamed: N'
            # and column matching fails even though the data is fine.
            # Auto-detect the real header row and retry once before giving up.
            header_row = self._detect_header_row(csv_path)
            if header_row > 0:
                logger.info("Retrying CSV read with header row detected at index %s", header_row)
                df = pd.read_csv(csv_path, skiprows=header_row, header=0)
                sales_col = self._find_column(df, self.SALES_COLUMN_CANDIDATES)
                title_col = self._find_column(df, self.TITLE_COLUMN_CANDIDATES)

        if not sales_col or not title_col:
            raise ValueError(
                "Could not find a sales-volume column or a title/product-name "
                f"column in this CSV. Found columns: {list(df.columns)}. "
                "If your export uses different header names, add them to "
                "SALES_COLUMN_CANDIDATES / TITLE_COLUMN_CANDIDATES in "
                "modules/competitor_analyzer.py."
            )

        cols = {
            "sales": sales_col,
            "title": title_col,
            "rating": self._find_column(df, self.RATING_COLUMN_CANDIDATES),
            "reviews": self._find_column(df, self.REVIEWS_COLUMN_CANDIDATES),
            "brand": self._find_column(df, self.BRAND_COLUMN_CANDIDATES),
            "price": self._find_column(df, self.PRICE_COLUMN_CANDIDATES),
            "asin": self._find_column(df, self.ASIN_COLUMN_CANDIDATES),
            # NEW signals — any of these may be None if the CSV doesn't have
            # them; every downstream use already handles None gracefully.
            "bsr": self._find_column(df, self.BSR_COLUMN_CANDIDATES),
            "revenue": self._find_column(df, self.REVENUE_COLUMN_CANDIDATES),
            "fulfillment": self._find_column(df, self.FULFILLMENT_COLUMN_CANDIDATES),
            "category": self._find_column(df, self.CATEGORY_COLUMN_CANDIDATES),
            "sellers": self._find_column(df, self.SELLERS_COLUMN_CANDIDATES),
        }
        return df, cols

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

    def _row_to_dict(self, row, cols: Dict[str, Optional[str]]) -> Dict[str, Any]:
        def _str(field: str) -> str:
            col = cols.get(field)
            return str(row[col]) if col and pd.notna(row.get(col)) else ""

        sales_val = self._safe_float(row.get(cols["sales"]))
        rating_val = self._safe_float(row.get(cols["rating"])) if cols.get("rating") else None
        reviews_val = self._safe_float(row.get(cols["reviews"])) if cols.get("reviews") else None
        bsr_val = self._safe_float(row.get(cols["bsr"])) if cols.get("bsr") else None
        sellers_val = self._safe_float(row.get(cols["sellers"])) if cols.get("sellers") else None

        return {
            "title": str(row[cols["title"]]),
            "avg_monthly_sales": sales_val if sales_val is not None else 0.0,
            "star_rating": rating_val,
            "review_count": int(reviews_val) if reviews_val is not None else None,
            "brand": _str("brand"),
            "price": _str("price"),
            "asin": _str("asin"),
            # NEW signals (per your request to use more of the sheet):
            "bsr": int(bsr_val) if bsr_val is not None else None,
            "monthly_revenue": _str("revenue"),
            "fulfillment": _str("fulfillment"),
            "category": _str("category"),
            "seller_count": int(sellers_val) if sellers_val is not None else None,
        }

    # ------------------------------------------------------------------
    # Ranking #1 — by sales volume (drives the main gap report)
    # ------------------------------------------------------------------
    def get_top5_by_avg_monthly_sales(self, csv_path: str) -> List[Dict[str, Any]]:
        df, cols = self._load_and_map_columns(csv_path)

        df = df.copy()
        df["_sales_numeric"] = df[cols["sales"]].apply(self._safe_float).fillna(0.0)
        top5 = df.sort_values(by="_sales_numeric", ascending=False).head(5)

        return [self._row_to_dict(row, cols) for _, row in top5.iterrows()]

    # ------------------------------------------------------------------
    # Ranking #2 — by star rating (customer-satisfaction signal you asked for)
    # Filters out low-review-count outliers so a 5.0 on 3 reviews doesn't win.
    # ------------------------------------------------------------------
    def get_top5_by_star_rating(self, csv_path: str) -> List[Dict[str, Any]]:
        df, cols = self._load_and_map_columns(csv_path)

        if not cols.get("rating"):
            logger.warning("No star-rating column found in this CSV; skipping rating ranking.")
            return []

        df = df.copy()
        df["_rating_numeric"] = df[cols["rating"]].apply(self._safe_float)
        df = df[df["_rating_numeric"].notna()]

        if cols.get("reviews"):
            df["_reviews_numeric"] = df[cols["reviews"]].apply(self._safe_float).fillna(0.0)
            df = df[df["_reviews_numeric"] >= MIN_REVIEWS_FOR_RATING_SIGNAL]

        if df.empty:
            return []

        top5 = df.sort_values(by="_rating_numeric", ascending=False).head(5)
        return [self._row_to_dict(row, cols) for _, row in top5.iterrows()]

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
    def _draw_table_or_fallback(self, pdf: FPDF, rows: List[Dict[str, Any]]) -> None:
        """Outermost safety net around _draw_competitor_table: if the table
        layout itself fails (e.g. an fpdf2 version difference on a
        deployment host raises on the header row, before per-row recovery
        even gets a chance to run), fall back to a plain bullet list rather
        than crashing the whole PDF/competitor-analysis flow."""
        try:
            self._draw_competitor_table(pdf, rows)
        except Exception as e:  # noqa: BLE001
            logger.warning("Competitor table failed entirely, falling back to bullet list: %s", e)
            pdf.set_font("Helvetica", "", 10)
            for c in rows:
                rating_str = f"{c['star_rating']:.1f}/5" if c.get("star_rating") else "N/A"
                line = (f"- {sanitize_pdf_text(c.get('title', ''))[:80]}  |  "
                        f"Sales: {c.get('avg_monthly_sales', 0):.0f}  |  Rating: {rating_str}")
                pdf.multi_cell(0, 6, line)

    def _draw_competitor_table(self, pdf: FPDF, rows: List[Dict[str, Any]]) -> None:
        """Real table with fixed columns: ASIN, Brand, Price, Monthly Units
        Sold, Star Rating, Keywords — matching a proper market-matrix layout
        instead of a flat bullet list.

        FIX: "Not enough horizontal space to render a single character" —
        an fpdf2 exception seen on Streamlit Cloud (different fpdf2 version
        than local) when column widths sum too close to the page's usable
        width, or a cell value is empty/oddly formatted. Column widths now
        total 165mm (comfortable margin under A4's ~190mm usable width,
        instead of the old 190mm which left zero room for error), every
        cell value has a non-empty fallback, and the whole per-row render
        is wrapped so a single bad row can't take down the entire PDF —
        it falls back to a plain-text line for that row instead.
        """
        col_widths = [20, 20, 16, 22, 14, 73]  # mm, sums to 165 (safe margin under ~190mm usable)
        headers = ["ASIN", "Brand", "Price", "Monthly Sold", "Rating", "Title Keywords"]

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(230, 230, 230)
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, h, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for c in rows:
            try:
                title_kws = ", ".join(self.extract_keywords([c.get("title", "")], top_n=4))
                rating_str = f"{c['star_rating']:.1f}" if c.get("star_rating") else "N/A"
                cells = [
                    sanitize_pdf_text(c.get("asin", "") or "-")[:12] or "-",
                    sanitize_pdf_text(c.get("brand", "") or "-")[:12] or "-",
                    sanitize_pdf_text(c.get("price", "") or "-")[:8] or "-",
                    f"{c.get('avg_monthly_sales', 0):.0f}",
                    rating_str,
                    (sanitize_pdf_text(title_kws)[:45] or sanitize_pdf_text(c.get("title", ""))[:45] or "-"),
                ]
                for w, val in zip(col_widths, cells):
                    pdf.cell(w, 7, val, border=1)
                pdf.ln()
            except Exception as e:  # noqa: BLE001
                # Never let one row's layout quirk crash the whole PDF —
                # fall back to a single plain-text line for this row.
                logger.warning("Competitor table row failed to render, falling back to text: %s", e)
                pdf.set_font("Helvetica", "", 8)
                fallback_line = sanitize_pdf_text(
                    f"{c.get('asin', '-')}  {c.get('brand', '-')}  "
                    f"{c.get('price', '-')}  {c.get('avg_monthly_sales', 0):.0f}  "
                    f"{c.get('title', '')[:60]}"
                )
                pdf.cell(0, 7, fallback_line, border=1, ln=True)

    def _build_pdf_report_impl(self, gap_report: Dict[str, Any], brand: str = "") -> str:
        pdf = SafeFPDF()
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
            self._draw_table_or_fallback(pdf, top5)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, "No competitor rows found in this export.")
        pdf.ln(4)

        top_rated = gap_report.get("top_rated", [])
        if top_rated:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Top Rated Competitors (Customer Satisfaction Signal)", ln=True)
            self._draw_table_or_fallback(pdf, top_rated)
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

        # --- NEW: Market Insights — BSR, Revenue, Fulfillment, Sellers ---
        if top5:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Market Insights", ln=True)
            pdf.set_font("Helvetica", "", 10)

            with_bsr = [c for c in top5 if c.get("bsr")]
            if with_bsr:
                best_bsr = min(with_bsr, key=lambda c: c["bsr"])
                pdf.multi_cell(0, 6, sanitize_pdf_text(
                    f"Most competitive by Best Seller Rank: "
                    f"{best_bsr.get('title', '')[:70]} (BSR #{best_bsr['bsr']:,}) "
                    "— lower BSR means it outsells more of the category than raw "
                    "unit counts alone show."
                ))

            with_revenue = [c for c in top5 if c.get("monthly_revenue")]
            if with_revenue:
                pdf.multi_cell(0, 6, sanitize_pdf_text(
                    "Monthly revenue (top 5): " +
                    ", ".join(f"{c.get('monthly_revenue', '-')}" for c in with_revenue)
                ))

            fba_count = sum(1 for c in top5 if str(c.get("fulfillment", "")).upper() == "FBA")
            if fba_count:
                pdf.multi_cell(0, 6, sanitize_pdf_text(
                    f"Fulfillment: {fba_count}/{len(top5)} top competitors use FBA "
                    "(Prime-eligible, harder to compete against on shipping)."
                ))

            seller_counts = [c["seller_count"] for c in top5 if c.get("seller_count")]
            if seller_counts:
                crowded = [c for c in top5 if c.get("seller_count", 0) and c["seller_count"] > 3]
                if crowded:
                    pdf.multi_cell(0, 6, sanitize_pdf_text(
                        f"{len(crowded)} of your top 5 competitors already have 3+ "
                        "sellers on the same listing — expect price competition there."
                    ))
            pdf.ln(3)

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

    def build_pdf_report(self, gap_report: Dict[str, Any], brand: str = "") -> str:
        """
        Public entry point — wraps the full report build in a top-level
        safety net. Whatever the exact fpdf2 version-specific cause of
        "Not enough horizontal space to render a single character" turns out
        to be on a given deployment host, this guarantees the competitor
        analysis flow ALWAYS produces a PDF (even a minimal one) instead of
        crashing the whole tool.
        """
        try:
            return self._build_pdf_report_impl(gap_report, brand)
        except Exception as e:  # noqa: BLE001
            logger.error("Full competitor PDF report failed (%s) — falling back "
                         "to a minimal plain-text version.", e)
            return self._build_minimal_fallback_pdf(gap_report, brand)

    def _build_minimal_fallback_pdf(self, gap_report: Dict[str, Any], brand: str = "") -> str:
        """Ultra-safe fallback: no tables, no fixed-width cells — just
        full-width multi_cell() text, which cannot trigger the same
        column-width failure mode as the table layout."""
        pdf = SafeFPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Competitor Analysis Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text(f"Brand: {brand or 'N/A'}"))
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top Competitors (by Monthly Units Sold)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for c in gap_report.get("top5", []):
            rating_str = f"{c['star_rating']:.1f}/5" if c.get("star_rating") else "N/A"
            pdf.multi_cell(0, 6, sanitize_pdf_text(
                f"- {c.get('title', '')[:90]} | Sales: {c.get('avg_monthly_sales', 0):.0f} "
                f"| Rating: {rating_str} | Price: {c.get('price', '-')}"
            ))
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Keywords & Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text(
            "Keywords: " + ", ".join(gap_report.get("competitor_keywords", [])[:15])
        ))
        pdf.multi_cell(0, 6, sanitize_pdf_text(gap_report.get("summary", "")))

        filename = f"competitor_report_{uuid.uuid4().hex[:8]}.pdf"
        path = os.path.join(REPORTS_DIR, filename)
        pdf.output(path)
        logger.info("Minimal fallback competitor PDF written to %s", path)
        return path
