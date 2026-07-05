"""
modules/review_analyzer.py
----------------------------
# SYNC-CHECK: v3-SafeFPDF (search this exact string on GitHub to confirm
# this file actually updated — if you don't see it, the push didn't take)
All-platform tool. Input: pasted reviews (or ASIN-resolved reviews from Phase 1).
Process: complaints, praises, gaps, sentiment.
Output: structured insight data + downloadable PDF report.

Also owns the two other PDF builders used by the pipeline (final listing summary)
and the manual-report-upload parser used by core/manager.py:run_listing_only(),
since 'Report Upload Box' -> re-hydrating prior reports is conceptually a review/
report-processing concern.

All PDF text goes through scripts/pdf_text_utils.sanitize_pdf_text() before
reaching FPDF — pasted customer reviews and LLM-generated listing copy are
exactly the kind of text likely to contain curly quotes, em-dashes, emoji, or
non-Latin-1 characters that would otherwise crash pdf.output().
"""

import os
import re
import uuid
import logging
from typing import Any, Dict, List

# FPDF imported for type-reference only — SafeFPDF (below) is what we actually instantiate
from fpdf import FPDF

try:
    from scripts.pdf_text_utils import sanitize_pdf_text, SafeFPDF
except ImportError:  # pragma: no cover
    # FIX: same partial-deploy protection as competitor_analyzer.py — never
    # let a missing SafeFPDF crash the entire app at import time.
    from scripts.pdf_text_utils import sanitize_pdf_text
    from fpdf import FPDF as SafeFPDF

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

logger = logging.getLogger("review_analyzer")

REPORTS_DIR = os.getenv("REPORTS_DIR", "generated_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


class ReviewAnalyzer:
    """
    Standalone-runnable module (spec: 'run tools independently, e.g. only Review
    Analysis'). Also called from core/manager.py Phase 3 and Phase 8.
    """

    # ------------------------------------------------------------------
    # Phase 3 support
    # ------------------------------------------------------------------
    def build_pdf_report(self, insights: Dict[str, Any], brand: str = "") -> str:
        pdf = SafeFPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Review Analysis Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, sanitize_pdf_text(f"Brand: {brand or 'N/A'}"), ln=True)
        score = insights.get("sentiment_score", 0.0)
        pdf.cell(0, 8, f"Overall Sentiment Score: {score:.2f} (-1 to 1)", ln=True)
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Complaints / Pain Points", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for c in insights.get("complaints", []):
            issue = c.get("issue", "") if isinstance(c, dict) else str(c)
            freq = c.get("frequency", "") if isinstance(c, dict) else ""
            pdf.multi_cell(0, 6, sanitize_pdf_text(f"- [{freq}] {issue}"))
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Praises", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for p in insights.get("praises", []):
            pdf.multi_cell(0, 6, sanitize_pdf_text(f"- {p}"))
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Improvement Strategies", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for s in insights.get("improvement_strategies", []):
            pdf.multi_cell(0, 6, sanitize_pdf_text(f"- {s}"))

        filename = f"review_report_{uuid.uuid4().hex[:8]}.pdf"
        path = os.path.join(REPORTS_DIR, filename)
        pdf.output(path)
        logger.info("Review PDF report written to %s", path)
        return path

    # ------------------------------------------------------------------
    # Phase 8 support — final listing summary PDF
    # ------------------------------------------------------------------

    def build_listing_summary_pdf(self, brand: str, platform: str,
                                   sections: Dict[str, Any],
                                   gap_report: Dict[str, Any],
                                   keywords: Dict[str, Any],
                                   competitive_summary: str,
                                   referee_verdict: Dict[str, Any] = None,
                                   risk_assessment: Dict[str, Any] = None) -> str:
        """
        Final Optimization Report — the clean, publish-ready listing plus the
        research that justified it. Per your request, this NO LONGER includes
        an audit/PASS-FAIL section — that lives in the UI now
        (state.full_audit_report), not the PDF, so the PDF stays a clean
        deliverable without QA annotations mixed into the content.

        `sections` is whatever core/manager.py's platform-aware
        _split_sections() returned — its shape differs by platform:
          amazon:  {"title": str, "bullets": [str], "description": str}
          etsy:    {"title": str, "tags": [str], "description": str}
          shopify: {"seo_title": str, "meta_description": str,
                     "features": [str], "description": str}

        referee_verdict / risk_assessment are OPTIONAL (default None).
        """
        pdf = SafeFPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Listing Generator - Final Optimization Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, sanitize_pdf_text(f"Brand: {brand or 'N/A'}  |  Platform: {platform.capitalize()}"), ln=True)
        pdf.ln(4)

        # --- 1. FINAL LISTING (platform-specific layout, no audit clutter) ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "1. FINAL LISTING", ln=True)

        if platform == "etsy":
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Title", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, sanitize_pdf_text(sections.get("title", "") or "(not generated)"))
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Tags", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, sanitize_pdf_text(", ".join(sections.get("tags", [])) or "(not generated)"))
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Description", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, sanitize_pdf_text(sections.get("description", "") or "(not generated)"))

        elif platform == "shopify":
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "SEO Title", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, sanitize_pdf_text(sections.get("seo_title", "") or "(not generated)"))
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Meta Description", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, sanitize_pdf_text(sections.get("meta_description", "") or "(not generated)"))
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Key Features", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for feat in sections.get("features", []):
                pdf.multi_cell(0, 6, sanitize_pdf_text(f"- {feat}"))
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Product Description", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, sanitize_pdf_text(sections.get("description", "") or "(not generated)"))

        else:  # amazon
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Title", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, sanitize_pdf_text(sections.get("title", "") or "(not generated)"))
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Bullet Points", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for b in sections.get("bullets", []):
                pdf.multi_cell(0, 6, sanitize_pdf_text(f"* {b}"))
                pdf.ln(1)
            pdf.ln(1)

            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Description", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, sanitize_pdf_text(sections.get("description", "") or "(not generated)"))

        pdf.ln(3)

        # --- 2. COMPETITOR GAP ANALYSIS ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "2. COMPETITOR GAP ANALYSIS", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text(
            "Missing keywords: " + (", ".join(gap_report.get("missing_keywords", [])) or "None")
        ))
        pdf.multi_cell(0, 6, sanitize_pdf_text(
            "Missing features: " + (", ".join(gap_report.get("missing_features", [])) or "None")
        ))
        for g in gap_report.get("positioning_gaps", []):
            pdf.multi_cell(0, 6, sanitize_pdf_text(f"- Opportunity: {g}"))
        if gap_report.get("summary"):
            pdf.ln(1)
            pdf.multi_cell(0, 6, sanitize_pdf_text(gap_report["summary"]))
        pdf.ln(3)

        # --- 3. MASTER KEYWORD LIST ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "3. MASTER KEYWORD LIST", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text("Tier 1 (primary): " + ", ".join(keywords.get("tier1", []))))
        pdf.multi_cell(0, 6, sanitize_pdf_text("Tier 2 (secondary): " + ", ".join(keywords.get("tier2", []))))
        pdf.multi_cell(0, 6, sanitize_pdf_text("Tier 3 (long-tail): " + ", ".join(keywords.get("tier3", []))))
        if platform == "amazon":
            pdf.multi_cell(0, 6, sanitize_pdf_text("Bullet openers: " + ", ".join(keywords.get("bullet_openers", []))))
        pdf.ln(3)

        # --- 4. COMPETITIVE ADVANTAGE SUMMARY ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "4. COMPETITIVE ADVANTAGE SUMMARY", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, sanitize_pdf_text(competitive_summary or "N/A"))

        # --- 5. REFEREE LLM VERDICT (new — Podcast 5 semantic gate) ---
        if referee_verdict:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "5. REFEREE LLM VERDICT (Semantic Policy Check)", ln=True)
            verdict = referee_verdict.get("verdict", "UNKNOWN")
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, sanitize_pdf_text(f"Verdict: {verdict}"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            concerns = referee_verdict.get("concerns", [])
            if concerns:
                pdf.cell(0, 6, "Concerns flagged:", ln=True)
                for c in concerns:
                    pdf.multi_cell(0, 6, sanitize_pdf_text(f"- {c}"))
            else:
                pdf.multi_cell(0, 6, "No semantic policy concerns flagged.")

        # --- 6. BRUTAL RISK ASSESSMENT (new — capstone roadmap item 5) ---
        if risk_assessment:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "6. BRUTAL RISK ASSESSMENT (Before You Publish)", ln=True)
            level = risk_assessment.get("risk_level", "UNKNOWN")
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, sanitize_pdf_text(f"Risk Level: {level}"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            risks = risk_assessment.get("risks", [])
            if risks:
                for r in risks:
                    pdf.multi_cell(0, 6, sanitize_pdf_text(f"- {r}"))
            else:
                pdf.multi_cell(0, 6, "No specific risks identified.")
            if risk_assessment.get("recommendation"):
                pdf.set_font("Helvetica", "I", 10)
                pdf.multi_cell(0, 6, sanitize_pdf_text(
                    f"Recommendation: {risk_assessment['recommendation']}"
                ))

        filename = f"listing_summary_{uuid.uuid4().hex[:8]}.pdf"
        path = os.path.join(REPORTS_DIR, filename)
        pdf.output(path)
        logger.info("Listing summary PDF written to %s", path)
        return path

    # ------------------------------------------------------------------
    # Manual Override support (Report Upload Box)
    # ------------------------------------------------------------------
    def parse_uploaded_reports(self, paths: List[str]) -> Dict[str, Any]:
        """
        Best-effort text extraction from previously downloaded PDF reports so the
        Listing Generator can reference them when a user skipped 'Auto-Add'.
        Heuristically buckets extracted text into gap_report / review_report by
        filename/content keywords — good enough as a fallback path; Auto-Add
        remains the recommended flow since it preserves structured data.
        """
        result: Dict[str, Any] = {"gap_report": {}, "review_report": {}}
        if PdfReader is None:
            logger.warning("pypdf not installed; cannot parse uploaded PDFs.")
            return result

        for path in paths:
            try:
                reader = PdfReader(path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to parse uploaded report %s: %s", path, e)
                continue

            lower_name = os.path.basename(path).lower()
            if "competitor" in lower_name or "competitor analysis" in text.lower():
                result["gap_report"].setdefault("summary", "")
                result["gap_report"]["summary"] += text[:2000]
                result["gap_report"]["missing_keywords"] = self._extract_line_list(
                    text, "Missing Keywords"
                )
            else:
                result["review_report"].setdefault("complaints", [])
                complaints = re.findall(r"-\s*\[(.*?)\]\s*(.+)", text)
                result["review_report"]["complaints"].extend(
                    [{"frequency": f.strip(), "issue": i.strip()} for f, i in complaints]
                )

        return result

    @staticmethod
    def _extract_line_list(text: str, section_header: str) -> List[str]:
        match = re.search(rf"{section_header}\n(.+)", text)
        if not match:
            return []
        return [w.strip() for w in match.group(1).split(",") if w.strip()]
