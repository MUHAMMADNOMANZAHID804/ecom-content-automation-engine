"""
core/manager.py
----------------
Anti-Gravity 2.0 — Pipeline Orchestrator

Uses Groq's openai/gpt-oss-120b as the ORCHESTRATOR model to make routing /
retry decisions, while delegating narrow tasks to the small-model sub-agents in
core/subagents.py (openai/gpt-oss-20b).

8-PHASE PIPELINE (integrity preserved, 2 modules + 1 ASIN phase folded in):

  Phase 1 - ASIN Resolution / Data Ingestion   (NEW: modules/*, subagents.ASINResolverAgent)
  Phase 2 - Competitor Gap Analysis            (modules/competitor_analyzer.py)
  Phase 3 - Review Sentiment / Pain-Point       (modules/review_analyzer.py)
  Phase 4 - RAG retrieval from KB               (scripts/build-kb.py output)
  Phase 5 - 3-source keyword synthesis          (competitor + review + RAG)
  Phase 6 - Listing generation (2 drafts -> 1 final)
  Phase 7 - Character/structure audit + auto-fix (<=2 retries)
  Phase 8 - Competitive advantage summary + PDF report

Existing safety-governor.py, circuit-breaker.py, audit-logger.py, keywords-check.py
and count-words.py hooks are preserved and called at the same points they always
were; nothing about their contract changes.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.subagents import (
    ASINResolverAgent,
    CompetitorGapAgent,
    ReviewInsightAgent,
    RAGRetrieverAgent,
    KeywordSynthesisAgent,
    ListingGeneratorAgent,
    AuditFixAgent,
    CompetitiveAdvantageAgent,
    RefereeAgent,             # NEW — Podcast 5 semantic gate
    RiskAssessmentAgent,      # NEW — Brutal Risk Assessment
)
from core.safety_governor import SafetyGovernor          # existing module, untouched
from scripts.circuit_breaker import CircuitBreaker         # existing module, untouched
from scripts.audit_logger import AuditLogger                # existing module, untouched
from scripts.keywords_check import check_keyword_stuffing   # existing module, untouched
from scripts.count_words import count_chars                 # existing module, untouched
from scripts.pii_redactor import redact_pii                 # NEW — enforces spec/security-policy.yaml
from scripts.file_message_bus import maybe_offload           # NEW — large-payload offloading
from tools.sandboxed_executor import SandboxedExecutor    # NEW — wires in the previously-orphaned harness
from modules.competitor_analyzer import CompetitorAnalyzer
from modules.review_analyzer import ReviewAnalyzer

logger = logging.getLogger("manager")

MAX_AUTOFIX_RETRIES = 2

TITLE_RANGE = (185, 200)
BULLET_RANGE = (300, 350)
DESC_RANGE = (1700, 1800)
N_BULLETS = 5


@dataclass
class PipelineState:
    """Carried across all 8 phases; also what 'Auto-Add' persists between tools."""
    platform: str = "amazon"                 # amazon | etsy | shopify
    asin: Optional[str] = None
    brand: str = ""
    features: str = ""
    manual_reviews_text: Optional[str] = None
    jungle_scout_csv_path: Optional[str] = None
    uploaded_report_paths: List[str] = field(default_factory=list)  # manual override

    ingested_product_data: Dict[str, Any] = field(default_factory=dict)
    gap_report: Dict[str, Any] = field(default_factory=dict)
    review_report: Dict[str, Any] = field(default_factory=dict)
    rag_context: Dict[str, Any] = field(default_factory=dict)
    keywords: Dict[str, Any] = field(default_factory=dict)
    drafts: List[str] = field(default_factory=list)
    final_listing: str = ""
    audit_violations: List[str] = field(default_factory=list)
    competitive_summary: str = ""
    pdf_paths: Dict[str, str] = field(default_factory=dict)
    phase_log: List[str] = field(default_factory=list)

    # NEW: mirrors the "Creativity" slider / "Auto-fix violations" toggle /
    # "Competitor Data (Titles + Negative Reviews)" manual box from the
    # reference Listing Generator UI. All optional — sensible defaults if
    # the UI never sets them (e.g. when driven from run_pipeline.py CLI).
    creativity: float = 0.65            # 0.0-1.0, maps to draft temperature
    auto_fix_enabled: bool = True        # if False, Phase 7 audits but never rewrites
    competitor_data_manual: Optional[str] = None  # manual paste, used if no CSV/Auto-Add

    # NEW (podcast-gap closures): all default to empty/None so existing code
    # that constructs PipelineState() with no args, or reads these fields
    # before they're populated, behaves exactly as before.
    referee_verdict: Dict[str, Any] = field(default_factory=dict)   # Podcast 5 semantic gate
    risk_assessment: Dict[str, Any] = field(default_factory=dict)    # Brutal Risk Assessment
    pii_redaction_stats: Dict[str, Any] = field(default_factory=dict)  # what was redacted, and where
    full_audit_report: List[Dict[str, str]] = field(default_factory=list)  # PASS/FAIL per section, UI-only


class PipelineManager:
    """
    The orchestrator. One instance per listing job.
    Call `.run_full_pipeline(state)` for the chained flow, or call individual
    `.phaseN_*` methods directly to support "run tools independently" (Section 3
    of the spec: Amazon needs Competitor+Review+Listing; Etsy/Shopify need only
    Review+Listing).
    """

    def __init__(self, scraper=None, kb_client=None):
        self.scraper = scraper          # tools/scrapper-mcp.py instance, injected
        self.kb_client = kb_client      # scripts/build-kb.py vector store handle
        self.safety = SafetyGovernor()
        self.breaker = CircuitBreaker()
        self.audit_log = AuditLogger()

        self.asin_agent = ASINResolverAgent()
        self.gap_agent = CompetitorGapAgent()
        self.review_agent = ReviewInsightAgent()
        self.rag_agent = RAGRetrieverAgent()
        self.kw_agent = KeywordSynthesisAgent()
        self.listing_agent = ListingGeneratorAgent()
        self.fix_agent = AuditFixAgent()
        self.advantage_agent = CompetitiveAdvantageAgent()
        self.referee_agent = RefereeAgent()              # NEW
        self.risk_agent = RiskAssessmentAgent()           # NEW
        self.sandbox = SandboxedExecutor()                # NEW — wires in tools/sandboxed_executor.py

        self.competitor_module = CompetitorAnalyzer()
        self.review_module = ReviewAnalyzer()

        # NEW: log the actual model routing once per run so it's always
        # visible in logs/run_*.jsonl which model handled which job — useful
        # after a model swap like this one, without changing any pipeline logic.
        self.audit_log.log("model_routing", {
            "ASINResolverAgent": self.asin_agent.model,
            "CompetitorGapAgent": self.gap_agent.model,
            "ReviewInsightAgent": self.review_agent.model,
            "RAGRetrieverAgent": self.rag_agent.model,
            "KeywordSynthesisAgent": self.kw_agent.model,
            "ListingGeneratorAgent": self.listing_agent.model,
            "AuditFixAgent": self.fix_agent.model,
            "CompetitiveAdvantageAgent": self.advantage_agent.model,
            "RefereeAgent": self.referee_agent.model,
            "RiskAssessmentAgent": self.risk_agent.model,
        })

    # ------------------------------------------------------------------
    # Phase 1 — ASIN Resolution / Data Ingestion (NEW)
    # ------------------------------------------------------------------
    def phase1_ingest(self, state: PipelineState) -> PipelineState:
        self.audit_log.log("phase1_start", {"asin": state.asin, "platform": state.platform})
        if state.asin:
            if not self.scraper:
                raise RuntimeError(
                    "ASIN provided but no scraper injected. Wire tools/scrapper-mcp.py "
                    "into PipelineManager(scraper=...)."
                )
            structured = self.breaker.call(
                lambda: self.asin_agent.run(state.asin, self.scraper)
            )
            state.ingested_product_data = structured
            if not state.manual_reviews_text and structured.get("reviews"):
                state.manual_reviews_text = "\n---\n".join(structured["reviews"])
        else:
            # Manual path: user already pasted reviews / gave a Jungle Scout CSV.
            state.ingested_product_data = {
                "title": None, "reviews": (state.manual_reviews_text or "").splitlines(),
            }
        state.phase_log.append("phase1_ingest")
        self.audit_log.log("phase1_done", {"has_asin_data": bool(state.asin)})
        return state

    # ------------------------------------------------------------------
    # Phase 2 — Competitor Gap Analysis (Amazon only; skip for Etsy/Shopify)
    # ------------------------------------------------------------------
    def phase2_competitor_analysis(self, state: PipelineState) -> PipelineState:
        if state.platform != "amazon":
            state.phase_log.append("phase2_skipped_non_amazon")
            return state
        if not state.jungle_scout_csv_path:
            state.phase_log.append("phase2_skipped_no_csv")
            return state

        self.audit_log.log("phase2_start", {"csv": state.jungle_scout_csv_path})
        top5 = self.competitor_module.get_top5_by_avg_monthly_sales(
            state.jungle_scout_csv_path
        )
        # NEW: also surface which competitors customers actually rate highest —
        # a high sales rank doesn't mean customers love the product, and a
        # separate "what do satisfied buyers respond to" signal is valuable
        # for the keyword/positioning strategy fed into the Listing Generator.
        top_rated = self.competitor_module.get_top5_by_star_rating(
            state.jungle_scout_csv_path
        )

        competitor_titles = [c["title"] for c in top5]
        competitor_keywords = self.competitor_module.extract_keywords(competitor_titles)
        top_rated_keywords = self.competitor_module.extract_keywords(
            [c["title"] for c in top_rated]
        ) if top_rated else []
        # Merge, de-duplicate, keep sales-driven keywords first since they
        # anchor search relevance; rating-driven keywords add conversion signal.
        merged_keywords = list(dict.fromkeys(competitor_keywords + top_rated_keywords))

        # NEW: Data Compaction protocol — fixes "Request too large" / 413
        # rate_limit_exceeded errors from Groq's free-tier TPM cap. Only
        # ASIN/Price/Sales/Title reach the LLM prompt (column filtering);
        # everything else (brand, rating, review count...) stays in
        # state.gap_report below for the UI/PDF, never touching the prompt.
        compact = self.competitor_module.build_compact_payload(
            top5, top_rated, state.features
        )
        self.audit_log.log("phase2_compaction", {
            "estimated_tokens": compact["estimated_tokens"],
            "competitor_count": compact["competitor_count"],
            "stepped_down_to_top3": compact["stepped_down"],
        })

        try:
            gap = self.breaker.call(
                lambda: self.gap_agent.run(
                    state.features, competitor_titles, merged_keywords,
                    compact_context=compact["compact_text"],
                )
            )
        except Exception as e:  # noqa: BLE001
            err_text = str(e).lower()
            if "413" in err_text or "too large" in err_text or "rate_limit" in err_text:
                # Last-resort fallback: force Top 3, no rated block, trimmed
                # keywords — smallest payload we're willing to send.
                logger.warning("Gap analysis still hit a token limit after compaction "
                                "(%s) — retrying once with Top 3 only.", e)
                minimal = self.competitor_module.build_compact_payload(
                    top5[:3], [], state.features,
                    target_tokens=1500, hard_cap_tokens=1500,
                )
                gap = self.breaker.call(
                    lambda: self.gap_agent.run(
                        state.features, competitor_titles[:3], merged_keywords[:10],
                        compact_context=minimal["compact_text"],
                    )
                )
            else:
                raise

        state.gap_report = {
            **gap,
            "top5": top5,
            "top_rated": top_rated,
            "competitor_keywords": merged_keywords,
        }

        # PDF output required by spec ("Every analysis step -> downloadable PDF")
        pdf_path = self.competitor_module.build_pdf_report(state.gap_report, brand=state.brand)
        state.pdf_paths["competitor_report"] = pdf_path

        state.phase_log.append("phase2_competitor_analysis")
        self.audit_log.log("phase2_done", {"pdf": pdf_path})
        return state

    # ------------------------------------------------------------------
    # Phase 3 — Review Sentiment / Pain-Point Analysis (all platforms)
    # ------------------------------------------------------------------
    def phase3_review_analysis(self, state: PipelineState) -> PipelineState:
        reviews_text = state.manual_reviews_text or ""
        if not reviews_text.strip():
            state.phase_log.append("phase3_skipped_no_reviews")
            return state

        # NEW: spec/security-policy.yaml declares pii_redaction: enabled, but
        # nothing previously enforced it. Customers pasting real reviews can
        # easily include an email/phone/address by accident — strip it
        # BEFORE it reaches the LLM prompt or gets embedded in a PDF.
        reviews_text, pii_counts = redact_pii(reviews_text)
        if any(pii_counts.values()):
            state.pii_redaction_stats["phase3_reviews"] = pii_counts
            logger.info("PII redacted from pasted reviews: %s", pii_counts)

        self.audit_log.log("phase3_start", {"chars": len(reviews_text),
                                               "pii_redacted": pii_counts})
        insights = self.breaker.call(lambda: self.review_agent.run(reviews_text))
        state.review_report = insights

        pdf_path = self.review_module.build_pdf_report(insights, brand=state.brand)
        state.pdf_paths["review_report"] = pdf_path

        state.phase_log.append("phase3_review_analysis")
        self.audit_log.log("phase3_done", {"pdf": pdf_path})
        return state

    # ------------------------------------------------------------------
    # Phase 4 — RAG Retrieval from KB
    # ------------------------------------------------------------------
    def phase4_rag_retrieval(self, state: PipelineState) -> PipelineState:
        self.audit_log.log("phase4_start", {})
        snippets: List[str] = []
        if self.kb_client:
            query = f"{state.brand} {state.features} {state.platform} listing policy"
            snippets = self.kb_client.query(query, top_k=8)
        rag = self.breaker.call(
            lambda: self.rag_agent.run(snippets, f"{state.brand}: {state.features}")
        )
        state.rag_context = rag
        state.phase_log.append("phase4_rag_retrieval")
        self.audit_log.log("phase4_done", {"n_snippets": len(snippets)})
        return state

    # ------------------------------------------------------------------
    # Phase 5 — 3-Source Keyword Synthesis
    # ------------------------------------------------------------------
    def phase5_keyword_synthesis(self, state: PipelineState) -> PipelineState:
        self.audit_log.log("phase5_start", {})
        competitor_kws = state.gap_report.get("competitor_keywords", [])
        review_kws = state.review_report.get("improvement_strategies", [])
        rag_kws = state.rag_context.get("tier1_keywords", [])

        kws = self.breaker.call(
            lambda: self.kw_agent.run(competitor_kws, review_kws, rag_kws)
        )
        # Deduplicate / cap defensively even though the sub-agent is instructed to.
        kws["tier1"] = list(dict.fromkeys(kws.get("tier1", [])))[:5]
        kws["bullet_openers"] = list(dict.fromkeys(kws.get("bullet_openers", [])))[:5]
        state.keywords = kws

        state.phase_log.append("phase5_keyword_synthesis")
        self.audit_log.log("phase5_done", {"tier1": kws["tier1"]})
        return state

    # ------------------------------------------------------------------
    # Phase 6 — Listing Generation (2 drafts -> 1 final)
    # ------------------------------------------------------------------
    def phase6_listing_generation(self, state: PipelineState) -> PipelineState:
        # Safety Governance / policy compliance check BEFORE we draft anything.
        policy_ok, policy_notes = self.safety.check(
            platform=state.platform, brand=state.brand, features=state.features
        )
        if not policy_ok:
            raise ValueError(f"Safety governor blocked generation: {policy_notes}")

        self.audit_log.log("phase6_start", {"creativity": state.creativity,
                                               "platform": state.platform})

        # If the user pasted competitor titles/negative-reviews directly into
        # the Listing Generator (the "Competitor Data" box, matching the
        # reference UI) fold it into the gap report the template sees — this
        # works even on Etsy/Shopify where there's no CSV analyzer at all,
        # and as a supplement on Amazon even when Auto-Add already ran.
        effective_gap = dict(state.gap_report)
        if state.competitor_data_manual and state.competitor_data_manual.strip():
            # NEW: this box is free text a human pastes in — same PII risk as
            # pasted reviews in Phase 3, so it gets the same treatment.
            redacted_notes, pii_counts = redact_pii(state.competitor_data_manual.strip())
            if any(pii_counts.values()):
                state.pii_redaction_stats["phase6_manual_competitor_data"] = pii_counts
                logger.info("PII redacted from manual competitor data: %s", pii_counts)
            effective_gap["manual_competitor_notes"] = redacted_notes

        # SINGLE-SHOT generation (per your request): one call produces the
        # final listing directly for the SELECTED PLATFORM's own template —
        # no more 2 drafts + a 3rd merge call. state.drafts is kept (empty)
        # only so nothing else that reads it breaks; nothing populates it now.
        final = self.breaker.call(
            lambda: self.listing_agent.generate_listing(
                state.brand, state.features, state.keywords,
                effective_gap, state.rag_context,
                platform=state.platform, temperature=state.creativity,
            )
        )
        state.final_listing = final
        state.drafts = []

        # Safety net: subagents.ListingGeneratorAgent already retries once
        # internally on empty output, but if it's STILL empty here, don't
        # silently hand an empty listing to Phase 7 (that's what previously
        # produced "TITLE length 0" audit results with no clear error).
        first_header = {"amazon": "--- TITLE", "etsy": "--- TITLE",
                         "shopify": "--- SEO TITLE"}.get(state.platform, "--- TITLE")
        if not state.final_listing.strip() or first_header not in state.final_listing:
            raise RuntimeError(
                "Listing generation returned empty/incomplete output after "
                "retries. This usually means the model ran out of its token "
                "budget on internal reasoning — check LISTING_DATA_MODEL and "
                "LISTING_MAX_TOKENS in core/subagents.py, or try again."
            )

        state.phase_log.append("phase6_listing_generation")
        # NEW: File Message Bus (Podcast 3) — log a lightweight reference
        # instead of the full gap-report blob once it's large, so audit logs
        # stay inspectable instead of ballooning on big competitor datasets.
        self.audit_log.log("phase6_done", {
            "platform": state.platform,
            "effective_gap_ref": maybe_offload("phase6_effective_gap", effective_gap),
        })
        return state

    # ------------------------------------------------------------------
    # Phase 7 — Character/Structure Audit + Auto-Fix (<= 2 retries)
    # ------------------------------------------------------------------
    def phase7_audit_and_fix(self, state: PipelineState) -> PipelineState:
        self.audit_log.log("phase7_start", {"auto_fix_enabled": state.auto_fix_enabled,
                                               "platform": state.platform})

        if not state.auto_fix_enabled:
            # Audit only, never rewrite — matches the reference UI's
            # "Auto-fix violations" toggle when switched off.
            state.audit_violations = self._structural_violations(state.final_listing, state.platform)
            stuffing_flags = check_keyword_stuffing(state.final_listing)
            if stuffing_flags:
                state.audit_violations.extend(stuffing_flags)
            state.full_audit_report = self._full_audit_report(state.final_listing, state.platform)
            state.phase_log.append("phase7_audit_and_fix")
            self.audit_log.log("phase7_done", {"remaining_violations": state.audit_violations,
                                                  "auto_fix": False})
            return state

        for attempt in range(MAX_AUTOFIX_RETRIES + 1):
            violations = self._structural_violations(state.final_listing, state.platform)
            state.audit_violations = violations
            if not violations:
                break
            if attempt == MAX_AUTOFIX_RETRIES:
                logger.warning("Max auto-fix retries reached; violations remain: %s",
                                violations)
                break
            state.final_listing = self.breaker.call(
                lambda: self.fix_agent.fix(state.final_listing, violations)
            )

        # keyword-stuffing / banned-word check reuses the existing script untouched
        stuffing_flags = check_keyword_stuffing(state.final_listing)
        if stuffing_flags:
            state.audit_violations.extend(stuffing_flags)

        # NEW: full PASS/FAIL breakdown for every section — this is what the
        # UI shows now instead of the PDF's old "4. LISTING AUDIT REPORT".
        state.full_audit_report = self._full_audit_report(state.final_listing, state.platform)

        state.phase_log.append("phase7_audit_and_fix")
        self.audit_log.log("phase7_done", {"remaining_violations": state.audit_violations})
        return state

    # ------------------------------------------------------------------
    # Platform-aware target ranges — reads spec/<platform>.yaml (via the
    # already-loaded SafetyGovernor policy) so the YAML files stay the
    # single source of truth, with sensible fallback defaults if a field
    # is missing.
    # ------------------------------------------------------------------
    def _platform_targets(self, platform: str) -> Dict[str, Any]:
        policy = self.safety._platform_policy(platform)

        if platform == "etsy":
            title = policy.get("title", {})
            tags = policy.get("tags", {})
            desc = policy.get("description", {})
            return {
                "title": (title.get("min_length", 80), title.get("max_length", 140)),
                "tags": {"count": tags.get("max_count", 13),
                         "max_len": tags.get("max_length_each", 20)},
                "description": (desc.get("min_length", 600), desc.get("max_length", 1200)),
            }

        if platform == "shopify":
            seo = policy.get("seo_title", {})
            meta = policy.get("meta_description", {})
            feat = policy.get("features", {})
            desc = policy.get("description", {})
            return {
                "seo_title": (seo.get("min_length", 40), seo.get("max_length", 70)),
                "meta_description": (meta.get("min_length", 120), meta.get("max_length", 160)),
                "features": {"min_count": feat.get("min_count", 3),
                             "max_count": feat.get("max_count", 5)},
                "description": (desc.get("min_length", 500), desc.get("max_length", 1000)),
            }

        # amazon (default)
        title = policy.get("title", {})
        bullets = policy.get("bullets", {})
        desc = policy.get("description", {})
        return {
            "title": (title.get("min_length", TITLE_RANGE[0]), title.get("max_length", TITLE_RANGE[1])),
            "bullets": {"count": bullets.get("count", N_BULLETS),
                        "range": (bullets.get("min_length", BULLET_RANGE[0]),
                                  bullets.get("max_length", BULLET_RANGE[1]))},
            "description": (desc.get("min_length", DESC_RANGE[0]), desc.get("max_length", DESC_RANGE[1])),
        }

    def _structural_violations(self, listing_text: str, platform: str = "amazon") -> List[str]:
        """Backward-compatible violations-only list (used for the auto-fix loop)."""
        return [entry["detail"] for entry in self._full_audit_report(listing_text, platform)
                if entry["status"] == "FAIL"]

    def _full_audit_report(self, listing_text: str, platform: str = "amazon") -> List[Dict[str, str]]:
        """
        Full PASS/FAIL breakdown for every section of the platform-specific
        template — this is the data the UI renders as the audit report.
        Replaces the PDF's old "4. LISTING AUDIT REPORT" section entirely
        per your request (audit belongs in the tool, not the PDF).
        """
        targets = self._platform_targets(platform)
        sections = self._split_sections(listing_text, platform)
        report: List[Dict[str, str]] = []

        def _check(name: str, length: int, lo: int, hi: int) -> None:
            status = "PASS" if lo <= length <= hi else "FAIL"
            report.append({"section": name, "status": status,
                            "detail": f"{name} length {length} (target {lo}-{hi})"})

        if platform == "etsy":
            t_lo, t_hi = targets["title"]
            _check("TITLE", count_chars(sections.get("title", "")), t_lo, t_hi)

            tags = sections.get("tags", [])
            expected_count = targets["tags"]["count"]
            max_len = targets["tags"]["max_len"]
            tag_count_status = "PASS" if len(tags) == expected_count else "FAIL"
            report.append({"section": "TAGS COUNT", "status": tag_count_status,
                            "detail": f"Expected {expected_count} tags, found {len(tags)}"})
            for i, tag in enumerate(tags):
                t_len = len(tag.strip())
                status = "PASS" if t_len <= max_len else "FAIL"
                report.append({"section": f"TAG {i+1}", "status": status,
                                "detail": f"'{tag.strip()}' ({t_len}/{max_len} chars)"})

            d_lo, d_hi = targets["description"]
            _check("DESCRIPTION", count_chars(sections.get("description", "")), d_lo, d_hi)

        elif platform == "shopify":
            st_lo, st_hi = targets["seo_title"]
            _check("SEO TITLE", count_chars(sections.get("seo_title", "")), st_lo, st_hi)

            md_lo, md_hi = targets["meta_description"]
            _check("META DESCRIPTION", count_chars(sections.get("meta_description", "")), md_lo, md_hi)

            features = sections.get("features", [])
            min_f, max_f = targets["features"]["min_count"], targets["features"]["max_count"]
            f_status = "PASS" if min_f <= len(features) <= max_f else "FAIL"
            report.append({"section": "KEY FEATURES", "status": f_status,
                            "detail": f"Found {len(features)} bullets (target {min_f}-{max_f})"})

            d_lo, d_hi = targets["description"]
            _check("PRODUCT DESCRIPTION", count_chars(sections.get("description", "")), d_lo, d_hi)

        else:  # amazon
            t_lo, t_hi = targets["title"]
            _check("TITLE", count_chars(sections.get("title", "")), t_lo, t_hi)

            bullets = sections.get("bullets", [])
            expected_count = targets["bullets"]["count"]
            b_lo, b_hi = targets["bullets"]["range"]
            count_status = "PASS" if len(bullets) == expected_count else "FAIL"
            report.append({"section": "BULLET COUNT", "status": count_status,
                            "detail": f"Expected {expected_count} bullets, found {len(bullets)}"})
            for i, b in enumerate(bullets):
                b_len = count_chars(b)
                status = "PASS" if b_lo <= b_len <= b_hi else "FAIL"
                report.append({"section": f"BULLET {i+1}", "status": status,
                                "detail": f"{b_len} chars (target {b_lo}-{b_hi})"})

            d_lo, d_hi = targets["description"]
            _check("DESCRIPTION", count_chars(sections.get("description", "")), d_lo, d_hi)

        return report

    def _split_sections(self, listing_text: str, platform: str = "amazon") -> Dict[str, Any]:
        """Parses the fixed headers for the GIVEN PLATFORM's template. Each
        platform has a different structure (Amazon: title/bullets/description;
        Etsy: title/tags/description; Shopify: seo_title/meta_description/
        features/description) so the header set and resulting dict shape
        differ by platform."""
        if platform == "etsy":
            return self._split_sections_etsy(listing_text)
        if platform == "shopify":
            return self._split_sections_shopify(listing_text)
        return self._split_sections_amazon(listing_text)

    @staticmethod
    def _split_sections_amazon(listing_text: str) -> Dict[str, Any]:
        sections: Dict[str, Any] = {"title": "", "bullets": [], "description": ""}
        current = None
        for line in listing_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("--- TITLE"):
                current = "title"
                continue
            if stripped.upper().startswith("--- BULLET"):
                current = "bullets"
                continue
            if stripped.upper().startswith("--- DESCRIPTION"):
                current = "description"
                continue
            if not stripped or current is None:
                continue
            if current == "title":
                sections["title"] += stripped
            elif current == "bullets":
                if stripped.startswith("*"):
                    sections["bullets"].append(stripped.lstrip("* ").strip())
            elif current == "description":
                sections["description"] += stripped + " "
        sections["description"] = sections["description"].strip()
        return sections

    @staticmethod
    def _split_sections_etsy(listing_text: str) -> Dict[str, Any]:
        sections: Dict[str, Any] = {"title": "", "tags": [], "description": ""}
        current = None
        for line in listing_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("--- TITLE"):
                current = "title"
                continue
            if stripped.upper().startswith("--- TAGS"):
                current = "tags"
                continue
            if stripped.upper().startswith("--- DESCRIPTION"):
                current = "description"
                continue
            if not stripped or current is None:
                continue
            if current == "title":
                sections["title"] += stripped
            elif current == "tags":
                # Tags are comma-separated on one (or more) lines.
                sections["tags"].extend(t.strip() for t in stripped.split(",") if t.strip())
            elif current == "description":
                sections["description"] += stripped + " "
        sections["description"] = sections["description"].strip()
        return sections

    @staticmethod
    def _split_sections_shopify(listing_text: str) -> Dict[str, Any]:
        sections: Dict[str, Any] = {"seo_title": "", "meta_description": "",
                                     "features": [], "description": ""}
        current = None
        for line in listing_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("--- SEO TITLE"):
                current = "seo_title"
                continue
            if stripped.upper().startswith("--- META DESCRIPTION"):
                current = "meta_description"
                continue
            if stripped.upper().startswith("--- KEY FEATURES"):
                current = "features"
                continue
            if stripped.upper().startswith("--- PRODUCT DESCRIPTION"):
                current = "description"
                continue
            if not stripped or current is None:
                continue
            if current == "seo_title":
                sections["seo_title"] += stripped
            elif current == "meta_description":
                sections["meta_description"] += stripped + " "
            elif current == "features":
                if stripped.startswith("*"):
                    sections["features"].append(stripped.lstrip("* ").strip())
            elif current == "description":
                sections["description"] += stripped + " "
        sections["meta_description"] = sections["meta_description"].strip()
        sections["description"] = sections["description"].strip()
        return sections

    # ------------------------------------------------------------------
    # Phase 8 — Competitive Advantage Summary + Final PDF
    # ------------------------------------------------------------------
    def phase8_finalize_and_pdf(self, state: PipelineState) -> PipelineState:
        self.audit_log.log("phase8_start", {})
        summary = self.breaker.call(
            lambda: self.advantage_agent.run(
                state.gap_report, state.keywords, state.final_listing
            )
        )
        state.competitive_summary = summary

        # NEW — Referee LLM / Policy Server (Podcast 5 "Semantic Gating").
        # Runs AFTER safety_governor.py's rule-based checks (Phase 6) and
        # keywords_check.py's stuffing check (Phase 7) — this is a second,
        # smarter layer judging spirit-of-the-policy compliance, not a
        # replacement for either existing check.
        try:
            platform_policy = self.safety._platform_policy(state.platform)  # existing method, read-only use
            referee_verdict = self.breaker.call(
                lambda: self.referee_agent.run(state.final_listing, state.platform, platform_policy)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("RefereeAgent failed, flagging for manual review: %s", e)
            referee_verdict = {"verdict": "REJECTED",
                                "concerns": [f"Referee check failed to run: {e}"],
                                "confidence": 0.0}
        state.referee_verdict = referee_verdict
        self.audit_log.log("phase8_referee", {"verdict": referee_verdict.get("verdict")})

        # NEW — Brutal Risk Assessment (your capstone roadmap item 5).
        # Deliberately a SEPARATE agent from advantage_agent above — one
        # sells the listing, this one argues against publishing it.
        try:
            risk_assessment = self.breaker.call(
                lambda: self.risk_agent.run(state.final_listing, state.gap_report, referee_verdict)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("RiskAssessmentAgent failed, defaulting to MEDIUM risk: %s", e)
            risk_assessment = {"risk_level": "MEDIUM",
                                "risks": [f"Risk assessment failed to run: {e}"],
                                "recommendation": "Manually review before publishing."}
        state.risk_assessment = risk_assessment
        self.audit_log.log("phase8_risk_assessment", {"risk_level": risk_assessment.get("risk_level")})

        sections = self._split_sections(state.final_listing, state.platform)

        pdf_path = self.review_module.build_listing_summary_pdf(
            brand=state.brand,
            platform=state.platform,
            sections=sections,
            gap_report=state.gap_report,
            keywords=state.keywords,
            competitive_summary=summary,
            referee_verdict=referee_verdict,      # NEW optional kwarg, defaults to None
            risk_assessment=risk_assessment,      # NEW optional kwarg, defaults to None
            # NOTE: audit/violations are intentionally NOT passed anymore —
            # per your request, the audit report is shown in the tool (UI),
            # not the PDF. See state.full_audit_report / state.audit_violations.
        )
        state.pdf_paths["listing_summary"] = pdf_path

        state.phase_log.append("phase8_finalize_and_pdf")
        self.audit_log.log("phase8_done", {"pdf": pdf_path})
        return state

    # ------------------------------------------------------------------
    # NEW — optional helper wiring in tools/sandboxed_executor.py, which
    # previously existed but was never called anywhere. Rebuilds the RAG
    # knowledge base as an ISOLATED subprocess (Podcast 1's "Harness" /
    # Podcast 4's "Infrastructure Isolation") instead of importing
    # scripts/build_kb.py's heavy embedding model directly into this
    # process. Purely additive — nothing calls this automatically; wire it
    # to a UI button ("Rebuild KB") if/when you want it.
    # ------------------------------------------------------------------
    def rebuild_knowledge_base_sandboxed(self) -> str:
        self.audit_log.log("kb_rebuild_start", {})
        try:
            output = self.sandbox.run_python("scripts/build_kb.py")
            self.audit_log.log("kb_rebuild_done", {"output_preview": output[:300]})
            return output
        except Exception as e:  # noqa: BLE001
            self.audit_log.log("kb_rebuild_failed", {"error": str(e)})
            raise

    # ------------------------------------------------------------------
    # Full chained run (Auto-Add path: every phase feeds the next automatically)
    # ------------------------------------------------------------------
    def run_full_pipeline(self, state: PipelineState) -> PipelineState:
        state = self.phase1_ingest(state)
        state = self.phase2_competitor_analysis(state)
        state = self.phase3_review_analysis(state)
        state = self.phase4_rag_retrieval(state)
        state = self.phase5_keyword_synthesis(state)
        state = self.phase6_listing_generation(state)
        state = self.phase7_audit_and_fix(state)
        state = self.phase8_finalize_and_pdf(state)
        return state

    # ------------------------------------------------------------------
    # Manual-override entry point: user skipped Auto-Add and instead uploaded
    # previous PDF/report files directly into the Listing Generator.
    # ------------------------------------------------------------------
    def run_listing_only(self, state: PipelineState) -> PipelineState:
        """
        Section 3 spec: "If the user fails to use Auto-Add, the Listing Generator
        must provide an optional Report Upload Box." This parses any uploaded
        reports back into state.gap_report / state.review_report before jumping
        straight to Phase 4-8.
        """
        if state.uploaded_report_paths:
            parsed = self.review_module.parse_uploaded_reports(state.uploaded_report_paths)
            state.gap_report = state.gap_report or parsed.get("gap_report", {})
            state.review_report = state.review_report or parsed.get("review_report", {})

        state = self.phase4_rag_retrieval(state)
        state = self.phase5_keyword_synthesis(state)
        state = self.phase6_listing_generation(state)
        state = self.phase7_audit_and_fix(state)
        state = self.phase8_finalize_and_pdf(state)
        return state