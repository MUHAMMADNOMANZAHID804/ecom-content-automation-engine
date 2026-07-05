"""
core/subagents.py
------------------
Groq-backed sub-agents for the Anti-Gravity 2.0 listing pipeline.

Design:
- ORCHESTRATOR uses a large model (openai/gpt-oss-120b) and lives in core/manager.py.
- Every SUB-AGENT below uses a small/fast Groq model (openai/gpt-oss-20b) because each
  sub-agent has ONE narrow job (ASIN parsing, gap analysis, keyword merge, audit, etc.)
  and does not need frontier reasoning. This keeps latency and token cost low.

Each sub-agent is a thin class with a single `.run(**kwargs) -> dict` method so the
manager can call them uniformly and log everything through audit-logger.py.
"""

import os
import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

# THE FIX: nothing was ever loading .env into the process environment, so
# os.getenv("GROQ_API_KEY") returned None even when the key was correctly
# pasted into the file. load_dotenv() must run before any os.getenv() call
# below. Safe to call from every entry point (ui/app.py, run_pipeline.py) —
# it's a no-op if variables are already set (e.g. exported in the shell).
load_dotenv()

logger = logging.getLogger("subagents")

# ---------------------------------------------------------------------------
# Model routing (see scripts/audit_logger.py for cost tracking of each call)
#
# CORRECTION: an earlier version of this file claimed Groq deprecated
# llama-3.1-8b-instant / llama-3.3-70b-versatile on 2026-06-17. Groq's own
# docs (console.groq.com/docs/models) confirm both are still active
# Production Models — that claim was wrong. They are intentionally NOT used
# below anyway, per your explicit preference for a more varied model mix.
#
# FIVE distinct models across THREE labs (OpenAI OSS, Meta, Alibaba Qwen),
# each picked for a specific job rather than one-size-fits-all:
#
#   MANAGER_MODEL        openai/gpt-oss-120b                — biggest, highest-
#                         quality model. Used ONLY for judgment-heavy calls:
#                         rewriting violations, the Referee policy check, the
#                         Brutal Risk Assessment, and the competitive-advantage
#                         summary. This is the "fast big main manager" role.
#
#   LISTING_DATA_MODEL   meta-llama/llama-4-scout-17b-16e-instruct — highest
#                         TPM of any candidate (300K on Developer plan), 750
#                         t/s, 131K context. Used for the two jobs that need
#                         to comfortably ingest/produce a lot of text fast:
#                         processing the Jungle Scout competitor data, and
#                         writing the actual listing.
#
#   SUBAGENT_MODEL_FAST       openai/gpt-oss-20b       — fastest model on the
#                              platform (1000 t/s). Simple structured-extraction
#                              jobs: cleaning up scraped ASIN data.
#   SUBAGENT_MODEL_REASONING  qwen/qwen3-32b            — stronger reasoning for
#                              a genuinely judgment-based job: reading raw
#                              customer reviews for sentiment and pain points.
#   SUBAGENT_MODEL_RETRIEVAL  qwen/qwen3.6-27b          — third distinct lab,
#                              used for the two remaining mechanical jobs:
#                              filtering RAG snippets and merging keyword lists.
#
# All five are open-weight models on Groq's free/on-demand tier. Override any
# of them via the matching environment variable in .env without touching code.
# ---------------------------------------------------------------------------
MANAGER_MODEL = os.getenv("MANAGER_MODEL", "openai/gpt-oss-120b")
LISTING_DATA_MODEL = os.getenv("LISTING_DATA_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
SUBAGENT_MODEL_FAST = os.getenv("SUBAGENT_MODEL_FAST", "openai/gpt-oss-20b")
SUBAGENT_MODEL_REASONING = os.getenv("SUBAGENT_MODEL_REASONING", "qwen/qwen3-32b")
SUBAGENT_MODEL_RETRIEVAL = os.getenv("SUBAGENT_MODEL_RETRIEVAL", "qwen/qwen3.6-27b")

# Backward-compatible aliases — some older code/comments referred to these
# two names. Keep them defined so nothing importing them breaks.
ORCHESTRATOR_MODEL = MANAGER_MODEL
SUBAGENT_MODEL = SUBAGENT_MODEL_FAST

_client: Optional[Groq] = None


def get_client() -> Groq:
    """Lazy singleton Groq client. Reads GROQ_API_KEY from .env / environment."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Checklist:\n"
                "  1. Does a .env file exist in your PROJECT ROOT (same folder "
                "as run_pipeline.py)?\n"
                "  2. Does it contain a line exactly like: GROQ_API_KEY=gsk_...\n"
                "  3. Are you running the app FROM that root folder "
                "(e.g. `streamlit run ui/app.py` from the project root, not "
                "from inside ui/)? load_dotenv() looks in the current working "
                "directory by default."
            )
        _client = Groq(api_key=api_key)
    return _client


def _extract_json_block(text: str) -> str:
    """
    Some models (especially newer/Preview ones) don't reliably support Groq's
    strict response_format=json_object mode, or wrap valid JSON in markdown
    fences / a sentence of commentary even when asked not to. This pulls the
    actual JSON object/array out regardless of what surrounds it, so parsing
    doesn't depend on the model behaving perfectly.
    """
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    return match.group(1) if match else text


def _chat(model: str, system: str, user: str, temperature: float = 0.3,
          max_tokens: int = 1500, json_mode: bool = False,
          retries: int = 2, reasoning_effort: Optional[str] = None) -> str:
    """
    Shared, retrying chat wrapper. Raises after `retries` failed attempts.

    reasoning_effort: gpt-oss models (openai/gpt-oss-120b, openai/gpt-oss-20b)
    are REASONING models — they spend tokens "thinking" before writing the
    visible answer, and those reasoning tokens count against max_tokens. Pass
    reasoning_effort="low" for anything where we need the full max_tokens
    budget to go to the actual output (listing copy, audit fixes), otherwise
    the model can silently exhaust its budget on reasoning and return an
    empty or truncated answer with no error raised.

    json_mode resilience: NOT every model on Groq (particularly Preview
    models like Qwen3/Llama 4 Scout) reliably supports strict
    response_format={"type": "json_object"} — some reject the request
    outright with a 400 'json_validate_failed' and an EMPTY
    failed_generation, which used to retry the identical request 3 times and
    always fail the same way. Fix: if that specific error occurs, drop
    response_format entirely on the next attempt and rely on a plain-prompt
    JSON instruction + _extract_json_block() parsing instead — this works
    with every model, strict-JSON-capable or not.
    """
    client = get_client()
    last_err = None
    use_json_mode = json_mode
    json_instruction = (
        "\n\nReturn ONLY valid JSON — no markdown code fences, no commentary, "
        "no explanation before or after the JSON."
    )

    for attempt in range(retries + 1):
        try:
            system_content = system + (json_instruction if json_mode else "")
            kwargs: Dict[str, Any] = dict(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user},
                ],
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if reasoning_effort and "gpt-oss" in model:
                kwargs["reasoning_effort"] = reasoning_effort
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            if json_mode:
                content = _extract_json_block(content)
            return content
        except Exception as e:  # noqa: BLE001
            err_text = str(e)
            last_err = e
            if use_json_mode and (
                "json_validate_failed" in err_text or "Failed to validate JSON" in err_text
            ):
                logger.warning(
                    "Model %s rejected strict JSON mode (response_format) — "
                    "falling back to prompt-based JSON for the remaining attempts.",
                    model,
                )
                use_json_mode = False
                continue  # retry immediately with the same attempt budget, no sleep needed
            logger.warning("Groq call failed (attempt %s/%s): %s",
                            attempt + 1, retries + 1, e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Groq call failed after {retries + 1} attempts: {last_err}")


def _strip_to_first_header(text: str, header: str = "--- TITLE") -> str:
    """
    Reasoning models sometimes prepend visible chain-of-thought/preamble even
    with reasoning_effort='low'. Since the listing template ALWAYS starts
    with a fixed header, anything before the first occurrence of that header
    is noise (reasoning leakage, "Sure, here's the listing:", etc.) — slice
    it off instead of feeding it into the character-count audit.
    """
    if not text:
        return text
    idx = text.find(header)
    return text[idx:].strip() if idx != -1 else text.strip()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
@dataclass
class SubAgent:
    name: str
    model: str = SUBAGENT_MODEL
    system_prompt: str = ""
    log: List[Dict[str, Any]] = field(default_factory=list)

    def _record(self, phase: str, payload: Dict[str, Any]) -> None:
        entry = {"agent": self.name, "phase": phase, "ts": time.time(), **payload}
        self.log.append(entry)
        logger.info(json.dumps(entry, default=str)[:800])


# ---------------------------------------------------------------------------
# Phase 1 (NEW) — ASIN Resolver Sub-Agent
# ---------------------------------------------------------------------------
class ASINResolverAgent(SubAgent):
    """
    If the user supplies an ASIN instead of pasting reviews/product data manually,
    this sub-agent drives tools/scrapper-mcp.py to pull the product page + reviews,
    then normalizes the raw HTML/JSON into clean structured text for the rest of
    the pipeline (Review Analyzer, Competitor Analyzer).
    """

    def __init__(self):
        super().__init__(
            name="ASINResolverAgent",
            system_prompt=(
                "You are a data-cleaning assistant. You receive raw scraped Amazon "
                "product/review data and must return STRICT JSON with keys: "
                "'title', 'bullet_points' (list), 'price', 'rating', 'review_count', "
                "'reviews' (list of raw review text strings, max 50). "
                "Do not invent data. If a field is missing, use null or an empty list."
            ),
        )

    def run(self, asin: str, scraper) -> Dict[str, Any]:
        """
        `scraper` is an injected callable/object from tools/scrapper-mcp.py or
        tools/scrapper-mcp-gateway.py — kept as a dependency injection so this
        sub-agent stays testable without live network calls.
        """
        self._record("start", {"asin": asin})
        raw = scraper.fetch_product(asin)  # -> dict with raw html/json chunks
        raw_reviews = scraper.fetch_reviews(asin, max_pages=3)

        user_payload = json.dumps({"raw_product": raw, "raw_reviews": raw_reviews})[:12000]
        content = _chat(
            self.model, self.system_prompt, user_payload,
            temperature=0.0, max_tokens=2000, json_mode=True,
        )
        try:
            structured = json.loads(content)
        except json.JSONDecodeError:
            structured = {
                "title": None, "bullet_points": [], "price": None,
                "rating": None, "review_count": None, "reviews": [],
            }
        self._record("done", {"review_count": len(structured.get("reviews", []))})
        return structured


# ---------------------------------------------------------------------------
# Phase 2 — Competitor Gap Analysis Sub-Agent
# ---------------------------------------------------------------------------
class CompetitorGapAgent(SubAgent):
    def __init__(self):
        super().__init__(
            name="CompetitorGapAgent",
            model=LISTING_DATA_MODEL,  # processes Jungle Scout data — needs high TPM/speed
            system_prompt=(
                "You are an Amazon competitive-intelligence analyst. Given our product "
                "features and the top 5 competitor titles/keywords (ranked by Average "
                "Monthly Sales), identify: (1) keywords competitors use that we don't, "
                "(2) claimed features/benefits we are missing, (3) positioning gaps we "
                "can exploit. Return STRICT JSON: {'missing_keywords': [], "
                "'missing_features': [], 'positioning_gaps': [], 'summary': ''}"
            ),
        )

    def run(self, our_features: str, competitor_titles: List[str],
            competitor_keywords: List[str], compact_context: str = None) -> Dict[str, Any]:
        self._record("start", {"n_competitors": len(competitor_titles)})

        if compact_context:
            # NEW: Data Compaction protocol (fixes 413/rate_limit errors on
            # Groq's free tier). Uses the pre-built pipe-delimited compact
            # text instead of full JSON — no repeated key names, no braces/
            # quotes overhead, dramatically fewer tokens for the same signal.
            user = compact_context
        else:
            user = json.dumps({
                "our_features": our_features,
                "competitor_titles": competitor_titles,
                "competitor_keywords": competitor_keywords,
            })

        content = _chat(self.model, self.system_prompt, user, json_mode=True)
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"missing_keywords": [], "missing_features": [],
                      "positioning_gaps": [], "summary": content[:500]}
        self._record("done", {"gaps_found": len(result.get("positioning_gaps", []))})
        return result


# ---------------------------------------------------------------------------
# Phase 3 — Review Sentiment / Pain-Point Sub-Agent
# ---------------------------------------------------------------------------
class ReviewInsightAgent(SubAgent):
    def __init__(self):
        super().__init__(
            name="ReviewInsightAgent",
            model=SUBAGENT_MODEL_REASONING,  # sentiment/pain-point judgment needs more nuance
            system_prompt=(
                "You are a customer-feedback analyst. From the raw reviews given, extract "
                "STRICT JSON: {'complaints': [{'issue': str, 'frequency': 'high|medium|low'}], "
                "'praises': [str], 'improvement_strategies': [str], "
                "'sentiment_score': float (-1 to 1)}. Be specific and concise, no fluff."
            ),
        )

    def run(self, reviews_text: str) -> Dict[str, Any]:
        self._record("start", {"chars": len(reviews_text)})
        content = _chat(self.model, self.system_prompt, reviews_text[:12000],
                         json_mode=True)
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"complaints": [], "praises": [],
                      "improvement_strategies": [], "sentiment_score": 0.0}
        self._record("done", {"complaints": len(result.get("complaints", []))})
        return result


# ---------------------------------------------------------------------------
# Phase 4 — RAG Retrieval Sub-Agent (queries the local KB built by scripts/build-kb.py)
# ---------------------------------------------------------------------------
class RAGRetrieverAgent(SubAgent):
    def __init__(self):
        super().__init__(
            name="RAGRetrieverAgent",
            model=SUBAGENT_MODEL_RETRIEVAL,
            system_prompt=(
                "You are a retrieval assistant. Given retrieved KB snippets and the "
                "current product brief, extract only the facts/keywords/policy notes "
                "relevant to writing this listing. Return STRICT JSON: "
                "{'relevant_facts': [], 'tier1_keywords': [], 'policy_notes': []}"
            ),
        )

    def run(self, kb_snippets: List[str], product_brief: str) -> Dict[str, Any]:
        self._record("start", {"n_snippets": len(kb_snippets)})
        user = json.dumps({"kb_snippets": kb_snippets, "product_brief": product_brief})
        content = _chat(self.model, self.system_prompt, user[:12000], json_mode=True)
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"relevant_facts": [], "tier1_keywords": [], "policy_notes": []}
        self._record("done", {})
        return result


# ---------------------------------------------------------------------------
# Phase 5 — 3-Source Keyword Synthesis Sub-Agent
# (merges Competitor keywords + Review-derived keywords + RAG keywords)
# ---------------------------------------------------------------------------
class KeywordSynthesisAgent(SubAgent):
    def __init__(self):
        super().__init__(
            name="KeywordSynthesisAgent",
            model=SUBAGENT_MODEL_RETRIEVAL,
            system_prompt=(
                "You merge three keyword sources (competitor, review-derived, RAG/KB) "
                "into a de-duplicated, tiered keyword set for an Amazon/Etsy/Shopify "
                "listing. Return STRICT JSON: {'tier1': [max 5], 'tier2': [max 8], "
                "'tier3': [max 8], 'bullet_openers': [exactly 5, ALL CAPS 2-3 word phrases]}"
            ),
        )

    def run(self, competitor_kws: List[str], review_kws: List[str],
            rag_kws: List[str]) -> Dict[str, Any]:
        self._record("start", {})
        user = json.dumps({
            "competitor_keywords": competitor_kws,
            "review_keywords": review_kws,
            "rag_keywords": rag_kws,
        })
        content = _chat(self.model, self.system_prompt, user, json_mode=True)
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"tier1": [], "tier2": [], "tier3": [], "bullet_openers": []}
        self._record("done", {})
        return result


# ---------------------------------------------------------------------------
# Phase 6 — Listing Generator Sub-Agent (uses the EXACT locked template)
# ---------------------------------------------------------------------------
PLATFORM_TEMPLATES = {
    # --- AMAZON (unchanged — this exact template is already tested/proven) ---
    "amazon": (
        "TARGETS: TITLE 185-200 chars. BULLETS x5 each 300-350 chars. "
        "DESC 1700-1800 chars total.\n\n"
        "PLANNING (internal - do not output):\n"
        "- Select 5 ALL-CAPS keyword openers from Bullet Openers list\n"
        "- Map each competitor pain point to exactly one bullet\n"
        "- Count characters precisely before writing each section\n\n"
        "OUTPUT ONLY with these exact headers:\n\n"
        "--- TITLE (Max 200 characters) ---\n"
        "[Product] by [Brand] / [KW1] / [KW2] / [KW3] / [KW4] / [KW5] for [Use + Feature].\n\n"
        "--- BULLET POINTS (5 Points) ---\n"
        "(ALL-CAPS keyword opener - benefit copy. 300-350 chars each. No competitor names. "
        "2 keywords per bullet.)\n"
        "* ALL-CAPS KEYWORD - [300-350 chars solving competitor pain point with product feature]\n"
        "* ALL-CAPS KEYWORD - [300-350 chars. Different pain point. 2 keywords woven in.]\n"
        "* ALL-CAPS KEYWORD - [300-350 chars. Third pain point. 2 keywords.]\n"
        "* ALL-CAPS KEYWORD - [300-350 chars. Fourth pain point. 2 keywords.]\n"
        "* ALL-CAPS KEYWORD - [300-350 chars. Trust/reliability. 2 keywords.]\n\n"
        "--- DESCRIPTION (1700 to 1800 characters) ---\n"
        "(Total 1700-1800 chars. NO brand name anywhere. Use this product.)\n\n"
        "Product Details:\n[2-3 sentences. Materials, weight, dimensions, certifications.]\n\n"
        "Product Features:\n[3-4 sentences. Key features. Tier-2 keywords.]\n\n"
        "Product Benefits:\n[4-5 sentences. LONGEST section. Features to outcomes. "
        "Counter 3 pain points. Tier-3 keyword.]\n\n"
        "Use Instructions:\n[2-3 sentences. Clear directions. One pro tip.]\n\n"
        "Care Instructions:\n[2-3 sentences. Specific temperatures, storage.]\n\n"
        "Why Choose Us?:\n[2-3 sentences. Guarantee. CTA. Tier-1 keyword.]\n\n"
        "HARD CONSTRAINTS: Title 185-200. Bullets exactly 5, each 300-350 (marker excluded). "
        "Desc 1700-1800 total.\n"
        "NO brand name in description. Return listing ONLY, no commentary.\n\n"
        "BRAND: {brand}\nFEATURES:\n{features}\n\n"
        "MASTER KEYWORDS:\n{kws}\n\nGAP REPORT:\n{gap}\n\nRAG INSIGHTS:\n{rag}"
    ),

    # --- ETSY (new) — title+tags+description, no bullet-point concept ---
    "etsy": (
        "TARGETS: TITLE 80-140 chars. TAGS exactly 13, each max 20 chars. "
        "DESCRIPTION 600-1200 chars total.\n\n"
        "PLANNING (internal - do not output):\n"
        "- Etsy buyers respond to warmth/craftsmanship, not hard-sell copy\n"
        "- Select 13 DISTINCT tags: material, use-case, style, occasion, audience\n"
        "- Count characters precisely before writing each section\n\n"
        "OUTPUT ONLY with these exact headers:\n\n"
        "--- TITLE (Max 140 characters) ---\n"
        "[Product] - [Material/Style] - [Use Case] - Handmade by [Brand]\n\n"
        "--- TAGS (13 tags, each max 20 characters) ---\n"
        "(Comma-separated. No duplicates. Each 20 chars or fewer.)\n"
        "tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10, tag11, tag12, tag13\n\n"
        "--- DESCRIPTION (600 to 1200 characters) ---\n"
        "(Warm, personal, story-driven tone. What it is, materials/craftsmanship, who "
        "it's for, care instructions. NO aggressive 'buy now' pressure.)\n\n"
        "HARD CONSTRAINTS: Title 80-140 chars. Exactly 13 tags, each <=20 chars, "
        "comma-separated, no duplicates. Description 600-1200 chars total.\n"
        "Return listing ONLY, no commentary.\n\n"
        "BRAND: {brand}\nFEATURES:\n{features}\n\n"
        "MASTER KEYWORDS:\n{kws}\n\nGAP REPORT:\n{gap}\n\nRAG INSIGHTS:\n{rag}"
    ),

    # --- SHOPIFY (new) — SEO title + meta description + features + body ---
    "shopify": (
        "TARGETS: SEO TITLE 40-70 chars. META DESCRIPTION 120-160 chars. "
        "KEY FEATURES 3-5 bullets. PRODUCT DESCRIPTION 500-1000 chars.\n\n"
        "PLANNING (internal - do not output):\n"
        "- SEO title front-loads the primary keyword (Google truncates ~60 chars)\n"
        "- Meta description needs a clear call-to-action within 160 chars\n"
        "- Count characters precisely before writing each section\n\n"
        "OUTPUT ONLY with these exact headers:\n\n"
        "--- SEO TITLE (Max 70 characters) ---\n"
        "[Primary Keyword] - [Product] by [Brand]\n\n"
        "--- META DESCRIPTION (120 to 160 characters) ---\n"
        "[Compelling summary, primary keyword + CTA, 120-160 chars]\n\n"
        "--- KEY FEATURES (3-5 bullets) ---\n"
        "* [Feature 1 - benefit-focused]\n"
        "* [Feature 2 - benefit-focused]\n"
        "* [Feature 3 - benefit-focused]\n\n"
        "--- PRODUCT DESCRIPTION (500 to 1000 characters) ---\n"
        "[Persuasive body copy: what it solves, key benefits, trust signals]\n\n"
        "HARD CONSTRAINTS: SEO Title 40-70 chars. Meta Description 120-160 chars. "
        "3 to 5 Key Features bullets. Product Description 500-1000 chars.\n"
        "Return listing ONLY, no commentary.\n\n"
        "BRAND: {brand}\nFEATURES:\n{features}\n\n"
        "MASTER KEYWORDS:\n{kws}\n\nGAP REPORT:\n{gap}\n\nRAG INSIGHTS:\n{rag}"
    ),
}

# First header of each platform's template — used to strip reasoning-leakage
# preamble and to validate the response actually contains real content.
PLATFORM_FIRST_HEADER = {
    "amazon": "--- TITLE",
    "etsy": "--- TITLE",
    "shopify": "--- SEO TITLE",
}
# A second marker that MUST also be present for a response to count as
# complete (catches a truncated response that only got the title written).
PLATFORM_REQUIRED_MARKER = {
    "amazon": "--- BULLET",
    "etsy": "--- TAGS",
    "shopify": "--- META DESCRIPTION",
}


class ListingGeneratorAgent(SubAgent):
    """
    Runs on LISTING_DATA_MODEL (meta-llama/llama-4-scout-17b-16e-instruct) —
    highest TPM of any candidate model and 750 t/s, matching your requirement
    that "the subagent writing the listing... have high tokens and speed."
    Previously ran on the MANAGER_MODEL; moved deliberately per your request
    for a more varied model mix across the pipeline.

    SINGLE-SHOT generation (no more 2-drafts-then-merge): per your request to
    minimize token consumption, this now makes ONE call that produces the
    final listing directly, instead of 2 draft calls + a 3rd merge call.
    """

    def __init__(self):
        super().__init__(
            name="ListingGeneratorAgent",
            model=LISTING_DATA_MODEL,
            system_prompt=(
                "You are a senior e-commerce listing copywriter working across "
                "Amazon, Etsy, and Shopify. Follow the user's structural template "
                "for the specified platform with zero deviation. Obey every "
                "character constraint exactly. Do not add commentary, markdown, "
                "or explanations outside the requested headers."
            ),
        )

    def _fill_template(self, platform: str, brand: str, features: str,
                        kws: Dict[str, Any], gap: Dict[str, Any],
                        rag: Dict[str, Any]) -> str:
        template = PLATFORM_TEMPLATES.get(platform, PLATFORM_TEMPLATES["amazon"])
        return template.format(
            brand=brand,
            features=features,
            kws=json.dumps(kws),
            gap=json.dumps(gap),
            rag=json.dumps(rag),
        )

    # Reasoning models (gpt-oss) spend part of max_tokens on invisible
    # "thinking" — a title(~50) + 5 bullets(~350 each, ~1750) + description
    # (~1800 chars, ~450 tokens) needs ~2300 tokens of VISIBLE output alone.
    # 4000 leaves real headroom even with reasoning overhead.
    LISTING_MAX_TOKENS = 4000

    def generate_listing(self, brand: str, features: str, kws: Dict[str, Any],
                          gap: Dict[str, Any], rag: Dict[str, Any],
                          platform: str = "amazon", temperature: float = 0.55) -> str:
        """
        Single call, final listing directly — replaces the old
        generate_drafts() + generate_final() two-step (three total LLM calls)
        with one. Cuts listing-generation token/API usage roughly in half to
        a third for this phase.
        """
        self._record("generate_start", {"platform": platform, "temperature": temperature})
        prompt = self._fill_template(platform, brand, features, kws, gap, rag)
        content = self._generate_with_safety_net(prompt, self.system_prompt,
                                                    temperature=temperature, platform=platform)
        self._record("generate_done", {"platform": platform})
        return content

    def _generate_with_safety_net(self, prompt: str, system: str,
                                   temperature: float, platform: str = "amazon") -> str:
        """
        Calls the model, strips any reasoning-leakage preamble, and retries
        ONCE with an emphasized instruction if the result comes back empty
        or missing the required headers for THIS platform's template — this
        is what was silently producing empty listings before (reasoning
        model burned its token budget with no visible output and no
        exception was ever raised).
        """
        first_header = PLATFORM_FIRST_HEADER.get(platform, "--- TITLE")
        required_marker = PLATFORM_REQUIRED_MARKER.get(platform, "--- BULLET")

        content = _chat(self.model, system, prompt, temperature=temperature,
                         max_tokens=self.LISTING_MAX_TOKENS, reasoning_effort="low")
        content = _strip_to_first_header(content, header=first_header)

        if not content or first_header not in content or required_marker not in content:
            logger.warning(
                "ListingGeneratorAgent got an empty/incomplete response for "
                "platform=%s — retrying once with a stronger 'output only' instruction.",
                platform,
            )
            emphasized_system = system + (
                f" CRITICAL: Output ONLY the required headers and their content. "
                f"Do not think out loud, do not explain, do not add any text "
                f"before '{first_header}'. Your entire response must start with "
                f"'{first_header}'."
            )
            content = _chat(self.model, emphasized_system, prompt,
                             temperature=temperature,
                             max_tokens=self.LISTING_MAX_TOKENS,
                             reasoning_effort="low")
            content = _strip_to_first_header(content, header=first_header)

        return content.strip()


# ---------------------------------------------------------------------------
# Phase 7 — Audit + Auto-Fix Sub-Agent
# ---------------------------------------------------------------------------
class AuditFixAgent(SubAgent):
    def __init__(self):
        super().__init__(
            name="AuditFixAgent",
            model=MANAGER_MODEL,  # fixing must preserve quality, use the big model
            system_prompt=(
                "You are a strict compliance/QA editor. You will be given a listing "
                "and a list of structural violations (character counts, missing "
                "headers, banned words). Rewrite ONLY the violating sections to fix "
                "them while preserving everything else verbatim. Return the FULL "
                "corrected listing, same headers, no commentary."
            ),
        )

    def fix(self, listing_text: str, violations: List[str]) -> str:
        self._record("fix_start", {"violations": violations})
        user = f"VIOLATIONS TO FIX:\n{json.dumps(violations)}\n\nLISTING:\n{listing_text}"
        fixed = _chat(self.model, self.system_prompt, user, temperature=0.2,
                       max_tokens=4000, reasoning_effort="low")
        fixed = _strip_to_first_header(fixed)
        if not fixed or "--- TITLE" not in fixed:
            logger.warning("AuditFixAgent returned empty/incomplete output — "
                            "keeping the pre-fix listing instead of discarding it.")
            return listing_text
        self._record("fix_done", {})
        return fixed.strip()


# ---------------------------------------------------------------------------
# Phase 8 — Competitive Advantage Summary Sub-Agent (feeds the final PDF)
# ---------------------------------------------------------------------------
class CompetitiveAdvantageAgent(SubAgent):
    def __init__(self):
        super().__init__(
            name="CompetitiveAdvantageAgent",
            model=MANAGER_MODEL,
            system_prompt=(
                "You write a short 'Why This Listing Wins' summary (120-180 words) "
                "for an internal PDF report, referencing the gap report and keyword "
                "strategy used. Plain text, no markdown headers."
            ),
        )

    def run(self, gap: Dict[str, Any], kws: Dict[str, Any], final_listing: str) -> str:
        self._record("start", {})
        user = json.dumps({"gap": gap, "keywords": kws,
                            "final_listing_preview": final_listing[:600]})
        summary = _chat(self.model, self.system_prompt, user, temperature=0.5, max_tokens=350)
        self._record("done", {})
        return summary.strip()


# ---------------------------------------------------------------------------
# NEW — Referee LLM / Policy Server (Podcast 5, "Semantic Gating")
# ---------------------------------------------------------------------------
# Podcast 5 explicitly names this as missing from earlier builds: "a Referee
# LLM that checks if the agent's action matches your guidelines" — distinct
# from core/safety_governor.py, which only does RULE-based checks (banned
# words, length limits). This agent does SEMANTIC judgment: does the listing
# actually comply with the platform's policy in spirit, not just in literal
# banned-word matching? It runs AFTER safety_governor.py, never instead of it
# — both stay active, this just adds a second, smarter layer.
class RefereeAgent(SubAgent):
    def __init__(self):
        super().__init__(
            name="RefereeAgent",
            model=MANAGER_MODEL,
            system_prompt=(
                "You are a strict compliance referee for e-commerce listings. "
                "You will be given a generated listing, the target platform, and "
                "that platform's policy rules. Judge whether the listing complies "
                "IN SPIRIT, not just literally — e.g. implied medical claims, "
                "misleading superlatives, or borderline exaggeration that a rule-"
                "based banned-word filter would miss. Return STRICT JSON: "
                "{'verdict': 'APPROVED' or 'REJECTED', 'concerns': [list of "
                "specific sentence-level concerns, empty if none], "
                "'confidence': float 0-1}."
            ),
        )

    def run(self, listing_text: str, platform: str, policy: Dict[str, Any]) -> Dict[str, Any]:
        self._record("start", {"platform": platform})
        user = json.dumps({
            "platform": platform,
            "policy_rules": policy,
            "listing": listing_text[:4000],
        })
        content = _chat(self.model, self.system_prompt, user, temperature=0.2,
                         max_tokens=600, json_mode=True)
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Fail safe: if the referee itself breaks, don't silently pass —
            # flag it for human review rather than pretend it was approved.
            result = {"verdict": "REJECTED", "concerns": [
                "Referee agent returned invalid output — flagged for manual review."
            ], "confidence": 0.0}
        self._record("done", {"verdict": result.get("verdict")})
        return result


# ---------------------------------------------------------------------------
# NEW — Brutal Risk Assessment (your capstone roadmap, Podcast 5 item 5)
# ---------------------------------------------------------------------------
# Distinct from CompetitiveAdvantageAgent (which sells the listing). This
# agent's ONLY job is to argue AGAINST publishing — the reviewer needs the
# critical case, not another positive summary, before they click approve.
class RiskAssessmentAgent(SubAgent):
    def __init__(self):
        super().__init__(
            name="RiskAssessmentAgent",
            model=MANAGER_MODEL,
            system_prompt=(
                "You are a skeptical, critical reviewer whose ONLY job is to find "
                "reasons NOT to publish this listing as-is. Be brutally honest, "
                "not diplomatic. Consider: unverifiable claims, policy risk, weak "
                "or generic bullets, keyword stuffing, missing differentiation "
                "from competitors, tone mismatches. Return STRICT JSON: "
                "{'risk_level': 'LOW' or 'MEDIUM' or 'HIGH', "
                "'risks': [specific, concrete risks — empty list only if you "
                "genuinely find none], 'recommendation': one sentence, direct}."
            ),
        )

    def run(self, listing_text: str, gap: Dict[str, Any],
            referee_verdict: Dict[str, Any]) -> Dict[str, Any]:
        self._record("start", {})
        user = json.dumps({
            "listing": listing_text[:4000],
            "gap_report_summary": gap.get("summary", ""),
            "referee_verdict": referee_verdict,
        })
        content = _chat(self.model, self.system_prompt, user, temperature=0.3,
                         max_tokens=600, json_mode=True)
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"risk_level": "MEDIUM", "risks": [
                "Risk-assessment agent returned invalid output — treat as unverified."
            ], "recommendation": "Manually review before publishing."}
        self._record("done", {"risk_level": result.get("risk_level")})
        return result