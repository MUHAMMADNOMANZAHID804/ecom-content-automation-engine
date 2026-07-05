"""
================================================================================
MANAGER BACKUP -- core/manager_backup.py
Deterministic Fallback Orchestrator (Circuit-Breaker Companion)
Anti Gravity Listing Engine v5.1
================================================================================

WHY THIS FILE EXISTS:
  manager.py's OrchestratorBrain calls the Groq API to interpret governor
  SOFT_FAIL verdicts (PROCEED / SOFT_FIX / HARD_STOP) and to grant
  permissions. If the Groq API is down, rate-limited, or the API key is
  invalid, that LLM call can throw -- and a thrown exception mid-pipeline
  is worse than a slightly-less-flexible deterministic fallback.

  This file is the "circuit breaker" target referenced in
  scripts/circuit_breaker.py: when N consecutive Groq API failures are
  detected for orchestrator-level decisions, scripts/circuit_breaker.py
  flips traffic from core.manager.LeadOrchestrator to
  core.manager_backup.LeadOrchestratorBackup for the rest of the session.

  It implements the IDENTICAL 10-phase pipeline and the IDENTICAL
  permission/governor contract as manager.py -- it just replaces the
  *brain* (LLM-based verdict interpretation) with deterministic
  rule-based logic:

    SOFT_FAIL  -> always PROCEED (governor already chose SOFT not HARD)
    HARD_FAIL  -> always ABORT (no LLM override possible)

  This matches Podcast 5's "Structural Gating" principle: binary
  Yes/No rules instead of a Referee LLM, used specifically as a
  degraded-mode fallback, not as the primary path.

Podcast Principles:
  - Circuit Breaker pattern: deterministic fallback when LLM unavailable
  - Structural Gating over Semantic Gating when LLM is the failure point
  - Same File Message Bus contract as manager.py (Podcast 3)
  - Same JIT Downscoping permission lifecycle (Podcast 4)
  - No YOLO: user checkpoint at Phase 7 is preserved exactly (Podcast 5)
"""

import os
import json
import uuid
import yaml
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="[%(asctime)s UTC] %(levelname)s  ORCHESTRATOR-BACKUP ▶ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "orchestrator_backup.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("manager_backup")

FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "llama-3.3-70b-versatile")
FAST_MODEL     = os.getenv("FAST_MODEL",     "llama3-8b-8192")

from core.safety_governor import SafetyGovernor
from core.sub_agents import ResearchAgent, SEOAgent, WriterAgent
from tools.scraper_mcp import ScraperMCP
from tools.file_mcp import FileMCP
from tools.secure_mcp_gateway import SecureMCPGateway
from modules.competitor_analyzer import run_competitor_analysis
from modules.review_analyzer import run_review_analysis


# ════════════════════════════════════════════════════════════════════
# DETERMINISTIC VERDICT INTERPRETER (replaces OrchestratorBrain LLM)
# ════════════════════════════════════════════════════════════════════

class DeterministicBrain:
    """
    Rule-based replacement for OrchestratorBrain.
    No Groq API calls. Pure structural gating (Podcast 5).

    SOFT_FAIL  -> PROCEED   (the governor already decided this is non-fatal)
    HARD_FAIL  -> the caller already routes this to ABORT before reaching
                  here, so this class only ever needs to resolve SOFT_FAIL
    """

    def decide_on_verdict(self, phase: int, verdict: dict, context: dict) -> dict:
        violations = verdict.get("violations", [])
        soft_count = sum(1 for v in violations if v.get("severity") == "SOFT_FAIL")

        # Conservative rule: more than 5 SOFT_FAILs in one phase = escalate
        # to HARD_STOP even though governor said SOFT_FAIL. This is the
        # deterministic equivalent of "use your judgment" -- a fixed
        # threshold instead of an LLM call.
        if soft_count > 5:
            return {
                "decision": "HARD_STOP",
                "reason": f"{soft_count} SOFT_FAIL violations exceeds backup-mode threshold (5)",
                "user_message": (
                    f"Phase {phase} produced {soft_count} quality issues. "
                    f"Running in degraded (no-LLM-brain) mode, so this is "
                    f"escalated to a hard stop for safety. Review violations "
                    f"and re-run."
                ),
                "fix_instructions": "",
            }

        return {
            "decision": "PROCEED",
            "reason": f"{soft_count} SOFT_FAIL violation(s) -- within backup-mode threshold",
            "user_message": "",
            "fix_instructions": "",
        }

    def grant_permission(self, agent: str, phase: int,
                          ptype: str, resource: str, purpose: str) -> dict:
        """Always grants -- the real access control is PermissionController,
        not this decision layer. Matches manager.py's permissive default."""
        return {
            "status"    : "GRANTED",
            "agent"     : agent,
            "resource"  : resource,
            "scope"     : f"phase {phase} only",
            "conditions": "deterministic backup mode -- standard scope rules apply",
            "reason"    : "Auto-granted under circuit-breaker fallback",
        }

    def generate_phase_summary(self, phase: int, output_data: dict, platform: str) -> str:
        return f"Phase {phase} completed (backup mode -- no LLM summary generated)."


# ════════════════════════════════════════════════════════════════════
# PERMISSION CONTROLLER -- IDENTICAL to manager.py (JIT Downscoping)
# ════════════════════════════════════════════════════════════════════

class PermissionController:
    def __init__(self):
        self._grants = []

    def grant(self, agent, phase, ptype, resource, conditions=None):
        g = {
            "token_id"  : str(uuid.uuid4())[:8],
            "agent"     : agent,
            "phase"     : phase,
            "type"      : ptype,
            "resource"  : resource,
            "conditions": conditions or {},
            "revoked"   : False,
            "granted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._grants.append(g)
        log.info(f"PERMISSION_GRANT  token={g['token_id']} | {agent} | {ptype} | phase={phase}")
        return g

    def revoke_phase(self, phase):
        for g in self._grants:
            if g["phase"] == phase and not g["revoked"]:
                g["revoked"] = True
                log.info(f"PERMISSION_REVOKED  token={g['token_id']} | phase={phase}")

    def is_granted(self, agent, phase, ptype):
        return any(
            g["agent"] == agent and g["phase"] == phase
            and g["type"] == ptype and not g["revoked"]
            for g in self._grants
        )

    def active_summary(self):
        return [g for g in self._grants if not g["revoked"]]


# ════════════════════════════════════════════════════════════════════
# SESSION LOG -- IDENTICAL to manager.py
# ════════════════════════════════════════════════════════════════════

class SessionLog:
    def __init__(self, session_id):
        self.session_id   = session_id
        self.started_at   = datetime.now(timezone.utc).isoformat()
        self.platform     = None
        self.product      = None
        self.policy_file  = None
        self.phases       = {}
        self.governor_flags = []
        self.user_decisions = []
        self.hard_fails     = []
        self.scrape_tiers   = {}
        self.mode            = "BACKUP_DETERMINISTIC"
        self.ai_model_used   = {
            "orchestrator"  : "deterministic_rules (no LLM -- circuit breaker active)",
            "research_agent": FRONTIER_MODEL,
            "seo_agent"     : FRONTIER_MODEL,
            "writer_agent"  : FAST_MODEL,
        }

    def record_phase(self, phase, status, note):
        self.phases[f"phase_{phase}"] = {"status": status, "note": note,
                                          "at": datetime.now(timezone.utc).isoformat()}
        log.info(f"PHASE {phase} ▶ {status} — {note}")

    def record_governor(self, phase, verdict):
        self.governor_flags.append({"phase": phase, "verdict": verdict})

    def record_user_decision(self, phase, decision, notes=""):
        self.user_decisions.append({"phase": phase, "decision": decision, "notes": notes})

    def record_hard_fail(self, phase, reason):
        self.hard_fails.append({"phase": phase, "reason": str(reason),
                                 "at": datetime.now(timezone.utc).isoformat()})
        log.error(f"HARD FAIL  phase={phase}  reason={reason}")

    def save(self, output_dir: Path):
        path = output_dir / "session_log.json"
        path.write_text(json.dumps(self.__dict__, indent=2, default=str), encoding="utf-8")
        log.info(f"Session log saved (backup mode) → {path}")

    def to_dict(self):
        return self.__dict__


# ════════════════════════════════════════════════════════════════════
# LEAD ORCHESTRATOR BACKUP -- 10-phase pipeline, deterministic brain
# ════════════════════════════════════════════════════════════════════

class LeadOrchestratorBackup:
    """
    Drop-in replacement for core.manager.LeadOrchestrator when the
    Groq API is unavailable for orchestrator-level decisions.

    Used by scripts/circuit_breaker.py:
        from core.manager_backup import LeadOrchestratorBackup as LeadOrchestrator
        # ... same .run(user_input) call signature as the primary class

    Same 10-phase structure, same file paths, same governor contract,
    same user checkpoint at Phase 7. ONLY difference: SOFT_FAIL handling
    uses DeterministicBrain instead of an LLM call.
    """

    SUPPORTED_PLATFORMS = ["amazon", "shopify", "etsy"]

    def __init__(self):
        self.brain       = DeterministicBrain()
        self.governor    = SafetyGovernor()
        self.permissions = PermissionController()
        self.file_mcp    = FileMCP(PROJECT_ROOT)
        self.gateway     = SecureMCPGateway()

        self.research_agent = ResearchAgent()
        self.seo_agent      = SEOAgent()
        self.writer_agent   = WriterAgent()

        self.session    = None
        self.output_dir = None

        log.warning("="*60)
        log.warning("  CIRCUIT BREAKER ACTIVE — Running in BACKUP mode")
        log.warning("  Orchestrator decisions use deterministic rules,")
        log.warning("  NOT the Groq-backed OrchestratorBrain.")
        log.warning("="*60)

    # ── PUBLIC ENTRY POINT (identical signature to manager.py) ───────

    def run(self, user_input: dict) -> dict:
        session_id      = str(uuid.uuid4())[:8].upper()
        self.session    = SessionLog(session_id)
        self.output_dir = PROJECT_ROOT / "output" / session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"{'═'*60}")
        log.info(f"  SESSION START (BACKUP MODE)  ID={session_id}")
        log.info(f"{'═'*60}")

        try:
            v = self._validate_input(user_input)
            if not v["ok"]:
                return self._abort(None, v["error"])

            platform = user_input["platform"].lower()
            self.session.platform = platform
            self.session.product  = user_input.get("product_name", "")

            policy_result = self._load_policy(platform)
            if not policy_result["ok"]:
                return self._abort(None,
                    f"❌ Cannot generate listing: policy file missing for '{platform}'.\n"
                    f"Expected: {policy_result['expected']}")

            policy_pack = policy_result["policy"]
            self.session.policy_file = policy_result["path"]

            kb_check = self._check_kb(platform)
            if not kb_check["ok"]:
                return self._abort(None,
                    f"❌ Cannot generate listing: knowledge base not ready for '{platform}'.\n"
                    f"{kb_check['error']}")

            return self._pipeline(user_input, platform, policy_pack)

        except KeyboardInterrupt:
            return {"status": "INTERRUPTED", "session_id": session_id}
        except Exception as e:
            log.exception(f"Unexpected backup-mode error: {e}")
            return self._abort(None, f"Unexpected error (backup mode): {e}")

    # ── 10-PHASE PIPELINE (structurally identical to manager.py) ─────

    def _pipeline(self, user_input: dict, platform: str, policy_pack: dict) -> dict:
        context = {
            "platform"    : platform,
            "product_name": user_input.get("product_name"),
            "session_id"  : self.session.session_id,
        }

        # PHASE 0 — Competitor Intelligence
        self._banner(0, "Competitor Intelligence", "Data Collector")
        asins        = user_input.get("competitor_asins", [])
        manual_comps = user_input.get("manual_competitors", [])

        if manual_comps and not asins:
            competitor_data = {
                "products"  : manual_comps,
                "asins"     : [c.get("asin","") for c in manual_comps],
                "tiers_used": {c.get("asin",""): "tier3_manual" for c in manual_comps},
            }
        elif asins:
            self.permissions.grant("data_collector", 0, "TOOL", "scraper_mcp",
                {"asins": asins, "max_retries": 3, "min_delay": 3})
            scraper = ScraperMCP(self.gateway)
            scrape_result = scraper.collect(
                asins=asins,
                marketplace=user_input.get("target_market", "amazon.com"),
                data_mode=user_input.get("data_mode", "auto"),
            )
            competitor_data = {
                "products"  : scrape_result.get("products", []),
                "asins"     : scrape_result.get("asins", asins),
                "tiers_used": scrape_result.get("tiers_used", {}),
            }
            self.session.scrape_tiers.update(scrape_result.get("tiers_used", {}))
        else:
            competitor_data = {"products": [], "asins": [], "source": "none"}

        comp_path = self.output_dir / "competitor_data.json"
        comp_path.write_text(json.dumps(competitor_data, indent=2), encoding="utf-8")
        if self._gate(0, competitor_data, platform, policy_pack, context) == "ABORT":
            return self._abort(0, "Governor HARD_FAIL at Phase 0")
        self.permissions.revoke_phase(0)
        self.session.record_phase(0, "PASS", f"{len(asins)} ASINs")

        # PHASE 1 — Competitor Gap Analysis
        self._banner(1, "Competitor Gap Analysis", "Research Agent")
        self.permissions.grant("research_agent", 1, "READ", str(comp_path), {})
        gap_report = self.research_agent.run_phase1(
            competitor_data_path=str(comp_path),
            product_details=user_input.get("product_details", ""),
            category=user_input.get("category", ""),
            platform=platform,
        )
        gap_path = self.output_dir / "gap_report.json"
        gap_path.write_text(json.dumps(gap_report, indent=2), encoding="utf-8")
        if self._gate(1, gap_report, platform, policy_pack, context) == "ABORT":
            return self._abort(1, "Governor HARD_FAIL at Phase 1")
        self.permissions.revoke_phase(1)
        self.session.record_phase(1, "PASS", f"{len(gap_report.get('ranked_gaps',[]))} gaps")

        # PHASE 2 — Top-5 Competitor Analyzer Module (Amazon only)
        self._banner(2, "Top-5 Competitor Analyzer", "Competitor Analyzer Module")
        competitor_analysis = {"status": "SKIPPED", "reason": "Module only runs for Amazon"}
        if platform == "amazon":
            self.permissions.grant("competitor_analyzer", 2, "READ", str(comp_path), {})
            try:
                competitor_analysis = run_competitor_analysis(
                    jungle_scout_bytes  = user_input.get("jungle_scout_bytes"),
                    jungle_scout_name   = user_input.get("jungle_scout_name", ""),
                    manual_competitors  = competitor_data.get("products", []),
                    product_details     = user_input.get("product_details", ""),
                    category            = user_input.get("category", ""),
                    session_id          = self.session.session_id,
                )
            except Exception as e:
                log.warning(f"Competitor Analyzer error (non-fatal): {e}")
                competitor_analysis = {"status": "ERROR", "error": str(e)}
        comp_analysis_path = self.output_dir / "competitor_analysis.json"
        comp_analysis_path.write_text(json.dumps(competitor_analysis, indent=2, default=str), encoding="utf-8")
        if self._gate(2, competitor_analysis, platform, policy_pack, context) == "ABORT":
            return self._abort(2, "Governor HARD_FAIL at Phase 2")
        self.permissions.revoke_phase(2)
        self.session.record_phase(2, "PASS", f"status={competitor_analysis.get('status')}")

        # PHASE 3 — Review Sentiment Analyzer Module (all platforms)
        self._banner(3, "Review Sentiment Analysis", "Review Analyzer Module")
        review_analysis = {"status": "SKIPPED", "reason": "No reviews provided"}
        manual_reviews = user_input.get("manual_reviews_by_competitor", [])
        if manual_reviews:
            self.permissions.grant("review_analyzer", 3, "READ", "manual_reviews", {})
            try:
                review_analysis = run_review_analysis(
                    reviews_by_competitor = manual_reviews,
                    platform              = platform,
                    product_category      = user_input.get("category", ""),
                    session_id            = self.session.session_id,
                )
            except Exception as e:
                log.warning(f"Review Analyzer error (non-fatal): {e}")
                review_analysis = {"status": "ERROR", "error": str(e)}
        review_analysis_path = self.output_dir / "review_analysis.json"
        review_analysis_path.write_text(json.dumps(review_analysis, indent=2, default=str), encoding="utf-8")
        if self._gate(3, review_analysis, platform, policy_pack, context) == "ABORT":
            return self._abort(3, "Governor HARD_FAIL at Phase 3")
        self.permissions.revoke_phase(3)
        self.session.record_phase(3, "PASS", f"status={review_analysis.get('status')}")

        # Merge module outputs into gap_report.json (same as primary manager.py)
        if competitor_analysis.get("status") == "SUCCESS":
            gap_report["competitor_analysis"] = {
                "title_keyword_gaps"   : competitor_analysis.get("gap_analysis",{}).get("title_keyword_gaps",[]),
                "competitor_weaknesses": competitor_analysis.get("gap_analysis",{}).get("competitor_weaknesses",[]),
                "usp_suggestion"       : competitor_analysis.get("gap_analysis",{}).get("usp_suggestion",""),
            }
        if review_analysis.get("status") == "SUCCESS":
            gap_report["review_sentiment"] = {
                "top_complaints": review_analysis.get("analysis",{}).get("top_complaints",[]),
                "buyer_language": review_analysis.get("analysis",{}).get("buyer_language",{}),
                "critical_gaps" : review_analysis.get("analysis",{}).get("critical_gaps",[]),
            }
        gap_path.write_text(json.dumps(gap_report, indent=2), encoding="utf-8")

        # PHASE 4 — RAG Knowledge Base Retrieval
        self._banner(4, "RAG Knowledge Base Retrieval", "Research Agent")
        self.permissions.grant("research_agent", 4, "READ", f"knowledge_base/{platform}/", {})
        knowledge_pack = self.research_agent.run_phase2(
            gap_report_path=str(gap_path), platform=platform,
            category=user_input.get("category", ""),
            kb_root=str(PROJECT_ROOT / "knowledge_base"), policy_pack=policy_pack,
        )
        kb_path = self.output_dir / "knowledge_pack.json"
        kb_path.write_text(json.dumps(knowledge_pack, indent=2), encoding="utf-8")
        if self._gate(4, knowledge_pack, platform, policy_pack, context) == "ABORT":
            return self._abort(4, "Governor HARD_FAIL at Phase 4")
        self.permissions.revoke_phase(4)
        self.session.record_phase(4, "PASS", f"{len(knowledge_pack.get('rules',[]))} rules")

        # PHASE 5 — 3-Source Keyword Synthesis
        self._banner(5, "3-Source Keyword Synthesis", "SEO Agent")
        self.permissions.grant("seo_agent", 5, "READ", "gap_report + knowledge_pack", {})
        keyword_strategy = self.seo_agent.run_phase3(
            gap_report_path=str(gap_path), knowledge_pack_path=str(kb_path), platform=platform,
        )
        kw_path = self.output_dir / "keyword_strategy.json"
        kw_path.write_text(json.dumps(keyword_strategy, indent=2), encoding="utf-8")
        if self._gate(5, keyword_strategy, platform, policy_pack, context) == "ABORT":
            return self._abort(5, "Governor HARD_FAIL at Phase 5")
        self.permissions.revoke_phase(5)
        self.session.record_phase(5, "PASS", f"{len(keyword_strategy.get('primary',[]))} keywords")

        # PHASE 6 — Initial Listing Draft
        self._banner(6, "Initial Listing Draft", "SEO Agent")
        self.permissions.grant("seo_agent", 6, "WRITE", "draft_listing.json", {})
        draft_listing = self.seo_agent.run_phase4(
            keyword_strategy_path=str(kw_path), knowledge_pack_path=str(kb_path),
            product_name=user_input.get("product_name", ""),
            product_details=user_input.get("product_details", ""),
            special_notes=user_input.get("special_notes", ""), platform=platform,
        )
        draft_path = self.output_dir / "draft_listing.json"
        draft_path.write_text(json.dumps(draft_listing, indent=2), encoding="utf-8")
        if self._gate(6, draft_listing, platform, policy_pack, context) == "ABORT":
            return self._abort(6, "Governor HARD_FAIL at Phase 6")
        self.permissions.revoke_phase(6)
        self.session.record_phase(6, "PASS", "Draft created")

        # PHASE 7 — Audit + User Checkpoint
        self._banner(7, "Character & Structure Audit", "Writer Agent")
        self.permissions.grant("writer_agent", 7, "READ", "draft + knowledge_pack", {})
        audit_report = self.writer_agent.run_phase5(
            draft_listing_path=str(draft_path), knowledge_pack_path=str(kb_path),
            platform=platform, policy_pack=policy_pack,
        )
        audit_path = self.output_dir / "audit_report.json"
        audit_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
        if self._gate(7, audit_report, platform, policy_pack, context) == "ABORT":
            return self._abort(7, "Governor HARD_FAIL at Phase 7")
        self.permissions.revoke_phase(7)
        self.session.record_phase(7, "PASS", f"Quality: {audit_report.get('quality_index',0)}/100")

        user_dec = self._user_checkpoint(draft_listing, audit_report)
        self.session.record_user_decision(7, user_dec["decision"], user_dec.get("notes",""))

        if user_dec["decision"] == "REJECTED":
            fix_result = self._phase8(draft_path, audit_path, kb_path,
                                      platform, policy_pack, user_dec.get("notes", ""))
            if fix_result["status"] == "ESCALATE":
                return self._abort(8, fix_result["reason"])
            draft_path = fix_result["fixed_path"]

        # PHASE 9 — Final Output
        self._banner(9, "Final Output & Competitive Advantage", "Writer Agent")
        self.permissions.grant("writer_agent", 9, "WRITE", "final outputs", {})
        final_output = self.writer_agent.run_phase7(
            approved_listing_path=str(draft_path), gap_report_path=str(gap_path),
            audit_report_path=str(audit_path), keyword_strategy_path=str(kw_path),
            session_log=self.session.to_dict(), platform=platform,
            output_dir=str(self.output_dir),
        )
        if self._gate(9, final_output, platform, policy_pack, context) == "ABORT":
            return self._abort(9, "Governor HARD_FAIL at Phase 9")
        self.permissions.revoke_phase(9)
        self.session.record_phase(9, "PASS", f"Final: {final_output.get('quality_index',0)}/100")
        self.session.save(self.output_dir)

        log.info(f"PIPELINE COMPLETE (BACKUP MODE)  session={self.session.session_id}")

        return {
            "status"     : "SUCCESS",
            "session_id" : self.session.session_id,
            "output_dir" : str(self.output_dir),
            "final_score": final_output.get("quality_index", 0),
            "platform"   : platform,
            "mode"       : "BACKUP_DETERMINISTIC",
            "files": {
                "listing"            : str(self.output_dir / "final_listing.txt"),
                "pdf"                : str(self.output_dir / "listing_package.pdf"),
                "report"             : str(self.output_dir / "competitive_report.json"),
                "session"            : str(self.output_dir / "session_log.json"),
                "competitor_analysis": str(comp_analysis_path),
                "review_analysis"    : str(review_analysis_path),
            }
        }

    def _phase8(self, draft_path, audit_path, kb_path,
                platform, policy_pack, user_notes) -> dict:
        MAX_RETRIES = 2
        current_draft = draft_path
        current_audit = audit_path

        for attempt in range(1, MAX_RETRIES + 1):
            self._banner(8, f"Auto-Fix Attempt {attempt}/{MAX_RETRIES}", "Writer Agent")
            self.permissions.grant("writer_agent", 8, "WRITE", f"fixed_listing_r{attempt}.json", {})
            fixed = self.writer_agent.run_phase6(
                draft_listing_path=str(current_draft), audit_report_path=str(current_audit),
                knowledge_pack_path=str(kb_path), platform=platform,
                user_notes=user_notes, attempt=attempt,
            )
            fixed_path = self.output_dir / f"fixed_listing_r{attempt}.json"
            fixed_path.write_text(json.dumps(fixed, indent=2), encoding="utf-8")

            re_audit = self.writer_agent.run_phase5(
                draft_listing_path=str(fixed_path), knowledge_pack_path=str(kb_path),
                platform=platform, policy_pack=policy_pack,
            )
            re_audit_path = self.output_dir / f"audit_r{attempt}.json"
            re_audit_path.write_text(json.dumps(re_audit, indent=2), encoding="utf-8")

            new_score = re_audit.get("quality_index", 0)
            self.session.record_phase(8, f"RETRY_{attempt}", f"Score: {new_score}/100")

            verdict = self._gate(8, re_audit, platform, policy_pack,
                                  {"platform": platform, "product_name": ""})
            if verdict != "ABORT" and new_score >= 65:
                self.permissions.revoke_phase(8)
                return {"status": "FIXED", "fixed_path": fixed_path}

            self.permissions.revoke_phase(8)
            current_draft = fixed_path
            current_audit = re_audit_path

        return {
            "status": "ESCALATE",
            "reason": f"Auto-fix exhausted {MAX_RETRIES} retries (backup mode). Human review required."
        }

    # ── GOVERNOR GATE — deterministic interpretation ──────────────────

    def _gate(self, phase, data, platform, policy_pack, context) -> str:
        verdict = self.governor.audit(phase=phase, data=data,
                                       platform=platform, policy_pack=policy_pack)
        self.session.record_governor(phase, verdict)

        if verdict["overall_status"] == "PASS":
            return "PROCEED"
        if verdict["overall_status"] == "HARD_FAIL":
            self.session.record_hard_fail(phase, verdict.get("violations", []))
            return "ABORT"
        if verdict["overall_status"] == "SOFT_FAIL":
            decision = self.brain.decide_on_verdict(phase, verdict, context)
            return "ABORT" if decision.get("decision") == "HARD_STOP" else "PROCEED"
        return "PROCEED"

    # ── USER CHECKPOINT — identical Vibe Diff prompt as manager.py ────

    def _user_checkpoint(self, draft: dict, audit: dict) -> dict:
        score      = audit.get("quality_index", 0)
        violations = audit.get("violations", [])

        print("\n" + "═"*62)
        print("  ⟳  USER CHECKPOINT — PHASE 7  |  VIBE DIFF (BACKUP MODE)")
        print("═"*62)
        print(f"\n  📊 Quality Index : {score}/100")
        if score < 75:
            print(f"  ⚠️  WARNING: Score below recommended 75")

        for field, val in draft.items():
            if field in ["platform", "changes_log"]:
                continue
            if isinstance(val, list):
                print(f"\n  {field.upper()}:")
                for i, v in enumerate(val, 1):
                    print(f"    {i}. {str(v)[:120]}")
            else:
                print(f"\n  {field.upper()}:\n    {str(val)[:200]}")

        if violations:
            print(f"\n  ── VIOLATIONS ({len(violations)}) ──")
            for v in violations[:5]:
                print(f"  [{v.get('layer')}] {v.get('severity','?'):10} {v.get('field'):15} → {v.get('finding','')[:70]}")

        print("\n  APPROVE to proceed  |  REJECT + notes to auto-fix")
        print("═"*62)
        raw = input("\n  Your decision: ").strip()
        if raw.upper().startswith("APPROVE"):
            return {"decision": "APPROVED", "notes": ""}
        return {"decision": "REJECTED", "notes": raw.replace("REJECT","").strip()}

    # ── VALIDATION + POLICY/KB HELPERS — identical to manager.py ──────

    def _validate_input(self, inp: dict) -> dict:
        required = ["platform", "product_name", "product_details", "category", "target_market"]
        missing  = [k for k in required if not str(inp.get(k,"")).strip()]
        if missing:
            return {"ok": False, "error": f"Missing required fields: {', '.join(missing)}"}
        if inp["platform"].lower() not in self.SUPPORTED_PLATFORMS:
            return {"ok": False, "error": f"Platform must be one of: {self.SUPPORTED_PLATFORMS}"}
        return {"ok": True}

    def _load_policy(self, platform: str) -> dict:
        spec_path = PROJECT_ROOT / "spec" / f"{platform}.yaml"
        if not spec_path.exists():
            return {"ok": False, "expected": str(spec_path)}
        with open(spec_path, encoding="utf-8") as f:
            policy = yaml.safe_load(f)
        return {"ok": True, "policy": policy or {}, "path": str(spec_path)}

    def _check_kb(self, platform: str) -> dict:
        raw_dir   = PROJECT_ROOT / "knowledge_base" / "raw_data"
        faiss_dir = PROJECT_ROOT / "knowledge_base" / "faiss_index"
        if raw_dir.exists():
            files = list(raw_dir.glob(f"{platform}*.txt")) + list(raw_dir.glob(f"{platform}*.json")) + list(raw_dir.glob("*.txt"))
            if files:
                return {"ok": True}
        if faiss_dir.exists() and list(faiss_dir.glob(f"{platform}*.index")):
            return {"ok": True}
        return {"ok": False, "error": f"No knowledge base data found for '{platform}'."}

    def _banner(self, phase, name, agent):
        p = self.session.platform.upper() if self.session else "?"
        log.info(f"{'─'*60}")
        log.info(f"  ▶ PHASE {phase}/9  —  {name}  [BACKUP MODE]")
        log.info(f"  Agent: {agent}  |  Platform: {p}  |  ACTIVE")
        log.info(f"{'─'*60}")

    def _abort(self, phase, reason) -> dict:
        msg = json.dumps(reason, indent=2) if isinstance(reason, dict) else str(reason)
        log.error(f"PIPELINE ABORTED (BACKUP MODE)  phase={phase}")
        if self.session and self.output_dir:
            self.session.record_hard_fail(phase or 0, msg)
            try:
                self.session.save(self.output_dir)
            except Exception:
                pass
        return {"status": "ABORTED", "phase": phase, "reason": reason, "mode": "BACKUP_DETERMINISTIC"}


# Alias so scripts/circuit_breaker.py can do a clean drop-in import swap
LeadOrchestrator = LeadOrchestratorBackup