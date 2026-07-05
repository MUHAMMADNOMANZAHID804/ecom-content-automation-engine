# Anti-Gravity 2.0 — Listing Agent Skill

## What this system does
Generates SEO-optimized, policy-compliant Amazon/Etsy/Shopify listings through
an 8-phase pipeline, using competitor and review data as grounding instead of
un-sourced generation.

## When to use which entry point
- `ui/app.py` — interactive use, human-in-the-loop review at every phase.
- `run_pipeline.py` (or `scripts/run_pipeline.py`) — CLI/cron use for batch jobs.
- `core/manager.py:PipelineManager` — import directly for programmatic use
  (e.g. from another agent or notebook).

## Architecture at a glance
```
core/manager.py        Orchestrator — openai/gpt-oss-120b drives phase logic
core/subagents.py       8 narrow-scope sub-agents (mostly openai/gpt-oss-20b)
core/safety_governor.py Policy compliance gate, reads spec/*.yaml
modules/competitor_analyzer.py   Jungle Scout CSV -> top 5 -> gap PDF
modules/review_analyzer.py       Reviews -> sentiment/pain points -> PDF
tools/scrapper_mcp*.py  ASIN -> raw product/review data (Phase 1)
tools/file_mcp.py       Sandboxed file read/write for uploads + reports
tools/sandboxed_executor.py  Isolated subprocess runner for utility scripts
scripts/*.py            circuit_breaker, audit_logger, keywords_check,
                         count_words, build_kb — deterministic support code
spec/*.yaml             Per-platform + global policy source of truth
```

## Golden rules (do not violate)
1. **Logic in code, not prompts.** Character counting, keyword-stuffing
   detection, and CSV ranking are deterministic Python — never delegate these
   to an LLM call.
2. **No hardcoded secrets.** `GROQ_API_KEY` lives in `.env` only.
3. **Human-in-the-loop by default.** `run_full_pipeline` always stops with a
   final PDF for review; nothing auto-publishes to a live storefront.
4. **8 phases, in order, always logged.** Every phase call goes through
   `AuditLogger` and `CircuitBreaker`. Do not call sub-agents directly from
   the UI — always go through `PipelineManager`.
5. **Small model for narrow jobs, big model for copy + fixes.** Don't route
   listing generation or the audit-fix rewrite through the small model — it
   degrades character-count precision.

## Extending the pipeline
To add a 9th phase, add a `phaseN_*` method to `PipelineManager`, a matching
sub-agent class in `subagents.py`, wire it into `run_full_pipeline`, and add
its status row to `PHASE_LABELS` in `ui/app.py`. Do not renumber existing
phases — downstream logs and PDFs reference phase names by string.