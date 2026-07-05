import sys
import os

# --- FIX: Streamlit Cloud has no real TTY, so libraries that use `rich`
# (directly or via a dependency, e.g. the Groq SDK's error/log rendering)
# can fail to auto-detect terminal width and get back 0, which makes rich
# throw "Not enough horizontal space to render a single character."
# Forcing COLUMNS/LINES here fixes it at the environment level, before any
# backend module is imported. Backend logic is completely untouched.
os.environ.setdefault("COLUMNS", "200")
os.environ.setdefault("LINES", "50")
os.environ.setdefault("TERM", "xterm-256color")
os.environ.setdefault("FORCE_COLOR", "0")

# Yeh line Python ko batayegi ke root folder ko bhi check kare
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from core.manager import PipelineManager
from tools.scrapper_mcp import ScraperMCP
"""
ui/app.py
---------
Streamlit dashboard for the multi-tool e-commerce listing platform.

THIS FILE IS PRESENTATION-ONLY. Every call into core/manager.py, every
PipelineState field, and every render_*() function's logic is IDENTICAL to
the previous version — only layout, navigation, and styling changed to match
the "Anti-Gravity 2.0" reference design (dark theme, pill platform switch,
vertical phase-nav, System Status card, styled upload dropzone, gradient CTA
buttons). No manager method signatures, no business logic, was touched.

Layout:
- Header: logo badge + platform pill switch (Amazon/Etsy/Shopify) + live
  system status indicator.
- Left sidebar: "AGENTIC WORKFLOW" — vertical nav between the tools
  available for the selected platform (Competitor Analyzer / Review Analyzer
  / Listing Generator), each labeled with its pipeline phase number, with
  the active tool highlighted. Below it, a "System Status" card showing
  Platform / Competitor Data / Review Data at a glance.
- Main panel: only the active tool's UI renders here, in a bordered card.
- Results are shown in plain English / tables, never raw JSON (unchanged).
"""

import os
import tempfile

import streamlit as st

from core.manager import PipelineManager, PipelineState
from tools.scrapper_mcp import ScraperMCP  # existing tool, untouched

st.set_page_config(
    page_title="Anti-Gravity 2.0 — Listing Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

PHASE_LABELS = [
    "1. ASIN / Data Ingestion",
    "2. Competitor Gap Analysis",
    "3. Review Sentiment Analysis",
    "4. RAG Retrieval",
    "5. Keyword Synthesis",
    "6. Listing Generation (2 drafts \u2192 final)",
    "7. Structure Audit + Auto-Fix",
    "8. Competitive Advantage + PDF",
]

# Tool -> (icon, pipeline-phase label shown in the nav, tab title)
TOOL_META = {
    "competitor": {"icon": "📊", "phase": "Phase 2", "label": "Competitor Analyzer"},
    "review":     {"icon": "💬", "phase": "Phase 3", "label": "Review Analyzer"},
    "listing":    {"icon": "🧩", "phase": "Phase 8", "label": "Listing Generator"},
}


# ---------------------------------------------------------------------------
# THEME — dark, card-based, matching the Anti-Gravity 2.0 reference design.
# Pure CSS/visual layer. No logic lives here.
# ---------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown("""
    <style>
        :root {
            --ag-bg: #0b0b10;
            --ag-panel: #15151c;
            --ag-card: #17171f;
            --ag-border: #2a2a35;
            --ag-border-active: #7c6cf6;
            --ag-accent: #7c6cf6;
            --ag-accent-grad: linear-gradient(135deg, #7c6cf6 0%, #5b8def 100%);
            --ag-text: #eaeaf2;
            --ag-text-dim: #9494a6;
            --ag-success: #34d399;
            --ag-highlight-bg: rgba(124, 108, 246, 0.10);
            --ag-highlight-border: #a78cf7;
            /* Tells the browser/OS to render native controls (checkboxes,
               dropdown menus, scrollbars, date pickers) in dark mode too —
               without this, some devices render those as light/white boxes
               regardless of the page's own CSS. */
            color-scheme: dark;
        }

        html, body { color-scheme: dark; }

        .stApp {
            background: var(--ag-bg);
            color: var(--ag-text);
        }

        /* Force dark background on every top-level Streamlit container,
           including ones that otherwise fall back to OS light/dark
           preference on some mobile browsers. */
        [data-testid="stAppViewContainer"], [data-testid="stHeader"],
        [data-testid="stToolbar"], [data-testid="stBottomBlockContainer"] {
            background: var(--ag-bg) !important;
        }

        section[data-testid="stSidebar"] {
            background: var(--ag-panel);
            border-right: 1px solid var(--ag-border);
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1.2rem;
        }

        /* Header logo badge */
        .ag-logo-badge {
            display: inline-flex; align-items: center; justify-content: center;
            width: 40px; height: 40px; border-radius: 10px;
            background: var(--ag-accent-grad);
            color: white; font-weight: 700; font-size: 0.95rem;
            margin-right: 10px;
        }
        .ag-header-row { display: flex; align-items: center; gap: 10px; }
        .ag-title { font-size: 1.4rem; font-weight: 700; color: var(--ag-text); }
        .ag-title-version { color: var(--ag-text-dim); font-weight: 400; }

        /* System status pill (top right) */
        .ag-status-pill {
            text-align: right; font-size: 0.75rem; color: var(--ag-text-dim);
        }
        .ag-status-dot {
            display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            background: var(--ag-success); margin-right: 6px;
        }

        /* Sidebar nav section title */
        .ag-nav-heading {
            font-size: 0.72rem; letter-spacing: 0.08em; color: var(--ag-text-dim);
            text-transform: uppercase; margin: 0.4rem 0 0.6rem 0; font-weight: 600;
        }

        /* Sidebar nav item — active (non-clickable div) */
        .ag-nav-active {
            display: flex; align-items: center; gap: 10px;
            background: rgba(124, 108, 246, 0.12);
            border: 1px solid var(--ag-border-active);
            border-radius: 10px; padding: 10px 12px; margin-bottom: 8px;
            color: var(--ag-accent); font-weight: 600; font-size: 0.9rem;
        }
        .ag-nav-active .ag-nav-phase { color: var(--ag-accent); opacity: 0.75;
            font-size: 0.72rem; font-weight: 500; display: block; }

        .ag-nav-inactive-wrap { margin-bottom: 8px; }
        .ag-nav-inactive-wrap .stButton > button {
            width: 100%; text-align: left; justify-content: flex-start;
            background: transparent; border: 1px solid var(--ag-border);
            color: var(--ag-text-dim); font-weight: 500; border-radius: 10px;
            padding: 10px 12px;
        }
        .ag-nav-inactive-wrap .stButton > button:hover {
            border-color: var(--ag-border-active); color: var(--ag-text);
        }

        /* System status card in sidebar */
        .ag-status-card {
            border: 1px solid var(--ag-border); border-radius: 10px;
            padding: 12px 14px; margin-top: 1rem; background: var(--ag-card);
        }
        .ag-status-card-title {
            font-size: 0.75rem; color: var(--ag-text-dim); font-weight: 600;
            margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.06em;
        }
        .ag-status-row { font-size: 0.85rem; color: var(--ag-text); margin: 3px 0; }
        .ag-status-value-done { color: var(--ag-success); font-weight: 600; }
        .ag-status-value-pending { color: var(--ag-text-dim); font-weight: 600; }

        /* Main content card wrapper */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--ag-card);
            border-color: var(--ag-border) !important;
            border-radius: 14px !important;
        }

        /* Primary CTA buttons — gradient, rounded, prominent */
        .stButton > button[kind="primary"], .stDownloadButton > button {
            background: var(--ag-accent-grad); border: none; border-radius: 10px;
            font-weight: 600; padding: 0.6rem 1.4rem; color: white;
        }
        .stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
            filter: brightness(1.08);
        }

        /* Segmented control / radio pills for platform switch */
        div[data-testid="stSegmentedControl"] button {
            border-radius: 8px !important;
        }

        /* File uploader dropzone — dashed, centered, matches reference */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--ag-bg) !important;
            border: 1.5px dashed var(--ag-border-active) !important;
            border-radius: 12px !important;
        }

        h1, h2, h3 { color: var(--ag-text) !important; }
        p, span, label { color: var(--ag-text); }
        .stCaption, [data-testid="stCaptionContainer"] { color: var(--ag-text-dim) !important; }

        /* --- Dark-mode consistency for native form widgets on ALL devices ---
           Text inputs, text areas, selectboxes, and their dropdown popovers
           are the elements most likely to render light/white on some mobile
           browsers regardless of page CSS. Force them dark explicitly. */
        input, textarea, select {
            background-color: var(--ag-card) !important;
            color: var(--ag-text) !important;
            border-color: var(--ag-border) !important;
        }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div, [data-baseweb="popover"] {
            background-color: var(--ag-card) !important;
            color: var(--ag-text) !important;
            border-color: var(--ag-border) !important;
        }
        [data-baseweb="menu"] li { background-color: var(--ag-card) !important; color: var(--ag-text) !important; }

        /* Expanders, dataframes/tables, metrics — keep them dark too */
        [data-testid="stExpander"], [data-testid="stDataFrame"],
        [data-testid="stMetric"], [data-testid="stMetricValue"] {
            background-color: var(--ag-card) !important;
            color: var(--ag-text) !important;
            border-color: var(--ag-border) !important;
        }

        /* Info/success/warning/error banners — keep readable in dark mode
           with a slightly tinted background instead of Streamlit's default
           near-white one on some versions/devices. */
        [data-testid="stAlertContainer"] {
            background-color: var(--ag-card) !important;
            border: 1px solid var(--ag-border) !important;
            color: var(--ag-text) !important;
        }

        /* --- Highlighted "tip" callouts (the quotes/guidance you asked for) --- */
        .ag-tip {
            background: var(--ag-highlight-bg);
            border-left: 3px solid var(--ag-highlight-border);
            border-radius: 6px;
            padding: 8px 12px;
            margin: 6px 0 12px 0;
            font-size: 0.85rem;
            font-style: italic;
            color: var(--ag-text);
        }
        .ag-tip b { font-style: normal; color: var(--ag-accent); }

        /* --- Mobile responsiveness: usable on phones/tablets, not just desktop --- */
        @media (max-width: 640px) {
            .ag-header-row { flex-wrap: wrap; }
            .ag-title { font-size: 1.1rem; }
            .ag-status-pill { text-align: left; margin-top: 6px; }
            div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
            .stButton > button, .stDownloadButton > button { width: 100%; }
        }
    </style>
    """, unsafe_allow_html=True)


def _get_manager() -> PipelineManager:
    if "manager" not in st.session_state:
        st.session_state.manager = PipelineManager(scraper=ScraperMCP())
    return st.session_state.manager


def _get_state() -> PipelineState:
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = PipelineState()
    return st.session_state.pipeline_state


def tip(text: str) -> None:
    """Highlighted guidance callout — purely additive usability sugar."""
    st.markdown(f'<div class="ag-tip">💡 {text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header — logo, title, platform pill switch, live status
# ---------------------------------------------------------------------------
def render_header(state: PipelineState) -> None:
    col_logo, col_platform, col_status = st.columns([2.2, 2, 1.3])
    with col_logo:
        st.markdown(
            '<div class="ag-header-row">'
            '<span class="ag-logo-badge">AG</span>'
            '<span class="ag-title">ANTI-GRAVITY <span class="ag-title-version">v2.0</span></span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_platform:
        platforms = ["amazon", "etsy", "shopify"]
        labels = ["Amazon", "Etsy", "Shopify"]
        if hasattr(st, "segmented_control"):
            choice = st.segmented_control(
                "Platform", labels, default=labels[platforms.index(state.platform)],
                label_visibility="collapsed",
            )
            state.platform = platforms[labels.index(choice)] if choice else state.platform
        else:
            choice = st.radio("Platform", labels, horizontal=True,
                               index=platforms.index(state.platform),
                               label_visibility="collapsed")
            state.platform = platforms[labels.index(choice)]
    with col_status:
        st.markdown(
            '<div class="ag-status-pill">SYSTEM STATUS<br>'
            '<span class="ag-status-dot"></span>Pipeline Active</div>',
            unsafe_allow_html=True,
        )
    st.write("")


# ---------------------------------------------------------------------------
# Sidebar — "AGENTIC WORKFLOW" nav + "System Status" card
# ---------------------------------------------------------------------------
def render_sidebar_nav(state: PipelineState) -> str:
    available_tools = ["competitor", "review", "listing"] if state.platform == "amazon" \
        else ["review", "listing"]

    if "active_tool" not in st.session_state or st.session_state.active_tool not in available_tools:
        st.session_state.active_tool = available_tools[0]

    with st.sidebar:
        st.markdown('<div class="ag-nav-heading">Agentic Workflow</div>', unsafe_allow_html=True)
        for tool in available_tools:
            meta = TOOL_META[tool]
            if tool == st.session_state.active_tool:
                st.markdown(
                    f'<div class="ag-nav-active">{meta["icon"]}&nbsp;&nbsp;'
                    f'<span><span class="ag-nav-phase">{meta["phase"]}</span>'
                    f'{meta["label"]}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="ag-nav-inactive-wrap">', unsafe_allow_html=True)
                if st.button(f'{meta["icon"]}  {meta["phase"]}: {meta["label"]}',
                             key=f"nav_{tool}", use_container_width=True):
                    st.session_state.active_tool = tool
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # --- System Status card ---
        competitor_status = "Done" if state.gap_report else "Pending"
        review_status = "Done" if state.review_report else "Pending"
        listing_status = "Done" if state.final_listing else "Pending"

        def _row(label, value):
            cls = "ag-status-value-done" if value == "Done" else "ag-status-value-pending"
            return f'<div class="ag-status-row">{label}: <span class="{cls}">{value}</span></div>'

        status_html = (
            '<div class="ag-status-card">'
            '<div class="ag-status-card-title">System Status</div>'
            f'<div class="ag-status-row">Platform: <strong>{state.platform.capitalize()}</strong></div>'
            + (_row("Competitor Data", competitor_status) if state.platform == "amazon" else "")
            + _row("Review Data", review_status)
            + _row("Listing", listing_status)
            + '</div>'
        )
        st.markdown(status_html, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="ag-nav-heading">Pipeline Progress</div>', unsafe_allow_html=True)
        done = set(state.phase_log)
        phase_keys = [
            "phase1_ingest", "phase2_competitor_analysis", "phase3_review_analysis",
            "phase4_rag_retrieval", "phase5_keyword_synthesis",
            "phase6_listing_generation", "phase7_audit_and_fix",
            "phase8_finalize_and_pdf",
        ]
        for label, key in zip(PHASE_LABELS, phase_keys):
            matched = any(k.startswith(key) or key in k for k in done) or key in done
            st.checkbox(label, value=matched, disabled=True, key=f"progress_{key}")

        st.caption(
            "Auto-Add is the system's built-in advantage: when it's on, results "
            "from Competitor Analyzer / Review Analyzer are carried forward "
            "automatically as extra context for the Listing Generator."
        )

    return st.session_state.active_tool


# ---------------------------------------------------------------------------
# Plain-English renderers — NEVER show raw JSON/dict output to the user.
# (Unchanged from the previous version.)
# ---------------------------------------------------------------------------
def render_competitor_table(rows: list, caption: str) -> None:
    if not rows:
        st.info("No competitor rows to show.")
        return
    st.write(f"**{caption}**")
    table_rows = []
    for c in rows:
        table_rows.append({
            "ASIN": c.get("asin", ""),
            "Brand": c.get("brand", ""),
            "Price": c.get("price", ""),
            "Monthly Units Sold": f"{c.get('avg_monthly_sales', 0):.0f}",
            "Star Rating": f"{c['star_rating']:.1f}" if c.get("star_rating") else "N/A",
            "Title": c.get("title", "")[:90],
        })
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_gap_report(gap_report: dict) -> None:
    if not gap_report:
        st.info("No competitor gap data yet.")
        return

    st.markdown("#### Phase 1: Top 5 Competitors Market Matrix")
    if gap_report.get("top5"):
        render_competitor_table(gap_report["top5"], "Ranked by Monthly Units Sold")
    if gap_report.get("top_rated"):
        st.write("")
        render_competitor_table(gap_report["top_rated"], "Top Rated (Customer Satisfaction Signal)")

    st.write("")
    st.markdown("#### Phase 2: Actionable Sourcing & Copywriting Solutions")
    st.write("**How to beat these competitors:**")

    if gap_report.get("positioning_gaps"):
        st.write("_Sourcing & positioning upgrades:_")
        for g in gap_report["positioning_gaps"]:
            st.write(f"- {g}")

    top_keywords = gap_report.get("competitor_keywords", [])[:15]
    if top_keywords:
        st.write("_Use these high-converting keywords, pulled from competitor titles:_")
        st.write(", ".join(f'"{k}"' for k in top_keywords))

    if gap_report.get("missing_keywords"):
        st.write("**Keywords your product is missing:**")
        st.write(", ".join(gap_report["missing_keywords"]))

    if gap_report.get("missing_features"):
        st.write("**Features competitors have that you don't:**")
        st.write(", ".join(gap_report["missing_features"]))

    if gap_report.get("summary"):
        st.write("**Summary:**")
        st.write(gap_report["summary"])


def render_review_insights(insights: dict) -> None:
    if not insights:
        st.info("No review analysis yet.")
        return
    score = insights.get("sentiment_score", 0.0)
    st.metric("Overall Sentiment", f"{score:.2f}", help="-1 (very negative) to 1 (very positive)")

    complaints = insights.get("complaints", [])
    if complaints:
        st.write("**Customer complaints / pain points:**")
        for c in complaints:
            if isinstance(c, dict):
                st.write(f"- [{c.get('frequency', '')}] {c.get('issue', '')}")
            else:
                st.write(f"- {c}")
    else:
        st.write("**Customer complaints / pain points:** None found.")

    praises = insights.get("praises", [])
    if praises:
        st.write("**What customers like:**")
        for p in praises:
            st.write(f"- {p}")

    strategies = insights.get("improvement_strategies", [])
    if strategies:
        st.write("**Suggested improvements:**")
        for s in strategies:
            st.write(f"- {s}")


# ---------------------------------------------------------------------------
# Tool bodies — SAME logic/calls as before, now rendered inside a card
# and driven by the sidebar nav instead of st.tabs.
# ---------------------------------------------------------------------------
def competitor_analyzer_tool(manager: PipelineManager, state: PipelineState) -> None:
    with st.container(border=True):
        st.subheader("📊 Competitor Analyzer")
        st.caption("Upload a Jungle Scout CSV export. The system finds your top 5 "
                   "competitors by sales volume, plus a separate top-5 by star "
                   "rating, and extracts the keywords they use in their titles.")
        st.markdown("**1. Upload Jungle Scout CSV**")
        tip("Export from Jungle Scout's <b>Extension</b> or <b>Chrome plugin</b> as CSV. "
            "The system only needs the Title, Monthly Units Sold, Star Rating, and "
            "Price columns — everything else is ignored automatically.")
        csv_file = st.file_uploader("Upload Jungle Scout CSV", type=["csv"],
                                     label_visibility="collapsed")
        auto_add_1 = st.checkbox(
            "Auto-Add results to Listing Generator (recommended)", value=True,
            key="autoadd_competitor",
            help="Keeps this competitor data as extra context for the Listing "
                 "Generator automatically — one less thing to re-enter."
        )
        if csv_file and st.button("Analyze Top 5 Competitors", type="primary"):
            tmp_path = os.path.join(tempfile.gettempdir(), csv_file.name)
            with open(tmp_path, "wb") as f:
                f.write(csv_file.getbuffer())
            state.jungle_scout_csv_path = tmp_path
            try:
                with st.spinner("Finding your top 5 competitors..."):
                    updated = manager.phase2_competitor_analysis(state)
                st.session_state.pipeline_state = updated
                st.success("Competitor analysis complete.")
                render_gap_report(updated.gap_report)
                if updated.pdf_paths.get("competitor_report"):
                    with open(updated.pdf_paths["competitor_report"], "rb") as f:
                        st.download_button("Download Competitor PDF Report", f,
                                            file_name="competitor_report.pdf")
                if not auto_add_1:
                    updated.gap_report = {}
            except Exception as e:  # noqa: BLE001
                st.error(f"Competitor analysis failed: {e}")


def review_analyzer_tool(manager: PipelineManager, state: PipelineState) -> None:
    with st.container(border=True):
        st.subheader("💬 Review Analyzer")
        st.caption("Paste customer reviews (or an ASIN, on Amazon) and the system "
                   "finds complaints, praises, and concrete ways to improve the listing.")
        input_mode = st.radio("Input method", ["Paste reviews", "ASIN (Amazon only)"],
                               horizontal=True)
        if input_mode == "ASIN (Amazon only)" and state.platform == "amazon":
            tip("The system fetches the product page and its reviews for you — "
                "just paste the 10-character code from the product URL.")
            state.asin = st.text_input("ASIN", state.asin or "",
                                        placeholder="e.g. B08N5WRWNW")
        else:
            tip("Paste reviews exactly as customers wrote them — one per line works "
                "best. The more honest/critical ones you include, the better the "
                "pain-point analysis.")
            state.manual_reviews_text = st.text_area(
                "Paste customer reviews (one per line)", state.manual_reviews_text or "",
                height=150,
                placeholder=(
                    "Great quality but runs small, had to size up.\n"
                    "Stopped working after 2 weeks, disappointed.\n"
                    "Exactly as described, fast shipping, would buy again."
                ),
            )
        auto_add_2 = st.checkbox(
            "Auto-Add results to Listing Generator (recommended)", value=True,
            key="autoadd_review",
            help="Keeps the pain points and praises found here as extra context "
                 "for the Listing Generator automatically."
        )
        if st.button("Run Review Analysis", type="primary"):
            try:
                with st.spinner("Analyzing sentiment and pain points..."):
                    if state.asin:
                        state = manager.phase1_ingest(state)
                    state = manager.phase3_review_analysis(state)
                st.session_state.pipeline_state = state
                st.success("Review analysis complete.")
                render_review_insights(state.review_report)
                if state.pdf_paths.get("review_report"):
                    with open(state.pdf_paths["review_report"], "rb") as f:
                        st.download_button("Download Review PDF Report", f,
                                            file_name="review_report.pdf")
                if not auto_add_2:
                    state.review_report = {}
            except Exception as e:  # noqa: BLE001
                st.error(f"Review analysis failed: {e}")


def listing_generator_tool(manager: PipelineManager, state: PipelineState) -> None:
    with st.container(border=True):
        st.subheader("🧩 Listing Generator")
        st.caption(
            "Combines: the product info you type below, PLUS whatever Auto-Add "
            "carried over from Competitor Analyzer / Review Analyzer, PLUS the "
            "knowledge base — to write a complete, platform-correct listing in "
            "one pass (not a summary)."
        )

        # Brand/Features live ONLY here now — per your feedback, they weren't
        # useful on the Competitor Analyzer / Review Analyzer tabs.
        col1, col2 = st.columns([1, 2])
        with col1:
            state.brand = st.text_input("Brand name", state.brand, placeholder="e.g. Acme")
        with col2:
            state.features = st.text_area(
                "Product features (GSM, certifications, dimensions, etc.)",
                state.features, height=68,
                placeholder="e.g. 100% organic cotton, 220 GSM, OEKO-TEX certified, true-to-size XS-4XL",
            )

        has_autoadd_data = bool(state.gap_report or state.review_report)
        if has_autoadd_data:
            st.success(
                "Auto-Added data detected from earlier tabs — it will be used "
                "automatically as extra context below."
            )
        else:
            st.info(
                "No Auto-Added data yet. Either run Competitor/Review Analyzer "
                "first, or upload previous PDF reports below as a manual override."
            )

        uploaded_reports = st.file_uploader(
            "Manual Override: Report Upload Box (previous PDFs)",
            type=["pdf"], accept_multiple_files=True,
        )
        if uploaded_reports:
            paths = []
            for uf in uploaded_reports:
                p = os.path.join(tempfile.gettempdir(), uf.name)
                with open(p, "wb") as f:
                    f.write(uf.getbuffer())
                paths.append(p)
            state.uploaded_report_paths = paths

        state.competitor_data_manual = st.text_area(
            "Competitor Data (Titles + Negative Reviews) — optional manual input",
            state.competitor_data_manual or "",
            height=140,
            help="Paste competitor titles and their negative reviews directly here. "
                 "Useful on Etsy/Shopify (no CSV analyzer), or to add extra context "
                 "on top of whatever Auto-Add already carried over.",
            placeholder=(
                "Competitor 1: Hanes Beefy-T Short Sleeve Crew\n"
                "Keywords: Short Sleeve, Crew Neck, Cotton T-Shirt, Classic Fit\n"
                "Negative: Shrinks 2 sizes after first wash. Colors fade fast.\n\n"
                "Competitor 2: Gildan Heavy Cotton Adult T-Shirt\n"
                "Keywords: Heavy Cotton, Adult T-Shirt, Classic Fit, Unisex\n"
                "Negative: Stiff uncomfortable fabric. Sizing inconsistent."
            ),
        )

        col_a, col_b = st.columns(2)
        with col_a:
            state.creativity = st.slider(
                "Creativity", 0.0, 1.0, state.creativity, 0.01,
                help="Lower = safer, more literal to your inputs. Higher = more "
                     "varied, higher-energy phrasing."
            )
        with col_b:
            state.auto_fix_enabled = st.toggle(
                "Auto-fix violations", value=state.auto_fix_enabled,
                help="If a section falls outside the platform's required "
                     "character range, automatically rewrite just that section "
                     "(up to 2 retries). Turn off to see the raw, unfixed output."
            )

        tip("Keep Auto-fix ON for your first run — it silently corrects any "
            "section that falls outside this platform's character limits "
            "before you ever see it. Generation is single-shot now (one final "
            "listing, not drafts) to keep token usage minimal.")

        if st.button("Generate Full Listing + PDF Report", type="primary"):
            try:
                with st.spinner("Running phases 4-8: RAG \u2192 keywords \u2192 generate "
                                 "\u2192 audit \u2192 final PDF..."):
                    if has_autoadd_data or state.uploaded_report_paths:
                        state = manager.run_listing_only(state)
                    else:
                        state = manager.run_full_pipeline(state)
                st.session_state.pipeline_state = state
                st.success("Listing generated.")

                st.write("**Final Listing**")
                st.text_area("Final Listing", state.final_listing, height=350,
                             label_visibility="collapsed")

                # --- Audit report lives HERE now, not in the PDF ---
                st.write("**Audit Report**")
                if state.full_audit_report:
                    for entry in state.full_audit_report:
                        if entry["status"] == "PASS":
                            st.success(f"✅ {entry['detail']}", icon="✅")
                        else:
                            st.error(f"❌ {entry['detail']}", icon="❌")
                else:
                    st.info("No audit data available.")

                if state.audit_violations:
                    st.warning(f"{len(state.audit_violations)} item(s) still need attention "
                               f"after auto-fix retries — see above.")
                else:
                    st.info("Passed structural + policy audit.")

                if state.referee_verdict:
                    verdict = state.referee_verdict.get("verdict", "UNKNOWN")
                    if verdict == "APPROVED":
                        st.success(f"Referee (semantic policy check): {verdict}")
                    else:
                        st.warning(f"Referee (semantic policy check): {verdict}")
                        for c in state.referee_verdict.get("concerns", []):
                            st.write(f"- {c}")

                if state.risk_assessment:
                    level = state.risk_assessment.get("risk_level", "UNKNOWN")
                    st.write(f"**Brutal Risk Assessment: {level}**")
                    for r in state.risk_assessment.get("risks", []):
                        st.write(f"- {r}")

                if state.pdf_paths.get("listing_summary"):
                    with open(state.pdf_paths["listing_summary"], "rb") as f:
                        st.download_button("Download Final Optimization Report (PDF)", f,
                                            file_name="listing_summary.pdf")
            except Exception as e:  # noqa: BLE001
                st.error(
                    f"Listing generation failed: {e}\n\n"
                    "If this mentions token budget or empty output, see "
                    "core/subagents.py's LISTING_MAX_TOKENS setting."
                )


def main() -> None:
    inject_theme()
    manager = _get_manager()
    state = _get_state()

    render_header(state)

    active_tool = render_sidebar_nav(state)

    if active_tool == "competitor" and state.platform == "amazon":
        competitor_analyzer_tool(manager, state)
    elif active_tool == "review":
        review_analyzer_tool(manager, state)
    elif active_tool == "listing":
        listing_generator_tool(manager, state)

    st.session_state.pipeline_state = state


if __name__ == "__main__":
    main()
