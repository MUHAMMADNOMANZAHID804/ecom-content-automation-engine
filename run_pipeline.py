"""
run_pipeline.py (root — the ONLY copy, no wrapper/duplicate anymore)
------------------------------------------------------------------------
CLI runner. Mirrors what ui/app.py does but scriptable / cron-able.

Usage:
  python run_pipeline.py --platform amazon --brand "Acme" \\
      --features "Stainless steel water bottle, 32oz" \\
      --csv data/jungle_scout_export.csv \\
      --reviews-file data/reviews.txt

  python run_pipeline.py --platform etsy --brand "Acme" \\
      --features "Hand-poured soy candle"
"""

import argparse
import logging
import os
import sys

# Same defensive fix as ui/app.py — ensures project-root imports resolve
# correctly regardless of the directory this script is invoked from.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.manager import PipelineManager, PipelineState
from scripts.build_kb import KBClient
from tools.scrapper_mcp import ScraperMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Run the Anti-Gravity 2.0 listing pipeline")
    p.add_argument("--platform", choices=["amazon", "etsy", "shopify"], required=True)
    p.add_argument("--brand", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--csv", help="Path to Jungle Scout CSV (Amazon Competitor Analyzer)")
    p.add_argument("--reviews-file", help="Path to a text file of pasted reviews")
    p.add_argument("--asin", help="ASIN to auto-resolve product + review data")
    p.add_argument("--listing-only", action="store_true",
                    help="Skip Phases 1-3 and use --upload-reports instead")
    p.add_argument("--upload-reports", nargs="*", default=[],
                    help="Paths to previously generated PDF reports (manual override)")
    return p.parse_args()


def main():
    args = parse_args()

    state = PipelineState(
        platform=args.platform,
        brand=args.brand,
        features=args.features,
        asin=args.asin or None,
        jungle_scout_csv_path=args.csv,
        uploaded_report_paths=args.upload_reports,
    )
    if args.reviews_file:
        with open(args.reviews_file, "r", encoding="utf-8") as f:
            state.manual_reviews_text = f.read()

    try:
        kb_client = KBClient()
    except Exception as e:  # noqa: BLE001
        logger.warning("KB client unavailable (%s) — Phase 4 will run with empty context.", e)
        kb_client = None

    manager = PipelineManager(scraper=ScraperMCP(), kb_client=kb_client)

    if args.listing_only:
        state = manager.run_listing_only(state)
    else:
        state = manager.run_full_pipeline(state)

    print("\n=== FINAL LISTING ===\n")
    print(state.final_listing)
    print("\n=== AUDIT NOTES ===")
    print(state.audit_violations or "None")
    print("\n=== PDF REPORTS ===")
    for name, path in state.pdf_paths.items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
