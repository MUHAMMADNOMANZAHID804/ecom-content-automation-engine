"""
core/safety_governor.py
------------------------
Policy Compliance Gate. Called by core/manager.py Phase 6 BEFORE any listing
copy is drafted, and again (implicitly, via keywords-check.py) after Phase 7.

Loads spec/security-policy.yaml plus the per-platform policy file
(spec/amazon.yaml, spec/Etsy.yaml, spec/Shopify.yaml) and checks:
  - banned claims / words (medical claims, guarantees Amazon disallows, etc.)
  - banned characters (emoji in titles, ALL-CAPS abuse, etc.)
  - required disclosures for certain product categories
  - brand-name usage rules
"""

import os
import re
import logging
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger("safety_governor")

SPEC_DIR = os.getenv("SPEC_DIR", "spec")

PLATFORM_SPEC_FILES = {
    "amazon": "amazon.yaml",
    "etsy": "Etsy.yaml",
    "shopify": "Shopify.yaml",
}


class SafetyGovernor:
    def __init__(self, spec_dir: str = SPEC_DIR):
        self.spec_dir = self._resolve_spec_dir(spec_dir)
        self.global_policy = self._load_yaml("security-policy.yaml")
        self._platform_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _resolve_spec_dir(spec_dir: str) -> str:
        """
        FIX for "Spec file not found: spec/security-policy.yaml": the old
        code only ever looked for `spec_dir` relative to the CURRENT WORKING
        DIRECTORY — which silently breaks the moment the app is launched
        from anywhere other than the project root (e.g. `streamlit run
        ui/app.py` from a different folder, a cron job, an IDE's default
        run directory, etc.). This tries several sensible locations and
        uses the first one that actually exists:
          1. spec_dir as given (CWD-relative) — unchanged default behavior
          2. spec_dir relative to the PROJECT ROOT, computed from this
             file's own location (core/safety_governor.py -> project root
             is one level up), which works regardless of CWD
        Falls back to the original path (with the existing warning-and-
        allow-by-default behavior) only if neither exists.
        """
        if os.path.isdir(spec_dir):
            return spec_dir

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, spec_dir)
        if os.path.isdir(candidate):
            logger.info("Resolved spec_dir via project root: %s", candidate)
            return candidate

        logger.warning(
            "Could not find a '%s' directory relative to CWD or project root "
            "(%s). Falling back to '%s' as given — policy checks will "
            "allow-by-default until this is fixed. Set SPEC_DIR in .env to "
            "an absolute path if your layout is non-standard.",
            spec_dir, project_root, spec_dir,
        )
        return spec_dir

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.spec_dir, filename)
        if not os.path.exists(path):
            logger.warning("Spec file not found: %s (governor will allow-by-default)", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _platform_policy(self, platform: str) -> Dict[str, Any]:
        platform = platform.lower()
        if platform not in self._platform_cache:
            filename = PLATFORM_SPEC_FILES.get(platform)
            self._platform_cache[platform] = self._load_yaml(filename) if filename else {}
        return self._platform_cache[platform]

    # ------------------------------------------------------------------
    # Pre-generation check (brand + feature text, before any copy exists)
    # ------------------------------------------------------------------
    def check(self, platform: str, brand: str, features: str) -> Tuple[bool, List[str]]:
        notes: List[str] = []
        policy = self._platform_policy(platform)
        banned_words = set(
            w.lower() for w in
            (self.global_policy.get("banned_words", []) + policy.get("banned_words", []))
        )
        blob = f"{brand} {features}".lower()

        for word in banned_words:
            if re.search(rf"\b{re.escape(word)}\b", blob):
                notes.append(f"Banned term detected in input: '{word}'")

        max_brand_len = policy.get("max_brand_length")
        if max_brand_len and len(brand) > max_brand_len:
            notes.append(f"Brand name exceeds {platform} max length ({max_brand_len})")

        ok = len(notes) == 0
        if not ok:
            logger.warning("SafetyGovernor blocked generation: %s", notes)
        return ok, notes

    # ------------------------------------------------------------------
    # Post-generation check (final listing text, used by keywords-check.py too)
    # ------------------------------------------------------------------
    def check_listing_text(self, platform: str, listing_text: str) -> Tuple[bool, List[str]]:
        notes: List[str] = []
        policy = self._platform_policy(platform)
        banned_words = set(
            w.lower() for w in
            (self.global_policy.get("banned_words", []) + policy.get("banned_claims", []))
        )
        lowered = listing_text.lower()
        for word in banned_words:
            if word in lowered:
                notes.append(f"Banned claim/word found in generated listing: '{word}'")

        max_caps_ratio = policy.get("max_all_caps_word_ratio", 0.3)
        words = re.findall(r"[A-Za-z]+", listing_text)
        caps_words = [w for w in words if w.isupper() and len(w) > 1]
        if words and (len(caps_words) / len(words)) > max_caps_ratio:
            notes.append("Excessive ALL-CAPS usage relative to platform policy")

        return len(notes) == 0, notes