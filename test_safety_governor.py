"""
core/test_safety_governor.py
------------------------------
Run with: pytest core/test_safety_governor.py -v
"""

import os
import tempfile
import textwrap

import pytest
import yaml

from core.safety_governor import SafetyGovernor


@pytest.fixture
def spec_dir(tmp_path):
    global_policy = {
        "banned_words": ["cure", "guaranteed", "fda approved"],
    }
    amazon_policy = {
        "banned_words": ["best seller"],
        "banned_claims": ["clinically proven"],
        "max_brand_length": 50,
        "max_all_caps_word_ratio": 0.25,
    }
    (tmp_path / "security-policy.yaml").write_text(yaml.dump(global_policy))
    (tmp_path / "amazon.yaml").write_text(yaml.dump(amazon_policy))
    (tmp_path / "Etsy.yaml").write_text(yaml.dump({}))
    (tmp_path / "Shopify.yaml").write_text(yaml.dump({}))
    return str(tmp_path)


def test_clean_input_passes(spec_dir):
    gov = SafetyGovernor(spec_dir=spec_dir)
    ok, notes = gov.check("amazon", brand="Acme", features="A durable steel water bottle")
    assert ok
    assert notes == []


def test_banned_word_blocks(spec_dir):
    gov = SafetyGovernor(spec_dir=spec_dir)
    ok, notes = gov.check("amazon", brand="Acme",
                           features="This product is a guaranteed cure for dehydration")
    assert not ok
    assert any("guaranteed" in n or "cure" in n for n in notes)


def test_platform_specific_banned_word(spec_dir):
    gov = SafetyGovernor(spec_dir=spec_dir)
    ok, notes = gov.check("amazon", brand="Acme", features="Our #1 best seller item")
    assert not ok
    assert any("best seller" in n for n in notes)


def test_brand_length_limit(spec_dir):
    gov = SafetyGovernor(spec_dir=spec_dir)
    long_brand = "A" * 60
    ok, notes = gov.check("amazon", brand=long_brand, features="Water bottle")
    assert not ok
    assert any("Brand name exceeds" in n for n in notes)


def test_listing_text_banned_claim(spec_dir):
    gov = SafetyGovernor(spec_dir=spec_dir)
    listing = "--- TITLE ---\nClinically Proven Steel Bottle by Acme"
    ok, notes = gov.check_listing_text("amazon", listing)
    assert not ok
    assert any("clinically proven" in n.lower() for n in notes)


def test_excessive_caps_flagged(spec_dir):
    gov = SafetyGovernor(spec_dir=spec_dir)
    listing = "THIS IS A REALLY LOUD ALL CAPS TITLE THAT SHOUTS EVERY SINGLE WORD"
    ok, notes = gov.check_listing_text("amazon", listing)
    assert not ok
    assert any("ALL-CAPS" in n for n in notes)


def test_missing_spec_files_allow_by_default(tmp_path):
    empty_dir = tmp_path / "empty_spec"
    empty_dir.mkdir()
    gov = SafetyGovernor(spec_dir=str(empty_dir))
    ok, notes = gov.check("etsy", brand="Acme", features="Handmade candle")
    assert ok
    assert notes == []