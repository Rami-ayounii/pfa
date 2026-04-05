"""
tests/test_agent2.py
────────────────────
Unit tests for agents/agent2_social_scraper.py

All network calls, Selenium, and external APIs are mocked.
"""

import re
from unittest.mock import MagicMock, patch

import pytest

import agents.agent2_social_scraper as a2
from agents.agent2_social_scraper import (
    SocialProfile,
    HallucinationChecker,
    IG_BLACKLIST,
    _ig_handle_plausible,
    step1_resolve_urls,
    step2b_tripadvisor,
    step4_google_maps,
    _osm_fallback,
    Agent2SocialScraper,
)


# ── step1_resolve_urls ────────────────────────────────────────────────────────

def test_step1_skips_when_ddgs_none(monkeypatch):
    monkeypatch.setattr(a2, "DDGS", None)
    result = step1_resolve_urls("Dar El Jeld", location="Tunisia")
    assert result == {"wikipedia_url": "", "tripadvisor_url": ""}


def test_step1_uses_location_in_query(monkeypatch):
    captured_queries = []

    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, query, max_results=10):
            captured_queries.append(query)
            return []

    monkeypatch.setattr(a2, "DDGS", FakeDDGS)
    step1_resolve_urls("Test Brand", location="Morocco")

    assert len(captured_queries) == 1
    assert "Morocco" in captured_queries[0]
    assert "Test Brand" in captured_queries[0]


def test_step1_extracts_wikipedia_url(monkeypatch):
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, query, max_results=10):
            return [
                {"href": "https://en.wikipedia.org/wiki/Dar_El_Jeld", "body": ""},
                {"href": "https://www.tripadvisor.com/Restaurant_Review-123", "body": ""},
            ]

    monkeypatch.setattr(a2, "DDGS", FakeDDGS)
    result = step1_resolve_urls("Dar El Jeld", location="Tunisia")

    assert "wikipedia.org" in result["wikipedia_url"]
    assert "tripadvisor.com" in result["tripadvisor_url"]


# ── step2b_tripadvisor ────────────────────────────────────────────────────────

def test_tripadvisor_query_uses_location(monkeypatch):
    captured_queries = []

    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def text(self, query, max_results=8):
            captured_queries.append(query)
            return []

    monkeypatch.setattr(a2, "DDGS", FakeDDGS)
    profile = SocialProfile(brand="Dar El Jeld")
    step2b_tripadvisor(profile, "", apify_token="", location="Morocco")

    assert any("Morocco" in q for q in captured_queries)


# ── Selenium availability ─────────────────────────────────────────────────────

def test_selenium_skipped_when_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(a2, "_SELENIUM_AVAILABLE", False)
    profile = SocialProfile(brand="Test Brand")
    step4_google_maps(profile, "Test Brand", location="Tunisia")
    captured = capsys.readouterr()
    assert "SKIP" in captured.out or "not installed" in captured.out.lower()


def test_selenium_skipped_when_serpapi_filled(monkeypatch):
    monkeypatch.setattr(a2, "_SELENIUM_AVAILABLE", True)
    profile = SocialProfile(brand="Test Brand")
    profile.venue_data_source = "serpapi"
    profile.fsq_rating = 4.5
    # If SerpApi already filled data, step4 should copy and return early
    step4_google_maps(profile, "Test Brand", location="Tunisia")
    # gm_rating should now be populated from fsq_rating
    assert profile.gm_rating == 4.5


# ── _osm_fallback ─────────────────────────────────────────────────────────────

def test_osm_fallback_uses_location(monkeypatch):
    captured_bodies = []

    class FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"elements": [{"id": 1, "tags": {"name": "Test", "amenity": "restaurant"}}]}

    def fake_post(url, data=None, timeout=None):
        captured_bodies.append(data)
        return FakeResponse()

    import httpx
    monkeypatch.setattr("httpx.post", fake_post)

    profile = SocialProfile(brand="Test")
    _osm_fallback(profile, "Test", location="Morocco")

    assert any("Morocco" in str(b) for b in captured_bodies)


# ── HallucinationChecker ──────────────────────────────────────────────────────

def test_hallucination_checker_no_flags_on_valid_data(mock_groq_client):
    profile = SocialProfile(brand="Dar El Jeld")
    profile.gm_rating       = 4.5
    profile.gm_review_count = 320
    profile.ta_rating       = 4.3
    profile.ta_review_count = 200
    profile.gm_phone        = "+216 71 560 916"
    profile.gm_website      = "https://dareljeld.com"
    profile.ig_handle       = "dareljeld"
    profile.ig_followers    = 5000
    profile.ig_posts        = 120
    profile.fb_handle       = "dareljeld"

    checker = HallucinationChecker(profile, groq_client=mock_groq_client, location="Tunisia")
    checker._run_rules()

    rule_flags = [f for f in checker.flags if "rule:" in f.validation_method]
    assert len(rule_flags) == 0


def test_hallucination_checker_flags_bad_phone(mock_groq_client):
    profile = SocialProfile(brand="Test Brand")
    profile.gm_phone = "+33 1 23 45 67 89"  # French number, not Tunisia +216

    checker = HallucinationChecker(profile, groq_client=mock_groq_client, location="Tunisia")
    checker._run_rules()

    phone_flags = [f for f in checker.flags if f.field_name == "phone"]
    assert len(phone_flags) > 0


def test_hallucination_checker_flags_rating_out_of_range(mock_groq_client):
    profile = SocialProfile(brand="Test Brand")
    profile.gm_rating = 6.0  # out of 0-5 range

    checker = HallucinationChecker(profile, groq_client=mock_groq_client)
    checker._run_rules()

    rating_flags = [f for f in checker.flags if f.field_name == "gm_rating"]
    assert len(rating_flags) > 0


def test_hallucination_checker_flags_rating_conflict(mock_groq_client):
    profile = SocialProfile(brand="Test Brand")
    profile.gm_rating = 5.0
    profile.ta_rating = 2.0  # diverges by 3.0 > 1.5 threshold

    checker = HallucinationChecker(profile, groq_client=mock_groq_client)
    checker._run_rules()

    conflict_flags = [f for f in checker.flags if f.field_name == "rating_consistency"]
    assert len(conflict_flags) > 0


def test_hallucination_checker_location_in_prompt(mock_groq_client):
    """Location should appear in the LLM validation prompt."""
    profile = SocialProfile(brand="Test Brand")
    captured_prompts = []

    with patch("agents.agent2_social_scraper.query_llm") as mock_q:
        mock_q.return_value = {
            "raw_response": '{"issues":[],"llm_confidence":1.0,"summary":"ok"}',
            "total_tokens": 30,
        }
        checker = HallucinationChecker(profile, groq_client=mock_groq_client,
                                       location="Morocco")
        checker._llm_validate()
        prompts = [str(c) for c in mock_q.call_args_list]

    assert any("Morocco" in p for p in prompts)


# ── IG handle plausibility ────────────────────────────────────────────────────

def test_ig_handle_plausible_match():
    assert _ig_handle_plausible("dareljeld", "Dar El Jeld") is True


def test_ig_handle_plausible_partial():
    assert _ig_handle_plausible("dareljeld_tunis", "Dar El Jeld") is True


def test_ig_handle_blacklisted():
    assert _ig_handle_plausible("explore", "My Brand") is False


def test_ig_handle_empty():
    assert _ig_handle_plausible("", "My Brand") is False


# ── Agent2 location propagation ───────────────────────────────────────────────

def test_location_propagates_from_agent_class():
    """Agent2SocialScraper should store and expose the location param."""
    agent = Agent2SocialScraper(
        brands=["Test Brand"],
        location="Morocco",
    )
    assert agent.location == "Morocco"


def test_agent2_default_location_is_tunisia():
    agent = Agent2SocialScraper(brands=["Test Brand"])
    assert agent.location == "Tunisia"
