"""
tests/test_query_parser.py
══════════════════════════
Tests for core/query_parser.py — NL input → {domain, location}.
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_llm_response(domain: str, location: str):
    """Build a fake query_llm() return value for the parser."""
    import json
    return {
        "raw_response": json.dumps({"domain": domain, "location": location}),
        "completion_tokens": 20, "prompt_tokens": 10, "total_tokens": 30,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestQueryParserLLM:
    """LLM-based extraction path."""

    def test_llm_parses_simple_query(self):
        """Should extract domain and location via LLM."""
        with patch("core.query_parser._get_llm_helpers") as mock_helpers:
            mock_query_llm = MagicMock(return_value=_mock_llm_response("Restaurants", "Sfax"))
            mock_parse_json = MagicMock(return_value={"domain": "Restaurants", "location": "Sfax"})
            mock_helpers.return_value = (mock_query_llm, mock_parse_json, "llama-3.1-8b-instant")

            from core.query_parser import parse_query
            result = parse_query("Restaurants in Sfax")

        assert result["domain"]   == "Restaurants"
        assert result["location"] == "Sfax"

    def test_llm_parses_french_query(self):
        """Should handle French-language queries."""
        with patch("core.query_parser._get_llm_helpers") as mock_helpers:
            mock_query_llm = MagicMock(
                return_value=_mock_llm_response("Meilleurs cafés", "Tunis"))
            mock_parse_json = MagicMock(
                return_value={"domain": "Meilleurs cafés", "location": "Tunis"})
            mock_helpers.return_value = (mock_query_llm, mock_parse_json, "llama-3.1-8b-instant")

            from core.query_parser import parse_query
            result = parse_query("Meilleurs cafés à Tunis")

        assert "caf" in result["domain"].lower()
        assert result["location"] == "Tunis"

    def test_llm_failure_falls_through_to_regex(self):
        """When LLM fails, regex fallback should succeed."""
        from core.query_parser import parse_query
        with patch("core.query_parser._get_llm_helpers") as mock_helpers:
            mock_helpers.side_effect = Exception("API down")
            result = parse_query("Sushi spots near Hammamet")

        assert result["location"].lower() == "hammamet"
        assert "sushi" in result["domain"].lower()


class TestQueryParserRegex:
    """Regex fallback path."""

    @pytest.mark.parametrize("query,expected_domain,expected_location", [
        ("Restaurants in Sfax",      "Restaurants",      "Sfax"),
        ("Cafes near Tunis",         "Cafes",            "Tunis"),
        ("Street food around Sousse","Street food",      "Sousse"),
        ("Pizza in Djerba",          "Pizza",            "Djerba"),
    ])
    def test_regex_patterns(self, query, expected_domain, expected_location):
        """Regex should handle common English prepositions."""
        from core.query_parser import parse_query
        with patch("core.query_parser._get_llm_helpers") as mock_helpers:
            mock_helpers.side_effect = Exception("force regex fallback")
            result = parse_query(query)

        assert result["domain"].lower()   == expected_domain.lower()
        assert result["location"].lower() == expected_location.lower()


class TestQueryParserFallback:
    """Final fallback: no location found."""

    def test_empty_query_returns_defaults(self):
        from core.query_parser import parse_query
        result = parse_query("")
        assert result["domain"]   == "restaurants"
        assert result["location"] == "Tunisia"

    def test_query_without_location_defaults_to_tunisia(self):
        """Query with no recognisable location pattern → location='Tunisia'."""
        with patch("core.query_parser._get_llm_helpers") as mock_helpers:
            # LLM returns missing location
            mock_query_llm = MagicMock(return_value={"raw_response": '{"domain":"food"}',
                                                      "completion_tokens": 5,
                                                      "prompt_tokens": 5, "total_tokens": 10})
            mock_parse_json = MagicMock(return_value={"domain": "food"})
            mock_helpers.return_value = (mock_query_llm, mock_parse_json, "llama-3.1-8b-instant")

            from core.query_parser import parse_query
            result = parse_query("Good food")

        assert result["location"] == "Tunisia"
        # Final fallback: domain = raw_query when LLM result lacks 'location' key
        assert result["domain"]   == "Good food"
