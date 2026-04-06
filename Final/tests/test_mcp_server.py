"""
Final/tests/test_mcp_server.py
================================
Tests that MCP server tools in Final/mcp_servers/ always return expected types
and that the tool functions are importable and callable without real API keys.

Final/ has three MCP servers:
  - mcp_servers/scrape_server.py  (scrape_instagram, scrape_facebook, extract_website_socials)
  - mcp_servers/search_server.py  (google_maps_search, tripadvisor_search, ddg_search)
  - mcp_servers/wiki_server.py    (wikipedia_summary, etc.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── scrape_server ─────────────────────────────────────────────────────────────

def test_scrape_instagram_returns_dict_without_key(monkeypatch):
    """scrape_instagram should return a dict even when APIFY_API_TOKEN is absent."""
    monkeypatch.setenv("APIFY_API_TOKEN", "")
    from mcp_servers.scrape_server import scrape_instagram
    result = scrape_instagram("testhandle")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "ig_followers" in result


def test_scrape_instagram_no_handle_returns_dict(monkeypatch):
    """Empty handle should return a default dict, not raise."""
    monkeypatch.setenv("APIFY_API_TOKEN", "")
    from mcp_servers.scrape_server import scrape_instagram
    result = scrape_instagram("")
    assert isinstance(result, dict)
    assert result.get("ig_followers", 0) == 0


def test_scrape_facebook_returns_dict_without_key(monkeypatch):
    """scrape_facebook should return a dict even when API token is absent."""
    monkeypatch.setenv("APIFY_API_TOKEN", "")
    from mcp_servers.scrape_server import scrape_facebook
    result = scrape_facebook("testpage")
    assert isinstance(result, dict)
    assert "fb_followers" in result or "status" in result


def test_extract_website_socials_returns_dict():
    """extract_website_socials should return a dict for any URL."""
    from mcp_servers.scrape_server import extract_website_socials
    result = extract_website_socials("https://example.com")
    assert isinstance(result, dict)


# ── search_server ─────────────────────────────────────────────────────────────

def test_google_maps_search_no_key_returns_dict(monkeypatch):
    """google_maps_search should return a dict with status='no_key' when SERPAPI_KEY absent."""
    monkeypatch.setenv("SERPAPI_KEY", "")
    from mcp_servers.search_server import google_maps_search
    result = google_maps_search("Dar El Jeld", "Tunisia")
    assert isinstance(result, dict)
    assert result.get("status") == "no_key"


def test_tripadvisor_search_no_key_returns_dict(monkeypatch):
    """tripadvisor_search should return a dict even without API keys."""
    monkeypatch.setenv("SERPAPI_KEY", "")
    from mcp_servers.search_server import tripadvisor_search
    result = tripadvisor_search("Dar El Jeld")
    assert isinstance(result, dict)


def test_ddg_search_returns_list():
    """ddg_search should return a list (possibly empty) for any query."""
    from mcp_servers.search_server import ddg_search
    # Mock DDGS to avoid real network call
    from unittest.mock import patch, MagicMock
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=False)
    mock_ddgs.text.return_value = [{"href": "https://example.com", "body": "text"}]

    with patch("mcp_servers.search_server.DDGS", return_value=mock_ddgs):
        result = ddg_search("Tunisian restaurants", max_results=3)
    assert isinstance(result, list)


# ── Tool presence and callability checks ─────────────────────────────────────

def test_no_tool_returns_none():
    """All expected tools should be importable, non-None, and callable."""
    import mcp_servers.scrape_server as ss
    import mcp_servers.search_server as se

    for name in ["scrape_instagram", "scrape_facebook", "extract_website_socials"]:
        fn = getattr(ss, name, None)
        assert fn is not None and callable(fn), \
            f"Tool {name} missing or not callable in scrape_server"

    for name in ["google_maps_search", "tripadvisor_search", "ddg_search"]:
        fn = getattr(se, name, None)
        assert fn is not None and callable(fn), \
            f"Tool {name} missing or not callable in search_server"
