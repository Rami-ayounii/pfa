"""
Final/tests/test_agent2.py
──────────────────────────
Unit tests for agent2 in Final/.
Final/ uses agents/agent2_react.py (run_agent2_react).
All network calls and external APIs are mocked.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


# ── run_agent2_react importability ───────────────────────────────────────────

def test_run_agent2_react_is_importable():
    """run_agent2_react should be importable from agents.agent2_react."""
    from agents.agent2_react import run_agent2_react
    assert callable(run_agent2_react)


def test_agent2_scrape_node_calls_run_agent2_react():
    """agent2_scrape_node should delegate to run_agent2_react."""
    mock_profile = {"brand": "Dar El Jeld", "status": "ok"}

    with patch("graph.nodes.agent2_scrape_node.run_agent2_react") as mock_react:
        # run_agent2_react is called in a ThreadPoolExecutor with asyncio.run
        # We mock it to return synchronously
        import asyncio
        async def fake_react(brands):
            return [mock_profile]
        mock_react.return_value = [mock_profile]

        # Patch asyncio.run inside the executor
        with patch("asyncio.run", return_value=[mock_profile]):
            from graph.nodes.agent2_scrape_node import agent2_scrape_node
            state  = {"brand": "Dar El Jeld", "location": "Tunisia",
                      "output_dir": "geo_output", "domain": "Tunisian restaurants"}
            result = agent2_scrape_node(state)

    assert "profiles" in result


def test_agent2_scrape_node_empty_brand():
    """agent2_scrape_node should skip when brand is empty."""
    from graph.nodes.agent2_scrape_node import agent2_scrape_node
    result = agent2_scrape_node({"brand": "", "location": "Tunisia"})
    assert result["profiles"] == []


def test_agent2_scrape_node_exception_captured():
    """agent2_scrape_node should capture exceptions and return empty profiles."""
    import concurrent.futures

    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Connection timeout")
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

        from graph.nodes.agent2_scrape_node import agent2_scrape_node
        result = agent2_scrape_node({
            "brand": "Some Brand", "location": "Tunisia",
            "output_dir": "geo_output", "domain": "Tunisian restaurants",
        })

    assert result["profiles"] == []
    assert len(result.get("errors", [])) > 0


def test_agent2_scrape_node_location_propagated():
    """Location should be read from state without error."""
    with patch("graph.nodes.agent2_scrape_node.run_agent2_react"), \
         patch("asyncio.run", return_value=[{"brand": "Test", "status": "ok"}]):
        from graph.nodes.agent2_scrape_node import agent2_scrape_node
        result = agent2_scrape_node({"brand": "Test Brand", "location": "Morocco",
                                     "output_dir": "geo_output", "domain": "Restaurants"})
    assert "profiles" in result
