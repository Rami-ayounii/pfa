"""
Final/tests/test_graph_nodes.py
================================
Unit tests for individual LangGraph nodes in Final/.
Includes supervisor node tests added for the merged pipeline.
All LLM calls and agent.run() methods are mocked.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# ── parse_query_node ──────────────────────────────────────────────────────────

class TestParseQueryNode:
    def test_node_returns_domain_and_location(self):
        with patch("graph.nodes.parse_node.parse_query") as mock_parse:
            mock_parse.return_value = {"domain": "Restaurants", "location": "Sfax"}

            from graph.nodes.parse_node import parse_query_node
            state  = {"query": "Restaurants in Sfax", "groq_client": None}
            result = parse_query_node(state)

        assert result["domain"]   == "Restaurants"
        assert result["location"] == "Sfax"

    def test_node_uses_groq_client_from_state(self):
        fake_client = MagicMock()
        with patch("graph.nodes.parse_node.parse_query") as mock_parse:
            mock_parse.return_value = {"domain": "Cafes", "location": "Tunis"}

            from graph.nodes.parse_node import parse_query_node
            state  = {"query": "Cafes in Tunis", "groq_client": fake_client}
            parse_query_node(state)

        mock_parse.assert_called_once_with("Cafes in Tunis", groq_client=fake_client)


# ── agent0_node ───────────────────────────────────────────────────────────────

class TestAgent0Node:
    def test_node_returns_prompt_df(self, tmp_path):
        sample_df = pd.DataFrame([{
            "prompt_id": "p1", "intent_id": "i1",
            "language": "fr", "variant_id": "v1",
            "prompt_text": "Quels sont les meilleurs restaurants a Sfax?",
        }])

        with patch("graph.nodes.agent0_node.agent0_run") as mock_agent0:
            mock_agent0.return_value = sample_df

            from graph.nodes.agent0_node import agent0_node
            state = {
                "domain": "Restaurants", "location": "Sfax",
                "n_intents": 2, "n_variants": 2, "languages": ["fr"],
                "analyst_model": "llama-3.3-70b-versatile",
                "output_dir": str(tmp_path), "groq_client": None,
            }
            result = agent0_node(state)

        assert "prompt_df" in result
        assert len(result["prompt_df"]) == 1


# ── agent1_load_node ──────────────────────────────────────────────────────────

class TestAgent1LoadNode:
    def test_load_valid_prompt_df(self):
        df = pd.DataFrame([
            {"prompt_id": "p1", "intent_id": "i1", "language": "fr",
             "prompt_text": "test prompt"},
        ])

        from graph.nodes.agent1_nodes import agent1_load_node
        state  = {"prompt_df": df}
        result = agent1_load_node(state)

        assert "prompts" in result
        assert len(result["prompts"]) == 1
        assert result["prompts"][0]["prompt_id"] == "p1"

    def test_load_missing_prompt_df(self):
        from graph.nodes.agent1_nodes import agent1_load_node
        result = agent1_load_node({"prompt_df": None})

        assert result["prompts"] == []
        assert len(result.get("errors", [])) > 0

    def test_load_missing_required_column(self):
        df = pd.DataFrame([{"prompt_id": "p1"}])   # missing intent_id, language, prompt_text

        from graph.nodes.agent1_nodes import agent1_load_node
        result = agent1_load_node({"prompt_df": df})

        assert result["prompts"] == []
        assert "errors" in result


# ── agent1_aggregate_node ─────────────────────────────────────────────────────

class TestAgent1AggregateNode:
    def test_aggregate_valid_responses(self, tmp_path):
        raw_responses = [
            {"prompt_id": "p1", "model_id": "llama-3.1-8b-instant",
             "run_idx": 1, "prompt_text": "test",
             "response_text": "Some restaurants: Dar El Jeld, Le Corsaire",
             "completion_tokens": 50, "prompt_tokens": 10, "total_tokens": 60},
        ]

        with patch("graph.nodes.agent1_nodes._fire_csv_backup"):
            from graph.nodes.agent1_nodes import agent1_aggregate_node
            state  = {"raw_responses": raw_responses, "output_dir": str(tmp_path)}
            result = agent1_aggregate_node(state)

        df = result["entities_df"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "response_id" in df.columns

    def test_aggregate_empty_responses(self, tmp_path):
        from graph.nodes.agent1_nodes import agent1_aggregate_node
        result = agent1_aggregate_node({"raw_responses": [], "output_dir": str(tmp_path)})

        assert "warnings" in result


# ── agent2_scrape_node ────────────────────────────────────────────────────────

class TestAgent2ScrapeNode:
    def test_scrape_empty_brand_skipped(self):
        from graph.nodes.agent2_scrape_node import agent2_scrape_node
        result = agent2_scrape_node({"brand": "", "location": "Tunis"})
        assert result["profiles"] == []

    def test_scrape_exception_captured_in_errors(self):
        import concurrent.futures

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
            mock_future = MagicMock()
            mock_future.result.side_effect = Exception("Connection timeout")
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

            from graph.nodes.agent2_scrape_node import agent2_scrape_node
            result = agent2_scrape_node({
                "brand": "Some Brand", "location": "Sfax",
                "output_dir": "geo_output", "domain": "Tunisian restaurants",
            })

        assert result["profiles"] == []
        assert len(result.get("errors", [])) > 0


# ── agent1_llm_query_all_node ─────────────────────────────────────────────────

class TestRouteLLMQueries:
    def test_all_node_collects_responses(self):
        """agent1_llm_query_all_node runs models sequentially per prompt."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        mock_result = {"raw_response": "some text", "completion_tokens": 10,
                       "prompt_tokens": 5, "total_tokens": 15}

        with patch("graph.nodes.agent1_llm_query_node.async_query_llm",
                   new=AsyncMock(return_value=mock_result)):
            from graph.nodes.agent1_llm_query_node import agent1_llm_query_all_node
            state = {
                "prompts":      [{"prompt_id": "p1", "prompt_text": "q1"},
                                 {"prompt_id": "p2", "prompt_text": "q2"}],
                "query_models": ["model-a", "model-b"],
                "n_runs":       1,
                "domain":       "Restaurants",
            }
            result = asyncio.run(agent1_llm_query_all_node(state))
            # 2 prompts x 2 models x 1 run = 4 responses
            assert len(result["raw_responses"]) == 4

    def test_all_node_skips_empty_responses(self):
        """agent1_llm_query_all_node drops rows where raw_response is empty."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        empty_result = {"raw_response": "", "completion_tokens": 0,
                        "prompt_tokens": 0, "total_tokens": 0}

        with patch("graph.nodes.agent1_llm_query_node.async_query_llm",
                   new=AsyncMock(return_value=empty_result)):
            from graph.nodes.agent1_llm_query_node import agent1_llm_query_all_node
            state = {
                "prompts":      [{"prompt_id": "p1", "prompt_text": "q1"}],
                "query_models": ["model-a"],
                "n_runs":       1,
                "domain":       "Restaurants",
            }
            result = asyncio.run(agent1_llm_query_all_node(state))
            assert result["raw_responses"] == []


# ── monitor_node ──────────────────────────────────────────────────────────────

class TestMonitorNode:
    """Unit tests for the zero-LLM quality monitor node."""

    def _make_global_df(self, n=12, mention_counts=None, stability_scores=None):
        rows = []
        for i in range(n):
            rows.append({
                "canonical_entity": f"brand_{i}",
                "mention_count":    (mention_counts[i] if mention_counts else 1),
                "stability_score":  (stability_scores[i] if stability_scores else 0.5),
            })
        return pd.DataFrame(rows)

    def test_returns_perfect_score_on_healthy_data(self):
        """12 entities, diversity=1.0, std~0.0 -> quality_score == 10.0"""
        from graph.nodes.monitor_node import monitor_node
        df = self._make_global_df(n=12)
        result = monitor_node({"global_df": df})
        assert result["quality_score"] == pytest.approx(10.0)
        assert result["quality_warnings"] == []

    def test_warns_on_low_entity_count(self):
        """4 entities (< default threshold 10) -> score < 10 and warning emitted."""
        from graph.nodes.monitor_node import monitor_node
        df = self._make_global_df(n=4)
        result = monitor_node({"global_df": df})
        assert result["quality_score"] < 10.0
        assert any("Low entity count" in w for w in result["quality_warnings"])

    def test_handles_none_global_df(self):
        """None global_df -> quality_score == 0.0 with a warning."""
        from graph.nodes.monitor_node import monitor_node
        result = monitor_node({"global_df": None})
        assert result["quality_score"] == pytest.approx(0.0)
        assert len(result["quality_warnings"]) > 0

    def test_handles_empty_global_df(self):
        from graph.nodes.monitor_node import monitor_node
        result = monitor_node({"global_df": pd.DataFrame()})
        assert result["quality_score"] == pytest.approx(0.0)

    def test_penalises_low_diversity(self):
        """Many mentions per entity -> low diversity -> score reduced."""
        from graph.nodes.monitor_node import monitor_node
        df = self._make_global_df(n=12, mention_counts=[100] * 12)
        result = monitor_node({"global_df": df})
        assert result["quality_score"] < 10.0
        assert any("diversity" in w.lower() for w in result["quality_warnings"])

    def test_penalises_high_variance(self):
        """Half entities at 0.0, half at 1.0 -> large std -> score reduced."""
        from graph.nodes.monitor_node import monitor_node
        scores = [0.0] * 6 + [1.0] * 6
        df = self._make_global_df(n=12, stability_scores=scores)
        result = monitor_node({"global_df": df})
        assert result["quality_score"] < 10.0
        assert any("variance" in w.lower() for w in result["quality_warnings"])


# ── route_brand_scraping ──────────────────────────────────────────────────────

class TestRouteBrandScraping:
    def test_sends_one_per_brand(self):
        from graph.pipeline_graph import route_brand_scraping
        state = {
            "brands":   ["Brand A", "Brand B", "Brand C"],
            "location": "Sfax",
            "output_dir": "geo_output",
            "serpapi_key": "", "apify_token": "",
            "openrouter_key": "", "groq_client": None,
            "ig_user": "", "ig_pass": "", "is_colab": False,
        }
        sends = route_brand_scraping(state)
        assert len(sends) == 3
        assert all(s.node == "agent2_scrape_node" for s in sends)

    def test_empty_brands_routes_to_export(self):
        from graph.pipeline_graph import route_brand_scraping
        state = {
            "brands": [],
            "location": "Tunisia",
            "output_dir": "geo_output",
            "serpapi_key": "", "apify_token": "",
            "openrouter_key": "", "groq_client": None,
            "ig_user": "", "ig_pass": "", "is_colab": False,
        }
        sends = route_brand_scraping(state)
        assert len(sends) == 1
        assert sends[0].node == "export_node"


# ── supervisor0_node ──────────────────────────────────────────────────────────

def test_supervisor0_continue(monkeypatch):
    """supervisor0 passes good prompt sets."""
    from graph.nodes.supervisor_nodes import supervisor0_node, route_after_supervisor0
    monkeypatch.setattr("graph.nodes.supervisor_nodes.call_llm_json",
        lambda *a, **kw: {"decision": "continue", "quality_score": 8, "notes": "ok"})
    state = {
        "prompts": [{"prompt_id": i, "prompt_text": f"prompt {i}"} for i in range(6)],
        "agent0_retries": 0, "max_retries": 1, "supervisor_notes": []
    }
    result = supervisor0_node(state)
    assert result.get("agent0_quality_score", 0) >= 7
    route = route_after_supervisor0({**state, **result})
    assert route == "agent1_load_node"


def test_supervisor0_retry(monkeypatch):
    """supervisor0 triggers retry on low quality."""
    from graph.nodes.supervisor_nodes import supervisor0_node, route_after_supervisor0
    monkeypatch.setattr("graph.nodes.supervisor_nodes.call_llm_json",
        lambda *a, **kw: {"decision": "retry", "quality_score": 3, "notes": "low quality"})
    state = {
        "prompts": [{"prompt_id": 0, "prompt_text": "p"}],
        "agent0_retries": 0, "max_retries": 1, "supervisor_notes": []
    }
    result = supervisor0_node(state)
    route = route_after_supervisor0({**state, **result})
    assert route == "agent0_node"


def test_supervisor0_abort_on_empty_prompts(monkeypatch):
    """supervisor0 aborts immediately when no prompts are produced."""
    from graph.nodes.supervisor_nodes import supervisor0_node, route_after_supervisor0
    state = {
        "prompts": [], "agent0_retries": 0, "max_retries": 2, "supervisor_notes": []
    }
    result = supervisor0_node(state)
    route = route_after_supervisor0({**state, **result})
    assert route == "__end__"


def test_supervisor0_max_retries_forces_continue(monkeypatch):
    """When agent0_retries >= max_retries, supervisor0 should not retry (continue instead)."""
    from graph.nodes.supervisor_nodes import supervisor0_node, route_after_supervisor0
    monkeypatch.setattr("graph.nodes.supervisor_nodes.call_llm_json",
        lambda *a, **kw: {"decision": "retry", "quality_score": 2, "reason": "low"})
    state = {
        "prompts": [{"prompt_id": 0, "prompt_text": "p", "language": "fr", "intent_id": "i1"}],
        "agent0_retries": 2, "max_retries": 2, "supervisor_notes": [],
        "domain": "Restaurants", "n_intents": 4, "languages": ["fr"],
    }
    result = supervisor0_node(state)
    route = route_after_supervisor0({**state, **result})
    # Max retries exceeded -> should NOT retry -> goes to agent1_load_node
    assert route == "agent1_load_node"
