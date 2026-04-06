"""
Final/tests/test_pipeline.py
─────────────────────────────
Integration-style tests for the Final/ pipeline.
All agent and LLM calls are mocked so no real API access is needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sample_prompt_df():
    return pd.DataFrame([
        {"prompt_id": "p1", "intent_id": "i1", "language": "fr",
         "variant_id": "v1", "prompt_text": "Meilleurs restaurants?"},
    ])


def _make_global_df():
    return pd.DataFrame([
        {"canonical_entity": "Dar El Jeld",
         "mention_rate": 0.8, "stability_score": 0.75,
         "consistency_label": "stable", "mention_count": 3},
    ])


# ── build_graph smoke test ────────────────────────────────────────────────────

def test_build_graph_returns_compiled_graph():
    """build_graph() should return a compiled LangGraph StateGraph without error."""
    from graph.pipeline_graph import build_graph
    graph = build_graph()
    assert graph is not None


# ── GEOState keys ─────────────────────────────────────────────────────────────

def test_geo_state_has_required_keys():
    """GEOState TypedDict should declare all pipeline-critical fields."""
    from graph.state import GEOState
    import typing
    hints = typing.get_type_hints(GEOState)
    required_keys = ["query", "domain", "location", "prompts", "raw_responses",
                     "brands", "profiles"]
    for key in required_keys:
        assert key in hints, f"GEOState missing field: {key}"


# ── agent0_node smoke test ────────────────────────────────────────────────────

def test_agent0_node_returns_prompt_df(tmp_path):
    """agent0_node should return a prompt_df in state."""
    sample_df = _make_sample_prompt_df()

    with patch("graph.nodes.agent0_node.agent0_run", return_value=sample_df):
        from graph.nodes.agent0_node import agent0_node
        state = {
            "domain": "Tunisian restaurants", "location": "Tunisia",
            "n_intents": 2, "n_variants": 2, "languages": ["fr"],
            "analyst_model": "llama-3.3-70b-versatile",
            "output_dir": str(tmp_path), "groq_client": None,
        }
        result = agent0_node(state)

    assert "prompt_df" in result
    assert isinstance(result["prompt_df"], pd.DataFrame)
    assert len(result["prompt_df"]) >= 1


# ── parse_query_node smoke test ───────────────────────────────────────────────

def test_parse_query_node_extracts_domain_and_location():
    """parse_query_node should return domain and location from query string."""
    with patch("graph.nodes.parse_node.parse_query",
               return_value={"domain": "Cafes", "location": "Tunis"}):
        from graph.nodes.parse_node import parse_query_node
        result = parse_query_node({"query": "Cafes in Tunis", "groq_client": None})

    assert result["domain"]   == "Cafes"
    assert result["location"] == "Tunis"


# ── supervisor integration ────────────────────────────────────────────────────

def test_supervisor0_node_continues_on_good_prompts(monkeypatch):
    """supervisor0_node should set current_step=supervisor0_ok on good output."""
    monkeypatch.setattr("graph.nodes.supervisor_nodes.call_llm_json",
        lambda *a, **kw: {"decision": "continue", "quality_score": 8, "reason": "good"})

    from graph.nodes.supervisor_nodes import supervisor0_node
    state = {
        "prompts": [
            {"prompt_id": i, "prompt_text": f"p {i}", "language": "fr", "intent_id": "i1"}
            for i in range(6)
        ],
        "agent0_retries": 0, "max_retries": 2, "supervisor_notes": [],
        "domain": "Tunisian restaurants", "n_intents": 4, "languages": ["fr"],
    }
    result = supervisor0_node(state)
    assert result.get("current_step") == "supervisor0_ok"
    assert result.get("agent0_quality_score", 0) >= 7


def test_supervisor1_node_continues_on_good_entities(monkeypatch):
    """supervisor1_node should set current_step=supervisor1_ok on good entity output."""
    monkeypatch.setattr("graph.nodes.supervisor_nodes.call_llm_json",
        lambda *a, **kw: {"decision": "continue", "quality_score": 8, "reason": "good"})

    from graph.nodes.supervisor_nodes import supervisor1_node
    state = {
        "entity_features_global": [
            {"canonical_entity": f"brand_{i}", "mention_rate": 0.7,
             "stability_score": 0.6, "consistency_label": "stable"}
            for i in range(6)
        ],
        "prompts": [{"prompt_id": i} for i in range(4)],
        "agent1_retries": 0, "max_retries": 2, "supervisor_notes": [],
        "domain": "Tunisian restaurants",
    }
    result = supervisor1_node(state)
    assert result.get("current_step") == "supervisor1_ok"
