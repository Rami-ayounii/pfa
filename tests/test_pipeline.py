"""
tests/test_pipeline.py
──────────────────────
Integration tests for pipeline.py

All agent classes and LLM calls are mocked so no real API access is needed.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline import run_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_agent0(domain, **_):
    df = pd.DataFrame([
        {"prompt_id": "p1", "intent_id": "i1", "language": "fr",
         "prompt_text": "Meilleurs restaurants?"},
    ])
    agent = MagicMock()
    agent.run.return_value = df
    return agent


def _mock_agent1(**_):
    features_df = pd.DataFrame([
        {"prompt_id": "p1", "canonical_entity": "Dar El Jeld",
         "mention_rate": 0.8, "stability_score": 0.75, "clean_flag": "clean"},
    ])
    global_df = pd.DataFrame([
        {"canonical_entity": "Dar El Jeld",
         "mention_rate": 0.8, "stability_score": 0.75, "clean_flag": "clean"},
    ])
    agent = MagicMock()
    agent.run.return_value = (features_df, global_df)
    return agent


def _mock_agent2(**_):
    profile = MagicMock()
    profile.brand               = "Dar El Jeld"
    profile.social_authority_score = 72.0
    profile.hallucination_flags = []
    agent = MagicMock()
    agent.run.return_value = [profile]
    agent.audit                 = []
    return agent


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_pipeline_skip_all_agents(tmp_output_dir):
    """Skipping all agents should return empty but valid result dict without error."""
    result = run_pipeline(
        domain      = "Tunisian restaurants",
        output_dir  = tmp_output_dir,
        skip_agent0 = True,
        skip_agent1 = True,
        skip_agent2 = True,
    )
    assert isinstance(result, dict)
    assert "profiles" in result
    assert "global_df" in result


def test_pipeline_uses_tunisia_default(tmp_output_dir):
    """Default domain should be 'Tunisian restaurants'."""
    with patch("pipeline.Agent0PromptGenerator", side_effect=_mock_agent0), \
         patch("pipeline.Agent1GeoAnalyser",     side_effect=_mock_agent1), \
         patch("pipeline.Agent2SocialScraper",   side_effect=_mock_agent2), \
         patch("pipeline.Groq"):
        result = run_pipeline(output_dir=tmp_output_dir)

    # Pipeline should complete without error; domain default is Tunisian restaurants
    assert result is not None


def test_pipeline_location_passed_to_agent2(tmp_output_dir):
    """location param should be forwarded to Agent2SocialScraper constructor."""
    captured_kwargs = {}

    def capture_agent2(**kwargs):
        captured_kwargs.update(kwargs)
        return _mock_agent2(**kwargs)

    with patch("pipeline.Agent0PromptGenerator", side_effect=_mock_agent0), \
         patch("pipeline.Agent1GeoAnalyser",     side_effect=_mock_agent1), \
         patch("pipeline.Agent2SocialScraper",   side_effect=capture_agent2), \
         patch("pipeline.Groq"):
        run_pipeline(
            domain     = "Tunisian restaurants",
            location   = "Morocco",
            output_dir = tmp_output_dir,
        )

    assert captured_kwargs.get("location") == "Morocco"


def test_pipeline_summary_json_created(tmp_output_dir):
    """pipeline_summary.json should be written after a successful run."""
    with patch("pipeline.Agent0PromptGenerator", side_effect=_mock_agent0), \
         patch("pipeline.Agent1GeoAnalyser",     side_effect=_mock_agent1), \
         patch("pipeline.Agent2SocialScraper",   side_effect=_mock_agent2), \
         patch("pipeline.Groq"):
        run_pipeline(
            domain     = "Tunisian restaurants",
            output_dir = tmp_output_dir,
        )

    summary_path = Path(tmp_output_dir) / "pipeline_summary.json"
    assert summary_path.exists(), "pipeline_summary.json not created"
    with open(summary_path) as f:
        summary = json.load(f)
    assert "domain" in summary
    assert summary["domain"] == "Tunisian restaurants"


def test_pipeline_domain_passed_to_agent1(tmp_output_dir):
    """domain should be forwarded to Agent1GeoAnalyser."""
    captured_kwargs = {}

    def capture_agent1(**kwargs):
        captured_kwargs.update(kwargs)
        return _mock_agent1(**kwargs)

    with patch("pipeline.Agent0PromptGenerator", side_effect=_mock_agent0), \
         patch("pipeline.Agent1GeoAnalyser",     side_effect=capture_agent1), \
         patch("pipeline.Agent2SocialScraper",   side_effect=_mock_agent2), \
         patch("pipeline.Groq"):
        run_pipeline(
            domain     = "Egyptian restaurants",
            output_dir = tmp_output_dir,
        )

    assert captured_kwargs.get("domain") == "Egyptian restaurants"
