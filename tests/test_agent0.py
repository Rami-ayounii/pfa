"""
tests/test_agent0.py
────────────────────
Unit tests for agents/agent0_prompt_generator.py
"""

import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.agent0_prompt_generator import Agent0PromptGenerator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_intent_response(intents=None):
    """Fake LLM response for intent discovery step."""
    intents = intents or [
        {"intent_id": "i1", "intent_label": "Best restaurants"},
        {"intent_id": "i2", "intent_label": "Must-try dishes"},
    ]
    return {"raw_response": json.dumps(intents), "total_tokens": 60}


def _make_prompt_response(n=3):
    """Fake LLM response for prompt generation step."""
    prompts = [f"Prompt variant {i+1}" for i in range(n)]
    return {"raw_response": json.dumps(prompts), "total_tokens": 80}


def _make_reflection_response(score=9):
    """Fake LLM reflection response."""
    payload = {"quality_score": score, "proceed": score >= 7, "issues": []}
    return {"raw_response": json.dumps(payload), "total_tokens": 40}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_returns_dataframe(tmp_output_dir, mock_groq_client):
    """Agent0.run() should return a DataFrame with required columns."""
    with patch("agents.agent0_prompt_generator.query_llm") as mock_q:
        mock_q.side_effect = [
            _make_intent_response(),    # intent discovery
            _make_prompt_response(3),   # prompts for intent i1
            _make_prompt_response(3),   # prompts for intent i2
            _make_reflection_response(9),  # reflection — pass
        ]
        agent = Agent0PromptGenerator(
            domain="Tunisian restaurants",
            languages=["fr"],
            n_intents=2,
            n_variants=3,
            max_reflection_loops=1,
            output_dir=tmp_output_dir,
            groq_client=mock_groq_client,
        )
        df = agent.run()

    assert isinstance(df, pd.DataFrame)
    for col in ["prompt_id", "intent_id", "language", "prompt_text"]:
        assert col in df.columns, f"Missing column: {col}"


def test_default_language_is_fr(tmp_output_dir, mock_groq_client):
    """When languages=None, Agent0 should default to French."""
    with patch("agents.agent0_prompt_generator.query_llm") as mock_q:
        mock_q.side_effect = [
            _make_intent_response(),
            _make_prompt_response(2),
            _make_prompt_response(2),
            _make_reflection_response(9),
        ]
        agent = Agent0PromptGenerator(
            domain="Tunisian restaurants",
            languages=None,  # should default to ["fr"]
            n_intents=2,
            n_variants=2,
            max_reflection_loops=1,
            output_dir=tmp_output_dir,
            groq_client=mock_groq_client,
        )
        df = agent.run()

    assert (df["language"] == "fr").all()


def test_output_file_created(tmp_output_dir, mock_groq_client):
    """A CSV file should be written to output_dir after run()."""
    with patch("agents.agent0_prompt_generator.query_llm") as mock_q:
        mock_q.side_effect = [
            _make_intent_response(),
            _make_prompt_response(2),
            _make_prompt_response(2),
            _make_reflection_response(9),
        ]
        agent = Agent0PromptGenerator(
            domain="Tunisian restaurants",
            languages=["fr"],
            n_intents=2,
            n_variants=2,
            max_reflection_loops=1,
            output_dir=tmp_output_dir,
            groq_client=mock_groq_client,
        )
        agent.run()

    csv_files = list(Path(tmp_output_dir).glob("prompt_set_*.csv"))
    assert len(csv_files) == 1


def test_n_intents_n_variants_respected(tmp_output_dir, mock_groq_client):
    """DataFrame should have n_intents × n_variants rows (for one language)."""
    n_intents, n_variants = 2, 3
    with patch("agents.agent0_prompt_generator.query_llm") as mock_q:
        mock_q.side_effect = [
            _make_intent_response(
                [{"intent_id": f"i{i}", "intent_label": f"Intent {i}"}
                 for i in range(n_intents)]
            ),
            *[_make_prompt_response(n_variants) for _ in range(n_intents)],
            _make_reflection_response(9),
        ]
        agent = Agent0PromptGenerator(
            domain="Tunisian restaurants",
            languages=["fr"],
            n_intents=n_intents,
            n_variants=n_variants,
            max_reflection_loops=1,
            output_dir=tmp_output_dir,
            groq_client=mock_groq_client,
        )
        df = agent.run()

    assert len(df) == n_intents * n_variants


def test_reflection_loop_skipped_when_quality_ok(tmp_output_dir, mock_groq_client):
    """When quality_score >= threshold, no extra reflection calls should be made."""
    with patch("agents.agent0_prompt_generator.query_llm") as mock_q:
        calls = [
            _make_intent_response(),
            _make_prompt_response(2),
            _make_prompt_response(2),
            _make_reflection_response(score=9),  # high score — should not loop
        ]
        mock_q.side_effect = calls
        agent = Agent0PromptGenerator(
            domain="Tunisian restaurants",
            languages=["fr"],
            n_intents=2,
            n_variants=2,
            max_reflection_loops=2,
            output_dir=tmp_output_dir,
            groq_client=mock_groq_client,
        )
        agent.run()

    # Verify we didn't call the LLM more than needed (1 intent + 2 prompt + 1 reflect = 4)
    assert mock_q.call_count <= 5
