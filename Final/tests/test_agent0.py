"""
Final/tests/test_agent0.py
──────────────────────────
Unit tests for agents/agent0.py (Final/ version).
Adapted from Project 2's tests/test_agent0.py — uses agent0_run() function.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd
import pytest

from agents.agent0 import agent0_run


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_intents(n=2):
    return [
        {"intent_id": f"i{i}", "intent_name": f"Intent {i}",
         "description": f"Description for intent {i}"}
        for i in range(n)
    ]


def _make_prompts(n_variants=3, languages=None):
    langs = languages or ["fr"]
    rows = []
    for lang in langs:
        for v in range(n_variants):
            rows.append({"prompt_id": f"{lang}_v{v}", "intent_id": "i0",
                         "language": lang, "variant_id": f"v{v}",
                         "prompt_text": f"Prompt variant {v} in {lang}"})
    return rows


@contextmanager
def _mock_agent0_internals(n_variants=2, languages=None):
    """Patch all three agent0 sub-functions with canned high-quality responses."""
    sample_df = pd.DataFrame(_make_prompts(n_variants=n_variants, languages=languages))
    with patch("agents.agent0.agent0_generate_intents", return_value=_make_intents(2)), \
         patch("agents.agent0.agent0_generate_prompts", return_value=sample_df), \
         patch("agents.agent0.agent0_reflect",
               return_value={"quality_score": 9, "approved": True, "issues": []}):
        yield sample_df


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_agent0_run_returns_dataframe():
    """agent0_run() should return a DataFrame."""
    with _mock_agent0_internals():
        df = agent0_run(domain="Tunisian restaurants", languages=["fr"],
                        n_intents=2, n_variants=2, max_reflection_loops=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_agent0_run_has_required_columns():
    """Returned DataFrame must include prompt_id, intent_id, language, prompt_text."""
    with _mock_agent0_internals():
        df = agent0_run(domain="Tunisian restaurants", languages=["fr"],
                        n_intents=2, n_variants=2, max_reflection_loops=1)
    for col in ["prompt_id", "intent_id", "language", "prompt_text"]:
        assert col in df.columns, f"Missing column: {col}"


def test_agent0_run_default_languages():
    """When languages=None, agent0_run should default to ['fr', 'ar']."""
    with _mock_agent0_internals(languages=["fr", "ar"]):
        df = agent0_run(domain="Tunisian restaurants", languages=None,
                        n_intents=2, n_variants=2, max_reflection_loops=1)
    assert isinstance(df, pd.DataFrame)


def test_agent0_run_reflection_not_looped_on_high_score():
    """When quality_score >= threshold, no extra loops should be triggered."""
    sample_df = pd.DataFrame(_make_prompts(n_variants=2))
    call_count = {"n": 0}

    def count_intents(*a, **kw):
        call_count["n"] += 1
        return _make_intents(2)

    with patch("agents.agent0.agent0_generate_intents", side_effect=count_intents), \
         patch("agents.agent0.agent0_generate_prompts", return_value=sample_df), \
         patch("agents.agent0.agent0_reflect",
               return_value={"quality_score": 9, "approved": True, "issues": []}):
        agent0_run(domain="Tunisian restaurants", languages=["fr"],
                   n_intents=2, n_variants=2, max_reflection_loops=3)

    assert call_count["n"] == 1
