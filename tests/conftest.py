"""
tests/conftest.py
─────────────────
Shared fixtures and mocks for the GEO multi-agent test suite.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── LLM response helpers ──────────────────────────────────────────────────────

def _make_llm_resp(content: str, tokens: int = 50) -> dict:
    return {
        "raw_response":      content,
        "completion_tokens": tokens,
        "prompt_tokens":     tokens,
        "total_tokens":      tokens * 2,
    }


@pytest.fixture
def llm_entity_response():
    """Returns a valid entity extraction response."""
    return _make_llm_resp('["Dar El Jeld", "Le Corsaire", "Dar Zarrouk"]')


@pytest.fixture
def llm_json_response():
    """Returns a generic JSON object response."""
    return _make_llm_resp('{"quality_score": 9, "proceed": true, "suspicious": []}')


@pytest.fixture
def mock_groq_client():
    """A MagicMock Groq client that returns a plausible completion object."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = '["Brand A", "Brand B"]'
    usage = MagicMock()
    usage.completion_tokens = 20
    usage.prompt_tokens = 30
    usage.total_tokens = 50
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    client.chat.completions.create.return_value = resp
    return client


# ── DataFrame fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_prompt_df():
    """Minimal prompt_set DataFrame as produced by Agent 0."""
    return pd.DataFrame([
        {"prompt_id": "p1", "intent_id": "i1", "language": "fr",
         "prompt_text": "Quels sont les meilleurs restaurants tunisiens?"},
        {"prompt_id": "p2", "intent_id": "i1", "language": "fr",
         "prompt_text": "Recommandez des restaurants tunisiens populaires."},
        {"prompt_id": "p3", "intent_id": "i2", "language": "fr",
         "prompt_text": "Les restaurants tunisiens les plus visités."},
        {"prompt_id": "p4", "intent_id": "i2", "language": "fr",
         "prompt_text": "Où manger tunisien à Tunis?"},
    ])


@pytest.fixture
def sample_prompt_csv(tmp_path, sample_prompt_df):
    """Saves sample_prompt_df to a CSV file and returns its path."""
    path = tmp_path / "prompt_set_Tunisian_restaurants.csv"
    sample_prompt_df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


@pytest.fixture
def sample_raw_responses():
    """Raw LLM responses DataFrame."""
    return pd.DataFrame([
        {"response_id": "r1", "prompt_id": "p1", "model_id": "m1__run1",
         "response_text": "Dar El Jeld est le meilleur restaurant. Le Corsaire est aussi excellent."},
        {"response_id": "r2", "prompt_id": "p2", "model_id": "m1__run1",
         "response_text": "Je recommande Dar El Jeld et Dar Zarrouk."},
        {"response_id": "r3", "prompt_id": "p3", "model_id": "m1__run1",
         "response_text": "Dar El Jeld, Le Corsaire et Dar Zarrouk sont populaires."},
        {"response_id": "r4", "prompt_id": "p4", "model_id": "m1__run1",
         "response_text": "Le Corsaire est incontournable."},
    ])


@pytest.fixture
def sample_entities():
    """Extracted entities DataFrame."""
    return pd.DataFrame([
        {"response_id": "r1", "entity": "dar el jeld", "ranking_position": 1,
         "brand_raw_text": "Dar El Jeld est le meilleur", "description_length": 20},
        {"response_id": "r1", "entity": "le corsaire",  "ranking_position": 2,
         "brand_raw_text": "Le Corsaire est excellent",   "description_length": 18},
        {"response_id": "r2", "entity": "dar el jeld",  "ranking_position": 1,
         "brand_raw_text": "Dar El Jeld et Dar Zarrouk",  "description_length": 15},
        {"response_id": "r2", "entity": "dar zarrouk",  "ranking_position": 2,
         "brand_raw_text": "Dar Zarrouk",                 "description_length": 10},
        {"response_id": "r3", "entity": "dar el jeld",  "ranking_position": 1,
         "brand_raw_text": "Dar El Jeld",                 "description_length": 10},
        {"response_id": "r3", "entity": "le corsaire",  "ranking_position": 2,
         "brand_raw_text": "Le Corsaire",                 "description_length": 10},
        {"response_id": "r3", "entity": "dar zarrouk",  "ranking_position": 3,
         "brand_raw_text": "Dar Zarrouk",                 "description_length": 10},
        {"response_id": "r4", "entity": "le corsaire",  "ranking_position": 1,
         "brand_raw_text": "Le Corsaire",                 "description_length": 12},
    ])


@pytest.fixture
def tmp_output_dir(tmp_path):
    """A temporary output directory."""
    d = tmp_path / "geo_output"
    d.mkdir()
    return str(d)
