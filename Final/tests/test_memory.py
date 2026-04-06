"""
Final/tests/test_memory.py
===========================
Unit tests for core/memory.py (Final/ version) — KnowledgeBase (JSON persistence).
All tests use tmp_path to avoid touching real geo_output/geo_knowledge.json.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest

from core.memory import KnowledgeBase


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_kb(tmp_path) -> KnowledgeBase:
    """Return a fresh KnowledgeBase backed by a temp JSON file."""
    return KnowledgeBase(kb_path=str(tmp_path / "test_kb.json"))


# ── Schema ────────────────────────────────────────────────────────────────────

class TestKBSchema:
    def test_kb_creates_schema(self, tmp_path):
        """A freshly created KB has the three expected top-level sections."""
        kb = make_kb(tmp_path)
        kb.close()
        with open(str(tmp_path / "test_kb.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert "entities"         in data
        assert "entity_decisions" in data
        assert "run_history"      in data


# ── Entity decisions ──────────────────────────────────────────────────────────

class TestEntityDecisions:
    def test_recall_returns_none_when_empty(self, tmp_path):
        kb = make_kb(tmp_path)
        result = kb.recall_entity_decision("dar el jeld", "dar al jeld", "restaurant")
        assert result is None
        kb.close()

    def test_save_and_recall_high_confidence(self, tmp_path):
        kb = make_kb(tmp_path)
        kb.save_entity_decision(
            "dar el jeld", "dar al jeld", "restaurant",
            decision="SAME", canonical="dar el jeld", confidence="high",
        )
        result = kb.recall_entity_decision("dar el jeld", "dar al jeld", "restaurant")
        assert result is not None
        assert result["decision"]   == "SAME"
        assert result["canonical"]  == "dar el jeld"
        assert result["confidence"] == "high"
        kb.close()

    def test_low_confidence_not_returned(self, tmp_path):
        """Confidence='low' is stored but recall_entity_decision should return None."""
        kb = make_kb(tmp_path)
        kb.save_entity_decision(
            "a", "b", "restaurant",
            decision="DIFFERENT", canonical="", confidence="low",
        )
        result = kb.recall_entity_decision("a", "b", "restaurant")
        assert result is None
        kb.close()

    def test_pair_hash_order_independent(self, tmp_path):
        """(a, b) and (b, a) should resolve to the same cached decision."""
        kb = make_kb(tmp_path)
        kb.save_entity_decision(
            "le corsaire", "le corsaire restaurant", "restaurant",
            decision="SAME", canonical="le corsaire", confidence="high",
        )
        # Recall with arguments in reverse order
        result = kb.recall_entity_decision(
            "le corsaire restaurant", "le corsaire", "restaurant"
        )
        assert result is not None
        assert result["canonical"] == "le corsaire"
        kb.close()


# ── Entity store ──────────────────────────────────────────────────────────────

class TestEntityStore:
    def test_save_and_recall_entities(self, tmp_path):
        kb = make_kb(tmp_path)
        kb.save_entities(["dar el jeld", "le corsaire"], "restaurant", "tunis")
        recalled = kb.recall_known_entities("restaurant", "tunis")
        assert "dar el jeld" in recalled
        assert "le corsaire" in recalled
        kb.close()

    def test_upsert_increments_mention_count(self, tmp_path):
        kb = make_kb(tmp_path)
        kb.save_entities(["dar el jeld"], "restaurant", "tunis")
        kb.save_entities(["dar el jeld"], "restaurant", "tunis")
        kb.close()

        with open(str(tmp_path / "test_kb.json"), encoding="utf-8") as f:
            data = json.load(f)
        key = "restaurant|tunis"
        count = data["entities"][key]["dar el jeld"]["mention_count"]
        assert count == 2


# ── Run history ───────────────────────────────────────────────────────────────

class TestRunHistory:
    def test_save_run_and_get_best_params(self, tmp_path):
        """get_best_params returns the *params* dict from the highest-quality run."""
        kb = make_kb(tmp_path)
        kb.save_run("run_001", "restaurant", "tunis",
                    {"n_runs": 1}, quality_score=6.5, entity_count=8, tokens_used=1000)
        kb.save_run("run_002", "restaurant", "tunis",
                    {"n_runs": 2}, quality_score=9.0, entity_count=15, tokens_used=2000)
        best = kb.get_best_params("restaurant")
        # get_best_params returns the params dict of the highest-quality-score run
        assert best is not None
        assert best.get("n_runs") == 2   # run_002 had quality_score=9.0 (highest)
        kb.close()

    def test_duplicate_run_id_raises(self, tmp_path):
        kb = make_kb(tmp_path)
        kb.save_run("run_dup", "restaurant", "tunis",
                    {}, quality_score=5.0, entity_count=5, tokens_used=500)
        with pytest.raises(ValueError, match="run_id"):
            kb.save_run("run_dup", "restaurant", "tunis",
                        {}, quality_score=5.0, entity_count=5, tokens_used=500)
        kb.close()

    def test_get_run_history_returns_most_recent(self, tmp_path):
        kb = make_kb(tmp_path)
        for i in range(5):
            kb.save_run(f"r_{i}", "restaurant", "tunis",
                        {}, quality_score=float(i), entity_count=i, tokens_used=i * 100)
        history = kb.get_run_history("restaurant", limit=3)
        assert len(history) == 3
        kb.close()
