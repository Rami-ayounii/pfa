"""
core/memory.py
══════════════
Persistent knowledge base for the GEO pipeline.

Uses a single JSON file (geo_output/geo_knowledge.json) — human-readable,
inspectable with any text editor, no external dependencies required.

Structure of geo_knowledge.json:
{
  "entities": {
    "<domain>|<location>": {
      "<entity_name>": {
        "mention_count": int,
        "confirmed_real": bool,
        "updated_at": "YYYY-MM-DD"
      }
    }
  },
  "entity_decisions": {
    "<md5_hash>": {
      "entity_a": str, "entity_b": str, "domain": str,
      "decision": "SAME"|"DIFFERENT",
      "canonical": str, "confidence": "high"|"low",
      "used_count": int, "updated_at": "YYYY-MM-DD"
    }
  },
  "run_history": [
    {
      "run_id": str, "timestamp": str, "domain": str, "location": str,
      "params": dict, "quality_score": float,
      "entity_count": int, "tokens_used": int
    }
  ]
}

Atomic writes: data is written to <path>.tmp first, then os.replace() is
used to swap it in — prevents file corruption on crash.

Thread safety: all reads happen at __init__; writes are synchronous and
protected by _save() which is only called at the end of a sequential run.
No concurrent-write safety is needed because the pipeline is single-process.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import hashlib
from datetime import date, datetime
from typing import Optional
from loguru import logger

DEFAULT_KB_PATH = os.environ.get(
    "KB_PATH", os.path.join("geo_output", "geo_knowledge.json")
)

_EMPTY_DATA: dict = {
    "entities":         {},
    "entity_decisions": {},
    "run_history":      [],
}


class KnowledgeBase:
    """
    JSON-backed persistent knowledge store for the GEO pipeline.

    Three top-level sections:
      entities         — known entity names with metadata per (domain, location)
      entity_decisions — cached LLM arbitration decisions for entity pairs
      run_history      — per-run metadata and quality scores

    Usage:
        kb = KnowledgeBase()          # load from disk (or start fresh)
        kb.save_entity_decision(...)  # write to in-memory dict
        kb.close()                    # flush to disk (atomic write)
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self, kb_path: str = DEFAULT_KB_PATH):
        self._path = kb_path
        self._data: dict = {
            "entities":         {},
            "entity_decisions": {},
            "run_history":      [],
        }
        self._load()

    def _load(self) -> None:
        """Load JSON from disk; silently start fresh if file is absent or corrupt."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Merge keys so newly added sections are always present
            for key in _EMPTY_DATA:
                if key in loaded:
                    self._data[key] = loaded[key]
            logger.debug(f"[KB] Loaded from {self._path} — "
                         f"{len(self._data['entity_decisions'])} decisions, "
                         f"{len(self._data['run_history'])} runs")
        except Exception as exc:
            logger.warning(f"[KB] Could not load {self._path}: {exc} — starting fresh")

    def _save(self) -> None:
        """Atomically write JSON to disk: write .tmp → os.replace()."""
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
            logger.debug(f"[KB] Saved to {self._path}")
        except Exception as exc:
            logger.warning(f"[KB] Save failed: {exc}")

    def close(self) -> None:
        """Flush in-memory data to disk. Call once at the end of a pipeline run."""
        self._save()

    # ── Entity pair decisions ─────────────────────────────────────────────────

    @staticmethod
    def _pair_hash(entity_a: str, entity_b: str, domain: str) -> str:
        """
        Deterministic, order-independent MD5 hash for an entity pair.
        Sorting ensures (A, B) and (B, A) map to the same key.
        """
        key = "|".join(sorted([entity_a.lower(), entity_b.lower()])) \
              + "|" + domain.lower()
        return hashlib.md5(key.encode()).hexdigest()

    def recall_entity_decision(
        self,
        entity_a: str,
        entity_b: str,
        domain: str,
    ) -> Optional[dict]:
        """
        Return a cached arbitration decision for this entity pair, or None.

        Policy: only HIGH-confidence decisions are replayed.
        Low-confidence entries are stored for auditability but never reused —
        ambiguous pairs benefit from fresh LLM judgment on every run.
        """
        key = self._pair_hash(entity_a, entity_b, domain)
        entry = self._data["entity_decisions"].get(key)
        if entry is None:
            return None
        if entry.get("confidence") != "high":
            return None
        # Increment usage counter (in-memory; persisted on close())
        entry["used_count"] = entry.get("used_count", 0) + 1
        return entry

    def save_entity_decision(
        self,
        entity_a: str,
        entity_b: str,
        domain: str,
        decision: str,
        canonical: str,
        confidence: str,
    ) -> None:
        """
        Upsert an entity pair decision.

        decision   : "SAME" or "DIFFERENT"
        canonical  : canonical entity name (populated when decision == "SAME")
        confidence : "high" or "low"
        """
        key = self._pair_hash(entity_a, entity_b, domain)
        existing = self._data["entity_decisions"].get(key, {})
        self._data["entity_decisions"][key] = {
            "entity_a":  entity_a.lower(),
            "entity_b":  entity_b.lower(),
            "domain":    domain.lower(),
            "decision":  decision,
            "canonical": canonical,
            "confidence": confidence,
            "used_count": existing.get("used_count", 0),
            "updated_at": str(date.today()),
        }

    # ── Entities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _entity_key(domain: str, location: str) -> str:
        return f"{domain.lower()}|{location.lower()}"

    def recall_known_entities(self, domain: str, location: str) -> list:
        """
        Return a list of confirmed entity names for this (domain, location).
        Only returns entities with confirmed_real == True.
        """
        bucket = self._data["entities"].get(self._entity_key(domain, location), {})
        return [
            name for name, meta in bucket.items()
            if meta.get("confirmed_real", True)
        ]

    def save_entities(
        self,
        entities: list,
        domain: str,
        location: str,
    ) -> None:
        """
        Upsert entity records for (domain, location).
        Increments mention_count on repeated entries.
        """
        key = self._entity_key(domain, location)
        if key not in self._data["entities"]:
            self._data["entities"][key] = {}
        bucket = self._data["entities"][key]
        today = str(date.today())
        for entity in entities:
            name = entity.strip().lower()
            if not name:
                continue
            if name in bucket:
                bucket[name]["mention_count"] = bucket[name].get("mention_count", 1) + 1
                bucket[name]["updated_at"]    = today
            else:
                bucket[name] = {
                    "mention_count": 1,
                    "confirmed_real": True,
                    "updated_at": today,
                }

    # ── Run history ───────────────────────────────────────────────────────────

    def save_run(
        self,
        run_id: str,
        domain: str,
        location: str,
        params: dict,
        quality_score: float,
        entity_count: int,
        tokens_used: int,
    ) -> None:
        """
        Append a run record to run_history.
        Raises ValueError if run_id already exists (duplicate run).
        """
        existing_ids = {r["run_id"] for r in self._data["run_history"]}
        if run_id in existing_ids:
            raise ValueError(f"[KB] Duplicate run_id: {run_id}")
        self._data["run_history"].append({
            "run_id":       run_id,
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "domain":       domain,
            "location":     location,
            "params":       params,
            "quality_score": round(float(quality_score), 4),
            "entity_count": entity_count,
            "tokens_used":  tokens_used,
        })

    def get_best_params(self, domain: str) -> Optional[dict]:
        """
        Return the `params` dict from the highest-quality run for this domain.
        Returns None if no history exists for the domain.
        """
        runs = [
            r for r in self._data["run_history"]
            if r.get("domain", "").lower() == domain.lower()
        ]
        if not runs:
            return None
        best = max(runs, key=lambda r: r.get("quality_score", 0.0))
        return best.get("params")

    def get_run_history(self, domain: str = "", limit: int = 10) -> list:
        """
        Return the most recent run records (optionally filtered by domain).
        Sorted newest-first.
        """
        runs = self._data["run_history"]
        if domain:
            runs = [r for r in runs if r.get("domain", "").lower() == domain.lower()]
        runs = sorted(runs, key=lambda r: r.get("timestamp", ""), reverse=True)
        return runs[:limit]
