"""
core/status_tracker.py
══════════════════════
Lightweight file-based status tracker for the GEO pipeline.

Each graph node calls emit() at start (status='active') and end (status='done').
status_server.py reads pipeline_status.json to power the live tracker dashboard.
"""

import json
import time
import os
import threading
from pathlib import Path

_lock = threading.Lock()

# Canonical stage order — used to determine pipeline completion
STAGE_ORDER = [
    "parse_query",
    "agent0",
    "agent1_load",
    "agent1_query",
    "agent1_aggregate",
    "agent1_extract",
    "agent1_enrich",
    "agent1_clean",
    "agent1_metrics",
    "agent2_scrape",
    "export",
]


def _load(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(path: Path, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass  # never crash the pipeline over a status write


def emit(stage_id: str, status: str, meta: dict | None = None,
         output_dir: str = "geo_output") -> None:
    """
    Update pipeline_status.json with current stage state.

    stage_id : one of STAGE_ORDER
    status   : 'active' | 'done' | 'error'
    meta     : optional dict of extra fields (tokens, model, entities, etc.)
               For 'active' calls, meta is merged — safe to call repeatedly
               for progress updates without resetting start time.
    """
    path = Path(output_dir) / "pipeline_status.json"
    os.makedirs(output_dir, exist_ok=True)

    with _lock:
        data = _load(path)

        if "stages" not in data:
            data["stages"] = {}
        if "started_at" not in data:
            data["started_at"] = time.time()

        stage = data["stages"].setdefault(stage_id, {})

        if status == "active":
            stage["status"] = "active"
            # Preserve start time on repeated calls (progress updates)
            if "started_at" not in stage:
                stage["started_at"] = time.time()
            data["active_stage"]    = stage_id
            data["pipeline_running"] = True
            if meta:
                stage.update(meta)

        elif status in ("done", "error"):
            stage["status"]     = status
            stage["ended_at"]   = time.time()
            stage["duration_s"] = round(
                time.time() - stage.get("started_at", time.time()), 2
            )
            if meta:
                stage.update(meta)

            # Clear active_stage; the next stage sets it via its own 'active' emit.
            # This prevents the tracker from showing the wrong node as active when
            # supervisor/monitor nodes (which don't emit) run between tracked stages.
            if data.get("active_stage") == stage_id:
                data["active_stage"] = None

            # Mark pipeline complete when the last tracked stage finishes
            idx = STAGE_ORDER.index(stage_id) if stage_id in STAGE_ORDER else -1
            if idx == len(STAGE_ORDER) - 1:
                data["pipeline_running"] = False
                data["completed_at"]     = time.time()

        data["updated_at"] = time.time()
        _save(path, data)


def increment(stage_id: str, field: str,
              output_dir: str = "geo_output") -> int:
    """
    Atomically increment an integer counter field inside a stage's meta.
    Returns the new value. Useful for fan-out progress (brands scraped, etc.).
    """
    path = Path(output_dir) / "pipeline_status.json"
    os.makedirs(output_dir, exist_ok=True)

    with _lock:
        data  = _load(path)
        stage = data.setdefault("stages", {}).setdefault(stage_id, {})
        new_val = stage.get(field, 0) + 1
        stage[field]       = new_val
        data["updated_at"] = time.time()
        _save(path, data)
        return new_val


def reset(output_dir: str = "geo_output") -> None:
    """Clear pipeline_status.json for a fresh run."""
    path = Path(output_dir) / "pipeline_status.json"
    with _lock:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
