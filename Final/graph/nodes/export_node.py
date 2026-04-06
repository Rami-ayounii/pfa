"""
graph/nodes/export_node.py
══════════════════════════
LangGraph final node: export all outputs to disk.

Writes:
  - agent2_output/social_profiles.csv
  - agent2_output/hallucination_flags.csv
  - agent2_output/audit_log.json
  - pipeline_summary.json

Input state:  profiles, global_df, features_df, domain, location,
              output_dir, query, brands, errors, warnings
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Final/ is root

import os
import json
import pandas as pd
from loguru import logger

from graph.state import GEOState
from core.llm_client import TOKEN_USAGE


def _serialize(obj):
    """
    Recursively serialize objects to JSON-safe types.
    Handles dataclass instances, plain dicts (new ReAct output),
    lists/tuples, pandas Series, and numpy scalars.
    """
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _serialize(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    elif hasattr(obj, 'to_dict'):  # pandas Series
        return obj.to_dict()
    elif hasattr(obj, 'item'):     # numpy scalar
        return obj.item()
    else:
        return obj


def export_node(state: GEOState) -> dict:
    """
    Final pipeline node — export all collected data to disk and print summary.
    """
    profiles         = state.get("profiles", [])
    global_df        = state.get("global_df")
    features_df      = state.get("features_df")
    output_dir       = state.get("output_dir", "geo_output")
    domain           = state.get("domain", "")
    location         = state.get("location", "")
    query            = state.get("query", "")
    brands           = state.get("brands", [])
    errors           = state.get("errors", [])
    warnings         = state.get("warnings", [])
    quality_score    = state.get("quality_score", 0.0)
    quality_warnings = state.get("quality_warnings", [])
    run_id           = state.get("run_id", "")

    agent2_dir = os.path.join(output_dir, "agent2_output")
    os.makedirs(agent2_dir, exist_ok=True)

    # ── Export social profiles ─────────────────────────────────────────────────
    if profiles:
        _export_profiles_manual(profiles, agent2_dir)
        logger.success(f"[ExportNode] {len(profiles)} social profiles exported → {agent2_dir}")

    # ── Compute avg authority from profiles (dict or dataclass) ───────────────
    avg_authority = 0.0
    if profiles:
        scores = []
        for p in profiles:
            if isinstance(p, dict):
                score = p.get("social_authority_score") or p.get("authority_score") or 0
            else:
                score = getattr(p, "social_authority_score", None) or 0
            scores.append(score)
        avg_authority = round(sum(scores) / len(scores), 1) if scores else 0.0

    # ── Derive brand names from profiles ──────────────────────────────────────
    profiled_brands = []
    for p in profiles:
        if isinstance(p, dict):
            profiled_brands.append(p.get("brand", str(p)))
        else:
            profiled_brands.append(getattr(p, "brand", str(p)))

    # ── Export pipeline summary ────────────────────────────────────────────────
    summary = {
        "run_id":            run_id,
        "query":             query,
        "domain":            domain,
        "location":          location,
        "brands_targeted":   brands,
        "brands_profiled":   profiled_brands,
        "entities_found":    len(global_df) if global_df is not None else 0,
        "token_usage":       TOKEN_USAGE.copy(),
        "avg_authority":     avg_authority,
        "quality_score":     quality_score,
        "quality_warnings":  quality_warnings,
        "errors":            errors,
        "warnings":          warnings,
    }
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_serialize(summary), f, indent=2, ensure_ascii=False)

    # ── Print final banner ─────────────────────────────────────────────────────
    logger.success("=" * 60)
    logger.success(" PIPELINE COMPLETE")
    logger.success("=" * 60)
    logger.info(f"  Query           : {query}")
    logger.info(f"  Domain          : {domain}")
    logger.info(f"  Location        : {location}")
    logger.info(f"  Output dir      : {output_dir}/")
    logger.info(f"  Total tokens    : {TOKEN_USAGE['total_tokens']}")
    if global_df is not None:
        logger.info(f"  Entities found  : {len(global_df)}")
    if profiles:
        logger.info(f"  Brands profiled : {len(profiles)}")
        logger.info(f"  Avg authority   : {avg_authority}/100")
    logger.info(f"  Quality score   : {quality_score}/10.0")
    if quality_warnings:
        for qw in quality_warnings:
            logger.warning(f"  [!]  {qw}")
    if errors:
        logger.warning(f"  Errors          : {len(errors)}")
    logger.info(f"  Summary saved   : {summary_path}")
    logger.success("=" * 60)

    return {}


def _export_profiles_manual(profiles, output_dir: str) -> None:
    """Export profiles to CSV — handles both dataclass and plain dict profiles."""
    rows = []
    for p in profiles:
        if hasattr(p, '__dataclass_fields__'):
            rows.append({k: _serialize(v) for k, v in vars(p).items()})
        elif isinstance(p, dict):
            rows.append({k: _serialize(v) for k, v in p.items()})
        elif hasattr(p, "__dict__"):
            rows.append(p.__dict__)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "social_profiles.csv"),
            index=False, encoding="utf-8-sig"
        )
