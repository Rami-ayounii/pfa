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

import os
import json
import pandas as pd
from loguru import logger

from graph.state import GEOState
from core.llm_client import TOKEN_USAGE


def export_node(state: GEOState) -> dict:
    """
    Final pipeline node — export all collected data to disk and print summary.
    """
    profiles    = state.get("profiles", [])
    global_df   = state.get("global_df")
    features_df = state.get("features_df")
    output_dir  = state.get("output_dir", "geo_output")
    domain      = state.get("domain", "")
    location    = state.get("location", "")
    query       = state.get("query", "")
    brands      = state.get("brands", [])
    errors      = state.get("errors", [])
    warnings    = state.get("warnings", [])

    agent2_dir = os.path.join(output_dir, "agent2_output")
    os.makedirs(agent2_dir, exist_ok=True)

    # ── Export social profiles ─────────────────────────────────────────────────
    if profiles:
        try:
            from agents.agent2_social_scraper import step8_export
            step8_export(profiles, [], agent2_dir)
            logger.success(f"[ExportNode] {len(profiles)} social profiles exported → {agent2_dir}")
        except Exception as e:
            logger.warning(f"[ExportNode] step8_export failed ({e}), falling back to manual export")
            _export_profiles_manual(profiles, agent2_dir)

    # ── Export pipeline summary ────────────────────────────────────────────────
    avg_authority = 0.0
    if profiles:
        scores = [getattr(p, "social_authority_score", None) or 0 for p in profiles]
        avg_authority = round(sum(scores) / len(scores), 1) if scores else 0.0

    summary = {
        "query":            query,
        "domain":           domain,
        "location":         location,
        "brands_targeted":  brands,
        "brands_profiled":  [getattr(p, "brand", str(p)) for p in profiles],
        "entities_found":   len(global_df) if global_df is not None else 0,
        "token_usage":      TOKEN_USAGE.copy(),
        "avg_authority":    avg_authority,
        "errors":           errors,
        "warnings":         warnings,
    }
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print final banner ─────────────────────────────────────────────────────
    logger.success("═" * 60)
    logger.success(" PIPELINE COMPLETE")
    logger.success("═" * 60)
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
    if errors:
        logger.warning(f"  Errors          : {len(errors)}")
    logger.info(f"  Summary saved   : {summary_path}")
    logger.success("═" * 60)

    return {}


def _export_profiles_manual(profiles, output_dir: str) -> None:
    """Fallback manual export when step8_export is not available."""
    rows = []
    for p in profiles:
        if hasattr(p, "__dict__"):
            rows.append(p.__dict__)
        elif isinstance(p, dict):
            rows.append(p)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "social_profiles.csv"),
            index=False, encoding="utf-8-sig"
        )
