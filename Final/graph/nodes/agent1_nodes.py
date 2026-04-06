"""
graph/nodes/agent1_nodes.py
═══════════════════════════
Sequential LangGraph nodes for Agent 1 (post fan-out merge):

  agent1_load_node      → converts prompt_df → prompts list
  agent1_aggregate_node → converts raw_responses list → pd.DataFrame + async CSV
  agent1_extract_node   → entity extraction (step 3)
  agent1_enrich_node    → entity enrichment (step 4)
  agent1_clean_node     → entity cleaning + dedup (step 5)
  agent1_metrics_node   → GEO feature computation (step 6) + derive brand list
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Final/ is root

import os
import time
import csv
import pandas as pd
from loguru import logger
from tqdm import tqdm

from graph.state import GEOState
from core.llm_client import MODEL_STRONG


# ── Helper: CSV backup (sync — safe to call from any thread) ─────────────────
def _fire_csv_backup(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to CSV synchronously. Thread-safe, no event-loop required."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.debug(f"CSV saved → {path}")
    except Exception as e:
        logger.warning(f"CSV write failed ({e})")


# ════════════════════════════════════════════════════════════════════════════════
# Node 1 — Load prompts from in-memory DataFrame
# ════════════════════════════════════════════════════════════════════════════════

def agent1_load_node(state: GEOState) -> dict:
    """
    Convert prompt_df (in-memory DataFrame from Agent 0) → list of prompt dicts.
    Validates required columns before passing downstream.
    """
    prompt_df = state.get("prompt_df")
    if prompt_df is None or len(prompt_df) == 0:
        logger.error("[Agent1LoadNode] No prompt_df found in state")
        return {"prompts": [], "errors": ["Agent1: prompt_df is empty or missing"]}

    required = ["prompt_id", "intent_id", "language", "prompt_text"]
    missing  = [c for c in required if c not in prompt_df.columns]
    if missing:
        msg = f"Agent1: prompt_df missing columns: {missing}"
        logger.error(f"[Agent1LoadNode] {msg}")
        return {"prompts": [], "errors": [msg]}

    prompts = prompt_df.to_dict(orient="records")
    logger.info(
        f"[Agent1LoadNode] {len(prompts)} prompts | "
        f"intents: {prompt_df['intent_id'].unique().tolist()} | "
        f"langs: {prompt_df['language'].unique().tolist()}"
    )
    return {"prompts": prompts}


# ════════════════════════════════════════════════════════════════════════════════
# Node 2 — Aggregate fan-in raw_responses list → DataFrame
# ════════════════════════════════════════════════════════════════════════════════

def agent1_aggregate_node(state: GEOState) -> dict:
    """
    Merge all raw_responses accumulated from parallel LLM fan-out nodes into
    a single DataFrame. Adds derived fields (response_id, run_id, timestamp).
    Fires async CSV backup.
    """
    raw_responses = state.get("raw_responses", [])
    output_dir    = state.get("output_dir", "geo_output")

    if not raw_responses:
        logger.warning("[Agent1AggNode] No raw_responses received from fan-out")
        return {"entities_df": pd.DataFrame(), "warnings": ["Agent1: no LLM responses"]}

    logger.info(f"[Agent1AggNode] Aggregating {len(raw_responses)} responses")

    rows = []
    for r in raw_responses:
        if not r.get("response_text"):
            continue
        prompt_id = r.get("prompt_id", "")
        model_id  = r.get("model_id", "")
        run_idx   = r.get("run_idx", 1)
        run_id    = f"run_{run_idx}"
        response_id = f"{prompt_id}__{model_id.replace('/', '--')}__{run_id}"
        rows.append({
            "response_id":       response_id,
            "prompt_id":         prompt_id,
            "model_id":          model_id,
            "run_id":            run_id,
            "prompt_text":       r.get("prompt_text", ""),
            "response_text":     r.get("response_text", ""),
            "completion_tokens": r.get("completion_tokens", 0),
            "prompt_tokens":     r.get("prompt_tokens", 0),
            "total_tokens":      r.get("total_tokens", 0),
            "timestamp":         time.time(),
        })

    df_raw = pd.DataFrame(rows)
    logger.success(f"[Agent1AggNode] {len(df_raw)} valid responses aggregated")

    # Async CSV backup
    csv_path = os.path.join(output_dir, "raw_responses.csv")
    os.makedirs(output_dir, exist_ok=True)
    _fire_csv_backup(df_raw, csv_path)

    # Store df_raw as entities_df placeholder so downstream nodes can read it
    # (actual entity extraction happens in agent1_extract_node)
    return {"entities_df": df_raw}   # reused as df_raw carrier until step3 replaces it


# ════════════════════════════════════════════════════════════════════════════════
# Node 3 — Entity extraction (Step 3)
# ════════════════════════════════════════════════════════════════════════════════

def agent1_extract_node(state: GEOState) -> dict:
    """
    Run step3_extract_entities on the aggregated raw responses DataFrame.
    entities_df at this point holds df_raw (set by aggregate node).
    """
    from agents.agent1 import Agent1GeoAnalyser

    df_raw      = state.get("entities_df")      # df_raw from aggregate node
    output_dir  = state.get("output_dir", "geo_output")
    mistral_key = state.get("mistral_key", "")
    groq_client = state.get("groq_client")

    if df_raw is None or len(df_raw) == 0:
        logger.warning("[Agent1ExtractNode] df_raw is empty, skipping extraction")
        return {"entities_df": pd.DataFrame()}

    logger.info(f"[Agent1ExtractNode] Extracting entities from {len(df_raw)} responses")

    output_path = os.path.join(output_dir, "extracted_entities.csv")
    os.makedirs(output_dir, exist_ok=True)

    from agents.agent1_geo_analyser import step3_extract_entities
    # step3 needs df_raw with response_id column
    df_entities = step3_extract_entities(
        df_raw, output_path, mistral_key or None, groq_client
    )

    logger.success(
        f"[Agent1ExtractNode] {df_entities['entity'].nunique()} unique entities extracted"
    )
    return {"entities_df": df_entities}


# ════════════════════════════════════════════════════════════════════════════════
# Node 4 — Entity enrichment (Step 4)
# ════════════════════════════════════════════════════════════════════════════════

def agent1_enrich_node(state: GEOState) -> dict:
    """
    Run step4_enrich_entities: add ranking_position + description_length.
    Needs both entities_df (from step 3) and the raw_responses for joining.
    """
    df_entities = state.get("entities_df")
    output_dir  = state.get("output_dir", "geo_output")
    groq_client = state.get("groq_client")

    # Reconstruct df_raw from raw_responses for join
    raw_responses = state.get("raw_responses", [])
    df_raw_rows = []
    for r in raw_responses:
        if not r.get("response_text"):
            continue
        prompt_id   = r.get("prompt_id", "")
        model_id    = r.get("model_id", "")
        run_idx     = r.get("run_idx", 1)
        run_id      = f"run_{run_idx}"
        response_id = f"{prompt_id}__{model_id.replace('/', '--')}__{run_id}"
        df_raw_rows.append({
            "response_id":       response_id,
            "prompt_id":         prompt_id,
            "model_id":          model_id,
            "run_id":            run_id,
            "response_text":     r.get("response_text", ""),
            "completion_tokens": r.get("completion_tokens", 0),
        })
    df_raw = pd.DataFrame(df_raw_rows)

    if df_entities is None or len(df_entities) == 0:
        logger.warning("[Agent1EnrichNode] entities_df is empty, skipping enrichment")
        return {"enriched_df": pd.DataFrame()}

    logger.info(f"[Agent1EnrichNode] Enriching {len(df_entities)} entity rows")

    output_path = os.path.join(output_dir, "enriched_entities.csv")
    os.makedirs(output_dir, exist_ok=True)

    from agents.agent1_geo_analyser import step4_enrich_entities
    df_enriched = step4_enrich_entities(df_entities, df_raw, output_path, groq_client)

    logger.success(f"[Agent1EnrichNode] {len(df_enriched)} enriched rows")
    return {"enriched_df": df_enriched}


# ════════════════════════════════════════════════════════════════════════════════
# Node 5 — Entity cleaning (Step 5)
# ════════════════════════════════════════════════════════════════════════════════

def agent1_clean_node(state: GEOState) -> dict:
    """
    Run step5_clean_entities: fuzzy dedup + LLM arbitration + self-reflection.
    """
    df_enriched   = state.get("enriched_df")
    df_entities   = state.get("entities_df")
    output_dir    = state.get("output_dir", "geo_output")
    domain        = state.get("domain", "Tunisian restaurant")
    location      = state.get("location", "Tunisia")
    analyst_model = state.get("analyst_model", MODEL_STRONG)
    groq_client   = state.get("groq_client")
    kb            = state.get("knowledge_base")

    # Reconstruct df_raw for step5 (used for context lookup)
    raw_responses = state.get("raw_responses", [])
    df_raw_rows = []
    for r in raw_responses:
        if not r.get("response_text"):
            continue
        prompt_id   = r.get("prompt_id", "")
        model_id    = r.get("model_id", "")
        run_idx     = r.get("run_idx", 1)
        run_id      = f"run_{run_idx}"
        response_id = f"{prompt_id}__{model_id.replace('/', '--')}__{run_id}"
        df_raw_rows.append({
            "response_id":   response_id,
            "prompt_id":     prompt_id,
            "response_text": r.get("response_text", ""),
        })
    df_raw = pd.DataFrame(df_raw_rows)

    if df_enriched is None or len(df_enriched) == 0:
        logger.warning("[Agent1CleanNode] enriched_df is empty, skipping cleaning")
        return {"clean_df": pd.DataFrame()}

    logger.info(
        f"[Agent1CleanNode] Cleaning {df_enriched['entity'].nunique()} unique entities"
    )

    output_path = os.path.join(output_dir, "clean_entities.csv")
    log_path    = os.path.join(output_dir, "cleaning_log.csv")
    os.makedirs(output_dir, exist_ok=True)

    from agents.agent1_geo_analyser import step5_clean_entities
    df_clean = step5_clean_entities(
        df_enriched, df_raw, df_entities,
        analyst_model, output_path, log_path,
        groq_client, domain=domain,
        kb=kb, location=location,
    )

    logger.success(
        f"[Agent1CleanNode] {df_clean['canonical_entity'].nunique()} canonical entities"
    )
    return {"clean_df": df_clean}


# ════════════════════════════════════════════════════════════════════════════════
# Node 6 — GEO feature computation + brand list (Step 6)
# ════════════════════════════════════════════════════════════════════════════════

def agent1_metrics_node(state: GEOState) -> dict:
    """
    Run step6_compute_metrics and derive the top-N brand list for Agent 2.
    """
    df_clean    = state.get("clean_df")
    prompts     = state.get("prompts", [])
    output_dir  = state.get("output_dir", "geo_output")
    top_n       = state.get("top_n_brands", 10)

    # Reconstruct df_raw for metric computation
    raw_responses = state.get("raw_responses", [])
    df_raw_rows = []
    for r in raw_responses:
        if not r.get("response_text"):
            continue
        prompt_id   = r.get("prompt_id", "")
        model_id    = r.get("model_id", "")
        run_idx     = r.get("run_idx", 1)
        run_id      = f"run_{run_idx}"
        response_id = f"{prompt_id}__{model_id.replace('/', '--')}__{run_id}"
        df_raw_rows.append({"response_id": response_id})
    df_raw = pd.DataFrame(df_raw_rows)

    if df_clean is None or len(df_clean) == 0:
        logger.warning("[Agent1MetricsNode] clean_df is empty, skipping metrics")
        return {"features_df": pd.DataFrame(), "global_df": pd.DataFrame(), "brands": []}

    logger.info(f"[Agent1MetricsNode] Computing GEO metrics")

    output_path = os.path.join(output_dir, "entity_features.csv")
    global_path = os.path.join(output_dir, "entity_features_global.csv")
    os.makedirs(output_dir, exist_ok=True)

    from agents.agent1_geo_analyser import step6_compute_metrics
    df_features, df_global = step6_compute_metrics(
        df_clean, df_raw, prompts, output_path, global_path
    )

    # Derive top-N brand list by stability_score
    df_top = df_global.copy()
    if "clean_flag" in df_top.columns:
        df_top = df_top[df_top["clean_flag"] != "invalid"]
    brand_list = (
        df_top
        .sort_values("stability_score", ascending=False)
        ["canonical_entity"]
        .dropna()
        .unique()
        [:top_n]
        .tolist()
    )

    logger.success(
        f"[Agent1MetricsNode] {len(df_global)} total entities | "
        f"top-{top_n} brands: {brand_list}"
    )
    return {
        "features_df": df_features,
        "global_df":   df_global,
        "brands":      brand_list,
    }
