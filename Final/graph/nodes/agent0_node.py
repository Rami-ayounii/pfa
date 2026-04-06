"""
graph/nodes/agent0_node.py
══════════════════════════
LangGraph node: run Agent 0 (Prompt Generator).

Input state:  domain, location, n_intents, n_variants, languages,
              analyst_model, output_dir, groq_client
Output state: prompt_df
Side-effect:  writes prompt_set CSV asynchronously to output_dir
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Final/ is root

import os
import pandas as pd
from loguru import logger

from graph.state import GEOState
from agents.agent0 import agent0_run
from core.llm_client import MODEL_STRONG


def agent0_node(state: GEOState) -> dict:
    """
    Generate a diverse prompt set for the domain/location using Agent 0.
    The resulting DataFrame is returned in-memory; CSV backup fires async.
    """
    domain         = state.get("domain", "Tunisian restaurants")
    location       = state.get("location", "Tunisia")
    n_intents      = state.get("n_intents",  3)
    n_variants     = state.get("n_variants", 4)
    languages      = state.get("languages",  ["fr"])
    analyst_model  = state.get("analyst_model", MODEL_STRONG)
    output_dir     = state.get("output_dir", "geo_output")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"[Agent0Node] Generating prompts — domain='{domain}', location='{location}'")

    MAX_PROMPTS            = 10   # absolute upper bound regardless of settings
    max_prompts_per_intent = state.get("max_prompts_per_intent", 2)
    max_reflection_loops   = state.get("max_reflection_loops", 2)

    prompt_df = agent0_run(
        domain=domain,
        languages=languages,
        n_intents=n_intents,
        n_variants=n_variants,
        max_reflection_loops=max_reflection_loops,
    )

    if "intent_id" in prompt_df.columns and len(prompt_df) > max_prompts_per_intent:
        # Stratified sampling: keep at most max_prompts_per_intent rows per intent.
        # groupby.head(n) is the canonical pandas idiom — always preserves all columns.
        n_intents_before = prompt_df["intent_id"].nunique()
        sampled = (
            prompt_df
            .groupby("intent_id")
            .head(max_prompts_per_intent)
            .reset_index(drop=True)
        )
        if len(sampled) > MAX_PROMPTS:
            sampled = sampled.head(MAX_PROMPTS).reset_index(drop=True)
        if len(sampled) < len(prompt_df):
            logger.warning(
                f"[Agent0Node] Stratified sampling: {len(prompt_df)} → {len(sampled)} prompts "
                f"({max_prompts_per_intent}/intent × {n_intents_before} intents)"
            )
        prompt_df = sampled
    elif len(prompt_df) > MAX_PROMPTS:
        logger.warning(
            f"[Agent0Node] Capping prompts {len(prompt_df)} → {MAX_PROMPTS} (absolute limit)"
        )
        prompt_df = prompt_df.head(MAX_PROMPTS).reset_index(drop=True)

    _n_intents = prompt_df["intent_id"].nunique() if "intent_id" in prompt_df.columns else len(prompt_df)
    logger.success(
        f"[Agent0Node] Generated {len(prompt_df)} prompts across {_n_intents} intents"
    )

    return {"prompt_df": prompt_df}
