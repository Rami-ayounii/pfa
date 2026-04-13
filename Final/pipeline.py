"""
Final/pipeline.py
══════════════════
GEO Multi-Agent Pipeline — Master Orchestrator (merged Final/ version)

LangGraph mode only — runs the full pipeline via a StateGraph with
parallel LLM fan-out (Agent 1) and parallel brand-scraping fan-out (Agent 2).

  python pipeline.py --query "Restaurants in Sfax"
  python pipeline.py --query "Best cafes in Tunis" --top_n 5

Flow:
  parse_query → Agent 0 (prompt gen)
    → supervisor0 (quality gate)
      → [sequential LLM queries] → Agent 1 (entity extract, clean, GEO metrics)
        → supervisor1 (quality gate)
          → monitor → [parallel brand fan-out] → Agent 2 (social scraping)
            → export (CSV + JSON)
              → supervisor2 → END
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import uuid
import asyncio
import concurrent.futures
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from loguru import logger

# ── Configure loguru ──────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

from graph.pipeline_graph import build_graph
from graph.state import initial_state
from core.memory import KnowledgeBase
from core.status_tracker import reset as _reset_status
from config import (
    OUTPUT_DIR,
    N_RUNS,
    N_INTENTS,
    N_VARIANTS,
    LANGUAGES,
    MAX_RETRIES,
    DEFAULT_QUERY_MODELS,
    MODEL_STRONG,
)
from core.llm_client import TOKEN_USAGE


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_graph_pipeline(
    query: str,
    languages: list = None,
    n_intents: int = None,
    n_variants: int = None,
    n_runs: int = None,
    max_prompts_per_intent: int = 6,
    top_n_brands: int = 10,
    output_dir: str = None,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """
    Run the GEO pipeline via LangGraph with a single natural-language query.

    Args:
        query                 : e.g. "Restaurants in Sfax"
        languages             : query languages for Agent 0 (default: LANGUAGES from config)
        n_intents             : intent types for Agent 0 (default: N_INTENTS from config)
        n_variants            : prompt variants per intent/language (default: N_VARIANTS)
        n_runs                : LLM runs per (prompt x model) (default: N_RUNS from config)
        max_prompts_per_intent: max prompt variants kept per intent after stratified sampling
        top_n_brands          : how many top-stability brands to pass to Agent 2
        output_dir            : root output folder (default: OUTPUT_DIR from config)
        max_retries           : supervisor retry budget (default: MAX_RETRIES from config)

    Returns:
        Final GEOState dict with all pipeline results
    """
    languages  = languages  or LANGUAGES
    n_intents  = n_intents  if n_intents  is not None else N_INTENTS
    n_variants = n_variants if n_variants is not None else N_VARIANTS
    n_runs     = n_runs     if n_runs     is not None else N_RUNS
    output_dir = output_dir or OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    _reset_status(output_dir)  # clear stale pipeline_status.json from any previous run

    kb     = KnowledgeBase(kb_path=os.path.join(output_dir, "geo_knowledge.json"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]

    logger.info(f"[Pipeline] LangGraph mode — query: '{query}' | run_id: {run_id}")
    logger.info(
        f"[Pipeline] Models: {DEFAULT_QUERY_MODELS} | n_runs={n_runs} | "
        f"max_prompts/intent={max_prompts_per_intent} | "
        f"~{max_prompts_per_intent * n_intents * len(DEFAULT_QUERY_MODELS) * n_runs} LLM tasks"
        f" | top_n={top_n_brands}"
    )

    state = initial_state(
        query                  = query,
        languages              = languages,
        n_intents              = n_intents,
        n_variants             = n_variants,
        n_runs                 = n_runs,
        max_prompts_per_intent = max_prompts_per_intent,
        query_models           = DEFAULT_QUERY_MODELS,
        analyst_model          = MODEL_STRONG,
        output_dir             = output_dir,
        top_n_brands           = top_n_brands,
        run_id                 = run_id,
        knowledge_base         = kb,
        max_retries            = max_retries,
    )

    graph = build_graph()
    try:
        loop = asyncio.get_running_loop()
        # Already inside an async context (e.g. FastMCP / Jupyter).
        # Run the coroutine in a separate thread that has its own event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            final_state = pool.submit(asyncio.run, graph.ainvoke(state)).result()  # type: ignore[union-attr]
    except RuntimeError:
        # No running loop — normal CLI execution.
        final_state = asyncio.run(graph.ainvoke(state))  # type: ignore[union-attr]

    try:
        global_df     = final_state.get("global_df")
        entity_count  = len(global_df) if global_df is not None else 0
        quality_score = final_state.get("quality_score", 0.0)
        domain_out    = final_state.get("domain", query)
        location_out  = final_state.get("location", "")
        params = {
            "n_runs":                 n_runs,
            "max_prompts_per_intent": max_prompts_per_intent,
            "models":                 DEFAULT_QUERY_MODELS,
        }
        kb.save_run(
            run_id        = run_id,
            domain        = domain_out,
            location      = location_out,
            params        = params,
            quality_score = quality_score,
            entity_count  = entity_count,
            tokens_used   = TOKEN_USAGE.get("total_tokens", 0),
        )
        best = kb.get_best_params(domain_out)
        if best:
            logger.info(f"[Pipeline] Historical best params for '{domain_out}': {best}")
    except Exception as _kb_err:
        logger.warning(f"[Pipeline] KB save_run failed (non-fatal): {_kb_err}")
    finally:
        kb.close()

    return final_state


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEO Pipeline (merged Final/ version)")
    parser.add_argument("--query", required=True,
                        help="Natural-language query, e.g. 'Restaurants in Sfax'")
    parser.add_argument("--languages",             nargs="+", default=["fr"])
    parser.add_argument("--n_intents",             type=int,  default=4)
    parser.add_argument("--n_variants",            type=int,  default=3)
    parser.add_argument("--n_runs",                type=int,  default=2)
    parser.add_argument("--max_prompts_per_intent", type=int, default=6)
    parser.add_argument("--top_n",                 type=int,  default=10, dest="top_n_brands")
    parser.add_argument("--output_dir",            default=OUTPUT_DIR)
    parser.add_argument("--max_retries",           type=int,  default=1)
    args = parser.parse_args()

    result = run_graph_pipeline(**vars(args))
    print("Pipeline complete.")
