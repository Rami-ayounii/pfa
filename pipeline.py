"""
pipeline.py
═══════════
GEO Multi-Agent Pipeline — Master Orchestrator

Two execution modes:

  1. LangGraph mode  (--query)     ← NEW primary mode
     ─────────────────────────────────────────────────────────────
     Single natural-language input drives the entire pipeline via
     a LangGraph StateGraph with parallel LLM fan-out (Agent 1)
     and parallel brand-scraping fan-out (Agent 2).

       python pipeline.py --query "Restaurants in Sfax"
       python pipeline.py --query "Best cafes in Tunis" --top_n 5

  2. Legacy mode  (--domain + --location)
     ─────────────────────────────────────────────────────────────
     Original sequential pipeline kept for backward compatibility.

       python pipeline.py --domain "Tunisian restaurants" --location "Tunisia"
       python pipeline.py --brands_only "dar el jeld,le corsaire"

Flow (LangGraph mode):
  parse_query → Agent 0 (prompt gen)
    → [parallel LLM fan-out] → Agent 1 (entity extract, clean, GEO metrics)
      → [parallel brand fan-out] → Agent 2 (social scraping)
        → export (CSV + JSON)
"""

import os
import sys
import json
import asyncio
import argparse
from dotenv import load_dotenv

load_dotenv()  # Load .env before any agent/client imports read os.environ

from groq import Groq
from loguru import logger

# ── Configure loguru ──────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

import pandas as pd

from agents.agent0_prompt_generator import Agent0PromptGenerator
from agents.agent1_geo_analyser     import Agent1GeoAnalyser
from agents.agent2_social_scraper   import Agent2SocialScraper
from core.llm_client                import TOKEN_USAGE, MODEL_STRONG, MODEL_COMPOUND, MODEL_FAST, MODEL_QWEN


def _load_env() -> dict:
    """Load API keys from environment variables."""
    return {
        "groq_key":       os.environ.get("GROQ_API_KEY",       ""),
        "mistral_key":    os.environ.get("MISTRAL_API_KEY",    ""),
        "openrouter_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "serpapi_key":    os.environ.get("SERPAPI_KEY",        ""),
        "apify_token":    os.environ.get("APIFY_API_TOKEN",    ""),
        "ig_user":        os.environ.get("IG_USERNAME",        ""),
        "ig_pass":        os.environ.get("IG_PASSWORD",        ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_graph_pipeline(
    query:                  str,
    languages:              list  = None,
    n_intents:              int   = 3,
    n_variants:             int   = 4,
    n_runs:                 int   = 1,    # was 2 — 1 run × stratified prompts keeps quality while halving requests
    max_prompts_per_intent: int   = 2,    # 2 prompts/intent × 3 intents = 6 prompts max
    query_models:           list  = None,
    analyst_model:          str   = MODEL_COMPOUND,
    output_dir:             str   = "geo_output",
    top_n_brands:           int   = 10,
    is_colab:               bool  = False,
) -> dict:
    """
    Run the GEO pipeline via LangGraph with a single natural-language query.

    Args:
        query        : e.g. "Restaurants in Sfax"
        languages    : query languages for Agent 0 (default: ["fr"])
        n_intents              : intent types for Agent 0
        n_variants             : prompt variants per intent/language
        n_runs                 : LLM runs per (prompt × model) — default 1 to conserve tokens
        max_prompts_per_intent : max prompt variants kept per intent after stratified sampling
        query_models           : Groq model IDs for parallel LLM querying
        analyst_model: model for Agent 0 prompt generation (default: compound-beta)
        output_dir   : root output folder
        top_n_brands : how many top-stability brands to pass to Agent 2
        is_colab     : set True when running in Google Colab

    Returns:
        Final GEOState dict with all pipeline results
    """
    from graph.pipeline_graph import build_graph

    languages    = languages    or ["fr"]
    # Default: fast Llama 8B + multilingual Qwen3-32B → diversity with low token cost
    query_models = query_models or [MODEL_FAST, MODEL_QWEN]

    env    = _load_env()
    client = Groq(api_key=env["groq_key"]) if env["groq_key"] else None

    os.makedirs(output_dir, exist_ok=True)

    _expected_tasks = max_prompts_per_intent * n_intents * len(query_models) * n_runs
    logger.info(f"[Pipeline] LangGraph mode — query: '{query}'")
    logger.info(
        f"[Pipeline] Models: {query_models} | n_runs={n_runs} | "
        f"max_prompts/intent={max_prompts_per_intent} | "
        f"~{_expected_tasks} LLM tasks | top_n={top_n_brands}"
    )

    initial_state = {
        # Input
        "query":          query,
        "domain":         "",        # filled by parse_query_node
        "location":       "",        # filled by parse_query_node
        # Config
        "n_intents":              n_intents,
        "n_variants":             n_variants,
        "n_runs":                 n_runs,
        "max_prompts_per_intent": max_prompts_per_intent,
        "languages":              languages,
        "query_models":           query_models,
        "analyst_model":          analyst_model,
        "output_dir":             output_dir,
        "top_n_brands":           top_n_brands,
        "is_colab":               is_colab,
        # Clients & keys
        "groq_client":    client,
        "mistral_key":    env["mistral_key"],
        "openrouter_key": env["openrouter_key"],
        "serpapi_key":    env["serpapi_key"],
        "apify_token":    env["apify_token"],
        "ig_user":        env["ig_user"],
        "ig_pass":        env["ig_pass"],
        # Fan-in accumulators (must start as empty lists)
        "raw_responses":  [],
        "profiles":       [],
        "errors":         [],
        "warnings":       [],
    }

    graph       = build_graph()
    final_state = asyncio.run(graph.ainvoke(initial_state))
    return final_state


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY: original sequential pipeline (kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    domain:         str       = "Tunisian restaurants",
    languages:      list      = None,
    n_intents:      int       = 3,
    n_variants:     int       = 4,
    n_runs:         int       = 1,    # was 2
    query_models:   list      = None,
    analyst_model:  str       = MODEL_COMPOUND,
    output_dir:     str       = "geo_output",
    brands_only:    list      = None,
    skip_agent0:    bool      = False,
    skip_agent1:    bool      = False,
    skip_agent2:    bool      = False,
    is_colab:       bool      = False,
    top_n_brands:   int       = 10,
    location:       str       = "Tunisia",
) -> dict:
    """
    Run the full GEO multi-agent pipeline (legacy sequential mode).

    Args:
        domain        : research domain (e.g. "Tunisian restaurants")
        languages     : query languages for Agent 0 (default: ["fr"])
        n_intents     : intent types for Agent 0 to discover
        n_variants    : prompt variants per intent/language
        n_runs        : LLM runs per prompt × model (Agent 1)
        query_models  : Groq model IDs for Agent 1 querying
        analyst_model : model for Agent 0 prompt generation (default: compound-beta)
        output_dir    : root output folder
        brands_only   : skip Agents 0+1 and run Agent 2 on this brand list
        skip_agent*   : skip individual agents (for partial reruns)
        is_colab      : set True when running in Google Colab
        top_n_brands  : how many top-stability brands to pass to Agent 2

    Returns:
        dict with keys: prompt_df, features_df, global_df, profiles, audit
    """
    languages    = languages    or ["fr"]
    query_models = query_models or [MODEL_FAST, MODEL_QWEN]

    env    = _load_env()
    client = Groq(api_key=env["groq_key"]) if env["groq_key"] else None

    os.makedirs(output_dir, exist_ok=True)
    slug = domain.replace(" ", "_")

    result = {
        "prompt_df":  None,
        "features_df": None,
        "global_df":   None,
        "profiles":    [],
        "audit":       [],
    }

    # ── Agent 0 ───────────────────────────────────────────────────────────────
    prompt_set_path = os.path.join(output_dir, f"prompt_set_{slug}.csv")

    if not brands_only and not skip_agent0:
        logger.info("=" * 60)
        logger.info(" AGENT 0 — Prompt Generator")
        logger.info("=" * 60)
        agent0 = Agent0PromptGenerator(
            domain               = domain,
            model                = analyst_model,
            languages            = languages,
            n_intents            = n_intents,
            n_variants           = n_variants,
            max_reflection_loops = 2,
            output_dir           = output_dir,
            groq_client          = client,
        )
        _df = agent0.run()
        if len(_df) > 10:
            logger.warning(f"[Pipeline] Capping prompts {len(_df)} → 10 (token limit)")
            _df = _df.head(10).reset_index(drop=True)
        result["prompt_df"] = _df
    elif os.path.exists(prompt_set_path):
        logger.info(f"[Pipeline] Loading existing prompt set: {prompt_set_path}")
        result["prompt_df"] = pd.read_csv(prompt_set_path, encoding="utf-8-sig")

    # ── Agent 1 ───────────────────────────────────────────────────────────────
    global_features_path = os.path.join(output_dir, "entity_features_global.csv")

    if not brands_only and not skip_agent1 and result["prompt_df"] is not None:
        logger.info("=" * 60)
        logger.info(" AGENT 1 — GEO Analyser")
        logger.info("=" * 60)
        agent1 = Agent1GeoAnalyser(
            prompt_set_path = prompt_set_path,
            output_dir      = output_dir,
            query_models    = query_models,
            n_runs          = n_runs,
            analyst_model   = analyst_model,
            groq_client     = client,
            mistral_key     = env["mistral_key"],
            domain          = domain,
        )
        result["features_df"], result["global_df"] = agent1.run()

    elif os.path.exists(global_features_path):
        logger.info(f"[Pipeline] Loading existing global features: {global_features_path}")
        result["global_df"] = pd.read_csv(global_features_path, encoding="utf-8-sig")

    # ── Derive brand list for Agent 2 ─────────────────────────────────────────
    if brands_only:
        brand_list = brands_only
    elif result["global_df"] is not None:
        df_global = result["global_df"]
        if "clean_flag" in df_global.columns:
            df_global = df_global[df_global["clean_flag"] != "invalid"]
        brand_list = (
            df_global
            .sort_values("stability_score", ascending=False)
            ["canonical_entity"]
            .dropna()
            .unique()
            [:top_n_brands]
            .tolist()
        )
        logger.info(f"[Pipeline] Top {len(brand_list)} brands: {brand_list}")
    else:
        logger.warning("[Pipeline] No brand list available — skipping Agent 2")
        brand_list = []

    # ── Agent 2 ───────────────────────────────────────────────────────────────
    if brand_list and not skip_agent2:
        logger.info("=" * 60)
        logger.info(" AGENT 2 — Social Scraper & Brand Intelligence")
        logger.info("=" * 60)
        agent2 = Agent2SocialScraper(
            brands         = brand_list,
            output_dir     = os.path.join(output_dir, "agent2_output"),
            serpapi_key    = env["serpapi_key"],
            apify_token    = env["apify_token"],
            openrouter_key = env["openrouter_key"],
            groq_client    = client,
            ig_user        = env["ig_user"],
            ig_pass        = env["ig_pass"],
            is_colab       = is_colab,
            location       = location,
        )
        result["profiles"] = agent2.run()
        result["audit"]    = agent2.audit

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.success("=" * 60)
    logger.success(" PIPELINE COMPLETE (legacy mode)")
    logger.success("=" * 60)
    logger.info(f"  Domain          : {domain}")
    logger.info(f"  Output dir      : {output_dir}/")
    logger.info(f"  Total tokens    : {TOKEN_USAGE['total_tokens']}")
    if result["global_df"] is not None:
        logger.info(f"  Entities found  : {len(result['global_df'])}")
    if result["profiles"]:
        avg_auth = sum(p.social_authority_score or 0
                       for p in result["profiles"]) / len(result["profiles"])
        logger.info(f"  Brands profiled : {len(result['profiles'])}")
        logger.info(f"  Avg authority   : {avg_auth:.1f}/100")

    summary = {
        "domain":         domain,
        "languages":      languages,
        "n_intents":      n_intents,
        "n_variants":     n_variants,
        "n_runs":         n_runs,
        "query_models":   query_models,
        "output_dir":     output_dir,
        "token_usage":    TOKEN_USAGE.copy(),
        "brands_profiled": [p.brand for p in result["profiles"]],
    }
    with open(os.path.join(output_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary saved   : {output_dir}/pipeline_summary.json")
    logger.success("=" * 60)

    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GEO Multi-Agent Pipeline — Brand Visibility Analysis")

    # ── New primary arg ────────────────────────────────────────────────────────
    parser.add_argument(
        "--query", type=str, default=None,
        help='Single natural-language query — triggers LangGraph mode. '
             'Example: "Restaurants in Sfax"')

    # ── Shared args ────────────────────────────────────────────────────────────
    parser.add_argument("--languages",    nargs="+", default=["fr"],
                        help="Query languages (e.g. fr ar)")
    parser.add_argument("--n_intents",   type=int, default=3)
    parser.add_argument("--n_variants",  type=int, default=4)
    parser.add_argument("--n_runs",      type=int, default=1,
                        help="LLM runs per (prompt × model). Default 1 — increase for more stable GEO estimates.")
    parser.add_argument("--max_prompts_per_intent", type=int, default=2,
                        help="Max prompt variants per intent after stratified sampling. "
                             "Default 2 → 3 intents × 2 = 6 prompts max. "
                             "Set to 1 for minimal token usage.")
    parser.add_argument("--output_dir",  default="geo_output")
    parser.add_argument("--top_n",       type=int, default=10)
    parser.add_argument("--colab",       action="store_true")

    # ── Legacy-only args ───────────────────────────────────────────────────────
    parser.add_argument("--domain",       default="Tunisian restaurants",
                        help="[Legacy] Research domain")
    parser.add_argument("--location",     default="Tunisia",
                        help="[Legacy] Geographic location")
    parser.add_argument("--brands_only",  type=str, default=None,
                        help="[Legacy] Comma-separated brand list — skip Agents 0+1")
    parser.add_argument("--skip_agent0",  action="store_true")
    parser.add_argument("--skip_agent1",  action="store_true")
    parser.add_argument("--skip_agent2",  action="store_true")

    args = parser.parse_args()

    if args.query:
        # ── LangGraph mode (new) ───────────────────────────────────────────────
        logger.info(f"[CLI] LangGraph mode — query: '{args.query}'")
        run_graph_pipeline(
            query                  = args.query,
            languages              = args.languages,
            n_intents              = args.n_intents,
            n_variants             = args.n_variants,
            n_runs                 = args.n_runs,
            max_prompts_per_intent = args.max_prompts_per_intent,
            output_dir             = args.output_dir,
            top_n_brands           = args.top_n,
            is_colab               = args.colab,
        )
    else:
        # ── Legacy mode ────────────────────────────────────────────────────────
        logger.info(f"[CLI] Legacy mode — domain: '{args.domain}'")
        brands_list = [b.strip() for b in args.brands_only.split(",")
                       ] if args.brands_only else None
        run_pipeline(
            domain        = args.domain,
            languages     = args.languages,
            n_intents     = args.n_intents,
            n_variants    = args.n_variants,
            n_runs        = args.n_runs,
            output_dir    = args.output_dir,
            brands_only   = brands_list,
            top_n_brands  = args.top_n,
            skip_agent0   = args.skip_agent0,
            skip_agent1   = args.skip_agent1,
            skip_agent2   = args.skip_agent2,
            is_colab      = args.colab,
            location      = args.location,
        )
