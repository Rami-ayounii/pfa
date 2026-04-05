"""
mcp_server.py
=============
Single MCP server for the GEO Multi-Agent Pipeline.

Exposes the LangGraph pipeline and individual agents as tools
for Claude Desktop / VS Code integration.

Uses FastMCP API with stdio transport.

Usage:
    python mcp_server.py

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "geo-pipeline": {
          "command": "C:\\Users\\ayoun\\anaconda3\\python.exe",
          "args": ["D:\\Rami IDSD\\Projects\\pfa agentic\\mcp_server.py"]
        }
      }
    }
"""

import os
import sys
import json
import dataclasses
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from groq import Groq
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "geo-pipeline",
    instructions=(
        "GEO Multi-Agent Pipeline for measuring F&B brand visibility in LLM outputs. "
        "Use run_pipeline for the full LangGraph workflow, or call individual agent tools."
    ),
)

# ── Imports from project ─────────────────────────────────────────────────────
from core.llm_client import (
    TOKEN_USAGE, MODEL_FAST, MODEL_STRONG, MODEL_COMPOUND, MODEL_QWEN,
)

OUTPUT_DIR = str(ROOT / "geo_output")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _env() -> dict:
    return {
        "groq_key":       os.environ.get("GROQ_API_KEY", ""),
        "mistral_key":    os.environ.get("MISTRAL_API_KEY", ""),
        "openrouter_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "serpapi_key":    os.environ.get("SERPAPI_KEY", ""),
        "apify_token":    os.environ.get("APIFY_API_TOKEN", ""),
        "ig_user":        os.environ.get("IG_USERNAME", ""),
        "ig_pass":        os.environ.get("IG_PASSWORD", ""),
    }


def _groq_client() -> Groq | None:
    key = os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None


def _serialize(obj):
    """Make an object JSON-safe (DataFrames, dataclasses, etc.)."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()
                if k not in ("groq_client",)}
    return obj


def _json(obj) -> str:
    """Serialize to indented JSON string."""
    return json.dumps(_serialize(obj), indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def run_pipeline(
    query: str,
    languages: str = "fr",
    n_intents: int = 3,
    n_variants: int = 4,
    n_runs: int = 1,
    max_prompts_per_intent: int = 2,
    top_n_brands: int = 10,
    output_dir: str = "geo_output",
) -> str:
    """Run the full GEO pipeline via LangGraph.

    Accepts a natural-language query (e.g. "Restaurants in Sfax") and executes:
    parse → Agent 0 (prompts) → Agent 1 (LLM fan-out, entity extraction, GEO metrics)
    → Agent 2 (social scraping) → export.

    May take several minutes depending on the number of prompts and brands.
    """
    from pipeline import run_graph_pipeline

    try:
        lang_list = [l.strip() for l in languages.split(",")]
        result = run_graph_pipeline(
            query=query,
            languages=lang_list,
            n_intents=n_intents,
            n_variants=n_variants,
            n_runs=n_runs,
            max_prompts_per_intent=max_prompts_per_intent,
            top_n_brands=top_n_brands,
            output_dir=output_dir,
        )

        summary = {
            "status": "success",
            "query": result.get("query"),
            "domain": result.get("domain"),
            "location": result.get("location"),
            "prompts_generated": len(result.get("prompt_df", []))
                if result.get("prompt_df") is not None else 0,
            "entities_found": len(result.get("global_df", []))
                if result.get("global_df") is not None else 0,
            "brands_profiled": len(result.get("profiles", [])),
            "errors": result.get("errors", []),
            "warnings": result.get("warnings", []),
            "token_usage": TOKEN_USAGE.copy(),
        }
        return _json(summary)

    except Exception as e:
        return _json({"status": "error", "error": str(e)})


@mcp.tool()
def generate_prompts(
    domain: str = "Tunisian restaurants",
    languages: str = "fr",
    n_intents: int = 3,
    n_variants: int = 4,
    max_reflection_loops: int = 2,
    output_dir: str = "geo_output",
) -> str:
    """Run Agent 0 only — generate diverse GEO prompts for a domain.

    Returns the generated prompt set as JSON records.
    """
    from agents.agent0_prompt_generator import Agent0PromptGenerator

    try:
        lang_list = [l.strip() for l in languages.split(",")]
        agent = Agent0PromptGenerator(
            domain=domain,
            model=MODEL_COMPOUND,
            languages=lang_list,
            n_intents=n_intents,
            n_variants=n_variants,
            max_reflection_loops=max_reflection_loops,
            output_dir=output_dir,
            client=_groq_client(),
        )
        prompt_df = agent.run()
        return _json({
            "status": "success",
            "domain": domain,
            "prompts_count": len(prompt_df),
            "prompts": prompt_df.to_dict(orient="records"),
        })
    except Exception as e:
        return _json({"status": "error", "error": str(e)})


@mcp.tool()
def analyze_entities(
    prompt_set_path: str = "geo_output/prompt_set_Tunisian_restaurants.csv",
    domain: str = "Tunisian restaurants",
    n_runs: int = 1,
    output_dir: str = "geo_output",
) -> str:
    """Run Agent 1 only — GEO entity analysis on an existing prompt set.

    Requires a prompt_set CSV from Agent 0. Performs LLM querying,
    entity extraction, enrichment, cleaning, and GEO metric computation.
    """
    from agents.agent1_geo_analyser import Agent1GeoAnalyser

    try:
        agent = Agent1GeoAnalyser(
            prompt_set_path=prompt_set_path,
            output_dir=output_dir,
            n_runs=n_runs,
            groq_client=_groq_client(),
            domain=domain,
        )
        agent.run()
        # Read generated output files
        result = {"status": "success", "domain": domain}
        gf = Path(output_dir) / "entity_features_global.csv"
        if gf.exists():
            df = pd.read_csv(gf)
            result["entities_count"] = len(df)
            result["top_brands"] = df.nlargest(5, "stability_score")["canonical_entity"].tolist() if "stability_score" in df.columns else []
        return _json(result)
    except Exception as e:
        return _json({"status": "error", "error": str(e)})


@mcp.tool()
def scrape_brands(
    brands: str,
    location: str = "Tunisia",
    output_dir: str = "geo_output/agent2_output",
) -> str:
    """Run Agent 2 only — scrape social profiles for given brands.

    Pass brands as a comma-separated string (e.g. "Dar El Jeld,Le Golfe").
    """
    from agents.agent2_social_scraper import scrape_single_brand

    try:
        env = _env()
        brand_list = [b.strip() for b in brands.split(",") if b.strip()]
        profiles = []
        for brand in brand_list:
            p = scrape_single_brand(
                brand=brand,
                location=location,
                output_dir=output_dir,
                serpapi_key=env["serpapi_key"],
                apify_token=env["apify_token"],
                openrouter_key=env["openrouter_key"],
                groq_client=_groq_client(),
                ig_user=env["ig_user"],
                ig_pass=env["ig_pass"],
            )
            profiles.append(p)
        return _json({
            "status": "success",
            "brands_scraped": len(profiles),
            "profiles": profiles,
        })
    except Exception as e:
        return _json({"status": "error", "error": str(e)})


@mcp.tool()
def get_results(output_dir: str = "geo_output") -> str:
    """Read the last pipeline summary from pipeline_summary.json."""
    p = Path(output_dir) / "pipeline_summary.json"
    if not p.exists():
        return _json({"status": "no_data", "message": "No pipeline run found. Run the pipeline first."})
    return p.read_text(encoding="utf-8")


@mcp.tool()
def list_output_files(output_dir: str = "geo_output") -> str:
    """List all CSV and JSON output files in the output directory."""
    d = Path(output_dir)
    if not d.exists():
        return _json({"files": [], "message": "Output directory does not exist."})
    files = []
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.suffix in (".csv", ".json"):
            files.append({
                "path": str(f.relative_to(d)),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return _json({"files": files})


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

@mcp.resource("geo://overview")
def pipeline_overview() -> str:
    """GEO pipeline architecture and capabilities."""
    return _json({
        "name": "GEO Multi-Agent Pipeline",
        "description": "Measures F&B brand visibility in LLM outputs via Generative Engine Optimization",
        "agents": {
            "agent0": {"name": "Prompt Generator", "role": "Discover intents, generate diverse prompts"},
            "agent1": {"name": "GEO Analyser", "role": "Extract brands, compute GEO features, identify top brands"},
            "agent2": {"name": "Social Scraper", "role": "Profile brands via web scraping, assess social authority"},
        },
        "flow": "parse_query → Agent 0 → Agent 1 (parallel LLM fan-out) → Agent 2 (parallel brand scraping) → export",
        "orchestration": "LangGraph StateGraph with Send() fan-out",
    })


@mcp.resource("geo://models")
def available_models() -> str:
    """Available LLM models configured for the pipeline."""
    return _json({
        "MODEL_FAST": MODEL_FAST,
        "MODEL_STRONG": MODEL_STRONG,
        "MODEL_COMPOUND": MODEL_COMPOUND,
        "MODEL_QWEN": MODEL_QWEN,
        "query_defaults": [MODEL_FAST, MODEL_QWEN],
        "analyst_default": MODEL_COMPOUND,
    })


@mcp.resource("geo://last-run")
def last_run_info() -> str:
    """Last pipeline execution summary."""
    p = Path(OUTPUT_DIR) / "pipeline_summary.json"
    if not p.exists():
        return _json({"status": "no_data"})
    return p.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run(transport="stdio")
