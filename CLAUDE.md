# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment

**Always use Anaconda Python** — the system `python` points to MSYS2 and is missing packages.

```bash
# Run pipeline / tests
/c/Users/ayoun/anaconda3/python.exe pipeline.py --query "Restaurants in Sfax"
/c/Users/ayoun/anaconda3/python.exe -m pytest tests/ -v

# Run a single test file or test
/c/Users/ayoun/anaconda3/python.exe -m pytest tests/test_agent1.py -v
/c/Users/ayoun/anaconda3/python.exe -m pytest tests/test_graph_nodes.py::test_name -v

# Streamlit dashboard
/c/Users/ayoun/anaconda3/python.exe -m streamlit run web_interface/dashboard.py

# MCP server (stdio mode)
/c/Users/ayoun/anaconda3/python.exe mcp_server.py
```

## Common CLI Invocations

```bash
# Primary mode (LangGraph)
python pipeline.py --query "Restaurants in Sfax"
python pipeline.py --query "Best cafes in Tunis" --top_n 5

# Eco mode — minimal tokens (3–6 LLM tasks instead of default 12)
python pipeline.py --query "Restaurants in Sfax" --n_runs 1 --max_prompts_per_intent 1

# Rate-limit-safe run
MAX_CONCURRENT_LLM=2 python pipeline.py --query "Restaurants in Sfax"

# Legacy mode
python pipeline.py --domain "Tunisian restaurants" --location "Tunisia"
python pipeline.py --brands_only "brand1,brand2"
```

## Architecture

### Data Flow

```
pipeline.py → LangGraph StateGraph (graph/pipeline_graph.py)
    → parse_query_node          NL → {domain, location}
    → agent0_node               Prompt generation with self-reflection
    → agent1_load_node          prompts list prep
    → [FAN-OUT] agent1_llm_query_node × (prompts × models × runs)   ← async
    → agent1_aggregate_node     merge raw_responses
    → agent1_extract/enrich/clean/metrics nodes
    → [FAN-OUT] agent2_scrape_node × brand                          ← sync ThreadPool
    → export_node               write geo_output/
```

### Key Files

| File | Role |
|------|------|
| `pipeline.py` | Entry point; builds initial GEOState and calls `graph.ainvoke()` |
| `graph/state.py` | `GEOState` TypedDict — the single state passed through all nodes |
| `graph/pipeline_graph.py` | StateGraph assembly; `route_llm_queries()` and `route_brand_scraping()` fan-out routers |
| `graph/nodes/agent1_llm_query_node.py` | **Async** fan-out target — one LLM call per (prompt × model × run) |
| `core/llm_client.py` | ALL LLM calls — `query_llm`, `async_query_llm`, `query_provider`; semaphore; retry queue |
| `core/query_parser.py` | NL → `{domain, location}` via LLM → regex → fallback |
| `agents/agent1_geo_analyser.py` | Entity extraction, fuzzy dedup, LLM arbitration, GEO scoring |
| `agents/agent2_social_scraper.py` | Social scraping per brand (SerpAPI/Apify/Selenium/Instaloader) |
| `mcp_server.py` | FastMCP server — 6 tools, 3 resources (`geo://overview`, `geo://models`, `geo://last-run`) |

### GEOState Fan-in Fields

Three fields use `Annotated[list, operator.add]` for LangGraph automatic fan-in merging:
- `raw_responses` — accumulates LLM query results
- `profiles` — accumulates Agent 2 social profiles
- `errors` / `warnings` — accumulate diagnostics

### Parallelism Controls (env vars)

```env
MAX_CONCURRENT_LLM=5      # asyncio.Semaphore for Groq calls (default 5)
MAX_CONCURRENT_BRANDS=3   # parallel brand scrapes (default 3)
```

### LLM Routing

All LLM calls go through `core/llm_client.py`. Use `query_provider(provider, model, prompt, ...)` for provider-agnostic calls. Model constants (`MODEL_FAST`, `MODEL_STRONG`, `MODEL_COMPOUND`, etc.) are read from env vars with defaults set in `llm_client.py`.

### Agent 1 Cleaning Pipeline

Step 5 runs four phases in sequence:
- **Phase A** — Rule-based normalization
- **Phase B** — Fuzzy string clustering (rapidfuzz)
- **Phase C** — LLM arbitration (`MODEL_STRONG`) via `_arbitration_system(domain)`
- **Phase D** — Self-reflection via `_reflection_system(domain)`

### Agent 2 Scraping Fallback Chain

Each step has a primary + fallback: SerpApi → OSM, Instaloader → Apify, DDG → Apify. Selenium (`step4`) is gated by `_SELENIUM_AVAILABLE` flag at module top and skips gracefully if Chrome is absent.

## Groq Rate Limits (Free Tier)

Default load = 6 prompts × 2 models × 1 run = **12 requests/burst**.

If 429 errors occur: use `--n_runs 1`, set `MAX_CONCURRENT_LLM=2`, or run eco mode.
Fatal errors (`invalid api key`, `model not found`) are not retried — check `.env`.

## Outputs

All pipeline outputs land in `geo_output/` (git-ignored). The main analytical output is `entity_features_global.csv`. Pipeline run metadata is in `pipeline_summary.json`.
