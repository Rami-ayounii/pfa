# GEO Pipeline — Full Architecture
> **Last updated:** 2026-04-05 (Session 8)
> Single source of truth for the pipeline design. Update this file whenever the architecture changes.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Full Pipeline Flow](#2-full-pipeline-flow)
3. [Node Reference](#3-node-reference)
4. [Model Assignments](#4-model-assignments)
5. [Data Flow — GEOState](#5-data-flow--geostate)
6. [Parallelism Design](#6-parallelism-design)
7. [Project File Structure](#7-project-file-structure)
8. [Configuration Reference](#8-configuration-reference)
9. [CLI Usage](#9-cli-usage)
10. [Dashboard](#10-dashboard--web_interfacedashboardpy)
11. [Known Issues & Limits](#11-known-issues--limits)

---

## 1. Overview

**GEO** (Generative Engine Optimization) pipeline — measures F&B brand visibility in LLM-generated outputs.

```
Single natural-language query  →  structured brand visibility report
"Restaurants in Sfax"          →  features.csv + profiles.csv + summary.json
```

**Stack:**
- Orchestration: LangGraph `StateGraph` with `Send` fan-out
- LLM provider: Groq (sync + async) + OpenRouter (Agent 2 fallback)
- Scraping: SerpAPI · Apify · Selenium (Instagram)
- Logging: loguru · tqdm

---

## 2. Full Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  ENTRY POINT: pipeline.py                                            │
│  asyncio.run( graph.ainvoke(initial_state) )                         │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                  ╔══════▼══════╗
                  ║  LangGraph  ║   graph/pipeline_graph.py
                  ║ StateGraph  ║
                  ╚══════╤══════╝
                         │
      ┌──────────────────▼────────────────────┐
      │  1. parse_query_node                  │  core/query_parser.py
      │     "Restaurants in Sfax"             │
      │     Strategy 1 → LLM (MODEL_EXTRACTOR)│
      │     Strategy 2 → regex ("X in Y")     │
      │     Strategy 3 → fallback defaults    │
      │     ─────────────────────────────────  │
      │     OUT: domain="Restaurants"         │
      │          location="Sfax"              │
      └──────────────────┬────────────────────┘
                         │
      ┌──────────────────▼────────────────────┐
      │  2. agent0_node                       │  agents/agent0_prompt_generator.py
      │     model: compound-beta              │
      │     location: "Tunisia" (default)     │  ← hardcoded; all prompts reference Tunisia
      │                                       │
      │     Loop 1/2:                         │
      │       generate n_intents (in Tunisia) │
      │       generate n_variants per intent  │
      │         (reference Tunisian cities)   │
      │       self-reflect (quality score)    │
      │       if score < threshold → loop 2   │
      │     Loop 2/2: regenerate + reflect    │
      │     ─────────────────────────────────  │
      │     cap at MAX 10 prompts             │  ← token limit guard
      │     ─────────────────────────────────  │
      │     OUT: prompt_df (in-memory)        │
      │          geo_output/prompt_set_*.csv  │
      └──────────────────┬────────────────────┘
                         │
      ┌──────────────────▼────────────────────┐
      │  3. agent1_load_node                  │  graph/nodes/agent1_nodes.py
      │     prompt_df → prompts list          │
      │     validates required columns        │
      │     OUT: prompts [list of dicts]      │
      └──────────────────┬────────────────────┘
                         │
              route_llm_queries()   ← FAN-OUT ROUTER
              prompts × query_models × n_runs
              = up to 10 × 2 × 2 = 40 parallel tasks
                         │
         ┌───────────────┼──────────────────┐
         │               │                  │
   ┌─────▼──────┐  ┌─────▼──────┐    ┌─────▼──────┐
   │  4. llm    │  │  4. llm    │    │  4. llm    │   graph/nodes/agent1_llm_query_node.py
   │  query     │  │  query     │ …  │  query     │   ASYNC — runs concurrently
   │  node      │  │  node      │    │  node      │   MAX_CONCURRENT_LLM semaphore
   │            │  │            │    │            │
   │ llama-3.1  │  │ llama-3.3  │    │ llama-3.1  │
   │ max_tok    │  │ max_tok    │    │ run 2      │
   │ =1024      │  │ =8192      │    │            │
   └─────┬──────┘  └─────┬──────┘    └─────┬──────┘
         │               │                  │
         └───────────────┴──────────────────┘
                         │  fan-in: operator.add
                         │  raw_responses list grows with each result
                         │
      ┌──────────────────▼────────────────────┐
      │  5. agent1_aggregate_node             │  graph/nodes/agent1_nodes.py
      │     raw_responses list → DataFrame    │
      │     async CSV backup                  │
      │     OUT: entities_df (raw)            │
      └──────────────────┬────────────────────┘
                         │
      ┌──────────────────▼────────────────────┐
      │  6. agent1_extract_node               │  agents/agent1_geo_analyser.py step3
      │     model: MODEL_EXTRACTOR (llama-8b) │
      │     NER: extract restaurant names     │
      │     from each LLM response            │
      │     OUT: entities_df                  │
      └──────────────────┬────────────────────┘
                         │
      ┌──────────────────▼────────────────────┐
      │  7. agent1_enrich_node                │  agents/agent1_geo_analyser.py step4
      │     add: cuisine, city, category      │
      │     add: position (rank in response)  │
      │     OUT: enriched_df                  │
      └──────────────────┬────────────────────┘
                         │
      ┌──────────────────▼────────────────────┐
      │  8. agent1_clean_node                 │  agents/agent1_geo_analyser.py step5
      │     model: MODEL_STRONG (llama-70b)   │
      │     Phase A: fuzzy string clustering  │
      │     Phase B: alias deduplication      │
      │     Phase C: LLM arbitration          │
      │     Phase D: self-reflection          │
      │     OUT: clean_df                     │
      └──────────────────┬────────────────────┘
                         │
      ┌──────────────────▼────────────────────┐
      │  9. agent1_metrics_node               │  agents/agent1_geo_analyser.py step6
      │     GEO features per brand:           │
      │       - visibility_score              │
      │       - citation_count                │
      │       - response_frequency            │
      │       - prompt_coverage               │
      │       - stability_score               │
      │     extract top-N brand list          │
      │     OUT: features_df, global_df,      │
      │          brands [list]                │
      └──────────────────┬────────────────────┘
                         │
              route_brand_scraping()  ← FAN-OUT ROUTER
              one Send per brand
              if no brands → skip to export_node
                         │
         ┌───────────────┼──────────────────┐
         │               │                  │
   ┌─────▼──────┐  ┌─────▼──────┐    ┌─────▼──────┐
   │ 10. scrape │  │ 10. scrape │ …  │ 10. scrape │   graph/nodes/agent2_scrape_node.py
   │  node      │  │  node      │    │  node      │   SYNC — ThreadPoolExecutor
   │  brand A   │  │  brand B   │    │  brand N   │   MAX_CONCURRENT_BRANDS semaphore
   │            │  │            │    │            │
   │ step1 Maps │  │ step1 Maps │    │ step1 Maps │
   │ step2 Trip │  │ step2 Trip │    │ step2 Trip │
   │ step3 FB   │  │ step3 FB   │    │ step3 FB   │
   │ step4 IG   │  │ step4 IG   │    │ step4 IG   │
   │ step5 hall.│  │ step5 hall.│    │ step5 hall.│
   │ step6 refl.│  │ step6 refl.│    │ step6 refl.│
   │ step7 prof.│  │ step7 prof.│    │ step7 prof.│
   └─────┬──────┘  └─────┬──────┘    └─────┬──────┘
         │               │                  │
         └───────────────┴──────────────────┘
                         │  fan-in: operator.add
                         │  profiles list grows with each SocialProfile
                         │
      ┌──────────────────▼────────────────────┐
      │  11. export_node                      │  graph/nodes/export_node.py
      │     writes to geo_output/:            │
      │       features.csv                    │
      │       global_metrics.csv             │
      │       social_profiles.csv            │
      │       pipeline_summary.json          │
      │     prints summary banner            │
      └──────────────────┬────────────────────┘
                         │
                        END
```

---

## 3. Node Reference

| # | Node | File | Type | Role |
|---|------|------|------|------|
| 1 | `parse_query_node` | `graph/nodes/parse_node.py` | Sync | NL → {domain, location} |
| 2 | `agent0_node` | `graph/nodes/agent0_node.py` | Sync | Prompt set generation (compound-beta) |
| 3 | `agent1_load_node` | `graph/nodes/agent1_nodes.py` | Sync | prompt_df → prompts list |
| 4 | `agent1_llm_query_node` | `graph/nodes/agent1_llm_query_node.py` | **Async** | One LLM call per (prompt × model × run) |
| 5 | `agent1_aggregate_node` | `graph/nodes/agent1_nodes.py` | Sync | Merge raw_responses → DataFrame |
| 6 | `agent1_extract_node` | `graph/nodes/agent1_nodes.py` | Sync | NER entity extraction |
| 7 | `agent1_enrich_node` | `graph/nodes/agent1_nodes.py` | Sync | Category/location enrichment |
| 8 | `agent1_clean_node` | `graph/nodes/agent1_nodes.py` | Sync | Fuzzy dedup + LLM arbitration |
| 9 | `agent1_metrics_node` | `graph/nodes/agent1_nodes.py` | Sync | GEO scoring → brand list |
| 10 | `agent2_scrape_node` | `graph/nodes/agent2_scrape_node.py` | Sync | Full social scrape for one brand |
| 11 | `export_node` | `graph/nodes/export_node.py` | Sync | Write all outputs |

**Fan-out routers:**

| Router | Triggers | Target node | Parallelism |
|--------|----------|-------------|-------------|
| `route_llm_queries()` | after `agent1_load_node` | `agent1_llm_query_node` | prompts × models × runs |
| `route_brand_scraping()` | after `agent1_metrics_node` | `agent2_scrape_node` (or skip to export) | one per brand |

---

## 4. Model Assignments

| Stage | Constant | Default | Reason |
|-------|----------|---------|--------|
| Query parsing | `MODEL_EXTRACTOR` | `llama-3.1-8b-instant` | Fast structured extraction |
| Prompt generation (Agent 0) | `MODEL_COMPOUND` | `groq/compound` | Groq compound agentic system |
| LLM queries — fast (Agent 1 fan-out) | `MODEL_FAST` | `llama-3.1-8b-instant` | Fastest + cheapest, `max_tokens=1024` |
| LLM queries — multilingual (Agent 1 fan-out) | `MODEL_QWEN` | `qwen/qwen3-32b` | 100+ langs (fr/ar), `max_tokens=2048` |
| Entity cleaning / reflection | `MODEL_STRONG` | `llama-3.3-70b-versatile` | Accurate deduplication |
| Agent 2 reflection | OpenRouter | `meta-llama/llama-3.1-8b-instruct:free` | Brand profile enrichment |

**Additional model constants (available for `query_models` override):**

| Constant | Default model ID | Notes |
|----------|-----------------|-------|
| `MODEL_STRONG` | `llama-3.3-70b-versatile` | High-quality, higher RPM cost |
| `MODEL_KIMI` | `moonshotai/kimi-k2-instruct-0905` | Moonshot Kimi K2 — ⚠ verify Groq availability |
| `MODEL_LLAMA4` | `meta-llama/llama-4-scout-17b-16e-instruct` | Llama 4 Scout MoE (preview) |
| `MODEL_GPTOSS` | `openai/gpt-oss-20b` | Lightweight reasoning model |
| `MODEL_ANALYST` | `openai/gpt-oss-120b` | Analyst/cleaning tasks |

**Token limits per role (`_model_params` in `core/llm_client.py`):**

| Role | Small model (`8b`/`instant`) | Large model |
|------|------------------------------|-------------|
| `query` | `max_tokens=1024` | `max_tokens=2048` ← reduced from 8192 |
| `extractor` | `max_tokens=1024` | `max_tokens=1024` |
| `analyst` | `max_tokens=8192` | `max_tokens=8192` |

Override any model via `.env`:
```env
MODEL_FAST=llama-3.1-8b-instant
MODEL_QWEN=qwen/qwen3-32b
MODEL_STRONG=llama-3.3-70b-versatile
MODEL_COMPOUND=groq/compound
MODEL_EXTRACTOR=llama-3.1-8b-instant
MODEL_KIMI=moonshotai/kimi-k2-instruct-0905
MODEL_LLAMA4=meta-llama/llama-4-scout-17b-16e-instruct
```

---

## 5. Data Flow — GEOState

`GEOState` (TypedDict) is the single shared state object passed through every node.

```
GEOState
├── INPUT
│   ├── query          str         "Restaurants in Sfax"
│   ├── domain         str         "Restaurants"          ← set by parse_query_node
│   └── location       str         "Sfax"                 ← set by parse_query_node
│
├── AGENT 0
│   └── prompt_df      DataFrame   10 rows max, in-memory
│
├── AGENT 1 INTERMEDIATES
│   ├── prompts        list        [{"prompt_id", "prompt_text", "intent_id", ...}]
│   ├── raw_responses  list ⊕      fan-in accumulator (operator.add)
│   ├── entities_df    DataFrame   after extract
│   ├── enriched_df    DataFrame   after enrich
│   ├── clean_df       DataFrame   after dedup/arbitration
│   ├── features_df    DataFrame   per-prompt GEO scores
│   ├── global_df      DataFrame   aggregated GEO scores
│   └── brands         list        top-N brand names → Agent 2
│
├── AGENT 2
│   └── profiles       list ⊕      fan-in accumulator (SocialProfile objects)
│
├── CONFIG (carried through all nodes)
│   ├── n_intents      int         default 3
│   ├── n_variants     int         default 4
│   ├── n_runs         int         default 2
│   ├── languages      list        default ["fr"]
│   ├── query_models   list        default [MODEL_FAST, MODEL_STRONG]
│   ├── analyst_model  str         default MODEL_COMPOUND (Agent 0)
│   ├── output_dir     str         default "geo_output"
│   └── top_n_brands   int         default 10
│
├── API CLIENTS / KEYS
│   ├── groq_client    Groq        pre-built client instance
│   ├── mistral_key    str
│   ├── openrouter_key str
│   ├── serpapi_key    str
│   ├── apify_token    str
│   ├── ig_user/pass   str
│   └── is_colab       bool
│
└── DIAGNOSTICS
    ├── errors         list ⊕      fan-in accumulator
    └── warnings       list ⊕      fan-in accumulator

⊕ = Annotated[list, operator.add] — LangGraph merges automatically on fan-in
```

---

## 6. Parallelism Design

### Fan-out 1 — LLM Queries

```
Default: 6 prompts × 2 models × 1 run = 12 async tasks   ← 70% reduction from previous 40
Maximum: 10 prompts × N models × N runs (fully configurable)
```

**Prompt reduction pipeline (agent0_node → route_llm_queries):**
1. Agent 0 generates `n_intents × n_variants` prompts (up to ~24)
2. **Stratified sampling**: keep `max_prompts_per_intent` (default 2) per intent → ~6 prompts
3. **Text deduplication**: router drops identical `prompt_text` values before fan-out
4. Absolute cap: 10 prompts

- Each task is an independent `async def agent1_llm_query_node(state)`
- Controlled by `asyncio.Semaphore(MAX_CONCURRENT_LLM)` — default 5
- Results merged via `raw_responses: Annotated[list, operator.add]`
- Failed tasks pushed to retry queue (`drain_retry_queue()`)
- Fatal errors (`invalid api key`, `model not found`) return immediately — not retried

### Fan-out 2 — Brand Scraping

```
top_n_brands tasks (default 10), one per brand
```

- Each task is a sync `def agent2_scrape_node(state)` running in LangGraph's ThreadPoolExecutor
- Controlled by `MAX_CONCURRENT_BRANDS` env var — default 3
- Results merged via `profiles: Annotated[list, operator.add]`
- If `brands = []` → router skips directly to `export_node`

### Concurrency Controls

```env
MAX_CONCURRENT_LLM=5      # parallel async Groq calls
MAX_CONCURRENT_BRANDS=3   # parallel brand scrapes
```

---

## 7. Project File Structure

```
pfa agentic/
│
├── pipeline.py                    ← entry point (LangGraph + legacy modes)
├── mcp_server.py                  ← ✨ FastMCP server (6 tools, 3 resources)
├── requirements.txt               ← updated: mcp>=1.0.0, no flask
├── .env                           ← API keys (git-ignored)
├── .env.example                   ← template with all keys
├── .gitignore                     ← updated: no mcp_servers, +llm_cache.json
├── README.md                      ← updated: MCP single-server architecture
├── ARCHITECTURE.md                ← this file
│
├── core/
│   ├── llm_client.py              ← ALL LLM calls (Groq sync/async, Mistral, OpenRouter)
│   └── query_parser.py            ← NL → {domain, location}
│
├── agents/
│   ├── agent0_prompt_generator.py ← intent + variant generation with self-reflection
│   ├── agent1_geo_analyser.py     ← entity extraction, enrichment, dedup, GEO scoring
│   └── agent2_social_scraper.py   ← social media scraping per brand
│
├── graph/
│   ├── state.py                   ← GEOState TypedDict
│   ├── pipeline_graph.py          ← StateGraph assembly + Send routers
│   └── nodes/
│       ├── parse_node.py          ← node 1
│       ├── agent0_node.py         ← node 2
│       ├── agent1_llm_query_node.py ← node 4 (async fan-out target)
│       ├── agent1_nodes.py        ← nodes 3, 5, 6, 7, 8, 9
│       ├── agent2_scrape_node.py  ← node 10 (sync fan-out target)
│       └── export_node.py         ← node 11
│
├── web_interface/
│   └── dashboard.py               ← Streamlit v3 dashboard (Session 7)
│
├── tests/
│   ├── conftest.py
│   ├── test_llm_client.py
│   ├── test_agent0.py
│   ├── test_agent1.py
│   ├── test_agent2.py
│   ├── test_pipeline.py
│   ├── test_query_parser.py       ← 9 tests
│   └── test_graph_nodes.py        ← 15 tests
│
└── geo_output/                    ← generated outputs (git-ignored)
    ├── prompt_set_*.csv
    ├── raw_responses.csv
    ├── extracted_entities.csv
    ├── enriched_entities.csv
    ├── clean_entities.csv
    ├── cleaning_log.csv
    ├── entity_features.csv
    ├── entity_features_global.csv ← main analytical output
    ├── pipeline_summary.json
    └── agent2_output/
        ├── social_profiles.csv
        ├── hallucination_flags.csv
        └── audit_log.json
```

### Deleted in Session 8
- `mcp_servers/` — 5 old MCP servers (mcp_agent0_server.py, etc.)
- `deployment/` — Docker configs for old servers
- `scripts/` — launcher scripts (start_mcp_servers.sh, .ps1)
- `docs/` — MCP integration guides (replaced by inline README.md + ARCHITECTURE.md)
- `web_interface/app.py` — Flask app (replaced by Streamlit dashboard v3)
- `.sixth/` — empty framework placeholder
- `test_client.py` — experimental FastMCP client
- `CONVERSATION_LOG.md` — 52 KB historical log
- `LANGGRAPH_REFACTOR.md` — completed refactor notes
- `llm_cache.json` — cache from previous runs

---

## 8. Configuration Reference

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `MISTRAL_API_KEY` | — | Optional. Mistral entity extraction |
| `OPENROUTER_API_KEY` | — | Optional. Agent 2 brand reflection |
| `SERPAPI_KEY` | — | Optional. Google Maps data (Agent 2) |
| `APIFY_API_TOKEN` | — | Optional. TripAdvisor / FB / Instagram (Agent 2) |
| `IG_USERNAME` | — | Optional. Instagram scraping |
| `IG_PASSWORD` | — | Optional. Instagram scraping |
| `MODEL_FAST` | `llama-3.1-8b-instant` | Fast query model (production) |
| `MODEL_STRONG` | `llama-3.3-70b-versatile` | Strong query + analyst model (production) |
| `MODEL_QWEN` | `qwen/qwen3-32b` | Multilingual query model — fr/ar (preview) |
| `MODEL_KIMI` | `moonshotai/kimi-k2-instruct-0905` | Moonshot Kimi K2 (⚠ verify availability) |
| `MODEL_LLAMA4` | `meta-llama/llama-4-scout-17b-16e-instruct` | Llama 4 Scout MoE (preview) |
| `MODEL_GPTOSS` | `openai/gpt-oss-20b` | Lightweight reasoning model (production) |
| `MODEL_COMPOUND` | `groq/compound` | Agent 0 prompt generation model (production) |
| `MODEL_EXTRACTOR` | `llama-3.1-8b-instant` | Query parsing + extraction model |
| `MAX_CONCURRENT_LLM` | `5` | Max parallel async LLM calls |
| `MAX_CONCURRENT_BRANDS` | `3` | Max parallel brand scrapes |
| `GROQ_BASE_URL` | — | Override to use any OpenAI-compatible endpoint |

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--query` | — | Natural-language input (LangGraph mode) |
| `--domain` | `Tunisian restaurants` | Domain (legacy mode) |
| `--location` | `Tunisia` | Location (legacy mode) |
| `--n_intents` | `3` | Intent types for Agent 0 |
| `--n_variants` | `4` | Prompt variants per intent |
| `--n_runs` | `1` | LLM runs per (prompt × model) — increase for more stable GEO estimates |
| `--max_prompts_per_intent` | `2` | Max prompt variants kept per intent (stratified sampling) |
| `--top_n` | `10` | Top-N brands passed to Agent 2 |
| `--languages` | `fr` | Query languages (comma-separated) |

---

## 9. CLI Usage

```bash
# ── LangGraph mode (primary) — default: 12 LLM tasks ─────────────────────────
python pipeline.py --query "Restaurants in Sfax"
python pipeline.py --query "Best cafes in Tunis" --top_n 5

# ── Eco mode — absolute minimum tokens (3–6 LLM tasks) ──────────────────────
python pipeline.py --query "Restaurants in Sfax" --n_runs 1 --max_prompts_per_intent 1

# ── More stable GEO estimates — increased sampling ───────────────────────────
python pipeline.py --query "Restaurants in Sfax" --n_runs 2 --max_prompts_per_intent 3

# ── Custom models — use specific Groq models ─────────────────────────────────
# (set MODEL_KIMI / MODEL_LLAMA4 in .env to override defaults)
# Then override query_models via Python API:
#   run_graph_pipeline("...", query_models=["llama-3.1-8b-instant", "moonshotai/kimi-k2-instruct-0905"])

# ── Legacy mode (backward-compatible) ────────────────────────────────────────
python pipeline.py --domain "Tunisian restaurants" --location "Tunisia"
python pipeline.py --brands_only "Dar El Jeld,Le Corsaire,Fondouk El Attarine"

# ── Rate-limit-safe run ───────────────────────────────────────────────────────
MAX_CONCURRENT_LLM=2 python pipeline.py --query "Restaurants in Sfax"

# ── Run tests ─────────────────────────────────────────────────────────────────
python -m pytest tests/ -v
```

> **Python env:** Always use the Anaconda environment.
> If using the shell directly: `/c/Users/ayoun/anaconda3/envs/pfa/python.exe`

---

## 10. Dashboard — `web_interface/dashboard.py` (v3 — Session 7)

### Run command
```bash
C:\Users\ayoun\anaconda3\python.exe -m streamlit run web_interface/dashboard.py
```

### CSS architecture (v3 — critical)
**Rule:** every custom HTML block carries its own `<style>` tag inside `components.html()`.
`st.markdown("<style>")` is **not** used for custom component styling — it does not reliably scope across separate `st.markdown()` calls.

| Constant | Used by | Content |
|----------|---------|---------|
| `_WORKFLOW_CSS` | `render_agent_workflow()` | agent cards, grid, tool tags, connectors |
| `_PIPELINE_CSS` | `render_pipeline_flow()` | node cards, legend, `@keyframes pulse` |

### Real-time architecture
```
pipeline.py subprocess
    └─► stdout+stderr → reader thread → queue.Queue
                                            │
                        session_state ← _drain_queue()
                             │
                        st.rerun() every 400 ms  (while is_running)
```

### Layout (top → bottom)
| Section | Always visible? | Notes |
|---------|----------------|-------|
| Header | ✅ | Title + caption |
| **Query input** | ✅ | Text input + ▶ Run button in main area |
| **Agent workflow** | ✅ | 3-card grid — per-agent description |
| Live node status | ✅ | 11 HTML node cards with state animation |
| KPI bar | After first run | 5 metrics |
| Tabs | ✅ | Live Console / Agent 0 / Agent 1 / Agent 2 / Downloads |

### Query input (v2)
- Prominent `st.text_input` at the top of main content — no CLI needed
- Placeholder: `"Restaurants in Sfax · Best cafes in Tunis · Patisseries in Monastir"`
- `st.columns([5, 1])` — input + **▶ Run** primary button
- Sidebar = parameters only (n_runs, top_n, n_intents, max/intent, languages, python path)
- Legacy domain/location mode in collapsed `st.expander` inside sidebar

### Agent workflow cards (`render_agent_workflow()`)
CSS grid: `grid-template-columns: 1fr 52px 1fr 52px 1fr` (3 cards + 2 arrows)

Each card: agent badge · icon + title · subtitle · step bullets · tool tags · IN/OUT

| Card | Tools shown | Data artifact |
|------|------------|---------------|
| Agent 0 — Prompt Generator | Groq Compound, LLM reflection, fr/ar/en | → prompts CSV |
| Agent 1 — GEO Analyser | async fan-out, Groq Fast/Strong, Qwen3-32B | → top-N brands |
| Agent 2 — Social Scraper | DuckDuckGo, SerpApi, Selenium, Apify, OpenRouter | → profiles CSV |

### Pipeline node status
11 node cards in `.pipeline-wrap` (overflow-x:auto). State classes:

| State | Border | Background | Label colour | Extra |
|-------|--------|------------|-------------|-------|
| `pending` | `#e2e8f0` | `#f8fafc` | `#94a3b8` | opacity .55 |
| `running` | `#3b82f6` | `#eff6ff` | `#2563eb` bold | pulse-blue animation |
| `done` | `#22c55e` | `#f0fdf4` | `#15803d` bold | — |
| `error` | `#ef4444` | `#fef2f2` | `#dc2626` bold | — |
| `skip` | `#f59e0b` | `#fffbeb` | `#b45309` bold | — |

Node transitions driven by `NODE_TRIGGERS`: `list[tuple[str, list[str]]]` matching log substrings.

### Tabs
| Tab | Content |
|-----|---------|
| Live Console | Dark terminal via `components.html`, auto-scroll JS, filter dropdown |
| Agent 0 | Prompt set table, filterable by intent |
| Agent 1 | Brand visibility bar chart (Plotly) + global metrics table |
| Agent 2 | Authority bar chart + radar chart + hallucination metrics |
| Downloads | `st.download_button` per file in `geo_output/` |

### KPI bar
Brands Scraped · Avg Authority · Top Brand · Prompts Queried · Run Status

### Charts (Plotly)
- Horizontal bar — `social_authority_score` per brand (`color_continuous_scale="Greens"`)
- Radar — 4 scoring dimensions per brand (25 pts each): Review Quality, Social Reach, Web Completeness, Cross-Platform

---

## 11. Known Issues & Limits

### Groq Free-Tier Rate Limits

| Model | TPM | RPM |
|-------|-----|-----|
| `llama-3.1-8b-instant` | 6,000 | 30 |
| `llama-3.3-70b-versatile` | 6,000 | 30 |
| `compound-beta` | ~12,000 | 30 |

**Default load (Session 3):** 6 prompts × 2 models × 1 run = **12 requests/burst** (70% reduction)
**Worst case:** 10 prompts × N models × N runs = configurable

**Mitigations in place:**
| Mitigation | Where | Effect |
|------------|-------|--------|
| `max_tokens=1024` for 8b models | `_model_params()` | Prevents 413 per-request size limit |
| Prompt cap at 10 | `agent0_node.py` | Limits burst to 40 max tasks |
| `asyncio.Semaphore(MAX_CONCURRENT_LLM)` | `async_query_llm()` | Controls concurrency |
| Exponential back-off on 429 | `async_query_llm()` retry loop | Auto-recovery from rate limits |
| Fatal error early return | `async_query_llm()` | No retry for auth/model errors |

**If 429 errors persist:**
```bash
# Option 1 — halve requests
python pipeline.py --query "..." --n_runs 1

# Option 2 — reduce concurrency
MAX_CONCURRENT_LLM=2 python pipeline.py --query "..."

# Option 3 — single model
python pipeline.py --query "..." --n_runs 1
# (then set MODEL_STRONG="" in .env to use only fast model)
```

### Other Known Limitations

| Issue | Status | Notes |
|-------|--------|-------|
| 413 Request too large (llama-3.1-8b) | ✅ Mitigated | `max_tokens` capped at 1024 |
| 429 Rate limit under full load | ⚠️ Ongoing | Use `--n_runs 1` workaround |
| Selenium Chrome binary not found | ✅ Fixed | `_find_chrome_binary()` checks Windows paths; graceful skip if absent |
| Selenium Instagram scraping | Graceful fallback | Skipped if Selenium not installed |
| `compound-beta` token cost | Monitor | Higher quality but uses more TPM |
| Agent 0 location context | ✅ Fixed | `location="Tunisia"` threaded through all 3 prompt steps |

---

*Update this file when: adding nodes, changing models, modifying GEOState fields, or changing parallelism strategy.*
