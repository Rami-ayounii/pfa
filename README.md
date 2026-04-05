# GEO Multi-Agent System
### Brand Visibility Analysis via Generative Engine Optimization

---

## 📁 Project Organization

The project is organized for clean separation of concerns:

```
core/               # Shared LLM utilities
agents/             # Core agent implementations (Agent 0, 1, 2)
graph/              # LangGraph orchestration (StateGraph + nodes)
web_interface/      # Streamlit dashboard
tests/              # Unit and integration tests
geo_output/         # Pipeline outputs (generated at runtime)
```

**Key entry points:**
- CLI: `python pipeline.py --query "your query"`
- MCP: `python mcp_server.py` (for Claude Desktop, VS Code)
- Dashboard: `streamlit run web_interface/dashboard.py`

---

## What this is

This system measures how **visible a brand is inside AI-generated answers**.
It discovers which restaurant/F&B brands LLMs mention, how prominently they rank them,
and what social data validates or contradicts that visibility — all autonomously.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT 0 — Prompt Generator                                         │
│                                                                     │
│  Step 1 · Discover intent types for the domain           (LLM)     │
│  Step 2 · Generate n_variants prompts per intent/language (LLM)    │
│  Step 3 · Self-reflect on quality → regenerate if needed  (LLM)    │
│                                                                     │
│  Output: prompt_set_{domain}.csv                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ prompt_set CSV
┌──────────────────────────────▼──────────────────────────────────────┐
│  AGENT 1 — GEO Analyser                                             │
│                                                                     │
│  Step 1 · Load prompt set                                           │
│  Step 2 · Query LLMs (multi-model × multi-run, progressive save)   │
│  Step 3 · Extract brand entities per response       (Mistral/Groq)  │
│  Step 4 · Enrich: ranking position + description length             │
│  Step 5 · Clean: fuzzy dedup → LLM arbitration → self-reflection   │
│  Step 6 · Compute GEO features (mention rate, stability score…)    │
│                                                                     │
│  Output: raw_responses · extracted_entities · enriched_entities     │
│          clean_entities · cleaning_log                              │
│          entity_features · entity_features_global   ← key output   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ top-N brands by stability score
┌──────────────────────────────▼──────────────────────────────────────┐
│  AGENT 2 — Social Scraper & Brand Intelligence Synthesiser          │
│                                                                     │
│  Step 1  · DuckDuckGo URL resolution (Wikipedia + TripAdvisor)     │
│  Step 2a · Wikipedia summary + categories                           │
│  Step 2b · TripAdvisor rating (DDG snippet → Apify fallback)       │
│  Step 3  · Venue data: SerpApi Google Maps → OSM fallback           │
│  Step 4  · Google Maps via Selenium (rating, phone, socials)        │
│  Step 5a · Instagram: Instaloader (free) → Apify fallback           │
│  Step 5b · Facebook:  Apify → DDG parse fallback                   │
│  Step 6  · Hallucination checker (rules + LLM semantic)             │
│  Step 7  · LLM reflection: authority score + GEO expansion recs     │
│  Step 8  · Export CSV + JSON                                        │
│                                                                     │
│  Output: social_profiles · hallucination_flags · audit_log          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Model Context Protocol (MCP) Integration

The GEO system now exposes all agents via a **single, unified MCP server** (`mcp_server.py`), enabling integration with Claude Desktop, VS Code, and other AI assistants.

### MCP Capabilities

- 🤖 **Full LangGraph pipeline** — Run complete GEO analysis via `run_pipeline()` tool
- 🔬 **Individual agents** — Call Agent 0, 1, 2 separately for fine-grained control
- 📊 **Live data resources** — Query outputs as structured resources (`geo://overview`, `geo://models`, `geo://last-run`)
- 📈 **Monitor execution** — Track prompts generated, entities extracted, token usage
- 🔄 **Resume on failure** — Progressive saves allow safe restarts

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the MCP server (stdio mode for Claude Desktop / VS Code)
python mcp_server.py
```

### Claude Desktop Integration

Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "geo-pipeline": {
      "command": "python",
      "args": ["/path/to/pfa_agentic/mcp_server.py"]
    }
  }
}
```

Then in Claude, ask:
```
Use the geo-pipeline MCP server to run a GEO analysis for restaurants in Sfax
```

### Available Tools

| Tool | Purpose |
|------|---------|
| `run_pipeline()` | Full LangGraph orchestration (all 3 agents) |
| `generate_prompts()` | Agent 0: Prompt generation + self-reflection |
| `analyze_entities()` | Agent 1: Entity extraction + GEO features |
| `scrape_brands()` | Agent 2: Social profile scraping |
| `get_results()` | Read pipeline_summary.json |
| `list_output_files()` | List all CSVs/JSONs in output directory |

### Available Resources

| Resource | Purpose |
|----------|---------|
| `geo://overview` | Pipeline architecture + data flow |
| `geo://models` | Available LLM models (MODEL_FAST, MODEL_STRONG, etc.) |
| `geo://last-run` | Last pipeline_summary.json contents |

---

## 🏗️ Project Structure

```
.
├── core/
│   ├── llm_client.py                    # LLM abstraction (Groq, Mistral, etc.)
│   └── query_parser.py                  # NL query → {domain, location}
│
├── agents/
│   ├── agent0_prompt_generator.py       # Prompt generation + self-reflection
│   ├── agent1_geo_analyser.py           # Entity extraction + GEO features
│   └── agent2_social_scraper.py         # Social profile + venue scraping
│
├── graph/
│   ├── pipeline_graph.py                # LangGraph StateGraph + fan-out
│   ├── state.py                         # GEOState TypedDict
│   └── nodes/                           # 11 graph nodes
│
├── web_interface/
│   ├── dashboard.py                     # Streamlit v3 dashboard
│   └── (CSS embedded in dashboard.py)
│
├── tests/
│   ├── conftest.py                      # Shared fixtures
│   ├── test_llm_client.py
│   ├── test_agent0.py
│   ├── test_agent1.py
│   ├── test_agent2.py
│   ├── test_pipeline.py
│   ├── test_query_parser.py
│   └── test_graph_nodes.py
│
├── pipeline.py                          # Main CLI orchestrator
├── mcp_server.py                        # ✨ FastMCP server (6 tools, 3 resources)
├── requirements.txt
├── .env.example
├── .gitignore
├── ARCHITECTURE.md
├── README.md                            # This file
└── geo_output/                          # Runtime outputs
    ├── prompt_set_*.csv
    ├── raw_responses.csv
    ├── entity_features_global.csv       # Main analytical output
    └── agent2_output/
        ├── social_profiles.csv
        └── audit_log.json
```

---

## Key agentic patterns

| Pattern | Where used |
|---|---|
| **Self-reflection loop** | Agent 0 (quality score → regenerate), Agent 1 Step 5 Phase D, Agent 2 Step 7 |
| **Progressive saving + resume** | Agent 1 Steps 2, 3, 4 (crash-safe, restart from last record) |
| **Primary + fallback** | Agent 2 every step (e.g. SerpApi → OSM, Instaloader → Apify) |
| **LLM cache** | Agent 2 synthesis (avoid redundant API calls on re-runs) |
| **Multi-layer JSON parser** | All agents (markdown fences, trailing commas, qwen3 thinking blocks) |
| **Token tracking** | Global `TOKEN_USAGE` across all agents |
| **Hallucination detection** | Agent 2 Step 6: rules → LLM semantic cross-check → weighted confidence |
| **Agentic cleaning** | Agent 1 Step 5: A (rules) → B (fuzzy) → C (LLM) → D (reflect) → E (apply) |

---

## GEO features produced (entity_features_global.csv)

| Column | Description |
|---|---|
| `mention_rate` | Fraction of responses that mention this brand |
| `average_ranking_position` | Mean position when mentioned (1 = first) |
| `rank_variance` | Stability of ranking across runs |
| `top1_rate` | How often the brand appears first |
| `stability_score` | `mention_rate × 1/(1+rank_variance)` — composite GEO signal |
| `consistency_label` | stable / variable / unstable |
| `cross_model_rate` | Fraction of models that mention the brand |
| `prompt_type_response` | Mention rate broken down by intent type |
| `co_mention_rate` | How often each brand appears alongside others |

---

## Setup

```bash
pip install groq mistralai rapidfuzz ddgs wikipedia-api httpx \
            apify-client instaloader selenium webdriver-manager \
            google-search-results pandas tabulate
```

**Environment variables:**

```bash
export GROQ_API_KEY="gsk_..."
export MISTRAL_API_KEY="..."           # optional — Groq fallback used if absent
export OPENROUTER_API_KEY="sk-or-..."  # optional — for Agent 2 synthesis
export SERPAPI_KEY="..."               # optional — OSM fallback used if absent
export APIFY_API_TOKEN="..."           # optional — DDG fallback used if absent
export IG_USERNAME=""                  # optional — anonymous mode works for public accounts
export IG_PASSWORD=""
```

---

## Usage

### Full pipeline (CLI)

```bash
# Default: Tunisian restaurants, French prompts
python pipeline.py

# Custom domain, multilingual
python pipeline.py \
    --domain "Tunisian restaurants" \
    --languages fr ar \
    --n_intents 4 \
    --n_variants 5 \
    --n_runs 3 \
    --output_dir my_output

# Skip Agents 0+1, run Agent 2 on a known brand list
python pipeline.py \
    --brands_only "dar el jeld,le corsaire,dar zarrouk"
```

### Individual agents (Python)

```python
from agents.agent0_prompt_generator import Agent0PromptGenerator
from agents.agent1_geo_analyser     import Agent1GeoAnalyser
from agents.agent2_social_scraper   import Agent2SocialScraper

# Agent 0
a0 = Agent0PromptGenerator(
    domain="Tunisian restaurants", languages=["fr"], n_intents=3, n_variants=4)
prompt_df = a0.run()   # → prompt_set_Tunisian_restaurants.csv

# Agent 1
a1 = Agent1GeoAnalyser(prompt_set_path="geo_output/prompt_set_Tunisian_restaurants.csv")
features_df, global_df = a1.run()   # → geo_output/*.csv

# Agent 2
a2 = Agent2SocialScraper(brands=["dar el jeld", "le corsaire"])
profiles = a2.run()    # → agent2_output/social_profiles.csv

# Full pipeline (importable)
from pipeline import run_pipeline
result = run_pipeline(domain="Tunisian restaurants", languages=["fr"])
```

---

## Output files

```
geo_output/
├── prompt_set_{domain}.csv          Agent 0 → Agent 1 handoff
├── raw_responses.csv                Raw LLM responses (Step 2)
├── extracted_entities.csv           Brand names per response (Step 3)
├── enriched_entities.csv            + ranking position + description length (Step 4)
├── clean_entities.csv               Canonical entities + quality flags (Step 5)
├── cleaning_log.csv                 Every cleaning decision with phase tag
├── entity_features.csv              Per (prompt × entity) GEO features (Step 6)
├── entity_features_global.csv       Global GEO features — main analytical output
├── pipeline_summary.json            Run config + token usage
└── agent2_output/
    ├── social_profiles.csv          Full social + venue profile per brand
    ├── hallucination_flags.csv      All anomaly flags with confidence scores
    └── audit_log.json               Per-brand scraper choices + confidence
```
