"""Agent 2 (ReAct/MCP) - Web feature extraction via MCP tool servers.

Replaces the hardcoded process_entity() loop with a LangGraph ReAct agent
that decides which tools to call per entity based on what it finds.

All entities are researched inside a single async context so MCP server
subprocesses are started once and reused — avoiding per-entity startup cost.
"""

import sys
import os
import json
import asyncio
import csv
import re
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GROQ_API_KEY, MODEL_STRONG as GROQ_MODEL

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# MCP server definitions
# Use sys.executable so the correct Python interpreter is always found.
# ---------------------------------------------------------------------------

_AGENT_DIR      = Path(__file__).parent
_MCP_SERVERS_DIR = _AGENT_DIR.parent / "mcp_servers"

_PYTHON = sys.executable

MCP_SERVERS = {
    "search": {
        "command": _PYTHON,
        "args": [str(_MCP_SERVERS_DIR / "search_server.py")],
        "transport": "stdio",
    },
    "scrape": {
        "command": _PYTHON,
        "args": [str(_MCP_SERVERS_DIR / "scrape_server.py")],
        "transport": "stdio",
    },
    "wiki": {
        "command": _PYTHON,
        "args": [str(_MCP_SERVERS_DIR / "wiki_server.py")],
        "transport": "stdio",
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a research agent for a GEO (Generative Engine Optimization) pipeline.
Your job is to collect web presence data for a restaurant entity.

For each entity you MUST follow this research protocol:
1. Call google_maps_search to get Google Maps data (rating, reviews, address, website).
2. Call wikipedia_lookup to check Wikipedia presence.
3. Call tripadvisor_search to get TripAdvisor data.
4. If google_maps_search returned a website URL, call extract_website_socials on that URL.
5. If an Instagram handle was found (from website socials), call scrape_instagram.
6. If a Facebook handle was found, call scrape_facebook.
7. If no social handles found from website, use ddg_search to find them:
   - Query: "<entity> site:instagram.com"
   - Query: "<entity> site:facebook.com"

After completing all research, output ONLY a JSON object with these exact keys:
{
  "canonical_entity": "<entity name>",
  "is_restaurant": true/false,
  "gm_rating": <float or null>,
  "gm_review_count": <int or null>,
  "gm_address": "<string or null>",
  "gm_phone": "<string or null>",
  "gm_website": "<string or null>",
  "gm_category": "<string or null>",
  "ta_rating": <float or null>,
  "ta_review_count": <int or null>,
  "ta_url": "<string or null>",
  "has_wikipedia": true/false,
  "ig_handle": "<string or null>",
  "ig_followers": <int or 0>,
  "ig_posts": <int or 0>,
  "ig_engagement_rate": <float or 0.0>,
  "ig_bio": "<string or null>",
  "fb_handle": "<string or null>",
  "fb_page_likes": <int or 0>,
  "fb_post_engagement": <float or 0.0>,
  "overall_confidence": <0.0-1.0>,
  "scrape_timestamp": "<ISO timestamp>"
}
Output ONLY the JSON. No explanation, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def _compute_derived(row: dict) -> dict:
    try:
        from rapidfuzz import fuzz
        bio_match = fuzz.partial_ratio(
            (row.get("canonical_entity") or "").lower(),
            (row.get("ig_bio") or "").lower()
        ) > 60
    except Exception:
        bio_match = False

    hw  = bool(row.get("gm_website"))
    hp  = bool(row.get("gm_phone"))
    ha  = bool(row.get("gm_address"))
    hi  = bool(row.get("ig_handle"))
    hf  = bool(row.get("fb_handle"))
    ht  = bool(row.get("ta_url"))
    hwk = bool(row.get("has_wikipedia"))

    return {
        "has_website":           hw,
        "has_phone":             hp,
        "has_address":           ha,
        "has_instagram":         hi,
        "has_facebook":          hf,
        "has_tripadvisor":       ht,
        "has_wikipedia":         hwk,
        "data_source_count":     sum([hw, hp, hi, hf, ht, hwk]),
        "review_total":          (row.get("gm_review_count") or 0) + (row.get("ta_review_count") or 0),
        "ig_bio_mentions_brand": bio_match,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

OUTPUT_DIR      = "agent2_output"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "web_features.csv")


def _load_done() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(CHECKPOINT_FILE)
        return set(df["canonical_entity"].dropna().tolist())
    except Exception:
        return set()


def _save_row(row: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exists = os.path.exists(CHECKPOINT_FILE)
    with open(CHECKPOINT_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=row.keys(), quoting=csv.QUOTE_ALL)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Async research — all entities share one MCP client session
# ---------------------------------------------------------------------------

def _parse_row(text: str, entity: str) -> dict:
    """Extract JSON from agent response."""
    clean = text.strip()
    if "```" in clean:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean)
        if m:
            clean = m.group(1).strip()
    # Try full parse first
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # Try extracting first {...} block
    m = re.search(r"\{[\s\S]*\}", clean)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {
        "canonical_entity": entity,
        "overall_confidence": 0.0,
        "parse_error": clean[:300],
    }


async def _research_all_async(entities: list[str], llm: ChatGroq,
                              servers: dict) -> list[dict]:
    """Research all entities inside a single MCP client session."""
    results = []

    async with MultiServerMCPClient(servers) as mcp_client:
        tools = mcp_client.get_tools()
        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

        for i, entity in enumerate(entities, 1):
            print(f"\n[{i}/{len(entities)}] Researching: {entity}")
            try:
                response = await agent.ainvoke({
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"Research this restaurant entity:\n"
                            f"Entity: {entity}\n"
                            f"Domain: Tunisian restaurants\n"
                            f"Follow the research protocol and return the JSON."
                        ),
                    }]
                })
                last_content = response["messages"][-1].content
                row = _parse_row(last_content, entity)
            except Exception as e:
                print(f"  -> ERROR: {e}")
                row = {"canonical_entity": entity, "error": str(e)[:200],
                       "overall_confidence": 0.0}

            row["scrape_timestamp"] = datetime.now().isoformat()
            results.append(row)
            print(f"  -> confidence={row.get('overall_confidence', '?')} "
                  f"| gm_rating={row.get('gm_rating', '?')}")

    return results


# ---------------------------------------------------------------------------
# Main orchestrator (sync entry point)
# ---------------------------------------------------------------------------

def run_agent2_react(entities: list[str], mcp_servers_dir: str = None,
                     fresh_start: bool = False) -> list[dict]:
    """Research all entities using MCP tools. Returns list of feature dicts.

    Args:
        entities:        List of canonical entity names to research.
        mcp_servers_dir: Optional override for MCP servers directory path.
                         Defaults to Final/mcp_servers/ relative to this file.
        fresh_start:     If True, delete existing checkpoint before starting.

    Returns:
        List of feature dicts, one per entity.
    """
    if mcp_servers_dir is not None:
        _dir = Path(mcp_servers_dir)
        servers = {
            "search": {"command": _PYTHON, "args": [str(_dir / "search_server.py")], "transport": "stdio"},
            "scrape": {"command": _PYTHON, "args": [str(_dir / "scrape_server.py")], "transport": "stdio"},
            "wiki":   {"command": _PYTHON, "args": [str(_dir / "wiki_server.py")],   "transport": "stdio"},
        }
    else:
        servers = MCP_SERVERS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if fresh_start and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    done  = _load_done()
    queue = [e for e in entities if e not in done]
    print(f"\n[Agent2-ReAct] {len(queue)} to research, {len(done)} already done")

    if not queue:
        import pandas as pd
        return pd.read_csv(CHECKPOINT_FILE).to_dict(orient="records")

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.0)

    # Single event loop for all entities
    new_rows = asyncio.run(_research_all_async(queue, llm, servers))

    all_rows = []
    for row in new_rows:
        derived = _compute_derived(row)
        row.update(derived)
        _save_row(row)
        all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def run_agent2_node(state: dict) -> dict:
    """LangGraph node: runs Agent 2 (ReAct/MCP) and merges results into state."""
    errors = list(state.get("errors", []))

    entities = [
        row["canonical_entity"]
        for row in state.get("entity_features_global", [])
        if row.get("canonical_entity")
    ]

    if not entities:
        errors.append("agent2: no entities in state - skipping")
        return {**state, "errors": errors, "current_step": "agent2_skipped"}

    try:
        web_rows = run_agent2_react(entities, fresh_start=False)
    except Exception as exc:
        errors.append(f"agent2: {exc}")
        web_rows = []

    return {
        **state,
        "web_features": web_rows,
        "errors":       errors,
        "current_step": "agent2",
    }
