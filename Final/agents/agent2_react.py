"""Agent 2 — Web feature extraction via MCP tool servers.

Calls MCP tools in a fixed sequence directly (no ReAct LLM loop), which
reduces per-entity LLM calls from ~10–12 down to zero for the research step.

All entities share one MCP client session so the three subprocesses
(search_server, scrape_server, wiki_server) are started once per run.
"""

import sys
import os
import asyncio
import csv
import re
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AGENT2_OUTPUT_DIR

from langchain_mcp_adapters.client import MultiServerMCPClient

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

# Anchor output to Final/ so the path is stable regardless of working directory.
_FINAL_DIR      = Path(__file__).parent.parent
OUTPUT_DIR      = str(_FINAL_DIR / AGENT2_OUTPUT_DIR)
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

_TOOL_TIMEOUT   = 60.0   # seconds per individual tool call
_ENTITY_TIMEOUT = 300.0  # seconds for all tools on one entity combined


async def _research_entity_direct(
    tools_map: dict,
    entity: str,
    location: str = "Tunisia",
) -> dict:
    """Execute the fixed research protocol via direct tool calls.

    No LLM is involved — each step calls the MCP tool directly and
    assigns the result fields. Confidence is computed from data coverage.
    """
    row: dict = {
        "canonical_entity": entity, "is_restaurant": True,
        "gm_rating": None, "gm_review_count": None, "gm_address": None,
        "gm_phone": None, "gm_website": None, "gm_category": None,
        "ta_rating": None, "ta_review_count": None, "ta_url": None,
        "ta_price_range": None, "ta_cuisine": None,
        "has_wikipedia": False,
        "ig_handle": None, "ig_followers": 0, "ig_posts": 0,
        "ig_engagement_rate": 0.0, "ig_bio": None,
        "fb_handle": None, "fb_page_likes": 0, "fb_followers": 0,
        "fb_post_engagement": 0.0,
        "overall_confidence": 0.0,
    }

    async def _call(name: str, **kwargs) -> dict:
        tool = tools_map.get(name)
        if tool is None:
            return {}
        try:
            res = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=_TOOL_TIMEOUT)
            return res if isinstance(res, dict) else {}
        except asyncio.TimeoutError:
            print(f"    [{name}] timed out after {_TOOL_TIMEOUT:.0f}s")
            return {}
        except Exception as e:
            print(f"    [{name}] error: {e}")
            return {}

    # Step 1: Google Maps
    gm = await _call("google_maps_search", entity=entity, location=location)
    for k in ("gm_name", "gm_rating", "gm_review_count", "gm_address",
              "gm_phone", "gm_website", "gm_category"):
        if gm.get(k) is not None:
            row[k] = gm[k]

    # Step 2: Wikipedia
    wiki = await _call("wikipedia_lookup", entity=entity)
    row["has_wikipedia"] = bool(wiki.get("has_wikipedia") or wiki.get("found"))

    # Step 3: TripAdvisor
    ta = await _call("scrape_tripadvisor", entity=entity, location=location)
    for k in ("ta_name", "ta_rating", "ta_review_count", "ta_url",
              "ta_price_range", "ta_cuisine"):
        if ta.get(k) is not None:
            row[k] = ta[k]

    # Step 4: Website socials (only if Google Maps found a website)
    ig_handle: str | None = None
    fb_handle: str | None = None
    if row.get("gm_website"):
        socials = await _call("extract_website_socials", website_url=row["gm_website"])
        ig_handle = socials.get("website_ig")
        fb_handle = socials.get("website_fb")

    # Step 5: Instagram — from website or DDG fallback
    if not ig_handle:
        ddg = await _call("ddg_search", query=f"{entity} site:instagram.com", max_results=3)
        if isinstance(ddg, list):
            for r in ddg:
                m = re.search(r"instagram\.com/([A-Za-z0-9_.]{3,30})", r.get("href", ""))
                if m:
                    ig_handle = m.group(1).lower()
                    break

    if ig_handle:
        row["ig_handle"] = ig_handle
        ig = await _call("scrape_instagram", handle=ig_handle)
        for k in ("ig_followers", "ig_posts", "ig_engagement_rate", "ig_bio"):
            if ig.get(k) is not None:
                row[k] = ig[k]

    # Step 6: Facebook — from website or DDG fallback
    if not fb_handle:
        ddg_fb = await _call("ddg_search", query=f"{entity} site:facebook.com", max_results=3)
        if isinstance(ddg_fb, list):
            for r in ddg_fb:
                m = re.search(r"facebook\.com/([A-Za-z0-9_.]{3,50})", r.get("href", ""))
                if m:
                    fb_handle = m.group(1).lower()
                    break

    if fb_handle:
        row["fb_handle"] = fb_handle
        fb = await _call("scrape_facebook", handle=fb_handle)
        for k in ("fb_page_likes", "fb_followers", "fb_post_engagement"):
            if fb.get(k) is not None:
                row[k] = fb[k]

    # Confidence from data coverage — no LLM call needed
    signals = [
        row.get("gm_rating") is not None,
        row.get("ta_rating") is not None,
        bool(row.get("ig_handle")),
        bool(row.get("fb_handle")),
        bool(row.get("has_wikipedia")),
        bool(row.get("gm_address")),
    ]
    row["overall_confidence"] = round(sum(signals) / len(signals), 2)

    return row


async def _research_all_async(entities: list[str], servers: dict,
                              location: str = "Tunisia") -> list[dict]:
    """Research all entities inside a single MCP client session."""
    results = []

    mcp_client = MultiServerMCPClient(servers)
    tools = await mcp_client.get_tools()
    tools_map = {t.name: t for t in tools}

    for i, entity in enumerate(entities, 1):
        print(f"\n[{i}/{len(entities)}] Researching: {entity}")
        try:
            row = await asyncio.wait_for(
                _research_entity_direct(tools_map, entity, location),
                timeout=_ENTITY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            print(f"  -> TIMEOUT after {_ENTITY_TIMEOUT:.0f}s")
            row = {"canonical_entity": entity, "overall_confidence": 0.0,
                   "error": f"timeout after {_ENTITY_TIMEOUT:.0f}s"}
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

    location = "Tunisia"
    new_rows = asyncio.run(_research_all_async(queue, servers, location))

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
