"""MCP Search Server — Google Maps (SerpApi), TripAdvisor (DDG), DuckDuckGo.

Run as a standalone MCP server:
    python mcp_servers/search_server.py

Tools exposed:
    google_maps_search(entity, location) -> dict
    tripadvisor_search(entity)           -> dict
    ddg_search(query, max_results)       -> list[dict]
"""

import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SERPAPI_KEY

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search-server")


def _delay(lo=0.3, hi=0.8):
    time.sleep(random.uniform(lo, hi))


def _key_ok(k):
    return bool(k and len(k) > 10)


@mcp.tool()
def google_maps_search(entity: str, location: str = "Tunisia") -> dict:
    """Search Google Maps via SerpApi for a restaurant entity.

    Returns structured venue data: name, rating, review_count, address,
    phone, website, category, status.
    """
    result = {
        "gm_name": None, "gm_rating": None, "gm_review_count": None,
        "gm_address": None, "gm_phone": None, "gm_website": None,
        "gm_category": None, "status": "no_results",
    }
    if not _key_ok(SERPAPI_KEY):
        result["status"] = "no_key"
        return result

    try:
        from serpapi import GoogleSearch

        places: list = []
        for query in [f"{entity} restaurant {location}", f"{entity} {location}"]:
            params = {
                "engine": "google_maps",
                "q": query,
                "type": "search",
                "api_key": SERPAPI_KEY,
            }
            data = GoogleSearch(params).get_dict()
            places = data.get("local_results", [])
            if not places:
                single = data.get("place_results")
                if single:
                    places = [single]
            if places:
                break

        if not places:
            return result

        p = places[0]
        result["gm_name"]         = p.get("title")
        result["gm_rating"]       = p.get("rating")
        result["gm_review_count"] = p.get("reviews")
        result["gm_address"]      = p.get("address")
        result["gm_phone"]        = p.get("phone")
        result["gm_website"]      = p.get("website")

        types_raw = p.get("types", [])
        if types_raw and isinstance(types_raw[0], dict):
            result["gm_category"] = ", ".join(t.get("name", "") for t in types_raw)[:80]
        elif types_raw and isinstance(types_raw[0], str):
            result["gm_category"] = ", ".join(types_raw)[:80]
        else:
            result["gm_category"] = p.get("type", "")

        result["status"] = "success"

    except Exception as e:
        return {"status": "error", "error": str(e)}

    _delay()
    return result


@mcp.tool()
def tripadvisor_search(entity: str) -> dict:
    """Search TripAdvisor via DuckDuckGo snippets for a restaurant entity.

    Returns: ta_rating, ta_review_count, ta_url.
    """
    result: dict[str, object] = {"ta_rating": None, "ta_review_count": None, "ta_url": None}
    _delay(1.5, 2.5)

    try:
        from ddgs import DDGS
        query = f"{entity} restaurant tripadvisor Tunisia"
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=8))

        ta_snippets = []
        for hit in hits:
            if "tripadvisor.com" not in hit.get("href", ""):
                continue
            if not result["ta_url"]:
                result["ta_url"] = hit["href"]
            snippet = hit.get("body", "")
            title   = hit.get("title", "")
            if snippet:
                ta_snippets.append(f"{title} - {snippet}")

        if not ta_snippets:
            return result

        # Return raw snippets — the ReAct agent will extract with LLM
        result["raw_snippets"] = ta_snippets[:3]

    except Exception as e:
        return {"status": "error", "error": str(e)}

    return result


@mcp.tool()
def ddg_search(query: str, max_results: int = 5) -> list:
    """General DuckDuckGo text search.

    Returns list of {title, href, body} dicts.
    Useful for finding Instagram/Facebook handles and general entity info.
    """
    _delay(1.0, 2.0)
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return [{"status": "error", "error": str(e)}]


if __name__ == "__main__":
    mcp.run(transport="stdio")
